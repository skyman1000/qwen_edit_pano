"""CPU math, schema, text scoring and complete fake-model integration tests.

Synthetic responses verify software, never semantic quality of real predictions.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..common import read, sha, write
from .causal_probability import CausalGraph
from .frozen_causal_core import (FrozenChoiceScorer, build_model, infer, normalize_proposal,
                                normalized_probabilities, shifted_token_logprob)
from .frozen_causal_layout import solve_layout
from .frozen_causal_world import code_hashes, load_case, run
from .scene_completion import map_plan, view_check
from .test_causal_world_completion import make_case


def proposal():
    return dict(contexts=[dict(description='Conversational living room configuration', reason='An uncertain use hypothesis.'),
        dict(description='Living room also arranged for watching TV', reason='An alternative unobserved intention.')],
        worlds=[dict(description='Hidden storage area elsewhere around the camera.',
            additions=[dict(id='n1', name='hidden bookshelf', object_type='bookshelf',
                reason='A possible existing storage choice, not mandatory equipment.',
                placement='Beside the camera outside the original view.')], relations=[])])


class FakeScorer:
    """Known synthetic tables make posterior and interventions testable."""
    def __init__(self): self.calls = []
    def score(self, description, parents, states):
        self.calls.append((description, deepcopy(parents), deepcopy(states)))
        ids = [s['id'] for s in states]
        if not parents:
            probs = dict(z1=.45, z2=.45, z_other=.10)
        elif ids == ['present', 'not_present']:
            p = .9 if 'Conversational' in parents['configuration_context'] else .1
            probs = dict(present=p, not_present=1-p)
        else:
            p = .8 if 'Conversational' in parents['configuration_context'] else .2
            probs = dict(w01=p, w_empty=.95-p, w_other=.05)
        return dict(probabilities=probs, synthetic_test=True)


class FakeModel:
    def __init__(self, broken=False, unmapped=False):
        self.broken = broken; self.unmapped = unmapped; self.calls = []
    def ask(self, image, prompt, path, **options):
        self.calls.append(path.stem)
        if path.stem.startswith('proposal'):
            value = proposal()
            if self.unmapped: value['worlds'][0]['additions'][0]['object_type'] = 'unlisted_specific_appliance'
        else:
            value = dict(objects=[dict(id='w01_n001', center=[1.5, .05, 3.2],
                size=[.8, 1.8, .4], reason='Coarse initial layout in the unphotographed surroundings.')])
            if self.broken: value['objects'][0]['size'][0] = 0
        path.write_text(json.dumps(value)); write(path.with_suffix('.json'), value)
        write(path.with_suffix('.request.json'), dict(prompt=prompt, **options))
        return value


class CoreTests(unittest.TestCase):
    def test_prior_evidence_update_and_do_are_external_math(self):
        case = dict(context=[])
        domain = normalize_proposal(proposal(), case)
        scorer = FakeScorer(); saved = {}
        spec = build_model(domain, ['observed sofa'], scorer, lambda n, row: saved.update({n:row}))
        before = deepcopy(spec); result = infer(spec)
        expected_context = .45*.9/(.45*.9+.45*.1+.1*.1)
        self.assertAlmostEqual(result['posterior_context']['probabilities']['z1'], expected_context)
        expected_hidden = expected_context*.8+(1-expected_context)*.2
        self.assertAlmostEqual(result['posterior_hidden']['probabilities']['w01'], expected_hidden)
        self.assertAlmostEqual(result['context_interventions']['z2']['probabilities']['w01'], .2)
        self.assertNotAlmostEqual(result['prior_hidden']['probabilities']['w01'], expected_hidden)
        for sid, p in result['prior_hidden']['probabilities'].items():
            self.assertAlmostEqual(result['do_evidence_control']['probabilities'][sid], p, places=12)
        self.assertEqual(spec, before)
        # No observed evidence or image enters the root or hidden mechanism.
        for desc, parents, states in scorer.calls:
            if 'AT LEAST' not in desc:
                self.assertNotIn('observed sofa', desc+json.dumps(parents)+json.dumps(states))
        self.assertEqual(len(saved), 7)
        self.assertFalse(result['causal_gain_validated'])

    def test_invalid_scorer_distribution_rejected(self):
        class Bad(FakeScorer):
            def score(self, *args): return dict(probabilities={'extra':1.})
        with self.assertRaises(ValueError):
            build_model(normalize_proposal(proposal(), dict(context=[])), [], Bad(), lambda *args:None)

    def test_id_remapping_is_explicit_and_observations_cannot_be_reused(self):
        raw = proposal(); raw['worlds'][0]['relations'] = [dict(source='n1', target='obs', kind='near',
            max_distance_m=5, reason='Hypothesized adjacency.')]
        case = dict(context=[dict(id='obs')]); original = deepcopy(raw)
        d = normalize_proposal(raw, case)
        self.assertEqual(d['worlds'][0]['relations'][0]['source'], 'w01_n001')
        self.assertEqual(raw, original)
        raw['worlds'][0]['additions'][0]['id'] = 'obs'
        with self.assertRaisesRegex(ValueError, 'observed-instance'):
            normalize_proposal(raw, case)

    def test_duplicate_bundles_unknown_edges_and_capacity_rejected(self):
        for mode in ('duplicate', 'edge', 'capacity'):
            value = proposal()
            if mode == 'duplicate': value['worlds'] *= 2
            if mode == 'edge': value['worlds'][0]['relations'] = [dict(source='missing',target='n1',kind='near',reason='x')]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                normalize_proposal(value, dict(context=[]), capacity=0 if mode=='capacity' else 256)

    def test_empty_and_other_are_explicit_not_redistributed(self):
        domain = normalize_proposal(proposal(), dict(context=[]))
        self.assertIn('w_empty', [w['id'] for w in domain['worlds']])
        spec = build_model(domain, [], FakeScorer(), lambda *args:None)
        r = infer(spec)
        self.assertAlmostEqual(r['unresolved_world_mass'], .05)
        self.assertAlmostEqual(sum(r['posterior_hidden']['probabilities'].values()), 1.)

    def test_stable_probability_normalization(self):
        self.assertAlmostEqual(sum(normalized_probabilities([-10000., -10001.])), 1.)
        for x in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError): normalized_probabilities(x)

    def test_teacher_forcing_uses_previous_positions_and_common_terminator(self):
        import torch
        logits = torch.zeros(1, 5, 7)
        logits[0, 2, 4] = 3.; logits[0, 3, 6] = 2.
        # Prompt has 3 tokens. Candidate token 4 comes from position 2,
        # EOS 6 from position 3, never from the final position 4.
        logits[0, 4, 0] = 100.
        expected = logits[0, 2].log_softmax(-1)[4]+logits[0, 3].log_softmax(-1)[6]
        self.assertAlmostEqual(shifted_token_logprob(logits, 3, [4,6]), float(expected))

    def test_scorer_runs_without_images_updates_or_shared_mutable_cache(self):
        import torch
        class Tokenizer:
            eos_token_id = 6
            def encode(self, value, **kwargs): return [2,3] if value == 'PREFIX' else [ord(value)-65]
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, *args, **kwargs): return 'PREFIX'
        class Model:
            device = 'cpu'
            def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
                self_nonlocal_calls.append((input_ids.clone(), use_cache))
                self.assert_positions = logits_to_keep.tolist()
                return SimpleNamespace(logits=torch.zeros(1,len(logits_to_keep),8))
        self_nonlocal_calls = []
        q = SimpleNamespace(processor=Processor(), model=Model(), torch=torch)
        r = FrozenChoiceScorer(q).score('test', {}, [dict(id='x',description='one'),dict(id='y',description='two')])
        self.assertEqual(r['probabilities'], dict(x=.5,y=.5))
        self.assertTrue(all(not cache for _,cache in self_nonlocal_calls))
        self.assertEqual(len(self_nonlocal_calls), 4)

    def test_actual_local_Qwen3VL_forward_api_on_tiny_CPU_model(self):
        # Random tiny weights, no downloads/training/GPU or real 8B loading.
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128,
                rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
            vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,
                out_hidden_size=32,patch_size=2,spatial_merge_size=1,temporal_patch_size=1,
                num_position_embeddings=16,deepstack_visual_indexes=[]),
            image_token_id=28,video_token_id=29,vision_start_token_id=30,vision_end_token_id=31)
        model = Qwen3VLForConditionalGeneration(config).eval()
        ids = torch.tensor([[1,2,3,4,5]])
        with torch.inference_mode():
            full=model(input_ids=ids,use_cache=False).logits
            short=model(input_ids=ids,use_cache=False,logits_to_keep=torch.tensor([2,3])).logits
        self.assertTrue(torch.allclose(full[:,[2,3]],short,atol=1e-6))
        from .frozen_causal_core import continuation_logprob
        self.assertAlmostEqual(shifted_token_logprob(full,3,[4,5]),continuation_logprob(short[0],[4,5]))


class LayoutTests(unittest.TestCase):
    def setup_plan(self):
        return normalize_proposal(proposal(),dict(context=[]))['worlds'][0]

    def test_old_trust_cube_failure_is_solved_without_resizing(self):
        plan = self.setup_plan(); mapping = map_plan(plan, 'semantic_lexicon_v1')
        raw = dict(objects=[dict(id='w01_n001',center=[1.5,.05,3.2],size=[.08,.12,.08],reason='test')])
        original = deepcopy(raw)
        items, review = solve_layout(raw, plan, mapping, dict(objects=[]), 90, [100,100])
        self.assertTrue(review['after']['numeric_constraints_passed'])
        self.assertGreater(max(abs(v) for v in items[0]['translation_m']), .75)
        self.assertEqual(raw, original); self.assertEqual(items[0]['size'], [.08,.12,.08])

    def test_full_box_not_just_center_and_asymmetric_vertical_fov(self):
        plan = self.setup_plan(); mapping = map_plan(plan, 'semantic_lexicon_v1')
        for c, size, wh in [([3.,0.,2.],[3.,1.,1.],[100,100]),([0.,1.,2.],[.2,1.,.2],[200,100]),
                            ([0.,0.,0.],[1.,1.,1.],[100,200]),([0.,0.,-3.],[1.,1.,1.],[100,100])]:
            raw = dict(objects=[dict(id='w01_n001',center=c,size=size,reason='test')])
            out, review = solve_layout(raw, plan, mapping, dict(objects=[]),90,wh)
            self.assertFalse(view_check(out[0]['center'],size,90,wh)['intersects_input_box_proxy'])
            self.assertTrue(review['after']['numeric_constraints_passed'])

    def test_soft_relation_conflict_does_not_delete_objects_or_change_observations(self):
        plan = self.setup_plan(); plan['relations'] = [dict(source='w01_n001',target='obs',kind=k,reason='test')
                                                         for k in ('left_of','right_of')]
        observed = dict(objects=[dict(track_id='obs',center_local_m={'value':[0,0,-3]},size_aabb_local_m={'value':[2,2,2]})])
        before = deepcopy(observed)
        raw = dict(objects=[dict(id='w01_n001',center=[0,0,-3],size=[1,1,1],reason='test')])
        out, review = solve_layout(raw, plan, map_plan(plan,'semantic_lexicon_v1'), observed,90,[100,100])
        self.assertEqual(len(out),1); self.assertEqual(observed,before)
        self.assertGreater(review['after']['soft_relation_violations'],0)
        self.assertTrue(review['after']['numeric_constraints_passed'])


class IntegrationTests(unittest.TestCase):
    def test_complete_world_and_visualization_preserve_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); case, observed = make_case(root); case['image_size']=[64,64]
            hashes = deepcopy(case['source_sha256']); output=root/'new'; output.mkdir()
            r = run(FakeModel(),FakeScorer(),case,output,worlds=2,sweeps=1)
            self.assertEqual(r['status'],'hypotheses_ready_needs_review')
            self.assertTrue(r['world_state_ready']); self.assertFalse(r['adapter_connected'])
            folder=output/case['id']; full=read(folder/'hypothesis_01/G_full.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(len(full['objects']),len(observed['objects'])+1)
            self.assertTrue((folder/'hypothesis_01/world/scene_3d.html').is_file())
            self.assertIn('全部保留在G_full',(folder/'hypothesis_01/world/index.html').read_text())
            self.assertTrue((folder/'causal_graph.svg').is_file())
            for p,h in hashes.items(): self.assertEqual(sha(p),h)
            write(output/'config.json', dict(case=case,code_sha256=code_hashes(),arguments={'worlds':2}))
            from .frozen_causal_audit import audit
            self.assertEqual(audit(output)['status'],'postflight_contract_passed')
            damaged=read(folder/'hypothesis_01/G_full.json')
            damaged['objects'][0]['center_local_m']['value'][0]+=1
            write(folder/'hypothesis_01/G_full.json',damaged)
            with self.assertRaisesRegex(ValueError,'Observed prefix changed'):
                audit(output)

    def test_failed_layout_is_not_hidden_or_replaced_with_successful_empty_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64];out=root/'out';out.mkdir()
            r=run(FakeModel(broken=True),FakeScorer(),case,out,worlds=2,sweeps=0)
            self.assertEqual(r['status'],'partial_failure')
            self.assertEqual(r['hypothesis_failures'],1)
            self.assertTrue((out/'case/hypothesis_01/boxes_repair.txt').is_file())
            self.assertTrue((out/'case/hypothesis_01/error_traceback.txt').is_file())
            self.assertEqual(read(out/'case/G_obs.json'),obs)
            self.assertFalse((out/'case/G_full.json').exists())

    def test_unmapped_object_is_retained_in_full_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64];out=root/'out';out.mkdir()
            r=run(FakeModel(unmapped=True),FakeScorer(),case,out,worlds=1,sweeps=0)
            self.assertEqual(r['status'],'hypotheses_ready_needs_review')
            full=read(out/'case/G_full.json')
            self.assertEqual(len(full['objects']),len(obs['objects'])+1)
            self.assertFalse(full['objects'][-1]['category_id']['valid'])
            self.assertTrue(full['semantic_plan_fully_exported'])

    def test_custom_case_uses_predicted_state_and_checks_safe_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);path=root/'input.json'
            write(path,{k:case[k] for k in ('id','image','state','hfov')})
            loaded=load_case(path);self.assertEqual(loaded['image_size'],[64,64])
            write(path,{**read(path),'id':'../escape'})
            with self.assertRaises(ValueError):load_case(path)


if __name__ == '__main__': unittest.main()
