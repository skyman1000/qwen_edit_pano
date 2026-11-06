"""CPU regressions for observed v1 failures; no real prediction quality claims."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from ..common import read,sha,write
from .conditional_world import run
from .conditional_world_core import (aggregate_rotations,conditional_model,contexts_answer,
                                     worlds_answer,VisualMechanismScorer)
from .conditional_world_geometry import compile_layout,support_plane
from .qwen_upstream import DEFAULT_MODEL
from .scene_completion import map_plan
from .test_causal_world_completion import make_case


def world():
    return dict(worlds=[dict(description='A separate existing off-camera storage area is plausible, not observed.',
        additions=[dict(id='n1',name='bookshelf',object_type='bookshelf',
            reason='One possible existing storage choice compatible with this room; uncertain.',
            placement='Unphotographed space behind this camera.',support=dict(kind='floor',target=None))])])


class Scorer:
    def score(self,description,parents,states):
        if not parents:values=[1.] if len(states)==1 else [.75,.25]
        else:
            p=.8 if parents['configuration']=='conversation' else .2
            values=[p,1-p] if len(states)==2 else [1.]
        return dict(probabilities=dict(zip([s['id'] for s in states],values)),
                    permutation_max_total_variation=0.,synthetic_test=True)


class Model:
    def __init__(self,front=False):self.front=front;self.calls=[]
    def ask(self,image,prompt,path,**options):
        self.calls.append(path.stem)
        if path.stem=='contexts':
            value=dict(contexts=[dict(description='conversation',reason='Possible room use.',evidence_ids=[])],
                       floor_anchor_ids=['observed_0000'])
        elif path.stem.startswith(('contents','review')):value=world()
        else:
            value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=0 if self.front else 150,
                range_m=3.,size=[.8,1.8,.4],center_y_m=.9,reason='One rough off-camera hypothesis.')])
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,**options))
        return value


class CoreTests(unittest.TestCase):
    def test_all_rotations_cancel_additive_position_bias(self):
        ids=['a','b','c'];content={'a':1.,'b':3.,'c':0.};bias=[20.,-5.,10.];rounds=[]
        for i in range(3):
            ordered=ids[i:]+ids[:i]
            rounds.append(dict(state_ids=ordered,log_scores=[content[s]+bias[j] for j,s in enumerate(ordered)]))
        result=aggregate_rotations(rounds,ids)
        from .frozen_causal_core import normalized_probabilities
        expected=normalized_probabilities([content[s] for s in ids])
        for s,p in zip(ids,expected):self.assertAlmostEqual(result['probabilities'][s],p)
        self.assertGreater(result['permutation_max_total_variation'],.9)

    def test_incomplete_rotation_is_not_called_balanced(self):
        with self.assertRaises(ValueError):
            aggregate_rotations([dict(state_ids=['a','b','c'],log_scores=[0,1,2])],['a','b','c'])

    def test_conditional_model_has_no_invented_evidence_or_catchall(self):
        c=[dict(id='z1',description='conversation'),dict(id='z2',description='reading')]
        w=[dict(id='w1',scene_hypothesis='storage',additions=[dict(object_type='bookshelf')]),
           dict(id='empty',scene_hypothesis='empty',additions=[])]
        spec,trace=conditional_model(c,w,Scorer(),lambda *args:None)
        self.assertEqual([n['id'] for n in spec['nodes']],['Z','H'])
        self.assertAlmostEqual(trace['world_mixture']['probabilities']['w1'],.75*.8+.25*.2)
        self.assertIsNone(trace['coverage_probability'])
        self.assertEqual(trace['context_regimes']['z2']['probabilities']['w1'],.2)
        self.assertFalse(trace['actual_world_causal_effect_identified'])

    def test_typed_support_cycles_and_dangling_prose_are_rejected(self):
        case=dict(context=[dict(id='observed_0000')])
        bad=world();bad['worlds'][0]['additions'][0]['placement']='on top of n7'
        with self.assertRaisesRegex(ValueError,'Dangling prose'):worlds_answer(bad,case,'z1')
        bad=world();a=bad['worlds'][0]['additions'][0];b=deepcopy(a);b['id']='n2'
        a['support']=dict(kind='object',target='n2');b['support']=dict(kind='object',target='n1')
        bad['worlds'][0]['additions'].append(b)
        with self.assertRaisesRegex(ValueError,'Cyclic'):worlds_answer(bad,case,'z1')

    def test_context_evidence_must_refer_to_observed_instances(self):
        with self.assertRaises(ValueError):
            contexts_answer(dict(contexts=[dict(description='x',reason='y',evidence_ids=['missing'])],
                                 floor_anchor_ids=[]),dict(context=[]))

    def test_review_reordered_identifiers_are_rebound_in_prose_and_support(self):
        w=world();a=w['worlds'][0]['additions'][0];a['id']='z1_w1_n2'
        b=deepcopy(a);b['id']='z1_w1_n1';b['name']='box';b['object_type']='box'
        b['support']=dict(kind='object',target='z1_w1_n2');b['placement']='on z1_w1_n2'
        w['worlds'][0]['additions'].append(b)
        normalized=worlds_answer(w,dict(context=[]),'z1')[0]
        self.assertEqual(normalized['additions'][1]['support']['target'],'z1_w1_n1')
        self.assertEqual(normalized['additions'][1]['placement'],'on z1_w1_n1')


class GeometryTests(unittest.TestCase):
    def setup_geometry(self):
        observed=dict(objects=[dict(track_id='obs',center_local_m={'value':[0.,-1.5,3.]},
            size_aabb_local_m={'value':[2.,1.,1.]})])
        case=dict(context=[dict(id='obs')]);plan=worlds_answer(world(),case,'z1')[0]
        return observed,plan,map_plan(plan,'semantic_lexicon_v1'),support_plane(observed,['obs'])

    def test_bearing_support_compiles_back_camera_and_keeps_floor_height(self):
        obs,plan,mapping,plane=self.setup_geometry();before=deepcopy(obs)
        value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=150,range_m=3.,size=[.8,1.8,.4],center_y_m=.9,reason='test')])
        items,review=compile_layout(value,plan,mapping,obs,plane,90,[100,100])
        self.assertLess(items[0]['center'][2],0)
        self.assertAlmostEqual(items[0]['center'][1],-1.1)
        self.assertAlmostEqual(items[0]['center'][1]-items[0]['size'][1]/2,plane['floor_y_m'])
        self.assertTrue(review['geometry_contract_passed']);self.assertFalse(review['position_was_projected'])
        self.assertEqual(obs,before);self.assertFalse(plane['floor_measured'])

    def test_in_fov_input_is_not_teleported_to_boundary(self):
        obs,plan,mapping,plane=self.setup_geometry()
        value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=0,range_m=3.,size=[.8,1.8,.4],reason='test')])
        items,review=compile_layout(value,plan,mapping,obs,plane,90,[100,100])
        self.assertEqual(items[0]['center'][0],0.)
        self.assertEqual(items[0]['center'][2],3.)
        self.assertFalse(review['numeric_constraints_passed'])
        self.assertFalse(review['position_was_projected'])

    def test_replan_cannot_shrink_sizes_and_free_xyz_is_rejected(self):
        obs,plan,mapping,plane=self.setup_geometry()
        value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=150,range_m=3.,size=[.8,1.8,.4],reason='test')])
        with self.assertRaisesRegex(ValueError,'resize'):
            compile_layout(value,plan,mapping,obs,plane,90,[100,100],{'z1_w1_n1':[1.,2.,1.]})
        value['objects'][0]['center']=[0,0,3]
        with self.assertRaisesRegex(ValueError,'free xyz'):compile_layout(value,plan,mapping,obs,plane,90,[100,100])

    def test_directly_above_camera_is_representable_without_fake_horizontal_offset(self):
        obs,plan,mapping,plane=self.setup_geometry()
        plan['additions'][0]['support']=dict(kind='unspecified',target=None)
        value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=0,range_m=0.,size=[.2,.2,.2],center_y_m=2.,reason='Overhead hypothesis.')])
        items,review=compile_layout(value,plan,mapping,obs,plane,90,[100,100])
        self.assertEqual(items[0]['center'],[0.,2.,0.])
        self.assertTrue(review['geometry_contract_passed'])
        self.assertIsNone(review['support_checks'][0]['proxy_passed'])

    def test_object_support_uses_target_top_and_reports_horizontal_gap(self):
        obs,plan,mapping,plane=self.setup_geometry()
        plan['additions'][0]['support']=dict(kind='object',target='obs')
        value=dict(objects=[dict(id='z1_w1_n1',azimuth_deg=180,range_m=3.,size=[.8,.4,.4],reason='test')])
        items,review=compile_layout(value,plan,mapping,obs,plane,90,[100,100])
        self.assertAlmostEqual(items[0]['center'][1],-.8)
        self.assertEqual(review['support_proxy_failures'],1)
        self.assertFalse(review['geometry_contract_passed'])


class IntegrationTests(unittest.TestCase):
    def test_world_state_visuals_and_frozen_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            # Fixture IDs are obs0/obs1 rather than real validation identifiers.
            anchor=case['context'][0]['id']
            class FixtureModel(Model):
                def ask(self,image,prompt,path,**options):
                    if path.stem=='contexts':
                        v=dict(contexts=[dict(description='conversation',reason='Possible use.',evidence_ids=[])],floor_anchor_ids=[anchor])
                        path.write_text(json.dumps(v));write(path.with_suffix('.json'),v);return v
                    return super().ask(image,prompt,path,**options)
            out=root/'out';out.mkdir();model=FixtureModel()
            r=run(model,lambda *args:Scorer(),case,out,worlds=2)
            self.assertEqual(r['status'],'candidates_exported_needs_review')
            full=read(out/'case/G_full.json')
            self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects'])
            self.assertEqual(len(full['objects']),len(observed['objects'])+1)
            self.assertTrue((out/'case/hypothesis_01/world/scene_3d.html').is_file())
            self.assertTrue(any(c.startswith('review') for c in model.calls))
            for p,h in case['source_sha256'].items():self.assertEqual(sha(p),h)
            from .conditional_world import code_hashes
            from .conditional_world_audit import audit
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments={'worlds':2}))
            self.assertEqual(audit(out)['status'],'postflight_contract_passed')

    def test_invalid_geometric_replan_preserves_first_complete_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,observed=make_case(root);case['image_size']=[64,64]
            anchor=case['context'][0]['id']
            class BadReplan(Model):
                def ask(self,image,prompt,path,**options):
                    if path.stem=='contexts':
                        value=dict(contexts=[dict(description='conversation',reason='Uncertain use.',evidence_ids=[])],
                                   floor_anchor_ids=[anchor])
                    else:
                        value=super().ask(image,prompt,path,**options)
                        if path.stem.startswith('bearing_1'):value['objects'][0]['size'][0]=.2
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
            out=root/'out';out.mkdir()
            result=run(BadReplan(front=True),lambda *args:Scorer(),case,out,worlds=1)
            self.assertEqual(result['geometry_unresolved'],1)
            full=read(out/'case/hypothesis_01/G_full.json')
            self.assertEqual(len(full['objects']),len(observed['objects'])+1)
            self.assertEqual(full['objects'][-1]['size_aabb_local_m']['value'],[.8,1.8,.4])
            attempts=read(out/'case/hypothesis_01/layout_attempts.json')
            self.assertEqual(attempts['selected_attempt'],0)
            self.assertFalse(attempts['attempts'][-1]['schema_valid'])
            self.assertFalse((out/'case/G_full.json').exists())

    def test_real_local_Qwen_visual_forward_on_tiny_CPU_weights(self):
        # Verify actual processor expansion + image-grid + logits_to_keep API.
        # Uses local tokenizer/processor only, never 8B weights or GPU/training.
        import torch
        from PIL import Image
        from types import SimpleNamespace
        from transformers import AutoProcessor,Qwen3VLConfig,Qwen3VLForConditionalGeneration
        processor=AutoProcessor.from_pretrained(str(DEFAULT_MODEL),local_files_only=True)
        config=Qwen3VLConfig(
            text_config=dict(vocab_size=152064,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=4096,
                rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=32,
                patch_size=16,spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=256,
                deepstack_visual_indexes=[]))
        tiny=Qwen3VLForConditionalGeneration(config).eval()
        scorer=VisualMechanismScorer(SimpleNamespace(model=tiny,processor=processor,torch=torch),
                                    Image.new('RGB',(64,64)),[dict(description='sofa')])
        result=scorer.score('test mechanism',{},[dict(id='a',description='one'),dict(id='b',description='two')])
        self.assertAlmostEqual(sum(result['probabilities'].values()),1.)
        self.assertEqual(len(result['rounds']),2)


if __name__=='__main__':unittest.main()
