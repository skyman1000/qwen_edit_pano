"""Real failed outputs, independent projection checks, and actual HF CPU APIs."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write
from .scene_completion import view_check
from .scm_typed_queries import ContinuationMechanismScorer,TokenTrie,finite_choice,normalized_logs
from .scm_typed_layout import canonical_angle,numeric_body,project_2d,replay_typed,solve_center,solve_typed
from .scm_world_layout import instantiate
from .scm_world import code_hashes,run
from .scm_world_audit import audit
from .test_causal_world_completion import make_case
from .test_scm_world import fixture_spec,SyntheticModel,SyntheticScorer
from .conditional_world_geometry import support_plane

FIXTURE=Path(__file__).with_name('testdata')/'scm_run002_layout_failures.json'


def body_for(record):
    return numeric_body({k:record[k] for k in ('azimuth_deg','range_m','size','center_y_m','reason')})


def fake_choice(model,image,prompt,candidates,path):
    # Select hosts first, then accessories. This is explicitly synthetic.
    if path.stem.endswith('_order'):
        choice=next((c for c in candidates if 'shelf' in c['name']),candidates[0])
    else:
        host=next((c for c in candidates if c['kind']=='object' and 'shelf' in c['description'] and 'scm_new' in c['target']),None)
        choice=host or next((c for c in candidates if c['kind']=='floor'),candidates[-1])
    path.write_text(json.dumps(choice));write(path.with_suffix('.json'),choice)
    write(path.with_suffix('.request.json'),dict(method='finite_token_trie_greedy_layout_v1',candidates=candidates))
    return choice


class GeometryTests(unittest.TestCase):
    def test_real_270_degrees_is_periodic_coordinate_not_invalid_geometry(self):
        raw=read(FIXTURE)['chair_270'];before=deepcopy(raw);b=body_for(raw)
        self.assertEqual(b['canonical_azimuth_deg'],-90.)
        self.assertEqual(raw,before)
        self.assertAlmostEqual(math.sin(math.radians(raw['azimuth_deg'])),math.sin(math.radians(b['canonical_azimuth_deg'])))
        for a in (float('nan'),float('inf'),'270',True):
            with self.assertRaises(ValueError):canonical_angle(a)

    def test_real_boundary_desk_is_solved_with_size_and_height_fixed(self):
        b=body_for(read(FIXTURE)['desk']);before=deepcopy(b)
        old=[b['range_m']*math.sin(math.pi/4),b['center_y_m'],b['range_m']*math.cos(math.pi/4)]
        self.assertTrue(view_check(old,b['size'],90,[1024,1024])['intersects_input_box_proxy'])
        result=solve_center(b,dict(kind='unspecified',target=None),dict(floor_y_m=-2.022),{},90,[1024,1024])
        # Independent analytic projection onto x-z >= (sx+sz)/2+epsilon.
        delta=(sum((b['size'][0],b['size'][2]))/2+1e-4)/2
        np.testing.assert_allclose(result['center'],[old[0]+delta,old[1],old[2]-delta],atol=1e-10)
        self.assertFalse(view_check(result['center'],b['size'],90,[1024,1024])['intersects_input_box_proxy'])
        self.assertEqual(b,before);self.assertFalse(result['sizes_changed'])

    def test_projection_matches_independent_scipy_constrained_minimization(self):
        from scipy.optimize import minimize,LinearConstraint
        p=np.array([2.,3.]);A=np.array([[1.,-1.],[1.,0.],[0.,-1.]]);b=np.array([.8,.2,-2.8])
        x=project_2d(p,A,b)
        ref=minimize(lambda v:np.sum((v-p)**2),p,jac=lambda v:2*(v-p),
            constraints=LinearConstraint(A,b,np.inf),method='SLSQP',options=dict(ftol=1e-12))
        self.assertTrue(ref.success);np.testing.assert_allclose(x,ref.x,atol=1e-7)
        self.assertIsNone(project_2d(p,[[1,0],[-1,0]],[1,1]))

    def test_support_height_and_positive_overlap_preserved_no_observation_moved(self):
        b=body_for(dict(azimuth_deg=270,range_m=3,size=[.4,.5,.4],center_y_m=-999,reason='test'))
        boxes={'host':dict(center=[-3.,-.5,0.],size=[1.,1.,1.])};before=deepcopy(boxes)
        result=solve_center(b,dict(kind='object',target='host'),dict(floor_y_m=-2),boxes,90,[1024,512])
        self.assertAlmostEqual(result['center'][1],.25)
        self.assertGreater(.7-abs(result['center'][0]+3),0)
        self.assertFalse(view_check(result['center'],b['size'],90,[1024,512])['intersects_input_box_proxy'])
        self.assertEqual(boxes,before)

    def test_single_body_rejects_identity_rewriting_and_missing_sizes(self):
        b={k:v for k,v in read(FIXTURE)['desk'].items() if k!='id'}
        self.assertEqual(len(read(FIXTURE)['wrong_support_ids']['objects']),9)
        for invalid in (dict(objects=[]),{**b,'id':'observed_0000'},{**b,'size':[0,1,1]}):
            with self.assertRaises(ValueError):numeric_body(invalid)


class TypedIntegrationTests(unittest.TestCase):
    def test_fixed_inventory_support_dag_projection_export_visuals_and_audit(self):
        class TokenTraceScorer(SyntheticScorer):
            def score(self,description,parents,states):
                logs=[-1000.]*(len(states)-1)+[-1.] if states[0]['id'].isdigit() else [-1.]*len(states)
                return dict(probabilities=dict(zip([s['id'] for s in states],normalized_logs(logs))),
                    candidate_sequences=[dict(id=s['id'],text=s['description'],token_ids=[1],
                        token_log_probabilities=[v],log_probability=v) for s,v in zip(states,logs)],
                    parents=deepcopy(parents),states=deepcopy(states),rounds=[],image_used=False,
                    source='synthetic_token_trace_for_audit_test')
        class Model(SyntheticModel):
            def ask(self,image,prompt,path,**options):
                if path.stem.startswith('metric_'):
                    value=dict(azimuth_deg=270,range_m=3.,size=[.4,.5,.4],center_y_m=-1.,reason='Synthetic side proposal.')
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
                    return value
                return super().ask(image,prompt,path,**options)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64];out=root/'out';out.mkdir()
            with patch('qwen_edit_pano.world_pipeline.completion.scm_typed_layout.finite_choice',fake_choice):
                record=run(Model(case),TokenTraceScorer(),case,out,max_count=3,generic_types=3,worlds=2,seed=0,layout_backend='typed')
            self.assertEqual(record['status'],'posterior_samples_exported_needs_review',record)
            self.assertTrue(record['world_state_ready'])
            self.assertEqual(read(out/'case/G_full.json')['objects'][:len(obs['objects'])],obs['objects'])
            args=dict(generic_types=3,max_count=3,worlds=2,seed=0,max_objects=256,layout_backend='typed')
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(out)['status'],'scm_postflight_contract_passed')
            mechanism=out/'case/mechanisms/B.json';original=read(mechanism);broken=deepcopy(original)
            broken['candidate_sequences'][0]['token_log_probabilities'][0]-=1.
            write(mechanism,broken)
            with self.assertRaisesRegex(ValueError,'token scores'):audit(out)
            write(mechanism,original)
            file=out/'case/hypothesis_01/typed_layout.json';broken=read(file)
            broken['steps'][0]['id']='observed_0000';write(file,broken)
            with self.assertRaises(ValueError):audit(out)

    def test_order_support_targets_are_program_owned_and_cycles_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64]
            sample=dict(seed=1,B='b1',K='k1',counts=[dict(object_type='shelf',outside=1),dict(object_type='lamp',outside=1)])
            plan=instantiate(sample,case,fixture_spec());before=deepcopy(plan)
            class Model:
                def ask(self,image,prompt,path,**options):
                    value=dict(azimuth_deg=180,range_m=3.,size=[.5,.5,.5],center_y_m=-1.,reason='synthetic')
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
            solved,_,items,review=solve_typed(Model(),None,case,obs,plan,support_plane(obs,[case['context'][0]['id']]),
                root,dict(seed=0),fake_choice)
            self.assertEqual(plan,before);self.assertEqual(len(items),2);self.assertTrue(review['geometry_contract_passed'])
            self.assertEqual(solved['additions'][1]['support']['target'],solved['additions'][0]['id'])
            saved=read(root/'typed_layout.json')
            replayed,_=replay_typed(case,obs,plan,support_plane(obs,[case['context'][0]['id']]),saved)
            self.assertEqual(replayed,solved)
            saved['steps'][0]['support']=dict(kind='object',target='scm_new_0002',description='future')
            with self.assertRaises(ValueError):replay_typed(case,obs,plan,support_plane(obs,[case['context'][0]['id']]),saved)


class ActualHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoProcessor,Qwen3VLConfig,Qwen3VLForConditionalGeneration
        from .qwen_upstream import DEFAULT_MODEL
        processor=AutoProcessor.from_pretrained(str(DEFAULT_MODEL),local_files_only=True)
        config=Qwen3VLConfig(text_config=dict(vocab_size=152064,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=4096,
            rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=32,
            patch_size=16,spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=256,deepstack_visual_indexes=[]))
        model=Qwen3VLForConditionalGeneration(config).eval()
        cls.q=SimpleNamespace(model=model,processor=processor,torch=torch)

    def test_aligned_sequence_probabilities_match_independent_full_forward_and_order_invariant(self):
        q=self.q;states=[dict(id='0',description='Exactly zero lamps.'),dict(id='1',description='Exactly one lamp.')]
        scorer=ContinuationMechanismScorer(q);r=scorer.score('lamp count',{'B':'living room'},states)
        reverse=scorer.score('lamp count',{'B':'living room'},states[::-1])
        self.assertEqual(r['probabilities'],reverse['probabilities']);self.assertNotIn('STATES:',r['prompt'])
        prefix=q.processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=r['prompt'])])],
            tokenize=False,add_generation_prompt=True)
        ids=q.processor.tokenizer.encode(prefix,add_special_tokens=False);row=r['candidate_sequences'][0]
        full=q.torch.tensor([ids+row['token_ids']])
        with q.torch.inference_mode():logits=q.model(input_ids=full,use_cache=False,logits_to_keep=0).logits
        aligned=logits[0,len(ids)-1:-1].float().log_softmax(-1)
        ref=aligned.gather(1,q.torch.tensor(row['token_ids'])[:,None]).sum().item()
        self.assertAlmostEqual(row['log_probability'],ref,places=4)
        self.assertEqual(row['token_ids'][-1],q.processor.tokenizer.eos_token_id)

    def test_actual_transformers_trie_generation_cannot_emit_observed_ID(self):
        candidates=[dict(name='shelf',id='scm_new_0001'),dict(name='lamp',id='scm_new_0002')]
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'choice.txt'
            value=finite_choice(self.q,None,'Choose a new object from the fixed inventory.',candidates,file)
            self.assertIn(value,candidates);self.assertEqual(json.loads(file.read_text()),value)
            self.assertTrue(read(file.with_suffix('.generation.json'))['ended_with_eos'])
        trie=TokenTrie([[1,2,9],[1,3,9]],1)
        self.assertEqual(trie(0,self.q.torch.tensor([99])),[1])
        with self.assertRaises(ValueError):trie(0,self.q.torch.tensor([99,4]))


if __name__=='__main__':unittest.main()
