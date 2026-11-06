"""Meaningful finite-inference, structural, geometric and export regressions.

Fixtures and tiny random HF models test code, never real hidden-world quality.
"""
from copy import deepcopy
from contextlib import redirect_stderr,redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import new_output,read,sha,validate_new_output,write
from . import scene_program_world
from .conditional_world_geometry import support_plane
from .scene_program import (enumerate_assignments,instantiate,parameterize,program_answer,
    registry_answer,sample_modes)
from .scene_program_audit import audit
from .scene_program_fixture import fixture
from .scene_program_geometry import compile_program,relative_quadratic
from .scene_program_scores import aligned_events,finite_scores,verify_scores
from .scene_program_world import code_hashes,run,solve_assignment
from .test_causal_world_completion import make_case


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.case,self.obs=make_case(self.root);self.case['image_size']=[64,64]
        self.data=fixture(self.case);self.registry=registry_answer(self.data['registry'],self.case,3)
        self.config=self.registry['configurations'][0]
        self.program=program_answer(self.data['programs']['c0'],self.case,self.config)
        self.tables=self.data['tables']['c0'];self.plane=support_plane(self.obs,[self.case['context'][0]['id']])

    def test_joint_assignments_and_do_replace_mechanism_not_just_conditioning(self):
        rows=enumerate_assignments(self.program,self.tables)
        self.assertAlmostEqual(sum(r['probability'] for r in rows),1.)
        self.assertEqual(len(rows),6)
        p=sum(r['probability'] for r in rows if r['modes'][1]==1)
        self.assertAlmostEqual(p,.1*.4+.6*.8+.3*.5)
        do=enumerate_assignments(self.program,self.tables,{1:1})
        # Intervention on child cuts its mechanism. The parent keeps its prior.
        for i,mass in enumerate((.1,.6,.3)):
            self.assertAlmostEqual(sum(r['probability'] for r in do if r['modes'][0]==i),mass)
        conditioned=[r for r in rows if r['modes'][1]==1]
        conditioned_parent=[sum(r['probability']/p for r in conditioned if r['modes'][0]==i) for i in range(3)]
        self.assertFalse(np.allclose(conditioned_parent,[.1,.6,.3]))

    def test_shared_noise_replay_and_mechanism_intervention_propagation(self):
        self.assertEqual(sample_modes(self.program,self.tables,72),sample_modes(self.program,self.tables,72))
        for mode in range(3):
            sample=sample_modes(self.program,self.tables,72,{0:mode})
            self.assertEqual(sample['modes'][0],mode)
            self.assertEqual(sample['noise'],sample_modes(self.program,self.tables,72)['noise'])
            self.assertEqual(json.loads(json.dumps(sample)),sample)

    def test_joint_mode_changes_presence_variant_size_and_host(self):
        a=instantiate(self.program,[1,0],self.case,self.obs,self.plane)
        b=instantiate(self.program,[2,0],self.case,self.obs,self.plane)
        self.assertEqual(len(a[0]['additions']),2);self.assertEqual(len(b[0]['additions']),1)
        lamp='sp_g00_light';self.assertEqual(a[1][lamp]['size'],[.25,.45,.25])
        self.assertEqual(b[1][lamp]['size'],[.4,1.6,.4])
        self.assertEqual(a[0]['additions'][1]['support']['target'],'sp_g00_table')
        self.assertEqual(b[0]['additions'][0]['support']['kind'],'floor')
        empty=instantiate(self.program,[0,0],self.case,self.obs,self.plane)
        self.assertEqual(empty[0]['additions'],[])

    def test_new_host_translation_propagates_child_prior_without_extra_LM_calls(self):
        original=deepcopy(self.program)
        before=instantiate(original,[1,0],self.case,self.obs,self.plane)[3]
        original['groups'][0]['modes'][1]['objects'][0]['offset_m'][0]+=1.4
        after=instantiate(original,[1,0],self.case,self.obs,self.plane)[3]
        for oid in ('sp_g00_table','sp_g00_light'):
            self.assertAlmostEqual(after[oid][0]-before[oid][0],1.4)

    def test_invalid_reference_cycle_size_and_parent_graph_are_rejected(self):
        for mutation in ('missing_host','cycle','size','parents','duplicate_key'):
            raw=deepcopy(self.data['programs']['c0']);objects=raw['groups'][0]['modes'][0]['objects']
            if mutation=='missing_host':objects[1]['anchor']='ghost';objects[1]['support']['target']='ghost'
            elif mutation=='cycle':objects[0]['anchor']='light'
            elif mutation=='size':objects[0]['size_m'][0]=-1
            elif mutation=='parents':raw['groups'][0]['parents']=[1]
            else:objects[1]['key']=objects[0]['key']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):program_answer(raw,self.case,self.config)

    def test_missing_floor_keeps_explicit_Y_and_reports_unresolved(self):
        p=instantiate(self.program,[2,0],self.case,self.obs,dict(floor_y_m=None))
        self.assertEqual(p[0]['additions'][0]['support']['kind'],'unspecified')
        self.assertEqual(p[4],['sp_g00_light'])

    def test_relative_layout_preserves_new_host_and_immutable_state(self):
        old=deepcopy(self.obs)
        solved=solve_assignment(self.case,self.obs,self.plane,self.program,[1,1],dict(region_keep=3,beam_width=4,rounds=8))
        self.assertEqual(old,self.obs);self.assertEqual(len(solved['items']),4)
        self.assertTrue(solved['review']['geometry_contract_passed'])
        self.assertEqual(solved['review']['program_solver']['final_overlaps'],[])
        items={o['id']:o for o in solved['items']}
        lamp,table=items['sp_g00_light'],items['sp_g00_table']
        self.assertAlmostEqual(lamp['center'][1]-table['center'][1],(.45+.6)/2)
        self.assertTrue(all(abs(lamp['center'][d]-table['center'][d])<=table['size'][d]/2+1e-7 for d in (0,2)))
        for oid,body in solved['bodies'].items():self.assertEqual(items[oid]['size'],body['size'])

    def test_relative_objective_retains_requested_offsets(self):
        plan,bodies,groups,placements,_=instantiate(self.program,[1,1],self.case,self.obs,self.plane)
        ids=list(bodies);old={o['track_id']:dict(center=o['center_local_m']['value']) for o in self.obs['objects']}
        matrix,target,energy,start=relative_quadratic(ids,bodies,old)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(),0)
        self.assertAlmostEqual(energy(start),0.,places=10)

    def test_unmapped_categories_remain_in_complete_export(self):
        program=deepcopy(self.program)
        program['groups'][0]['modes'][2]['objects'][0]['category']='unlisted_test_object'
        program['groups'][0]['modes'][2]['objects'][0]['name']='unlisted_test_entity'
        solved=solve_assignment(self.case,self.obs,self.plane,program,[2,0],dict(region_keep=2,beam_width=2,rounds=4))
        from .scene_completion import export_states
        full,_,unmapped=export_states(self.obs,solved['plan'],solved['items'],solved['mappings'],solved['review'])
        self.assertEqual(len(full['objects']),len(self.obs['objects'])+1)
        self.assertEqual(len(unmapped),1);self.assertFalse(full['objects'][-1]['category_id']['valid'])

    def test_visible_host_infeasibility_rejects_complete_recipe_not_child_deletion(self):
        old_id=self.case['context'][0]['id']
        raw=deepcopy(self.data['programs']['c0']);raw['groups']=raw['groups'][:1]
        lamp=raw['groups'][0]['modes'][0]['objects'][1]
        lamp['anchor']=old_id;lamp['support']['target']=old_id
        raw['groups'][0]['modes']=[dict(description='test object on a wholly visible host',objects=[lamp])]
        program=program_answer(raw,self.case,self.config)
        with self.assertRaisesRegex(ValueError,'No off-frustum layout for complete group'):
            solve_assignment(self.case,self.obs,self.plane,program,[1],dict(region_keep=3,beam_width=4,rounds=8))
        empty=solve_assignment(self.case,self.obs,self.plane,program,[0],dict(region_keep=3,beam_width=4,rounds=8))
        self.assertEqual(empty['items'],[])
        self.assertEqual(len(program['groups'][0]['modes'][1]['objects']),1)

    def test_lexicographic_occupancy_repairs_overlapping_fixed_inventory(self):
        raw=deepcopy(self.data['programs']['c0']);raw['groups']=raw['groups'][:1]
        first=raw['groups'][0]['modes'][1]['objects'][0]
        first['size_m']=[.8,1.,.8];first['offset_m']=[0,0,-3]
        second=deepcopy(first);second['key']='other'
        raw['groups'][0]['modes']=[dict(description='explicit overlapping solver test',objects=[first,second])]
        program=program_answer(raw,self.case,self.config)
        solved=solve_assignment(self.case,self.obs,self.plane,program,[1],dict(region_keep=3,beam_width=4,rounds=8))
        self.assertTrue(solved['review']['program_solver']['initial_overlaps'])
        self.assertEqual(solved['review']['program_solver']['final_overlaps'],[])
        self.assertEqual(len(solved['items']),2)
        self.assertTrue(all(o['size']==[.8,1.,.8] for o in solved['items']))

    def test_optimizer_failure_retains_feasible_point_and_review_status(self):
        plan,bodies,groups,placements,_=instantiate(self.program,[2,0],self.case,self.obs,self.plane)
        with patch('qwen_edit_pano.world_pipeline.completion.scene_program_geometry.minimize',
                   return_value=SimpleNamespace(x=np.array([np.nan,np.nan]),success=False)):
            raw,review=compile_program(self.case,self.obs,plan,bodies,groups,placements)
        self.assertEqual(len(raw['objects']),1);self.assertFalse(review['solver_converged'])

    def test_parameterized_parent_context_is_explicit_and_domain_complete(self):
        calls=[]
        class Scorer:
            def score(_,description,parents,states):
                calls.append(deepcopy(parents));n=len(states)
                return dict(probabilities={str(i):1/n for i in range(n)})
        tables=parameterize(self.program,self.registry,Scorer(),lambda *args:None)
        self.assertEqual(len(calls),4)
        self.assertEqual(calls[0]['parent_groups'],{})
        self.assertEqual(calls[-1]['parent_groups']['g0']['mode'],self.program['groups'][0]['modes'][2]['description'])
        self.assertEqual(calls[0]['observed_context'],self.registry['instances'])
        self.assertAlmostEqual(sum(r['probability'] for r in enumerate_assignments(self.program,tables)),1)

    def test_end_to_end_all_states_export_2to1_replay_and_tamper_rejection(self):
        output=self.root/'output';output.mkdir();budgets=dict(configurations=3,groups=3,modes=2,max_new=24)
        search=dict(region_keep=3,beam_width=4,rounds=8)
        write(output/'config.json',dict(case=self.case,code_sha256=code_hashes(),budgets=budgets,search=search,arguments=dict(worlds=3)))
        record=run(None,self.case,output,budgets,search,fixture=self.data)
        self.assertTrue(record['world_state_ready'],record)
        self.assertEqual(record['candidate_assignments'],10)
        self.assertEqual(len(record['worlds']),3)
        self.assertTrue(all(w['new_instances']>0 for w in record['worlds']))
        result=audit(output,allow_fixture=True);self.assertEqual(result['status'],'scene_program_contract_passed')
        with self.assertRaisesRegex(ValueError,'not a model prediction'):audit(output)
        first=output/self.case['id']/record['worlds'][0]['folder']/'G_full.json'
        graph=read(first);graph['objects'][-1]['center_local_m']['value'][0]+=1.;write(first,graph)
        with self.assertRaisesRegex(ValueError,'manifest changed'):audit(output,allow_fixture=True)

    def test_model_path_raw_binding_and_score_records_audit(self):
        # Exercise the actual proposal/repair/parameterization path, with
        # explicit synthetic token-score records instead of loading 8B weights.
        data=self.data
        class Replies:
            def ask(_,image,prompt,path,**kwargs):
                value=data['registry'] if path.stem.startswith('registry') else data['programs'][path.parent.name]
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
                write(path.with_suffix('.request.json'),dict(prompt=prompt,synthetic_test=True))
                return deepcopy(value)
        def scores(q,image,field,parents,alternatives):
            n=len(alternatives);rotations=[]
            for shift in range(n):
                events=[dict(code=i,token_ids=[i,99],token_log_probabilities=[-float(i+1),-1.],
                             log_probability=-float(i+2)) for i in range(n)]
                vals=np.exp(-np.arange(n,dtype=float));vals/=vals.sum()
                rotations.append(dict(prompt='explicit synthetic test',events=events,image_used=image is not None,
                    probabilities=vals.tolist(),code_to_alternative=[(i+shift)%n for i in range(n)]))
            # Cyclic pooling makes these deliberately code-biased events uniform.
            return dict(probabilities={str(i):1/n for i in range(n)},rotations=rotations,
                parents=deepcopy(parents),alternatives=deepcopy(alternatives),field=field,
                image_used=image is not None,source='explicit_synthetic_test')
        output=self.root/'model_path';output.mkdir();budgets=dict(configurations=3,groups=3,modes=2,max_new=24)
        search=dict(region_keep=2,beam_width=2,rounds=4)
        write(output/'config.json',dict(case=self.case,code_sha256=code_hashes(),budgets=budgets,search=search,arguments=dict(worlds=3)))
        with patch('qwen_edit_pano.world_pipeline.completion.scene_program_world.finite_scores',scores),\
             patch('qwen_edit_pano.world_pipeline.completion.scene_program_scores.finite_scores',scores):
            result=run(Replies(),self.case,output,budgets,search)
        self.assertTrue(result['world_state_ready'],result)
        self.assertEqual(audit(output)['status'],'scene_program_contract_passed')


class OutputPreflightTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        root_patch=patch('qwen_edit_pano.world_pipeline.common.ROOT',self.root)
        root_patch.start();self.addCleanup(root_patch.stop)
        self.base=self.root/'qwen_edit_pano/outputs';self.base.mkdir(parents=True)

    def test_validation_and_creation_share_policy_for_relative_and_absolute_paths(self):
        output=self.base/'new_run'
        relative=Path(os.path.relpath(output,Path.cwd()))
        self.assertEqual(validate_new_output(relative),output)
        self.assertEqual(validate_new_output(output),output)
        self.assertFalse(output.exists())
        self.assertEqual(new_output(relative),output)
        self.assertTrue(output.is_dir())

    def test_invalid_output_one_is_rejected_before_case_or_model_in_both_modes(self):
        # Exact failed job 71404 argument. Preflight must not accept it.
        for mode in ([],['--preflight']):
            stderr=io.StringIO()
            with self.subTest(mode=mode),patch('sys.argv',['scene_program_world','--output','1',*mode]),\
                 patch.object(scene_program_world,'load_case') as load,\
                 patch.object(scene_program_world,'Qwen') as model,redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as result:scene_program_world.main()
                self.assertEqual(result.exception.code,2)
                load.assert_not_called();model.assert_not_called()
            self.assertIn("Output '1' resolves to",stderr.getvalue())
            self.assertIn('qwen_edit_pano/outputs/',stderr.getvalue())

    def test_valid_preflight_reports_canonical_output_without_creating_it(self):
        output=self.base/'valid';stdout=io.StringIO()
        args=['scene_program_world','--output',str(output),'--model',str(self.root),'--preflight']
        with patch('sys.argv',args),patch.object(scene_program_world,'load_case',return_value=dict(id='fixture',context=[])),\
             patch.object(scene_program_world,'Qwen') as model,redirect_stdout(stdout):
            scene_program_world.main();model.assert_not_called()
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['output'],str(output));self.assertTrue(result['output_policy_checked'])
        self.assertFalse(result['model_loaded']);self.assertFalse(output.exists())

    def test_existing_output_is_rejected_and_its_contents_preserved(self):
        output=new_output(self.base/'existing');marker=output/'result.json';marker.write_text('original')
        for check in (validate_new_output,new_output):
            with self.assertRaises(FileExistsError):check(output)
        self.assertEqual(marker.read_text(),'original')

    def test_allowed_roots_and_symlink_escapes_are_rejected(self):
        for root in (self.root/'qwen_edit_pano/data',self.base,self.root/'research/audits'):
            with self.assertRaises(ValueError):validate_new_output(root)
        outside=self.root/'other';outside.mkdir();link=self.base/'escape';link.symlink_to(outside,target_is_directory=True)
        for check in (validate_new_output,new_output):
            with self.assertRaises(ValueError):check(link/'new')
        self.assertFalse((outside/'new').exists())
        dangling=self.base/'dangling';dangling.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaises(FileExistsError):validate_new_output(dangling)


class HFMultimodalTests(unittest.TestCase):
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
        cls.q=SimpleNamespace(model=Qwen3VLForConditionalGeneration(config).eval(),processor=processor,torch=torch)

    def test_actual_multimodal_forward_matches_independent_full_sequence(self):
        q=self.q;image=Image.new('RGB',(64,64),'red');prompt='Return the index 0 or 1 for the image.'
        result=aligned_events(q,image,prompt,2)
        prefix=q.processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])],
            tokenize=False,add_generation_prompt=True)
        encoded=q.processor(text=[prefix],images=[image],return_tensors='pt');encoded.pop('token_type_ids',None)
        length=encoded.input_ids.shape[1];event=result['events'][1];ids=q.torch.tensor([event['token_ids']])
        full=dict(encoded);full['input_ids']=q.torch.cat((encoded.input_ids,ids),dim=1)
        full['attention_mask']=q.torch.cat((encoded.attention_mask,q.torch.ones_like(ids)),dim=1)
        with q.torch.inference_mode():logits=q.model(**full,use_cache=False,logits_to_keep=0).logits
        reference=logits[0,length-1:-1].float().log_softmax(-1).gather(1,ids[0,:,None]).sum().item()
        self.assertAlmostEqual(event['log_probability'],reference,places=4)
        self.assertEqual(event['token_ids'][-1],q.processor.tokenizer.eos_token_id)

    def test_cyclic_mechanism_records_replay_and_corruption_rejected(self):
        result=finite_scores(self.q,None,'Choose a joint recipe.',{'configuration':'test'},['absent','desk with monitor'])
        self.assertAlmostEqual(sum(result['probabilities'].values()),1.)
        self.assertEqual(len(verify_scores(result)),2)
        result['rotations'][0]['events'][0]['token_log_probabilities'][0]-=1
        with self.assertRaises(ValueError):verify_scores(result)


if __name__=='__main__':unittest.main()
