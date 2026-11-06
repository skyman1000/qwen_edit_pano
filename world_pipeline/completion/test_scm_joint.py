"""Actual run004 geometry regressions and complete export/audit on CPU.

Fixture installation roles are explicit TEST assumptions. They are not outputs
of the real Qwen checkpoint and cannot establish the new model's visual quality.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write
from .conditional_world_geometry import compile_layout
from .scene_completion import map_plan,view_check
from .scm_joint_geometry import compile_joint,linear_feasible,observed_host_feasible
from .scm_mechanisms import count_states
from .scm_numeric_mechanisms import NumericCountScorer
from .scm_relational_layout import (PROFILES,decorate_joint,host_choices,metric_validator,
    replay_relational,resolve_geometry,solve_relational)
from .scm_typed_layout import initial_boxes,numeric_body
from .scm_world import code_hashes,run
from .scm_world_audit import audit
from .scm_checkpoint import validate_checkpoint
from .test_causal_world_completion import make_case
from .test_scm_world import SyntheticModel,SyntheticScorer

DATA=Path(__file__).with_name('testdata')/'scm_run004_geometry.json'


def fixture_profiles(data,h):
    result={}
    for obj in data['observed']['objects']:
        oid=obj['track_id']
        if oid=='observed_0006':role=('surface','top')
        elif oid in ('observed_0000','observed_0001','observed_0008'):role=('floor','seat')
        elif oid=='observed_0007':role=('floor','none')
        else:role=('seat','none')
        result[oid]=dict(installation=role[0],surface=role[1])
    for obj in h['plan']['additions']:
        t=obj['object_type']
        if t in ('couch','chair'):role=('floor','seat')
        elif t in ('table','desk','bookshelf'):role=('floor','top')
        elif t=='monitor':role=('surface','none')
        elif t=='cushion':role=('seat','none')
        else:role=('floor','none')
        result[obj['id']]=dict(installation=role[0],surface=role[1])
    return result


def fixture_solution(data,h):
    profiles=fixture_profiles(data,h);plan=h['plan'];case=data['case'];obs=data['observed'];plane=data['plane']
    bodies={r['id']:r['body'] for r in h['typed']['metric_records']};selected={};old=initial_boxes(case,obs)
    order=sorted(plan['additions'],key=lambda o:(profiles[o['id']]['installation'] in ('surface','seat'),o['id']))
    for obj in order:
        oid=obj['id'];choices,_=host_choices(oid,profiles,bodies,plan,selected,old,plane,case)
        # Fixed test hosts: prefer sampled new desk/table/seat to avoid inventing
        # a host or reusing the visible tray. No model-quality claim is made.
        selected[oid]=next((c for c in choices if c['kind']=='floor'),None) or next(
            (c for c in choices if c['kind']=='object' and c['target'].startswith('scm_new_') and
             'desk' in c['description']),None) or next(
            (c for c in choices if c['kind']=='object' and c['target'].startswith('scm_new_')),choices[-1])
    solved,raw,solver,resolved,changes=resolve_geometry(case,obs,plan,plane,bodies,selected)
    items,review=compile_layout(raw,solved,map_plan(solved,'semantic_lexicon_v1'),obs,plane,case['hfov'],case['image_size'])
    saved=dict(profiles=profiles,resolved_plan=solved,solver=solver,program_resolutions=changes)
    decorate_joint(review,saved,items,obs)
    return solved,items,review


class ActualGeometryTests(unittest.TestCase):
    def test_optimizer_failure_is_reviewable_feasible_output_not_false_support_infeasibility(self):
        case=dict(context=[],hfov=90,image_size=[64,64]);obs=dict(objects=[])
        plan=dict(additions=[dict(id='a',name='a')],relations=[],placement_dependency_order=['a'])
        bodies={'a':numeric_body(dict(azimuth_deg=180,range_m=3,size=[.3,.6,.3],center_y_m=-1))}
        with patch('qwen_edit_pano.world_pipeline.completion.scm_joint_geometry.minimize',
                   return_value=SimpleNamespace(x=np.array([np.nan,np.nan]),success=False)):
            _,raw,review=compile_joint(case,obs,plan,dict(floor_y_m=-2),bodies,{'a':dict(kind='floor',target=None)})
        self.assertEqual(len(raw['objects']),1);self.assertFalse(review['solver_converged'])
        self.assertFalse(view_check([0.,-1.7,-3.],[.3,.6,.3],90,[64,64])['intersects_input_box_proxy'])
        with patch('qwen_edit_pano.world_pipeline.completion.scm_joint_geometry.linprog',
                   return_value=SimpleNamespace(status=4,success=False,message='numerical test failure')):
            with self.assertRaisesRegex(RuntimeError,'solver failed'):
                linear_feasible(np.array([[1.,0.]]),np.array([0.]))

    def test_unconditioned_child_position_cannot_drag_floor_host_towards_camera(self):
        case=dict(context=[],hfov=90,image_size=[64,64]);obs=dict(objects=[])
        plan=dict(additions=[dict(id=i,name=i) for i in ('desk','monitor')],relations=[],placement_dependency_order=['desk','monitor'])
        bodies=dict(desk=numeric_body(dict(azimuth_deg=90,range_m=3.5,size=[.6,.7,1.2],center_y_m=0)),
            monitor=numeric_body(dict(azimuth_deg=-90,range_m=3.5,size=[.04,.57,.36],center_y_m=-.2)))
        selected=dict(desk=dict(kind='floor',target=None),monitor=dict(kind='object',target='desk'))
        _,raw,review=compile_joint(case,obs,plan,dict(floor_y_m=-2),bodies,selected)
        for row in raw['objects']:
            x=row['range_m']*np.sin(np.deg2rad(row['azimuth_deg']))
            self.assertAlmostEqual(x,3.5,places=7)
        self.assertEqual(review['child_global_proposals_not_used_as_absolute_priors'],['monitor'])

    def test_all_three_real_inventories_preserve_counts_sizes_observations_and_remove_duplicates(self):
        data=read(DATA);before=deepcopy(data)
        for h in data['hypotheses']:
            plan,items,review=fixture_solution(data,h)
            self.assertEqual(len(items),len(h['plan']['additions']))
            original={r['id']:r['body']['size'] for r in h['typed']['metric_records']}
            self.assertTrue(review['geometry_contract_passed'],review)
            self.assertTrue(review['layout_proxy_passed'],review['joint_layout'])
            self.assertEqual(review['joint_layout']['final_overlaps'],[])
            for item in items:
                self.assertEqual(item['size'],original[item['id']])
                self.assertFalse(view_check(item['center'],item['size'],data['case']['hfov'],data['case']['image_size'])['intersects_input_box_proxy'])
            for a in range(len(items)):
                for b in range(a):self.assertGreater(np.linalg.norm(np.array(items[a]['center'])-items[b]['center']),1e-6)
        self.assertEqual(data,before)

    def test_floor_table_cannot_select_tray_but_monitor_can_select_role_compatible_host(self):
        data=read(DATA);h=data['hypotheses'][0];profiles=fixture_profiles(data,h)
        bodies={r['id']:r['body'] for r in h['typed']['metric_records']};old=initial_boxes(data['case'],data['observed'])
        floor,_=host_choices('scm_new_0001',profiles,bodies,h['plan'],{},old,data['plane'],data['case'])
        self.assertEqual([c['kind'] for c in floor],['floor'])
        monitor,_=host_choices('scm_new_0008',profiles,bodies,h['plan'],{},old,data['plane'],data['case'])
        self.assertTrue(any(c['target']=='scm_new_0007' for c in monitor))
        self.assertFalse(any(c['target']=='observed_0007' for c in monitor)) # lamp is not a desk surface

    def test_center_contact_rejects_old_epsilon_only_support(self):
        data=read(DATA);h=data['hypotheses'][0]
        body=next(r['body'] for r in h['typed']['metric_records'] if r['id']=='scm_new_0001')
        old=initial_boxes(data['case'],data['observed'])
        self.assertFalse(observed_host_feasible('scm_new_0001',body,'observed_0006',old,data['plane'],data['case']))

    def test_future_support_cycle_is_excluded_and_missing_host_is_explicit(self):
        profiles={i:dict(installation='surface',surface='top') for i in ('a','b')}
        case=dict(context=[],hfov=90,image_size=[64,64]);plan=dict(additions=[dict(id=i,name=i) for i in profiles])
        bodies={i:numeric_body(dict(azimuth_deg=180,range_m=3,size=[.3,.5,.3],center_y_m=-1)) for i in profiles}
        choices,_=host_choices('b',profiles,bodies,plan,{'a':dict(kind='object',target='b')},{},dict(floor_y_m=-2),case)
        self.assertEqual(len(choices),1);self.assertEqual(choices[0]['kind'],'unspecified')

    def test_infeasible_declared_host_is_preserved_as_unresolved_not_a_fake_success(self):
        data=read(DATA);h=data['hypotheses'][0]
        obj=h['plan']['additions'][0];plan=deepcopy(h['plan']);plan['additions']=[obj]
        b=next(r['body'] for r in h['typed']['metric_records'] if r['id']==obj['id'])
        selected={obj['id']:dict(kind='object',target='observed_0006')}
        out,raw,solver,resolved,changes=resolve_geometry(data['case'],data['observed'],plan,data['plane'],{obj['id']:b},selected)
        self.assertEqual(len(raw['objects']),1);self.assertEqual(len(changes),1)
        self.assertEqual(out['additions'][0]['support']['kind'],'unspecified')
        self.assertEqual(selected[obj['id']]['kind'],'object')

    def test_resolving_one_component_preserves_unrelated_valid_support_edges(self):
        data=read(DATA);h=data['hypotheses'][0];ids=['scm_new_0001','scm_new_0002','scm_new_0008']
        plan=deepcopy(h['plan']);plan['additions']=[o for o in plan['additions'] if o['id'] in ids]
        bodies={r['id']:r['body'] for r in h['typed']['metric_records'] if r['id'] in ids}
        selected={ids[0]:dict(kind='object',target='observed_0006'),ids[1]:dict(kind='floor',target=None),
            ids[2]:dict(kind='object',target=ids[1])}
        _,raw,_,resolved,changes=resolve_geometry(data['case'],data['observed'],plan,data['plane'],bodies,selected)
        self.assertEqual(len(raw['objects']),3);self.assertEqual([c['id'] for c in changes],[ids[0]])
        self.assertEqual(resolved[ids[2]],selected[ids[2]])

    def test_impossible_AABB_separation_remains_a_review_signal_without_deleting_objects(self):
        case=dict(context=[dict(id='old',description='tiny fixed shelf')],hfov=90,image_size=[64,64])
        obs=dict(objects=[dict(track_id='old',center_local_m=dict(value=[0.,-1.,-3.]),size_aabb_local_m=dict(value=[.1,.1,.1]))])
        plan=dict(additions=[dict(id=i,name=i) for i in ('a','b')],relations=[],placement_dependency_order=['a','b'])
        bodies={i:numeric_body(dict(azimuth_deg=180,range_m=3,size=[1.,.5,1.],center_y_m=-.7)) for i in ('a','b')}
        selected={i:dict(kind='object',target='old') for i in bodies}
        _,raw,review=compile_joint(case,obs,plan,dict(floor_y_m=-2),bodies,selected)
        self.assertEqual(len(raw['objects']),2);self.assertTrue(review['final_overlaps'])
        self.assertFalse(review['physical_geometry_verified'])

    def test_unmeasured_wall_floor_and_empty_inventory_keep_honest_status(self):
        with self.assertRaisesRegex(ValueError,'center_y_m'):
            metric_validator(dict(azimuth_deg=180,range_m=3,size=[.5,1,.5]),dict(installation='wall'),dict(floor_y_m=-2))
        plan=dict(additions=[],relations=[],placement_dependency_order=[]);case=dict(context=[],hfov=90,image_size=[64,64])
        out,raw,solver=compile_joint(case,dict(objects=[]),plan,dict(floor_y_m=None),{}, {})
        self.assertEqual(raw,dict(objects=[]));self.assertFalse(solver['physical_geometry_verified'])


class NumericEventTests(unittest.TestCase):
    def test_values_are_canonical_and_long_sentence_contract_is_preserved_for_control(self):
        values=count_states('couch',16,'json_integer_v1')
        self.assertEqual([s['description'] for s in values],[str(n) for n in range(17)])
        self.assertEqual([s['id'] for s in values],[str(n) for n in range(17)])
        self.assertIn('instances',count_states('couch',1)[1]['description'])

    def test_actual_HF_numeric_complete_events_match_full_forward(self):
        from .test_scm_typed import ActualHFTests
        ActualHFTests.setUpClass();q=ActualHFTests.q;scorer=NumericCountScorer(q)
        states=[dict(id=str(n),description=str(n)) for n in (0,1,10)]
        result=scorer.score('Whole-room count. Answer with one JSON integer.',{'B':'living room'},states)
        prefix=q.processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=result['prompt'])])],tokenize=False,add_generation_prompt=True)
        ids=q.processor.tokenizer.encode(prefix,add_special_tokens=False)
        for r in result['candidate_sequences']:
            self.assertEqual(r['text'],r['id'])
            full=q.torch.tensor([ids+r['token_ids']])
            with q.torch.inference_mode():logits=q.model(input_ids=full,use_cache=False,logits_to_keep=0).logits
            logp=logits[0,len(ids)-1:-1].float().log_softmax(-1)
            ref=logp.gather(1,q.torch.tensor(r['token_ids'])[:,None]).sum().item()
            self.assertAlmostEqual(ref,r['log_probability'],places=4)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1.)
        with self.assertRaises(ValueError):scorer.score('count',{},[dict(id='1',description='Exactly 1 chair.')])


def synthetic_choice(model,image,prompt,candidates,path):
    choice=next((c for c in candidates if c.get('installation')=='floor' and c.get('surface')=='none'),None)
    choice=choice or next((c for c in candidates if c.get('kind')=='floor'),candidates[-1])
    path.write_text(json.dumps(choice));write(path.with_suffix('.json'),choice)
    write(path.with_suffix('.request.json'),dict(method='finite_token_trie_greedy_layout_v1',candidates=candidates))
    return choice


class PipelineTests(unittest.TestCase):
    def test_empty_joint_sample_exports_observations_without_layout_queries(self):
        class Scorer(SyntheticScorer):
            count_event_contract='json_integer_v1'
            def score(self,description,parents,states):
                if states[0]['id'].isdigit():
                    values=[0.]*len(states);values[2 if 'type lamp.' in description else 0]=1.
                    return dict(probabilities=dict(zip([s['id'] for s in states],values)),rounds=[])
                return super().score(description,parents,states)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64];out=root/'out';out.mkdir()
            model=SyntheticModel(case)
            with patch('qwen_edit_pano.world_pipeline.completion.scm_relational_layout.finite_choice',side_effect=AssertionError('No layout query expected')):
                result=run(model,Scorer(),case,out,max_count=3,generic_types=3,worlds=1,seed=0,layout_backend='joint')
            self.assertEqual(result['hypotheses'][0]['new_instances'],0,result)
            self.assertEqual(read(out/'case/G_full.json')['objects'],obs['objects'])
            args=dict(generic_types=3,max_count=3,worlds=1,seed=0,max_objects=256,layout_backend='joint',mechanism_scoring='numeric')
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(out)['status'],'scm_postflight_contract_passed')

    def test_installation_queries_object_support_export_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64]
            plan=dict(scene_hypothesis='Synthetic test',additions=[dict(id=i,name=n,object_type=n,
                reason='test',placement='pending',support=dict(kind='unspecified',target=None))
                for i,n in [('host','desk'),('child','monitor')]],relations=[],placement_dependency_order=['host','child'])
            class Model:
                def ask(self,image,prompt,path,**options):
                    value=dict(azimuth_deg=180,range_m=3,size=[1,1,1] if path.stem=='metric_host' else [.4,.5,.3],center_y_m=-1)
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
            def choose(model,image,prompt,candidates,path):
                if path.stem.startswith('profile_'):
                    role=dict(installation='floor',surface='top') if path.stem=='profile_host' else dict(installation='surface',surface='none')
                    choice=next(c for c in candidates if c==role)
                else:choice=next(c for c in candidates if c.get('target')=='host')
                path.write_text(json.dumps(choice));write(path.with_suffix('.json'),choice)
                write(path.with_suffix('.request.json'),dict(method='finite_token_trie_greedy_layout_v1',candidates=candidates))
                return choice
            plane=dict(floor_y_m=-2)
            out,_,items,review=solve_relational(Model(),None,case,obs,plan,plane,root,dict(seed=0),choose)
            self.assertTrue(review['layout_proxy_passed'],review)
            self.assertEqual(out['additions'][1]['support']['target'],'host')
            self.assertEqual(len(items),2)
            replayed,_=replay_relational(case,obs,plan,plane,read(root/'joint_layout.json'))
            self.assertEqual(out,replayed)

    def test_numeric_semantics_joint_export_visuals_audit_and_tampering(self):
        class Scorer(SyntheticScorer):count_event_contract='json_integer_v1'
        class Model(SyntheticModel):
            def ask(self,image,prompt,path,**options):
                if path.stem.startswith('metric_'):
                    value=dict(azimuth_deg=180,range_m=3,size=[.3,.6,.3],center_y_m=-1)
                    path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
                if getattr(self,'forbid_semantics',False):raise AssertionError('Unexpected semantic LM call during reuse')
                return super().ask(image,prompt,path,**options)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,obs=make_case(root);case['image_size']=[64,64];out=root/'out';out.mkdir()
            with patch('qwen_edit_pano.world_pipeline.completion.scm_relational_layout.finite_choice',synthetic_choice):
                record=run(Model(case),Scorer(),case,out,max_count=3,generic_types=3,worlds=2,seed=0,layout_backend='joint')
            self.assertEqual(record['status'],'posterior_samples_exported_needs_review',record)
            self.assertTrue(record['world_state_ready'],record)
            self.assertTrue(all(h['layout_proxy_passed'] for h in record['hypotheses']))
            self.assertEqual(read(out/'case/G_full.json')['objects'][:len(obs['objects'])],obs['objects'])
            args=dict(generic_types=3,max_count=3,worlds=2,seed=0,max_objects=256,layout_backend='joint',mechanism_scoring='numeric')
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(out)['status'],'scm_postflight_contract_passed')
            validate_checkpoint(out,case,3,3,'numeric')
            with self.assertRaises(ValueError):validate_checkpoint(out,case,3,3,'continuation')
            reused=root/'reused';reused.mkdir();model=Model(case);model.forbid_semantics=True
            with patch('qwen_edit_pano.world_pipeline.completion.scm_relational_layout.finite_choice',synthetic_choice):
                second=run(model,None,case,reused,max_count=3,generic_types=3,worlds=2,seed=0,
                    layout_backend='joint',semantic_source=out)
            self.assertEqual(second['status'],'posterior_samples_exported_needs_review',second)
            write(reused/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            self.assertEqual(audit(reused)['status'],'scm_postflight_contract_passed')
            self.assertEqual(read(out/'case/hypothesis_01/posterior_sample.json'),read(reused/'case/hypothesis_01/posterior_sample.json'))
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments={**args,'mechanism_scoring':'continuation'}))
            with self.assertRaisesRegex(ValueError,'contract'):audit(out)
            write(out/'config.json',dict(case=case,code_sha256=code_hashes(),arguments=args))
            file=out/'case/hypothesis_01/joint_layout.json';saved=read(file);broken=deepcopy(saved)
            broken['profile_records'][0]['id']='invented';write(file,broken)
            with self.assertRaises(ValueError):audit(out)
            write(file,saved)
            specfile=out/'case/semantic_model.json';spec=read(specfile);spec['count_event_contract']='sentence_v1';write(specfile,spec)
            with self.assertRaisesRegex(ValueError,'contract'):audit(out)


if __name__=='__main__':unittest.main()
