"""v7 CPU contracts and saved-output regression fixtures, without model calls."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write,sha
from . import assembly_realization as model,assembly_realization_experiment as experiment
from . import assembly_realization_geometry as geometry,assembly_activation_geometry as v6geometry
from . import test_assembly as fixture
from .test_activation import ActivationFixture
from .test_modular import proposals
from .world_scm import forward,validate
from .assembly_world import run
from .conditional_world_geometry import support_plane


def raw_proposals():
    value=proposals()
    for group in value['alternatives'][0]['groups']:group['realization_mode']='new_group'
    group=deepcopy(value['alternatives'][0]['groups'][0]);group['realization_mode']='observed_extension';group['observed_refs']=['obs_000']
    for obj in group['new_objects']:
        obj['key']='extension_'+obj['key']
        if obj['attachment']['host']:obj['attachment']['host']='extension_'+obj['attachment']['host']
    value['alternatives'][0]['groups'].append(group)
    value['alternatives'][0]['relations']=[dict(source='extension_table',target='obs_000',kind='near',reason='serves observed anchor')]
    return value


class RealizationFixture(ActivationFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);name=path.stem
        if name.startswith('furniture_'):value=raw_proposals()
        elif name.startswith('activation_'):
            if image is not None:raise AssertionError('Image in activation mechanism')
            h=json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\n')[0])
            menu=json.loads(prompt.split('\nAVAILABLE_REALIZATION_CHANNELS: ')[1].split('\n')[0]);j=int(name.split('_')[-1])
            active=[]
            for channel in menu:
                if channel['realization_mode']=='observed_extension':active.append(channel['id'])
                elif 'reading' in h['functions'] and channel['function']=='reading' and j%2:active.append(channel['id'])
                elif 'reading' not in h['functions'] and channel['function']=='storage':active.append(channel['id'])
            if j==0:active=[]
            value=dict(active_channels=active)
        else:return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,canonical_image_sha256=None if image is None else 'fixture'))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class RealizationTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def bundle(self):
        root=self.temp/'realization'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=2,draws=4)
        for backend in ('qwen','api'):
            result=experiment.propose(root,backend,RealizationFixture(backend));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for backend in ('qwen','api'):experiment.mechanisms(root,backend,RealizationFixture(backend))
        experiment.pool_mechanisms(root);return root

    def fitted(self,policy='functional'):
        root=self.bundle();domain=read(root/'shared/domain.json')
        fitted=model.prepare_model(domain['base_model'],read(root/'shared/case.json'),dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',realization_policy=policy),None)
        return root,fitted

    def test_sparse_joint_law_is_exact_dense_dirichlet_predictive(self):
        channels=['a','b','c'];draws=['101','101','000',None];alpha=1.
        table=model.estimate_activation(draws,channels,alpha)
        for bits in itertools.product('01',repeat=3):
            state=''.join(bits);dense=(draws.count(state)+alpha/8)/(3+alpha)
            sparse=table['prior_weight']/8+table['empirical_component_weights'].get(state,0.)
            self.assertAlmostEqual(dense,sparse)
        with self.assertRaises(ValueError):model.estimate_activation([None],channels)

    def test_existing_extension_can_survive_without_new_zone_and_both_can_coexist(self):
        root,m=self.fitted();validate(m);noise={n['id']:.2 for n in m['nodes']}
        extension='reading__observed_extension';new='reading__new_group'
        factual=model.active_world(m,forward(m,noise,{'A_'+extension:'1','A_'+new:'1'}))
        after=model.active_world(m,forward(m,noise,{'A_'+extension:'1','A_'+new:'0'}))
        self.assertTrue(any(m['state_slot'][o['furniture_state']]==extension for o in after['objects']))
        self.assertFalse(any(m['state_slot'][o['furniture_state']]==new for o in after['objects']))
        self.assertGreater(len(factual['objects']),len(after['objects']))

    def test_exact_marginals_match_sampling_and_h_interventions_change_inventory(self):
        root,m=self.fitted();law=model.marginals(m)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1.)
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        self.assertLess(law['category_presence']['book'],law['expected_category_counts']['book'])
        empty=next(h['id'] for h in m['profiles'] if not h['functions']);reading=next(h['id'] for h in m['profiles'] if h['functions']==['reading'])
        self.assertGreater(model.marginals(m,reading)['activation_probabilities']['reading__new_group']-model.marginals(m,empty)['activation_probabilities']['reading__new_group'],.2)
        hist={};samples=model.sample_worlds(m,count=1200)
        for w in samples:
            self.assertEqual(forward(m,w['noise']),w['structural_state']);n=str(len(w['objects']));hist[n]=hist.get(n,0)+1
        for n,p in law['hidden_count_probabilities'].items():self.assertAlmostEqual(hist.get(n,0)/len(samples),p,delta=.045)

    def test_h_cut_preserves_channel_and_object_marginals(self):
        root,m=self.fitted();d=read(root/'shared/domain.json')
        cut=model.prepare_model(d['base_model'],read(root/'shared/case.json'),dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',realization_policy='function_free'),None)
        for field in ('activation_probabilities','object_presence'):
            a,b=model.marginals(m)[field],model.marginals(cut)[field]
            for k,p in a.items():self.assertAlmostEqual(p,b[k])
        self.assertTrue(all('H' not in n['parents'] for n in cut['nodes'] if n['id']!='H'))

    def test_diagnostic_profiles_do_not_acquire_observational_prior_mass(self):
        root,m=self.fitted();domain=read(root/'shared/domain.json');eligible=set(domain['recognition_profile_ids'])
        extra={h['id'] for h in domain['profiles']}-eligible;self.assertTrue(extra)
        for backend in ('qwen','api','pooled'):
            row=read(root/'shared/mechanisms'/(backend+'.json'))['recognition']
            self.assertTrue(all(row['probabilities'][h]==0 for h in extra));self.assertAlmostEqual(sum(row['probabilities'].values()),1.)
        for h in extra:self.assertAlmostEqual(sum(model.marginals(m,h)['hidden_count_probabilities'].values()),1.)

    def test_explicit_realization_modes_and_observed_binding_are_required(self):
        raw=raw_proposals();h=model.profile('z0',['reading'])
        raw['alternatives'][0]['groups'][-1]['observed_refs']=[]
        with self.assertRaisesRegex(ValueError,'observed_refs'):model.furniture_answer(raw,self.case,h)
        raw=raw_proposals();raw['alternatives'][0]['relations']=[]
        with self.assertRaisesRegex(ValueError,'explicitly relate'):model.furniture_answer(raw,self.case,h)

    def test_failed_stage_does_not_cancel_other_fixed_proposal_stages(self):
        root=self.temp/'partial'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):experiment.prepare(root,self.root)
        backend=RealizationFixture('qwen');original=backend.ask;seen=[]
        def fail_one(image,prompt,path,**kw):
            seen.append(Path(path).stem)
            if Path(path).stem=='furniture_z0_surrounding' or Path(path).stem=='furniture_z0_surrounding_repair':raise ValueError('deliberate isolated failure')
            return original(image,prompt,path,**kw)
        backend.ask=fail_one;result=experiment.propose(root,'qwen',backend)
        self.assertEqual(result['status'],'partial');self.assertIn('furniture_z0_belongings',seen);self.assertGreater(result['programs'],0)

    def test_cli_records_dependency_errors_in_new_error_directory(self):
        root=self.temp/'missing_bundle';root.mkdir()
        result=subprocess.run([sys.executable,'-m','qwen_edit_pano.world_pipeline.completion.assembly_realization_experiment','pool','--root',str(root)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,2,result.stderr)
        error=read(root/'phase_errors/pool_None.json');self.assertEqual(error['action'],'pool')
        self.assertIn('FileNotFoundError',error['error'])

    def test_full_six_branches_keep_observations_and_shared_control_samples(self):
        root=self.bundle();before=read(self.root/'qwen/observer3d/G_obs.json')
        for label in ('end_to_end','layout_control','function_free'):
            for backend in ('qwen','api'):
                s=run(RealizationFixture(backend),root/label,backend);self.assertNotEqual(s['status'],'failed',s)
                self.assertEqual(len(s['worlds'])+len(s['failures']),2)
                for record in s['worlds']+s['blocked_worlds']:
                    full=read(root/label/backend/'completion'/record['folder']/record.get('state_file','G_full.json'))
                    self.assertEqual(full['objects'][:len(before['objects'])],before['objects'])
        paths=[root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')]
        self.assertEqual(sha(paths[0]),sha(paths[1]))
        from .assembly_activation_visual import report
        page=report(root);self.assertTrue(page.is_file());self.assertIn('v7',page.read_text())

    def test_hard_depth_ordering_repairs_initial_region_without_changing_sizes_or_observation(self):
        observed=deepcopy(self.observed);world=dict(objects=[dict(id='cabinet',key='cabinet',name='cabinet',category='cabinet',variant='cabinet',
            distinct_instance_reason='fixture',placement_hint='farther than observed',group='g0',attachment=dict(kind='floor',host=None))],
            groups=[dict(key='g0',function='storage')],relations=[dict(source='cabinet',target='obs_000',kind='in_front_of',reason='larger camera Z')],hypothesis={},configuration='fixture')
        raw=dict(groups=[dict(id='g0',region='rear',range_m=3.,objects=[dict(id='cabinet',full_size_m=[.6,1.,.4],yaw_deg=0,offset_m=[0,0,0])])])
        layout=geometry.layout_answer(raw,world,90);plane=support_plane(observed,['obs_000'])
        old=v6geometry.compile_layout(self.case,observed,world,layout,plane);new=geometry.compile_layout(self.case,observed,world,layout,plane)
        self.assertFalse(old['review']['geometry_contract_passed']);self.assertTrue(new['review']['geometry_contract_passed'],new['review'])
        self.assertEqual(old['items'][0]['size'],new['items'][0]['size']);self.assertEqual(observed,self.observed)

    def test_functional_group_does_not_force_different_support_roots_to_share_hidden_face(self):
        world=dict(objects=[dict(id='chair',group='g',attachment=dict(kind='floor',host=None)),
            dict(id='picture',group='g',attachment=dict(kind='wall',host=None)),
            dict(id='book',group='g',attachment=dict(kind='top',host='chair'))],groups=[dict(key='g',function='reading')],
            relations=[dict(source='picture',target='left_anchor',kind='near'),dict(source='chair',target='right_anchor',kind='near')])
        observed=dict(objects=[dict(track_id='left_anchor',center_local_m=dict(value=[-4,0,4])),dict(track_id='right_anchor',center_local_m=dict(value=[4,0,4]))])
        layout={'g':dict(id='g',region_domain=['rear'],objects=[dict(id=k) for k in ('chair','picture','book')])};before=deepcopy(world)
        numeric,revised,mapping=geometry.support_tree_groups(dict(hfov=90),observed,world,layout)
        self.assertEqual(world,before);self.assertEqual(len(numeric['groups']),2)
        self.assertEqual(mapping['chair']['numeric_group'],mapping['book']['numeric_group'])
        self.assertNotEqual(mapping['chair']['numeric_group'],mapping['picture']['numeric_group'])
        self.assertEqual(revised[mapping['chair']['numeric_group']]['region_domain'][0],'right')
        self.assertEqual(revised[mapping['picture']['numeric_group']]['region_domain'][0],'left')

    def test_distant_collision_free_initialization_does_not_block_near_then_repair_search(self):
        from .assembly_spatial import articulated_repair
        observed=dict(objects=[dict(track_id='anchor',center_local_m=dict(value=[0,0,-3]),size_aabb_local_m=dict(value=[1,1,1]))])
        boxes={oid:dict(id=oid,center=[x,0,-3],size=[1,1,1],group='g') for oid,x in [('left',-4),('right',4)]}
        contracts={oid:dict(attachment=dict(kind='floor',host=None)) for oid in boxes}
        relations=[dict(source=oid,target='anchor',kind='near') for oid in boxes]
        solved,review=articulated_repair(dict(hfov=90,image_size=[1,1]),observed,boxes,contracts,
            {'g':dict(kind='behind_camera')},[],relations,hard_ordering=True,surface_proximity=True)
        self.assertFalse(review['final_overlaps'])
        self.assertLess(max(np.linalg.norm(np.maximum(0,np.abs(np.array(v['center'])-[0,0,-3])-1)) for v in solved.values()),.05)


if __name__=='__main__':unittest.main()
