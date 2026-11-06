"""Task-level contracts: fixed observed anchors and genuine hidden deltas."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from . import assembly_delta as delta,assembly_parametric as geometry
from . import test_assembly as fixture
from .world_scm import forward
from .conditional_world_geometry import support_plane
from ..common import read
from .assembly_world import run
from .assembly_review import review
from .assembly_observation import host_domains
from . import assembly_pose as pose
import numpy as np


def proposal():
    old=fixture.furniture();z=fixture.scm.hypotheses_answer(fixture.hypotheses())[0]
    normalized=fixture.scm.furniture_answer(old,dict(context=[]),z)
    for raw,f in zip(old['alternatives'],normalized):
        contents=fixture.contents(f)['bundles'][0]['alternatives'][0]
        raw['relations']+=contents['relations'];raw['groups'][0]['objects']+=contents['objects']
        for g in raw['groups']:
            g['observed_refs']=['obs_000'];g['new_objects']=g.pop('objects')
            for o in g['new_objects']:
                o['reason']=o.pop('distinct_instance_reason');o['placement']=o.pop('placement_hint')
    return old


class DeltaTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize
    def model(self):
        zs=delta.hypotheses_answer(fixture.hypotheses());fs=delta.furniture_answer(proposal(),self.case,zs[0])
        return delta.build_model(zs,{f'f{i}':f for i,f in enumerate(fs)})
    def world(self):
        m=self.model();noise={n['id']:0. for n in m['nodes']};w=delta.active_world(m,forward(m,noise));w['noise']=noise;return m,w
    def metric(self,w):
        raw=fixture.metric(w)
        for g in raw['groups']:
            g.pop('azimuth_deg');g['region']='rear'
            for r in g['objects']:r['full_size_m']=[r.pop(k) for k in ('width_m','height_m','depth_m')]
        return raw
    def test_observed_anchors_never_become_new_inventory(self):
        m,w=self.world()
        self.assertTrue(all(o['key']!='obs_000' for o in w['objects']))
        self.assertEqual(w['groups'][0]['observed_refs'],['obs_000'])
        self.assertEqual(len(w['objects']),9)
    def test_joint_mechanism_has_no_furniture_contents_partition(self):
        m,w=self.world();nodes={n['id']:n for n in m['nodes']}
        self.assertFalse(any(n.startswith('T_') for n in nodes))
        self.assertTrue({'table','lamp','book','plant'}<={o['key'] for o in w['objects']})
        self.assertTrue(all(nodes[o['presence_node']]['parents']==['F'] for o in w['objects']))
    def test_host_intervention_preserves_owned_belongings(self):
        m,w=self.world();table=next(o for o in w['objects'] if o['key']=='table')
        after=delta.active_world(m,forward(m,w['noise'],{table['presence_node']:'0'}));objects={o['key']:o for o in after['objects']}
        self.assertIn('lamp',objects);self.assertFalse(objects['lamp']['installation_ready']);self.assertTrue(objects['plant']['installation_ready'])
    def test_activity_intervention_reexecutes_the_joint_inventory_mechanism(self):
        z0=delta.hypotheses_answer(fixture.hypotheses())[0];z1={**deepcopy(z0),'id':'z1','description':'A different assumed activity organization.'}
        f=delta.furniture_answer(proposal(),self.case,z0);f[1]['hypothesis']=z1
        m=delta.build_model([z0,z1],{'f0':f[0],'f1':f[1]});noise={n['id']:0. for n in m['nodes']}
        w=delta.active_world(m,forward(m,noise));w['noise']=noise
        query=delta.intervention_diagnostics(m,w)['configuration_queries'][0]
        self.assertEqual(query['before_program'],'f0');self.assertEqual(query['after_program'],'f1')
        self.assertEqual([o['name'] for o in query['after_inventory']],['wardrobe'])
        self.assertFalse(query['model_called']);self.assertFalse(query['coordinate_counterfactual_generated'])
    def test_same_category_hidden_instance_is_allowed_without_blacklist(self):
        value=proposal();value['alternatives'][0]['groups'][0]['new_objects'][0]['category']=self.case['context'][0]['category']
        rows=delta.furniture_answer(value,self.case,delta.hypotheses_answer(fixture.hypotheses())[0])
        self.assertEqual(rows[0]['groups'][0]['objects'][0]['category'],self.case['context'][0]['category'])
    def test_observed_identity_cannot_be_redefined(self):
        value=proposal();value['alternatives'][0]['groups'][0]['new_objects'][0]['key']='obs_000'
        with self.assertRaisesRegex(ValueError,'existing IDs'):
            delta.furniture_answer(value,self.case,delta.hypotheses_answer(fixture.hypotheses())[0])
    def test_unknown_observed_reference_reports_real_ids(self):
        value=proposal();value['alternatives'][0]['groups'][0]['observed_refs']=['imaginary_obs']
        with self.assertRaisesRegex(ValueError,'obs_000'):
            delta.furniture_answer(value,self.case,delta.hypotheses_answer(fixture.hypotheses())[0])
    def test_finite_regions_remove_angular_boundary_decision(self):
        _,w=self.world();raw=self.metric(w)
        for hfov in (60,90,120):
            for region in ('left','right','rear'):
                for g in raw['groups']:g['region']=region
                clean=geometry.layout_answer(raw,w,hfov)
                self.assertTrue(all(abs(g['azimuth_deg'])>hfov/2 for g in clean.values()))
        prompt=geometry.layout_prompt(self.case,self.observed,w,support_plane(self.observed,['obs_000']))
        self.assertNotIn('"azimuth_deg":',prompt);self.assertNotIn('"center_m":',prompt);self.assertNotIn('"offset_frame":',prompt)
    def test_parametric_pipeline_retains_every_selected_instance(self):
        _,w=self.world();raw=self.metric(w);original=deepcopy(raw)
        clean=geometry.layout_answer(raw,w,90);solved=geometry.compile_layout(self.case,self.observed,w,clean,support_plane(self.observed,['obs_000']))
        self.assertTrue(solved['review']['geometry_contract_passed']);self.assertEqual(len(solved['items']),len(w['objects']))
        self.assertEqual(raw,original)
    def test_real_qwen_center_response_is_an_unknown_domain_not_a_rear_alias(self):
        _,w=self.world();raw=self.metric(w)
        for g in raw['groups']:g['region']='center'
        clean=geometry.layout_answer(raw,w,90)
        self.assertTrue(all(g['region_domain']==['left','right','rear'] for g in clean.values()))
        solved=geometry.compile_layout(self.case,self.observed,w,clean,support_plane(self.observed,['obs_000']))
        search=solved['review']['direction_search']
        self.assertEqual(search['combinations_previewed'],3)
        self.assertEqual(len(search['refined']),3)
        self.assertFalse(search['inventory_changed'])
        self.assertTrue(solved['review']['geometry_contract_passed'])
        self.assertEqual(len(solved['items']),len(w['objects']))
    def test_fully_in_view_observed_host_cannot_support_a_wholly_off_view_addition(self):
        obs=deepcopy(self.observed);obs['objects'][0]['center_local_m']['value']=[0,0,4]
        obs['objects'][0]['size_aabb_local_m']['value']=[1,1,1]
        case={**self.case,'observed_support_domains':host_domains(self.case,obs)}
        self.assertEqual(case['observed_support_domains']['obs_000']['contact_region_candidates']['top'],[])
        value=proposal();book=value['alternatives'][0]['groups'][0]['new_objects'][0]
        book['attachment']=dict(kind='top',host='obs_000')
        with self.assertRaisesRegex(ValueError,'entire possible contact envelope'):
            delta.furniture_answer(value,case,delta.hypotheses_answer(fixture.hypotheses())[0])
    def test_cropped_host_is_still_an_allowed_uncertain_support_hypothesis(self):
        obs=deepcopy(self.observed);obs['objects'][0]['center_local_m']['value']=[-4,0,4]
        obs['objects'][0]['size_aabb_local_m']['value']=[2,1,1]
        case={**self.case,'observed_support_domains':host_domains(self.case,obs)}
        self.assertIn('left',case['observed_support_domains']['obs_000']['contact_region_candidates']['top'])
        value=proposal();value['alternatives'][0]['groups'][0]['new_objects'][-1]['attachment']=dict(kind='top',host='obs_000')
        self.assertTrue(delta.furniture_answer(value,case,delta.hypotheses_answer(fixture.hypotheses())[0]))
    def test_observed_host_vertical_escape_is_considered_without_moving_the_host(self):
        _,w=self.world();w=deepcopy(w);item=w['objects'][0];item['attachment']=dict(kind='top',host='obs_000');w['objects']=[item];w['relations']=[]
        obs=deepcopy(self.observed);obs['objects'][0]['center_local_m']['value']=[0,-4,3]
        obs['objects'][0]['size_aabb_local_m']['value']=[1,.2,1]
        raw=self.metric(w);r=raw['groups'][0]['objects'][0];r['full_size_m']=[.1,.1,.1];r['offset_m']=[0,0,0]
        clean=geometry.layout_answer(raw,w,90);before=deepcopy(obs)
        solved=geometry.compile_layout(self.case,obs,w,clean,support_plane(obs,[]))
        self.assertEqual(obs,before);self.assertTrue(solved['review']['geometry_contract_passed'])
        regions=solved['review']['program_solver']['observed_host_region_search']
        self.assertIn('below',regions[item['id']]['feasible_faces'])
    def test_pose_equations_replay_and_propagate_host_motion_only_to_descendants(self):
        _,w=self.world();clean=geometry.layout_answer(self.metric(w),w,90)
        solved=geometry.compile_layout(self.case,self.observed,w,clean,support_plane(self.observed,['obs_000']))
        model=pose.fit(w,solved['items'],self.observed);before=pose.forward(model)
        report=pose.diagnostics(model,solved['items']);self.assertLess(report['exact_layout_replay_error_m'],1e-8)
        table=next(o for o in w['objects'] if o['key']=='table');lamp=next(o for o in w['objects'] if o['key']=='lamp')
        after=pose.forward(model,{table['id']:(np.array(before[table['id']]['center'])+[.5,.2,0]).tolist()})
        np.testing.assert_allclose(np.array(after[lamp['id']]['center'])-before[lamp['id']]['center'],[.5,.2,0],atol=1e-8)
        for o in w['objects']:
            if o['key'] not in ('table','lamp'):self.assertEqual(after[o['id']],before[o['id']])
        with self.assertRaisesRegex(ValueError,'observations stay fixed'):pose.forward(model,{'obs_000':[0,0,0]})
    def test_repetition_diagnostics_do_not_delete_or_cap_multiple_instances(self):
        _,w=self.world();copy=deepcopy(w['objects'][0]);copy['id']='another_table';w['objects'].append(copy)
        before=deepcopy(w);report=delta.semantic_diagnostics(w)
        self.assertEqual(w,before);self.assertFalse(report['objects_edited'])
        self.assertEqual(report['repeated_variant_on_same_host'][0]['count'],2)
    def test_left_right_regions_keep_objects_on_requested_side(self):
        _,w=self.world();raw=self.metric(w)
        for region,sign in [('left',-1),('right',1)]:
            for g in raw['groups']:g['region']=region
            clean=geometry.layout_answer(raw,w,90);solved=geometry.compile_layout(self.case,self.observed,w,clean,support_plane(self.observed,['obs_000']))
            self.assertTrue(solved['review']['geometry_contract_passed'])
            self.assertTrue(all(sign*o['center'][0]>0 for o in solved['items']))
    def test_empty_hidden_delta_remains_honest(self):
        z=delta.hypotheses_answer(fixture.hypotheses());f=delta.furniture_answer(dict(alternatives=[dict(description='No additional hidden objects proposed.',
            groups=[dict(function='Known observed area.',reason='Fixed evidence.',observed_refs=['obs_000'],new_objects=[])],relations=[])]),self.case,z[0])
        samples=delta.sample_worlds(delta.build_model(z,{'f0':f[0]}))
        self.assertEqual(samples[0]['objects'],[])
    def test_blocked_numeric_candidate_is_visualized_without_completed_state(self):
        original=geometry.compile_layout
        def blocked(*args,**kwargs):
            result=original(*args,**kwargs);result['review']['geometry_contract_passed']=False
            result['review']['hard_violations']=1
            return result
        self.base.observe(self.root,'api',True)
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_world.geometry.compile_layout',side_effect=blocked):
            summary=run(fixture.GrammarFixture(),self.root,'qwen')
        self.assertEqual(summary['worlds'],[]);self.assertEqual(len(summary['blocked_worlds']),2)
        for row in summary['blocked_worlds']:
            folder=self.root/'qwen/completion'/row['folder']
            self.assertFalse((folder/'G_full.json').exists())
            candidate=read(folder/'G_full_candidates.json');self.assertFalse(candidate['world_state_ready'])
            self.assertEqual(len(candidate['objects']),len(self.observed['objects'])+row['new_instances'])
            self.assertTrue((folder/'candidate_world/scene_3d.html').is_file())
        html=review(self.root).read_text();self.assertIn('诊断候选',html);self.assertEqual(html.count('<iframe class="scene"'),2)
    def test_cli_distinguishes_partial_candidates_from_no_export(self):
        from .assembly_experiment import main
        args=['assembly_experiment','world','--root',str(self.root),'--backend','qwen']
        partial=dict(status='partial_layouts_needs_review',worlds=[dict(folder='world_01')],
            failures=[dict(world=2,error='Synthetic blocked candidate')])
        with patch('sys.argv',args),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.provider'),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_world.run',return_value=partial),patch('builtins.print'):
            main()
        with patch('sys.argv',args),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.provider'),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_world.run',return_value=dict(status='no_world_exported',worlds=[],failures=[{}])),patch('builtins.print'):
            with self.assertRaises(SystemExit) as error:main()
        self.assertEqual(error.exception.code,1)
    def test_stateless_joint_and_metric_requests_keep_the_visual_evidence(self):
        self.base.observe(self.root,'api',True)
        summary=run(fixture.GrammarFixture(),self.root,'qwen')
        self.assertEqual(summary['status'],'worlds_exported_needs_review')
        calls=self.root/'qwen/completion/calls'
        evidence=read(calls/'hypotheses.request.json')['canonical_image_sha256']
        self.assertTrue(evidence)
        for stem in ('furniture_z0','layout_01','layout_02'):
            self.assertEqual(read(calls/(stem+'.request.json'))['canonical_image_sha256'],evidence)


if __name__=='__main__':unittest.main()
