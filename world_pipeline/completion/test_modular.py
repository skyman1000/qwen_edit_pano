"""Synthetic CPU tests of functional assembly probability and geometry semantics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ..common import read,write,sha
from . import assembly_modular as model,assembly_modular_experiment as experiment
from . import assembly_modular_geometry as geometry,assembly_parametric as parametric
from . import test_assembly as fixture
from .test_sdscene import SceneFixture
from .assembly_world import run
from .world_generation import image_bytes,MATCHED_OPTIONS,SYSTEM_PROMPT
from .world_scm import validate,forward
from .conditional_world_geometry import support_plane


def proposals():
    def item(key,category,kind='floor',host=None,aff=()):
        return dict(key=key,category=category,variant=category,reason='One possible unseen instance.',
            placement='An unphotographed surrounding area.',attachment=dict(kind=kind,host=host),affordances=list(aff))
    def group(role,objects):return dict(function=role,reason='Uncertain functional assembly.',observed_refs=[],new_objects=objects)
    return dict(alternatives=[dict(description='Optional reading, storage and incidental assemblies.',groups=[
        group('reading',[item('table','side table',aff=['top']),item('lamp','table lamp','top','table'),
                         item('book','book','top','table'),item('chair','chair',aff=['seat'])]),
        group('storage',[item('shelf','bookshelf',aff=['shelf']),item('archive_book','book','shelf','shelf')]),
        group('background',[item('plant','plant')])],relations=[])])


class ModularFixture(SceneFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);name=path.stem
        if name.startswith('layout_') or name=='hypotheses' or name.startswith('recognition_'):
            return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)
        if name.startswith('furniture_'):value=proposals()
        elif name.startswith('assembly_'):
            if image is not None:raise AssertionError('Nonparent image in assembly mechanism')
            h=json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\nFIXED_ROLE: ')[0]);role=prompt.split('\nFIXED_ROLE: ')[1].split('\n')[0]
            menu=json.loads(prompt.split('\nALLOWED_VALUE_DOMAIN: ')[1]);present=next(m['id'] for m in menu if m['id']!='none')
            j=int(name.split('_')[-1]);active=('reading' in h['functions'])
            choose=active if role=='reading' else not active if role=='storage' else j%2==0
            if self.backend=='api' and role=='storage':choose=not choose
            value=dict(choice=present if choose else 'none')
        else:raise AssertionError('Unexpected query '+name)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,system_prompt=SYSTEM_PROMPT,
            generation=MATCHED_OPTIONS,canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class ModularTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def bundle(self):
        root=self.temp/'modular'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=3,draws=4)
        for backend in ('qwen','api'):
            result=experiment.propose(root,backend,ModularFixture(backend));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for backend in ('qwen','api'):experiment.mechanisms(root,backend,ModularFixture(backend))
        experiment.previous.pool_mechanisms(root);return root

    def fitted(self,policy='functional'):
        root=self.bundle();d=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        m=model.prepare_model(d['base_model'],case,dict(sdscene_bundle=str(root/'shared'),
            mechanism_source='per_backend',execution_backend='qwen',modular_policy=policy),None)
        return root,m

    def test_profiles_are_pure_intervention_values_and_do_not_retain_inventory_captions(self):
        root=self.bundle();domain=read(root/'shared/domain.json')
        self.assertEqual(domain['profiles'][1],model.profile('h1',['reading','social']))
        for path in (root/'mechanism_calls/qwen').glob('assembly_*.request.json'):
            req=read(path);self.assertIsNone(req['canonical_image_sha256'])
            h=json.loads(req['prompt'].split('\nCURRENT_PARENT_H: ')[1].split('\nFIXED_ROLE: ')[0])
            self.assertEqual(h,model.profile(h['id'],h['functions']))
        self.assertEqual(read(root/'run.json')['api_model'],'deepseek-ai/DeepSeek-V4.1-Flash')

    def test_role_composition_is_proper_and_category_presence_is_not_expected_count(self):
        root,m=self.fitted();validate(m);law=model.marginals(m)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1.)
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        self.assertLess(law['category_presence']['book'],law['expected_category_counts']['book'])
        self.assertTrue(0<law['empty_probability']<1)
        worlds=model.sample_worlds(m,count=4000);hist={};mixed=False
        for world in worlds:
            self.assertEqual(forward(m,world['noise']),world['structural_state'])
            self.assertEqual(len(world['objects']),len({o['id'] for o in world['objects']}))
            active=sum(world['structural_state']['M_'+r]!='none' for r in m['slots']);mixed|=1<active<len(m['slots'])
            n=str(len(world['objects']));hist[n]=hist.get(n,0)+1
        self.assertTrue(mixed)
        for n,p in law['hidden_count_probabilities'].items():self.assertAlmostEqual(hist.get(n,0)/len(worlds),p,delta=.025)

    def test_function_edge_ablation_preserves_domain_and_removes_inventory_dependence(self):
        root,m=self.fitted();d=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        free=model.prepare_model(d['base_model'],case,dict(sdscene_bundle=str(root/'shared'),mechanism_source='per_backend',
            execution_backend='qwen',modular_policy='function_free'),None)
        self.assertEqual(m['component_registry'],free['component_registry']);self.assertEqual(m['slots'],free['slots'])
        a=model.marginals(m,'h0');b=model.marginals(m,'h1')
        self.assertGreater(abs(a['category_presence']['chair']-b['category_presence']['chair']),.1)
        c=model.marginals(free,'h0');d=model.marginals(free,'h1')
        self.assertEqual(c['category_presence'],d['category_presence'])
        for role,rows in free['assembly_rows'].items():self.assertEqual(rows['h0'],rows['h1'])
        self.assertTrue(all(not n['parents'] for n in free['nodes'] if n['id'].startswith('M_')))

    def test_host_removal_changes_installation_but_keeps_owned_child(self):
        root,m=self.fitted();noise={n['id']:.75 for n in m['nodes']};role='reading';state=next(s for s in m['slots'][role] if s!='none')
        do={'M_'+role:state};world=model.active_world(m,forward(m,noise,do));table=next(o for o in world['objects'] if o['key']=='table')
        lamp=next(o for o in world['objects'] if o['key']=='lamp');after=model.active_world(m,forward(m,noise,dict(do,**{table['presence_node']:'0'})))
        self.assertIn(lamp['id'],[o['id'] for o in after['objects']]);self.assertIn(lamp['id'],after['unresolved_installations'])

    def test_parser_rejects_real_side_table_on_sofa_and_cross_module_dependencies(self):
        h=model.profile('z0',['reading']);raw=proposals()
        raw['alternatives'][0]['groups'][0]['new_objects'][0]['attachment']=dict(kind='seat',host='obs_000')
        with self.assertRaisesRegex(ValueError,'cannot be installed'):model.furniture_answer(raw,self.case,h)
        raw=proposals();raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='shelf'
        raw['alternatives'][0]['groups'][1]['new_objects'][0]['affordances'].append('top')
        with self.assertRaisesRegex(ValueError,'SAME|same|one'):model.furniture_answer(raw,self.case,h)

    def test_parser_preserves_declared_role_for_observed_host_extension(self):
        raw=proposals();raw['alternatives'][0]['groups'][0]['new_objects']=[raw['alternatives'][0]['groups'][0]['new_objects'][2]]
        raw['alternatives'][0]['groups'][0]['new_objects'][0]['attachment']=dict(kind='seat',host='obs_000')
        case=deepcopy(self.case);case.pop('observed_support_domains',None)
        rows=model.furniture_answer(raw,case,model.profile('z0',['reading']))
        group=next(g for g in rows[0]['groups'] if any(o['key']=='book' for o in g['objects']))
        self.assertEqual(group['function'],'reading');self.assertEqual(group['assembly_key'],'g0')
        self.assertEqual(group['objects'][0]['attachment']['host'],'obs_000')

    def test_six_branches_export_keep_same_observations_and_control_samples_match(self):
        root=self.bundle();original=read(self.root/'qwen/observer3d/G_obs.json')
        for label in ('end_to_end','layout_control','function_free'):
            for backend in ('qwen','api'):
                s=run(ModularFixture(backend),root/label,backend)
                self.assertEqual(s['status'],'worlds_exported_needs_review',s)
                for record in s['worlds']:
                    full=read(root/label/backend/'completion'/record['folder']/'G_full.json')
                    self.assertEqual(full['objects'][:len(original['objects'])],original['objects'])
                    self.assertTrue(all('generative_presence_probability' in o for o in full['objects'][len(original['objects']):]))
        self.assertEqual(sha(root/'layout_control/qwen/completion/structural_samples.json'),sha(root/'layout_control/api/completion/structural_samples.json'))
        self.assertTrue(experiment.report(root).is_file());self.assertTrue((root/'functional_edge_ablation.json').is_file())

    def vertical_case(self):
        m=fixture.AssemblyTests.model(self);world=fixture.AssemblyTests.rich_world(self,m)
        bykey={o['key']:o for o in world['objects']};raw=fixture.metric(world)
        for group in raw['groups']:
            group.pop('azimuth_deg');group['region']='rear'
            for o in group['objects']:
                o['full_size_m']=[o.pop(k) for k in ('width_m','height_m','depth_m')]
                if o['id']==bykey['hooks']['id']:o['offset_m'][1]=-5.
        plane=support_plane(self.observed,['obs_000']);return world,bykey,raw,plane

    def test_vertical_projection_fixes_low_wall_root_and_transports_hanging_child(self):
        world,items,raw,plane=self.vertical_case();world['relations'].append(dict(source=items['table']['id'],target=items['hooks']['id'],kind='below',reason='Hooks above table.'))
        layout=geometry.layout_answer(raw,world,self.case['hfov']);before=parametric.instantiate(world,layout,self.observed,plane,90)[0]
        solved=geometry.compile_layout(self.case,self.observed,world,layout,plane);after={o['id']:o for o in solved['items']}
        self.assertTrue(solved['review']['vertical_order_projection']['feasible'])
        dy=after[items['hooks']['id']]['center'][1]-before[items['hooks']['id']]['center'][1]
        self.assertGreater(dy,0);self.assertAlmostEqual(after[items['bag']['id']]['center'][1]-before[items['bag']['id']]['center'][1],dy)
        r=next(r for r in solved['review']['declared_relation_review'] if r['kind']=='below')
        self.assertTrue(r['ordering_or_presence_satisfied']);self.assertEqual(len(after),len(world['objects']))
        prompt=geometry.layout_prompt(self.case,self.observed,world,plane)
        anchors=json.loads(prompt.split('\nREAD_ONLY_METRIC_ANCHORS: ')[1])
        self.assertEqual(anchors[0]['center_camera_m'],self.observed['objects'][0]['center_local_m']['value'])
        self.assertTrue(solved['review']['observed_absolute_positions_sent_to_metric_LM'])

    def test_infeasible_floor_order_is_blocked_without_moving_observations_or_dropping_inventory(self):
        world,items,raw,plane=self.vertical_case();world['relations']=[dict(source=items['table']['id'],target=items['chair']['id'],kind='above',reason='Deliberate contradiction.')]
        layout=geometry.layout_answer(raw,world,90);before=deepcopy(self.observed)
        solved=geometry.compile_layout(self.case,self.observed,world,layout,plane)
        self.assertFalse(solved['review']['geometry_contract_passed']);self.assertEqual(self.observed,before)
        self.assertEqual({o['id'] for o in solved['items']},{o['id'] for o in world['objects']})

    def test_all_absent_mechanisms_really_allow_all_empty_worlds(self):
        root,m=self.fitted()
        for role,states in m['slots'].items():
            for h in m['assembly_rows'][role]:m['assembly_rows'][role][h]=[float(s=='none') for s in states]
        for node in m['nodes']:
            if node['id'].startswith('M_'):
                node['cpt']={h:[float(s=='none') for s in node['states']] for h in node['cpt']}
        self.assertAlmostEqual(model.marginals(m)['empty_probability'],1.)
        self.assertTrue(all(not w['objects'] for w in model.sample_worlds(m,count=40)))

    def test_direction_search_is_bounded_with_more_than_four_modules(self):
        objects=[];groups=[];layout={}
        for i in range(5):
            gid='g'+str(i);oid='plant'+str(i)
            objects.append(dict(id=oid,group=gid,key=oid,name='plant',category='plant',variant='potted plant',
                distinct_instance_reason='Synthetic geometry test.',placement_hint='Surrounding area.',attachment=dict(kind='floor',host=None)))
            groups.append(dict(key=gid,function='background',objects=[]))
            layout[gid]=dict(id=gid,azimuth_deg=180,range_m=3,region_domain=['left','right','rear'],region_response='unspecified',
                objects=[dict(id=oid,width_m=.2,height_m=.4,depth_m=.2,size=[.2,.4,.2],yaw_deg=0,offset_m=[i*.5,0,0],
                    offset_frame='camera',surface_fraction=None,hanging_drop_m=None)])
        world=dict(objects=objects,groups=groups,relations=[],configuration='Synthetic five-group geometry test.',hypothesis={})
        solved=parametric.compile_layout(self.case,self.observed,world,layout,support_plane(self.observed,['obs_000']),max_direction_previews=20)
        search=solved['review']['direction_search'];self.assertEqual(search['total_combinations_in_domain'],243)
        self.assertEqual(search['combinations_previewed'],20)


if __name__=='__main__':unittest.main()
