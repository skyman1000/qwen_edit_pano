"""v11 probability, stable identities, spatial transport and execution contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import read,write
from . import assembly_holistic as model,assembly_holistic_experiment as experiment
from .test_panorama import PanoramaFixture,proposals
from . import test_assembly as fixture
from .assembly_world import run
from .world_scm import forward
from .world_generation import image_bytes


class HolisticFixture(PanoramaFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        path=Path(path);name=path.stem
        if name.startswith('furniture_'):
            value=proposals()
            for alt in value['alternatives']:
                for g in alt['groups']:g['layout_region']='unspecified'
        elif name.startswith(('reference_','joint_','rgb_joint_','direct_')):
            menu=json.loads(prompt.split('\nCANDIDATE_ASSEMBLIES: ')[1].split('\n')[0]);h=json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\n')[0]) if '\nCURRENT_PARENT_H: ' in prompt else None
            selected=menu[:2] if h is None or 'reading' in h['functions'] else []
            value=dict(assemblies=[dict(id=row['id'],region=('rear' if image is None and 'rear' in row['allowed_camera_regions'] else row['allowed_camera_regions'][0])) for row in selected])
        else:return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kw)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class HolisticTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def prepare(self):
        root=self.temp/'holistic'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=2,draws=2)
        return root

    def bundle(self):
        root=self.prepare()
        for b in ('qwen','api'):
            result=experiment.propose(root,b,HolisticFixture(b));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for b in ('qwen','api'):experiment.mechanisms(root,b,HolisticFixture(b))
        experiment.pool_mechanisms(root);return root

    def fitted(self,root,branch='end_to_end'):
        d=read(root/'shared/domain.json');config=read(root/branch/'experiment.json');config['execution_backend']='qwen'
        return model.prepare_model(d['base_model'],read(root/'shared/case.json'),config,None)

    def test_regions_are_probability_choices_and_preserve_canonical_identity(self):
        root=self.bundle();m=self.fitted(root);c=next(c for c,r in m['region_domains'].items() if set(r)>={'left','rear'})
        u={n['id']:.3 for n in m['nodes']};base={'M_'+k:'none' for k in m['slots']}
        left=model.active_world(m,forward(m,u,base|{'M_'+c:'left'}));rear=model.active_world(m,forward(m,u,base|{'M_'+c:'rear'}))
        self.assertEqual([o['id'] for o in left['objects']],[o['id'] for o in rear['objects']])
        self.assertTrue(all(g['layout_region']=='rear' for g in rear['groups']))
        self.assertEqual(rear['spatial_program'][c]['region'],'rear')
        self.assertEqual(len(rear['objects']),sum(len(g['objects']) for g in m['furniture_states'][m['slots'][c][1]]['groups']))

    def test_same_function_units_can_coexist_and_empty_is_retained(self):
        root=self.bundle();m=self.fitted(root);u={n['id']:.2 for n in m['nodes']}
        states={'M_'+c:regions[0] for c,regions in m['region_domains'].items()}
        world=model.active_world(m,forward(m,u,states));self.assertEqual(len(world['objects']),len(m['component_registry']))
        empty=model.active_world(m,forward(m,u,{k:'none' for k in states}));self.assertEqual(empty['objects'],[])

    def test_function_and_rgb_change_actual_inventory_or_region_probabilities(self):
        root=self.bundle();m=self.fitted(root);rgb=self.fitted(root,'rgb_inventory')
        h_on=next(h['id'] for h in m['profiles'] if 'reading' in h['functions']);h_off=next(h['id'] for h in m['profiles'] if not h['functions'])
        self.assertNotEqual(model.marginals(m,h_on)['object_presence'],model.marginals(m,h_off)['object_presence'])
        self.assertNotEqual(model.marginals(m)['channel_region_joint_presence'],model.marginals(rgb)['channel_region_joint_presence'])

    def test_symmetric_base_does_not_increase_inventory_mass_with_more_regions(self):
        root=self.bundle();m=self.fitted(root)
        for c,rows in m['component_probabilities'].items():self.assertEqual(rows['base'][0],.5)
        law=model.marginals(m);self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1)
        self.assertAlmostEqual(sum(law['expected_category_counts'].values()),law['expected_hidden_instances'])
        samples=model.sample_worlds(m,count=250)
        self.assertAlmostEqual(sum(len(w['objects']) for w in samples)/len(samples),law['expected_hidden_instances'],delta=1.)
        empirical_rear=sum(any(v['region']=='rear' for v in w['spatial_program'].values()) for w in samples)/len(samples)
        self.assertAlmostEqual(empirical_rear,law['any_declared_region_presence']['rear'],delta=.1)
        domain=read(root/'shared/domain.json');table=read(root/'shared/mechanisms/qwen.json')
        vectors=[json.loads(v) for r in [table['reference'],*table['joint'].values(),table['direct']] for v in r['counts']]
        for i,c in enumerate(domain['slots']):domain['region_domains'][c]=sorted({v[i] for v in vectors if v[i]!='none'}) or domain['region_domains'][c][:1]
        narrow=model.marginals(model.compile_model(domain['base_model'],domain,table))
        self.assertAlmostEqual(narrow['expected_hidden_instances'],law['expected_hidden_instances'])
        self.assertAlmostEqual(narrow['empty_probability'],law['empty_probability'])

    def test_host_removal_keeps_owned_children_and_invalidates_installation(self):
        root=self.bundle();m=self.fitted(root);u={n['id']:.3 for n in m['nodes']}
        before=model.active_world(m,forward(m,u,{'M_'+c:r[0] for c,r in m['region_domains'].items()}))
        child=next(o for o in before['objects'] if o['attachment']['host'] in m['component_registry'])
        host=m['component_registry'][child['attachment']['host']]
        do={'M_'+c:r[0] for c,r in m['region_domains'].items()};do[host['presence_node']]='0'
        after=model.active_world(m,forward(m,u,do))
        self.assertIn(child['id'],{o['id'] for o in after['objects']});self.assertIn(child['id'],after['unresolved_installations'])

    def test_H_cut_preserves_full_joint_component_distribution(self):
        root=self.bundle();a=self.fitted(root);b=self.fitted(root,'function_free')
        self.assertEqual(a['joint_vectors'],b['joint_vectors'])
        for j,p in model.marginals(a)['joint_component_probabilities'].items():self.assertAlmostEqual(p,model.marginals(b)['joint_component_probabilities'][j])
        for h in b['profiles']:self.assertEqual(b['joint_rows'][h['id']],b['joint_rows'][b['profiles'][0]['id']])

    def test_paired_backend_prompts_RGB_seeds_and_selection_budgets(self):
        root=self.bundle();a=read(root/'shared/mechanisms/qwen.json');b=read(root/'shared/mechanisms/api.json')
        self.assertEqual(a['original_parent_prompt_plan'],b['original_parent_prompt_plan'])
        plans={v['stage']:v for v in a['original_parent_prompt_plan'] if 'canonical_draw' in v}
        first=[v for k,v in plans.items() if k.startswith('joint_')];direct=[v for k,v in plans.items() if k.startswith('direct_')]
        self.assertEqual(len(first),len(direct))
        for v in first:
            other=plans['rgb_'+v['stage']]
            self.assertEqual(v['local_seed'],other['local_seed']);self.assertEqual(v['prompt_sha256'],other['prompt_sha256'])
            self.assertFalse(v['image_delivered']);self.assertTrue(other['image_delivered'])
            self.assertEqual(v['max_tokens'],other['max_tokens'])

    def test_host_identity_mismatch_rejects_only_the_dependent_unit(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0]
        bad=dict(function='reading',new_objects=[dict(key='bad_remote',category='remote',variant='remote',reason='wrong identity',placement='on table',
            attachment=dict(kind='top',host='obs_000'),observed_host_name='side table',affordances=[])])
        alt['groups'].append(bad)
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertTrue(any(p['groups'] for p in accepted))
        self.assertFalse(any(o['key']=='bad_remote' for p in accepted for g in p['groups'] for o in g['objects']))
        self.assertTrue(any('observed_host_name' in r['error'] for r in accepted[0]['proposal_validation']['rejected']))

    def test_positive_Z_order_excludes_rear_without_overriding_the_relation(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0]
        g=alt['groups'][0];g['layout_region']='unspecified';key=g['new_objects'][0]['key']
        alt['relations']=[dict(source=key,target='obs_000',kind='farther_than',reason='Explicit depth hypothesis')]
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        chosen=next(p for p in accepted if any(o['key']==key for g in p['groups'] for o in g['objects']))
        self.assertNotIn('rear',chosen['groups'][0]['region_options']);self.assertTrue(chosen['relations'])

    def test_all_branches_export_or_retain_complete_failed_candidates(self):
        root=self.bundle();observed=read(self.root/'qwen/observer3d/G_obs.json')
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                self.assertTrue(experiment.ready(root,exp,b)['ready'])
                result=run(HolisticFixture(b),root/exp,b)
                self.assertEqual(len(result['worlds'])+len(result['blocked_worlds']),2,result)
                for record in result['worlds']+result['blocked_worlds']:
                    w=root/exp/b/'completion'/record['folder'];state=read(w/record.get('state_file','G_full.json'));sem=read(w/'semantic_world.json')
                    self.assertEqual(state['objects'][:len(observed['objects'])],observed['objects'])
                    self.assertEqual(len(state['objects'])-len(observed['objects']),len(sem['objects']))
                    for obj in state['objects'][len(observed['objects']):]:self.assertIn('orientation_yaw_camera_deg',obj)
                    if record in result['worlds']:
                        numeric=read(w/'numeric_review.json')
                        self.assertFalse(any(v['required_proxy_failed'] for v in numeric['support_projection_review']))


if __name__=='__main__':unittest.main()
