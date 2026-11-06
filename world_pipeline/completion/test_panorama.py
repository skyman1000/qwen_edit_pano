"""v9 protected spatial-state contracts; no real model, training, or rear quota."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import read,write,sha
from . import assembly_panorama as model,assembly_panorama_experiment as experiment
from . import assembly_panorama_geometry as geometry
from .test_mechanism import MechanismFixture,proposals as old_proposals
from . import test_assembly as fixture
from .assembly_world import run
from .assembly_panorama_visual import report
from .conditional_world_geometry import support_plane
from .world_scm import forward,validate
from .world_generation import image_bytes


def proposals():
    raw=old_proposals()
    for i,alt in enumerate(raw['alternatives']):
        alt['relations']=[]
        for g in alt['groups']:
            g.update(layout_region='rear' if g['realization_mode']=='observed_extension' or i==0 else 'right',
                spatial_reason='One possible original-camera arrangement, not a measured wall.',functional_links=[])
            if g['realization_mode']=='observed_extension':
                root=next(o for o in g['new_objects'] if o['attachment']['host'] is None)
                g['functional_links']=[dict(source=root['key'],target='obs_000',kind='serves',purpose='An unseen counterpart for the observed arrangement, without a near claim')]
    return raw


class PanoramaFixture(MechanismFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kw):
        path=Path(path);name=path.stem
        if name.startswith('furniture_'):value=proposals()
        elif name.startswith('variant_'):
            if image is not None:raise AssertionError('Variant sees H,D only')
            menu=json.loads(prompt.split('\nALLOWED_VARIANTS: ')[1].split('\n')[0]);j=int(name.split('_')[-1]);value=dict(choice=menu[j%len(menu)]['id'])
        else:return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kw)
        write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,
            canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class PanoramaTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def prepare(self,name='panorama'):
        root=self.temp/name
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=2,draws=4)
        return root

    def bundle(self):
        root=self.prepare()
        for b in ('qwen','api'):
            result=experiment.propose(root,b,PanoramaFixture(b));self.assertEqual(result['status'],'complete',result)
        experiment.pool(root)
        for b in ('qwen','api'):experiment.mechanisms(root,b,PanoramaFixture(b))
        experiment.pool_mechanisms(root);return root

    def fitted(self,policy='functional'):
        root=self.bundle();d=read(root/'shared/domain.json');case=read(root/'shared/case.json')
        return root,model.prepare_model(d['base_model'],case,dict(sdscene_bundle=str(root/'shared'),mechanism_source='qwen',mechanism_policy=policy),None)

    def test_service_only_extension_is_valid_and_does_not_fabricate_near(self):
        raw=proposals();before=deepcopy(raw);p=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertEqual(raw,before)
        self.assertTrue(any(g['functional_links'] for v in p for g in v['groups']))
        self.assertTrue(all(not v['relations'] for v in p))
        prompt=model.furniture_prompt(self.case,model.profile('z',['reading']))
        self.assertNotIn('Describe that functional intent in placement instead.',prompt)

    def test_single_wire_schema_is_accepted_without_a_second_group_rationale(self):
        prompt=model.furniture_prompt(self.case,model.profile('z',['reading']))
        raw=json.loads(prompt.split('\nSINGLE RESPONSE SCHEMA: ')[1].split('\nSCENE HYPOTHESIS: ')[0])
        self.assertEqual(prompt.count('SINGLE RESPONSE SCHEMA:'),1)
        self.assertNotIn('reason',raw['alternatives'][0]['groups'][0])
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertEqual(sum(len(g['objects']) for g in accepted[0]['groups']),1)
        self.assertTrue(accepted[0]['interface_normalizations'])

    def test_recorded_wire_shapes_preserve_inventory_and_group_local_relations(self):
        canonical=proposals();canonical['alternatives']=canonical['alternatives'][:1]
        alt=canonical['alternatives'][0];g=alt['groups'][0];keys=[o['key'] for o in g['new_objects']]
        relation=dict(source=keys[1],target=keys[0],kind='near',reason='Declared adjacency is not optional parser decoration')
        alt['relations']=[relation]
        bound=model.furniture_answer(canonical,self.case,model.profile('z',['reading']))[0]
        raw=deepcopy(canonical)
        for group in raw['alternatives'][0]['groups']:group.pop('reason')
        raw['alternatives'][0]['groups'][0]['relations']=raw['alternatives'][0].pop('relations')
        before=deepcopy(raw);parsed=model.furniture_answer(raw,self.case,model.profile('z',['reading']))[0]
        self.assertEqual(raw,before)
        self.assertEqual(parsed['relations'],bound['relations'])
        self.assertEqual([o for group in parsed['groups'] for o in group['objects']],[o for group in bound['groups'] for o in group['objects']])
        bindings=parsed['interface_normalizations']
        self.assertTrue(any(b['action'].startswith('relocate_') for b in bindings))

    def test_nested_order_cannot_be_silently_ignored_after_rationale_binding(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];g=raw['alternatives'][0]['groups'][0]
        g.pop('reason');g['layout_region']='rear'
        key=next(o['key'] for o in g['new_objects'] if o['attachment']['host'] is None)
        g['relations']=[dict(source=key,target='obs_000',kind='farther_than',reason='An impossible declared rear hypothesis')]
        with self.assertRaisesRegex(ValueError,'inconsistent') as caught:
            model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        message=str(caught.exception)
        self.assertIn('fixed_observed_centers',message);self.assertIn('obs_000',message);self.assertIn('farther_than',message)

    def test_missing_semantic_fields_report_together_and_are_not_inferred(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];g=raw['alternatives'][0]['groups'][0]
        for key in ('realization_mode','layout_region','functional_links'):g.pop(key)
        before=deepcopy(raw)
        with self.assertRaises(ValueError) as caught:model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        for key in ('realization_mode','layout_region','functional_links'):self.assertIn(key,str(caught.exception))
        self.assertEqual(raw,before)

    def test_group_local_cross_assembly_dependencies_remain_invalid(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0]
        first,second=alt['groups'][:2]
        first['relations']=[dict(source=first['new_objects'][0]['key'],target=second['new_objects'][0]['key'],kind='near',reason='Illegal cross-assembly dependency')]
        with self.assertRaisesRegex(ValueError,'Cross-group'):
            model.furniture_answer(raw,self.case,model.profile('z',['reading']))

    def test_exact_anchor_binding_and_functional_links_cannot_cross_groups(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1]
        g=raw['alternatives'][0]['groups'][-1]
        g['functional_links'][0]['target']='obs_wrong'
        with self.assertRaisesRegex(ValueError,'No legal'):model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];g=raw['alternatives'][0]['groups'][-1];g['functional_links']=[]
        with self.assertRaisesRegex(ValueError,'explicitly serve'):model.furniture_answer(raw,self.case,model.profile('z',['reading']))

    def test_rear_and_farther_than_positive_observed_depth_are_rejected_as_inconsistent(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0];g=alt['groups'][0];g['layout_region']='rear'
        key=next(o['key'] for o in g['new_objects'] if o['attachment']['host'] is None)
        alt['relations']=[dict(source=key,target='obs_000',kind='farther_than',reason='incompatible camera depth')]
        with self.assertRaisesRegex(ValueError,'inconsistent'):model.furniture_answer(raw,self.case,model.profile('z',['reading']))

    def test_identical_inventory_in_different_regions_does_not_collapse_in_pool(self):
        p=model.furniture_answer(proposals(),self.case,model.profile('z',['reading']))[0]
        a=deepcopy(p);b=deepcopy(p)
        for g in b['groups']:g['layout_region']='right'
        m=model.build_model([model.profile('z',['reading'])],{'a':a,'b':b})
        self.assertEqual(m['proposal_bindings']['a'],'a');self.assertEqual(m['proposal_bindings']['b'],'b')
        c=deepcopy(a)
        for g in c['groups']:
            g['spatial_reason']='Only explanation wording changed'
            for link in g['functional_links']:link['purpose']='A paraphrase of the same service'
        unchanged=model.build_model([model.profile('z',['reading'])],{'a':a,'c':c})
        self.assertEqual(unchanged['proposal_bindings']['c'],'a')

    def test_spatial_state_probabilities_match_sampling_and_active_channel_mass(self):
        root,m=self.fitted();validate(m);law=model.marginals(m)
        for c,row in law['channel_region_joint_presence'].items():self.assertAlmostEqual(sum(row.values()),law['activation_probabilities'][c])
        samples=model.sample_worlds(m,count=1400);counts=dict.fromkeys(model.REGIONS,0)
        for w in samples:
            self.assertEqual(w['structural_state'],forward(m,w['noise']))
            for r in model.REGIONS:counts[r]+=any(p['region']==r for p in w['spatial_program'].values())
        for r,p in law['any_declared_region_presence'].items():self.assertAlmostEqual(counts[r]/len(samples),p,delta=.045)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1)

    def test_region_intervention_preserves_inventory_but_changes_geometry(self):
        root,m=self.fitted();noise={n['id']:.7 for n in m['nodes']};c='reading__observed_extension';s=m['slots'][c][1]
        do={'A_'+k:'0' for k in m['slots']};do.update({'A_'+c:'1','M_'+c:s,'R_'+c:'rear'})
        rear=model.active_world(m,forward(m,noise,do));left=model.active_world(m,forward(m,noise,dict(do,**{'R_'+c:'left'})))
        self.assertEqual([o['id'] for o in rear['objects']],[o['id'] for o in left['objects']])
        for w in (rear,left):
            root=next(o for o in w['objects'] if o['attachment']['host'] is None)
            w['relations'].append(dict(source=root['id'],target='obs_000',kind='near',reason='Soft proximity must not override sampled camera region'))
        p=PanoramaFixture();plane=support_plane(self.observed,['obs_000']);solved=[]
        for i,w in enumerate((rear,left)):
            path=self.temp/f'layout_{i}.txt';raw=p.ask(None,geometry.layout_prompt(self.case,self.observed,w,plane),path)
            layout=geometry.layout_answer(raw,w,90);solved.append(geometry.compile_layout(self.case,self.observed,w,layout,plane))
        self.assertTrue(solved[0]['review']['geometry_contract_passed'],solved[0]['review'])
        self.assertTrue(all(o['center'][2]+o['size'][2]/2<1e-5 for o in solved[0]['items']))
        self.assertNotEqual([o['center'] for o in solved[0]['items']],[o['center'] for o in solved[1]['items']])
        self.assertEqual([o['size'] for o in solved[0]['items']],[o['size'] for o in solved[1]['items']])
        source=rear['functional_links'][0]['source'];parent=next(o for o in rear['objects'] if o['id']==source)
        after=model.active_world(m,forward(m,noise,dict(do,**{parent['presence_node']:'0'})))
        self.assertTrue(after['unresolved_functional_links'])
        self.assertTrue(all(l['source']!=source for l in after['functional_links']))
        self.assertTrue(any(o['attachment']['host']==source for o in after['objects']))

    def test_direct_inventory_has_no_h_dependency_and_spatial_program_is_retained(self):
        root,m=self.fitted('direct');laws=[model.marginals(m,h['id']) for h in m['profiles']]
        for l in laws[1:]:
            for r,p in laws[0]['any_declared_region_presence'].items():self.assertAlmostEqual(l['any_declared_region_presence'][r],p)
        for n in m['nodes']:
            if n['id'].startswith('R_'):self.assertEqual(len(n['parents']),1);self.assertTrue(n['parents'][0].startswith('M_'))

    def test_region_is_part_of_variant_and_direct_menus_but_reference_gets_no_h_or_image(self):
        root=self.bundle();requests=list((root/'mechanism_calls/qwen').glob('*.request.json'))
        for path in requests:
            req=read(path)
            if path.name.startswith(('variant_','direct_')):self.assertIn('SPATIAL_VARIANTS',req['prompt']);self.assertIn('layout_region',req['prompt'])
            if path.name.startswith('reference_'):self.assertIn('SPATIAL_SUPPORT',req['prompt']);self.assertNotIn('CURRENT_PARENT_H',req['prompt']);self.assertIsNone(req['canonical_image_sha256'])
        for b in ('qwen','api'):
            t=read(root/'shared/mechanisms'/(b+'.json'));self.assertEqual(t['kind'],model.KIND)
        self.assertEqual(read(root/'run.json')['api_model'],'deepseek-ai/DeepSeek-V4.1-Flash')

    def test_all_eight_branches_preserve_observations_regions_inventory_and_functional_links(self):
        root=self.bundle();observed=read(self.root/'qwen/observer3d/G_obs.json')
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                s=run(PanoramaFixture(b),root/exp,b);self.assertNotEqual(s['status'],'failed',s)
                self.assertFalse(s['failures'],s)
                self.assertEqual(len(s['worlds'])+len(s['blocked_worlds']),2)
                for w in s['worlds']:
                    folder=root/exp/b/'completion'/w['folder'];full=read(folder/'G_full.json');review=read(folder/'numeric_review.json')
                    self.assertEqual(full['objects'][:len(observed['objects'])],observed['objects']);self.assertFalse(review['spatial_program_violations'])
                    self.assertIn('functional_links',full);self.assertIn('spatial_program',full)
        controls=[root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')]
        self.assertEqual(sha(controls[0]),sha(controls[1]));page=report(root);self.assertTrue((page.parent/'panoramic_angles.png').exists())
        audit=read(page.parent/'review.json');self.assertIn('panoramic_spatial_review',audit)

    def test_no_rear_candidate_is_reported_as_missing_support_not_fabricated(self):
        root=self.prepare();p=PanoramaFixture();original=p.ask
        def only_side(image,prompt,path,**kw):
            value=original(image,prompt,path,**kw)
            if Path(path).stem.startswith('furniture_'):
                for alt in value['alternatives']:
                    for g in alt['groups']:g['layout_region']='left'
                Path(path).write_text(json.dumps(value));write(Path(path).with_suffix('.json'),value)
            return value
        p.ask=only_side
        for b in ('qwen','api'):experiment.propose(root,b,p)
        result=experiment.pool(root);self.assertFalse(result['rear_program_proposed'])
        self.assertTrue(all('rear' not in states for states in result['spatial_support'].values()))

    def test_empty_world_is_allowed_and_regions_do_not_create_objects(self):
        root,m=self.fitted();noise={n['id']:.2 for n in m['nodes']}
        do={'A_'+c:'0' for c in m['slots']};do.update({'R_'+c:'rear' for c in m['slots']})
        w=model.active_world(m,forward(m,noise,do));self.assertFalse(w['objects']);self.assertFalse(w['spatial_program'])

    def test_all_blocked_dependencies_still_render_without_a_causal_model_file(self):
        root=self.prepare()
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):experiment.block(root,exp,b)
        page=report(root);self.assertIn('blocked_dependency',page.read_text())


if __name__=='__main__':unittest.main()
