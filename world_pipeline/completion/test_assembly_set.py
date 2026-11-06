"""Meaningful joint-set, dependency-atomic and matched-evidence contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import read,write,sha
from . import assembly_set as model,assembly_set_experiment as experiment
from .test_panorama import PanoramaFixture,proposals
from . import test_assembly as fixture
from .assembly_world import run
from .assembly_set_visual import report
from .world_scm import forward
from .world_generation import image_bytes
from .assembly_mechanism_visual import inventory_law


class AssemblySetFixture(PanoramaFixture):
    def ask(self,image,prompt,path,seed=3407,max_tokens=8192,**kwargs):
        path=Path(path);name=path.stem
        if name.startswith(('reference_','activation_','tag_activation_','direct_')):
            menu=json.loads(prompt.split('\nCANDIDATE_ASSEMBLIES: ')[1].split('\n')[0])
            functions=json.loads(prompt.split('\nCURRENT_PARENT_H: ')[1].split('\n')[0])['functions'] if '\nCURRENT_PARENT_H:' in prompt else []
            active=[m['id'] for m in menu] if image is not None and ('reading' in functions or name.startswith('direct_')) else [menu[0]['id']]
            value=dict(active_assemblies=active)
            write(path.with_suffix('.request.json'),dict(prompt=prompt,seed=seed,max_tokens=max_tokens,
                canonical_image_sha256=hashlib.sha256(image_bytes(image)).hexdigest() if image is not None else None))
            path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)
        return super().ask(image,prompt,path,seed=seed,max_tokens=max_tokens,**kwargs)


class AssemblySetTests(unittest.TestCase):
    setUp=fixture.AssemblyTests.setUp
    initialize=fixture.AssemblyTests.initialize

    def prepare(self):
        root=self.temp/'assembly_set'
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]),\
             patch('qwen_edit_pano.world_pipeline.completion.assembly_sdscene_experiment.new_output',side_effect=lambda p:(p.mkdir(),p)[1]):
            experiment.prepare(root,self.root,worlds=2,draws=4)
        return root

    def bundle(self):
        root=self.prepare()
        for b in ('qwen','api'):experiment.propose(root,b,AssemblySetFixture(b))
        experiment.pool(root)
        for b in ('qwen','api'):experiment.mechanisms(root,b,AssemblySetFixture(b))
        experiment.pool_mechanisms(root);return root

    def fitted(self,root,experiment_name='end_to_end'):
        d=read(root/'shared/domain.json');config=read(root/experiment_name/'experiment.json');config['execution_backend']='qwen'
        return model.prepare_model(d['base_model'],read(root/'shared/case.json'),config,None)

    def test_schema_has_no_extra_zone_or_anchor_gate(self):
        prompt=model.furniture_prompt(self.case,model.profile('z',['reading']))
        raw=json.loads(prompt.split('\nSINGLE RESPONSE SCHEMA: ')[1].split('\n')[0]);g=raw['alternatives'][0]['groups'][0]
        self.assertNotIn('realization_mode',g)
        g['observed_refs']=['obs_000'];g.pop('functional_links');g.pop('layout_region')
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertEqual(len(accepted),1);self.assertTrue(accepted[0]['groups'][0]['objects'])
        self.assertEqual(accepted[0]['groups'][0]['layout_region'],'unspecified')

    def test_bad_independent_unit_does_not_erase_valid_complete_units(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0]
        bad=deepcopy(alt['groups'][0]);renamed={o['key']:'bad_'+o['key'] for o in bad['new_objects']}
        for o in bad['new_objects']:
            o['key']=renamed[o['key']];o['attachment']['host']=renamed.get(o['attachment']['host'],o['attachment']['host'])
        for link in bad.get('functional_links',[]):
            for key in ('source','target'):link[key]=renamed.get(link[key],link[key])
        bad['new_objects'][0]['attachment']=dict(kind='top',host='unknown_host')
        alt['groups'].append(bad);before=deepcopy(raw)
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertEqual(raw,before)
        self.assertTrue(accepted[0]['proposal_validation']['rejected'])
        self.assertFalse(any(o['key'].startswith('bad_') for p in accepted for g in p['groups'] for o in g['objects']))
        self.assertEqual(sum(len(g['objects']) for p in accepted for g in p['groups']),sum(len(g['new_objects']) for g in alt['groups'][:-1]))

    def test_cross_unit_dependency_is_rejected_atomically(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0];first,second=alt['groups'][:2]
        first['new_objects'][0]['attachment']=dict(kind='top',host=second['new_objects'][0]['key'])
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        removed={o['key'] for g in (first,second) for o in g['new_objects']}
        self.assertFalse(any(o['key'] in removed for p in accepted for g in p['groups'] for o in g['objects']))
        self.assertTrue(accepted[0]['proposal_validation']['rejected'])

    def test_valid_cross_group_host_dependency_is_one_atomic_candidate(self):
        prompt=model.furniture_prompt(self.case,model.profile('z',['reading']))
        raw=json.loads(prompt.split('\nSINGLE RESPONSE SCHEMA: ')[1].split('\n')[0]);alt=raw['alternatives'][0];host=alt['groups'][0]
        child=dict(function='storage',new_objects=[dict(key='book_child',category='book',variant='hardcover novel',
            reason='A separate stored book',placement='on the bookcase shelf',attachment=dict(kind='shelf',host='unique_key'),affordances=[])])
        alt['groups'].append(child)
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertEqual(len(accepted),1)
        self.assertEqual(sum(len(g['objects']) for g in accepted[0]['groups']),2)
        self.assertEqual(accepted[0]['groups'][0]['function_tags'],['reading','storage'])

    def test_actual_camera_contradiction_still_rejects_whole_affected_unit(self):
        raw=proposals();raw['alternatives']=raw['alternatives'][:1];alt=raw['alternatives'][0];g=alt['groups'][0];g['layout_region']='rear'
        key=next(o['key'] for o in g['new_objects'] if o['attachment']['host'] is None)
        alt['relations']=[dict(source=key,target='obs_000',kind='farther_than',reason='contradictory rear hypothesis')]
        accepted=model.furniture_answer(raw,self.case,model.profile('z',['reading']))
        self.assertFalse(any(o['key']==key for p in accepted for g in p['groups'] for o in g['objects']))
        self.assertTrue(any('inconsistent' in r['error'] for r in accepted[0]['proposal_validation']['rejected']))

    def test_same_function_units_can_coexist_and_inventory_is_not_function_capped(self):
        root=self.bundle();d=read(root/'shared/domain.json');m=self.fitted(root)
        self.assertFalse(d['function_slot_exclusivity']);self.assertTrue(all(len(s)==2 for s in m['slots'].values()))
        counts={}
        for c,s in d['slots'].items():
            function=d['programs'][s[1]]['groups'][0]['function'];counts.setdefault(function,[]).append(c)
        channels=next(v for v in counts.values() if len(v)>=2)
        noise={n['id']:.5 for n in m['nodes']};do={'A_'+c:'0' for c in m['slots']};do.update({'A_'+c:'1' for c in channels[:2]})
        w=model.active_world(m,forward(m,noise,do))
        self.assertEqual(len({o['furniture_state'] for o in w['objects']}),2)
        self.assertEqual(len(w['objects']),sum(len(g['objects']) for c in channels[:2] for g in d['programs'][d['slots'][c][1]]['groups']))
        samples=model.sample_worlds(m,count=600);law=model.marginals(m)
        self.assertAlmostEqual(sum(law['hidden_count_probabilities'].values()),1)
        self.assertAlmostEqual(sum(len(w['objects']) for w in samples)/len(samples),law['expected_hidden_instances'],delta=.8)

    def test_rgb_and_h_change_numeric_probability_without_an_explanation_gate(self):
        root=self.bundle();full=self.fitted(root);tag=self.fitted(root,'tag_only')
        self.assertNotEqual(model.marginals(full)['object_presence'],model.marginals(tag)['object_presence'])
        hs=full['profiles'];reading=next(h['id'] for h in hs if 'reading' in h['functions']);empty=next(h['id'] for h in hs if not h['functions'])
        self.assertNotEqual(model.marginals(full,reading)['object_presence'],model.marginals(full,empty)['object_presence'])
        self.assertIn('O_RGB',full['fixed_context_parents']['joint_selection']);self.assertNotIn('O_RGB',tag['fixed_context_parents']['joint_selection'])

    def test_paired_prompts_evidence_and_initial_selection_budgets_match(self):
        root=self.bundle();first=read(root/'shared/mechanisms/qwen.json');second=read(root/'shared/mechanisms/api.json')
        self.assertEqual(first['original_parent_prompt_plan'],second['original_parent_prompt_plan'])
        plans=[p for p in first['original_parent_prompt_plan'] if 'prompt_sha256' in p]
        functional=[p for p in plans if p['stage'].startswith('activation_')];direct=[p for p in plans if p['stage'].startswith('direct_')]
        self.assertEqual(len(functional),len(direct));self.assertTrue(all(p['max_tokens']==512 for p in functional+direct))
        for p in plans:
            if p['stage'].startswith('activation_'):self.assertTrue(p['image_delivered']);self.assertIn('O_RGB',p['parents'])
            if p['stage'].startswith(('tag_activation_','reference_')):self.assertFalse(p['image_delivered'])

    def test_empty_inventory_and_host_installation_are_not_count_targets(self):
        root=self.bundle();m=self.fitted(root);noise={n['id']:.4 for n in m['nodes']}
        w=model.active_world(m,forward(m,noise,{'A_'+c:'0' for c in m['slots']}));self.assertFalse(w['objects']);self.assertFalse(w['spatial_program'])

    def test_h_cut_preserves_entire_unconditional_unit_inventory_law(self):
        root=self.bundle();full=inventory_law(self.fitted(root));cut=inventory_law(self.fitted(root,'function_free'))
        self.assertIsNotNone(full);self.assertIsNotNone(cut)
        self.assertLess(sum(abs(full.get(k,0)-cut.get(k,0)) for k in set(full)|set(cut)),1e-9)

    def test_tag_only_readiness_checks_its_own_rgb_protocol(self):
        root=self.bundle();path=root/'shared/mechanisms/qwen_tag_only.json';table=read(path);table['conditional_RGB']=True;write(path,table)
        self.assertTrue(experiment.ready(root,'end_to_end','qwen')['ready'])
        self.assertFalse(experiment.ready(root,'tag_only','qwen')['ready'])

    def test_all_ten_branches_preserve_full_inventory_and_observed_state(self):
        root=self.bundle();observed=read(self.root/'qwen/observer3d/G_obs.json')
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):
                self.assertTrue(experiment.ready(root,exp,b)['ready'])
                s=run(AssemblySetFixture(b),root/exp,b)
                self.assertEqual(len(s['worlds'])+len(s['blocked_worlds']),2,s)
                for record in s['worlds']+s['blocked_worlds']:
                    folder=root/exp/b/'completion'/record['folder'];state=read(folder/record.get('state_file','G_full.json'));sem=read(folder/'semantic_world.json')
                    self.assertEqual(state['objects'][:len(observed['objects'])],observed['objects'])
                    self.assertEqual(len(state['objects'])-len(observed['objects']),len(sem['objects']))
                    self.assertIn('spatial_program',state)
        paths=[root/'layout_control'/b/'completion/structural_samples.json' for b in ('qwen','api')];self.assertEqual(sha(paths[0]),sha(paths[1]))
        page=report(root);self.assertTrue(page.exists());self.assertIn('联合功能装配',page.read_text())

    def test_tag_only_dependency_and_all_blocked_visualization(self):
        root=self.prepare()
        for exp in experiment.EXPERIMENTS:
            for b in ('qwen','api'):experiment.block(root,exp,b)
        page=report(root);self.assertIn('blocked_dependency',page.read_text())


if __name__=='__main__':unittest.main()
