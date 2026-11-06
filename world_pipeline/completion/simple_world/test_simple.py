"""Causal and joint-inventory regressions; synthetic LM is not a quality test."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ...common import ROOT, read, sha
from ..world_scm import forward
from ..test_multiscene import CPUProvider
from ..candidate_repair.test_candidates import wire
from . import contract as c, model as m, workflow as w

SOURCE = ROOT / 'qwen_edit_pano/outputs/local_observation_fixed_002'
LATEST = ROOT / 'qwen_edit_pano/outputs/v13_candidates_fixed_001'
SID = '8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39'
CASE = LATEST / 'scenes' / SID / 'completion/shared/case.json'


def fixture():
    def obj(key, category, host=None):
        return dict(key=key, category=category, variant='Synthetic '+category,
            reason='Separate hypothetical unseen '+category, placement='Around the unseen workspace',
            attachment=dict(kind='top' if host else 'floor',host=host),affordances=['top'] if not host else [])
    return dict(alternatives=[dict(description='Synthetic coexisting work and storage world',groups=[
        dict(function='work',reason='Hypothetical work arrangement',layout_region='left',observed_refs=[],
            new_objects=[obj('desk','desk'),obj('lamp','table lamp','desk')],
            functional_links=[dict(source='lamp',target='desk',kind='illuminates',purpose='Lighting intent')]),
        dict(function='storage',reason='Hypothetical storage arrangement',layout_region='rear',observed_refs=[],
            new_objects=[obj('cabinet','cabinet'),obj('box','box','cabinet')],functional_links=[])],relations=[])])


class ProposalCPU:
    def __init__(self): self.requests=[]
    def ask(self,image,prompt,path,**kw):
        self.requests.append(str(path)); path=Path(path)
        value = dict(hypotheses=[dict(description='Possible joint work and storage organization',
            functions=['work','storage'],assumptions=['Unseen surrounding space may support both uses'])]) if path.stem.startswith('hypotheses') else fixture()
        path.write_text(json.dumps(value)); w.save(path.with_suffix('.json'),value)
        return value


class SimpleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'research/audits',prefix='simple_CPU_')
        self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        self.case=read(CASE)
        self.hs=c.hypotheses_answer(dict(hypotheses=[dict(description='Possible coexisting work and storage',
            functions=['work','storage'],assumptions=['Unknown hidden space'])]))

    def fitted(self,raw=None,hs=None):
        hs=hs or self.hs; programs={}
        for h in hs:
            for p in c.furniture_answer(raw or fixture(),self.case,h): programs['f'+str(len(programs))]=p
        return m.prepare_model(m.build_model(hs,programs),self.case,{},None)

    def test_whole_program_never_splits_roles_or_drops_objects(self):
        model=self.fitted(); worlds=m.sample_worlds(model,count=4)
        self.assertFalse(any(n['id'].startswith(('A_','M_','J','H')) for n in model['nodes']))
        for world in worlds:
            self.assertEqual(len(world['objects']),4)
            self.assertEqual(world['hypothesis'],self.hs[0])
            self.assertEqual({g['function'] for g in world['groups']},{'work','storage'})
            self.assertEqual([g['layout_region'] for g in world['groups']],['left','rear'])
            self.assertEqual(len(world['functional_links']),1)
            self.assertFalse(world['relations'])

    def test_host_intervention_keeps_child_owned_but_uninstalled(self):
        model=self.fitted(); world=m.sample_worlds(model,count=1)[0]
        host=next(o for o in world['objects'] if o['key']=='desk')
        child=next(o for o in world['objects'] if o['key']=='lamp')
        after=m.active_world(model,forward(model,world['noise'],{host['presence_node']:'0'}))
        self.assertIn(child['id'],{o['id'] for o in after['objects']})
        self.assertIn(child['id'],after['unresolved_installations'])
        self.assertTrue(m.intervention_diagnostics(model,world)['queries'])

    def test_hypothesis_intervention_changes_joint_program_with_same_noise(self):
        hs=deepcopy(self.hs); hs.append(dict(hs[0],id='z1',description='Different possible organization'))
        model=self.fitted(hs=hs); world=m.sample_worlds(model,count=1)[0]
        other=next(h['id'] for h in hs if h['id']!=world['structural_state']['Z'])
        after=m.active_world(model,forward(model,world['noise'],{'Z':other}))
        self.assertNotEqual(world['structural_state']['F'],after['structural_state']['F'])
        self.assertEqual(len(after['objects']),4)

    def test_empty_and_duplicate_samples_are_retained_without_new_null_mass(self):
        raw=dict(alternatives=[dict(description='Possible observed-only inventory',groups=[],relations=[])])
        model=self.fitted(raw); worlds=m.sample_worlds(model,count=4)
        self.assertEqual(len(worlds),4); self.assertTrue(all(not w['objects'] for w in worlds))
        self.assertEqual(len(model['furniture_states']),1)
        self.assertNotIn('f_null',model['furniture_states'])

    def test_newest_real_accepted_programs_roundtrip_without_inventory_change(self):
        count=0
        for path in LATEST.glob('scenes/*/completion/proposals/qwen/programs.json'):
            case=read(path.parents[2]/'shared/case.json')
            for p in read(path).values():
                raw=wire(p)
                if any(l['kind']=='contained_in' for g in raw['alternatives'][0]['groups'] for l in g['functional_links']): continue
                before=sha(path); result=c.furniture_answer(raw,case,p['hypothesis'])[0]
                self.assertEqual({o['key'] for g in p['groups'] for o in g['objects']},
                                 {o['key'] for g in result['groups'] for o in g['objects']})
                self.assertEqual(before,sha(path)); count+=1
        self.assertGreater(count,15)

    def test_redundant_containment_and_unknown_host_are_not_silently_fixed(self):
        raw=fixture(); raw['alternatives'][0]['groups'][0]['functional_links'].append(
            dict(source='lamp',target='desk',kind='contained_in',purpose='Contradicts top'))
        with self.assertRaisesRegex(ValueError,'do not duplicate'): c.furniture_answer(raw,self.case,self.hs[0])
        raw=fixture(); raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
        with self.assertRaises(ValueError): c.furniture_answer(raw,self.case,self.hs[0])

    def test_real_latest_containment_failure_has_actionable_feedback_without_guessing(self):
        comp=LATEST/'scenes/X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c/completion'
        path=comp/'proposal_contract_rounds/round_01/calls/furniture_z0_surrounding_attempt1.json'
        before=sha(path); raw=read(path)
        with self.assertRaisesRegex(ValueError,'do not duplicate'):
            c.furniture_answer(raw,read(comp/'shared/case.json'),self.hs[0])
        self.assertEqual(sha(path),before)

    def test_proposal_rejection_is_not_an_empty_world_or_execution_exception(self):
        comp=self.root/'case'; w.save(comp/'shared/case.json',self.case)
        w.save(comp/'run.json',dict(seed=3407)); w.save(comp/'context/qwen/observer2d/grounding.json',{})
        shutil.copyfile(LATEST/'scenes'/SID/'completion/shared/observation_board.png',comp/'shared/observation_board.png')
        class Rejected(ProposalCPU):
            def ask(inner,image,prompt,path,**kw):
                if Path(path).stem.startswith('joint'): raise ValueError('invalid typed host')
                return super().ask(image,prompt,path,**kw)
        result=w.propose_scene(comp,lambda:Rejected())
        self.assertEqual(result['status'],'empty_candidate_domain')
        self.assertFalse(result['execution_failures']); self.assertTrue(result['rejections'])
        self.assertFalse(read(comp/'shared/joint_domain.json')['programs'])
        self.assertFalse((comp/'end_to_end/qwen/completion/G_full.json').exists())

    def test_runtime_failure_is_kept_as_execution_failure(self):
        comp=self.root/'case'; w.save(comp/'shared/case.json',self.case)
        w.save(comp/'run.json',dict(seed=3407)); w.save(comp/'context/qwen/observer2d/grounding.json',{})
        shutil.copyfile(LATEST/'scenes'/SID/'completion/shared/observation_board.png',comp/'shared/observation_board.png')
        class Broken:
            def ask(inner,*a,**kw): raise RuntimeError('synthetic model runtime failure')
        result=w.propose_scene(comp,lambda:Broken())
        self.assertEqual(result['status'],'failed'); self.assertTrue(result['execution_failures'])
        self.assertFalse(result['rejections'])

    def test_cache_integrity_valueerror_is_not_a_semantic_warning(self):
        comp=self.root/'case'; w.save(comp/'shared/case.json',self.case)
        w.save(comp/'run.json',dict(seed=3407)); w.save(comp/'context/qwen/observer2d/grounding.json',{})
        shutil.copyfile(LATEST/'scenes'/SID/'completion/shared/observation_board.png',comp/'shared/observation_board.png')
        provider=ProposalCPU(); first=w.propose_scene(comp,lambda:provider)
        self.assertEqual(first['status'],'complete')
        path=comp/'proposals/qwen/calls/hypotheses_attempt0.txt'; path.write_text('corrupt')
        second=w.propose_scene(comp,lambda:provider)
        self.assertEqual(second['status'],'failed'); self.assertTrue(second['execution_failures'])
        self.assertFalse(second['rejections']); self.assertEqual(len(provider.requests),2)

    def test_full_script_path_original_geometry_real_render_and_resume(self):
        source=self.root/'source'; source.mkdir(); suite=read(SOURCE/'suite.json')
        suite['scenes']=[s for s in suite['scenes'] if s['id']==SID]; w.save(source/'suite.json',suite)
        shutil.copytree(SOURCE/'scenes'/SID/'observation',source/'scenes'/SID/'observation')
        root=self.root/'run'; proposal=ProposalCPU(); numeric=CPUProvider(all_present=True)
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'), \
             patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            w.prepare(root,source,worlds=2); w.phase(root,'prepare-scenes')
            with patch.object(w,'provider',return_value=proposal): w.phase(root,'propose')
            with patch.object(w,'provider',return_value=numeric): w.phase(root,'worlds')
            result=w.report(root)
            self.assertEqual(result['worlds_with_new_objects'],2, result.get('execution_errors'))
            self.assertFalse(result['execution_errors']); self.assertEqual(result['LM_probability_table_calls'],0)
            self.assertTrue(all(r['observed_preserved'] and r['placed_new']==4 for r in result['records']))
            self.assertEqual(len(proposal.requests),2)
            comp=root/'scenes'/SID/'completion'
            self.assertFalse((comp/'shared/domain.json').exists())
            self.assertFalse((comp/'shared/mechanisms/qwen.json').exists())
            self.assertTrue((root/'report'/SID/'world_01/views/overview.png').is_file())
            state=read(comp/'end_to_end/qwen/completion/world_01/G_full.json')
            self.assertEqual(len(state['functional_links']),1)
            with patch.object(w,'provider',side_effect=AssertionError('Resume made a new model call')):
                w.phase(root,'propose'); w.phase(root,'worlds'); second=w.report(root)
            self.assertEqual(result['records'],second['records'])


if __name__=='__main__': unittest.main()
