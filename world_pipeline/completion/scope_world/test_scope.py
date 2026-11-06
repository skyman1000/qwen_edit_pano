"""Global-scope contracts and executable dependency/layout integration."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ...common import ROOT,read,sha
from ..world_scm import forward
from ..test_multiscene import CPUProvider
from ..simple_world.test_simple import fixture,SOURCE,SID
from . import contract as c,model as m,workflow as w

LATEST=ROOT/'qwen_edit_pano/outputs/simple_world_001'
CASE=LATEST/'scenes'/SID/'completion/shared/case.json'


def scope_fixture():
    return dict(hypotheses=[dict(description='Possible unseen work and storage surrounding the observed lounge',
        room_kind='Lounge with possible surrounding work and storage areas',functions=['work','storage'],
        assumptions=['Extent and use of unseen space are uncertain'],visible_unmodelled=[],
        hidden_areas=[dict(id='a0',function='work',region='left',configuration='Desk and task-light arrangement',
            why_plausible='Possible quiet use in unphotographed floor space',anchor_refs=[]),
            dict(id='a1',function='storage',region='rear',configuration='Storage cabinet and supported box',
            why_plausible='Ordinary storage may coexist beyond the camera',anchor_refs=[])],
        unmodelled_space='Remaining room boundaries and other contents are unknown')])


def area_fixture(area):
    group=deepcopy(fixture()['alternatives'][0]['groups'][0 if area['function']=='work' else 1])
    return dict(status='instantiated',reason=group['reason'],new_objects=group['new_objects'],
        relations=[],functional_links=group['functional_links'])


class ScopeCPU:
    def __init__(self):self.requests=[]
    def ask(self,image,prompt,path,**kw):
        self.requests.append(str(path));path=Path(path)
        if path.stem.startswith('surrounding_scope'): value=scope_fixture()
        else:
            area=json.loads(prompt.split('\nREQUESTED_AREA: ')[1].split('\n')[0]); value=area_fixture(area)
        path.write_text(json.dumps(value));w.save(path.with_suffix('.json'),value);return value


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'research/audits',prefix='scope_CPU_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.case=read(CASE);self.h=c.hypotheses_answer(scope_fixture(),self.case)[0]

    def program(self,h=None,unresolved=None):
        h=h or self.h; answers={}
        for area in h['hidden_areas']:
            value=area_fixture(area) if area['id']!=unresolved else dict(status='unresolved',reason='Area specification unknown',new_objects=[])
            answers[area['id']]=c.area_answer(value,self.case,h,area,24)
        return c.assemble(self.case,h,answers)

    def fitted(self,programs):
        hs=[p['hypothesis'] for p in programs.values()]
        return m.prepare_model(m.build_model(hs,programs),self.case,{},None)

    def test_all_planned_areas_coexist_and_object_host_ids_are_namespaced(self):
        p=self.program();self.assertTrue(p['scope_complete_within_hypothesis'])
        self.assertEqual(len(p['scope_ledger']),2);self.assertEqual(len(p['groups']),2)
        objects={o['key']:o for g in p['groups'] for o in g['objects']}
        self.assertEqual(len(objects),4);self.assertEqual(objects['a0_lamp']['attachment']['host'],'a0_desk')

    def test_unknown_area_is_not_absence_and_other_area_is_not_deleted(self):
        p=self.program(unresolved='a1');self.assertFalse(p['scope_complete_within_hypothesis'])
        self.assertEqual(p['unresolved_areas'],['a1'])
        self.assertEqual(sum(len(g['objects']) for g in p['groups']),2)
        self.assertEqual(p['scope_ledger'][1]['status'],'unresolved')

    def test_no_area_means_unknown_surrounding_organization(self):
        h=deepcopy(self.h);h['hidden_areas']=[];p=c.assemble(self.case,h,{})
        self.assertFalse(p['groups']);self.assertFalse(p['scope_complete_within_hypothesis'])

    def test_schema_placeholders_are_not_accepted_as_room_hypotheses(self):
        for field in ('room_kind','description','unmodelled_space'):
            value=scope_fixture();value['hypotheses'][0][field]='<concrete surrounding configuration>'
            with self.assertRaisesRegex(ValueError,'placeholder'):c.hypotheses_answer(value,self.case)

    def test_explicit_empty_is_distinct_from_unknown(self):
        a=self.h['hidden_areas'][0]
        empty=c.area_answer(dict(status='assumed_empty',reason='Assume this area is open circulation space',new_objects=[]),self.case,self.h,a,24)
        self.assertEqual(empty['status'],'assumed_empty')
        with self.assertRaises(ValueError):c.area_answer(dict(status='unresolved',reason='Unknown',new_objects=area_fixture(a)['new_objects']),self.case,self.h,a,24)

    def test_region_and_remaining_capacity_are_contracts_not_random_reassignments(self):
        a=self.h['hidden_areas'][0]
        value=c.area_answer(area_fixture(a),self.case,self.h,a,24)
        self.assertEqual(value['group']['layout_region'],'left')
        with self.assertRaises(ValueError):c.area_answer(area_fixture(a),self.case,self.h,a,1)

    def test_export_covers_each_program_before_geometric_repeats(self):
        p0=self.program();h=deepcopy(self.h);h['id']='z1';h['description']='Alternative whole-room arrangement'
        h['hidden_areas'][0]['region']='right';p1=self.program(h)
        model=self.fitted(dict(f0=p0,f1=p1));worlds=m.sample_worlds(model,count=4)
        self.assertEqual([v['structural_state']['F'] for v in worlds],['f0','f1','f0','f1'])
        self.assertEqual([v['geometry_variant'] for v in worlds],[False,False,True,True])
        for world in worlds:self.assertEqual(forward(model,world['noise']),world['structural_state'])

    def test_host_removal_retains_owned_child_and_invalidates_installation(self):
        model=self.fitted(dict(f0=self.program()));world=m.sample_worlds(model,count=1)[0]
        host=next(o for o in world['objects'] if o['key']=='a0_desk');child=next(o for o in world['objects'] if o['key']=='a0_lamp')
        after=m.active_world(model,forward(model,world['noise'],{host['presence_node']:'0'}))
        self.assertIn(child['id'],[o['id'] for o in after['objects']]);self.assertIn(child['id'],after['unresolved_installations'])

    def test_real_previous_caption_cannot_pass_as_an_explicit_surrounding_plan(self):
        path=LATEST/'scenes'/SID/'completion/proposals/qwen/calls/hypotheses_attempt0.json';before=sha(path)
        with self.assertRaises(ValueError):c.hypotheses_answer(read(path),self.case)
        self.assertEqual(sha(path),before)

    def test_real_three_repairs_cannot_turn_nonempty_inventory_into_empty(self):
        paths=list(LATEST.glob('scenes/*/completion/proposals/qwen/calls/joint_*_attempt1.json'));count=0
        for path in paths:
            initial=read(path.with_name(path.name.replace('attempt1','attempt0')));repaired=read(path)
            raw=[o for a in initial['alternatives'] for g in a['groups'] for o in g['new_objects']]
            after=[o for a in repaired['alternatives'] for g in a['groups'] for o in g['new_objects']]
            if not raw or after:continue
            case=read(path.parents[3]/'shared/case.json');area=self.h['hidden_areas'][0]
            validator=c.AreaValidator(case,self.h,area,24)
            # Use the actual initial records. Their original host/installation
            # errors may reject them; that must not erase the semantic draw.
            try:validator(dict(status='instantiated',reason='Original proposed area',new_objects=raw,relations=[],functional_links=[]))
            except ValueError:pass
            self.assertEqual(validator.commitment,{o['key']:o['category'] for o in raw})
            with self.assertRaisesRegex(ValueError,'changed the proposed inventory'):
                validator(dict(status='assumed_empty',reason='A correction returned an empty inventory',new_objects=[]))
            count+=1
        self.assertEqual(count,3)

    def test_schema_correction_may_fix_fields_without_changing_instances(self):
        area=self.h['hidden_areas'][0];raw=area_fixture(area);raw['new_objects'][1]['attachment']['host']='unknown'
        validator=c.AreaValidator(self.case,self.h,area,24)
        with self.assertRaisesRegex(ValueError,'Preserve these exact proposed keys/categories'):validator(raw)
        fixed=area_fixture(area);result=validator(fixed)
        self.assertEqual(result['status'],'instantiated')
        self.assertEqual(len(result['group']['new_objects']),2)

    def test_failed_inventory_correction_is_replayable_without_empty_mass_or_calls(self):
        comp=self.root/'case';w.save(comp/'shared/case.json',self.case);w.save(comp/'run.json',dict(seed=3407))
        shutil.copyfile(LATEST/'scenes'/SID/'completion/shared/observation_board.png',comp/'shared/observation_board.png')
        class BrokenArea(ScopeCPU):
            def ask(inner,image,prompt,path,**kw):
                name=Path(path).stem
                if name.startswith('z0_a0'):
                    inner.requests.append(str(path));value=area_fixture(self.h['hidden_areas'][0])
                    if name.endswith('attempt0'):value['new_objects'][1]['attachment']['host']='unknown'
                    else:value=dict(status='assumed_empty',reason='Attempted to clear the failed proposal',new_objects=[])
                    Path(path).write_text(json.dumps(value));w.save(Path(path).with_suffix('.json'),value);return value
                return super().ask(image,prompt,path,**kw)
        provider=BrokenArea();first=w.propose_scene(comp,lambda:provider)
        self.assertFalse(first['execution_failures']);self.assertTrue(first['rejections'])
        p=read(comp/'shared/joint_domain.json')['programs']['f0']
        self.assertFalse(p['scope_complete_within_hypothesis'])
        self.assertEqual(p['scope_ledger'][0]['uninstantiated_inventory'],{'desk':'desk','lamp':'table lamp'})
        self.assertEqual(sum(len(g['objects']) for g in p['groups']),2)
        n=len(provider.requests);second=w.propose_scene(comp,lambda:provider)
        self.assertEqual(len(provider.requests),n);self.assertEqual(first,second)
        self.assertEqual(p,read(comp/'shared/joint_domain.json')['programs']['f0'])

    def test_full_frozen_observation_geometry_render_and_zero_call_resume(self):
        source=self.root/'source';source.mkdir();suite=read(SOURCE/'suite.json');suite['scenes']=[s for s in suite['scenes'] if s['id']==SID]
        w.save(source/'suite.json',suite);shutil.copytree(SOURCE/'scenes'/SID/'observation',source/'scenes'/SID/'observation')
        root=self.root/'run';proposal=ScopeCPU();numeric=CPUProvider(all_present=True)
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            w.prepare(root,source,worlds=2);w.phase(root,'prepare-scenes')
            with patch.object(w,'provider',return_value=proposal):w.phase(root,'propose')
            with patch.object(w,'provider',return_value=numeric):w.phase(root,'worlds')
            result=w.report(root);self.assertFalse(result['execution_errors']);self.assertEqual(result['worlds_with_new_objects'],2)
            self.assertFalse(result['scope_incomplete_worlds']);self.assertTrue(all(r['placed_new']==4 and r['observed_preserved'] for r in result['records']))
            self.assertEqual(len(proposal.requests),3);self.assertTrue((root/'scope.html').exists())
            self.assertTrue((root/'report'/SID/'world_01/views/overview.png').exists())
            with patch.object(w,'provider',side_effect=AssertionError('Resume made a new model call')):
                w.phase(root,'propose');w.phase(root,'worlds');again=w.report(root)
            self.assertEqual(result['records'],again['records'])


if __name__=='__main__':unittest.main()
