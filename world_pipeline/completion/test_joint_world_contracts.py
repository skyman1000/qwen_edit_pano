"""Regression tests for real-reply boundaries and downstream failure handling."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import ROOT,read,write
from . import joint_world as runner,joint_world_model as model,joint_world_geometry as geometry
from .joint_world_requests import numeric_fields
from .joint_world_replay import BASELINE,FAILED,SOURCE,baseline_replies,case_for,request_hypothesis
from . import test_joint_world as fixtures
from .test_joint_world import hypotheses,programs_reply


class Replies:
    def __init__(self,values):self.values=iter(values);self.prompts=[]
    def ask(self,image,prompt,path,**kwargs):
        self.prompts.append(prompt);v=next(self.values);Path(path).write_text(json.dumps(v));return v


class ContractTests(unittest.TestCase):
    setUp=fixtures.JointWorldTests.setUp

    def test_all_five_real_H_replies_no_longer_require_numeric_zones(self):
        for s in read(BASELINE/'suite.json')['scenes']:
            case,_=case_for(s['id'])
            p=FAILED/'scenes'/s['id']/'joint_completion/calls/hypotheses.json'
            hs=model.hypotheses_answer(read(p),case)
            self.assertTrue(hs)
            self.assertTrue(all(set(h)=={'id','description','functions','assumptions'} for h in hs))

    def test_all_stored_v13_replies_are_censused_and_preserve_accepted_instances(self):
        from .assembly_delta import furniture_answer
        for s in read(BASELINE/'suite.json')['scenes']:
            case,_=case_for(s['id']);accepted=0;seen=0
            for req in baseline_replies(s['id']):
                seen+=1;raw=read(req.with_name(req.name.replace('.request.json','.json')));h=request_hypothesis(req)
                # Compare every legal sibling to the unchanged v13 validator.
                try:r=model.programs_answer(raw,case,h)
                except ValueError:
                    with self.assertRaises(ValueError):furniture_answer(raw,case,h)
                    continue
                accepted+=1
                for p in r['valid']:
                    old=furniture_answer(dict(alternatives=[raw['alternatives'][p['source_alternative']]]),case,h)[0]
                    actual=deepcopy(p);actual.pop('source_alternative')
                    self.assertEqual(actual,old)
            self.assertGreater(seen,0);self.assertGreater(accepted,0,s['id'])

    def test_semantic_repair_can_replace_invalid_candidate_without_renaming_observations(self):
        bad=programs_reply();bad['alternatives'][0]['groups'][0]['new_objects'][0]['key']='obs_000'
        p=Replies([bad,programs_reply()]);case=dict(self.case,context=[dict(id='obs_000')])
        ask=runner.asker(p,dict(seed=3407),self.base/'calls')
        result=ask('programs_h0',None,'Complete hidden delta',lambda v:model.programs_answer(v,case,self.h))
        self.assertEqual(len(result['valid']),1);self.assertEqual(len(p.prompts),2)
        self.assertIn('Candidate identities are NOT frozen',p.prompts[1])
        self.assertIn('Never redeclare or rename observed objects',p.prompts[1])
        self.assertTrue((self.base/'calls/programs_h0.validation.json').exists())

    def test_numeric_normalization_is_lossless_named_and_audited(self):
        raw=dict(id='001',description='3.5 metres',full_size_m=['1.25','2','3e-1'],
            yaw_deg='15',host_offset_uv=['0.2','-.5'],attachment=dict(host='123'),floor_y_m='-1.5m')
        v,log=numeric_fields(raw)
        self.assertEqual(v['full_size_m'],[1.25,2.,.3]);self.assertEqual(v['id'],'001')
        self.assertEqual(v['attachment']['host'],'123');self.assertEqual(v['floor_y_m'],'-1.5m')
        self.assertEqual(v['host_offset_uv'][1],'-.5');self.assertEqual(len(log),5)
        self.assertIsInstance(raw['yaw_deg'],str)
        p=Replies([dict(yaw_deg='15')]);ask=runner.asker(p,dict(seed=3407),self.base/'metric_calls')
        self.assertEqual(ask('world_01_poses_0',None,'Pose',lambda x:x),dict(yaw_deg=15.))
        saved=read(self.base/'metric_calls/world_01_poses_0.accepted.json')
        self.assertEqual(len(saved['numeric_normalizations']),1);self.assertFalse(saved['semantic_identity_changes'])

    def test_size_repair_can_fix_size_then_pose_repair_freezes_it(self):
        def validate(v):
            if v.get('ok') is not True:raise ValueError('Malformed numeric fixture')
            return v
        for name in ('world_01_sizes_0','world_01_poses_0'):
            p=Replies([dict(ok=False),dict(ok=True)]);ask=runner.asker(p,dict(seed=3407),self.base/name)
            ask(name,None,'Requested fields',validate)
            self.assertIn('exact IDs and hosts are frozen',p.prompts[1])
            self.assertIn('Dimensions are being inferred' if '_sizes_' in name else 'Previously accepted dimensions are frozen',p.prompts[1])

    def test_valid_sibling_and_failed_sibling_remain_distinct_SCM_states(self):
        raw=programs_reply();bad=deepcopy(raw['alternatives'][0]);bad['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
        raw['alternatives'].append(bad)
        p=model.programs_answer(raw,self.case,self.h)
        self.assertEqual(len(p['valid']),1);self.assertEqual(len(p['failed']),1)
        picture=self.base/'image.png';Image.new('RGB',(32,32)).save(picture)
        case=dict(self.case,image=str(picture))
        def ask(name,image,prompt,validate,max_tokens=8192):return validate(hypotheses() if name=='hypotheses' else raw)
        runner.propose(case,dict(objects=[]),self.base,{},ask)
        d=read(self.base/'domain.json');self.assertTrue(model.has_valid_program(d))
        self.assertTrue(d['programs']['f_h0_failed_1']['proposal_failed'])
        states=list(d['programs']);tables=dict(recognition=model.predictive(['h0'],['h0']),
            programs={'h0':model.predictive(['f_h0_0'],states)})
        scm=model.compile_model(d['hypotheses'],d['programs'],tables)
        f=next(n for n in scm['nodes'] if n['id']=='F')
        self.assertGreater(f['cpt']['h0'][f['states'].index('f_h0_failed_1')],0.)

    def test_floor_is_explicit_and_anchor_bottoms_are_computed(self):
        observed=dict(objects=[dict(track_id='obs_000',center_local_m=dict(value=[0.,-.5,2.]),size_aabb_local_m=dict(value=[1.,2.,1.]))])
        plane=geometry.floor_answer(dict(anchor_ids=['obs_000'],floor_y_m=None,basis='Assumed floor furniture.'),observed)
        self.assertEqual(plane['floor_y_m'],-1.5);self.assertFalse(plane['floor_measured'])
        for v in (dict(anchor_ids=['unknown'],floor_y_m=None,basis='bad'),dict(anchor_ids=['obs_000'],floor_y_m=-1.,basis='bad'),
            dict(anchor_ids=[],floor_y_m=0.,basis='bad'),dict(anchor_ids=[],floor_y_m=True,basis='bad')):
            with self.assertRaises(ValueError):geometry.floor_answer(v,observed)
        unresolved=geometry.floor_answer(dict(anchor_ids=[],floor_y_m=None,basis='Not inferable.'),observed)
        self.assertIsNone(unresolved['floor_y_m'])

    def test_failed_domains_never_initialize_provider_and_report_has_working_observed_view(self):
        source=self.base/'source';source.mkdir();suite=read(SOURCE/'suite.json');suite['scenes']=suite['scenes'][:1]
        write(source/'suite.json',suite);sid=suite['scenes'][0]['id']
        shutil.copytree(SOURCE/'scenes'/sid/'observation',source/'scenes'/sid/'observation')
        root=self.base/'run'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            runner.prepare(root,source,worlds=2,draws=4)
        comp=root/'scenes'/sid/'joint_completion';comp.mkdir()
        write(comp/'domain.json',dict(hypotheses=[self.h],programs={'f_h0':model.failed_program(self.h,'invalid proposal')}))
        with patch.object(runner,'provider',side_effect=AssertionError('Blocked domain must not load a model')):
            for phase in ('mechanisms','worlds'):
                r=runner.phase(root,phase);self.assertEqual(r['records'][0]['status'],'blocked_proposal_domain')
                self.assertFalse(r['records'][0]['model_called'])
        self.assertFalse((comp/'causal_model.json').exists());self.assertFalse((comp/'structural_samples.json').exists())
        r=runner.report(root);self.assertEqual(r['counts'],dict(blocked_proposal_domain=2))
        self.assertEqual(read(comp/'observed_views/visual_validation.json')['labels'],16)
        page=(root/'index.html').read_text();self.assertIn('不是隐藏补全',page)
        for target in re.findall(r'(?:href|src)="([^"]+)"',page):self.assertTrue((root/target).is_file(),target)


if __name__=='__main__':unittest.main()
