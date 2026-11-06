"""Actual response regressions and retained-local-catalog integration."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ...common import ROOT,read,sha
from .. import assembly_multiscene as pipeline
from ..candidate_repair import contract as old_contract
from ..candidate_repair.test_candidates import ProposalCPU
from ..simple_world.test_simple import SOURCE,SID,fixture
from ..test_multiscene import CPUProvider
from . import workflow as w,contract as c
from .requests import PatchRequests

LATEST=ROOT/'qwen_edit_pano/outputs/simple_world_scope_001'


class StableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'research/audits',prefix='stable_CPU_')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.case=read(LATEST/'scenes'/SID/'completion/shared/case.json')
        self.h=dict(id='z0',description='Possible unseen work and storage',functions=['work','storage'],assumptions=[])

    def test_six_real_overlong_replies_report_real_constraints_not_prose_length(self):
        total=0;accepted=[]
        for err in LATEST.glob('scenes/*/completion/proposals/qwen/calls/*attempt0.error.json'):
            if 'group.reason requires concrete concise text' not in read(err)['error']:continue
            case=read(err.parents[3]/'shared/case.json')
            h=read(err.parent/'surrounding_scope_attempt0.json')['hypotheses'][0];h['id']='z0'
            aid=err.stem.split('_')[1];area=next(a for a in h['hidden_areas'] if a['id']==aid)
            raw=read(err.with_name(err.name.replace('.error.json','.json')))
            g=dict(function=area['function'],reason=raw['reason'],observed_refs=area['anchor_refs'],
                layout_region=area['region'],new_objects=raw['new_objects'],functional_links=raw.get('functional_links',[]))
            wire=dict(alternatives=[dict(description=h['description'],groups=[g],relations=raw.get('relations',[]))])
            with self.assertRaisesRegex(ValueError,'group.reason requires'):old_contract.furniture_answer(wire,case,h)
            try:program=c.furniture_answer(wire,case,h)[0]
            except ValueError as exc:self.assertNotIn('1..320',str(exc))
            else:
                accepted.append(err)
                self.assertTrue(all(group['reason']==raw['reason'] for group in program['groups']))
                after={o['key']:o for group in program['groups'] for o in group['objects']}
                for obj in raw['new_objects']:
                    self.assertEqual(after[obj['key']]['placement_hint'],obj['placement'])
                    self.assertEqual(after[obj['key']]['attachment'],obj['attachment'])
            total+=1
        self.assertEqual(total,6);self.assertEqual(len(accepted),3)

    def test_field_patch_cannot_erase_rename_or_reclassify_inventory(self):
        raw=fixture()
        for path,value in [('/alternatives',[]),('/alternatives/0/groups/0/new_objects',[]),
            ('/alternatives/0/groups/0/new_objects/0/key','changed'),
            ('/alternatives/0/groups/0/new_objects/0/category','changed')]:
            with self.assertRaisesRegex(ValueError,'Non-editable'):
                c.apply_patch(raw,dict(edits=[dict(path=path,value=value,reason='Attempted semantic redraw')]))
        with self.assertRaises(ValueError):c.apply_patch(raw,dict(alternatives=[]))

    def test_long_supported_containment_keeps_semantic_link_and_entire_reason(self):
        raw=fixture();group=raw['alternatives'][0]['groups'][1]
        group['new_objects'][1]['attachment']=dict(kind='shelf',host=group['new_objects'][0]['key'])
        group['new_objects'][0]['affordances']=['shelf']
        reason='Explicitly assumed interior shelf storage, not adjacency. '*12
        raw['alternatives'][0]['relations']=[dict(source=group['new_objects'][1]['key'],target=group['new_objects'][0]['key'],kind='inside',reason=reason)]
        p=c.furniture_answer(raw,self.case,self.h)[0]
        links=[r for g in p['groups'] for r in g.get('functional_links',[]) if r['kind']=='contained_in']
        self.assertEqual(len(links),1);self.assertEqual(links[0]['purpose'],reason)

    def test_typed_patch_changes_only_declared_field_and_preserves_other_objects(self):
        raw=fixture();raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
        fixed=c.apply_patch(raw,dict(edits=[dict(path='/alternatives/0/groups/0/new_objects/1/attachment/host',value='desk',reason='Use actual desk host')]))
        reference=deepcopy(raw);reference['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='desk'
        self.assertEqual(fixed,reference);self.assertNotEqual(raw,fixed)
        self.assertTrue(c.furniture_answer(fixed,self.case,self.h))

    def test_unknown_hosts_and_view_conflicts_remain_failures(self):
        raw=fixture();raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
        with self.assertRaisesRegex(ValueError,'unknown'):c.furniture_answer(raw,self.case,self.h)
        k=LATEST/'scenes/8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029/completion'
        h=read(k/'proposals/qwen/calls/surrounding_scope_attempt0.json')['hypotheses'][0];h['id']='z0'
        raw=read(k/'proposals/qwen/calls/z0_a1_attempt0.json')
        wire=dict(alternatives=[dict(description=h['description'],groups=[dict(function='hygiene',reason=raw['reason'],new_objects=raw['new_objects'])],relations=[])])
        with self.assertRaisesRegex(ValueError,'incompatible_observed_support'):c.furniture_answer(wire,read(k/'shared/case.json'),h)

    def test_patch_journal_replay_keeps_first_reply_and_uses_no_model(self):
        raw=fixture();raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
        class Provider:
            def __init__(inner):inner.calls=0
            def ask(inner,image,prompt,path,**kw):
                inner.calls+=1
                if inner.calls==1:value=raw
                else:
                    self.assertIn('COMPLETE_ORIGINAL_PROPOSAL',prompt)
                    value=dict(edits=[dict(path='/alternatives/0/groups/0/new_objects/1/attachment/host',value='desk',reason='Use the proposed host desk')])
                path.write_text(json.dumps(value));w.save(path.with_suffix('.json'),value);return value
        provider=Provider();request=PatchRequests(self.root/'calls',7,LATEST/'scenes'/SID/'completion/shared/observation_board.png',lambda:provider)
        first=request.ask('inventory','Task',lambda v:c.furniture_answer(v,self.case,self.h))
        self.assertEqual(provider.calls,2);self.assertEqual(read(self.root/'calls/inventory_attempt0.json'),raw)
        again=request.ask('inventory','Task',lambda v:c.furniture_answer(v,self.case,self.h))
        self.assertEqual(first,again);self.assertEqual(provider.calls,2)

    def test_retained_local_catalog_all_five_scenes_revalidates(self):
        suite,_=w.cached.check(w.BASELINE);count=0
        for scene in suite['scenes']:
            comp=w.BASELINE/'scenes'/scene['id']/'completion'
            for p in read(comp/'proposals/qwen/programs.json').values():
                clean=w.original.revalidate_program(p,read(comp/'shared/case.json'))
                self.assertEqual({o['key'] for g in clean['groups'] for o in g['objects']},{o['key'] for g in p['groups'] for o in g['objects']})
                count+=1
        self.assertGreater(count,0)

    def prepared(self,baseline=w.BASELINE):
        source=self.root/'source';source.mkdir();suite=read(SOURCE/'suite.json');suite['scenes']=[s for s in suite['scenes'] if s['id']==SID]
        w.save(source/'suite.json',suite);shutil.copytree(SOURCE/'scenes'/SID/'observation',source/'scenes'/SID/'observation')
        root=self.root/'run';w.prepare(root,source,worlds=2,draws=4,baseline=baseline);w.phase(root,'prepare-scenes');return root

    def test_new_scene_mode_needs_no_historical_catalog_or_API(self):
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            root=self.prepared(baseline=None);proposer=ProposalCPU()
            with patch.object(w,'provider',return_value=proposer):w.phase(root,'propose')
            summary=read(root/'scenes'/SID/'completion/proposals/qwen/summary.json')
            self.assertEqual(summary['retained_programs'],0);self.assertEqual(len(proposer.requests),6)
            self.assertGreater(summary['fresh_nonempty_programs'],0);self.assertFalse(summary['API_used'])
            self.assertFalse(read(root/'candidate_repair.json')['baseline_statistics'][SID]['comparable'])

    def test_no_catalog_and_explicit_empty_requests_remain_incomplete(self):
        class Empty(ProposalCPU):
            def ask(inner,image,prompt,path,**kwargs):
                if not Path(path).stem.startswith('furniture_'):return super().ask(image,prompt,path,**kwargs)
                value=dict(alternatives=[dict(description='Possible open space hypothesis',groups=[],relations=[])])
                Path(path).write_text(json.dumps(value));w.save(Path(path).with_suffix('.json'),value);return value
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            root=self.prepared(baseline=None)
            with patch.object(w,'provider',return_value=Empty()):w.phase(root,'propose')
            for phase in ('pool','mechanisms','worlds'):w.phase(root,phase)
            result=w.report(root);self.assertEqual(result['counts'],{'empty_candidate_domain':2})
            self.assertFalse(result['execution_errors']);self.assertEqual(result['status'],'incomplete_worlds')

    def test_full_original_v13_pipeline_retains_catalog_and_renders(self):
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            root=self.prepared();proposer=ProposalCPU();numeric=CPUProvider(all_present=True)
            self.assertIn(str(w.BASELINE),next(iter(read(root/'candidate_repair.json')['baseline_bindings'])))
            with patch.object(w,'provider',return_value=proposer),patch.object(pipeline,'provider',return_value=numeric):
                w.phase(root,'propose');w.phase(root,'pool');w.preview(root);w.phase(root,'mechanisms');w.phase(root,'worlds')
            baseline=read(w.BASELINE/'scenes'/SID/'completion/proposals/qwen/programs.json')
            current=read(root/'scenes'/SID/'completion/proposals/qwen/programs.json')
            self.assertTrue(all(current[k]==v for k,v in baseline.items()))
            result=w.report(root);self.assertFalse(result['execution_errors'],result)
            self.assertTrue(all(r['observed_preserved'] for r in result['records']))
            self.assertEqual(result['counts'],{'exported_with_new_objects':2})
            self.assertTrue((root/'report'/SID/'world_01/views/scene_3d.html').exists())
            self.assertGreater(len(numeric.requests),0)
            with patch.object(w,'provider',side_effect=AssertionError('Resume called model')):
                w.phase(root,'propose');again=w.report(root)
            self.assertEqual(result['counts'],again['counts'])

    def test_failed_fresh_proposal_does_not_remove_retained_catalog(self):
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            root=self.prepared();comp=root/'scenes'/SID/'completion'
            class RuntimeFailure:
                def ask(self,*a,**kw):raise RuntimeError('Explicit synthetic transport failure')
            result=w.propose_scene(comp,w.BASELINE,lambda:RuntimeFailure())
            self.assertTrue(result['execution_failures']);self.assertEqual(result['fresh_programs'],0)
            self.assertEqual(read(comp/'proposals/qwen/programs.json'),read(w.BASELINE/'scenes'/SID/'completion/proposals/qwen/programs.json'))

    def test_rejected_new_proposals_are_warnings_when_retained_domain_is_valid(self):
        class BadProposal(ProposalCPU):
            def ask(inner,image,prompt,path,**kwargs):
                if not Path(path).stem.startswith('furniture_'):return super().ask(image,prompt,path,**kwargs)
                raw=fixture();raw['alternatives'][0]['groups'][0]['new_objects'][1]['attachment']['host']='unknown'
                value=raw if Path(path).stem.endswith('attempt0') else dict(alternatives=[])
                Path(path).write_text(json.dumps(value));w.save(Path(path).with_suffix('.json'),value);return value
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG='',CAUPANO_LAYOUT_RGB='0'),patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            root=self.prepared()
            with patch.object(w,'provider',return_value=BadProposal()):w.phase(root,'propose')
            w.phase(root,'pool');result=w.preview(root)
            self.assertTrue(result['proposal_rejections']);self.assertFalse(result['execution_errors'])
            self.assertEqual(result['status'],'candidate_domains_ready_with_warnings')


if __name__=='__main__':unittest.main()
