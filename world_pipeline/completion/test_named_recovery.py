"""Replay the four real failures and mock rejected transport requests on CPU."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from ..common import ROOT,read,write,sha
from . import assembly_named_geometry as numeric,assembly_multiscene as suite
from .assembly_api_pacing import PacedAPI,RateGate,retry_after
from .assembly_multiscene_fork import fork_suite,verify_recovery,json_hashes
from .test_multiscene import CPUProvider,create
from .world_api import SiliconFlowJSON,APIRequestError

ACTUAL=ROOT/'qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903'
COUNTS={
    '8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029':(33,7),
    '8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':(28,3),
    'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6':(13,7),
    'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a':(23,9)}


class NamedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)

    def test_real_failures_preserve_valid_rows_and_refuse_ambiguous_extra_columns(self):
        for sid,(count,kept) in COUNTS.items():
            c=ACTUAL/'scenes'/sid/'completion/end_to_end/qwen/completion';d=c/'world_01';world=read(d/'semantic_world.json')
            rows,ranges,sources,rejected=numeric.capture_legacy(c,d,world,1)
            self.assertEqual(len(world['objects']),count);self.assertEqual(len(rows),kept);self.assertTrue(rejected)
            self.assertTrue(all(ranges.values()));self.assertEqual(set(rows),{k for k in sources if not k.startswith('range:')})
            # Wrong arity is a distinct error, rather than a false nonfinite claim.
            self.assertTrue(any('columns' in r['error'] for r in rejected))
            for obj in world['objects']:
                if obj['id'] in rows:self.assertEqual(numeric.row_answer(rows[obj['id']],obj),rows[obj['id']])

    def test_named_partial_patch_preserves_good_rows_and_reasks_only_bad_coordinates(self):
        sid=next(iter(COUNTS));c=ACTUAL/'scenes'/sid/'completion/end_to_end/qwen/completion';d=c/'world_01'
        world=read(d/'semantic_world.json');g=world['groups'][0];world=dict(world,groups=[g],objects=[o for o in world['objects'] if o['group']==g['key']])
        case=read(c/'case.json');obs=read(c/'G_obs.json');plane=read(c/'support_plane.json');dims=read(d/'dimension_stage.json')['dimensions']
        out=self.base/'partial';out.mkdir();p=CPUProvider();requested=[];fixed=None
        def ask(name,image,prompt,validator):
            nonlocal fixed
            value=p.ask(image,prompt,out/(name+'.txt'));required=json.loads(prompt.split('\nREQUIRED_FIELDS_BY_ID: ')[1].split('\n')[0]);requested.append(set(required))
            if len(requested)==1:
                value['roots'][0]['root_offset_m']=[None,0.,0.];fixed=deepcopy(value['hosted'][0])
            return validator(value)
        layout=numeric.placement(case,obs,world,plane,dims,None,ask,out,1)
        self.assertEqual(len(requested),2);self.assertEqual(len(requested[1]),1)
        stage=read(out/'named_numeric_stage_v2.json');rows={o['id']:o for o in stage['raw_placement']['groups'][0]['objects']}
        self.assertEqual(rows[fixed['id']],fixed);self.assertEqual(len(rows),len(world['objects']))
        for g in layout.values():
            for o in g['objects']:self.assertEqual(o['intrinsic_size_m'],dims[o['id']]['full_size_m'])

    def test_all_four_real_recoveries_keep_SCM_inventory_sizes_valid_rows_and_source(self):
        before=json_hashes(ACTUAL);root=self.base/'fork'
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch.object(suite,'new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            plan=fork_suite(root,ACTUAL,suite.DEFAULT_INPUTS,numeric_recovery=True)
        self.assertEqual(plan['reused_complete_worlds'],3);self.assertEqual(plan['new_scenes'],[])
        for sid,(count,kept) in COUNTS.items():
            comp=root/'scenes'/sid/'completion';c=comp/'end_to_end/qwen/completion';d=c/'world_01';w=read(d/'semantic_world.json')
            initial=numeric.capture_legacy(c,d,w,1);p=CPUProvider()
            result=numeric.recover_layouts(comp,'end_to_end','qwen',p)
            outcome=result['named_coordinate_recovery'][0]
            self.assertIn(outcome['status'],('exported','blocked_numeric'),outcome)
            saved=read(d/'named_numeric_stage_v2.json');rows={o['id']:o for g in saved['raw_placement']['groups'] for o in g['objects']}
            self.assertEqual(len(rows),count);self.assertEqual(len(saved['initially_reused_ids']),kept)
            for oid,row in initial[0].items():self.assertEqual(rows[oid],row)
            for req in p.requests:
                missing=json.loads(req['prompt'].split('\nREQUIRED_FIELDS_BY_ID: ')[1].split('\n')[0])
                self.assertFalse(set(missing)&set(initial[0]))
            state=read(d/('G_full.json' if outcome['status']=='exported' else 'G_full_candidates.json'))
            self.assertEqual(len(state['objects'])-len(read(c/'G_obs.json')['objects']),count)
            verify_recovery(root)
        self.assertEqual(before,json_hashes(ACTUAL));suite.verify_suite(root)


class Response(io.BytesIO):
    def __init__(self,value):super().__init__(json.dumps(value).encode())


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.now=1000.;self.sleeps=[]
    def sleep(self,n):self.sleeps.append(n);self.now+=n
    def gate(self):return RateGate(self.base/'gate.json',interval=3.,clock=lambda:self.now,sleep=self.sleep)
    def api(self,opener,retries=3,wait_budget=300.):
        with patch.dict(os.environ,CPU_TEST_API_KEY='fixture-secret'):
            inner=SiliconFlowJSON(key_env='CPU_TEST_API_KEY',opener=opener,stream=False,max_calls=4,generation_profile='matched_v1')
        return PacedAPI(inner,gate=self.gate(),backoff=20.,retries=retries,wait_budget=wait_budget)
    @staticmethod
    def answer(raw='{"choice":"value_0"}'):
        return Response(dict(choices=[dict(finish_reason='stop',message=dict(content=raw))]))

    def test_429_retries_same_request_only_and_publishes_original_cache_paths(self):
        calls=[]
        def opener(request,**kw):
            calls.append(request.data)
            if len(calls)==1:raise urllib.error.HTTPError(request.full_url,429,'limited',{'Retry-After':'24'},None)
            return self.answer()
        p=self.api(opener);path=self.base/'recognition.txt';value=p.ask(None,'fixed prompt',path,max_tokens=256)
        self.assertEqual(value,dict(choice='value_0'));self.assertEqual(p.logical_calls,1);self.assertEqual(p.calls,2)
        self.assertEqual(calls[0],calls[1]);self.assertEqual(sum(self.sleeps),24.)
        self.assertEqual(read(path.with_suffix('.json')),value);self.assertTrue(path.with_suffix('.request.json').exists())
        self.assertTrue((self.base/'recognition_wire_calls/wire_00.transport_error.json').exists())
        self.assertNotIn('fixture-secret',''.join(f.read_text() for f in self.base.rglob('*') if f.is_file()))
        # Published raw file is accepted by the original mechanism cache.
        self.assertEqual(sha(path.with_suffix('.json')),sha(self.base/'recognition_wire_calls/wire_01.json'))

    def test_permanent_429_is_bounded_and_long_retry_after_is_respected(self):
        calls=[]
        def opener(request,**kw):calls.append(1);raise urllib.error.HTTPError(request.full_url,429,'limited',{},None)
        p=self.api(opener,retries=2)
        with self.assertRaises(APIRequestError):p.ask(None,'fixed',self.base/'limited.txt')
        self.assertEqual(len(calls),3);self.assertEqual(p.logical_calls,1);self.assertEqual(sum(self.sleeps),60.)
        def slow(request,**kw):raise urllib.error.HTTPError(request.full_url,429,'limited',{'Retry-After':'9999'},None)
        other=self.api(slow,wait_budget=90.)
        with self.assertRaisesRegex(APIRequestError,'rate_limit_wait_budget_exhausted'):other.ask(None,'fixed',self.base/'long.txt')
        self.assertEqual(other.calls,1)

    def test_timeouts_and_invalid_model_json_are_never_transport_retried(self):
        calls=[]
        def timeout(request,**kw):calls.append(1);raise TimeoutError('fixture')
        p=self.api(timeout)
        with self.assertRaises(APIRequestError):p.ask(None,'fixed',self.base/'timeout.txt')
        self.assertEqual(len(calls),1)
        calls.clear()
        def invalid(request,**kw):calls.append(1);return self.answer('bad JSON')
        p=self.api(invalid)
        with self.assertRaises(ValueError):p.ask(None,'fixed',self.base/'invalid.txt')
        self.assertEqual(len(calls),1);self.assertEqual((self.base/'invalid.txt').read_text(),'bad JSON')

    def test_shared_gate_and_logical_budget_and_retry_after_parsing(self):
        a=self.gate();b=self.gate();self.assertEqual(a.wait(),0.);self.assertEqual(b.wait(),3.)
        p=self.api(lambda *a,**k:self.answer());p.max_calls=1;p.ask(None,'fixed',self.base/'one.txt')
        with self.assertRaisesRegex(APIRequestError,'logical_request_budget_exhausted'):p.ask(None,'fixed',self.base/'two.txt')
        self.assertEqual(p.calls,1);self.assertFalse((self.base/'two_wire_calls').exists())
        self.assertEqual(retry_after('NaN'),None);self.assertEqual(retry_after('9'),9.)
        self.assertEqual(retry_after('Thu, 01 Jan 1970 00:00:20 GMT',now=10.),10.)


if __name__=='__main__':unittest.main()
