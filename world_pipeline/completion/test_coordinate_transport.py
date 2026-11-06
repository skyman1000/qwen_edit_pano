"""Real Qwen envelope failure replays; mocked generation only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import assembly_coordinate_transport as transport,assembly_multiscene as suite
from .assembly_multiscene_fork import fork_suite,verify_recovery,json_hashes
from .test_multiscene import CPUProvider,create

ACTUAL=ROOT/'qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001'
COUNTS={
    '8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029':(33,14),
    '8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':(28,18),
    'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6':(13,8),
    'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a':(23,12)}


class CoordinateTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)

    def inputs(self,sid=None):
        sid=sid or next(iter(COUNTS));c=ACTUAL/'scenes'/sid/'completion/end_to_end/qwen/completion';d=c/'world_01'
        return c,d,read(c/'case.json'),read(c/'G_obs.json'),read(d/'semantic_world.json'),read(c/'support_plane.json'),read(d/'dimension_stage.json')['dimensions']

    def test_actual_group_wrapper_ID_dictionary_and_duplicate_wrong_frame_are_lossless(self):
        for sid,(count,kept) in COUNTS.items():
            c,d,case,obs,w,plane,dims=self.inputs(sid);original=deepcopy(w);rows,ranges,sources,bad,audit=transport.capture(c,d,w,1)
            self.assertEqual(len(w['objects']),count);self.assertEqual(len(rows),kept);self.assertEqual(w,original)
            for oid,row in rows.items():
                if 'named_recovery_calls_v2' not in sources[oid]:continue
                raw=read(Path(sources[oid]));g=next(g for g in w['groups'] if next(o for o in w['objects'] if o['id']==oid)['group']==g['key'])
                decoded=transport.response(raw,g,[o for o in w['objects'] if o['group']==g['key']]);self.assertEqual(decoded['rows'][oid],row)
            self.assertTrue(audit)

    def test_conflicting_ID_or_group_is_never_rebound_and_numbers_are_never_fabricated(self):
        c,d,case,obs,w,plane,dims=self.inputs();g=w['groups'][0];members=[o for o in w['objects'] if o['group']==g['key']];o=next(o for o in members if not o['attachment']['host'])
        oid=o['id'];row=dict(id=oid,yaw_deg=0.,root_offset_m=[0.,0.,0.]);value=dict(group_id=g['key'],objects=[row])
        self.assertEqual(transport.response(value,g,members)['rows'][oid],row)
        with self.assertRaises(ValueError):transport.response(dict(value,group_id='other'),g,members)
        with self.assertRaises(ValueError):transport.response({oid:dict(row,id='other')},g,members)
        with self.assertRaises(ValueError):transport.response({'unknown':row},g,members)
        for field,bad in [('yaw_deg',None),('root_offset_m',[0,0]),('root_offset_m',[0,False,0])]:
            invalid=dict(row,**{field:bad});self.assertNotIn(oid,transport.response(dict(value,objects=[invalid]),g,members)['rows'])
        missing=dict(row);missing.pop('yaw_deg');self.assertNotIn(oid,transport.response(dict(value,objects=[missing]),g,members)['rows'])
        conflict=dict(row,yaw_deg=90.)
        self.assertNotIn(oid,transport.response(dict(value,objects=[row,conflict]),g,members)['rows'])

    def test_partial_reply_retains_good_rows_before_requesting_only_missing_fields(self):
        c,d,case,obs,w,plane,dims=self.inputs();g=w['groups'][0];w=dict(w,groups=[g],objects=[o for o in w['objects'] if o['group']==g['key']])
        out=self.base/'partial';out.mkdir();p=CPUProvider();requests=[];kept={}
        def ask(name,image,prompt,validator):
            request=json.loads(prompt.split('\nREQUEST: ')[1].split('\n')[0]);requests.append(request)
            value=p.ask(image,prompt,out/(name+'.txt'))
            if len(requests)==1:
                first=value['objects'][0]['id'];kept.update({o['id']:deepcopy(o) for o in value['objects'][1:]});value['objects']=value['objects'][1:]
            return validator(value)
        layout=transport.placement(case,obs,w,plane,dims,None,ask,out,1)
        self.assertEqual(len(requests[1]['object_ids']),1);self.assertFalse(set(requests[1]['object_ids'])&set(kept))
        self.assertTrue(all(len(r['object_ids'])<=6 for r in requests))
        raw=read(out/'coordinate_stage_v3.json');rows={o['id']:o for g in raw['raw_placement']['groups'] for o in g['objects']}
        for oid,row in kept.items():self.assertEqual(rows[oid],row)
        self.assertEqual(len(rows),len(w['objects']))

    def test_invalid_range_is_requested_alone_without_replacing_accepted_numbers(self):
        c,d,case,obs,w,plane,dims=self.inputs();g=w['groups'][0];w=dict(w,groups=[g],objects=[o for o in w['objects'] if o['group']==g['key']][:1])
        out=self.base/'range';out.mkdir();p=CPUProvider();requests=[]
        def ask(name,image,prompt,validator):
            request=json.loads(prompt.split('\nREQUEST: ')[1].split('\n')[0]);requests.append(request);value=p.ask(image,prompt,out/(name+'.txt'))
            if len(requests)==1:value.pop('range_m')
            return validator(value)
        transport.placement(case,obs,w,plane,dims,None,ask,out,1)
        self.assertEqual(requests[1]['object_ids'],[]);self.assertTrue(requests[1]['range_required'])

    def test_real_four_scene_recovery_preserves_all_six_successes_and_has_no_API_dependency(self):
        root=self.base/'fork';before=json_hashes(ACTUAL)
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch.object(suite,'new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):plan=fork_suite(root,ACTUAL,suite.DEFAULT_INPUTS,numeric_recovery=True)
        self.assertEqual(plan['reused_complete_worlds'],6);self.assertEqual(plan['new_scenes'],[])
        ready=suite.coordinate_plan(root);self.assertEqual(ready['api_calls_required'],0);self.assertEqual(len(ready['pending_Qwen_coordinate_worlds']),4)
        all_results=[]
        for sid,(count,kept) in COUNTS.items():
            comp=root/'scenes'/sid/'completion';c=comp/'end_to_end/qwen/completion';d=c/'world_01';w=read(d/'semantic_world.json');saved=transport.capture(c,d,w,1)
            p=CPUProvider();result=transport.recover_layouts(comp,'end_to_end','qwen',p);outcome=result['coordinate_transport_recovery'][0]
            self.assertIn(outcome['status'],('exported','blocked_numeric'),outcome)
            stage=read(d/'coordinate_stage_v3.json');rows={o['id']:o for g in stage['raw_placement']['groups'] for o in g['objects']}
            self.assertEqual(len(rows),count);self.assertEqual(len(stage['initially_reused_ids']),kept)
            for oid,row in saved[0].items():self.assertEqual(rows[oid],row)
            for request in p.requests:
                wanted=json.loads(request['prompt'].split('\nREQUEST: ')[1].split('\n')[0])['object_ids'];self.assertFalse(set(wanted)&set(saved[0]))
            state=read(d/('G_full.json' if outcome['status']=='exported' else 'G_full_candidates.json'))
            self.assertEqual(len(state['objects'])-len(read(c/'G_obs.json')['objects']),count)
            all_results.append(dict(scene=sid,outcome=outcome['status'],hidden=count,kept=kept,mocked_requests=len(p.requests)))
        m=verify_recovery(root);self.assertEqual(json_hashes(ACTUAL),before)
        for p in m['reused_worlds']:self.assertEqual(sha(root/p),sha(ACTUAL/p))
        # Completed suites may be inspected again without model calls. A
        # genuinely blocked world must still fail the coordinate-only gate.
        if all(r['outcome']=='exported' for r in all_results):
            ready=suite.coordinate_plan(root);self.assertEqual(ready['pending_Qwen_coordinate_worlds'],[]);self.assertEqual(ready['api_calls_required'],0)
        else:
            with self.assertRaises(ValueError):suite.coordinate_plan(root)
        print('CPU_V3_REPLAY_RESULTS '+json.dumps(all_results),flush=True)


if __name__=='__main__':unittest.main()
