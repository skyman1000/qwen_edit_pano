"""Actual failure replays and CPU fixtures; no real model/API generation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import ROOT,read,write,sha
from . import assembly_io_recovery as io,assembly_multiscene as suite
from .assembly_multiscene_fork import fork_suite,verify_recovery,json_hashes
from .test_multiscene import CPUProvider,create

ACTUAL=ROOT/'qwen_edit_pano/outputs/organized_multiscene_20261027_203624_2435093'
BATH='x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6'
LOUNGE='8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39'


def obj(i):return dict(name='chair '+str(i),category='chair',bbox_1000=[i*10,100,i*10+8,900],on_floor=True)


class IORecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)

    def observation(self):
        root=self.base/'observation';(root/'qwen').mkdir(parents=True);Image.new('RGB',(320,320),'white').save(root/'input.png')
        config=read(ACTUAL/'scenes'/LOUNGE/'observation/experiment.json');write(root/'experiment.json',config);return root

    def test_real_initial_failures_are_repetition_and_null_template_not_zero_inventory(self):
        d=ACTUAL/'scenes'/LOUNGE/'observation/qwen/observer2d'
        self.assertEqual(read(d/'answer.generation.json')['generated_tokens'],8192);self.assertEqual(read(d/'answer.generation.json')['finish_reason'],'length')
        raw=(d/'answer.txt').read_text();self.assertGreater(raw.count('glass vase'),30);self.assertEqual(raw,(d/'answer_repair.txt').read_text())
        c=ACTUAL/'scenes'/BATH/'completion/end_to_end/qwen/completion'
        w=read(c/'world_01/semantic_world.json');self.assertEqual(len(w['objects']),13)
        initial=read(c/'calls/layout_01_placement.json');self.assertTrue(all(o['yaw_deg'] is None for g in initial['groups'] for o in g['objects']))

    def test_chunk_reply_validates_done_duplicates_and_budget(self):
        value=dict(objects=[obj(1)],evidence_description='visible chair',done=True)
        self.assertEqual(io.chunk_answer(value,[])['objects'],[obj(1)])
        self.assertEqual(len(io.chunk_answer(dict(value,objects=[obj(i) for i in range(9)]),[])['objects']),9)
        for bad in (dict(value,done=None),dict(value,objects=[],done=False),dict(value,objects=[obj(i) for i in range(65)])):
            with self.assertRaises(ValueError):io.chunk_answer(bad,[])
        with self.assertRaises(ValueError):io.chunk_answer(value,[obj(1)])

    def test_truncated_reply_is_not_pasted_into_retry_and_all_chunks_are_kept(self):
        root=self.observation();requests=[]
        raw=(ACTUAL/'scenes'/LOUNGE/'observation/qwen/observer2d/answer.txt').read_text()
        class Provider:
            def ask(inner,image,prompt,path,**kwargs):
                requests.append(prompt)
                if len(requests)==1:path.write_text(raw);raise ValueError('Unfinished local response reached token limit')
                value=dict(objects=[obj(i) for i in range(8)] if len(requests)==2 else [obj(8)],evidence_description='CPU visible objects',done=len(requests)==3)
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        result=io.observe_chunked(Provider(),root);self.assertEqual(result['objects'],9);self.assertEqual(result['actual_calls'],3)
        self.assertNotIn('glass vase',requests[1]);self.assertLess(len(requests[1]),10000)
        self.assertEqual(len(read(root/'qwen/observer2d/G_2d.json')['objects']),9)

    def test_unfinished_budget_does_not_export_partial_observation(self):
        root=self.observation()
        class P:
            def __init__(inner):inner.n=0
            def ask(inner,image,prompt,path,**kwargs):
                inner.n+=1;v=dict(objects=[obj(inner.n)],evidence_description='CPU incomplete',done=False);path.write_text(json.dumps(v));return v
        with self.assertRaisesRegex(ValueError,'eight chunks'):io.observe_chunked(P(),root)
        self.assertFalse((root/'qwen/observer2d/G_2d.json').exists());self.assertTrue((root/'qwen/observer2d/accepted_chunks.json').exists())

    def test_real_kitchen_duplicate_then_empty_cannot_erase_visible_inventory(self):
        root=self.observation();source=ACTUAL.parent/'organized_multiscene_recovery_20261028_094728_566800/scenes/8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029/observation/qwen/observer2d'
        replies=[read(source/'chunk_00_0.json'),read(source/'chunk_00_1.json'),read(source/'chunk_00_1.json')]
        class P:
            def ask(inner,image,prompt,path,**kwargs):
                value=replies.pop(0);path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        with self.assertRaisesRegex(ValueError,'Contradictory empty inventory'):io.observe_chunked(P(),root)
        self.assertFalse((root/'qwen/observer2d/G_2d.json').exists());self.assertEqual(read(root/'qwen/observer2d/summary.json')['status'],'failed')

    def test_genuinely_empty_first_reply_is_not_forced_to_invent_objects(self):
        root=self.observation()
        class P:
            def ask(inner,image,prompt,path,**kwargs):
                value=dict(objects=[],evidence_description='CPU fixture without identifiable object instances',done=True);path.write_text(json.dumps(value));return value
        self.assertEqual(io.observe_chunked(P(),root)['objects'],0)

    def bath_inputs(self):
        c=ACTUAL/'scenes'/BATH/'completion/end_to_end/qwen/completion';d=c/'world_01'
        return read(c/'case.json'),read(c/'G_obs.json'),read(d/'semantic_world.json'),read(c/'support_plane.json'),read(d/'dimension_stage.json')['dimensions']

    def test_compact_rows_keep_all_thirteen_objects_sizes_regions_and_hosts(self):
        case,obs,world,plane,dims=self.bath_inputs();before=deepcopy((world,dims));p=CPUProvider();dest=self.base/'numeric';dest.mkdir();prompts=[]
        def ask(name,image,prompt,validator):
            prompts.append(prompt);return validator(p.ask(image,prompt,dest/(name+'.txt')))
        layout=io.placement(case,obs,world,plane,dims,None,ask,dest,1)
        self.assertEqual(sum(len(g['objects']) for g in layout.values()),13);self.assertEqual((world,dims),before)
        self.assertEqual(len(prompts),3);self.assertTrue(all('ROW_COLUMNS_BY_ID' in p and '\nTEMPLATE:' not in p for p in prompts))
        for g in layout.values():
            for o in g['objects']:self.assertEqual(o['intrinsic_size_m'],dims[o['id']]['full_size_m'])

    def test_compact_rows_reject_nulls_unknown_ids_and_invalid_surface_values(self):
        case,obs,w,plane,d=self.bath_inputs();g=w['groups'][0];members=[o for o in w['objects'] if o['group']==g['key']]
        rows=[]
        for o in members:
            rows.append([o['id']]+[.5 if c=='surface_fraction' else .1 if c=='hanging_drop_m' else 0. for c in io.columns(o)[1:]])
        value=dict(group_id=g['key'],range_m=3.,placements=rows);io.group_answer(value,g,w,obs,d,case['hfov'])
        for bad in (dict(value,range_m=True),dict(value,placements=[]),dict(value,group_id='bad')):
            with self.assertRaises(ValueError):io.group_answer(bad,g,w,obs,d,case['hfov'])
        bad=deepcopy(value);bad['placements'][0][1]=None
        with self.assertRaises(ValueError):io.group_answer(bad,g,w,obs,d,case['hfov'])
        bad=deepcopy(value);bad['placements'][0][0]='other_group_object'
        with self.assertRaises(ValueError):io.group_answer(bad,g,w,obs,d,case['hfov'])

    def test_dependency_failure_never_initializes_model_or_fakes_an_empty_scene(self):
        root=self.base/'suite';(root/'phase_records').mkdir(parents=True);scene=root/'scenes/missing';scene.mkdir(parents=True)
        with patch.object(suite,'verify_suite',return_value=dict(scenes=[dict(id='missing')])),patch.object(suite,'provider',side_effect=AssertionError('No model for blocked stages')):
            result=suite.run_phase(root,'mechanisms','api')
        row=result['scenes'][0];self.assertEqual(row['status'],'blocked_dependency');self.assertFalse(row['model_called']);self.assertNotIn('traceback',row)
        self.assertFalse((scene/'completion').exists())

    def test_fork_preserves_successes_and_recovers_only_coordinates_with_fixed_SCM(self):
        root=self.base/'fork';before=json_hashes(ACTUAL)
        with patch.dict(os.environ,CAUPANO_PROGRAM_CATALOG=''),patch.object(suite,'new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.new_output',side_effect=create),\
            patch('qwen_edit_pano.world_pipeline.completion.assembly_experiment.preflight',return_value={}):
            plan=fork_suite(root,ACTUAL,suite.DEFAULT_INPUTS,numeric_recovery=True)
        self.assertEqual(plan['inputs'],5);self.assertEqual(plan['reused_complete_worlds'],3);self.assertEqual(len(plan['new_scenes']),2)
        self.assertFalse((root/'scenes'/LOUNGE/'observation/qwen/observer2d').exists())
        self.assertTrue((root/'scenes'/LOUNGE/'recovery_history/observer2d_original_failure/answer.txt').exists())
        suite.verify_suite(root);m=verify_recovery(root)
        p=CPUProvider();comp=root/'scenes'/BATH/'completion';result=io.recover_layouts(comp,'end_to_end','qwen',p)
        self.assertEqual(len(p.requests),3)
        dest=comp/'end_to_end/qwen/completion/world_01';state=read(dest/('G_full.json' if (dest/'G_full.json').exists() else 'G_full_candidates.json'))
        self.assertEqual(len(state['objects'])-len(read(comp/'end_to_end/qwen/completion/G_obs.json')['objects']),13)
        verify_recovery(root);self.assertEqual(before,json_hashes(ACTUAL))
        # Partial valid proposals are warnings; they do not trigger fresh draws.
        with patch.object(suite,'verify_suite',return_value=dict(scenes=[dict(id=BATH)])),patch.object(suite,'provider',side_effect=AssertionError('No re-proposal')):
            status=suite.run_phase(root,'propose','api')['status']
        self.assertEqual(status,'complete_with_warnings')
        key=next(k for k in m['protected_copy_json_sha256'] if k.endswith('/causal_model.json'));(root/key).write_text('{}')
        with self.assertRaises(ValueError):verify_recovery(root)


if __name__=='__main__':unittest.main()
