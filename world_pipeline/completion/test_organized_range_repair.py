"""CPU regressions using preserved real replies; synthetic ranges are not results."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ..common import ROOT,read,write,sha
from . import assembly_organized_geometry as geometry,assembly_organized_repair as repair
from .test_organized import toy_world
from .world_scm_runner import ask_valid

ACTUAL=ROOT/'qwen_edit_pano/outputs/organized_completion_20261027_152654_1370933'


def json_fixture(dest):
    def ignore(folder,names):
        return [n for n in names if not (Path(folder)/n).is_dir() and not n.endswith('.json') and n not in ('input.png','observation_board.png')]
    shutil.copytree(ACTUAL,dest,ignore=ignore)
    run=read(dest/'run.json');run['synthetic_fixture']=True;write(dest/'run.json',run)
    return dest


def new_fixture_output(path):
    if path.exists():raise FileExistsError(path)
    return path.resolve()


class RangeProvider:
    def __init__(self,invalid=False):self.requests=[];self.invalid=invalid
    def ask(self,image,prompt,path,**kwargs):
        if image is not None:raise AssertionError('The frozen v13 layout configuration is text-only')
        self.requests.append(dict(prompt=prompt,path=str(path),**kwargs))
        if '\nRANGE_TEMPLATE: ' not in prompt:raise AssertionError('Recovery must ask only for ranges')
        value=json.loads(prompt.split('\nRANGE_TEMPLATE: ')[1].split('\n')[0])
        for r in value['ranges']:r['range_m']=None if self.invalid else 3.0
        path=Path(path);path.write_text(json.dumps(value));write(path.with_suffix('.json'),value)
        return value


class RangeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=tempfile.TemporaryDirectory();cls.source=json_fixture(Path(cls.shared.name)/'source')
    @classmethod
    def tearDownClass(cls):cls.shared.cleanup()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'repair'
    def prepare(self):
        with patch.object(repair,'validate_new_output',side_effect=new_fixture_output):m=repair.prepare(self.root,self.source)
        return m

    def test_actual_six_initial_replies_have_only_nineteen_missing_ranges(self):
        m=self.prepare();self.assertEqual(len(m['pending']),6)
        self.assertEqual(sum(len(e['missing_group_ids']) for e in m['pending']),19)
        self.assertEqual(m['initial_calls_by_backend'],dict(qwen=6,api=0));self.assertEqual(m['completed_states_reused'],26)
        for e in m['pending']:
            _,_,case,obs,world,_,dims,raw=repair.inputs(self.root,e);before=deepcopy(raw)
            layout,completed=geometry.apply_ranges(raw,{g:3.0 for g in e['missing_group_ids']},world,obs,dims,case['hfov'])
            self.assertEqual(raw,before)
            self.assertEqual(sum(len(g['objects']) for g in layout.values()),len(world['objects']))
            for a,b in zip(raw['groups'],completed['groups']):
                self.assertEqual(a['objects'],b['objects'])
                if a['id'] not in e['missing_group_ids']:self.assertEqual(a,b)
                for o in layout[a['id']]['objects']:self.assertEqual(o['intrinsic_size_m'],dims[o['id']]['full_size_m'])
        repair.verify(self.root)

    def test_patch_schema_rejects_missing_duplicate_extra_nonfinite_and_bool(self):
        good=dict(ranges=[dict(id='g',range_m=2.5)]);self.assertEqual(geometry.ranges_answer(good,['g']),{'g':2.5})
        bad=[None,{},dict(ranges=[]),dict(ranges=good['ranges']*2),dict(ranges=[dict(id='wrong',range_m=2)]),
            dict(ranges=good['ranges'],objects=[]),dict(ranges=[dict(id='g',range_m=2,yaw_deg=0)])]
        bad.extend(dict(ranges=[dict(id='g',range_m=v)]) for v in (None,True,False,0,-1,'3',float('nan'),float('inf')))
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):geometry.ranges_answer(value,['g'])

    def test_capture_rejects_other_missing_fields_without_defaulting(self):
        m=self.prepare();e=m['pending'][0];_,_,case,obs,w,_,d,raw=repair.inputs(self.root,e)
        for field,value in (('yaw_deg',None),('root_offset_m',[None,0,0])):
            bad=deepcopy(raw);obj=next(o for g in bad['groups'] for o in g['objects'] if 'root_offset_m' in o);obj[field]=value
            with self.assertRaises(ValueError):geometry.placement_capture(bad,w,obs,d,case['hfov'])
        with self.assertRaises(ValueError):geometry.apply_ranges(raw,{},w,obs,d,case['hfov'])

    def future_inference(self,missing):
        world=toy_world();dims={o['id']:dict(id=o['id'],full_size_m=[1,.7,.8] if o['id']=='desk' else [.1,.1,.1]) for o in world['objects']}
        value=geometry.placement_template(world)
        for group in value['groups']:
            group['range_m']=None if missing else 3.
            for obj in group['objects']:
                obj['yaw_deg']=15.
                if 'root_offset_m' in obj:obj['root_offset_m']=[.2,0.,.1]
                else:obj['host_offset_uv']=[.3,-.2]
        calls=[]
        class P:
            def ask(inner,image,prompt,path,**kw):
                calls.append(path.stem)
                if path.stem.endswith('_dimensions'):v=dict(objects=list(dims.values()))
                elif path.stem.endswith('_placement'):v=value
                elif path.stem.endswith('_ranges'):v=dict(ranges=[dict(id='g0',range_m=4.)])
                else:raise AssertionError('A full placement retry would erase valid fields: '+path.stem)
                path.write_text(json.dumps(v));write(path.with_suffix('.json'),v);return v
        dest=Path(self.temp.name)
        def ask(name,image,prompt,validator):return ask_valid(P(),image,prompt,dest/(name+'.txt'),validator)
        layout=geometry.infer_layout(dict(hfov=90,image_size=[640,480]),dict(objects=[]),world,{},None,ask,dest,1)
        self.assertEqual(layout['g0']['objects'][0]['yaw_deg'],15.)
        self.assertEqual(layout['g0']['objects'][0]['offset_m'],[.2,0.,.1])
        return calls

    def test_future_partial_reply_uses_small_range_call_without_full_retry(self):
        self.assertEqual(self.future_inference(True),['layout_01_dimensions','layout_01_placement','layout_01_ranges'])
    def test_complete_reply_uses_no_extra_range_call(self):
        self.assertEqual(self.future_inference(False),['layout_01_dimensions','layout_01_placement'])

    def test_api_with_zero_pending_cases_never_initializes_provider(self):
        self.prepare()
        with patch.object(repair,'provider',side_effect=AssertionError('API must not be initialized')):result=repair.execute(self.root,'api')
        self.assertEqual(result['status'],'complete');self.assertEqual(result['cases'],[])

    def test_actual_six_range_recovery_keeps_all_inventory_and_frozen_artifacts(self):
        m=self.prepare();p=RangeProvider();result=repair.execute(self.root,'qwen',p)
        self.assertEqual(len(p.requests),6);self.assertEqual(len(result['cases']),6)
        for e,outcome in zip(m['pending'],result['cases']):
            self.assertNotEqual(outcome['status'],'repair_failed',outcome)
            _,dest,_,obs,w,_,dims,_=repair.inputs(self.root,e)
            path=dest/('G_full.json' if outcome['status']=='world_exported_needs_semantic_review' else 'G_full_candidates.json')
            state=read(path);self.assertEqual(state['objects'][:len(obs['objects'])],obs['objects'])
            self.assertEqual({o['track_id'] for o in state['objects'][len(obs['objects']):]},{o['id'] for o in w['objects']})
            for o in state['objects'][len(obs['objects']):]:self.assertEqual(o['intrinsic_size_m']['value'],dims[o['track_id']]['full_size_m'])
            self.assertTrue((dest/'range_repair_result.json').exists())
        repair.verify(self.root)
        for k,h in m['source_json_sha256'].items():
            if k.endswith('/G_full.json'):self.assertEqual(sha(self.root/k),h)
        with self.assertRaises(FileExistsError):repair.execute(self.root,'qwen',p)

    def test_invalid_patch_keeps_original_failures_and_has_fixed_budget(self):
        m=self.prepare();p=RangeProvider(invalid=True);result=repair.execute(self.root,'qwen',p)
        self.assertEqual(len(p.requests),12);self.assertEqual(result['status'],'partial_or_failed')
        self.assertTrue(all(c['status']=='repair_failed' for c in result['cases']))
        for e in m['pending']:
            c,dest,*_=repair.inputs(self.root,e)
            self.assertIn(e['original_failure'],read(c/'summary.json')['failures'])
            self.assertFalse((dest/'G_full.json').exists())
        repair.verify(self.root)

    def test_copy_tampering_and_existing_output_are_rejected(self):
        m=self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()
        e=m['pending'][0];_,dest,*_=repair.inputs(self.root,e);p=dest/'semantic_world.json'
        value=read(p);value['objects'].pop();write(p,value)
        with self.assertRaises(ValueError):repair.verify(self.root)

    def test_original_run_cannot_be_execution_target(self):
        before=repair.json_hashes(self.source)
        for action in ('run','report','pending'):
            args=[sys.executable,'-m','qwen_edit_pano.world_pipeline.completion.assembly_organized_repair',action,'--root',str(self.source),'--backend','qwen']
            proc=subprocess.run(args,capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(proc.returncode,0)
        self.assertEqual(before,repair.json_hashes(self.source))

    def test_report_pairs_all_four_branches_and_keeps_original_state_hashes(self):
        self.prepare();calls=[]
        def visual(root,source,out,**kwargs):
            calls.append((source,kwargs));out.mkdir();p=out/'index.html';p.write_text('<meta charset="utf-8"><p>fixture</p>');return p
        with patch('qwen_edit_pano.world_pipeline.completion.assembly_holistic_visual.report',side_effect=visual):
            first=repair.report(self.root);second=repair.report(self.root)
        self.assertNotEqual(first,second)
        self.assertEqual(calls[0][0],self.source)
        self.assertEqual(calls[0][1]['gallery_experiments'],['end_to_end','layout_control','function_free','geometry_control'])
        self.assertEqual(calls[0][1]['baseline_experiments'],calls[0][1]['gallery_experiments'])
        self.assertIn('not_attempted',first.read_text());self.assertIn('CPU 合成',first.read_text());repair.verify(self.root)

    def test_render_failure_never_exports_a_completed_state(self):
        m=self.prepare();e=m['pending'][0];_,dest,case,obs,w,_,dims,raw=repair.inputs(self.root,e)
        layout,_=geometry.apply_ranges(raw,{g:3.0 for g in e['missing_group_ids']},w,obs,dims,case['hfov'])
        with patch('qwen_edit_pano.world_pipeline.completion.semantic_layout_visual.render',side_effect=RuntimeError('fixture render failure')):
            with self.assertRaisesRegex(RuntimeError,'fixture render failure'):repair.materialize(self.root,e,layout)
        self.assertFalse((dest/'G_full.json').exists());repair.verify(self.root)


if __name__=='__main__':unittest.main()
