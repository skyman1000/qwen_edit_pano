"""CPU contracts and isolation tests for the raw-RGB structured-world baseline."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import ROOT,read,write,sha
from . import direct_rgb_world as direct,direct_rgb_review as review


def create(path):path.mkdir(parents=True,exist_ok=False);return path.resolve()


def answer():
    # Deliberately put a claimed hidden box INSIDE the input and overlapping
    # the visible sofa. The baseline must preserve this observable weakness.
    return dict(objects=[['v0','sofa','sofa','visible',0.,0.,3.,2.,1.,1.,0.],
        ['h0','small side table','table','outside_view',.1,0.,3.,.5,.5,.5,0.],
        ['h1','unknown fixture','unclassified_xyz','outside_view',0.,0.,-3.,.2,.3,.4,90.]],relations=[])


class Provider:
    def __init__(self,values=None):self.requests=[];self.values=values or [answer()]
    def ask(self,image,prompt,path,**kwargs):
        if image is None:raise AssertionError('The direct model must receive original RGB')
        self.requests.append(dict(prompt=prompt,pixels=direct.pixel_sha(image),**kwargs));path=Path(path)
        value=self.values[min(len(self.requests)-1,len(self.values)-1)]
        write(path.with_suffix('.request.json'),dict(prompt=prompt,canonical_image_sha256=__import__('hashlib').sha256(direct.image_bytes(image)).hexdigest(),**kwargs))
        if isinstance(value,Exception):path.write_text('BAD_REPETITIVE_OUTPUT '*1000);raise value
        path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return deepcopy(value)


class DirectRGBTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)

    def prepared(self,name='direct'):
        image=self.base/(name+'_input.png');Image.new('RGB',(320,240),'#aabbcc').save(image)
        manifest=self.base/(name+'.jsonl');manifest.write_text(json.dumps(dict(id='scene',local_rgb=str(image),local_sha256=sha(image),hfov_degrees=90,split='CPU_fixture'))+'\n')
        root=self.base/name
        with patch.object(direct,'new_output',side_effect=create):direct.prepare(root,manifest)
        config=read(root/'direct_experiment.json');config['synthetic_fixture']=True;write(root/'direct_experiment.json',config);return root

    def test_preparation_reads_RGB_only_and_has_no_observer_or_WildDet_requirement(self):
        with patch('qwen_edit_pano.world_pipeline.completion.fresh_pipeline.preflight',side_effect=AssertionError('No main-pipeline/WildDet preflight')):
            root=self.prepared()
        cfg=direct.verify(root)
        for flag in ('wilddet_used','observer_stage_used','explicit_causal_model_used','program_library_used','layout_solver_used','scene_GT_used'):self.assertFalse(cfg[flag])
        self.assertEqual(cfg['worlds_per_image_backend'],1);self.assertFalse((root/'scenes/scene/observer3d').exists())

    def test_structured_schema_rejects_bad_geometry_ids_and_relations(self):
        good=answer();self.assertEqual(direct.validate_answer(good),good)
        for value in (None,{},dict(good,extra=1),dict(good,relations=[['h0','near','missing']]),dict(good,relations=[['v0','near','v0']])):
            with self.assertRaises(ValueError):direct.validate_answer(value)
        for index,v in ((4,None),(5,True),(6,float('nan')),(7,0),(8,-1),(10,'90')):
            bad=deepcopy(good);bad['objects'][0][index]=v
            with self.subTest(index=index),self.assertRaises(ValueError):direct.validate_answer(bad)
        bad=deepcopy(good);bad['objects'][1][0]='v0'
        with self.assertRaises(ValueError):direct.validate_answer(bad)
        with self.assertRaises(ValueError):direct.validate_answer(good,max_objects=2)

    def test_intrinsic_sizes_yaw_transform_is_deterministic_not_pose_optimization(self):
        self.assertTrue(np.allclose(direct.camera_aabb([2,1,4],90),[4,1,2]))
        self.assertTrue(np.allclose(direct.camera_aabb([2,1,4],-90),[4,1,2]))
        self.assertTrue(np.allclose(direct.camera_aabb([2,1,4],0),[2,1,4]))

    def test_generation_never_calls_observer_causal_model_or_physical_optimizer(self):
        root=self.prepared();p=Provider()
        with (
            patch('qwen_edit_pano.world_pipeline.completion.fresh_wilddet.run',side_effect=AssertionError('No WildDet')),
            patch('qwen_edit_pano.world_pipeline.completion.fresh_observer.observe',side_effect=AssertionError('No observer')),
            patch('qwen_edit_pano.world_pipeline.completion.assembly_world._run',side_effect=AssertionError('No SCM')),
            patch('qwen_edit_pano.world_pipeline.completion.assembly_organized_geometry.compile_layout',side_effect=AssertionError('No optimizer')),
        ):
            result=direct.generate(root,'qwen',p)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(p.requests),1)
        full=read(root/'scenes/scene/qwen/G_full.json');self.assertEqual(len(full['objects']),3)
        for o,row in zip(full['objects'],answer()['objects']):
            self.assertEqual(o['track_id'],row[0]);self.assertEqual(o['center_local_m']['value'],row[4:7]);self.assertEqual(o['intrinsic_size_m']['value'],row[7:10]);self.assertEqual(o['yaw_camera_deg']['value'],row[10])
        self.assertFalse(full['objects'][2]['category_id']['valid'])
        metrics,_=review.measure(full,90,[320,240]);self.assertGreater(metrics['claimed_hidden_intersects_input'],0);self.assertGreater(metrics['overlap_nonhost_pairs'],0)

    def test_backends_receive_identical_primary_prompt_pixels_and_recipe(self):
        root=self.prepared();a=Provider();q=Provider();direct.generate(root,'api',a);direct.generate(root,'qwen',q)
        self.assertEqual(a.requests,q.requests);self.assertEqual(a.requests[0]['max_tokens'],8192)
        self.assertEqual(a.requests[0]['pixels'],read(root/'direct_experiment.json')['scenes'][0]['pixel_sha256'])

    def test_format_retry_has_fixed_budget_and_does_not_copy_bad_long_text(self):
        root=self.prepared();p=Provider([ValueError('Unfinished local response reached token limit'),answer()]);direct.generate(root,'qwen',p)
        self.assertEqual(len(p.requests),2);self.assertNotIn('BAD_REPETITIVE_OUTPUT',p.requests[1]['prompt'])
        self.assertEqual(read(root/'scenes/scene/qwen/summary.json')['model_calls'],2)

    def test_failed_and_completed_cases_are_not_resampled_on_resume(self):
        root=self.prepared();p=Provider([ValueError('Invalid JSON')]);result=direct.generate(root,'api',p)
        self.assertEqual(result['status'],'partial_or_failed');self.assertEqual(len(p.requests),2)
        with patch.object(direct,'provider',side_effect=AssertionError('No retry until success')):direct.generate(root,'api')
        self.assertFalse((root/'scenes/scene/api/G_full.json').exists())

    def test_interrupted_attempt_is_preserved_without_extra_draw(self):
        root=self.prepared();(root/'scenes/scene/qwen').mkdir();(root/'scenes/scene/qwen/raw.txt').write_text('partial')
        with patch.object(direct,'provider',side_effect=AssertionError('No repeated interrupted attempt')):r=direct.generate(root,'qwen')
        self.assertEqual(r['status'],'partial_or_failed');self.assertIn('Interrupted',r['results'][0]['error'])

    def test_API_really_sends_image_payload_with_fixed_model_and_no_key_in_logs(self):
        from .world_api import SiliconFlowJSON
        captured=[]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self):return json.dumps(dict(model='deepseek-ai/DeepSeek-V4.1-Flash',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(answer())))])).encode()
        def opener(request,timeout):captured.append(json.loads(request.data));return Response()
        with patch.dict(os.environ,SILICONFLOW_API_KEY='CPU_TEST_SECRET'):
            p=SiliconFlowJSON(model='deepseek-ai/DeepSeek-V4.1-Flash',base='https://api.siliconflow.cn/v1',stream=False,generation_profile='matched_v1',opener=opener)
            path=self.base/'api.txt';p.ask(Image.new('RGB',(32,32),'red'),'CPU prompt',path,max_tokens=8192)
        payload=captured[0];self.assertEqual(payload['model'],'deepseek-ai/DeepSeek-V4.1-Flash');self.assertEqual(payload['messages'][1]['content'][0]['type'],'image_url')
        self.assertNotIn('CPU_TEST_SECRET',path.with_suffix('.request.json').read_text());self.assertNotIn('seed',payload)

    def test_comparison_uses_same_RGB_camera_and_common_read_only_metrics(self):
        root=self.prepared();direct.generate(root,'qwen',Provider());direct.generate(root,'api',Provider());causal=self.base/'causal';folder=causal/'scenes/scene/observation';folder.mkdir(parents=True)
        shutil.copyfile(root/'scenes/scene/input.png',folder/'input.png');write(causal/'suite.json',dict(worlds=1,scenes=[dict(id='scene',hfov=90)]))
        for b in ('qwen','api'):
            target=causal/'scenes/scene/completion/end_to_end'/b/'completion';(target/'world_01').mkdir(parents=True)
            shutil.copyfile(root/'scenes/scene'/b/'G_full.json',target/'world_01/G_full.json');write(target/'summary.json',dict(status='CPU_fixture_export'))
        before={str(p):sha(p) for p in causal.rglob('*.json')}
        with patch.object(review,'new_output',side_effect=create):page=review.report(root,self.base/'visual',causal)
        metrics=read(page.parent/'metrics.json');self.assertEqual(len(metrics),4)
        for b in ('qwen','api'):
            rows=[m for m in metrics if m['backend']==b];self.assertEqual(rows[0]['hidden'],rows[1]['hidden']);self.assertEqual(rows[0]['overlap_nonhost_pairs'],rows[1]['overlap_nonhost_pairs'])
        v=read(page.parent/'validation.json');self.assertEqual(v['views'],4);self.assertEqual(v['broken_links'],[])
        self.assertEqual(before,{str(p):sha(p) for p in causal.rglob('*.json')})
        Image.new('RGB',(320,240),'black').save(folder/'input.png')
        with self.assertRaisesRegex(ValueError,'RGB/HFOV mismatch'):review.report(root,self.base/'bad_compare',causal)

    def test_optional_reference_reads_only_RGB_ID_HFOV_not_G_obs_or_CPT(self):
        reference=self.base/'reference';reference.mkdir();image=self.base/'raw.png';Image.new('RGB',(100,100),'white').save(image)
        write(reference/'suite.json',dict(scenes=[dict(id='raw_scene',image=str(image),image_sha256=sha(image),hfov=90)]))
        # There are deliberately no observer or completion files at all.
        with patch.object(direct,'new_output',side_effect=create):cfg=direct.prepare(self.base/'from_reference',reference=reference)
        self.assertEqual(len(cfg['scenes']),1);self.assertEqual(cfg['reference_data_used'],'RGB_ID_HFOV_only')

    def test_original_causal_directory_is_not_an_execution_target(self):
        root=self.base/'causal_only';root.mkdir();write(root/'suite.json',dict(kind='causal'))
        with self.assertRaises(FileNotFoundError):direct.verify(root)
        self.assertEqual(list(root.iterdir()),[root/'suite.json'])


if __name__=='__main__':unittest.main()
