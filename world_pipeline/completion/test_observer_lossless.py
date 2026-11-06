"""Saved five-scene failures and scene-isolation integration, CPU only.

Model replies are archived data or explicit test fixtures. No GPU/API inference.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import ROOT,read,write,sha
from . import assembly_io_recovery as io,assembly_multiscene as suite,local_observation as local
from .assembly_experiment import SeededProvider,stage_seed
from .fresh_observer import answer,state_from_answer
from .world_generation import image_bytes

SOURCE=ROOT/'qwen_edit_pano/outputs/local_observation_restored_001'
SAVED=ROOT/'qwen_edit_pano/outputs/kitchen_observer_20261028_144049/scenes'/local.KITCHEN/'observation'


def obj(i):return dict(name='fixture chair '+str(i),category='chair',bbox_1000=[i*10,100,i*10+8,900],on_floor=True)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
    def root(self):
        root=self.base/'observation';root.mkdir();Image.new('RGB',(100,100),'white').save(root/'input.png')
        write(root/'experiment.json',dict(id='fixture',max_tokens=8192));return root

    def test_page_size_is_not_instance_validity_but_total_budget_stays_bounded(self):
        for n in (0,8,9,10,16,64):
            raw=dict(objects=[obj(i) for i in range(n)],evidence_description='CPU fixture',done=True)
            self.assertEqual(io.chunk_answer(raw,[])['objects'],raw['objects'])
        raw=dict(objects=[obj(i) for i in range(65)],evidence_description='CPU fixture',done=True)
        with self.assertRaises(io.InventoryBudgetError):io.chunk_answer(raw,[])
        with self.assertRaises(io.InventoryBudgetError):
            io.chunk_answer(dict(raw,objects=[obj(i) for i in range(55,65)]),[obj(i) for i in range(55)])
        bad=dict(objects=[obj(i) for i in range(10)],evidence_description='CPU fixture',done=True)
        bad['objects'][8]['bbox_1000']=[0,0,0,0]
        with self.assertRaises(ValueError):io.chunk_answer(bad,[])
        bad['objects'][8]=deepcopy(bad['objects'][0])
        with self.assertRaisesRegex(ValueError,'Repeated exact'):io.chunk_answer(bad,[])

    def test_hard_budget_failure_keeps_raw_reply_and_never_asks_to_shrink_inventory(self):
        root=self.root()
        class P:
            calls=0
            def ask(self,image,prompt,path,**kwargs):
                self.calls+=1
                raw=dict(objects=[obj(i) for i in range(65)],evidence_description='CPU fixture',done=True)
                path.write_text(json.dumps(raw));return raw
        p=P()
        with self.assertRaises(io.InventoryBudgetError):io.observe_chunked(p,root)
        self.assertEqual(p.calls,1);self.assertFalse((root/'qwen/observer2d/G_2d.json').exists())
        raw=read(root/'qwen/observer2d/chunk_00_0.txt');self.assertEqual(len(raw['objects']),65)

    def test_overflow_cannot_bypass_duplicate_instance_handling(self):
        root=self.root();rows=[obj(i) for i in range(9)];raw=dict(objects=rows+[deepcopy(rows[0])],evidence_description='CPU fixture',done=True)
        class P:
            calls=0
            def ask(self,image,prompt,path,**kwargs):
                self.calls+=1
                if path.stem=='chunk_00_0':value=deepcopy(raw)
                elif path.stem=='whole_image_review':value=dict(objects=rows,evidence_description='CPU fixture')
                else:raise AssertionError('Unexpected new request: '+path.stem)
                path.write_text(json.dumps(value));write(path.with_suffix('.json'),value);return value
        p=P();result=io.observe_chunked(p,root)
        self.assertEqual(result['objects'],9);self.assertEqual(p.calls,2)
        self.assertEqual(result['exact_duplicate_copies_collapsed'],1)
        self.assertEqual(len(read(root/'qwen/observer2d/reconciliation_candidates.json')['objects']),9)

    def test_schema_repair_does_not_describe_nonempty_proposals_as_an_empty_scene(self):
        root=self.root();prompts=[]
        class P:
            def ask(self,image,prompt,path,**kwargs):
                prompts.append(prompt)
                value=dict(objects=[obj(1)],evidence_description='CPU visible chair')
                if len(prompts)>1:value['done']=True
                path.write_text(json.dumps(value));return value
        result=io.observe_chunked(P(),root)
        self.assertEqual(result['objects'],1);self.assertEqual(len(prompts),2)
        self.assertNotIn('No visible objects have been accepted',prompts[1]);self.assertNotIn('Correct duplicate records',prompts[1])
        self.assertIn('reported field error',prompts[1])


@unittest.skipUnless(SOURCE.exists(),'Saved restored001 unavailable')
class ActualFiveSceneReplayTests(unittest.TestCase):
    def test_all_five_actual_first_replies_keep_ids_boxes_and_original_requests(self):
        files={str(p):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
        expected_counts=[16,8,8,10,3];totals=[]
        for scene in read(SOURCE/'suite.json')['scenes']:
            source=SOURCE/'scenes'/scene['id']/'observation';archive=source/'qwen/observer2d';test=self
            class Replay:
                calls=[]
                def ask(self,image,prompt,path,**kwargs):
                    old=archive/path.name;request=read(old.with_suffix('.request.json'))
                    test.assertEqual(prompt,request['prompt']);test.assertEqual(kwargs['seed'],request['seed'])
                    test.assertEqual(kwargs['max_tokens'],request['max_tokens'])
                    test.assertEqual(hashlib.sha256(image_bytes(image)).hexdigest(),request['canonical_image_sha256'])
                    self.calls.append(path.name);value=read(old.with_suffix('.json'))
                    shutil.copyfile(old,path);write(path.with_suffix('.json'),value);return deepcopy(value)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for name in ('input.png','experiment.json'):shutil.copyfile(source/name,root/name)
                replay=Replay();result=io.observe_chunked(SeededProvider(replay,stage_seed(3407,'shared_observer')),root)
                state=read(root/'qwen/observer2d/G_2d.json');totals.append(result['objects'])
                if scene['id'].startswith('zsNo'):
                    clean=answer(read(archive/'chunk_00_0.json'))
                    expected,_=state_from_answer(clean,read(source/'experiment.json'),'qwen')
                    self.assertEqual(state,expected);self.assertEqual(replay.calls,['chunk_00_0.txt'])
                    audit=read(root/'qwen/observer2d/transport_audit.json')
                    self.assertEqual([len(p['object_indices']) for p in audit['responses'][0]['pages']],[8,2])
                    self.assertFalse((root/'qwen/observer2d/chunk_00_1.txt').exists())
                    self.assertEqual(result['soft_page_overflows'],1)
                else:self.assertEqual(state,read(archive/'G_2d.json'))
        self.assertEqual(totals,expected_counts)
        self.assertEqual(files,{p:sha(Path(p)) for p in files})


class SceneIsolationTests(unittest.TestCase):
    def test_all_blocked_scenes_do_not_initialize_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'phase_records').mkdir()
            config=dict(scenes=[dict(id=s) for s in ('a','b')])
            for s in ('a','b'):(root/'scenes'/s/'observation').mkdir(parents=True)
            with (patch.object(suite,'verify_suite',return_value=config),
                  patch.object(suite,'provider',side_effect=AssertionError('No Qwen')),
                  patch('qwen_edit_pano.world_pipeline.completion.fresh_wilddet.run',side_effect=AssertionError('No WildDet'))):
                result=suite.run_phase(root,'observer3d')
            self.assertEqual(result['status'],'partial_or_failed')
            self.assertTrue(all(r['status']=='blocked_dependency' and r['model_called'] is False for r in result['scenes']))

    def test_3d_runs_valid_scene_even_with_failed_and_missing_2d_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'phase_records').mkdir();calls=[]
            for name,status in [('valid','completed'),('failed','failed')]:
                obs=root/'scenes'/name/'observation';two=obs/'qwen/observer2d';two.mkdir(parents=True)
                # Deliberately leave a stale graph even for the failed scene.
                shutil.copyfile(SAVED/'qwen/observer2d/G_2d.json',two/'G_2d.json')
                write(two/'summary.json',dict(status=status,error='fixture failure' if status=='failed' else None))
            (root/'scenes/missing/observation').mkdir(parents=True)
            def lift(obs,backend,repo,checkpoint,check_experiment):
                calls.append(obs.parents[0].name)
                self.assertEqual(obs.parents[0].name,'valid')
                target=obs/'qwen/observer3d';target.mkdir()
                # Saved metric data is a CPU fixture, not a new 3D inference.
                shutil.copyfile(SAVED/'qwen/observer3d/G_obs.json',target/'G_obs.json')
                write(target/'summary.json',dict(status='completed',objects=3))
                return dict(status='completed',objects=3)
            config=dict(scenes=[dict(id=s) for s in ('valid','failed','missing')])
            with (patch.object(suite,'verify_suite',return_value=config),
                  patch.object(suite,'provider',side_effect=AssertionError('No Qwen for 3D')),
                  patch('qwen_edit_pano.world_pipeline.completion.fresh_wilddet.run',side_effect=lift)):
                result=suite.run_phase(root,'observer3d');suite.run_phase(root,'observer3d')
            rows={r['scene']:r for r in result['scenes']}
            self.assertEqual(calls,['valid']);self.assertEqual(rows['valid']['status'],'complete')
            self.assertEqual(rows['failed']['status'],'blocked_dependency');self.assertFalse(rows['failed']['model_called'])
            self.assertEqual(rows['missing']['status'],'blocked_dependency');self.assertFalse(rows['missing']['model_called'])
            self.assertEqual(result['status'],'partial_or_failed')


if __name__=='__main__':unittest.main()
