"""Exact archived-request replay of the restored observer; no model inference."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from ..common import ROOT,read,write,sha
from .global_world import observe2d
from .world_generation import image_bytes
from .fresh_wilddet import export

HIST=ROOT/'qwen_edit_pano/outputs'
CASES=[('organized_numeric_recovery_20261028_001','8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39',16),
       ('kitchen_observer_20261028_144049','8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029',3)]


class Archived:
    def __init__(self,source,test):self.source=source;self.test=test;self.calls=0
    def ask(self,image,prompt,path,**kwargs):
        self.calls+=1;path=Path(path);original=self.source/path.name
        request=read(original.with_suffix('.request.json'))
        self.test.assertEqual(prompt,request['prompt'])
        self.test.assertEqual(kwargs['seed'],request['seed'])
        self.test.assertEqual(kwargs['max_tokens'],request['max_tokens'])
        self.test.assertEqual(hashlib.sha256(image_bytes(image)).hexdigest(),request['canonical_image_sha256'])
        value=read(original.with_suffix('.json'))
        shutil.copyfile(original,path);write(path.with_suffix('.json'),value)
        return deepcopy(value)


@unittest.skipUnless((HIST/CASES[0][0]).exists(),'Archived observations unavailable')
class RestoredObserverTests(unittest.TestCase):
    def test_saved_bedroom_rows_and_mapping_keep_original_contract(self):
        # This older archive predates an extra explanatory sentence in the
        # original observer's >8-row repair prompt. Check its accepted rows,
        # not an assertion that a changed repair prompt was newly inferred.
        from .fresh_observer import answer,state_from_answer
        obs=HIST/'organized_numeric_recovery_20261028_001/scenes/zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a/observation'
        clean=answer(read(obs/'qwen/observer2d/chunk_00_1.json'))
        state,_=state_from_answer(clean,read(obs/'experiment.json'),'qwen')
        self.assertEqual(state,read(obs/'qwen/observer2d/G_2d.json'))
        self.assertEqual(len(state['objects']),8)

    def test_original_requests_and_observations_replayed_without_extra_review(self):
        for run,sid,count in CASES:
            source=HIST/run/'scenes'/sid/'observation'
            before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
            with tempfile.TemporaryDirectory() as tmp:
                obs=Path(tmp);shutil.copyfile(source/'input.png',obs/'input.png')
                shutil.copyfile(source/'experiment.json',obs/'experiment.json')
                client=Archived(source/'qwen/observer2d',self)
                observe2d(obs,lambda:client,3407)
                old2=read(source/'qwen/observer2d/G_2d.json');new2=read(obs/'qwen/observer2d/G_2d.json')
                self.assertEqual(len(new2['objects']),count)
                self.assertEqual(new2,old2)
                self.assertFalse((obs/'review_calls').exists());self.assertFalse((obs/'draft').exists())
                lifted=export(new2,read(source/'qwen/observer3d/native_boxes.json'),'qwen')
                old3=read(source/'qwen/observer3d/G_obs.json')
                for a,b in zip(lifted['objects'],old3['objects']):
                    for key in ('track_id','category_id','raw_category','bbox2d_visible_xyxy_norm','center_local_m','size_aabb_local_m'):
                        self.assertEqual(a[key],b[key])
                # Re-entry must not issue new stochastic requests.
                calls=client.calls;observe2d(obs,lambda:client,3407);self.assertEqual(client.calls,calls)
            self.assertEqual(before,{str(p):sha(Path(p)) for p in before})

    def test_actual_old_and_broken_first_requests_differ_only_in_seed(self):
        sid=CASES[0][1]
        old=read(HIST/CASES[0][0]/'scenes'/sid/'observation/qwen/observer2d/chunk_00_0.request.json')
        broken=read(HIST/'global_world_srun_002/scenes'/sid/'observation/draft/qwen/observer2d/chunk_00_0.request.json')
        self.assertEqual({k:v for k,v in old.items() if k!='seed'},{k:v for k,v in broken.items() if k!='seed'})
        self.assertEqual(old['seed'],1083124977);self.assertEqual(broken['seed'],3407)


if __name__=='__main__':unittest.main()
