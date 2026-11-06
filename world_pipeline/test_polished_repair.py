"""CPU tests of source preservation and explicit repair selection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .build_polished_world import PROFILE, FRAME, atomic_json, pair_identity, read_rows, summarize
from .prepare_polished_repair import prepare, MISSING_POSE


class RepairTests(unittest.TestCase):
    def fixture(self, root):
        source,candidates=root/'source',root/'candidates'
        source.mkdir();candidates.mkdir();(source/'BUILD.lock').touch()
        rows=[]
        for sid in ('done','unposed','weak'):
            rows.append(dict(id=sid,split='train',scene_id='house',source_view_id=sid,
                local_condition=dict(archive='local.zip',member=sid,face_index=2),
                image=dict(arrow_file='target.arrow',batch_index=0,row_index=0),
                target_alignment=dict(mirror=False,roll_fraction=.75),objects=None,
                object_region_supervision=None,state_status='UNBUILT'))
        names={'11':'bed'}
        atomic_json(candidates/'category_names.json',names)
        for split,values in [('train',rows),('val',[])]:
            (candidates/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
        record=deepcopy(rows[0])
        mask,calibration=source/'mask.npz',source/'calibration.json'
        mask.write_bytes(b'fixture');atomic_json(calibration,dict(profile=PROFILE))
        record.update(objects=[dict(instance_id='1',category_id=11,center_local_m=[0,0,3],
            size_aabb_local_m=[2,1,2],center_valid=True,size_valid=True)],
            frame=FRAME,object_region_supervision=dict(path=str(mask)),calibration=str(calibration),
            state_status='BUILT_MINIMAL_STATE')
        (source/'train.jsonl').write_text(json.dumps(record)+'\n');(source/'val.jsonl').write_text('')
        config=dict(profile=PROFILE,frame=FRAME,target='polished_ERP',Local='single_original_face2',
            size='camera_AABB_full_side_lengths_metres',height=512,min_instance_pixels=16,min_front_ncc=.65,
            object_loss_max_latitude=54.,sift_features=3600,ransac_trials=3000,rebuild_existing=False,
            names=names,sources=[pair_identity(row) for row in rows],raw_root='/raw',category_mapping='/mapping')
        atomic_json(source/'CONFIG.json',config)
        atomic_json(source/'BUILD_RESULT.json',dict(requested=3,completed=1,failures=[
            dict(id='unposed',split='train',error=MISSING_POSE),
            dict(id='weak',split='train',error='ValueError: No reliable registration consensus')]))
        return source,candidates,record,config

    def test_exclusion_requires_explicit_scope_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,record,config=self.fixture(root)
            output,candidate_output=root/'repaired',root/'selected'
            original=(source/'CONFIG.json').read_text()
            with self.assertRaisesRegex(ValueError,'explicitly pass'):
                prepare(source,candidates,output,candidate_output)
            self.assertFalse(output.exists())
            selection=prepare(source,candidates,output,candidate_output,True)
            self.assertEqual((selection['original_requested'],selection['selected_requested'],selection['seeded_successes'],selection['registration_retries']),(3,2,1,1))
            self.assertEqual((source/'CONFIG.json').read_text(),original)
            self.assertEqual(json.loads((output/'samples/done/record.json').read_text()),record)
            self.assertFalse((output/'samples/unposed').exists())
            actual=json.loads((output/'CONFIG.json').read_text())
            self.assertEqual(actual['registration_fit_parities'],[0,1])
            self.assertEqual(actual['min_front_ncc'],config['min_front_ncc'])
            self.assertIn('--registration-fit-parities 0 1',(output/'run_build.sh').read_text())
            selected=read_rows(candidate_output/'train.jsonl')
            pending=deepcopy(record)
            pending.update(pair_identity(next(row for row in selected if row['id']=='weak')))
            atomic_json(output/'samples/weak/record.json',pending)
            (output/'samples/weak/failure.json').unlink()
            result=summarize(output,selected)
            self.assertTrue(result['all_candidates_built'])
            self.assertFalse(result['all_original_candidates_built'])
            self.assertEqual((result['requested'],result['original_requested'],result['excluded_from_original']),(2,3,1))
            with self.assertRaisesRegex(ValueError,'NEW'):
                prepare(source,candidates,output,candidate_output,True)

    def test_missing_completed_asset_is_not_silently_seeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,_,_=self.fixture(root)
            (source/'mask.npz').unlink()
            with self.assertRaisesRegex(ValueError,'lacks'):
                prepare(source,candidates,root/'out',root/'selected',True)
            self.assertFalse((root/'out').exists())

    def test_resume_finishes_partial_preparation_and_keeps_existing_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,record,_=self.fixture(root)
            output,selected=root/'out',root/'selected'
            prepare(source,candidates,output,selected,True)
            saved=output/'samples/done/record.json'
            saved.unlink();(output/'run_build.sh').unlink()
            prepare(source,candidates,output,selected,True,resume=True)
            self.assertEqual(json.loads(saved.read_text()),record)
            self.assertTrue((output/'run_build.sh').is_file())
            before=saved.stat().st_mtime_ns
            prepare(source,candidates,output,selected,True,resume=True)
            self.assertEqual(saved.stat().st_mtime_ns,before)
            actual=json.loads(saved.read_text());actual['objects']=[]
            atomic_json(saved,actual)
            with self.assertRaisesRegex(ValueError,'completed record differs'):
                prepare(source,candidates,output,selected,True,resume=True)
            self.assertEqual(json.loads(saved.read_text()),actual)

    def test_resume_rejects_different_configuration_and_active_writer(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,_,_=self.fixture(root)
            output,selected=root/'out',root/'selected'
            prepare(source,candidates,output,selected,True)
            with (output/'BUILD.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'Another process'):
                    prepare(source,candidates,output,selected,True,resume=True)
            path=output/'CONFIG.json';config=json.loads(path.read_text());config['height']=128
            atomic_json(path,config)
            with self.assertRaisesRegex(ValueError,'configuration/selection differs'):
                prepare(source,candidates,output,selected,True,resume=True)
            self.assertEqual(json.loads(path.read_text())['height'],128)

    def test_audited_failures_need_explicit_exclusion_and_exact_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,_,_=self.fixture(root)
            audit=root/'audit.jsonl';audit.write_text(json.dumps(dict(id='weak',status='registration_failed',error='insufficient validation'))+'\n')
            with self.assertRaisesRegex(ValueError,'exclude-unregistered'):
                prepare(source,candidates,root/'out',root/'selected',True,audit)
            selection=prepare(source,candidates,root/'out',root/'selected',True,audit,True)
            self.assertEqual((selection['excluded_count'],selection['selected_requested'],selection['registration_retries']),(2,1,0))

    def test_completed_recovery_reused_only_with_matching_calibration_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,candidates,record,config=self.fixture(root)
            config['height']=2
            atomic_json(source/'CONFIG.json',config)
            review=root/'review';directory=review/'recovered/samples/weak'
            directory.mkdir(parents=True);(review/'REVIEW.lock').touch()
            atomic_json(review/'CONFIG.json',dict(protocol='polished_object_alignment_review_v1',
                source=str(source),candidates=str(candidates),source_build_config=config))
            pending=next(row for row in read_rows(candidates/'train.jsonl') if row['id']=='weak')
            recovered=deepcopy(record);recovered.update(pair_identity(pending))
            mask=directory/'object_regions.npz';calibration=directory/'calibration.json'
            valid=np.ones((2,4),dtype=bool)
            np.savez(mask,instance=np.ones((2,4),dtype=np.int32),instance_valid=valid,object_loss_valid=valid)
            recovered.update(calibration=str(calibration),object_region_supervision=dict(path=str(mask)))
            atomic_json(directory/'record.json',recovered)
            metrics=dict(sample_id='weak',profile=PROFILE,frame=FRAME,object_count=1,
                requested_fit_parities=[0,1],fit_inliers=80,heldout_inliers=40,heldout_observations=4,
                local_target_front_ncc=.9,sift_features=3600,ransac_trials=3000,object_loss_max_latitude=54.)
            atomic_json(calibration,metrics)
            selection=prepare(source,candidates,root/'out',root/'selected',True,recovered_review=review)
            self.assertEqual((selection['seeded_successes'],selection['recovered_successes'],selection['registration_retries']),(2,1,0))
            self.assertEqual(json.loads((root/'out/samples/weak/record.json').read_text()),recovered)
            self.assertFalse((root/'out/samples/weak/failure.json').exists())
            metrics['heldout_inliers']=29;atomic_json(calibration,metrics)
            with self.assertRaisesRegex(ValueError,'calibration differs'):
                prepare(source,candidates,root/'bad',root/'bad_selected',True,recovered_review=review)
            metrics['heldout_inliers']=40;atomic_json(calibration,metrics)
            np.savez(mask,instance=np.ones((2,4),dtype=np.int32),instance_valid=valid,object_loss_valid=~valid)
            with self.assertRaisesRegex(ValueError,'loss support differs'):
                prepare(source,candidates,root/'bad',root/'bad_selected',True,recovered_review=review)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':
    unittest.main()
