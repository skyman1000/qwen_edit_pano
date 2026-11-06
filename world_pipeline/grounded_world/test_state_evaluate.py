"""CPU checks of validation isolation, paired jobs and strict partial resume."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..build_polished_world import atomic_json
from ..object_world.data import identity
from .state_evaluate import validation_cases,jobs,expected_record,checked_pending,run


def inputs():
    rows={};cases=[]
    for i in range(7):
        row=dict(id=f'view{i}',split='val' if i<6 else 'train',scene_id=f'house{i}',source_view_id=str(i),
            local_condition=dict(archive='/existing.zip',member=f'{i}.jpg',face_index=2),
            image=dict(arrow_file='/existing.arrow',batch_index=0,row_index=i),
            target_alignment=dict(mirror=False,roll_fraction=.75))
        rows[row['id']]=row
        cases.append(dict(id=row['id'],split=row['split'],source=identity(row)))
    return dict(cases=cases),rows,['view6']


class NaturalStateTests(unittest.TestCase):
    def protocol(self):
        panel,byid,train_ids=inputs()
        return dict(cases=validation_cases(panel,byid,train_ids),arms=dict(pano_only=None,
            correct_step4000='/checkpoint4000',correct_step6000='/checkpoint6000'))

    def test_six_validation_views_and_matched_state_jobs(self):
        protocol=self.protocol();pairs=jobs(protocol)
        self.assertEqual(len(pairs),18)
        self.assertNotIn('view6',[case['id'] for case in protocol['cases']])
        for case in protocol['cases']:
            records=[expected_record(protocol,c,arm) for c,arm in pairs if c['id']==case['id']]
            self.assertEqual({r['seed'] for r in records},{case['seed']})
            self.assertEqual([r['mode'] for r in records],['world_off','correct','correct'])
            self.assertEqual([r['checkpoint'] for r in records],[None,'/checkpoint4000','/checkpoint6000'])

    def test_reject_validation_contamination_and_source_changes(self):
        panel,byid,train_ids=inputs()
        with self.assertRaises(ValueError):validation_cases(panel,byid,train_ids+['view0'])
        byid['view0']['target_alignment']['roll_fraction']=0.
        with self.assertRaises(ValueError):validation_cases(panel,byid,train_ids)
        panel,byid,train_ids=inputs();panel['cases'][1]=panel['cases'][0]
        with self.assertRaises(ValueError):validation_cases(panel,byid,train_ids)

    def test_resume_requires_matching_seed_checkpoint_and_runtime(self):
        protocol=self.protocol()
        with TemporaryDirectory() as directory:
            out=Path(directory);atomic_json(out/'PROTOCOL.json',protocol)
            self.assertEqual(len(checked_pending(out,protocol)),18)
            case,arm=jobs(protocol)[0];target=out/case['id'];target.mkdir()
            execution=dict(device='CPU-test-only')
            atomic_json(out/'EXECUTION.json',execution)
            record=dict(**expected_record(protocol,case,arm),execution=execution,
                branch_calls=dict(policy='both',cond=28,uncond=28,disabled_uncond=0))
            atomic_json(target/(arm+'.json'),record)
            self.assertEqual(len(checked_pending(out,protocol)),18)  # PNG missing.
            (target/(arm+'.png')).write_bytes(b'placeholder; dimensions are checked by summarize')
            self.assertEqual(len(checked_pending(out,protocol)),17)
            record['seed']+=1;atomic_json(target/(arm+'.json'),record)
            with self.assertRaises(ValueError):checked_pending(out,protocol)
            record['seed']-=1;record['checkpoint']='/other';atomic_json(target/(arm+'.json'),record)
            with self.assertRaises(ValueError):checked_pending(out,protocol)
            record['checkpoint']=None;record['execution']=dict(device='different');atomic_json(target/(arm+'.json'),record)
            with self.assertRaises(ValueError):checked_pending(out,protocol)

    def test_preflight_does_not_start_CUDA_or_write_output(self):
        protocol=self.protocol()
        with TemporaryDirectory() as directory:
            out=Path(directory)/'not-created'
            with patch(__package__+'.state_evaluate.prepare',return_value=({}, {}, None, protocol)),patch('torch.cuda.init') as cuda:
                run(Path(directory),(4000,6000),out,True)
                cuda.assert_not_called()
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
