"""CPU checks of repair rejection, quarantine, and interrupted commits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .source_repair_ops import choose_observations, accept_quarantined_region, repaired_world, PROFILE
from .common import read
from .checkpoint_io import atomic_json


class RepairTests(unittest.TestCase):
    def test_observation_guards(self):
        checks=[dict(observation_id=f'camera_{pitch}_{yaw}', count=20, median_relative=.01)
                for pitch in range(3) for yaw in range(6)]
        checks[0].update(count=0,median_relative=None)
        checks[1]['median_relative']=.2
        good,bad,ok=choose_observations(checks)
        self.assertTrue(ok);self.assertEqual(len(good),16);self.assertEqual(len(bad),2)
        self.assertFalse(choose_observations(checks[:8])[2])
        self.assertFalse(choose_observations([r for r in checks if not r['observation_id'].endswith('_5')])[2])

    def test_quarantine_refuses_other_errors(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);atomic_json(p/'region_validation.json',dict(errors=['label','bad geometry']))
            with self.assertRaises(ValueError):
                accept_quarantined_region(p,[dict(validation_error='label')])
            self.assertEqual(read(p/'region_validation.json')['errors'],['label','bad geometry'])

    def test_quarantine_preserves_original(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);report=dict(status='FAIL',errors=['label'])
            atomic_json(p/'region_validation.json',report)
            accept_quarantined_region(p,[dict(validation_error='label')])
            self.assertEqual(read(p/'region_validation.original.json'),report)
            self.assertFalse(read(p/'region_validation.json')['ready_for_world_supervision'])

    def run_world(self,missing=False):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);index=p/'index';index.mkdir();geo=p/'geo';geo.mkdir();out=p/'world'
            atomic_json(p/'semantic_quarantine.json',dict(conflicts=[dict(instance_id=2)]))
            (index/'panorama_index.jsonl').write_text(json.dumps(dict(sample_id='x',region_index_path=str(p/'region_index.json')))+'\n')
            atomic_json(geo/'geometry_report.json',dict(sample_id='x'))
            def build(*args):
                out.mkdir()
                atomic_json(out/'objects.json',dict(objects=[dict(id=i,category_id=15,
                    supervision_mask=dict(category=True,position=True)) for i in ([1] if missing else [1,2])]))
                atomic_json(out/'world_state.json',dict(sample_id='x'))
                atomic_json(out/'build_report.json',dict(status='BUILT'))
            with patch('qwen_edit_pano.world_pipeline.lean_world.build',side_effect=build):
                if missing:
                    with self.assertRaises(ValueError):repaired_world(index,geo,out)
                    self.assertFalse((out/'build_report.json').exists())
                else:
                    repaired_world(index,geo,out)
                    objs=read(out/'objects.json')['objects']
                    self.assertTrue(objs[0]['supervision_mask']['category'])
                    self.assertFalse(objs[1]['supervision_mask']['category'])
                    self.assertEqual(objs[1]['category_id'],15)
                    self.assertTrue(objs[1]['supervision_mask']['position'])
                    self.assertEqual(read(out/'build_report.json')['repair_profile'],PROFILE)

    def test_catalog_masks_only_conflict(self):self.run_world()
    def test_incomplete_catalog_not_committed(self):self.run_world(missing=True)


if __name__=='__main__':unittest.main()
