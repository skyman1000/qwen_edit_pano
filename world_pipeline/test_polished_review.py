"""CPU checks of review fidelity; these are not model/data accuracy tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .build_polished_world import FRAME
from .review_polished_world import encode_ids, contours, choose_existing, render_case, make_page


class ReviewTests(unittest.TestCase):
    def test_ids_remain_exact_and_only_conditioned_instances_are_shown(self):
        raw=np.array([[-1,0,1,255,256,65793,9]],dtype=np.int32)
        encoded,ids=encode_ids(raw,raw>0,{1,255,256,65793})
        decoded=(encoded[...,0].astype(np.int64)<<16)+(encoded[...,1].astype(np.int64)<<8)+encoded[...,2]
        np.testing.assert_array_equal(ids,decoded)
        self.assertEqual(ids.tolist(),[[0,0,1,255,256,65793,0]])

    def test_latitude_is_not_periodic(self):
        ids=np.ones((3,4),dtype=np.int32);ids[2]=0
        edge=contours(ids)
        self.assertFalse(edge[0].any())
        self.assertTrue(edge[1].all())

    def test_selection_is_deterministic_unique_and_does_not_change_records(self):
        records=[dict(id=str(i),scene_id=str(i//2),objects=[{}]*(i+1),
                      state_status='REUSED_EXISTING_STATE' if i%2 else 'BUILT_MINIMAL_STATE') for i in range(16)]
        calibrations={str(i):dict(local_target_front_ncc=.7+i*.01,supervised_pixel_fraction=.2) for i in range(16)}
        first=choose_existing(records,calibrations,12)
        self.assertEqual(first,choose_existing(records,calibrations,12))
        self.assertEqual(len({row['id'] for row in first}),12)
        self.assertTrue(any(row['reason']=='many_objects' for row in first))
        self.assertNotIn('reason',records[0])

    def test_wrong_loss_support_is_rejected_and_full_object_statistics_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mask=root/'regions.npz'
            ids=np.ones((8,16),dtype=np.int32);valid=np.ones(ids.shape,dtype=bool)
            lat=90-(np.arange(8)+.5)/8*180;loss=np.broadcast_to(np.abs(lat)[:,None]<=54,ids.shape).copy()
            np.savez_compressed(mask,instance=ids,instance_valid=valid,object_loss_valid=loss)
            obj=dict(instance_id='1',category_id=11,center_local_m=[0,0,3],size_aabb_local_m=[2,1,2],center_valid=True,size_valid=True)
            row=dict(id='sample',split='val',scene_id='house',frame=FRAME,objects=[obj],state_status='BUILT_MINIMAL_STATE',
                     object_region_supervision=dict(path=str(mask)),calibration='calibration.json')
            calibration=dict(sample_id='sample',object_loss_max_latitude=54)
            with patch('qwen_edit_pano.world_pipeline.review_polished_world.aligned_images',
                       return_value=(np.zeros((512,512,3),dtype=np.uint8),np.zeros((8,16,3),dtype=np.uint8))):
                case=render_case(row,calibration,{'11':'bed'},dict(height=8,object_loss_max_latitude=54),root/'case','test')
            self.assertEqual(case['objects'][0]['ERP_pixels'],128)
            self.assertEqual(case['objects'][0]['region_loss_pixels'],64)
            self.assertFalse(case['ready_for_training'])
            ids[0,0]=0
            np.savez_compressed(mask,instance=ids,instance_valid=valid,object_loss_valid=loss)
            with self.assertRaisesRegex(ValueError,'positive IDs'):
                render_case(row,calibration,{'11':'bed'},dict(height=8,object_loss_max_latitude=54),root/'case','test')
            ids[0,0]=1
            np.savez_compressed(mask,instance=ids,instance_valid=valid,object_loss_valid=np.ones(ids.shape,dtype=bool))
            with self.assertRaisesRegex(ValueError,'support differs'):
                render_case(row,calibration,{'11':'bed'},dict(height=8,object_loss_max_latitude=54),root/'case','test')

    def test_page_is_offline_and_does_not_embed_unescaped_script_terminators(self):
        page=make_page([dict(id='</script><script>bad()</script>')],[])
        self.assertIn('\\u003c/script>',page)
        self.assertNotIn('</script><script>bad()',page)
        self.assertNotIn('fetch(',page)


if __name__=='__main__':
    unittest.main()
