"""CPU tests for image budget, immutable panel identity and spherical mask policy."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .evaluation import (POLICY,circular_box,generation_budget,make_panel,
                         object_report,resolve_panel,select_images,write_crop_review)
from .runtime import WorldBranch
from .train import gradient_norms


class EvaluationTests(unittest.TestCase):
    def test_six_images_does_not_expand_six_buildings(self):
        rows=[dict(id=f'{house}_{i}',scene_id=str(house),split='val') for house in range(8) for i in range(100)]
        selected=select_images(rows,'val',6)
        self.assertEqual(len(selected),6)
        self.assertEqual(len({r['scene_id'] for r in selected}),6)
        self.assertEqual(generation_budget(6,['baseline','correct','shuffled'],28)['generated_panoramas'],18)
        self.assertEqual(generation_budget(6,['baseline','correct','shuffled'],28)['total_denoising_iterations'],504)
        self.assertEqual(len(select_images(rows[:200],'val',6)),6)
        with self.assertRaises(ValueError):select_images(rows[:2],'val',6)
        with self.assertRaises(ValueError):generation_budget(6,['correct','correct'],28)

    def test_seam_crop_and_spherical_latitude(self):
        mask=np.zeros((100,200),bool);mask[45:55,[0,199]]=True
        self.assertEqual(circular_box(mask),dict(x_start=199,width=2,y_start=45,height=10))
        self.assertIsNone(circular_box(np.zeros_like(mask)))
        instance=np.zeros((100,200),int)
        instance[45:55,20:30]=1;instance[:10,20:30]=2
        row=dict(id='x',objects=[dict(instance_id=str(i),category_id=3) for i in (1,2)])
        report=object_report(row,{'3':'chair'},instance,np.ones_like(mask),np.ones_like(mask))
        first,second=report['objects']
        self.assertEqual(first['supported_pixels'],second['supported_pixels'])
        self.assertGreater(first['sphere_fraction'],second['sphere_fraction'])

    def test_review_filters_do_not_filter_training_objects(self):
        ids=np.zeros((20,40),int);ids[8:12,18:22]=1;ids[5:10,1:10]=2;ids[1:3,1:3]=3
        support=np.ones_like(ids,bool);support[8:12,18:20]=False
        row=dict(id='x',objects=[dict(instance_id=str(i),category_id=39 if i==2 else 3) for i in (1,2,3)])
        original=copy.deepcopy(row)
        policy=dict(POLICY,minimum_supported_pixels=8,minimum_sphere_fraction=0.,minimum_support_retention=.6)
        report=object_report(row,{'3':'chair','39':'objects'},ids,np.ones_like(support),support,policy)
        self.assertIn('insufficient_supported_fraction',report['objects'][0]['reasons'])
        self.assertIn('generic_semantics',report['objects'][1]['reasons'])
        self.assertIn('small_supported_pixel_count',report['objects'][2]['reasons'])
        self.assertEqual(row,original)

    def test_panel_pins_ids_seeds_donors_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.npz'
            ids=np.zeros((8,16),np.int32);ids[:,4:12]=1
            np.savez(path,instance=ids,instance_valid=np.ones_like(ids,bool),object_loss_valid=np.ones_like(ids,bool))
            rows=[]
            for i in range(8):
                rows.append(dict(id=str(i),scene_id=str(i//2),split='val',source_view_id=str(i),
                    local_condition=dict(archive='a',member=str(i),face_index=2),
                    image=dict(arrow_file='b',batch_index=0,row_index=i),
                    target_alignment=dict(mirror=False,roll_fraction=0.),
                    object_region_supervision=dict(path=str(path)),objects=[dict(instance_id='1',category_id=3)]))
            panel=make_panel({'names':{'3':'chair'}},rows,count=6,seed=7)
            selected=resolve_panel(panel,rows,'val',6,7)
            self.assertEqual(len(selected),6)
            self.assertEqual([s['seed'] for s in panel['samples']],list(range(7,13)))
            wrong=copy.deepcopy(panel);wrong['samples'][0]['source']['target_alignment']['mirror']=True
            with self.assertRaises(ValueError):resolve_panel(wrong,rows,'val',6,7)
            with self.assertRaises(ValueError):resolve_panel(panel,rows,'val',7,7)
            from PIL import Image
            for name in ('gt_erp','baseline','correct','shuffled'):
                Image.new('RGB',(32,16),'blue').save(Path(directory)/(name+'.png'))
            entry=panel['samples'][0]
            write_crop_review(Path(directory),entry,['baseline','correct','shuffled'])
            self.assertTrue((Path(directory)/'object_review.jpg').is_file())

    def test_gradient_diagnostics_preserve_gradients_and_partition_norm(self):
        torch.manual_seed(9)
        world=WorldBranch(32,torch.randn(42,12),(0,2),16)
        before={}
        for name,p in world.named_parameters():
            p.grad=torch.randn_like(p);before[name]=p.grad.clone()
        norms=gradient_norms(world)
        expected=sum(float(g.float().square().sum()) for g in before.values())
        self.assertAlmostEqual(sum(v*v for v in norms.values())/expected,1.,places=5)
        for name,p in world.named_parameters():torch.testing.assert_close(p.grad,before[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
