"""CPU checks of the real fixed scene and migration of the old interventions."""
import copy
import json
from pathlib import Path
import unittest

import torch

from .state_probe import SID,OLD_PLAN,ROOT,variants,changes,EXPANDED_CASES
from ..object_world.data import tensorize


class StateProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT/'qwen_edit_pano/data/object_world_polished_v1/samples.jsonl').open() as f:
            cls.row=next(r for r in map(json.loads,f) if r['id']==SID)
        cls.original=copy.deepcopy(cls.row)
        torch.set_num_threads(2)
        cls.cases={name:row for name,_,row in variants(cls.row,'legacy')}
        cls.expanded={name:row for name,_,row in variants(cls.row,'expanded')}

    def test_variants_never_change_original_and_only_edit_object_fields(self):
        self.assertEqual(self.row,self.original)
        for row in self.cases.values():
            self.assertEqual({k:v for k,v in row.items() if k!='objects'},
                             {k:v for k,v in self.original.items() if k!='objects'})

    def test_old_tv_probes_are_preserved_exactly(self):
        from research.world_evaluation.state_probe import variants as old_variants
        plan=json.loads(OLD_PLAN.read_text())
        graph_path=Path(next(c['path'] for c in plan['cases'] if c['name']=='correct'))
        graph=json.loads(graph_path.read_text())
        for name,_,legacy in old_variants(graph):
            expected={o['track_id']:dict(category_id=o['category_id']['value'],
                center_local_m=o['center_local_m']['value'],size_aabb_local_m=o['size_aabb_local_m']['value'])
                for o in legacy['objects']}
            actual={o['instance_id']:{k:o[k] for k in ('category_id','center_local_m','size_aabb_local_m')}
                    for o in self.cases[name]['objects']}
            self.assertEqual(expected,actual)

    def test_add_remove_and_modify_are_isolated(self):
        for name,row in self.cases.items():
            d=changes(self.original,row)
            expected={'correct':(0,0,0),'remove_tv':(1,0,0),'add_plant':(0,1,0)}.get(name,(0,0,1))
            self.assertEqual(tuple(len(d[k]) for k in ('removed','added','modified')),expected)
            if d['modified']:self.assertEqual(d['modified'][0]['instance_id'],'360')
            if d['added']:self.assertEqual(d['added'][0]['category_id'],14)

    def test_tensor_inputs_ignore_training_masks_and_pack_caches(self):
        for row in self.expanded.values():
            minimal={k:row[k] for k in ('id','frame','objects')}
            full_tensor=tensorize(row);minimal_tensor=tensorize(minimal)
            for a,b in zip(full_tensor,minimal_tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(full_tensor[0].shape,(1,len(row['objects'])+1))
            self.assertTrue(full_tensor[2].all())

    def test_expanded_class_edits_preserve_geometry_and_other_objects(self):
        self.assertEqual(set(self.expanded),set(EXPANDED_CASES)-{'baseline','shuffled'})
        edits={'picture_tv':('360',6),'hidden_chair_to_plant':('328',14),
            'hidden_chair_to_cabinet':('328',7),'hidden_chair_to_stool':('328',19),
            'dining_table_to_bed':('335',11),'dining_table_to_sofa':('335',10)}
        for name,(iid,category) in edits.items():
            diff=changes(self.original,self.expanded[name])
            self.assertEqual(len(diff['modified']),1);self.assertFalse(diff['removed'] or diff['added'])
            before,after=diff['modified'][0]['before'],diff['modified'][0]['after']
            self.assertEqual(before['instance_id'],iid);self.assertEqual(after['category_id'],category)
            self.assertEqual({k:v for k,v in before.items() if k!='category_id'},
                             {k:v for k,v in after.items() if k!='category_id'})
        self.assertEqual(self.row,self.original)

    def test_floor_additions_share_geometry_and_keep_every_original_object(self):
        boxes=[]
        for name,category in [('add_floor_plant',14),('add_floor_chair',3),('add_floor_cabinet',7)]:
            d=changes(self.original,self.expanded[name]);self.assertFalse(d['modified'] or d['removed'])
            self.assertEqual(len(d['added']),1);self.assertEqual(d['added'][0]['category_id'],category)
            self.assertLess(d['added'][0]['center_local_m'][2],0)
            boxes.append({k:v for k,v in d['added'][0].items() if k!='category_id'})
        self.assertEqual(boxes[0],boxes[1]);self.assertEqual(boxes[1],boxes[2])

    def test_expanded_states_use_only_supported_classes_and_valid_geometry(self):
        from ..build_polished_world import validate_objects
        names=json.loads((ROOT/'qwen_edit_pano/data/object_world_polished_v1/PLAN.json').read_text())['names']
        for row in self.expanded.values():validate_objects(row['objects'],names)
        for name,iid in [('remove_hidden_chair','328'),('remove_dining_table','335')]:
            d=changes(self.original,self.expanded[name]);self.assertFalse(d['added'] or d['modified'])
            self.assertEqual([o['instance_id'] for o in d['removed']],[iid])

    def test_review_projection_and_html_with_all_cases_on_cpu(self):
        import tempfile
        import numpy as np
        from PIL import Image
        from .state_probe_review import write_review,draw_box,review_groups
        names=['correct_step001000',*EXPANDED_CASES]
        with tempfile.TemporaryDirectory(prefix='expanded-world-review-') as temporary:
            output=Path(temporary)
            y,x=np.mgrid[:48,:96];rgb=np.stack((x*255//95,y*255//47,np.full_like(x,80)),axis=-1).astype(np.uint8)
            for name in ['gt_erp',*names]:Image.fromarray(rgb).save(output/(name+'.png'))
            states={name:row for name,row in self.expanded.items()}
            (output/'input_states.json').write_text(json.dumps(states))
            write_review(output,self.row,names)
            page=(output/'comparison.html').read_text()
            for token in ['滑动对比','餐桌','add_floor_cabinet','clipPath','_boxes.jpg']:self.assertIn(token,page)
            for name in ['tv','dining','floor','hidden_chair']:
                self.assertTrue((output/'reviews'/(name+'_comparison.jpg')).is_file())
            # Looking directly at a box center must put the projected center at
            # the middle of the perspective. Symmetric x/y extents give a
            # symmetric projection under a 0-degree camera.
            box=dict(center_local_m=[0.,0.,3.],size_aabb_local_m=[1.,1.,1.])
            painted=draw_box(Image.new('RGB',(100,100)),box,dict(yaw=0.,pitch=0.,fov=90.),(255,0,0),'input box')
            a=np.array(painted);self.assertTrue((a[40:65,39:42,0]>0).any())
            groups=review_groups(self.row,names);self.assertEqual(len(groups),6)
            for group in groups:
                target=group['target']
                if target is None:continue
                v=np.array(target['center_local_m']);yaw,pitch=np.deg2rad([group['yaw'],group['pitch']])
                ray=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
                np.testing.assert_allclose(ray,v/np.linalg.norm(v),atol=1e-12)


if __name__=='__main__':unittest.main()
