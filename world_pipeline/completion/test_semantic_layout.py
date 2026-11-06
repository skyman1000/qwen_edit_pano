"""Numeric export/direction/visualization checks; no model-quality claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read, write
from ..oracle_model import tensorize
from .semantic_layout import build_states,diagnostics,run,validate_boxes
from .semantic_layout_visual import erp_map,erp_uv
from .test_semantic_probe import state


def plan():
    return dict(scene_type='living room',scene_hypothesis='A surrounding seating space.',additions=[
        dict(id='planned_chairs',name='two chairs',reason='Additional seating.',placement='Outside the photo.'),
        dict(id='planned_rug',name='rug',reason='A decorative hypothesis.',placement='On an unmeasured floor.')])


def proposal():
    return dict(objects=[dict(id='chair1',plan_id='planned_chairs',category='chair',center=[3.,-.5,-2.],size=[.6,1.,.6],unseen_reason='outside_fov',reason='A seating corner.'),
        dict(id='chair2',plan_id='planned_chairs',category='chair',center=[-3.,-.5,-2.],size=[.6,1.,.6],unseen_reason='outside_fov',reason='Same seating function.'),
        dict(id='rug1',plan_id='planned_rug',category=None,center=[0.,-1.,-2.],size=[2.,.05,2.],unseen_reason='outside_fov',reason='Unsupported category retained as a candidate.')])


class SemanticLayoutTests(unittest.TestCase):
    def test_multiplied_instances_and_unmapped_candidates_preserve_observed_encoder_prefix(self):
        observed=state();original=deepcopy(observed)
        boxes=validate_boxes(proposal(),plan(),observed)
        all_candidates,full,unmapped=build_states(observed,plan(),boxes)
        self.assertEqual(observed,original)
        self.assertEqual(len(all_candidates['objects']),len(observed['objects'])+3)
        self.assertEqual(len(full['objects']),len(observed['objects'])+2)
        self.assertEqual(full['unmapped_candidate_ids'],['rug1'])
        self.assertFalse(full['semantic_plan_fully_exported'])
        self.assertFalse(unmapped[0]['category_id']['valid'])
        for before,after in zip(tensorize(observed,160),tensorize(full,160)):
            self.assertTrue(before[:,:len(observed['objects'])+1].equal(after[:,:len(observed['objects'])+1]))

    def test_missing_items_unknown_categories_duplicate_ids_and_geometry_are_not_silently_fixed(self):
        for kind in ('missing','duplicate','unsupported','nan','negative','foreign_plan','observed_id'):
            answer=proposal();box=answer['objects'][0]
            if kind=='missing':answer['objects'].pop()
            elif kind=='duplicate':answer['objects'][1]['id']=box['id']
            elif kind=='unsupported':box['category']='new_embedding_id'
            elif kind=='nan':box['center'][0]=float('nan')
            elif kind=='negative':box['size'][0]=-1
            elif kind=='foreign_plan':box['plan_id']='uncommitted_object'
            elif kind=='observed_id':box['id']=state()['objects'][0]['track_id']
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_boxes(answer,plan(),state())

    def test_positions_anywhere_around_camera_are_retained_without_direction_repair(self):
        for angle in np.linspace(-np.pi,np.pi,25):
            answer=proposal();center=[3*float(np.sin(angle)),.2,3*float(np.cos(angle))]
            answer['objects'][0]['center']=center
            candidates,_,_=build_states(state(),plan(),validate_boxes(answer,plan(),state()))
            self.assertEqual(candidates['objects'][-3]['center_local_m']['value'],center)
        front=proposal()['objects'][0];front['center']=[0.,0.,3.]
        self.assertFalse(diagnostics([front],90,(768,768))[0]['outside_input_fov_certified'])
        front['center']=[0.,0.,-3.]
        self.assertTrue(diagnostics([front],90,(768,768))[0]['outside_input_fov_certified'])

    def test_erp_directions_and_back_seam_do_not_draw_a_false_line_through_front(self):
        uv=erp_uv([[0,0,1],[1,0,0],[-1,0,0],[0,1,0]],1200,600)
        np.testing.assert_allclose(uv,[[600,300],[900,300],[300,300],[600,0]])
        observed=state();observed['objects']=[]
        p=dict(scene_hypothesis='Back object.',additions=[plan()['additions'][0]])
        item=proposal()['objects'][0];item['center']=[0.,0.,-3.]
        candidates,_,_=build_states(observed,p,[item])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.png';erp_map(candidates,90,(768,768),path)
            with Image.open(path) as image:pixels=np.array(image)
            orange=np.all(pixels==[232,140,19],axis=-1)
            self.assertTrue(orange[:, :80].any() or orange[:, -80:].any())
            self.assertFalse(orange[40:550,500:700].any())

    def test_fake_model_exports_files_and_colored_visuals_without_mutating_inputs(self):
        observed=state();original=deepcopy(observed)
        case=dict(id='case',image='',hfov=90,observed=observed,plan=plan(),prompt='test prompt',source_sha256={})
        class Model:
            def ask(self,image,prompt,path,**options):
                self.assert_rgb=image.mode=='RGB'
                write(path.with_suffix('.json'),proposal())
                return proposal()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image.png';Image.new('RGB',(128,128)).save(image)
            case['image']=str(image)
            model=Model()
            self.assertTrue(run(model,[case],root))
            self.assertTrue(model.assert_rgb)
            self.assertEqual(read(root/'case/G_obs.json'),original)
            self.assertEqual(read(root/'case/summary.json')['unmapped_instances'],1)
            for name in ('scene_3d.html','erp_state_map.png','layout_xz.png','local_projection.png','index.html'):
                self.assertTrue((root/'case'/name).exists())
            html=(root/'case/scene_3d.html').read_text()
            self.assertIn('hypothesized',html)
            self.assertIn('unmapped candidate',html)
            with Image.open(root/'case/erp_state_map.png') as image:self.assertEqual(image.size,(1200,600))
        self.assertEqual(observed,original)


if __name__=='__main__':unittest.main()
