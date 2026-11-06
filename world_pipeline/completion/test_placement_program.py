"""CPU geometry properties and real run006 regression; not model-quality tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read
from ..oracle_model import tensorize
from .box_layout import frustum_margin
from .placement_program import compile_program
from .semantic_world import make_graph, run
from .test_semantic_world import fixture


def program():
    return dict(room_type='bedroom',scene_hypothesis='A reading area in the unobserved part of the room.',
        groups=[dict(id='reading',region='back',along_m=-1.5,clearance_m=.4,objects=[
            dict(id='chair',category='chair',size=[.82,.98,.76],height_anchor_id='observed_0000',
                 height_alignment='bottom',height_offset_m=0.,
                 description='chair',reason='Reading area seating.'),
            dict(id='table',category='table',size=[.68,.72,.75],anchor_id='chair',direction='right',gap_m=.2,
                 vertical_alignment='bottom',
                 description='side table',reason='A table beside the chair.'),
            dict(id='picture',category='picture',size=[.48,.62,.58],anchor_id='chair',direction='above',gap_m=.3,
                 description='art',reason='Art above the reading area.')])])


class PlacementProgramTests(unittest.TestCase):
    def test_height_translation_follows_observation_not_a_fixed_floor(self):
        observed,_,layout=fixture();proposal=program();original=deepcopy(layout)
        first,report=compile_program(proposal,observed,layout,90.,8)
        shifted=deepcopy(layout)
        for box in shifted['objects']:box['center'][1]+=2.37
        second,_=compile_program(proposal,observed,shifted,90.,8)
        for a,b in zip(first['additions'],second['additions']):
            np.testing.assert_allclose(np.array(b['center'])-a['center'],[0.,2.37,0.],atol=1e-12)
            self.assertEqual(a['size'],b['size'])
        self.assertEqual(layout,original)
        self.assertEqual(report['height_source'],'observed_camera_y_aabb_prior_not_measured_floor')

    def test_root_reference_edges_and_signed_offset(self):
        observed,_,layout=fixture()
        for edge,factor in (('bottom',-.5),('center',0.),('top',.5)):
            proposal=program();root=proposal['groups'][0]['objects'][0]
            root.update(height_alignment=edge,height_offset_m=-.17)
            answer,_=compile_program(proposal,observed,layout,90.,8)
            self.assertAlmostEqual(answer['additions'][0]['center'][1],-.7+factor*(1.-.98)-.17)

    def test_height_references_must_be_observed_and_unambiguous(self):
        observed,_,layout=fixture()
        for bad in ('missing','new','non_observed_layout','absolute','unknown_edge','nan','child_alignment'):
            with self.subTest(bad=bad):
                proposal=program();root,child=proposal['groups'][0]['objects'][:2]
                if bad=='missing':root.pop('height_anchor_id')
                elif bad=='new':root['height_anchor_id']='table'
                elif bad=='non_observed_layout':root['height_anchor_id']='wardrobe'
                elif bad=='absolute':root['center_y_m']=.35
                elif bad=='unknown_edge':root['height_alignment']='floor'
                elif bad=='nan':root['height_offset_m']=float('nan')
                else:child.pop('vertical_alignment')
                with self.assertRaises(ValueError):compile_program(proposal,observed,layout,90.,8)

    def test_run007_furniture_heights_use_same_explicit_camera_y_edge(self):
        # Actual observed bed AABB and run007 proposed sizes; this is a compiler
        # regression, not a new Qwen result or measured floor reconstruction.
        observed,_,layout=fixture();layout=deepcopy(layout)
        layout['objects'][0].update(center=[.5216971635818481,-1.100923627614975,3.3491668701171875],
            size=[2.4022305011749268,1.481960117816925,2.3960113525390625])
        proposal=program();group=proposal['groups'][0]
        group.update(region='right',along_m=1.8,clearance_m=.5)
        chair,table,lamp=group['objects']
        chair.update(size=[.65,.9,.45])
        table.update(size=[.7,.6,.5],direction='left')
        lamp.update(category='lighting',size=[.3,.4,.3],anchor_id='table',gap_m=.1)
        answer,report=compile_program(proposal,observed,layout,90.,8)
        chair,table,lamp=answer['additions']
        self.assertAlmostEqual(chair['center'][1],-1.3919036865234375)
        self.assertAlmostEqual(table['center'][1],-1.5419036865234375)
        self.assertAlmostEqual(lamp['center'][1],-.9419036865234375)
        self.assertAlmostEqual(lamp['center'][1]-.2-(table['center'][1]+.3),.1)
        # The declared above gap remains .1m; no silent on-top/support repair.
        self.assertEqual(report['groups'][0]['translation'][1],0.)

    def test_run006_sizes_can_form_separated_boxes_without_changing_observation(self):
        observed,_,layout=fixture(); proposal=program(); original=deepcopy(proposal)
        answer,report=compile_program(proposal,observed,layout,90.,8)
        boxes={o['id']:o for o in answer['additions']}
        chair,table,picture=[boxes[k]for k in ('chair','table','picture')]
        self.assertAlmostEqual(table['center'][0]-chair['center'][0],(.82+.68)/2+.2)
        self.assertAlmostEqual(picture['center'][1]-chair['center'][1],(.98+.62)/2+.3)
        self.assertAlmostEqual(chair['center'][1]-.98/2,-.7-1./2)
        self.assertAlmostEqual(table['center'][1]-.72/2,chair['center'][1]-.98/2)
        self.assertEqual([o['size'] for o in answer['additions']],[o['size']for o in original['groups'][0]['objects']])
        self.assertEqual(report['overlap_report']['pairs'],[])
        self.assertFalse(report['floor_snapping'])
        self.assertEqual(proposal,original)

    def test_whole_group_stays_outside_for_varied_fov_sizes_and_regions(self):
        observed,_,layout=fixture();rng=np.random.default_rng(3407)
        for hfov in (30.,60.,90.,120.,160.):
            for region in ('left','right','back'):
                for _ in range(8):
                    proposal=program();g=proposal['groups'][0]
                    g.update(region=region,along_m=float(rng.uniform(0,4)),clearance_m=float(rng.uniform(.01,1)))
                    for o in g['objects']:o['size']=rng.uniform(.1,4,3).tolist()
                    answer,_=compile_program(proposal,observed,layout,hfov,8)
                    tangent=np.tan(np.deg2rad(hfov)/2)
                    for box in answer['additions']:
                        self.assertLess(frustum_margin(np.array(box['center']),np.array(box['size']),tangent,tangent),0)
                        # Independent 8-corner half-space check, including large
                        # child boxes whose CENTER alone would be insufficient.
                        corners=np.array(box['center'])+np.array([[x,y,z]for x in (-1,1)for y in (-1,1)for z in (-1,1)])*box['size']/2
                        if region=='left': self.assertTrue(np.all(corners[:,0]+tangent*corners[:,2]<0))
                        elif region=='right': self.assertTrue(np.all(-corners[:,0]+tangent*corners[:,2]<0))
                        else:self.assertTrue(np.all(corners[:,2]<0))

    def test_invalid_references_sizes_and_capacity_do_not_drop_objects(self):
        observed,_,layout=fixture()
        for kind in ('reference','reference_type','duplicate','size','gap','capacity','center'):
            with self.subTest(kind=kind):
                proposal=program();items=proposal['groups'][0]['objects']; capacity=8
                if kind=='reference':items[1]['anchor_id']='observed_0000'
                elif kind=='reference_type':items[1]['anchor_id']=['chair']
                elif kind=='duplicate':items[1]['id']='chair'
                elif kind=='size':items[0]['size']=[0,1,1]
                elif kind=='gap':items[1]['gap_m']=-.1
                elif kind=='capacity':capacity=2
                else:items[0]['center']=[0,0,3]
                with self.assertRaises(ValueError):compile_program(proposal,observed,layout,90.,capacity)

    def test_overlapping_siblings_are_reported_not_pushed_or_deleted(self):
        observed,_,layout=fixture();proposal=program()
        duplicate=deepcopy(proposal['groups'][0]['objects'][1]);duplicate['id']='another_table'
        proposal['groups'][0]['objects'].append(duplicate)
        answer,report=compile_program(proposal,observed,layout,90.,8)
        self.assertEqual(len(answer['additions']),4)
        self.assertTrue(any({p['a'],p['b']}=={'table','another_table'}for p in report['overlap_report']['pairs']))

    def test_one_call_real_encoder_and_observed_state_preserved(self):
        observed,_,layout=fixture();state=make_graph(observed,layout,'test','coarse','observed')
        calls=[]
        class Model:
            def ask(self,image,prompt,path,**kwargs):
                calls.append(path.stem)
                return program()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            summary=run(Model(),Image.new('RGB',(768,768)),90.,output,'test',reused=observed,
                        fixed_observed_state=state,completion_mode='program')
            full=read(output/'G_full.json')
            self.assertEqual(calls,['placement_program'])
            self.assertEqual(full['objects'][:2],state['objects'])
            self.assertEqual(summary['outside_fov_verified'],3)
            self.assertEqual(summary['suspected_aabb_overlap_pairs'],0)
            self.assertFalse(summary['scene_semantics_verified'])
            _,geometry,mask=tensorize(full,max_objects=32)
            self.assertEqual(int(mask.sum()),6)
            self.assertTrue(np.isfinite(geometry.numpy()).all())

    def test_actual_run006_overlap_is_visible_to_read_only_diagnostic(self):
        from .placement_program import overlap_report
        path=Path(__file__).parent/'testdata/scene_boxes_box_world_run006_icl.json'
        answer=read(path)
        report=overlap_report(answer['additions'],dict(objects=[]))
        pair=next(p for p in report['pairs']if {p['a'],p['b']}=={'new_0000','new_0001'})
        self.assertAlmostEqual(pair['fraction_of_smaller_aabb'],.7063725490196073)

    def test_zero_fov_failures_do_not_hide_overlapping_new_objects(self):
        observed,_,layout=fixture();state=make_graph(observed,layout,'test','coarse','observed')
        proposal=program();duplicate=deepcopy(proposal['groups'][0]['objects'][1]);duplicate['id']='another_table'
        proposal['groups'][0]['objects'].append(duplicate)
        class Model:
            def ask(self,*args,**kwargs):return proposal
        with tempfile.TemporaryDirectory() as tmp:
            summary=run(Model(),Image.new('RGB',(768,768)),90.,Path(tmp),'test',reused=observed,
                        fixed_observed_state=state,completion_mode='program')
            self.assertEqual(summary['outside_fov_failed'],0)
            self.assertGreater(summary['suspected_aabb_overlap_pairs'],0)
            self.assertEqual(summary['quality_status'],'needs_aabb_overlap_review')
            self.assertEqual(summary['hypothesized'],4)


if __name__=='__main__':
    unittest.main()
