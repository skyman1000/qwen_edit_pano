"""CPU checks for camera/gravity separation, ID relations and support placement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read
from .box_layout import camera_layout
from .planned_completion import complete, select_layout, validate_contents
from .scene_geometry import camera_geometry, ground_center, half_extent, horizontal_axes, observation_report
from .scene_relations import check_relation, ordered_objects, validate_relations
from .semantic_world import make_graph, run
from .test_semantic_world import fixture


def support_plan():
    return dict(room_type='bedroom',scene_hypothesis='Seating with a pillow',
        additions=[dict(id='seat',category='chair',description='reading chair',reason='seating',placement_intent='outside photo'),
                   dict(id='pillow',category='cushion',description='gray pillow',reason='comfort',placement_intent='on seat')],
        relations=[dict(subject='pillow',type='on',target='seat')])


class SceneConsistencyTests(unittest.TestCase):
    def test_above_below_use_camera_y_and_preserve_real_failed_plan(self):
        observed = read(Path('qwen_edit_pano/outputs/box_world_run002/observed_inventory.json'))
        contents = read(Path(__file__).parent/'testdata/room_contents_box_world_run002.json')
        original = deepcopy(contents)
        validate_contents(contents, observed, 8)
        self.assertEqual(contents, original)
        boxes = dict(new_01=dict(center=[0,0,3],size=[.5,.5,.5]),
                     new_02=dict(center=[0,1,3],size=[.2,.2,.2]))
        geometry = camera_geometry([0,.8,.6])
        rel = dict(subject='new_02',type='above',target='new_01')
        self.assertEqual(check_relation(rel,boxes,geometry)['status'], 'satisfied')
        rel['type']='below'
        self.assertEqual(check_relation(rel,boxes,geometry)['status'], 'violated')
        boxes['new_02']['center'][1]=-1
        self.assertEqual(check_relation(rel,boxes,geometry)['status'], 'satisfied')
        # A spatial ordering statement is not an on/support dependency.
        self.assertEqual(ordered_objects(contents), contents['additions'])

    def test_relation_errors_distinguish_unknown_type_from_unknown_ID(self):
        contents = support_plan()
        contents['relations'][0]['type']='touches'
        with self.assertRaisesRegex(ValueError, "Unsupported relation type 'touches'"):
            validate_relations(contents, [])
        contents = support_plan()
        contents['relations'][0]['target']='missing_chair'
        with self.assertRaisesRegex(ValueError, "target ID: 'missing_chair'"):
            validate_relations(contents, [])

    def test_observe_only_requests_geometry_then_stops_before_planning(self):
        observed,_,layout=fixture(); calls=[]
        class Model:
            def ask(self,image,prompt,path,**kwargs):
                calls.append(path.stem)
                if path.stem!='observed_layout': raise AssertionError('Unexpected completion')
                return dict(objects=[dict(id=o['id'],depth_m=o['center'][2],center_uv=[.6,.55],size=o['size']) for o in layout['objects'][:2]])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            summary=run(Model(),Image.new('RGB',(768,768)),90,output,'test',reused=observed,observe_only=True)
            self.assertEqual(calls,['observed_layout'])
            self.assertFalse(summary['completion_executed'])
            self.assertTrue((output/'observation_geometry_report.json').is_file())
            self.assertFalse((output/'G_full.json').exists())
            full=read(output/'G_obs.json')
            np.testing.assert_allclose(full['objects'][0]['center_local_m']['value'],[.6,-.3,3.],atol=1e-12)

    def test_tilted_gravity_plane_support_uses_full_camera_AABB(self):
        geometry=camera_geometry([.1,.98,.15],1.5)
        n=np.array(geometry['up_local']); size=[.6,.8,.5]
        center=ground_center(size,0,3,90,geometry)
        self.assertAlmostEqual(np.dot(n,center)+1.5-half_extent(size,n),0,places=12)
        right,up,forward=horizontal_axes(geometry)
        np.testing.assert_allclose(np.array([right,up,forward])@np.array([right,up,forward]).T,np.eye(3),atol=1e-12)
        self.assertFalse(geometry['levelled']); self.assertFalse(geometry['floor_verified'])
        self.assertEqual(geometry['up_source'],'user_supplied_direction')

    def test_floor_prior_never_becomes_a_confirmed_collision(self):
        observed,_,layout=fixture()
        state=make_graph(observed,layout,'test','coarse','observed'); original=deepcopy(state)
        report=observation_report(state,90,(768,768),camera_geometry())
        self.assertFalse(report['camera_geometry']['gravity_verified'])
        self.assertTrue(all(not o['floor_collision_verified'] for o in report['objects']))
        self.assertEqual(state,original)

    def test_projected_full_center_is_not_forced_to_visible_fragment_center(self):
        observed,_,_=fixture(); observed=observed[:1]
        answer=dict(objects=[dict(id=observed[0]['id'],depth_m=3.,center_uv=[.8,.4],size=[2.,1.,2.])])
        layout,report=camera_layout(answer,observed,90,(768,768))
        np.testing.assert_allclose(layout['objects'][0]['center'],[1.8,.6,3.],atol=1e-12)
        self.assertEqual(report['explicit_projected_center_ids'],[observed[0]['id']])
        self.assertEqual(report['visible_bbox_center_proxy_ids'],[])

    def test_on_relation_references_and_support_cycles_are_checked(self):
        plan=support_plan(); validate_contents(plan,[],8)
        plan['relations'][0]['target']='table_only_in_prose'
        with self.assertRaisesRegex(ValueError,'existing/planned target ID'):
            validate_relations(plan,[])
        plan=support_plan(); plan['relations'].append(dict(subject='seat',type='on',target='pillow'))
        with self.assertRaisesRegex(ValueError,'Support dependency cycle'):
            validate_relations(plan,[])

    def test_near_is_scored_and_face_to_is_explicitly_unverified(self):
        obj=dict(id='a',category='chair',size=[.6,.8,.6])
        contents=dict(additions=[obj],relations=[dict(subject='a',type='near',target='old',max_gap_m=.5)])
        choices={'a':[dict(candidate_id='far',center=[3,-1,0]),dict(candidate_id='near',center=[-3,-1,0])]}
        old=dict(objects=[dict(id='old',center=[-2,-1,0],size=[.6,.8,.6])])
        answer,_=select_layout(contents,choices,dict(placements=[dict(id='a',candidate_ids=['far','near'])]),observed_layout=old)
        self.assertEqual(answer['additions'][0]['center'],[-3,-1,0])
        face=check_relation(dict(subject='a',type='face_to',target='old'),{'a':dict(center=[0,0,0],size=[1,1,1]),'old':old['objects'][0]},camera_geometry())
        self.assertEqual(face['status'],'unverified')

    def test_complete_support_pipeline_uses_surface_not_parent_AABB_top(self):
        observed,_,layout=fixture(); state=make_graph(observed,layout,'test','coarse','observed')
        geometry=camera_geometry([.04,.99,.1])
        class Model:
            def ask(self,image,prompt,path,**kwargs):
                if path.stem=='room_contents': return support_plan()
                if path.stem=='room_geometry':
                    return dict(objects=[dict(id='seat',size=[.6,1.2,.6],bottom_height_m=0),
                                         dict(id='pillow',size=[.3,.3,.3],bottom_height_m=0)],
                                support_surfaces=[dict(id='seat',offset_m=-.15)])
                if path.stem=='room_placement':
                    candidates=read(path.parent/'placement_candidates.json')['candidates']
                    return dict(placements=[dict(id=oid,candidate_ids=[p['candidate_id'] for p in options[:5]]) for oid,options in candidates.items()])
                raise AssertionError(path.stem)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            summary=run(Model(),Image.new('RGB',(768,768)),90,output,'test',reused=observed,
                        fixed_observed_state=state,completion_mode='planned',up_local=geometry['up_local'])
            report=read(output/'relation_diagnostic.json')
            self.assertEqual(report['satisfied'],1); self.assertEqual(report['violated'],0)
            full=read(output/'G_full.json'); seat,pillow=full['objects'][-2:]
            n=np.array(geometry['up_local']); delta=np.array(pillow['center_local_m']['value'])-seat['center_local_m']['value']
            self.assertAlmostEqual(np.dot(delta,n),-.15+half_extent([.3,.3,.3],n))
            self.assertEqual(full['objects'][:2],state['objects'])
            self.assertEqual(full['relations'],support_plan()['relations'])
            self.assertEqual(full['object_commitment']['authoritative_object_ids'],[o['track_id'] for o in full['objects']])
            from ..oracle_model import tensorize
            before=tensorize(full,max_objects=160)
            no_metadata=deepcopy(full);no_metadata['relations']=[];no_metadata.pop('object_commitment')
            after=tensorize(no_metadata,max_objects=160)
            for a,b in zip(before,after): self.assertTrue((a==b).all())


if __name__=='__main__': unittest.main()
