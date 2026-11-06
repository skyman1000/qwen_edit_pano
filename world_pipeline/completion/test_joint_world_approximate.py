"""Real repaired_002 regressions and approximate-layout invariants."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from ..common import ROOT,read,write,sha
from . import joint_world_model as model,joint_world_geometry as geometry
from . import test_joint_world as fixture
from .joint_world_acceptance import evaluate
from .joint_world_initialization import instantiate
from .joint_world_repair import cached_geometry
from .joint_world_replay import case_for

RUN=ROOT/'qwen_edit_pano/outputs/joint_world_repaired_002'
LIVING='X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c'
BATH='x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6'
KITCHEN='8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'


def saved(sid,index):
    case,obs=case_for(sid);folder=RUN/'scenes'/sid/'joint_completion'/f'world_{index:02d}'
    w=read(folder/'semantic_world.json');dims=read(folder/'dimension_stage.json')['dimensions']
    raw=read(folder/'continuous_placement.json');plane=read(folder/'support_plane.json')
    layout=geometry.convert(w,obs,dims,raw['origins'],raw['poses'],case['hfov'])
    return case,obs,w,dims,plane,layout


class ApproximateTests(unittest.TestCase):
    setUp=fixture.JointWorldTests.setUp
    compiled=fixture.JointWorldTests.compiled

    def test_actual_unmeasured_wall_support_is_assumption_not_automatic_rejection(self):
        case,obs,w,dims,plane,layout=saved(BATH,2)
        approximate=geometry.compile_layout(case,obs,w,layout,plane)
        strict=geometry.compile_layout(case,obs,w,layout,plane,policy='strict')
        self.assertTrue(approximate['review']['geometry_contract_passed'])
        self.assertFalse(strict['review']['geometry_contract_passed'])
        self.assertEqual(len(approximate['review']['environment_support_assumptions']),3)
        self.assertFalse(approximate['review']['physical_geometry_verified'])
        self.assertEqual({o['id'] for o in approximate['items']},set(dims))

    def test_real_cartesian_side_projection_is_not_reinterpreted_as_rear_ray(self):
        case,obs,w,dims,plane,layout=saved(LIVING,3)
        boxes,_,certificates,_=instantiate(w,layout,obs,plane,case['hfov'])
        origin=layout['g0']['origin_initialization'];self.assertLess(origin['feasible_center_xz_m'][1],0.)
        self.assertEqual(origin['region'],'left');self.assertEqual(certificates['g0']['kind'],'left')
        self.assertAlmostEqual(certificates['g0']['used_range_m'],layout['g0']['range_m'])
        self.assertLess(certificates['g0']['used_range_m'],5.)
        row=layout['g0']['objects'][0];expected=np.array(origin['feasible_center_xz_m'])+np.array(row['offset_m'])[[0,2]]
        self.assertTrue(np.allclose(np.array(boxes[row['id']]['center'])[[0,2]],expected))

    def test_all_16_saved_metric_worlds_keep_inventory_and_pass_approximate_checks(self):
        from .scene_completion import view_check
        count=0
        for scene in read(RUN/'suite.json')['scenes']:
            p=RUN/'scenes'/scene['id']/'joint_completion'
            for path in p.glob('world_*/continuous_placement.json'):
                index=int(path.parent.name.split('_')[-1]);case,obs,w,dims,plane,layout=saved(scene['id'],index)
                solved=geometry.compile_layout(case,obs,w,layout,plane)
                self.assertTrue(solved['review']['geometry_contract_passed'],(scene['id'],index,solved['review']))
                self.assertEqual({o['id'] for o in solved['items']},set(dims))
                for group in layout.values():
                    for row in group['objects']:self.assertEqual(row['intrinsic_size_m'],dims[row['id']]['full_size_m'])
                for obj in solved['items']:
                    check=view_check(obj['center'],obj['size'],case['hfov'],case['image_size'])
                    self.assertFalse(check['intersects_input_box_proxy']);self.assertFalse(check['camera_inside_box_proxy'])
                count+=1
        self.assertEqual(count,16)

    def test_valid_proposal_conditioning_keeps_original_mass_visible_and_uses_kitchen_alternative(self):
        p=RUN/'scenes'/KITCHEN/'joint_completion';d=read(p/'domain.json');tables=read(p/'empirical_tables.json')
        raw=model.compile_model(d['hypotheses'],d['programs'],tables);before=deepcopy(raw)
        conditioned,coverage=model.available_proposal_model(raw)
        self.assertEqual(raw,before);self.assertGreater(coverage['excluded_reference_mass'],.5)
        worlds=model.sample_worlds(conditioned,3407,100)
        self.assertTrue(all(not w['proposal_failed'] and w['objects'] for w in worlds))
        self.assertEqual({w['hypothesis']['id'] for w in worlds},{'h1'})
        self.assertFalse(coverage['real_world_posterior_claimed'])

    def test_genuine_empty_world_is_not_removed_by_available_support_conditioning(self):
        conditioned,coverage=model.available_proposal_model(self.compiled(failed=True))
        worlds=model.sample_worlds(conditioned,51,100)
        self.assertTrue(any(not w['objects'] for w in worlds));self.assertTrue(any(w['objects'] for w in worlds))
        self.assertFalse(any(w['proposal_failed'] for w in worlds));self.assertGreater(coverage['excluded_reference_mass'],0.)

    def test_all_failed_proposals_still_block_instead_of_becoming_empty(self):
        raw=self.compiled()
        for p in raw['furniture_states'].values():p['proposal_failed']=True
        with self.assertRaisesRegex(ValueError,'All semantic proposals unresolved'):model.available_proposal_model(raw)

    def numeric_fixture(self,separation):
        items=[dict(id='a',center=[0.,0.,0.],size=[1.,1.,1.]),dict(id='b',center=[separation,0.,0.],size=[1.,1.,1.])]
        solved=dict(items=items,review=dict(hard_violations=0,program_solver=dict(final_overlaps=[dict(a='a',b='b')])))
        world=dict(objects=[dict(id=oid,attachment=dict(kind='free',host=None)) for oid in ('a','b')])
        return solved,world

    def test_small_AABB_uncertainty_is_warned_but_major_interpenetration_rejected(self):
        solved,world=self.numeric_fixture(.98)
        review=evaluate(solved,dict(objects=[]),world,dict(floor_y_m=None),['remaining_AABB_collisions'])
        self.assertTrue(review['geometry_contract_passed']);self.assertEqual(review['soft_layout_warnings'][0]['code'],'minor_AABB_overlap')
        solved,world=self.numeric_fixture(.4)
        review=evaluate(solved,dict(objects=[]),world,dict(floor_y_m=None),['remaining_AABB_collisions'])
        self.assertFalse(review['geometry_contract_passed'])

    def test_camera_contradiction_and_missing_floor_are_not_relaxed(self):
        solved,world=self.numeric_fixture(2.);solved['review']['program_solver']['final_overlaps']=[]
        solved['review']['hard_violations']=1
        world['objects'][0]['attachment']['kind']='floor'
        review=evaluate(solved,dict(objects=[]),world,dict(floor_y_m=None),[])
        self.assertIn('original_camera_visibility_or_camera_intersection',review['joint_contract_errors'])
        self.assertIn('unresolved_floor:a',review['joint_contract_errors'])

    def test_saved_metric_cache_checks_identity_and_bindings(self):
        sid=LIVING;out=self.base/'run/scenes'/sid/'joint_completion';out.mkdir(parents=True)
        root=out.parents[2];source=RUN/'scenes'/sid/'joint_completion'
        observation=out.parent/'observation/qwen/observer3d';observation.mkdir(parents=True)
        write(observation/'G_obs.json',read(RUN/'scenes'/sid/'observation/qwen/observer3d/G_obs.json'))
        files={str(p.relative_to(RUN)):sha(p) for p in source.glob('world_*/*.json')
            if p.name in ('semantic_world.json','dimension_stage.json','continuous_placement.json','support_plane.json')}
        write(root/'repair_source_bindings.json',dict(source=str(RUN),files=files))
        write(root/'joint_world_run.json',dict(repair_source=str(RUN),repair_source_bindings_sha256=sha(root/'repair_source_bindings.json')))
        world=read(source/'world_01/semantic_world.json');cached=cached_geometry(out,world,1)
        self.assertIsNotNone(cached);self.assertEqual(set(cached[0]),{o['id'] for o in world['objects']})
        changed=deepcopy(world);changed['objects'][0]['variant']='Different semantic instance; cache must not be reused.'
        self.assertIsNone(cached_geometry(out,changed,1))
        write(root/'repair_source_bindings.json',dict(source=str(RUN),files={}))
        with self.assertRaisesRegex(ValueError,'bindings changed'):cached_geometry(out,world,1)


if __name__=='__main__':unittest.main()
