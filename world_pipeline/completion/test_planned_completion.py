"""CPU tests of the planner/placer boundary, not of Qwen's scene quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read
from .box_layout import frustum_margin
from .planned_completion import candidate_positions, complete_geometry, overlaps, select_layout, validate_contents, validate_geometry
from .semantic_world import make_graph, run
from .test_semantic_world import fixture


def contents():
    return dict(room_type='bedroom', scene_hypothesis='A bedroom with additional storage and seating.',
                additions=[dict(id=oid, category=category, description=category,
                    reason='room function', placement_intent='near existing furniture',
                    size=[.6, .8, .6], bottom_height_m=0)
                    for oid, category in [('new_01', 'chair'), ('new_02', 'cabinet')]])


class PlannedCompletionTests(unittest.TestCase):
    def test_joint_boxes_preserve_xyz_without_floor_or_candidate_conversion(self):
        observed, _, layout = fixture()
        state = make_graph(observed,layout,'test','coarse','observed')
        answer = dict(room_type='bedroom',scene_hypothesis='Storage and elevated art in the unseen room.',
                      additions=[dict(id='new_01',category='cabinet',center=[-2.,-.6,-2.],size=[1.2,2.,.6],
                                      description='storage cabinet',reason='bedroom storage outside the photo',unseen_reason='outside_fov'),
                                 dict(id='new_02',category='picture',center=[-2.,.9,-2.],size=[.5,.6,.2],
                                      description='wall art',reason='art placed above the cabinet',unseen_reason='outside_fov')])
        original = deepcopy(answer); calls=[]
        class Model:
            def ask(self,image,prompt,path,**kwargs):
                calls.append(path.stem)
                if path.stem != 'scene_boxes': raise AssertionError('Unexpected stage: '+path.stem)
                return answer
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(Model(),Image.new('RGB',(768,768)),90,output,'test',reused=observed,
                          fixed_observed_state=state,completion_mode='boxes')
            full=read(output/'G_full.json')
            self.assertEqual(calls,['scene_boxes'])
            self.assertEqual(full['objects'][:2],state['objects'])
            for obj,expected in zip(full['objects'][2:],original['additions']):
                self.assertEqual(obj['center_local_m']['value'],expected['center'])
                self.assertEqual(obj['size_aabb_local_m']['value'],expected['size'])
            self.assertEqual(summary['hypothesized'],2)
            self.assertEqual(summary['outside_fov_verified'],2)
            self.assertFalse(read(output/'completion_conversion.json')['candidate_grid_used'])
            self.assertEqual(answer,original)

    def test_joint_boxes_allow_unverified_occlusion_and_reject_invalid_geometry(self):
        from .box_completion import complete
        observed,_,layout=fixture()
        answer=dict(room_type='bedroom',scene_hypothesis='A small object hidden within the camera view.',
                    additions=[dict(id='hidden',category='cabinet',center=[0,-.8,4],size=[.5,.7,.5],
                                    description='cabinet',reason='hypothesized behind a visible foreground wall',unseen_reason='occluded')])
        value,_=complete(lambda *a:answer,observed,layout,90,8)
        self.assertEqual(value['additions'][0]['center'],[0,-.8,4])
        self.assertEqual(value['additions'][0]['unseen_reason'],'occluded')
        answer['additions'][0]['size']=[.5,0,.5]
        with self.assertRaisesRegex(ValueError,'positive full camera-AABB'):
            complete(lambda *a:answer,observed,layout,90,8)

    def test_reuse_failed_contents_continues_geometry_without_replanning(self):
        source = Path('qwen_edit_pano/outputs/box_world_run002')
        observed = read(source/'observed_inventory.json')
        state = read(source/'G_obs.json')
        contents = read(Path(__file__).parent/'testdata/room_contents_box_world_run002.json')
        original = deepcopy(contents)
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem == 'room_geometry':
                    return dict(objects=[dict(id='new_01',size=[.6,.7,.6],bottom_height_m=0),
                                         dict(id='new_02',size=[.5,.7,.2],bottom_height_m=1.2)])
                if path.stem == 'room_placement':
                    candidates = read(path.parent/'placement_candidates.json')['candidates']
                    return dict(placements=[dict(id=oid,candidate_ids=[p['candidate_id'] for p in options[:5]])
                                            for oid,options in candidates.items()])
                raise AssertionError('Reused contents must not trigger '+path.stem)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB',(768,768)),90,output,state['vocabulary_sha256'],
                          reused=observed,fixed_observed_state=state,completion_mode='planned',cached_contents=contents)
            self.assertEqual(calls, ['room_geometry','room_placement'])
            self.assertTrue(summary['content_plan_reused'])
            self.assertEqual(summary['hypothesized'],2)
            full = read(output/'G_full.json')
            self.assertEqual(full['objects'][:17],state['objects'])
            self.assertEqual(full['relations'],original['relations'])
            self.assertEqual(contents,original)

    def test_empty_model_plan_is_reported_without_fabricating_objects(self):
        observed, _, layout = fixture()
        state = make_graph(observed, layout, 'test', 'coarse', 'observed')
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem != 'room_contents':
                    raise AssertionError('Empty contents must not request geometry or placements')
                return dict(room_type='bedroom', scene_hypothesis='Sparse room hypothesis.', additions=[], relations=[])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB', (768,768)), 90, output, 'test',
                          reused=observed, fixed_observed_state=state, completion_mode='planned')
            self.assertEqual(calls, ['room_contents'])
            self.assertEqual(summary['completion_outcome'], 'no_new_objects')
            self.assertEqual(read(output/'G_full.json')['objects'], state['objects'])

    def test_real_run012_missing_fields_are_estimated_without_rewriting_semantics(self):
        raw = read(Path(__file__).parent/'testdata/semantic_contents_run012.json')
        original = deepcopy(raw)
        semantic = validate_contents(raw, [], 8)
        calls = []
        def ask(prompt, stage, tokens):
            calls.append(stage)
            # Synthetic geometry exercises protocol, not actual room quality.
            return dict(objects=[dict(id=o['id'], size=[.5,1.,.3], bottom_height_m=0,
                                      description='must not overwrite') for o in raw['additions']])
        result = complete_geometry(semantic, dict(objects=[]), 1.5, ask)
        self.assertEqual(calls, ['room_geometry'])
        self.assertEqual(raw, original)
        self.assertEqual(semantic, original)
        for old, new in zip(original['additions'], result['additions']):
            self.assertEqual({k:new[k] for k in old}, old)
            validate_geometry(new)

    def test_missing_and_invalid_geometry_are_distinguished(self):
        with self.assertRaisesRegex(ValueError, 'new_01: missing geometry fields'):
            validate_geometry(dict(id='new_01'))
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            validate_geometry(dict(id='new_01', size=[0,1,1], bottom_height_m=0))
        raw = contents(); raw['additions'][0]['size'] = [-1,1,1]
        with self.assertRaisesRegex(ValueError, 'invalid provided size'):
            complete_geometry(raw, dict(objects=[]), 1.5, lambda *args: self.fail('Do not overwrite invalid sizes'))

    def test_geometry_rejects_missing_ids_and_preserves_supplied_values(self):
        raw = contents()
        del raw['additions'][0]['bottom_height_m']
        def wrong_id(*args):
            return dict(objects=[dict(id='different', size=[1,1,1], bottom_height_m=0)])
        with self.assertRaisesRegex(ValueError, 'every requested ID'):
            complete_geometry(raw, dict(objects=[]), 1.5, wrong_id)
        def changed_size(*args):
            return dict(objects=[dict(id='new_01', size=[1,1,1], bottom_height_m=0)])
        with self.assertRaisesRegex(ValueError, 'attempted to change provided size'):
            complete_geometry(raw, dict(objects=[]), 1.5, changed_size)

    def test_candidates_respect_full_box_and_keep_observation_unchanged(self):
        observed, _, layout = fixture()
        original = deepcopy(layout)
        obj = contents()['additions'][0]
        options = candidate_positions(obj, layout, 90, (768, 768), 1.5)
        self.assertGreater(len(options), 5)
        for option in options:
            box = dict(center=option['center'], size=obj['size'])
            self.assertLess(frustum_margin(np.array(box['center']), np.array(box['size']), 1, 1), 0)
            self.assertFalse(any(overlaps(box, b) for b in layout['objects']))
            self.assertAlmostEqual(box['center'][1]-.4, -1.5)
        self.assertEqual(layout, original)

    def test_joint_choice_uses_alternative_without_deleting_objects(self):
        data = contents()
        candidates = {o['id']: [dict(candidate_id='p0', center=[-2, -.5, 0]),
                                dict(candidate_id='p1', center=[2, -.5, 0])] for o in data['additions']}
        reply = dict(placements=[dict(id=o['id'], candidate_ids=['p0','p1']) for o in data['additions']])
        result, report = select_layout(data, candidates, reply)
        self.assertEqual(len(result['additions']), 2)
        self.assertFalse(overlaps(result['additions'][0], result['additions'][1]))
        self.assertEqual(sorted(p['preference_rank'] for p in report['selected']), [1,2])
        self.assertEqual([o['category'] for o in result['additions']], ['chair','cabinet'])

    def test_infeasible_choices_do_not_delete_or_invent_positions(self):
        data = contents()
        candidates = {o['id']: [dict(candidate_id='p0', center=[-2,-.5,0])] for o in data['additions']}
        reply = dict(placements=[dict(id=o['id'], candidate_ids=['p0']) for o in data['additions']])
        original = deepcopy(data)
        with self.assertRaisesRegex(ValueError, 'No joint layout'):
            select_layout(data, candidates, reply)
        self.assertEqual(data, original)

    def test_empty_plan_is_explicit_not_a_forced_object_count(self):
        data = contents(); data['additions'] = []
        result, report = select_layout(validate_contents(data, [], 8), {}, dict(placements=[]))
        self.assertEqual(result['additions'], [])
        self.assertEqual(report['planned_count'], 0)

    def test_unknown_candidate_missing_id_and_duplicate_ids_rejected(self):
        data = contents()
        candidates = {o['id']: [dict(candidate_id='p0', center=[-2,-.5,0])] for o in data['additions']}
        for rows in ([dict(id='new_01', candidate_ids=['p0'])],
                     [dict(id='new_01', candidate_ids=['p0'])]*2,
                     [dict(id=o['id'], candidate_ids=['invented']) for o in data['additions']]):
            with self.assertRaises(ValueError):
                select_layout(data, candidates, dict(placements=rows))

    def test_complete_pipeline_preserves_local_state_and_encoder_contract(self):
        observed, _, layout = fixture()
        state = make_graph(observed, layout, 'test', 'coarse', 'observed')
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem == 'room_contents':
                    answer = contents()
                    for obj in answer['additions']:
                        del obj['size']; del obj['bottom_height_m']
                    return answer
                if path.stem == 'room_geometry':
                    return dict(objects=[{k:o[k] for k in ('id','size','bottom_height_m')}
                                         for o in contents()['additions']])
                if path.stem == 'room_placement':
                    candidates = read(path.parent/'placement_candidates.json')['candidates']
                    return dict(placements=[dict(id=oid, candidate_ids=[p['candidate_id'] for p in options[:5]])
                                            for oid, options in candidates.items()])
                raise AssertionError('Unexpected model request: '+path.stem)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB', (768,768)), 90, output, 'test',
                          reused=observed, fixed_observed_state=state, completion_mode='planned')
            full = read(output/'G_full.json')
            self.assertEqual(calls, ['room_contents','room_geometry','room_placement'])
            self.assertEqual(full['objects'][:2], state['objects'])
            self.assertEqual(summary['hypothesized'], 2)
            self.assertEqual(summary['completion_outcome'], 'objects_added')
            self.assertEqual(summary['outside_fov_failed'], 0)
            self.assertEqual(summary['completion_withdrawn'], 0)
            self.assertFalse(summary['completion_revision_used'])
            self.assertEqual(full['scene_hypothesis'], contents()['scene_hypothesis'])
            from ..oracle_model import tensorize
            categories, geometry, mask = tensorize(full, max_objects=160)
            self.assertEqual(categories[0, 1:5].tolist(), [11,28,3,7])
            self.assertEqual(mask.sum().item(), 5)


if __name__ == '__main__':
    unittest.main()
