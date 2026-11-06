"""CPU behavioral checks; fake model outputs do not validate Qwen scene quality."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from ..common import read, write
from .box_layout import BoxProblem, camera_layout, directional_layout, frustum_margin, inspect_layout, refine, support_constraints, validate_layout
from .completion_feedback import conflicts, revise_once
from .semantic_world import (RelationValidationError, decode_plan, make_graph, reuse_inventory,
                             layout_prompt, observed_state_layout, run, scene_objects, validated_plan, visible_inventory)


def fixture():
    observed = visible_inventory([
        dict(category='bed', description='bed', bbox_2d=[320, 350, 680, 750]),
        dict(category='lighting', description='lamp', bbox_2d=[700, 200, 820, 380]),
    ], 32)
    answer = dict(room_type='bedroom', additions=[dict(id='wardrobe', category='cabinet',
        description='wardrobe', unseen_reason='outside_fov', reason='storage beside the camera')],
        relations=[dict(a='observed_0000', type='left_of', b='observed_0001')])
    plan = decode_plan(answer, observed, 8)
    layout = dict(room=dict(x_min=-4, x_max=4, floor_y=-1.5, ceiling_y=1.5, z_min=-4, z_max=6),
        objects=[dict(id='observed_0000', center=[0, -.7, 3], size=[2, 1, 2], support='floor'),
                 dict(id='observed_0001', center=[1.6, .8, 3], size=[.4, .7, .4], support='free'),
                 dict(id='wardrobe', center=[.1, -.3, 2.9], size=[1, 2, .6], support='floor')])
    return observed, plan, validate_layout(layout, plan['objects'])


class SemanticWorldTests(unittest.TestCase):
    def test_saved_completion_only_runs_revision_and_preserves_observation(self):
        observed, _, layout = fixture()
        state = make_graph(observed, layout, 'test', 'coarse_hypothesis', 'observed')
        raw = read(Path(__file__).parent/'testdata/semantic_completion_run010.json')
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                return dict(additions=[], withdrawn=[dict(id='new_01', reason='uncertain placement'),
                                                      dict(id='new_03', reason='unsupported fixture')])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB', (768, 768)), 90, output, 'test',
                          reused=observed, fixed_observed_state=state, cached_completion=raw)
            self.assertEqual(calls, ['completion_revision'])
            self.assertEqual(summary['completion_withdrawn'], 2)
            self.assertEqual(summary['completion_conflicts_before'], 2)
            self.assertEqual(summary['completion_conflicts_after'], 0)
            self.assertEqual(read(output/'G_full.json')['objects'][:2], state['objects'])
            self.assertEqual(read(output/'G_full.json')['objects'][2]['track_id'], 'new_02')

    def test_actual_run010_feedback_revises_only_failed_candidates(self):
        raw = read(Path(__file__).parent/'testdata/semantic_completion_run010.json')
        answer, _ = directional_layout(raw, 1.5)
        original = deepcopy(answer)
        self.assertEqual([o['id'] for o in conflicts(answer, 90, (768, 768))], ['new_01', 'new_03'])
        calls = []
        def ask(prompt, stage, tokens):
            calls.append(stage)
            self.assertIn('COMPUTED GEOMETRY FEEDBACK', prompt)
            chair = deepcopy(raw['additions'][0])
            chair.update(azimuth_deg=-65, description='chair outside left edge')
            return dict(additions=[chair], withdrawn=[dict(id='new_03', reason='Unsupported visible-wall fixture')])
        result, report = revise_once(answer, 'context', ask, lambda a: scene_objects(a, [], 8), 90, (768, 768), 1.5)
        self.assertEqual(calls, ['completion_revision'])
        self.assertEqual(report['after'], [])
        self.assertEqual(result['additions'][1], original['additions'][1])
        self.assertEqual([o['id'] for o in result['additions']], ['new_01', 'new_02'])
        self.assertEqual(answer, original)

    def test_revision_does_not_loop_or_hide_unresolved_candidates(self):
        raw = read(Path(__file__).parent/'testdata/semantic_completion_run010.json')
        answer, _ = directional_layout(raw, 1.5)
        calls = []
        def ask(*args):
            calls.append(1)
            return dict(additions=[raw['additions'][0], raw['additions'][2]], withdrawn=[])
        result, report = revise_once(answer, '', ask, lambda a: scene_objects(a, [], 8), 90, (768, 768), 1.5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(report['after']), 2)
        self.assertEqual(len(result['additions']), 3)

    def test_invalid_revision_keeps_original_and_reports_error(self):
        raw = read(Path(__file__).parent/'testdata/semantic_completion_run010.json')
        answer, _ = directional_layout(raw, 1.5)
        def ask(*args):
            return dict(additions=raw['additions'], withdrawn=[])
        result, report = revise_once(answer, '', ask, lambda a: scene_objects(a, [], 8), 90, (768, 768), 1.5)
        self.assertEqual(result, answer)
        self.assertIsNotNone(report['revision_error'])
        self.assertEqual(len(report['after']), 2)

    def test_valid_completion_does_not_request_revision(self):
        raw = read(Path(__file__).parent/'testdata/semantic_completion_run010.json')
        raw['additions'] = [raw['additions'][1]]
        answer, _ = directional_layout(raw, 1.5)
        def ask(*args):
            self.fail('Valid geometry must not trigger revision')
        result, report = revise_once(answer, '', ask, lambda a: None, 90, (768, 768), 1.5)
        self.assertFalse(report['used'])
        self.assertEqual(result, answer)

    def test_continuous_azimuth_is_not_snapped_to_three_rays(self):
        proposal = dict(additions=[dict(id='cabinet', azimuth_deg=125, distance_m=3.,
                                       bottom_height_m=0., size=[1., 2., .5])])
        original = deepcopy(proposal)
        converted, report = directional_layout(proposal, 1.5)
        center = converted['additions'][0]['center']
        self.assertAlmostEqual(np.degrees(np.arctan2(center[0], center[2])), 125)
        self.assertAlmostEqual(np.hypot(center[0], center[2]), 3)
        self.assertAlmostEqual(center[1], -.5)
        self.assertEqual(report['legacy_region_ids'], [])
        self.assertEqual(proposal, original)
        for invalid in (True, float('nan'), 181, 'left'):
            proposal['additions'][0]['azimuth_deg'] = invalid
            with self.assertRaises(ValueError):
                directional_layout(proposal, 1.5)

    def test_coverage_exposes_tiny_box_even_when_overlap_passes(self):
        observed, _, layout = fixture()
        observed = observed[:1]
        observed[0]['bbox_2d'] = [.25, .25, .75, .75]
        layout['objects'] = [dict(id=observed[0]['id'], center=[0, 0, 3], size=[.1, .1, .1])]
        original = deepcopy(layout)
        result = inspect_layout(layout, observed, 90, (768, 768))['objects'][0]
        self.assertTrue(result['projection_overlaps_evidence'])
        self.assertLess(result['visible_bbox_coverage'], .01)
        self.assertEqual(layout, original)

    def test_directional_hypothesis_compiles_side_back_and_floor_height(self):
        answer = dict(room_type='bedroom', additions=[dict(id=region, category='cabinet',
            description='cabinet', reason='room storage', camera_region=region,
            distance_m=3., bottom_height_m=0., size=[1., 2., .5])
            for region in ('camera-left', 'camera-right', 'behind-camera')])
        original = deepcopy(answer)
        compiled, report = directional_layout(answer, 1.5)
        self.assertEqual([o['center'] for o in compiled['additions']], [[-3., -.5, 0.], [3., -.5, 0.], [0., -.5, -3.]])
        for obj in compiled['additions']:
            self.assertEqual(obj['center'][1]-obj['size'][1]/2, -1.5)
            self.assertEqual(obj['unseen_reason'], 'outside_fov')
        self.assertEqual(answer, original)
        self.assertEqual(report['region_bearings_deg']['camera-left'], -90)

    def test_directional_compilation_does_not_hide_large_box_entering_view(self):
        answer = dict(room_type='bedroom', additions=[dict(id='new_cabinet', category='cabinet',
            description='cabinet', reason='storage', camera_region='camera-left',
            distance_m=.2, bottom_height_m=0., size=[2., 2., 2.])])
        compiled, _ = directional_layout(answer, 1.5)
        self.assertEqual(compiled['additions'][0]['center'], [-.2, -.5, 0.])
        obj = compiled['additions'][0]
        self.assertGreater(frustum_margin(np.array(obj['center']), np.array(obj['size']), 1, 1), 0)

    def test_directional_model_ranges_and_elevated_bottom_are_preserved(self):
        answer = dict(additions=[dict(id='lamp', camera_region='camera-right', distance_m=2.4,
                                    bottom_height_m=.7, size=[.3, .5, .3])])
        compiled, _ = directional_layout(answer, 1.5)
        np.testing.assert_allclose(compiled['additions'][0]['center'], [2.4, -.55, 0.])
        for key, bad in (('camera_region', 'left_of_bed'), ('distance_m', 0), ('bottom_height_m', -.1), ('size', [1, 0, 1])):
            invalid = deepcopy(answer); invalid['additions'][0][key] = bad
            with self.assertRaises(ValueError):
                directional_layout(invalid, 1.5)

    def test_actual_run006_array_and_wrapper_have_identical_geometry(self):
        data = read(Path(__file__).parent/'testdata/semantic_layout_run006.json')
        answer, objects = data['answer'], data['scene']['objects']
        original = deepcopy(answer)
        self.assertEqual(len(answer), 21)
        array_layout, array_report = camera_layout(answer, objects, 90, (768, 768))
        wrapper_layout, _ = camera_layout(dict(objects=answer), objects, 90, (768, 768))
        self.assertEqual(array_layout, wrapper_layout)
        self.assertEqual(answer, original)
        self.assertEqual(array_report['response_container'], 'array')
        self.assertEqual(len(array_layout['objects']), 21)
        self.assertEqual([o['id'] for o in array_layout['objects']], [o['id'] for o in objects])
        for normalized, raw in zip(array_layout['objects'], answer):
            self.assertEqual(normalized['size'], raw['size'])

    def test_array_acceptance_still_rejects_missing_or_duplicate_ids(self):
        data = read(Path(__file__).parent/'testdata/semantic_layout_run006.json')
        answer, objects = data['answer'], data['scene']['objects']
        with self.assertRaisesRegex(ValueError, '20 entries for 21'):
            camera_layout(answer[:-1], objects, 90, (768, 768))
        duplicate = deepcopy(answer); duplicate[-1]['id'] = duplicate[0]['id']
        with self.assertRaisesRegex(ValueError, 'IDs differ'):
            camera_layout(duplicate, objects, 90, (768, 768))
        invalid = deepcopy(answer); invalid[0]['size'][0] = 0
        with self.assertRaisesRegex(ValueError, 'Positive FULL'):
            camera_layout(invalid, objects, 90, (768, 768))

    def test_actual_run004_uses_rays_not_invalid_model_room_or_xy(self):
        data = read(Path(__file__).parent/'testdata/semantic_layout_run004.json')
        answer, objects = data['answer'], data['scene']['objects']
        original = deepcopy(answer)
        with self.assertRaisesRegex(ValueError, 'camera origin'):
            validate_layout(answer, objects)
        result, report = camera_layout(answer, objects, 90, (768, 768), 1.5, 2.7)
        self.assertEqual(answer, original)
        self.assertFalse(report['model_room_coordinates_used'])
        self.assertEqual(result['room']['floor_y'], -1.5)
        self.assertAlmostEqual(result['room']['ceiling_y'], 1.2)
        self.assertEqual(len(report['reused_legacy_depth_ids']), 17)
        for box, obj in zip(result['objects'], objects):
            self.assertEqual(box['size'], next(o['size'] for o in answer['objects'] if o['id']==obj['id']))
            if obj['role'] == 'observed':
                x, y, z = box['center']; target = np.array(obj['bbox_2d'])
                np.testing.assert_allclose([.5+x/(2*z), .5-y/(2*z)], (target[:2]+target[2:])/2, atol=1e-12)
        # The bedside lamp at image-left of its mate stays to that mate's left.
        by_id = {o['id']: o for o in result['objects']}
        self.assertLess(by_id['observed_0011']['center'][0]/by_id['observed_0011']['center'][2],
                        by_id['observed_0010']['center'][0]/by_id['observed_0010']['center'][2])

    def test_new_depth_only_contract_and_non_square_camera(self):
        _, plan, layout = fixture()
        answer = deepcopy(layout); answer.pop('room')
        for box in answer['objects'][:2]:
            box['depth_m'] = box.pop('center')[2]
        result, report = camera_layout(answer, plan['objects'], 90, (800, 400))
        self.assertEqual(report['reused_legacy_depth_ids'], [])
        x, y, z = result['objects'][0]['center']
        bbox = plan['objects'][0]['bbox_2d']
        self.assertAlmostEqual(.5-y/z, (bbox[1]+bbox[3])/2)
        answer['objects'][0]['depth_m'] = 0
        with self.assertRaisesRegex(ValueError, 'depth'):
            camera_layout(answer, plan['objects'], 90, (800, 400))

    def test_finish_saved_layout_never_calls_a_model(self):
        observed, plan, layout = fixture()
        answer = dict(room_type=plan['room_type'], additions=[{k:v for k,v in plan['objects'][-1].items() if k!='role'}])
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(None, Image.new('RGB', (200, 160)), 90, Path(tmp), 'test',
                          reused=observed, cached_plan=answer, cached_layout=layout)
            self.assertTrue(summary['cached_layout_postprocessed'])
            self.assertTrue((Path(tmp)/'G_full.json').is_file())
            self.assertTrue((Path(tmp)/'camera_initialization.json').is_file())
            self.assertFalse(summary['optimizer_enabled'])
            self.assertIsNone(summary['optimizer_success'])

    def test_run006_keeps_every_observed_state_and_reports_bad_completion(self):
        data = read(Path(__file__).parent/'testdata/semantic_layout_run006.json')
        objects = data['scene']['objects']
        observed = [o for o in objects if o['role'] == 'observed']
        answer = dict(room_type=data['scene']['room_type'],
                      additions=[o for o in objects if o['role'] == 'hypothesized'])
        expected, _ = camera_layout(data['answer'], objects, 90, (768, 768))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(None, Image.new('RGB', (768, 768)), 90, output, 'test',
                          reused=observed, cached_plan=answer, cached_layout=data['answer'])
            self.assertEqual(read(output/'layout_refined.json'), expected)
            obs, full = read(output/'G_obs.json'), read(output/'G_full.json')
            self.assertEqual(full['objects'][:17], obs['objects'])
            self.assertEqual(summary['outside_fov_failed'], 2)
            self.assertEqual(summary['occlusion_unverified'], 2)
            self.assertEqual(full['relations'], [])
            self.assertFalse(summary['optimizer_enabled'])
            self.assertFalse(any(w['issue'] == 'observed_projection_no_overlap'
                                 for w in read(output/'geometry_report.json')['after']['warnings']))

    def test_read_only_inspection_never_moves_even_invalid_outside_proposals(self):
        _, plan, layout = fixture()
        original = deepcopy(layout)
        report = inspect_layout(layout, plan['objects'], 90, (768, 768))
        self.assertEqual(layout, original)
        self.assertIn('outside_fov_not_satisfied', [w['issue'] for w in report['warnings']])
        layout['objects'][-1]['center'] = [-4., 0., -2.]
        report = inspect_layout(layout, plan['objects'], 90, (768, 768))
        self.assertNotIn('outside_fov_not_satisfied', [w['issue'] for w in report['warnings']])

    def test_refinement_improves_fixed_scene_without_changing_sizes(self):
        observed, plan, layout = fixture()
        corrected, report = refine(layout, plan['objects'], plan['relations'], 90, (768, 768))
        before = sum(v['squared_sum'] for v in report['before']['weighted_residuals'].values())
        after = sum(v['squared_sum'] for v in report['after']['weighted_residuals'].values())
        self.assertLess(after, before*.5)
        self.assertEqual([o['size'] for o in corrected['objects']], [o['size'] for o in layout['objects']])
        self.assertEqual([o['id'] for o in corrected['objects']], [o['id'] for o in layout['objects']])
        self.assertEqual(corrected['room'], layout['room'])

    def test_frustum_checks_entire_box_and_both_camera_sides(self):
        # Center is outside the right edge but the large box still enters view.
        self.assertGreater(frustum_margin(np.array([3.1, 0, 3]), np.array([2., 1, 2]), 1, 1), 0)
        for x in (-5., 5.):
            self.assertLess(frustum_margin(np.array([x, 0, 2]), np.ones(3), 1, 1), 0)
        self.assertLess(frustum_margin(np.array([0., 0, -2]), np.ones(3), 1, 1), 0)

    def test_support_is_not_penalized_as_collision(self):
        observed, plan, layout = fixture()
        plan['relations'] = [dict(a='observed_0001', type='on_top_of', b='observed_0000')]
        layout['objects'][1]['center'] = [0, 0, 3]
        problem = BoxProblem(layout, plan['objects'], plan['relations'], 90, (768, 768))
        self.assertNotIn((0, 1), [tuple(pair) for pair in problem.pairs])
        corrected, _ = refine(layout, plan['objects'], plan['relations'], 90, (768, 768))
        bed, lamp = corrected['objects'][:2]
        self.assertLess(abs(lamp['center'][1]-bed['center'][1]-(lamp['size'][1]+bed['size'][1])/2), .1)

    def test_plan_rejects_observed_duplication_and_support_cycle(self):
        observed, _, _ = fixture()
        answer = dict(room_type='bedroom', additions=[dict(id=observed[0]['id'], category='bed',
            description='duplicate', unseen_reason='occluded', reason='x')], relations=[])
        with self.assertRaisesRegex(ValueError, 'distinct ID'):
            decode_plan(answer, observed, 8)
        answer.update(additions=[], relations=[
            dict(a=observed[0]['id'], type='on_top_of', b=observed[1]['id']),
            dict(a=observed[1]['id'], type='on_top_of', b=observed[0]['id'])])
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            decode_plan(answer, observed, 8)

    def test_layout_cannot_drop_instances_or_use_zero_size(self):
        _, plan, layout = fixture()
        bad = deepcopy(layout); bad['objects'].pop()
        with self.assertRaises(ValueError):
            validate_layout(bad, plan['objects'])
        bad = deepcopy(layout); bad['objects'][0]['size'][1] = 0
        with self.assertRaises(ValueError):
            validate_layout(bad, plan['objects'])

    def test_reports_multiple_parents_and_cycles_together(self):
        observed, plan, _ = fixture()
        answer = dict(room_type='bedroom', additions=[{k: v for k, v in plan['objects'][-1].items() if k != 'role'}],
            relations=[dict(a=observed[0]['id'], type='on_top_of', b=observed[1]['id']),
                       dict(a=observed[0]['id'], type='on_top_of', b='wardrobe'),
                       dict(a=observed[1]['id'], type='on_top_of', b=observed[0]['id'])])
        with self.assertRaises(RelationValidationError) as caught:
            decode_plan(answer, observed, 8)
        self.assertIn('single parent', str(caught.exception))
        self.assertIn('Cyclic', str(caught.exception))

    def test_single_relation_correction_preserves_objects(self):
        observed, plan, _ = fixture()
        answer = dict(room_type='bedroom', additions=[{k: v for k, v in plan['objects'][-1].items() if k != 'role'}],
            relations=[dict(a=observed[0]['id'], type='on_top_of', b=observed[1]['id']),
                       dict(a=observed[1]['id'], type='on_top_of', b=observed[0]['id'])])
        original = deepcopy(answer)
        calls = []
        def ask(prompt, name, tokens):
            calls.append(name)
            self.assertIn('Cyclic', prompt)
            return dict(relations=plan['relations'])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result, diagnostic = validated_plan(answer, observed, 8, ask, output)
            self.assertEqual(calls, ['relation_correction'])
            self.assertEqual(result['objects'], plan['objects'])
            self.assertEqual(answer, original)
            self.assertTrue(diagnostic['repair_attempted'])
            self.assertTrue(diagnostic['final_graph_valid'])
            self.assertTrue(read(output/'relation_validation.json')['removed_relations'])

    def test_bad_correction_stops_after_one_attempt(self):
        observed, _, _ = fixture()
        answer = dict(room_type='bedroom', additions=[], relations=[
            dict(a=observed[0]['id'], type='on_top_of', b=observed[1]['id']),
            dict(a=observed[1]['id'], type='on_top_of', b=observed[0]['id'])])
        calls = []
        def ask(prompt, name, tokens):
            calls.append(name)
            return dict(relations=deepcopy(answer['relations']))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RelationValidationError):
                validated_plan(answer, observed, 8, ask, Path(tmp))
            self.assertEqual(len(calls), 1)
            self.assertFalse(read(Path(tmp)/'relation_validation.json')['final_graph_valid'])

    def test_above_below_use_vertical_order_without_support(self):
        _, plan, layout = fixture()
        relations = [dict(a='observed_0001', type='above', b='observed_0000'),
                     dict(a='observed_0000', type='below', b='observed_0001')]
        problem = BoxProblem(layout, plan['objects'], relations, 90, (768, 768))
        self.assertEqual(problem.parents, {})
        self.assertEqual(problem.residuals(problem.initial.ravel(), grouped=True)['relations']['max_abs'], 0)
        swapped = problem.initial.copy(); swapped[:, 1] *= -1
        self.assertGreater(problem.residuals(swapped.ravel(), grouped=True)['relations']['max_abs'], 1)

    def test_actual_run002_excess_relations_reach_single_correction(self):
        data = read(Path(__file__).parent/'testdata/semantic_plan_run002.json')
        answer, observed = data['answer'], data['observed']
        original = deepcopy(answer)
        self.assertEqual(len(answer['relations']), 46)
        with self.assertRaises(RelationValidationError) as caught:
            decode_plan(answer, observed, 8)
        message = str(caught.exception)
        for issue in ('46 > 40', 'Duplicate', 'single parent', 'Cyclic physical support', 'Contradictory z order'):
            self.assertIn(issue, message)
        calls = []
        # This is a stand-in model response, not a claimed GPU correction result.
        relations = [dict(a='observed_0002', type='left_of', b='observed_0003'),
                     dict(a='add_0003', type='near', b='observed_0000')]
        def ask(prompt, name, tokens):
            calls.append(name)
            self.assertIn('46 > 40', prompt)
            self.assertIn('Contradictory z order', prompt)
            return dict(relations=relations)
        with tempfile.TemporaryDirectory() as tmp:
            plan, diagnostic = validated_plan(answer, observed, 8, ask, Path(tmp))
            self.assertEqual(calls, ['relation_correction'])
            self.assertEqual(len(plan['objects']), 20)
            self.assertTrue(diagnostic['final_graph_valid'])
            self.assertEqual(read(Path(tmp)/'semantic_plan_corrected.json')['additions'], answer['additions'])
            self.assertEqual(answer, original)

    def test_relation_container_error_also_enters_correction(self):
        observed, _, _ = fixture()
        answer = dict(room_type='bedroom', additions=[], relations={})
        calls = []
        def ask(prompt, name, tokens):
            calls.append(name)
            return dict(relations=[dict(a=observed[0]['id'], type='left_of', b=observed[1]['id'])])
        with tempfile.TemporaryDirectory() as tmp:
            _, diagnostic = validated_plan(answer, observed, 8, ask, Path(tmp))
            self.assertEqual(len(calls), 1)
            self.assertTrue(diagnostic['final_graph_valid'])

    def test_relation_count_alone_is_not_semantic_failure_or_truncated(self):
        observed, _, _ = fixture()
        many = [dict(deepcopy(observed[0]), id=f'object_{i}') for i in range(10)]
        relations = [dict(a=many[i]['id'], type='near', b=many[j]['id'])
                     for i in range(10) for j in range(i+1, 10)]
        answer = dict(room_type='bedroom', additions=[], relations=relations)
        def no_call(*args):
            raise AssertionError('Count alone should not require GPU correction')
        with tempfile.TemporaryDirectory() as tmp:
            plan, diagnostic = validated_plan(answer, many, 8, no_call, Path(tmp))
            self.assertEqual(len(plan['relations']), 45)
            self.assertTrue(diagnostic['relation_budget_exceeded'])
            self.assertFalse(diagnostic['repair_attempted'])

    def test_transitive_axis_cycle_is_rejected(self):
        observed, plan, _ = fixture()
        answer = dict(room_type='bedroom', additions=[{k: v for k, v in plan['objects'][-1].items() if k != 'role'}],
            relations=[dict(a='observed_0000', type='left_of', b='observed_0001'),
                       dict(a='observed_0001', type='left_of', b='wardrobe'),
                       dict(a='wardrobe', type='left_of', b='observed_0000')])
        with self.assertRaisesRegex(RelationValidationError, 'Cyclic x order'):
            decode_plan(answer, observed, 8)

    def test_continuation_estimates_layout_without_reusing_failed_graph(self):
        observed, plan, layout = fixture()
        answer = dict(room_type=plan['room_type'], additions=[{k: v for k, v in plan['objects'][-1].items() if k != 'role'}],
            relations=[dict(a='observed_0000', type='on_top_of', b='observed_0001'),
                       dict(a='observed_0001', type='on_top_of', b='observed_0000')])
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem == 'coarse_layout':
                    if 'on_top_of' in prompt or '"relations"' in prompt:
                        raise AssertionError('Failed relation graph leaked into layout request')
                    return deepcopy(layout)
                raise AssertionError('Unexpected repeated inference: '+path.stem)
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(FakeModel(), Image.new('RGB', (200, 160)), 90, Path(tmp), 'test',
                          reused=observed, cached_plan=answer)
            self.assertEqual(calls, ['coarse_layout'])
            self.assertFalse(summary['relation_correction_used'])
            self.assertFalse(summary['generated_relation_graph_used'])
            self.assertEqual(read(Path(tmp)/'planning_protocol.json')['legacy_relation_count'], 2)

    def test_real_failed_graph_is_not_a_numeric_layout_condition(self):
        data = read(Path(__file__).parent/'testdata/semantic_plan_run002.json')
        answer = deepcopy(data['answer'])
        original = deepcopy(answer)
        plan = scene_objects(answer, data['observed'], 8)
        self.assertEqual(len(plan['objects']), 20)
        self.assertNotIn('relations', plan)
        prompt = layout_prompt(plan, 90, (768, 768))
        self.assertNotIn('"on_top_of"', prompt)
        self.assertNotIn('"relations"', prompt)
        self.assertEqual(answer, original)

    def test_direct_support_reference_is_child_to_parent(self):
        _, plan, layout = fixture()
        layout['objects'][1]['support'] = 'observed_0000'
        layout = validate_layout(layout, plan['objects'])
        relations, warnings = support_constraints(layout)
        self.assertEqual(relations, [dict(a='observed_0001', type='on_top_of', b='observed_0000')])
        self.assertEqual(warnings, [])
        problem = BoxProblem(layout, plan['objects'], relations, 90, (768, 768))
        self.assertEqual(problem.parents, {1: 0})

    def test_invalid_auxiliary_support_is_reported_without_false_constraint(self):
        _, plan, layout = fixture()
        layout['objects'][0]['support'] = 'observed_0001'
        layout['objects'][1]['support'] = 'observed_0000'
        layout['objects'][2]['support'] = ['observed_0000', 'observed_0001']
        layout = validate_layout(layout, plan['objects'])
        relations, warnings = support_constraints(layout)
        self.assertEqual(relations, [])
        self.assertEqual(len(warnings), 3)
        self.assertEqual(layout['objects'][0]['support'], 'observed_0001')
        self.assertEqual(layout['objects'][2]['support'], ['observed_0000', 'observed_0001'])
        corrected, _ = refine(layout, plan['objects'], relations, 90, (768, 768))
        self.assertTrue(np.isfinite([o['center'] for o in corrected['objects']]).all())
    def test_reuse_is_image_bound_and_discards_old_metric_guesses(self):
        observed, _, layout = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rgb = Image.new('RGB', (100, 100), 'white'); rgb.save(directory/'local.png')
            state = make_graph(observed, layout, 'test', 'bad_old_geometry', 'observed')
            write(directory/'G_obs.json', state)
            reused = reuse_inventory(directory/'G_obs.json', rgb, 'test', 32)
            self.assertEqual(reused, observed)
            self.assertNotIn('center', reused[0])
            with self.assertRaisesRegex(ValueError, 'different image'):
                reuse_inventory(directory/'G_obs.json', Image.new('RGB', rgb.size, 'black'), 'test', 32)

    def test_three_call_pipeline_and_real_encoder_cpu_contract(self):
        observed, plan, layout = fixture()
        inventory_answer = [dict(category=o['category'], description=o['description'], bbox_2d=[v*1000 for v in o['bbox_2d']]) for o in observed]
        observed_answer = dict(objects=[dict(id=o['id'], depth_m=o['center'][2], size=o['size']) for o in layout['objects'][:2]])
        addition = {k: v for k, v in plan['objects'][-1].items() if k != 'role'}
        addition.update(azimuth_deg=-90., distance_m=3., bottom_height_m=0., size=layout['objects'][-1]['size'])
        completion_answer = dict(room_type=plan['room_type'], additions=[addition])
        class FakeModel:
            def __init__(self):
                self.calls = 0
            def ask(self, image, prompt, path, **kwargs):
                if path.stem == 'scene_completion':
                    # Numeric local state exists before the completion request.
                    state = read(path.parent/'G_obs.json')
                    self.assert_fixed_state = deepcopy(state)
                    if not all(o['center_local_m']['valid'] for o in state['objects']):
                        raise AssertionError('Completion requested before observed state was established')
                    if 'FIXED OBSERVED STATE' not in prompt or 'observed_0000' not in prompt:
                        raise AssertionError('Observed state missing from completion context')
                answer = [inventory_answer, observed_answer, completion_answer][self.calls]
                self.calls += 1
                path.write_text(str(answer))
                return deepcopy(answer)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); model = FakeModel()
            summary = run(model, Image.new('RGB', (200, 160)), 90, output, 'test')
            self.assertEqual(model.calls, 3)
            self.assertEqual(summary['full_objects'], 3)
            self.assertEqual(summary['quality_status'], 'requires_visual_review')
            raw, corrected = read(output/'G_full_raw.json'), read(output/'G_full.json')
            self.assertEqual([o['track_id'] for o in raw['objects']], [o['track_id'] for o in corrected['objects']])
            from ..oracle_model import tensorize
            categories, geometry, mask = tensorize(corrected, max_objects=160)
            self.assertEqual(categories[0, 1:4].tolist(), [11, 28, 7])
            self.assertEqual(mask.sum().item(), 4)
            self.assertTrue((geometry[0, 1:4, 6:8] == 1).all())
            self.assertEqual(geometry[0, 1, 3:6].tolist(), [2., 1., 2.])
            self.assertTrue((output/'layout_refined.png').is_file())
            self.assertTrue((output/'projection_refined.png').is_file())
            self.assertEqual(corrected['objects'][:2], model.assert_fixed_state['objects'])
            self.assertEqual(read(output/'G_obs.json'), model.assert_fixed_state)
            self.assertEqual(read(output/'G_hidden.json')['objects'], corrected['objects'][2:])
            np.testing.assert_allclose(corrected['objects'][2]['center_local_m']['value'], [-3., -.5, 0.], atol=1e-12)
            self.assertEqual(summary['completion_parameterization'], 'camera_azimuth_range_bottom_height')

    def test_empty_completion_keeps_observed_world(self):
        observed, _, layout = fixture()
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem == 'observed_layout':
                    return dict(objects=layout['objects'][:2])
                return dict(room_type='bedroom', additions=[])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB', (200, 160)), 90, output, 'test', reused=observed)
            self.assertEqual(calls, ['observed_layout', 'scene_completion'])
            self.assertEqual(summary['hypothesized'], 0)
            self.assertEqual(read(output/'G_hidden.json')['objects'], [])
            self.assertEqual(read(output/'G_obs.json')['objects'], read(output/'G_full.json')['objects'])

    def test_reuse_numeric_state_calls_only_completion_and_keeps_off_ray_centers(self):
        observed, plan, layout = fixture()
        state = make_graph(observed, layout, 'test', 'fixed_numeric_hypothesis', 'observed')
        original = deepcopy(state)
        calls = []
        class FakeModel:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                if path.stem != 'scene_completion':
                    raise AssertionError('Re-estimated a frozen numeric observation')
                if read(path.parent/'G_obs.json') != original:
                    raise AssertionError('Observed input changed before completion')
                return dict(room_type='bedroom', additions=[dict(id='cabinet_new', category='cabinet',
                    description='storage cabinet', reason='plausible storage outside photo',
                    camera_region='behind-camera', distance_m=3., bottom_height_m=0., size=[1., 2., .5])])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(FakeModel(), Image.new('RGB', (200, 160)), 90, output, 'test',
                          reused=observed, fixed_observed_state=state)
            self.assertEqual(calls, ['scene_completion'])
            self.assertTrue(summary['numeric_observation_reused'])
            self.assertEqual(read(output/'G_obs.json'), original)
            self.assertEqual(read(output/'G_full.json')['objects'][:2], original['objects'])
            self.assertEqual(read(output/'layout_raw.json')['objects'][:2],
                             observed_state_layout(original, observed)['objects'])
            self.assertEqual(state, original)

    def test_reuse_numeric_state_rejects_unknown_geometry_or_wrong_ids(self):
        observed, _, layout = fixture()
        unknown = make_graph(observed, None, 'test', 'not_estimated', 'observed')
        with self.assertRaisesRegex(ValueError, 'Numeric G_obs required'):
            observed_state_layout(unknown, observed)
        state = make_graph(observed, layout, 'test', 'coarse_hypothesis', 'observed')
        with self.assertRaisesRegex(ValueError, 'IDs/scope'):
            observed_state_layout(state, observed[::-1])

    def test_completion_cannot_rewrite_observed_id_or_supply_invalid_size(self):
        observed, _, layout = fixture()
        for overwrite in (True, False):
            class FakeModel:
                def ask(self, image, prompt, path, **kwargs):
                    if path.stem == 'observed_layout':
                        return dict(objects=layout['objects'][:2])
                    return dict(room_type='bedroom', additions=[dict(
                        id=observed[0]['id'] if overwrite else 'new_object', category='cabinet',
                        description='wardrobe', reason='storage', camera_region='behind-camera',
                        distance_m=2., bottom_height_m=0., size=[1., 1., 1.] if overwrite else [0., 1., 1.])])
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                with self.assertRaises(ValueError):
                    run(FakeModel(), Image.new('RGB', (200, 160)), 90, output, 'test', reused=observed)
                self.assertTrue(all(o['center_local_m']['valid'] for o in read(output/'G_obs.json')['objects']))
                self.assertFalse((output/'G_full.json').exists())

    def test_occlusion_is_reported_unverified(self):
        _, plan, layout = fixture()
        plan['objects'][-1]['unseen_reason'] = 'occluded'
        problem = BoxProblem(layout, plan['objects'], [], 90, (768, 768))
        issues = problem.diagnose(problem.initial)['warnings']
        self.assertIn('occlusion_not_verified_by_box_solver', [o['issue'] for o in issues])


if __name__ == '__main__':
    unittest.main()
