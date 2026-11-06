"""CPU contract/geometry checks; fake model responses do not prove 8B quality."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..common import read, sha, write
from ..oracle_model import tensorize
from .scene_completion import (content_prompt, export_states, map_plan, review_boxes,
                               run, saved_plan, validate_layout, validate_plan, view_check)
from .test_semantic_probe import state


def plan():
    return dict(scene_type='study', scene_hypothesis='A plausible surrounding work area.', additions=[
        dict(id='new_0', name='book storage', object_type='bookshelf',
             reason='Stores books for work.', placement='Behind the camera.')])


def layout(center=None):
    return dict(objects=[dict(id='new_0', center=center or [0., -.1, -2.], size=[.8, 1.5, .4],
                              reason='A storage corner outside the photograph; height is hypothetical.')])


class CompletionTests(unittest.TestCase):
    def test_content_does_not_see_noisy_numeric_geometry_or_category_menu(self):
        context = [dict(id='o', category='table', description='work desk',
                        visible_bbox_xyxy_norm=[.1,.1,.5,.5], center=[12345,67890,99999],
                        full_aabb_size=[11111,22222,33333])]
        case = dict(context=context, hfov=90)
        before = deepcopy(case)
        question = content_prompt(case)
        self.assertNotIn('12345', question)
        self.assertNotIn('11111', question)
        self.assertNotIn('ALLOWED CATEGORIES', question)
        self.assertNotIn('height_anchor_id', question)
        self.assertEqual(case, before)

    def test_actual_noun_determines_mapping_and_unknown_nouns_remain_candidates(self):
        value = plan()
        value['additions'][0].update(name='wooden sauna heater', object_type='sauna heater')
        self.assertEqual(map_plan(value)[0]['category'], 'misc')
        self.assertEqual(value['additions'][0]['name'], 'wooden sauna heater')
        value['additions'][0]['object_type'] = 'UNSUPPORTED_TEST_NOUN'
        mappings = map_plan(value)
        self.assertIsNone(mappings[0]['category'])
        observed = state()
        items = validate_layout(layout(), value, mappings)
        checks = review_boxes(items, observed, 90, (128,128))
        candidates, full, unresolved = export_states(observed, value, items, mappings, checks)
        self.assertEqual(len(candidates['objects']), len(observed['objects'])+1)
        self.assertEqual(len(full['objects']), len(observed['objects']))
        self.assertEqual(unresolved[0]['description'], 'wooden sauna heater')
        self.assertFalse(full['semantic_plan_fully_exported'])

    def test_layout_cannot_change_names_categories_counts_or_instance_ids(self):
        for kind in ('missing', 'duplicate', 'unknown', 'category', 'nan', 'negative'):
            value = layout()
            if kind == 'missing': value['objects'] = []
            elif kind == 'duplicate': value['objects'] *= 2
            elif kind == 'unknown': value['objects'][0]['id'] = 'uncommitted'
            elif kind == 'category': value['objects'][0]['category'] = 'seating'
            elif kind == 'nan': value['objects'][0]['center'][0] = float('nan')
            elif kind == 'negative': value['objects'][0]['size'][0] = -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_layout(value, plan(), map_plan(plan()))

    def test_camera_cone_checks_front_back_straddling_and_camera_inside(self):
        self.assertTrue(view_check([0,0,3], [.5,.5,.5], 90, (128,128))['intersects_input_box_proxy'])
        self.assertFalse(view_check([0,0,-3], [.5,.5,.5], 90, (128,128))['intersects_input_box_proxy'])
        self.assertFalse(view_check([5,0,2], [.5,.5,.5], 90, (128,128))['intersects_input_box_proxy'])
        self.assertTrue(view_check([0,0,0], [1,1,1], 90, (128,128))['camera_inside_box_proxy'])
        # A tall thin photograph has a different vertical frustum.
        self.assertTrue(view_check([0,2,3], [.2,.2,.2], 90, (128,128))['intersects_input_box_proxy'])
        self.assertFalse(view_check([0,2,3], [.2,.2,.2], 90, (256,64))['intersects_input_box_proxy'])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Independent LP reference unavailable')
    def test_analytic_intersection_matches_independent_linear_program(self):
        from scipy.optimize import linprog
        rng = np.random.default_rng(23)
        for _ in range(80):
            center = rng.uniform(-6,6,3)
            size = rng.uniform(.05,4,3)
            hfov = rng.uniform(35,130)
            width, height = 320, int(rng.integers(100,600))
            tx = math.tan(math.radians(hfov)/2); ty = tx*height/width
            constraints = [[1,0,-tx],[-1,0,-tx],[0,1,-ty],[0,-1,-ty],[0,0,-1]]
            reference = linprog([0,0,0], A_ub=constraints, b_ub=[0,0,0,0,-1e-6],
                                bounds=list(zip(center-size/2,center+size/2)), method='highs')
            self.assertEqual(view_check(center,size,hfov,(width,height))['intersects_input_box_proxy'],
                             reference.success)

    def test_overlaps_are_reported_without_moving_boxes_or_claiming_physical_collision(self):
        observed = state()
        reference = observed['objects'][0]
        value = layout(reference['center_local_m']['value'])
        before = deepcopy(value)
        items = validate_layout(value, plan(), map_plan(plan()))
        checks = review_boxes(items, observed, 90, (128,128))
        self.assertTrue(checks['objects'][0]['box_proxy_overlaps'])
        self.assertFalse(checks['physical_support_verified'])
        self.assertEqual(value, before)

    def test_export_preserves_observed_objects_and_real_encoder_prefix(self):
        observed = state(); before = deepcopy(observed)
        mappings = map_plan(plan()); items = validate_layout(layout(), plan(), mappings)
        checks = review_boxes(items, observed, 90, (128,128))
        _, full, _ = export_states(observed, plan(), items, mappings, checks)
        self.assertEqual(observed, before)
        self.assertEqual(full['objects'][:len(before['objects'])], before['objects'])
        for a,b in zip(tensorize(observed,160),tensorize(full,160)):
            self.assertTrue(a[:,:len(before['objects'])+1].equal(b[:,:len(before['objects'])+1]))
        self.assertFalse(full['semantics_verified'])
        self.assertEqual(full['objects'][-1]['center_local_m']['value'], [0.,-.1,-2.])

    def test_bounded_visual_feedback_preserves_both_real_proposals(self):
        observed = state()
        class Model:
            def __init__(self): self.images = []
            def ask(self, image, question, path, **kwargs):
                self.images.append(image.size)
                answer = plan() if path.name == 'content.txt' else layout([0.,0.,3.])
                path.write_text(json.dumps(answer))
                write(path.with_suffix('.json'), answer)
                return answer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'input.png'; Image.new('RGB',(128,128)).save(image)
            state_path=root/'G_obs.json'; write(state_path, observed)
            case=dict(id='case', image=str(image), state=str(state_path), hfov=90,
                      context=[], source_sha256={str(image):sha(image), str(state_path):sha(state_path)})
            model=Model()
            self.assertTrue(run(model,[case],root,stage='full',layout_revisions=1))
            self.assertEqual(model.images,[(128,128),(128,128),(1536,1536)])
            folder=root/'case'
            self.assertTrue((folder/'layout_000/feedback.png').is_file())
            self.assertTrue((folder/'layout_001/proposal.json').is_file())
            self.assertEqual(read(folder/'G_obs.json'), observed)
            self.assertGreater(read(folder/'summary.json')['numeric_warning_objects'],0)
            self.assertEqual(len(read(folder/'layout_history.json')),2)
            self.assertFalse(read(root/'summary.json')['model_quality_validated'])
            self.assertTrue((folder/'G_full_proposal.json').is_file())
            self.assertFalse((folder/'G_full.json').exists())

    def test_usable_numeric_export_requires_all_planned_categories_and_no_numeric_warning(self):
        for noun, expected_export in [('bookshelf',True),('UNSUPPORTED_TEST_NOUN',False)]:
            class Model:
                def ask(self, image, question, path, **kwargs):
                    if path.name == 'content.txt':
                        value=plan();value['additions'][0]['object_type']=noun
                        return value
                    return layout()
            with self.subTest(noun=noun), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);image=root/'input.png';Image.new('RGB',(32,32)).save(image)
                state_path=root/'obs.json';write(state_path,state())
                case=dict(id='case',image=str(image),state=str(state_path),hfov=90,context=[],source_sha256={})
                self.assertTrue(run(Model(),[case],root,stage='full'))
                folder=root/'case'
                self.assertEqual((folder/'G_full.json').is_file(),expected_export)
                self.assertTrue((folder/'G_full_candidates.json').is_file())
                self.assertEqual(read(folder/'summary.json')['adapter_state_exported'],expected_export)

    def test_empty_content_is_explicit_and_never_reported_as_objects_added(self):
        observed=state()
        class Model:
            def ask(self, image, question, path, **kwargs):
                return dict(scene_type='study',scene_hypothesis='No further committed contents.',additions=[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'input.png'; Image.new('RGB',(32,32)).save(image)
            state_path=root/'obs.json';write(state_path,observed)
            case=dict(id='case',image=str(image),state=str(state_path),hfov=90,context=[],source_sha256={})
            self.assertTrue(run(Model(),[case],root,stage='full'))
            summary=read(root/'case/summary.json')
            self.assertEqual(summary['status'],'no_additions_proposed')
            self.assertEqual(summary['hypothesized_instances'],0)
            self.assertFalse((root/'case/G_full.json').exists())

    def test_saved_semantics_require_original_response_and_matching_evidence(self):
        case=dict(id='case',context=[],hfov=90,source_sha256={})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'case';folder.mkdir()
            write(folder/'input.json',case);write(folder/'content.json',plan());write(folder/'content.txt',plan())
            write(folder/'content.request.json',dict(prompt=content_prompt(case)))
            self.assertEqual(saved_plan(root,case,160)[0],plan())
            edited=plan();edited['additions']=[];write(folder/'content.json',edited)
            with self.assertRaisesRegex(ValueError,'raw model response'):saved_plan(root,case,160)


if __name__ == '__main__':
    unittest.main()
