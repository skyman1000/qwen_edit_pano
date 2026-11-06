"""Actual run004 regression; fake revisions test control flow, not model quality."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ..common import read
from .box_completion import projection_feedback, revise_boxes
from .semantic_world import make_graph, run, scene_objects
from .test_semantic_world import fixture


def actual_proposal():
    return read(Path(__file__).parent/'testdata/scene_boxes_box_world_run004.json')


class BoxCompletionTests(unittest.TestCase):
    def test_training_examples_condition_one_fresh_call_and_preserve_observation(self):
        observed,_,layout=fixture()
        state=make_graph(observed,layout,'test','coarse','observed')
        bank=dict(examples=[dict(input=dict(fixed_observed=[]),output=dict(additions=[]))])
        calls=[]
        class Model:
            def ask(self,image,prompt,path,**kwargs):
                calls.append(path.stem)
                assert [m['role'] for m in kwargs['examples']]==['user','assistant']
                assert 'PREVIOUS PROPOSAL' not in prompt
                return actual_proposal()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            summary=run(Model(),Image.new('RGB',(768,768)),90,output,'test',reused=observed,
                        fixed_observed_state=state,completion_mode='boxes',layout_examples=bank)
            self.assertEqual(calls,['scene_boxes'])
            self.assertEqual(read(output/'G_full.json')['objects'][:2],state['objects'])
            self.assertEqual(summary['layout_example_count'],1)
            self.assertEqual(summary['outside_fov_failed'],4)  # Examples never suppress real errors.

    def test_actual_four_offscreen_claims_project_wholly_inside_photo(self):
        answer = actual_proposal()
        original = deepcopy(answer)
        feedback = projection_feedback(answer, 90, (768, 768))
        self.assertEqual([o['id'] for o in feedback], [o['id'] for o in answer['additions']])
        self.assertTrue(all(o['whole_aabb_projects_inside_photo'] for o in feedback))
        self.assertTrue(all(27 < o['center_azimuth_deg'] < 33 for o in feedback))
        self.assertEqual(answer, original)

    def test_cached_revision_asks_once_and_preserves_observed_state_and_inventory(self):
        observed, _, layout = fixture()
        state = make_graph(observed, layout, 'test', 'coarse', 'observed')
        answer = actual_proposal()
        original = deepcopy(answer)
        calls = []
        class Model:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                self.assert_feedback(prompt)
                candidate = deepcopy(answer)
                for i, obj in enumerate(candidate['additions']):
                    obj['center'] = [i-1.5, 0., -3.]
                return candidate

            @staticmethod
            def assert_feedback(prompt):
                assert 'COMPUTED PROJECTION FEEDBACK' in prompt
                assert 'whole_aabb_projects_inside_photo' in prompt

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(Model(), Image.new('RGB', (768, 768)), 90, output, 'test',
                reused=observed, fixed_observed_state=state, completion_mode='boxes',
                cached_completion=answer)
            full = read(output/'G_full.json')
            self.assertEqual(calls, ['scene_boxes_revision'])
            self.assertEqual(full['objects'][:len(observed)], state['objects'])
            self.assertEqual([o['track_id'] for o in full['objects'][len(observed):]],
                             [o['id'] for o in answer['additions']])
            self.assertEqual(summary['completion_conflicts_before'], 4)
            self.assertEqual(summary['completion_conflicts_after'], 0)
            self.assertEqual(summary['completion_withdrawn'], 0)
            self.assertFalse(summary['scene_semantics_verified'])
            self.assertEqual(read(output/'completion_initial.json'), original)
        self.assertEqual(answer, original)

    def test_invalid_revision_cannot_delete_or_relabel_failed_objects(self):
        original = actual_proposal()
        for kind in ('delete', 'relabel', 'category'):
            with self.subTest(kind=kind):
                candidate = deepcopy(original)
                if kind == 'delete':
                    candidate['additions'].pop()
                else:
                    key, value = ('unseen_reason', 'occluded') if kind == 'relabel' else ('category', 'table')
                    candidate['additions'][0][key] = value
                result, report = revise_boxes(original, 'context', lambda *a: candidate,
                    lambda a: scene_objects(a, [], 8), 90, (768, 768))
                self.assertEqual(result, original)
                self.assertIsNotNone(report['revision_error'])
                self.assertEqual(len(report['after']), 4)

    def test_wrong_geometry_remains_flagged_after_one_attempt(self):
        answer = actual_proposal()
        calls = []
        def ask(*args):
            calls.append(1)
            return deepcopy(answer)
        result, report = revise_boxes(answer, '', ask,
            lambda a: scene_objects(a, [], 8), 90, (768, 768))
        self.assertEqual(calls, [1])
        self.assertEqual(len(report['after']), 4)
        self.assertEqual(result, answer)
        self.assertFalse(report['numeric_revision_changed'])

    def test_occlusion_is_not_treated_as_offscreen_or_certified(self):
        answer = actual_proposal()
        for obj in answer['additions']:
            obj['unseen_reason'] = 'occluded'
        def ask(*args):
            self.fail('No offscreen claim to revise')
        _, report = revise_boxes(answer, '', ask, lambda a: None, 90, (768, 768))
        self.assertFalse(report['used'])
        self.assertEqual(report['before'], [])


if __name__ == '__main__':
    unittest.main()
