"""CPU checks of the isolation boundary, not a measure of model capability."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .single_image_world import decode_objects
from .single_image_plan import semantic_context, run_plans, validate_unseen


class UnseenPlanTests(unittest.TestCase):
    def test_metric_estimates_are_not_supplied_to_semantic_planner(self):
        objects, _ = decode_objects([dict(category='bed', description='one bed',
            bbox_2d=[400, 500, 800, 900], center_local_m=[999, 999, 999],
            size_aabb_local_m=[999, 999, 999])], 'observed', 32)
        original = deepcopy(objects)
        context = semantic_context(dict(objects=objects))
        self.assertNotIn('999', str(context))
        self.assertEqual(context[0]['bbox_2d'], [.4, .5, .8, .9])
        self.assertEqual(objects, original)

    def test_no_visible_recovery_or_silent_geometry_in_unseen_plan(self):
        item = dict(id='h1', category='chair', description='one chair',
                    placement='behind CAMERA', unseen_reason='outside_fov', reason='hypothesis')
        validate_unseen(dict(additions=[item]), 8)
        for extra in [dict(role='visible_missing'), dict(bbox_2d=[0, 0, 1, 1]),
                      dict(center_local_m=[0, 0, -2]), dict(unseen_reason='uncertain')]:
            with self.assertRaises(ValueError):
                validate_unseen(dict(additions=[dict(item, **extra)]), 8)

    def test_failed_seed_and_empty_plan_remain_visible_without_retry_or_merge(self):
        class Model:
            calls = []
            def ask(self, image, prompt, path, **kwargs):
                self.calls.append(kwargs['seed'])
                if kwargs['seed'] == 3408:
                    raise ValueError('malformed response')
                return {'additions': []}
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = run_plans(model, Image.new('RGB', (64,64)), [], 'camera',
                               output, [3407, 3408, 3409])
            self.assertEqual(model.calls, [3407, 3408, 3409])
            self.assertEqual([r['status'] for r in result['results']], ['parsed', 'failed', 'parsed'])
            self.assertEqual(result['results'][0]['proposals'], 0)
            self.assertFalse((output/'G_full.json').exists())
            self.assertFalse(result['observed_geometry_used'])


if __name__ == '__main__':
    unittest.main()
