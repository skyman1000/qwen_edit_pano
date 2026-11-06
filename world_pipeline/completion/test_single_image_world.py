"""CPU boundary tests, including actual saved array responses; not model scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ..common import ROOT, read, validate_graph
from .geometry import parse_json
from . import single_image_world as single


BED = dict(category='bed', description='one bed', bbox_2d=[100, 200, 900, 950],
           center_local_m=[0., -.5, 3.], size_aabb_local_m=[2., 1., 2.])


class SingleImageTests(unittest.TestCase):
    def test_native_observer_keeps_pixels_and_does_not_certify_project_geometry(self):
        pixels = Image.new('RGB', (64,64), color=(67,89,123))
        objects, _ = single.decode_objects([{k: BED[k] for k in ('category','description','bbox_2d')}], 'observed', 32)
        class NativeModel:
            calls = []
            def ask(self, image, prompt, path, **kwargs):
                self.calls.append(path.stem)
                if image.tobytes() != pixels.tobytes():
                    raise AssertionError('Native inference image must not contain an added red annotation')
                if 'DIVIDED BY PI' in prompt or 'fx=' in prompt or 'size_aabb_local_m' in prompt:
                    raise AssertionError('Do not reparameterize the native camera/angle contract')
                if path.stem == 'completion':
                    raise AssertionError('Observer verification must not call completion')
                return [{'label':'bed','bbox_3d':[0,.5,3,2,1,2,0,0,0]}]
        model = NativeModel()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = single.run_observer(model, pixels, dict(hfov_deg=90), out, 'test',
                                         4096, 32, reused=(objects,[]))
            state = validate_graph(read(out/'G_obs.json'))
            self.assertEqual(model.calls, ['observed_0000'])
            self.assertEqual(result['native_numeric_boxes'], 1)
            self.assertEqual(result['project_numeric_boxes'], 0)
            self.assertFalse(state['observation_verified'])
            self.assertFalse(state['camera_mapping_verified'])
            self.assertFalse(state['objects'][0]['center_local_m']['valid'])
            self.assertEqual(state['objects'][0]['native_bbox_3d'], [0,.5,3,2,1,2,0,0,0])
            self.assertFalse((out/'G_full.json').exists())

    def test_reuse_rejects_wrong_image_and_gt_and_strips_old_geometry(self):
        from .geometry import graph
        objects, _ = single.decode_objects([BED], 'observed', 32)
        state = graph('test','val','observed',objects,'test')
        pixels = Image.new('RGB',(64,64), color=(67,89,123))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pixels.save(out/'local.png')
            single.write(out/'G_obs.json',state)
            reused, issues = single.reuse_visible_inventory(out/'G_obs.json',pixels)
            self.assertEqual(reused[0]['track_id'],objects[0]['track_id'])
            self.assertIsNone(reused[0]['center_local_m']['value'])
            self.assertIsNone(reused[0]['size_aabb_local_m']['value'])
            self.assertEqual(read(out/'G_obs.json'),state)
            with self.assertRaisesRegex(ValueError,'differs'):
                single.reuse_visible_inventory(out/'G_obs.json',Image.new('RGB',(64,64)))
            state['stage'] = 'gt'
            single.write(out/'G_obs.json',state)
            with self.assertRaisesRegex(ValueError,'not a GT'):
                single.reuse_visible_inventory(out/'G_obs.json',pixels)

    def test_reprojection_cannot_disambiguate_metric_scale(self):
        objects, _ = single.decode_objects([{k: BED[k] for k in ('category','description','bbox_2d')}], 'observed', 32)
        obj = objects[0]
        obj['native_bbox_3d'] = [1,.5,4,1,1,1,0,0,0]
        first = single.native_reprojection(obj,(768,768),90)
        obj['native_bbox_3d'] = [2,1,8,2,2,2,0,0,0]
        second = single.native_reprojection(obj,(768,768),90)
        self.assertEqual(first['projected_center'],second['projected_center'])
        self.assertEqual(first['projected_box'],second['projected_box'])
        self.assertEqual(first['status'],'diagnostic_only')
        self.assertIsNone(obj['center_local_m']['value'])

    def test_shapes_unknown_geometry_and_no_axis_repair(self):
        raw = deepcopy(BED)
        raw['center_local_m'] = [-2., 0., 0.]
        raw['size_aabb_local_m'] = None
        a, issues = single.decode_objects([raw], 'observed', 160)
        b, _ = single.decode_objects({'objects': [raw]}, 'observed', 160)
        self.assertEqual(a, b)
        self.assertEqual(a[0]['center_local_m']['value'], [-2., 0., 0.])
        self.assertIsNone(a[0]['size_aabb_local_m']['value'])
        self.assertFalse(a[0]['size_aabb_local_m']['valid'])
        self.assertTrue(any(x['error'] == 'visible_center_not_in_front_review_required' for x in issues))
        malformed = dict(BED, center_local_m=[1, 2], size_aabb_local_m=[-1, 2, 3])
        decoded, issues = single.decode_objects([malformed], 'observed', 160)
        self.assertIsNone(decoded[0]['center_local_m']['value'])
        self.assertIsNone(decoded[0]['size_aabb_local_m']['value'])
        self.assertEqual(len(issues), 2)
        for value in [None, 'list', {}, {'objects': {}}, [{}]]:
            with self.assertRaises(ValueError):
                single.decode_objects(value, 'observed', 160)

    def test_bounded_workflow_preserves_observation_and_separates_roles(self):
        additions = [dict(category='wardrobe', role='unseen', description='wooden wardrobe',
                          center_local_m=[-2., -.5, -2.], size_aabb_local_m=[1., 2., .6]),
                     dict(category='lighting', role='visible_missing', description='visible lamp',
                          bbox_2d=[100, 200, 200, 400], center_local_m=None, size_aabb_local_m=None),
                     dict(category='chair', description='uncertain additional chair',
                          center_local_m=None, size_aabb_local_m=None)]
        class BoundedModel:
            calls = []
            def ask(self, image, prompt, path, **kwargs):
                self.calls.append(path.stem)
                if kwargs['profile'] != 'official' or not kwargs['sample'] or not kwargs['guard_repetition']:
                    raise AssertionError('Use official sampling and the explicit loop guard')
                if kwargs['seed'] != 3407:
                    raise AssertionError('Initial request must record the official reproducibility seed')
                if path.stem == 'observation':
                    result = [{k: BED[k] for k in ('category', 'description', 'bbox_2d')}]
                    if kwargs['max_tokens'] > 2048:
                        raise AssertionError('Inventory budget not bounded')
                elif path.stem == 'completion':
                    context = json.loads(prompt.split('\nOBSERVED (bbox_2d uses 0..1000): ', 1)[1])
                    if context[0]['bbox_2d'] != BED['bbox_2d']:
                        raise AssertionError('Inconsistent 2D units')
                    result = [{k: v for k, v in a.items() if k not in ('center_local_m', 'size_aabb_local_m')} for a in additions]
                elif path.stem in ('observed_0000', 'added_0001'):
                    if kwargs['max_tokens'] > 512 or 'red rectangle' not in prompt:
                        raise AssertionError('Expected single-target native query')
                    result = [{'bbox_3d': [0, .5, 3, 2, 1, 2, 0, 0, 0], 'label': 'target'}]
                elif path.stem == 'added_0000':
                    if kwargs['max_tokens'] > 384 or 'ONLY this one' not in prompt:
                        raise AssertionError('Unseen placement must be bounded to one object')
                    result = [{k: additions[0][k] for k in ('center_local_m', 'size_aabb_local_m')}]
                else:
                    raise AssertionError('Unexpected query '+path.stem)
                path.write_text(json.dumps(result))
                return result
        model = BoundedModel()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = single.run(model, Image.new('RGB', (64, 64)), dict(hfov_deg=90), out, 'test', 12, 4096)
            self.assertEqual(model.calls, ['observation', 'observed_0000', 'completion', 'added_0000', 'added_0001'])
            self.assertEqual(summary['status'], 'partial')
            self.assertEqual((summary['unseen_proposals'], summary['visible_recoveries'], summary['unresolved_additions']), (1, 1, 1))
            obs, full = read(out/'G_obs.json'), read(out/'G_full.json')
            self.assertEqual(full['objects'][:1], obs['objects'])
            self.assertEqual(len(full['objects']), 4)
            self.assertEqual(full['quality_status'], 'not_validated')
            self.assertEqual(read(out/'G_hidden.json')['objects'][0]['center_local_m']['value'], [-2., -.5, -2.])
            self.assertEqual(read(out/'G_hidden.json')['objects'][0]['category_id']['value'], 7)
            for name in ('G_obs', 'G_hidden', 'G_visible_added', 'G_unresolved_added', 'G_full'):
                validate_graph(read(out/f'{name}.json'))
            from ..oracle_model import tensorize
            _, geometry, mask = tensorize(full, max_objects=160)
            self.assertEqual(int(mask.sum()), 5)
            self.assertEqual(geometry[0, 4, 6:].tolist(), [0., 0.])
            self.assertEqual(obs['objects'][0]['center_local_m']['value'], [0., -.5, 3.])
            self.assertTrue((out/'observed_overlay.png').exists())

    def test_geometry_failure_retains_semantic_object_as_unknown(self):
        class Model:
            calls = 0
            def ask(self, *args, **kwargs):
                self.calls += 1
                raise ValueError('response reached token limit')
        obj = single.decode_objects([{k: BED[k] for k in ('category','description','bbox_2d')}], 'observed', 32)[0][0]
        model, issues = Model(), []
        with tempfile.TemporaryDirectory() as tmp:
            single.visible_geometry(model, Image.new('RGB', (64,64)), 'camera', obj, Path(tmp), 512, issues)
        self.assertEqual(model.calls, 2)
        self.assertEqual(obj['category_id']['value'], 11)
        self.assertIsNone(obj['center_local_m']['value'])
        self.assertTrue(issues)

    def test_inventory_rejects_duplicate_loop_without_repair(self):
        from .json_generation import complete_array_prefix, repeated_object_tail
        source = ROOT/'qwen_edit_pano/outputs/single_image_qwen_run001/observation.txt'
        if source.exists():
            raw = source.read_text()
            prefix = complete_array_prefix(raw)
            self.assertEqual(len(prefix), 39)
            self.assertTrue(repeated_object_tail(raw))
            with self.assertRaises(ValueError):
                parse_json(raw)  # Do not repair the unfinished JSON into a world.
        repeated = [{k: BED[k] for k in ('category','description','bbox_2d')}]*3
        with self.assertRaisesRegex(ValueError, 'Repeated visible'):
            single.inventory(repeated, 'observed', 32)

    def test_real_array_responses_do_not_reproduce_previous_type_error(self):
        source = ROOT/'qwen_edit_pano/outputs/panorama_qwen_probe_v1/plan'
        paths = list(source.glob('*/*/plan.txt'))
        if not paths:
            self.skipTest('Historical raw responses not available')
        total = 0
        for path in paths:
            raw = parse_json(path.read_text())
            original = deepcopy(raw)
            decoded, issues = single.decode_objects(raw, 'added', 160)
            self.assertEqual(raw, original)
            self.assertEqual(len(decoded), len(raw))
            # Old plans contain no metric estimates or role; do not invent them.
            self.assertTrue(all(o['center_local_m']['value'] is None for o in decoded))
            self.assertTrue(all(o['completion_role'] == 'unresolved' for o in decoded))
            self.assertEqual(sum(x['field'] == 'role' for x in issues), len(raw))
            total += len(decoded)
        self.assertEqual(total, 56)


if __name__ == '__main__':
    unittest.main()
