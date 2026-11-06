"""CPU contract tests; these do not measure pretrained detector accuracy."""
from copy import deepcopy
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import ROOT, read, write
from ..oracle_model import tensorize
from .box_observer import camera_aabb, export_state, intrinsics, load_source, prompt_pixels, bbox_iou, infer_one, projection_preview
from .semantic_world import observed_state_layout, reuse_inventory, run


def example():
    state = read(ROOT/'qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json')
    return state


def corners(center=(1., 2., 5.), size=(2., 4., 6.), rotation=None):
    signs = np.array(list(itertools.product([-1., 1.], repeat=3)))
    rotation = np.eye(3) if rotation is None else rotation
    return np.array(center)+(signs*np.array(size)/2)@rotation.T


class BoxObserverTests(unittest.TestCase):
    def test_new_image_inventory_never_asks_qwen_for_numerical_geometry(self):
        calls = []
        class Model:
            def ask(self, image, prompt, path, **kwargs):
                calls.append(path.stem)
                return dict(objects=[dict(category='bed', description='bed', bbox_2d=[100, 200, 900, 900])])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run(Model(), Image.new('RGB', (768, 768)), 90, output, 'test', inventory_only=True)
            self.assertEqual(calls, ['observation'])
            self.assertFalse(summary['numeric_geometry_executed'])
            self.assertFalse(read(output/'G_obs.json')['objects'][0]['center_local_m']['valid'])

    def test_axis_flip_full_dimensions_and_no_ground_adjustment(self):
        center, size = camera_aabb([1, 2, 5], corners())
        np.testing.assert_allclose(center, [1, -2, 5])
        np.testing.assert_allclose(size, [2, 4, 6])

    def test_rotated_box_must_not_copy_local_dimensions(self):
        a = np.pi/4
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        center, size = camera_aabb([1, 2, 5], corners(rotation=rotation))
        np.testing.assert_allclose(center, [1, -2, 5])
        np.testing.assert_allclose(size, [8/np.sqrt(2), 4, 8/np.sqrt(2)])

    def test_invalid_native_geometry_is_not_repaired_with_defaults(self):
        for center, points in (([1, 2, -5], corners()), ([1, 2, 5], np.zeros((8, 3))),
                               ([1, 2, 5], np.full((8, 3), np.nan))):
            with self.assertRaises(ValueError):
                camera_aabb(center, points)

    def test_all_old_numeric_guesses_replaced_and_adapter_reads_only_seven_values(self):
        state = example(); original = deepcopy(state)
        records = [dict(id=o['track_id'], status='predicted', box3d_native=[1, 2, 5, 6, 2, 4, 1, 0, 0, 0],
                        corners_native=corners().tolist()) for o in state['objects']]
        result, simple = export_state(state, records)
        self.assertEqual(state, original)
        self.assertEqual(len(simple), 17)
        self.assertEqual(set(simple[0]), {'category', 'position', 'size'})
        cat, geo, mask = tensorize(result, max_objects=32)
        np.testing.assert_allclose(geo[0, 1, :6].numpy(), [1, -2, 5, 2, 4, 6])
        np.testing.assert_allclose(geo[0, 1, 6:].numpy(), [1, 1])
        self.assertEqual(int(cat[0, 1]), 11)
        self.assertEqual(int(mask.sum()), 18)
        for old, new in zip(original['objects'], result['objects']):
            self.assertEqual(new['category_id'], old['category_id'])
            self.assertEqual(new['bbox2d_visible_xyxy_norm'], old['bbox2d_visible_xyxy_norm'])
        changed = deepcopy(result)
        changed['room_hypothesis'] = None
        changed['relations'] = [{'arbitrary': 'ignored by Adapter'}]
        c2, g2, m2 = tensorize(changed, max_objects=32)
        self.assertTrue((geo == g2).all() and (cat == c2).all() and (mask == m2).all())

    def test_no_silent_id_loss_or_old_geometry_fallback(self):
        state = example()
        records = [dict(id=o['track_id'], status='failed') for o in state['objects']]
        result, simple = export_state(state, records)
        self.assertEqual(simple, [])
        self.assertEqual(len(result['objects']), 17)
        self.assertTrue(all(not o['center_local_m']['valid'] and not o['size_aabb_local_m']['valid']
                            for o in result['objects']))
        with self.assertRaisesRegex(ValueError, 'every inventory ID'):
            export_state(state, records[:-1])

    def test_original_pixels_and_fov_intrinsics(self):
        obj = dict(bbox2d_visible_xyxy_norm=dict(value=[.1, .2, .8, .9]))
        np.testing.assert_allclose(prompt_pixels(obj, (1000, 500)), [100, 100, 800, 450])
        np.testing.assert_allclose(intrinsics(1000, 500, 90), [[500, 0, 500], [0, 500, 250], [0, 0, 1]])
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_official_single_prompt_api_preserves_original_pixels_and_preprocessed_K(self):
        import torch
        obj = example()['objects'][0]
        data = dict(images=torch.zeros(1, 3, 16, 16), intrinsics=torch.eye(3),
                    input_hw=(16, 16), original_hw=(500, 1000), padding=(0, 0, 4, 4))
        calls = []
        def model(**kw):
            calls.append(kw)
            return ([torch.tensor([kw['input_boxes'][0]])],
                    [torch.tensor([[1., 2., 5., 6., 2., 4., 1., 0., 0., 0.]])],
                    [torch.tensor([.4])], [torch.tensor([.8])], [torch.tensor([.5])],
                    [torch.tensor([0])], None)
        modules = {'vis4d.data.const': SimpleNamespace(AxisMode=SimpleNamespace(OPENCV='opencv')),
                   'vis4d.op.box.box3d': SimpleNamespace(boxes3d_to_corners=lambda b, a: torch.tensor(corners()[None]))}
        with patch.dict('sys.modules', modules):
            result = infer_one(model, data, obj, (1000, 500), 'cpu')
            clock = deepcopy(obj)
            clock.update(raw_category='clock')
            clock['category_id']['value']=39
            clock_result = infer_one(model, data, clock, (1000, 500), 'cpu')
        self.assertEqual(result['id'], obj['track_id'])
        self.assertAlmostEqual(result['prompt_prediction_iou'], 1, places=6)
        self.assertEqual(calls[0]['prompt_text'], 'geometric: bed')
        self.assertEqual(calls[1]['prompt_text'], 'geometric: clock')
        self.assertEqual(clock_result['prompt_category'], 'clock')
        np.testing.assert_allclose(calls[0]['input_boxes'][0], prompt_pixels(obj, (1000, 500)))
        np.testing.assert_allclose(calls[0]['intrinsics'][0].numpy(), np.eye(3))

    def test_same_image_loader_and_existing_completion_reuse(self):
        path = ROOT/'qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json'
        source, image, _ = load_source(path, None, 90)
        records = [dict(id=o['track_id'], status='predicted', box3d_native=[1, 2, 5, 6, 2, 4, 1, 0, 0, 0],
                        corners_native=corners().tolist()) for o in source['objects']]
        result, _ = export_state(source, records)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            small = image.copy(); small.thumbnail((768, 768)); small.save(p/'local.png')
            write(p/'G_obs.json', result)
            projection_preview(image, result, intrinsics(*image.size, 90), p/'projection.png')
            self.assertTrue((p/'projection.png').is_file())
            inventory = reuse_inventory(p/'G_obs.json', small, result['vocabulary_sha256'], 32)
            layout = observed_state_layout(result, inventory)
            np.testing.assert_allclose(layout['objects'][0]['center'], [1, -2, 5])
            np.testing.assert_allclose(layout['objects'][0]['size'], [2, 4, 6])
            other = p/'other.png'; Image.new('RGB', image.size).save(other)
            with self.assertRaisesRegex(ValueError, 'Image differs'):
                load_source(path, other, 90)
        with self.assertRaisesRegex(ValueError, 'FOV differs'):
            load_source(path, None, 60)


if __name__ == '__main__':
    unittest.main()
