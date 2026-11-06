"""CPU tests of geometry bridge and the official add_object wrapper protocol."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from .hybrid import (context_tensors, decode_added, generate, local_box, make_context,
                     outside_view, matches_region, validate_proposal)
from .test_qwen_upstream import observed

STATS = dict(class_labels=['chair', 'table', 'start', 'end'],
    bounds_translations=[-20, -20, -20, 20, 20, 20],
    bounds_sizes=[0, 0, 0, 10, 10, 10], bounds_angles=[-np.pi, np.pi])


class FakeNetwork:
    def __init__(self, positions):
        self.positions = iter(positions)
        self.context_lengths = []

    def add_object(self, room_mask, class_label, boxes, device, text):
        assert text and room_mask.shape == (1, 1, 64, 64)
        assert boxes['class_labels'][..., -2:].sum() == 0, 'START/END leaked into context'
        self.context_lengths.append(boxes['class_labels'].shape[1])
        native = dict(category='table', location=next(self.positions), size=[.5, .5, .5], rotation=0.)
        new = context_tensors([native], STATS, device)
        new['class_labels'] = class_label
        result = {}
        for key, value in boxes.items():
            result[key] = torch.cat([torch.zeros_like(value[:, :1]), value, new[key], torch.zeros_like(value[:, :1])], 1)
        return result


class HybridTests(unittest.TestCase):
    def test_two_stage_files_with_fake_models(self):
        import argparse
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from ..common import read, write, write_rows
        from . import hybrid
        class SemanticModel:
            def __init__(self, model):
                pass
            def ask(self, image, prompt, path, **kwargs):
                assert 'GT_SECRET' not in prompt
                return dict(description='A room with a chair.',
                            candidates=[dict(category='table', relation='Behind the camera',
                                             evidence='A chair suggests a possible table beyond this partial view.',
                                             region='behind_camera')])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root/'bundle'; bundle.mkdir()
            observer = root/'observer'; (observer/'scene').mkdir(parents=True)
            state = observed(); write(observer/'scene/G_obs.json', state)
            gt = deepcopy(state); gt['room'] = {'secret': 'GT_SECRET'}
            write(root/'gt.json', gt)
            gt_hidden = deepcopy(gt); gt_hidden['scope'] = 'hidden'
            gt_hidden['objects'][0]['center_local_m']['value'] = [0, 0, -3]
            write(root/'gt_hidden.json', gt_hidden)
            Image.new('RGB', (64, 64)).save(root/'local.png')
            write_rows(bundle/'inputs.jsonl', [dict(id='scene', building='house', local_rgb=str(root/'local.png'), hfov_deg=90)])
            write_rows(bundle/'references.jsonl', [dict(id='scene', gt_obs=str(root/'gt.json'), gt_hidden=str(root/'gt_hidden.json'))])
            write(bundle/'bundle.json', dict(max_objects=160))
            args = argparse.Namespace(bundle=bundle, observer=observer, output=root/'out',
                repo=root, model=root/'fake', weights=root/'fake', device='cpu', camera_height=1.5, attempts=3, offscreen=True)
            with patch.object(hybrid, 'Qwen', SemanticModel), patch.object(hybrid, 'official_stats', return_value=STATS):
                hybrid.propose(args)
            from . import offscreen_review
            # Semantic review must work before M3D inference exists.
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--run', str(args.output)]):
                offscreen_review.main()
            self.assertNotIn('pred', read(args.output/'review/offscreen_metrics.json')['records'][0])
            with patch('qwen_edit_pano.world_pipeline.completion.complete.load_pretrained',
                       return_value=(FakeNetwork([[0, 1.5, -3], [0, 1.5, -3]]), STATS)):
                hybrid.run(args)
            for condition in ('gt', 'pred'):
                d = args.output/'completion'/condition/'scene/seed0'
                self.assertEqual(read(d/'status.json')['generated'], 1)
                self.assertEqual(read(d/'G_full.json')['objects'][:1], state['objects'])
            with patch('sys.argv', ['review', '--bundle', str(bundle), '--run', str(args.output)]):
                offscreen_review.main()
            record = read(args.output/'review/offscreen_metrics.json')['records'][0]
            self.assertEqual(record['pred']['raw_outside_view'], 1)
            self.assertEqual(record['gt_definitely_offscreen'], 1)

    def test_offscreen_is_not_in_frame_occlusion(self):
        from .offscreen_review import offscreen_reference
        state = observed(); state['scope'] = 'hidden'
        # Marking a visible-frustum box hidden does not make it an offscreen reference.
        self.assertEqual(len(offscreen_reference(state, 90, (100, 100))['objects']), 0)
        state['objects'][0]['center_local_m']['value'] = [0, 0, -3]
        self.assertEqual(len(offscreen_reference(state, 90, (100, 100))['objects']), 1)

    def test_region_not_confused_with_generic_hidden(self):
        _, right = local_box(dict(location=[5, 0, 3], size=[.5]*3, rotation=0), 0)
        self.assertTrue(outside_view(right, 90, (100, 100)))
        self.assertFalse(matches_region(right, 'left', 90, (100, 100)))
        self.assertTrue(matches_region(right, 'right', 90, (100, 100)))
        self.assertTrue(matches_region(right, 'unspecified', 90, (100, 100)))
        proposal = dict(description='Room', candidates=[dict(category='table', relation='outside',
                                                           region='occluded', evidence='chair')])
        with self.assertRaises(ValueError):
            validate_proposal(proposal, ['table'], True)

    def test_bridge_half_extents_and_frame(self):
        obs = observed()
        context, omitted = make_context(obs, STATS, 1.5)
        center, corners = local_box(context[0], 1.5)
        np.testing.assert_allclose(center, obs['objects'][0]['center_local_m']['value'])
        np.testing.assert_allclose(np.ptp(corners, axis=0), obs['objects'][0]['size_aabb_local_m']['value'])
        self.assertEqual(omitted, [])

    def test_full_box_frustum(self):
        def corners(center, half):
            return local_box(dict(location=center, size=[half]*3, rotation=0), 0)[1]
        self.assertFalse(outside_view(corners([0, 0, 3], .5), 90, (100, 100)))
        self.assertTrue(outside_view(corners([0, 0, -3], .5), 90, (100, 100)))
        # Centre outside is insufficient when the box crosses the right frustum plane.
        self.assertFalse(outside_view(corners([3.1, 0, 3], .5), 90, (100, 100)))
        self.assertTrue(outside_view(corners([5, 0, 3], .5), 90, (100, 100)))

    def test_rejected_not_recycled_and_observed_preserved(self):
        obs = observed(); original = deepcopy(obs)
        proposal = dict(description='A room', candidates=[dict(category='table', relation='behind camera')]*2)
        network = FakeNetwork([[0, 1.5, 3], [0, 1.5, -3], [3, 1.5, -3]])
        hidden, full, audit, omitted = generate(obs, proposal, STATS, network,
            (100, 100), 90, 1.5, 2, 160, 'cpu')
        self.assertEqual(network.context_lengths, [1, 1, 2])
        self.assertEqual(len(hidden['objects']), 2)
        self.assertEqual(full['objects'][:1], original['objects'])
        self.assertEqual(obs, original)
        self.assertEqual(audit[0]['reason'], 'not_wholly_outside_input_view')
        from ..oracle_model import tensorize
        _, geometry, mask = tensorize(full)
        self.assertEqual(int(mask.sum()), 4)  # global token + observed + two additions
        np.testing.assert_allclose(geometry[0, 2, :6].numpy(), [0, 0, -3, 1, 1, 1], atol=1e-5)

    def test_all_rejected_is_not_completion_success(self):
        obs = observed()
        hidden, full, audit, _ = generate(obs,
            dict(description='Room', candidates=[dict(category='table', relation='outside')]),
            STATS, FakeNetwork([[0, 1.5, 3]]), (100, 100), 90, 1.5, 1, 160, 'cpu')
        self.assertEqual(hidden['objects'], [])
        self.assertEqual(full['objects'], obs['objects'])
        self.assertEqual(audit[-1]['outcome'], 'no_accepted_geometry')


if __name__ == '__main__':
    unittest.main()
