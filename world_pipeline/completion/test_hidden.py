"""CPU contract tests; no model download or GPU work."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..common import FRAME, SCHEMA, field
from ..oracle_model import tensorize
from .hidden_data import CATEGORIES, VERSION, fit_bounds, prefix, split_scene
from .hidden_model import batch, build, encode, generate, objective
from .evaluate_hidden import matches


def obj(i, x, visible=True):
    return dict(track_id=str(i), category_id=field(3), center_local_m=field([x, 0., 2.]),
        size_aabb_local_m=field([1., 1., 1.]), bbox2d_visible_xyxy_norm=field(None),
        in_primary_room=field(None), local_evidence='visible' if visible else 'unobserved')


def graph(items, scope='full'):
    return dict(id='test', schema_version=SCHEMA, frame=FRAME, scope=scope, objects=items)


class HiddenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.observed, cls.hidden = split_scene(graph([obj(0, 0, False), obj(1, 3), obj(2, -2, False)]))
        cls.scene = dict(observed=cls.observed, hidden=cls.hidden)
        cls.meta = dict(version=VERSION, categories=CATEGORIES, max_objects=5,
                        bounds=fit_bounds([cls.scene]))
        repo = Path(__file__).resolve().parents[2] / 'third_party/M3DLayout-code'
        cls.model, _ = build(repo, 'cpu')

    def test_observed_is_not_spatial_prefix(self):
        for k in range(3):
            context, target = prefix(self.scene, k)
            self.assertEqual([o['id'] for o in context], ['1'] + ['2', '0'][:k])
            self.assertEqual(None if target is None else target['id'], ['2', '0', None][k])

    def test_empty_context_and_end(self):
        b = batch([(dict(observed=[], hidden=[]), 0)], self.meta, 'cpu')
        self.assertEqual(b['class_labels'].shape[1], 0)
        self.assertEqual(b['class_labels_tr'][0, 0, -1], 1)
        loss = objective(self.model(b), b)
        self.assertTrue(torch.isfinite(loss))

    def test_train_inference_encoding_and_padding(self):
        model = self.model.eval()
        mixed = batch([(self.scene, 0), (self.scene, 2)], self.meta, 'cpu')
        single = batch([(self.scene, 0)], self.meta, 'cpu')
        with torch.no_grad():
            a = model(mixed).class_labels[0]
            b = model(single).class_labels[0]
            boxes = model.start_symbol('cpu')
            for key, value in encode(self.observed, self.meta).items():
                boxes[key] = torch.cat([boxes[key], value[None]], 1)
            c = model.hidden2output.class_layer(model._encode(boxes, torch.ones(1, 1, 64, 64)))[0]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b, c, atol=1e-5, rtol=1e-5)

    def test_backward_and_end_geometry_mask(self):
        model = self.model
        model.zero_grad(set_to_none=True)
        b = batch([(self.scene, 2)], self.meta, 'cpu')
        objective(model(b), b).backward()
        self.assertGreater(model.hidden2output.class_layer.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.hidden2output.centroid_layer_x[0].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        objective(model(batch([(self.scene, 0)], self.meta, 'cpu')), batch([(self.scene, 0)], self.meta, 'cpu')).backward()
        self.assertGreater(model.hidden2output.centroid_layer_x[0].weight.grad.abs().sum().item(), 0)

    def test_generated_graph_preserves_observed_and_tensorizes(self):
        # Deterministic fake sampling; the real architecture was exercised above.
        model = copy.deepcopy(self.model).eval()
        head = model.hidden2output
        class Next(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, features):
                x = torch.full((1, 1, len(CATEGORIES) + 2), -1000.)
                x[..., 0 if self.calls == 0 else -1] = 0
                self.calls += 1
                return x
        head.class_layer = Next()
        g = graph([obj(7, 1)], 'observed')
        with patch('qwen_edit_pano.world_pipeline.completion.hidden_model.sample_geometry',
                   return_value=(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), torch.zeros(1, 1, 1))):
            result, status = generate(model, g, self.meta, 0, 'cpu')
        self.assertEqual(result['objects'][:1], g['objects'])
        self.assertTrue(status['ended'])
        category, geometry, mask = tensorize(result, max_objects=5)
        self.assertEqual(mask.sum().item(), 3)
        np.testing.assert_allclose(geometry[0, 2, 3:6], [.8, .8, .8])
        self.assertEqual(category[0, 2].item(), 3)

    def test_matching(self):
        self.assertEqual(matches(self.hidden, self.hidden, .25), (2, 0.))
        self.assertEqual(matches([], self.hidden, .25), (0, 0.))


if __name__ == '__main__':
    unittest.main()
