"""CPU regression of official-interface routing and graph assembly."""
import tempfile
import unittest
from pathlib import Path

import torch

from ..common import read, write
from ..oracle_model import tensorize
from .complete import main
from .geometry import graph, qwen_corners, world_object


class FakeOfficial:
    def token(self, index):
        label = torch.zeros(1, 1, 3); label[0, 0, index] = 1
        return dict(class_labels=label, translations=torch.zeros(1, 1, 3),
                    sizes=torch.zeros(1, 1, 3), angles=torch.zeros(1, 1, 1))

    def result(self, pieces):
        return {k: torch.cat([p[k] for p in pieces], 1) for k in pieces[0]}

    def generate_boxes(self, room_mask, max_boxes, device, text):
        self.text = text; self.mode = 'text'
        return self.result([self.token(1), self.token(0), self.token(2)])

    def complete_scene(self, boxes, room_mask, max_boxes, device, text):
        self.text = text; self.mode = 'joint'
        return self.result([self.token(1), boxes, self.token(0), self.token(2)])


class PretrainedTests(unittest.TestCase):
    def run_mode(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            center, corners = qwen_corners([0., 0., 2., 1., 1., 1., 0., 0., 0.])
            original = graph('test', 'val', 'observed',
                [world_object('obs_0000', 'chair', center, corners, 'observer', True)], 'test')
            write(root/'G_obs.json', original)
            write(root/'observed.json', dict(room_type='living room', boxes=[dict(
                category='chair', center_local=center.tolist(), corners_local=corners.tolist(), observed=True)]))
            (root/'caption.txt').write_text('A chair is visible in a partial room view.')
            network = FakeOfficial()
            stats = dict(class_labels=['chair', 'start', 'end'],
                bounds_translations=[-10., -10., -10., 10., 10., 10.],
                bounds_sizes=[.1, .1, .1, 2., 2., 2.], bounds_angles=[-3.14, 3.14])
            main(['--observed', str(root/'observed.json'), '--weights', str(root/'unused.pt'),
                  '--output', str(root/'out'), '--text-file', str(root/'caption.txt'),
                  '--seeds', '0', '--device', 'cpu', '--mode', mode], runtime=(network, stats))
            self.assertEqual(network.text, (root/'caption.txt').read_text())
            out = root/'out/seed000'
            self.assertTrue(read(out/'summary.json')['ended'])
            if mode == 'text-only':
                self.assertEqual(network.mode, 'text')
                self.assertFalse((out/'G_full.json').exists())
                self.assertEqual(len(read(out/'layout_scene.json')['objects']), 1)
            else:
                self.assertEqual(network.mode, 'joint')
                full = read(out/'G_full.json')
                self.assertEqual(full['objects'][:1], original['objects'])
                self.assertEqual(len(full['objects']), 2)
                self.assertEqual(tensorize(full, 160)[2].sum().item(), 3)

    def test_text_only_stays_in_native_scene_frame(self):
        self.run_mode('text-only')

    def test_joint_only_appends_new_objects(self):
        self.run_mode('completion')


if __name__ == '__main__':
    unittest.main()
