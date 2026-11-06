"""Small synthetic checks; no model loading, downloaded data, or GPU required."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import zipfile
import numpy as np

from .geometry import SIGNS
from .visible_probe import axis_maps, register, to_local
from .native_probe import ordered_objects
from . import visible_probe


class VisibleProbeTests(unittest.TestCase):
    def test_order_empty_and_single_scene(self):
        self.assertEqual(ordered_objects({'objects': []}), [])
        obj = dict(location=[1., 2., 3.])
        self.assertEqual(ordered_objects({'objects': [obj]}), [obj])

    def test_prepare_empty_and_oversized_rooms_after_interruption(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'layouts.zip'
            scenes = [dict(scene_id='house_region0', objects=[]),
                      dict(scene_id='house_region1', objects=[{}]*121)]
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('M3DLayout_json/mp3d_val.json', json.dumps(dict(scenes=scenes)))
            entries = []
            for region in range(2):
                directory = root/f'sample{region}'
                directory.mkdir()
                world = directory/'world_state.json'
                world.write_text(json.dumps(dict(room=dict(region_id=region, type='b', supervision_eligible=True))))
                (directory/'provenance.json').write_text(json.dumps(dict(sources={str(world): 'fixture'})))
                entries.append(dict(id=f'sample{region}', split='val', paired=dict(scene_id='house'),
                                    graph=str(directory/'G_full.json')))
            bundle = root/'prepared'
            # Reproduce the output left by an interrupted prepare.
            (bundle/'samples'/'sample0').mkdir(parents=True)
            args = SimpleNamespace(bundle=bundle, repo=root, archive=archive, oracle=root,
                                   selection_seed=0, per_building=3)
            with patch.object(visible_probe, 'official_stats', return_value={'class_labels': ['chair']}), \
                 patch.object(visible_probe, 'rows', side_effect=[[], entries]):
                with self.assertRaisesRegex(SystemExit, 'No unambiguous registrations'):
                    visible_probe.prepare(args)
            audit = json.loads((bundle/'audit.json').read_text())
            self.assertEqual(len(audit['records']), 2)
            self.assertTrue(all(r['reason']=='native_capacity_or_vocabulary' for r in audit['records']))
            self.assertEqual(json.loads((bundle/'bundle.json').read_text())['samples'], [])
            self.assertEqual((bundle/'inputs.jsonl').read_text(), '')
            # Completed manifests must not be silently overwritten.
            with self.assertRaises(FileExistsError):
                visible_probe.prepare(args)

    def test_unique_room_registration(self):
        source = [dict(house_object_id=i, category_id=cat, center_world=center, radii=size)
                  for i, (cat, center, size) in enumerate([
                      (3, [1., 2., .6], [.3, .4, .5]),
                      (5, [3., -1., .4], [.6, .7, .8]),
                      (11, [-2., 0., .7], [.9, 1., 1.1]),
                      (7, [0., 4., 1.], [1.2, 1.3, 1.4])])]
        matrix = list(axis_maps())[3]
        offset = np.array([2., -.3, -1.])
        names = ['chair', 'table', 'bed', 'cabinet']
        native = [dict(category=name, location=(np.asarray(s['center_world'])@matrix.T+offset).tolist(),
                       size=s['radii'], rotation=0.) for name, s in zip(names, source)]
        result, reason = register(native, source)
        self.assertIsNone(reason)
        np.testing.assert_allclose(result['world_to_native'], matrix)
        np.testing.assert_allclose(result['offset'], offset)
        self.assertEqual(result['source_indices'], [0, 1, 2, 3])
        # A per-object displacement must not be explained by a room-wide frame.
        native[0]['location'][0] += .4
        result, reason = register(native, source)
        self.assertIsNone(result)

    def test_camera_AABB_from_rotated_corners(self):
        obj = dict(category='table', location=[1., 2., 3.], size=[1., 2., .5], rotation=np.pi/4)
        identity = dict(world_to_native=np.eye(3).tolist(), offset=[0., 0., 0.])
        camera = dict(origin_world=[0., 0., 0.], basis_local_to_world=np.eye(3).tolist())
        local = to_local(obj, identity, camera)
        np.testing.assert_allclose(local['center_local_m'], [1, 2, 3])
        np.testing.assert_allclose(local['size_aabb_local_m'], [3/np.sqrt(2), 4, 3/np.sqrt(2)])
        # Axis reflection is allowed by the existing camera basis convention.
        camera['basis_local_to_world'] = np.diag([-1., 1., 1.]).tolist()
        reflected = to_local(obj, identity, camera)
        np.testing.assert_allclose(reflected['center_local_m'], [-1, 2, 3])
        np.testing.assert_allclose(reflected['size_aabb_local_m'], local['size_aabb_local_m'])

    def test_native_half_extents(self):
        obj = dict(category='chair', location=[0., 0., 0.], size=[.2, .3, .4], rotation=0.)
        registration = dict(world_to_native=np.eye(3).tolist(), offset=[1., 2., 3.])
        camera = dict(origin_world=[-1., -2., -3.], basis_local_to_world=np.eye(3).tolist())
        local = to_local(obj, registration, camera)
        np.testing.assert_allclose(local['corners_local'], SIGNS * [.2, .3, .4], atol=1e-12)
        np.testing.assert_allclose(local['size_aabb_local_m'], [.4, .6, .8])


if __name__ == '__main__':
    unittest.main()
