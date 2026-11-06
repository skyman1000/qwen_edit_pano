"""CPU regression tests for the Observer's model-output boundary."""
import json
import unittest

from .geometry import (graph, observer_boxes, observer_categories, parse_json,
                       qwen_corners, world_object)

BOX = [1., 2., 3., 2., 4., 6., 0., 0., 0.]


class ObserverResponsesTest(unittest.TestCase):
    def test_category_variants_and_unknown(self):
        names, room, changes, skipped = observer_categories(dict(categories=[
            'appliance', 'appliances', 'misc.', 'Dining Chairs', 'glass-panel-door',
            'unknown furniture thing', None, {'label': 'chair'}]))
        self.assertEqual(names, ['appliances', 'misc', 'chair', 'door'])
        self.assertEqual(room, 'unknown')
        self.assertEqual(len(changes), 4)
        self.assertEqual(len(skipped), 3)

    def test_same_adapter_category_is_queried_once(self):
        names, _, _, _ = observer_categories(dict(categories=['chair', 'armchair', 'dining_chair', 'tv', 'tv_monitor']))
        self.assertEqual(names, ['chair', 'tv_monitor'])

    def test_invalid_categories(self):
        for value in (None, [], {}, {'categories': 'chair'}):
            with self.assertRaises(ValueError):
                observer_categories(value)

    def test_single_list_and_wrapped_boxes(self):
        item = {'bbox_3d': BOX}
        for value in (item, [item], {'objects': [item]}, {'boxes': [item]}):
            self.assertEqual(observer_boxes(value, 'chair', 8, set()), ([BOX], []))

    def test_bad_items_do_not_lose_good_box(self):
        items = [None, {}, 'chair', {'bbox_3d': None}, {'bbox_3d': BOX[:8]}]
        for invalid in (None, '1.0', True, float('nan'), float('inf')):
            items.append({'bbox_3d': [invalid] + BOX[1:]})
        items.extend([{'bbox_3d': BOX[:3] + [0] + BOX[4:]}, {'bbox_3d': BOX}])
        boxes, skipped = observer_boxes(items, 'chair', 8, set())
        self.assertEqual(boxes, [BOX])
        self.assertEqual(len(skipped), len(items)-1)
        json.dumps(skipped, allow_nan=False)

    def test_duplicate_and_cap(self):
        seen = set()
        observer_boxes([{'bbox_3d': BOX}], 'chair', 8, seen)
        boxes, skipped = observer_boxes([{'bbox_3d': BOX}], 'dining_chair', 8, seen)
        self.assertEqual(boxes, [])
        self.assertEqual(skipped[0]['reason'], 'exact_duplicate')
        boxes, skipped = observer_boxes([{'bbox_3d': BOX}], 'table', 0, seen)
        self.assertEqual(skipped[0]['reason'], 'instance_cap')

    def test_json_fences_and_truncation(self):
        for raw in ('[]', '```json\n[]\n```', '```json []```'):
            self.assertEqual(parse_json(raw), [])
        for raw in ('```', '```json\n[{"bbox_3d":[1,2', '[{"bbox_3d":[]},'):
            with self.assertRaises(json.JSONDecodeError):
                parse_json(raw)

    def test_encoder_contract(self):
        from ..oracle_model import tensorize
        center, corners = qwen_corners(BOX)
        obj = world_object('obs_0000', 'appliances', center, corners, 'observer', True)
        categories, geometry, mask = tensorize(graph('test', 'val', 'observed', [obj], 'test'), 160)
        self.assertEqual(categories[0, 1].item(), 37)
        self.assertEqual(geometry[0, 1].tolist(), [1., -2., 3., 2., 4., 6., 1., 1.])
        self.assertEqual(mask.sum().item(), 2)


if __name__ == '__main__':
    unittest.main()
