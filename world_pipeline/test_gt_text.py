"""User-run CPU tests for the GT text projection; no model/GPU/network."""
import copy
import unittest
from .common import SCHEMA, FRAME, field
from .gt_text import spatial_text, direction, shuffle_sources


def fixture():
    def obj(key, center):
        return dict(track_id=key, category_id=field(3), center_local_m=field(center),
                    size_aabb_local_m=field([.5, 1., .5]), bbox2d_visible_xyxy_norm=field(None),
                    in_primary_room=field(None), local_evidence='unknown')
    return dict(schema_version=SCHEMA, frame=FRAME, id='PRIVATE_SAMPLE', split='train',
                stage='gt', scope='full', vocabulary_sha256='fixture',
                objects=[obj('PRIVATE_OBJECT_A',[1.,0.,0.]), obj('PRIVATE_OBJECT_B',[1.,0.,0.])],
                room={'secret': 'NEVER_USE_ROOM'}, caption='NEVER_USE_CAPTION'), {
                    'category_mapping_sha256':'fixture', 'categories':{'3':'chair'}}


class TextTests(unittest.TestCase):
    def test_camera_directions(self):
        self.assertEqual([direction(v) for v in ([0,0,1],[1,0,0],[0,0,-1],[-1,0,0])],
                         ['front','right','back','left'])
        self.assertEqual(direction([-1,0,1]), 'front-left')
        self.assertEqual(direction([0,2,0]), 'at-camera-horizontal-position')

    def test_counts_order_invariance_and_no_identity_caption_leak(self):
        graph, vocab = fixture()
        text, info = spatial_text(graph, vocab)
        self.assertIn('right: 2 x chair (near).', text)
        self.assertEqual(info['object_count'], 2)
        self.assertNotIn('PRIVATE', text)
        self.assertNotIn('NEVER_USE', text)
        graph['objects'].reverse()
        self.assertEqual(text, spatial_text(graph,vocab)[0])

    def test_unknown_is_not_origin_and_metric_has_size(self):
        graph, vocab = fixture()
        graph['objects'][0]['center_local_m'] = field(None)
        self.assertIn('position-unknown', spatial_text(graph,vocab)[0])
        metric = spatial_text(graph,vocab,'metric')[0]
        self.assertIn('center=unknown', metric)
        self.assertIn('extent=(0.50,1.00,0.50)', metric)
        self.assertIn('center=(1.00,0.00,0.00)', metric)

    def test_near_mid_far_thresholds(self):
        graph, vocab = fixture()
        graph['objects'][0]['center_local_m'] = field([0,0,2])
        graph['objects'][1]['center_local_m'] = field([0,0,4])
        text, _ = spatial_text(graph,vocab)
        self.assertIn('1 x chair (mid)',text)
        self.assertIn('1 x chair (far)',text)

    def test_only_full_gt_and_pinned_vocabulary(self):
        graph, vocab = fixture()
        bad = copy.deepcopy(graph)
        bad['stage'] = 'observer'
        with self.assertRaises(ValueError): spatial_text(bad,vocab)
        bad = copy.deepcopy(graph)
        bad['vocabulary_sha256'] = 'changed'
        with self.assertRaises(ValueError): spatial_text(bad,vocab)

    def test_shuffle_derangement_is_reproducible_and_bijective(self):
        ids = ['a','b','c','d']
        mapping = shuffle_sources(ids,12)
        self.assertEqual(mapping,shuffle_sources(ids[::-1],12))
        self.assertEqual(set(mapping.values()),set(ids))
        self.assertTrue(all(k != v for k,v in mapping.items()))
        with self.assertRaises(ValueError): shuffle_sources(['a'],0)


if __name__ == '__main__':
    unittest.main()
