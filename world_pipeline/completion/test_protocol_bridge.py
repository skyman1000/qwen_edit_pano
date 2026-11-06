"""Real val6 response regressions; these are CPU parsing/geometry, not model quality."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from ..common import read
from ..oracle_model import tensorize
from .category_bridge import category
from .semantic_world import visible_inventory,make_graph
from .placement_program import compile_program

DATA=read(Path(__file__).parent/'testdata/val6_run001_protocol_failures.json')


class ProtocolBridgeTests(unittest.TestCase):
    def test_real_clock_and_candles_use_official_coarse_ids_without_dropping_instances(self):
        for name in ('bedroom','bath'):
            raw=deepcopy(DATA[name]);original=deepcopy(raw)
            inventory=visible_inventory(raw,32)
            self.assertEqual(len(raw),len(inventory))
            graph=make_graph(inventory,None,'test','not_estimated','observed')
            cat,geometry,mask=tensorize(graph,32)
            for i,item in enumerate(raw):
                np.testing.assert_allclose(inventory[i]['bbox_2d'],np.array(item['bbox_2d'])/1000)
                if item['category'] in ('clock','candle'):
                    self.assertEqual(graph['objects'][i]['raw_category'],item['category'])
                    self.assertEqual(int(cat[0,i+1]),39)
                    self.assertEqual(graph['objects'][i]['category_mapping']['category_id'],39)
            self.assertEqual(int(mask.sum()),len(raw)+1)
            self.assertFalse(geometry[...,6:].any())
            self.assertEqual(raw,original)
        missing=next(o for o in visible_inventory(DATA['bath'],32) if o.get('description_source'))
        self.assertEqual(missing['description'],'candle')

    def test_unknown_labels_bad_boxes_and_structural_classes_still_fail(self):
        for name in ('fictional_new_category','floor','unlabeled'):
            with self.assertRaises(ValueError):category(name)
        bad=deepcopy(DATA['bath']);bad[0]['bbox_2d']=[100,200,50,400]
        with self.assertRaises(ValueError):visible_inventory(bad,32)

    def test_above_below_irrelevant_alignment_does_not_change_declared_geometry(self):
        for name in ('sauna','hallway'):
            fixture=DATA[name];raw=deepcopy(fixture['program']);original=deepcopy(raw)
            answer,report=compile_program(raw,fixture['observed'],fixture['layout'],90.,8)
            trimmed=deepcopy(raw)
            for group in trimmed['groups']:
                for o in group['objects']:
                    if o.get('direction') in ('above','below'):o.pop('vertical_alignment',None)
            expected,_=compile_program(trimmed,fixture['observed'],fixture['layout'],90.,8)
            self.assertEqual(answer,expected)
            self.assertEqual(len(report['unused_parameters']),1)
            self.assertEqual(raw,original)
            root,child=answer['additions']
            sign=1 if raw['groups'][0]['objects'][1]['direction']=='above' else -1
            self.assertAlmostEqual(sign*(child['center'][1]-root['center'][1]),
                (root['size'][1]+child['size'][1])/2+.1)


if __name__=='__main__':unittest.main()
