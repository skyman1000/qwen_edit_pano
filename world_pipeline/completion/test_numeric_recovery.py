"""Regression on the actual blocked layouts, with no model resampling."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import unittest
from unittest.mock import patch

from ..common import ROOT,read,sha
from . import assembly_spatial as spatial,assembly_organized_geometry as geometry
from .assembly_numeric_recovery import plan

ACTUAL=ROOT/'qwen_edit_pano/outputs/organized_multiscene_recovery_20261028_094728_566800'
FAILED={'8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':5,'8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029':23}


class NumericRecoveryTests(unittest.TestCase):
    def test_plan_reuses_eight_successes_and_requires_no_model(self):
        p=plan(ACTUAL);self.assertEqual(len(p['complete_worlds']),8);self.assertEqual(len(p['pending']),2)
        self.assertEqual(p['model_calls_required'],0);self.assertEqual({c['scene']:c['previous_overlaps'] for c in p['pending']},FAILED)

    def test_actual_failure_is_removed_by_search_alone_with_unchanged_constraints(self):
        for sid,overlaps in FAILED.items():
            c=ACTUAL/'scenes'/sid/'completion/end_to_end/qwen/completion';d=c/'world_01';execution=read(d/'execution.json')
            layout=execution['layout'];world=read(d/'semantic_world.json');observed=read(c/'G_obs.json');case=read(c/'case.json');plane=read(c/'support_plane.json')
            before=deepcopy((layout,world,observed));frozen={p:sha(p) for p in (d/'dimension_stage.json',d/'semantic_world.json',c/'causal_model.json',c/'G_obs.json')}
            original=spatial.articulated_repair
            with patch.object(spatial,'articulated_repair',partial(original,collision_rounds=8)):
                baseline=geometry.compile_layout(case,observed,world,layout,plane)
            self.assertEqual(len(baseline['review']['program_solver']['final_overlaps']),overlaps)
            self.assertFalse(baseline['review']['geometry_contract_passed'])
            solved=geometry.compile_layout(case,observed,world,layout,plane);r=solved['review'];s=r['program_solver']
            self.assertTrue(r['geometry_contract_passed']);self.assertEqual(s['final_overlaps'],[])
            self.assertGreater(s['collision_search_rounds_used'],8);self.assertLessEqual(s['collision_search_rounds_used'],s['collision_search_round_budget'])
            self.assertEqual(s['collision_search_termination'],'collision_free');self.assertFalse(s['collision_search_budget_exhausted'])
            self.assertEqual(s['contact_and_frustum_linear_violations'],0);self.assertEqual(r['attachment_contact_failures'],0)
            self.assertEqual(r['spatial_program_violations'],[]);self.assertEqual(r['unsatisfied_declared_ordering'],[])
            previous={o['id']:o for o in execution['solved']['items']};current={o['id']:o for o in solved['items']}
            self.assertEqual(set(current),set(previous))
            for oid,item in current.items():self.assertEqual(item['size'],previous[oid]['size']);self.assertEqual(item['center'][1],previous[oid]['center'][1])
            self.assertEqual((layout,world,observed),before)
            for p,h in frozen.items():self.assertEqual(sha(p),h)

    def test_search_budget_is_explicit_and_bounded(self):
        args=(dict(hfov=90),dict(objects=[]),{},{},{},{})
        for bad in (0,129,True,1.5):
            with self.assertRaises(ValueError):spatial.articulated_repair(*args,collision_rounds=bad)
        spatial.articulated_repair(*args,collision_rounds=8)


if __name__=='__main__':unittest.main()
