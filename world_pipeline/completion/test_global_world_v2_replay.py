"""Regression against the five-scene v1 failure, plus cross-scene invariants.

Stored Qwen outputs are replayed without claiming a new model inference. Explicit
space parameters below are test inputs, not inferred room truth or new worlds.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ..common import ROOT, read, sha, write
from . import global_world as pipeline, global_world_model as model, global_world_observation as contract

SOURCE = ROOT/'qwen_edit_pano/outputs/global_world_srun_001'


class SemanticContractTests(unittest.TestCase):
    def test_coarse_category_uses_only_explicit_name_and_keeps_provenance(self):
        raw = dict(objects=[dict(name='stainless steel range hood', category='appliance',
            bbox_1000=[100,100,300,400], on_floor=False)], evidence_description='one range hood', review_status='complete')
        before = deepcopy(raw);clean = contract.inventory_answer(raw)
        self.assertEqual(raw, before)
        self.assertEqual(clean['objects'][0]['category'], 'stainless steel range hood')
        self.assertEqual(clean['objects'][0]['reported_category'], 'appliance')
        state, _ = contract.export_inventory(clean, dict(id='fixture'))
        self.assertEqual(state['objects'][0]['raw_category'], 'stainless steel range hood')
        self.assertEqual(state['objects'][0]['description'], 'stainless steel range hood')

    def test_no_specific_name_is_rejected_instead_of_guessing(self):
        raw = dict(objects=[dict(name='furniture', category='furniture', bbox_1000=[1,1,20,20], on_floor=True)],
            evidence_description='unresolved object', review_status='complete')
        with self.assertRaisesRegex(ValueError, 'specific'):contract.inventory_answer(raw)

    def test_real_empty_scene_is_legal_but_unresolved_detection_is_not(self):
        empty = dict(objects=[], evidence_description='bare surfaces with no identifiable objects', review_status='empty_scene')
        self.assertEqual(contract.inventory_answer(empty)['objects'], [])
        empty['review_status'] = 'unresolved'
        with self.assertRaises(ValueError):contract.inventory_answer(empty)

    def test_nested_boxes_warn_without_automatic_instance_deletion(self):
        obj = dict(name='chair', category='chair', bbox_1000=[100,100,500,900], on_floor=True)
        small = dict(obj, bbox_1000=[200,300,300,400])
        clean = contract.inventory_answer(dict(objects=[obj,small], evidence_description='ambiguous overlapping chair boxes', review_status='complete'))
        audit = contract.quality_audit(clean)
        self.assertEqual(len(clean['objects']),2);self.assertEqual(len(audit['warnings']),1)


@unittest.skipUnless(SOURCE.is_dir(), 'Saved v1 five-scene experiment not available')
class ActualReplyReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base = Path(self.tmp.name)

    def test_all_34_saved_review_instances_retain_names_boxes_and_floor_flags(self):
        count = 0
        for path in SOURCE.glob('scenes/*/observation/review_calls/whole_image_grounding.call.json'):
            saved = read(path);self.assertEqual(saved['status'],'accepted')
            clean = contract.inventory_answer(saved['raw'])
            for original, normalized in zip(saved['raw']['objects'], clean['objects']):
                self.assertEqual(original['name'], normalized['name'])
                self.assertEqual(original['bbox_1000'], normalized['bbox_1000'])
                self.assertEqual(original['on_floor'], normalized['on_floor'])
                self.assertNotIn(normalized['category'].casefold(), contract.COARSE)
                self.assertEqual(normalized['reported_category'], original['category'])
                count += 1
        self.assertEqual(count,34)

    def test_containment_is_compiled_for_every_saved_metric_scene_and_slack(self):
        checked = 0
        for path in SOURCE.glob('scenes/*/completion/G_obs.json'):
            state = read(path);original = deepcopy(state)
            grounding = read(path.parents[1]/'observation/qwen/observer2d/grounding.json')
            packet = model.observation_packet(state, grounding)
            self.assertTrue(all(o['description'] for o in packet['objects']))
            # Zero and heterogeneous clearances are explicit boundary-test inputs.
            for values in ([0.]*6, [.2,1.,3.,.8,.1,.3]):
                space = dict(extra_clearance_m=dict(zip(model.SPACE_DIRECTIONS,values)))
                room,audit = model.compile_space(space,packet)
                lo=np.array([room['x_min'],room['floor_y'],room['z_min']])
                hi=np.array([room['x_max'],room['ceiling_y'],room['z_max']])
                for o in packet['objects']:
                    c=np.array(o['center_m']);size=np.array(o['full_camera_aabb_m'])
                    self.assertTrue(np.all(c-size/2>=lo-1e-9));self.assertTrue(np.all(c+size/2<=hi+1e-9))
                self.assertFalse(audit['support_floor']['measured'])
                self.assertFalse(audit['walls_measured'])
            self.assertEqual(state,original);checked += 1
        self.assertEqual(checked,3)

    def test_old_absolute_numeric_templates_are_not_silently_accepted_as_v2(self):
        count = 0
        for path in SOURCE.glob('scenes/*/completion/calls/global_hypotheses.try*.json'):
            if path.name.count('.') != 2:continue
            raw = read(path)
            if 'hypotheses' not in raw:continue
            comp=path.parents[1];state=read(comp/'G_obs.json')
            packet=model.observation_packet(state,read(comp.parent/'observation/qwen/observer2d/grounding.json'))
            with self.assertRaisesRegex(ValueError,'extra_clearance'):
                model.hypotheses_answer(raw,packet,3)
            count += len(raw['hypotheses'])
        self.assertEqual(count,16)

    def test_two_real_failed_drafts_reach_independent_grounding(self):
        failed = [s for s in read(SOURCE/'suite.json')['scenes'] if read(SOURCE/'scenes'/s['id']/'observer2d_result.json')['status']=='failed']
        self.assertEqual(len(failed),2)
        for scene in failed:
            obs=self.base/scene['id'];obs.mkdir()
            __import__('shutil').copytree(SOURCE/'scenes'/scene['id']/'observation/draft',obs/'draft')
            __import__('shutil').copyfile(SOURCE/'scenes'/scene['id']/'observation/input.png',obs/'input.png')
            write(obs/'experiment.json',dict(id=scene['id'],hfov=scene['hfov'],image_size=scene['image_size']))
            class Independent:
                calls=0
                def ask(self,image,prompt,path,**kwargs):
                    self.calls+=1
                    if 'DRAFT_CONTEXT:' not in prompt:raise AssertionError('Failed draft must not be resubmitted')
                    # Deliberate test reply, never represented as real recognition.
                    value=dict(objects=[dict(name='fixture chair',category='chair',bbox_1000=[100,100,200,300],on_floor=True)],
                        evidence_description='CPU fixture recognition',review_status='complete')
                    write(Path(path).with_suffix('.json'),value);Path(path).write_text(json.dumps(value));return value
            provider=Independent();pipeline.observe2d_review_v2(obs,lambda:provider,1)
            self.assertEqual(provider.calls,1)
            self.assertTrue((obs/'qwen/observer2d/G_2d.json').exists())

    def test_five_scene_migration_reuses_three_2d_reviews_but_no_metric_geometry(self):
        tracked=[p for p in SOURCE.rglob('*') if p.is_file()]
        before={str(p):sha(p) for p in tracked}
        root=self.base/'migrated'
        with patch.object(pipeline,'new_output',side_effect=lambda p:(Path(p).mkdir(parents=True),Path(p))[1]):
            pipeline.prepare(root,recover_from=SOURCE)
        pipeline.verify(root)
        plan=read(root/'recovery_plan.json')
        self.assertEqual(len(plan['scenes']),5)
        self.assertEqual(sum(s['reviewed_2d_reused'] for s in plan['scenes']),3)
        self.assertEqual(len(list(root.glob('scenes/*/observation/qwen/observer2d/G_2d.json'))),3)
        self.assertFalse(list(root.rglob('G_obs.json')))
        self.assertTrue(all(not s['metric_geometry_reused'] for s in plan['scenes']))
        for path in root.glob('scenes/*/observation/qwen/observer2d/G_2d.json'):
            for obj in read(path)['objects']:self.assertNotIn(obj['raw_category'].casefold(),contract.COARSE)
        self.assertEqual(before,{str(p):sha(p) for p in tracked})


class FreshFailurePathTests(unittest.TestCase):
    def test_new_draft_failure_also_reaches_independent_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs=Path(tmp);Image.new('RGB',(64,64),'white').save(obs/'input.png')
            write(obs/'experiment.json',dict(id='fixture',hfov=90,image_size=[64,64]))
            class Provider:
                calls=0
                def ask(self,image,prompt,path,**kwargs):
                    self.calls+=1
                    if 'DRAFT_CONTEXT:' in prompt:
                        value=dict(objects=[],evidence_description='empty room CPU fixture',review_status='empty_scene')
                    else:value=dict(objects=[],evidence_description='',done=True)
                    write(Path(path).with_suffix('.json'),value);Path(path).write_text(json.dumps(value));return value
            p=Provider();pipeline.observe2d_review_v2(obs,lambda:p,1)
            self.assertEqual(p.calls,3)  # two bad draft replies and one independent review
            self.assertTrue((obs/'draft_failure.json').exists())
            self.assertEqual(read(obs/'qwen/observer2d/G_2d.json')['objects'],[])


if __name__=='__main__':unittest.main()
