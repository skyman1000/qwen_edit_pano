"""CPU tests for the diagnostic, not model accuracy tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from ..common import read
from .geometry import parse_json
from . import panorama_probe as probe
from .test_qwen_upstream import observed


PLAN = dict(additions=[dict(id='new_lamp', category='lighting', description='bedside lamp',
                           placement='left of the camera, outside the image',
                           unseen_reason='outside_fov', reason='a plausible lighting arrangement')])
ANSWER = dict(hidden_objects=[dict(id='new_lamp', category='lighting',
                                   center_local_m=[-3., 0., 1.], size_aabb_local_m=[.4, .8, .4])])


class ProbeTests(unittest.TestCase):
    def test_bare_arrays_missing_reason_and_core_failures(self):
        # Shape actually returned by all six real plan calls, with an omitted
        # diagnostic field as observed in three responses.
        raw = deepcopy(PLAN['additions'])
        del raw[0]['reason']
        original = deepcopy(raw)
        plan, protocol = probe.decode_plan(raw, 12)
        self.assertEqual(plan['additions'], original)
        self.assertEqual(raw, original)
        self.assertNotIn('reason', plan['additions'][0])
        self.assertFalse(protocol['strict_protocol_conformant'])
        self.assertEqual(protocol['missing_or_invalid_reason_ids'], ['new_lamp'])
        self.assertEqual(protocol['normalization'], 'wrap_array_in_additions')
        self.assertEqual(protocol['semantic_quality'], 'not_validated')
        self.assertEqual(probe.validate_plan([], 12), {'additions': []})
        for invalid in [None, 'objects', {}, [1], dict(additions={})]:
            with self.assertRaises(ValueError):
                probe.validate_plan(invalid, 12)
        missing = deepcopy(raw)
        del missing[0]['placement']
        with self.assertRaisesRegex(ValueError, 'placement'):
            probe.validate_plan(missing, 12)

    def test_replay_real_plan_answers_without_loading_model_or_changing_sources(self):
        from ..common import ROOT
        source = ROOT/'qwen_edit_pano/outputs/panorama_qwen_probe_v1/plan'
        if not (source/'summary.json').exists():
            self.skipTest('Real plan responses unavailable')
        paths = list(source.glob('*/*/plan.txt')) + list(source.glob('*/*/plan.json')) + list(source.glob('*/*/status.json'))
        snapshots = {path: path.read_bytes() for path in paths}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'review'
            with patch.object(probe, 'Qwen', side_effect=AssertionError('No model may be loaded')):
                with patch.object(sys, 'argv', ['probe', 'replay-plan', '--source', str(source), '--output', str(output)]):
                    probe.main()
            summary = read(output/'summary.json')
            self.assertEqual(summary['total_cases'], 6)
            self.assertEqual(summary['extracted_cases'], 6)
            self.assertEqual(summary['strict_protocol_cases'], 0)
            self.assertEqual(summary['extracted_objects'], 56)
            self.assertEqual(summary['missing_reason_objects'], 35)
            self.assertEqual(summary['repeated_text_excess'], 14)
            for row in summary['records']:
                raw = parse_json((source/row['condition']/row['id']/'plan.txt').read_text())
                recovered = read(output/row['condition']/row['id']/'semantic_plan.json')
                self.assertEqual(recovered['additions'], raw)
            self.assertFalse(list(output.glob('**/G_full.json')))
        for path, data in snapshots.items():
            self.assertEqual(path.read_bytes(), data)

    def test_failed_real_direction_and_semantic_only_stage(self):
        class Model:
            calls = []
            def ask(self, image, prompt, path, **kwargs):
                self.calls.append(path.stem)
                if path.stem == 'copy':
                    return json.loads(prompt.split('\n', 1)[1])
                if path.stem == 'direction':
                    # Actual user-run answer: a model axis error, not a parser error.
                    return dict(center_local_m=[-2.0, 0.0, 0.0])
                if path.stem == 'plan':
                    return deepcopy(PLAN)
                raise AssertionError('Semantic-only stage must never request geometry')
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = probe.controls(model, root)
            self.assertTrue(result['copy_exact'])
            self.assertFalse(result['behind_camera_correct'])
            self.assertEqual(result['behind_camera_answer']['center_local_m'], [-2., 0., 0.])
            image = root/'image.png'
            Image.new('RGB', (64, 64), 'gray').save(image)
            obs = observed()
            planned = probe.plan_case(model, dict(local_rgb=str(image), hfov_deg=90),
                                      obs, root/'plan_case', 12)
            self.assertEqual(planned['stage'], 'semantic_plan_only')
            self.assertEqual(model.calls, ['copy', 'direction', 'plan'])
            self.assertEqual(read(root/'plan_case/semantic_plan.json'), PLAN)
            self.assertEqual(read(root/'plan_case/G_obs.json'), obs)
            self.assertFalse((root/'plan_case/G_full.json').exists())
            self.assertFalse((root/'plan_case/G_hidden.json').exists())
            self.assertFalse(read(root/'controls.json')['behind_camera_correct'])

    def test_plan_geometry_identity_and_failures(self):
        self.assertEqual(probe.validate_plan(PLAN, 1), PLAN)
        self.assertEqual(probe.validate_geometry(ANSWER, PLAN), ANSWER)
        for key, value in [('id', 'wrong'), ('category', 'chair')]:
            changed = deepcopy(ANSWER)
            changed['hidden_objects'][0][key] = value
            with self.assertRaises(ValueError):
                probe.validate_geometry(changed, PLAN)
        with self.assertRaises(ValueError):
            probe.validate_geometry(dict(hidden_objects=[]), PLAN)
        with self.assertRaises(ValueError):
            probe.validate_plan(dict(additions=PLAN['additions']*2), 2)

    def test_controls_and_case_preserve_state_and_do_not_skip_status(self):
        class Model:
            calls = 0
            def ask(self, image, prompt, path, **kwargs):
                self.calls += 1
                if kwargs['profile'] != 'completion' or kwargs['sample']:
                    raise AssertionError('Probe must use deterministic completion profile')
                if path.stem == 'copy':
                    return json.loads(prompt.split('\n', 1)[1])
                if path.stem == 'direction':
                    return dict(center_local_m=[0, 0, -2])
                if 'SECRET_GT_ROOM' in prompt:
                    raise AssertionError('GT room leaked')
                return deepcopy(PLAN if path.stem == 'plan' else ANSWER)
        model = Model()
        obs = observed()
        obs['room'] = dict(secret='SECRET_GT_ROOM')
        original = deepcopy(obs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = probe.controls(model, root)
            self.assertTrue(controls['copy_exact'] and controls['behind_camera_correct'])
            image = root/'image.png'
            Image.new('RGB', (64, 64), 'gray').save(image)
            entry = dict(local_rgb=str(image), hfov_deg=90)
            for _ in range(2):
                result = probe.complete_case(model, entry, obs, root/'case', 12)
                self.assertTrue(result['observed_preserved'])
            self.assertEqual(model.calls, 6)
            self.assertEqual(obs, original)
            full = read(root/'case/G_full.json')
            self.assertEqual(full['objects'][:1], obs['objects'])
            self.assertEqual(full['objects'][1]['completion_plan'], PLAN['additions'][0])
            self.assertEqual(full['objects'][1]['center_local_m']['value'], [-3., 0., 1.])

    def test_real_saved_geometry_is_not_moved_by_assembly(self):
        # Replay real saved model answers rather than just synthetic parser fixtures.
        from ..common import ROOT
        from .qwen_upstream_core import assemble
        branch = ROOT/'qwen_edit_pano/outputs/qwen_world_simple_run002/completion_pred/pred'
        replayed = 0
        for source in sorted(branch.glob('*/seed0/G_hidden.json')):
            d = source.parent
            obs = read(d/'G_obs.json')
            raw = parse_json((d/'proposal.txt').read_text())
            hidden, full = assemble(obs, raw, 160, 12)
            self.assertEqual(hidden, read(source))
            self.assertEqual(full, read(d/'G_full.json'))
            replayed += 1
        if not replayed:
            self.skipTest('Historical results unavailable')
        self.assertEqual(replayed, 10)


if __name__ == '__main__':
    unittest.main()
