"""Containment proofs and frozen-input replay; no model weights or GPU."""
import tempfile
from pathlib import Path
import unittest

from PIL import Image

from ..common import write
from .audit_semantic_layout import project_aabb
from .compare_semantic_layout import frozen_cases, checkpoint_metadata


class LayoutAuditTests(unittest.TestCase):
    def test_inside_partial_and_behind_are_distinguished(self):
        inside = project_aabb([0,0,3], [1,1,1], 90, (768,768))
        self.assertTrue(inside['entire_aabb_inside_input_frustum'])
        self.assertEqual(len(inside['projected_bbox_xyxy_norm']), 4)
        crossing = project_aabb([3,0,3], [2,1,2], 90, (768,768))
        self.assertFalse(crossing['entire_aabb_inside_input_frustum'])
        # Crossing is not thereby proven outside or occluded.
        self.assertGreater(crossing['projected_bbox_xyxy_norm'][2], 1)
        behind = project_aabb([0,0,-3], [1,1,1], 90, (768,768))
        self.assertFalse(behind['entire_aabb_inside_input_frustum'])
        self.assertIsNone(behind['projected_bbox_xyxy_norm'])

    def test_vertical_fov_depends_on_aspect_ratio_and_image_up_projects_up(self):
        square = project_aabb([0,1,3], [.2,.2,.2], 90, (768,768))
        short = project_aabb([0,1,3], [.2,.2,.2], 90, (768,192))
        self.assertTrue(square['entire_aabb_inside_input_frustum'])
        self.assertFalse(short['entire_aabb_inside_input_frustum'])
        self.assertLess(square['projected_bbox_xyxy_norm'][3], .5)

    def test_replay_uses_original_request_and_detects_changed_settings_or_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'sample'; folder.mkdir()
            case = dict(id='sample', prompt='frozen prompt', observed=dict(objects=[]),
                        plan=dict(additions=[]), source_sha256={}, image=str(root/'original.png'))
            Image.new('RGB',(16,12),'blue').save(root/'original.png')
            write(root/'run_config.json', dict(cases=[case], seed=3407, max_tokens=4096))
            write(folder/'G_obs.json', case['observed']); write(folder/'semantic_plan.json', case['plan'])
            Image.new('RGB',(16,12),'blue').save(folder/'local.png')
            request = dict(prompt=case['prompt'], seed=3407, sample=True, max_tokens=4096,
                profile='official', response_format='json', presence_penalty=1.5,
                generation=dict(do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.),
                guard_repetition=dict(repeated_complete_objects=3, check_every=32))
            write(folder/'layout.request.json', request)
            _, cases = frozen_cases(root)
            self.assertEqual(cases[0]['prompt'], case['prompt'])
            self.assertEqual(cases[0]['image'], str((folder/'local.png').resolve()))
            Image.new('RGB',(16,12),'red').save(folder/'local.png')
            # A loaded comparison binds its image bytes; it cannot reuse stale results.
            from ..common import sha
            self.assertNotEqual(sha(folder/'local.png'), cases[0]['source_sha256'][str((folder/'local.png').resolve())])
            with self.assertRaisesRegex(ValueError, 'resized RGB'): frozen_cases(root)
            Image.new('RGB',(16,12),'blue').save(folder/'local.png')
            request['generation']['temperature'] = .1
            write(folder/'layout.request.json', request)
            with self.assertRaisesRegex(ValueError, 'sampling'): frozen_cases(root)
            write(folder/'layout.request.json', dict(request, generation=dict(do_sample=True, temperature=.7, top_p=.8, top_k=20, repetition_penalty=1.)))
            write(folder/'G_obs.json', dict(objects=['changed']))
            with self.assertRaisesRegex(ValueError, 'observed state'): frozen_cases(root)

    def test_missing_checkpoint_shards_fail_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'config.json', dict(model_type='qwen3_vl'))
            write(root/'model.safetensors.index.json', dict(weight_map={'p':'missing.safetensors'}))
            with self.assertRaisesRegex(ValueError, 'shards missing'): checkpoint_metadata(root)


if __name__ == '__main__': unittest.main()
