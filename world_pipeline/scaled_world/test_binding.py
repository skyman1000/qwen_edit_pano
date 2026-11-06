"""Tiny CPU tests: scale actually changes frozen LoRA; new binding rejects mismatches."""
import json
import tempfile
from pathlib import Path
import unittest
import torch
from ..oracle_test import model, inputs, graph
from ..oracle_model import WorldBranch, tensorize
from .binding import apply_scale, check_scale, validate_scale, verified_posteriors, verify_binding, VERSION
from ..common import sha


class ScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_scale_and_reload(self):
        m = model(checkpointed=False)
        report = apply_scale(m, .5, 'default')
        self.assertEqual(report['checked_lora_layers'], 12)
        with torch.no_grad():
            half = m(**inputs())[0]
            apply_scale(m, 1., 'default')
            full = m(**inputs())[0]
        self.assertGreater(float((full-half).abs().max()), 0)
        # Scaling is runtime metadata, so restoring weights alone is insufficient.
        restored = model(checkpointed=False)
        restored.load_state_dict(m.state_dict())
        apply_scale(restored, .5, 'default')
        torch.testing.assert_close(restored(**inputs())[0], half, rtol=0, atol=0)
        with self.assertRaises(AssertionError):
            check_scale(restored, 1., 'default')

    def test_world_zero_and_backward_at_half_scale(self):
        m = model(checkpointed=True)
        apply_scale(m, .5, 'default')
        with torch.no_grad():
            base = m(**inputs())[0]
        world = WorldBranch(32, (0,2), 16)
        world.attach(m)
        world.set_world(tensorize(graph(), 3))
        zero = m(**inputs())[0]
        torch.testing.assert_close(zero, base, rtol=0, atol=0)
        for branch in world.branches.values():
            branch.gate.data.fill_(.1)
        m(**inputs())[0].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in world.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_invalid_scale(self):
        for s in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_scale(s)

    def test_posterior_provenance_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'target_posteriors').mkdir()
            post = root/'target_posteriors'/'a.safetensors'
            post.write_bytes(b'fixture-only')
            entry = dict(id='a', split='train', paired=dict(image='image-a',target_alignment='aligned'))
            (root/'samples.jsonl').write_text(json.dumps(entry)+'\n')
            old = dict(base_snapshot='model', height=1024, augmentation='none',bundle=str(root))
            (root/'config.json').write_text(json.dumps(old))
            (root/'parameters.json').write_text(json.dumps(dict(target_posterior_sha256={post.name:sha(post)})))
            self.assertEqual(verified_posteriors(root,old,[entry]),{'a':post})
            bad = dict(entry,paired=dict(image='different',target_alignment='aligned'))
            with self.assertRaises(ValueError):verified_posteriors(root,old,[bad])
            post.write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_posteriors(root,old,[entry])

    def test_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'adapter_resume.safetensors').write_bytes(b'fixture')
            bundle=dict(base_checkpoint=str(base))
            config=dict(experiment_version=VERSION,pano_lora_scale=.5,
                        base_checkpoint=str(base),pano_weight_sha256=sha(base/'adapter_resume.safetensors'))
            verify_binding(config,bundle)
            with self.assertRaises(ValueError):verify_binding(dict(config,base_checkpoint='wrong'),bundle)
            with self.assertRaises(ValueError):verify_binding(dict(config,experiment_version='old'),bundle)


if __name__ == '__main__':
    unittest.main()
