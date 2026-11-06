"""CPU input/provenance and three-condition wiring; no pretrained model/GPU."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from ..common import read, write, sha
from ..oracle_model import FEATURE_VERSION
from ..completion.semantic_world import make_graph
from ..completion.test_semantic_world import fixture
from .binding import VERSION
from .runtime import ENCODING
from .predicted_inference import scene_inputs, checkpoint_inputs, generate


def scene(root):
    directory = root/'scene'
    directory.mkdir()
    image = root/'source.png'
    Image.new('RGB', (32, 32), 'green').save(image)
    Image.open(image).save(directory/'local.png')
    write(directory/'config.json', dict(image=str(image), image_sha256=sha(image)))
    observed, plan, layout = fixture()
    obs = make_graph(observed, layout, 'test_vocab', 'cpu_fixture', 'observed')
    full = make_graph(plan['objects'], layout, 'test_vocab', 'cpu_fixture')
    full['objects'][:len(obs['objects'])] = deepcopy(obs['objects'])
    write(directory/'G_obs.json', obs)
    write(directory/'G_full.json', full)
    return directory


def checkpoint(root):
    base = root/'panorama'; base.mkdir()
    ckpt = root/'checkpoint'; ckpt.mkdir()
    bundle_dir = root/'bundle'; bundle_dir.mkdir()
    for path in (base/'adapter_resume.safetensors', base/'pytorch_lora_weights.safetensors', ckpt/'world.safetensors'):
        path.write_bytes(b'CPU hash fixture; never loaded as weights')
    pano = dict(base_snapshot='local_snapshot', height=32, inference_padding_columns=1,
                conditioning='edit2511_native_local_rgb_face2_v1', prompt='Expand the photo.')
    write(base/'pano_config.json', pano)
    vocab = root/'vocabulary.json'
    write(vocab, dict(category_mapping_sha256='test_vocab'))
    protected = {str(p): sha(p) for p in (vocab, base/'pano_config.json', base/'pytorch_lora_weights.safetensors')}
    # A nonexistent scene-GT path makes unintended blanket GT loading fail.
    protected[str(root/'NEVER_READ_GT.json')] = 'not_needed_for_predicted_inference'
    bundle = dict(feature_version=FEATURE_VERSION, base_checkpoint=str(base),
                  base_snapshot='local_snapshot', height=32, padding_columns=1,
                  max_objects=8, protected_files=protected)
    write(bundle_dir/'BUNDLE.json', bundle)
    config = dict(feature_version=FEATURE_VERSION, condition='full', bundle=str(bundle_dir),
        bundle_sha256=sha(bundle_dir/'BUNDLE.json'), base_checkpoint=str(base),
        experiment_version=VERSION, geometry_encoding=ENCODING, pano_lora_scale=.5,
        pano_weight_sha256=sha(base/'adapter_resume.safetensors'))
    write(ckpt/'config.json', config)
    write(ckpt/'COMPLETE.json', dict(feature_version=FEATURE_VERSION, world_sha256=sha(ckpt/'world.safetensors')))
    return ckpt


class PredictedInferenceTests(unittest.TestCase):
    def native_scene(self, root):
        directory=scene(root)
        config=read(directory/'config.json')
        observed_path=root/'fixed_observed.json'
        write(observed_path,read(directory/'G_obs.json'))
        image_path=Path(config['image'])
        write(directory/'input.json',dict(image=str(image_path),state=str(observed_path),hfov=90,
            source_sha256={str(p.resolve()):sha(p) for p in (image_path,observed_path)}))
        (directory/'config.json').unlink()
        return directory,observed_path

    def test_native_completion_input_json_connects_observed_and_completed_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,observed_path=self.native_scene(Path(tmp))
            image,graphs,tensors,hashes=scene_inputs(directory,'test_vocab',8)
            self.assertEqual(image.size,(32,32))
            self.assertEqual(int(tensors['observed'][2].sum()),3)
            self.assertEqual(int(tensors['completed'][2].sum()),4)
            self.assertIn(str(directory/'input.json'),hashes)
            self.assertIn(str(observed_path),hashes)
            self.assertEqual(graphs['completed']['objects'][:2],graphs['observed']['objects'])

    def test_native_completion_rejects_changed_or_different_observed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,observed_path=self.native_scene(Path(tmp))
            original=read(directory/'G_obs.json')
            changed=deepcopy(original)
            changed['objects'][0]['center_local_m']['value'][0]+=.2
            write(directory/'G_obs.json',changed)
            with self.assertRaisesRegex(ValueError,'recorded predicted source'):
                scene_inputs(directory,'test_vocab',8)
            write(directory/'G_obs.json',original)
            write(observed_path,changed)
            with self.assertRaisesRegex(ValueError,'source image/observation metadata changed'):
                scene_inputs(directory,'test_vocab',8)

    def test_real_tensorizer_preserves_observation_and_includes_additions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = scene(Path(tmp))
            image, graphs, tensors, hashes = scene_inputs(directory, 'test_vocab', 8)
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(int(tensors['observed'][2].sum()), 3)
            self.assertEqual(int(tensors['completed'][2].sum()), 4)
            self.assertEqual(len(hashes), 5)
            self.assertEqual(graphs['completed']['objects'][:2], graphs['observed']['objects'])

    def test_rejects_gt_changed_observation_wrong_vocab_and_missing_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = scene(Path(tmp)); original = read(directory/'G_full.json')
            for kind in ('gt', 'scope', 'observed', 'vocabulary', 'geometry'):
                with self.subTest(kind=kind):
                    graph = deepcopy(original)
                    if kind == 'gt': graph['stage'] = 'gt'
                    elif kind == 'scope': graph['scope'] = 'observed'
                    elif kind == 'observed': graph['objects'][0]['center_local_m']['value'][0] += .2
                    elif kind == 'vocabulary': graph['vocabulary_sha256'] = 'different'
                    else:
                        graph['objects'][-1]['center_local_m'].update(valid=False, value=None)
                    write(directory/'G_full.json', graph)
                    with self.assertRaises(ValueError): scene_inputs(directory, 'test_vocab', 8)
            write(directory/'G_full.json', original)
            with self.assertRaises(ValueError): scene_inputs(directory, 'test_vocab', 2)

    def test_rejects_mismatched_saved_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = scene(Path(tmp))
            Image.new('RGB', (32, 32), 'red').save(directory/'local.png')
            with self.assertRaises(ValueError): scene_inputs(directory, 'test_vocab', 8)

    def test_checkpoint_metadata_binding_without_scene_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = checkpoint(Path(tmp))
            config, _, bundle, _, vocabulary = checkpoint_inputs(ckpt)
            self.assertEqual(vocabulary, 'test_vocab')
            self.assertEqual(bundle['max_objects'], 8)
            config['condition'] = 'observed'
            write(ckpt/'config.json', config)
            with self.assertRaisesRegex(ValueError, 'condition=full'): checkpoint_inputs(ckpt)

    def test_checkpoint_rejects_changed_exported_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = checkpoint(Path(tmp))
            (Path(read(ckpt/'config.json')['base_checkpoint'])/'pytorch_lora_weights.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Panorama file changed'): checkpoint_inputs(ckpt)

    def test_three_modes_use_same_rgb_prompt_noise_and_only_change_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = scene(root)
            image, _, tensors, _ = scene_inputs(directory, 'test_vocab', 8)
            output = root/'result'; output.mkdir()
            class World:
                branches = {}
                current = None
                def set_world(self, state): self.current = state
                def diagnostics(self): return dict(active=self.current is not None)
            world = World(); calls = []
            def pipe(**kwargs):
                calls.append(dict(rgb=kwargs['image'].tobytes(), prompt=kwargs['prompt'],
                    noise=torch.randn(4, generator=kwargs['generator']),
                    count=None if world.current is None else int(world.current[2].sum()),
                    cfg=kwargs['true_cfg_scale'], steps=kwargs['num_inference_steps']))
                return SimpleNamespace(images=[Image.new('RGB', (64, 32))])
            # CPU-only transport substitute; no CUDA allocation or model calls.
            with patch.object(torch.Tensor, 'cuda', lambda t: t):
                generate(pipe, world, tensors, image, dict(prompt='same prompt'), dict(height=32),
                         output, ['baseline', 'observed', 'completed'], 28, 4., 123)
            self.assertEqual([c['count'] for c in calls], [None, 3, 4])
            for call in calls[1:]:
                self.assertTrue(torch.equal(call['noise'], calls[0]['noise']))
                self.assertEqual(call['rgb'], calls[0]['rgb'])
                self.assertEqual(call['prompt'], calls[0]['prompt'])
            self.assertIsNone(world.current)
            self.assertFalse(read(output/'completed.json')['oracle'])
            self.assertTrue((output/'observed_input.safetensors').is_file())
            self.assertTrue((output/'comparison.png').is_file())


if __name__ == '__main__':
    unittest.main()
