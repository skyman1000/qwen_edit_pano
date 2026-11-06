"""Checks shared by scaled-world training and inference; no GPU model loading."""
import math
from pathlib import Path
from ..common import read, rows, sha

VERSION = 'frozen_pano_scale_fourier_world_v1'


def validate_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Panorama LoRA scale must be finite and positive')
    return float(scale)


def apply_scale(transformer, scale, name):
    """Model API uses weights=; re-freeze after adapter activation."""
    scale = validate_scale(scale)
    transformer.set_adapters(name, weights=scale)
    transformer.requires_grad_(False).eval()
    return check_scale(transformer, scale, name)


def check_scale(transformer, scale, name):
    from peft.tuners.lora.layer import LoraLayer
    count = 0
    for module in transformer.modules():
        if isinstance(module, LoraLayer):
            if name not in module.scaling or list(module.active_adapters) != [name]:
                raise AssertionError('Wrong active Panorama adapter')
            if module.merged or module.disable_adapters:
                raise AssertionError('Expected enabled, unfused Panorama adapter')
            expected = scale * module.lora_alpha[name] / module.r[name]
            if not math.isclose(module.scaling[name], expected, rel_tol=1e-7, abs_tol=1e-9):
                raise AssertionError('Effective Panorama LoRA scale mismatch')
            count += 1
    if count == 0 or any(p.requires_grad for p in transformer.parameters()):
        raise AssertionError('Expected a frozen transformer with LoRA layers')
    return dict(pano_lora_scale=scale, checked_lora_layers=count, base_trainable=0)


def verify_binding(config, bundle):
    if config.get('experiment_version') != VERSION:
        raise ValueError('Use a checkpoint trained with fourier_world.train')
    from .runtime import ENCODING
    if config.get('geometry_encoding') != ENCODING:
        raise ValueError('Geometry encoding mismatch')
    validate_scale(config['pano_lora_scale'])
    if config['base_checkpoint'] != bundle['base_checkpoint']:
        raise ValueError('World checkpoint vs Panorama checkpoint mismatch')
    base = Path(bundle['base_checkpoint'])
    if config['pano_weight_sha256'] != sha(base / 'adapter_resume.safetensors'):
        raise ValueError('Panorama weights changed')


def verified_posteriors(source, bundle, train):
    """Reuse only original, hash-verified posteriors with identical VAE and targets."""
    if source is None:
        return {}
    source = Path(source).resolve()
    old = read(source / 'config.json')
    if any(old[k] != bundle[k] for k in ('base_snapshot', 'height')):
        raise ValueError('Posterior VAE snapshot or resolution changed')
    if old.get('augmentation') != 'none':
        raise ValueError('Posterior augmentation mismatch')
    old_rows = {r['id']: r for r in rows(Path(old['bundle']) / 'samples.jsonl') if r['split'] == 'train'}
    hashes = read(source / 'parameters.json')['target_posterior_sha256']
    result = {}
    for entry in train:
        if entry['id'] not in old_rows or old_rows[entry['id']]['paired'] != entry['paired']:
            raise ValueError('Posterior paired target or alignment mismatch: ' + entry['id'])
        filename = entry['id'] + '.safetensors'
        path = source / 'target_posteriors' / filename
        if filename not in hashes or sha(path) != hashes[filename]:
            raise ValueError('Posterior missing or changed: ' + str(path))
        result[entry['id']] = path
    return result
