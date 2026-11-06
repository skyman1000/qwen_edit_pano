"""Own experiment contract; reuse unchanged Local/target/region/lexical caches."""
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT, read, atomic_json
from ..object_world.data import DEFAULT_PACK, load_pack
from ..object_world.cache import check_condition, check_posterior, semantic_phrases
from ..binding_refine_world.data import stat, optimizer_step, restore, smoke_rows
from ..binding_refine_world.supervision import POLICY, sample_binding
from . import VERSION

OUTPUT = ROOT/'qwen_edit_pano/outputs/object_world_native_dual_v1'
FEATURES = DEFAULT_PACK/'class_features.safetensors'


def inputs():
    plan, records = load_pack(DEFAULT_PACK)
    pano = read(Path(plan['base_checkpoint'])/'pano_config.json')
    if any(pano[k] != plan[k] for k in ('base_snapshot', 'height', 'padding_columns', 'prompt')):
        raise ValueError('Frozen Panorama / existing caches differ')
    architecture = read(Path(plan['base_snapshot'])/'transformer/config.json')
    if (not architecture['zero_cond_t'] or architecture['num_layers'] != 60
            or architecture['joint_attention_dim'] != 3584
            or architecture['num_attention_heads']*architecture['attention_head_dim'] != 3072):
        raise ValueError('Unexpected native Edit architecture')
    if (plan['train_count'], plan['val_count'], plan['Local'], plan['target']) != (
            7390, 923, 'single_original_face2', 'polished_ERP'):
        raise ValueError('Unexpected task/data contract')
    feature_config = dict(policy=plan['semantic_policy'], names=plan['names'],
        phrases=semantic_phrases(plan['names']), base_snapshot=plan['base_snapshot'],
        pooling='mean of name tokens only; padding/prefix excluded')
    if read(FEATURES.with_suffix('.json')) != feature_config:
        raise ValueError('Existing category feature provenance differs')
    features = load_features()
    if any(features[int(k)].norm() == 0 for k in plan['names']):
        raise ValueError('Empty task category feature')
    return plan, [r for r in records if r['split'] == 'train']


def load_features():
    table = load_file(str(FEATURES), device='cpu')['features']
    if table.shape != (42, 3584) or not torch.isfinite(table).all():
        raise ValueError('Expected existing finite raw Qwen category features')
    return table


def configuration(plan, smoke=False):
    base = Path(plan['base_checkpoint'])
    return dict(version=VERSION, group='native', smoke=smoke, source_step=0,
        pack=str(DEFAULT_PACK), pack_plan=stat(DEFAULT_PACK/'PLAN.json'),
        pack_samples=stat(DEFAULT_PACK/'samples.jsonl'), semantic_features=stat(FEATURES),
        semantic_metadata=stat(FEATURES.with_suffix('.json')),
        base_files={n:stat(base/n) for n in ('pano_config.json','adapter_resume.safetensors')},
        base_checkpoint=plan['base_checkpoint'], base_snapshot=plan['base_snapshot'],
        height=plan['height'], padding_columns=plan['padding_columns'], prompt=plan['prompt'],
        blocks=[20,40,59], width=3072, rank=16, connector_rank=32, geometry_width=256,
        binding_block=40, category_encoding='frozen_Qwen_names_through_native_txt_norm_and_txt_in',
        pano_lora_scale=.5, region_weight=.25, coverage_weight=.01, concentration_weight=.001,
        binding_ramp_steps=200, policy=POLICY, lr=1e-4, warmup_steps=100, seed=0,
        accumulation=4, replicas=2, microbatch=1, local_accumulation=2,
        betas=[.9,.999], weight_decay=1e-5, adam_eps=1e-6, clip_norm=1., save_every=500,
        forward='decoupled layout pass: frozen native pre-RoPE Q, text K/V plus rank16 deltas, native output; target-only direct injection',
        object_modulation='native block txt_norm1 + current timestep txt_mod1',
        rotary_policy='layout Q and K BOTH pre-RoPE; original image/text RoPE unchanged',
        global_token='learned global initialized from projected generic class39; slot0 remains global',
        loss='flow+.5cube+.5yaw+.25region+.01*ramp*positive_CE+.001*ramp*reliable_other_BCE',
        initialization='new connector/geometry/deltas; pretrained semantic anchor and frozen native projections; fresh Adam, no old World',
        activation_storage='cpu', world_precision='FP32 masters/moments; BF16 autocast, cast cache disabled',
        train_count=7390, val_count=923, caption=False, spatial_routing=False,
        hashes_computed=False, automatic_evaluation=False)


def resume_step(checkpoint, config):
    if checkpoint is None:
        return 0
    checkpoint = Path(checkpoint)
    meta = read(checkpoint/'COMPLETE.json')
    step = meta.get('step')
    if (meta.get('version') != VERSION or type(step) is not int or step < 1
            or meta.get('optimizer_step') != step or checkpoint.name != f'checkpoint-step{step:06d}'
            or read(checkpoint/'config.json') != config
            or any(not (checkpoint/n).is_file() for n in ('world.safetensors','optimizer.pt'))):
        raise ValueError('Resume requires a complete same-version native experiment checkpoint')
    return step


def latest(folder, config):
    found = []
    for marker in Path(folder).glob('run-*/checkpoint-step*/COMPLETE.json'):
        step = resume_step(marker.parent, config)
        found.append((step, str(marker.parent), marker.parent))
    return max(found)[2] if found else None


def check(root):
    plan, rows = inputs()
    missing = {k:0 for k in ('condition_cache','posterior_cache','region_cache','object_region_supervision')}
    for row in rows:
        for key in missing:
            path = row[key]['path'] if key == 'object_region_supervision' else row[key]
            missing[key] += not Path(path).is_file()
    if any(missing.values()):
        raise ValueError('Existing caches missing: '+str(missing))
    support = []
    for row in smoke_rows(rows):
        check_condition(row['condition_cache'])
        check_posterior(row['posterior_cache'], plan['height'])
        _, meta = sample_binding(row, plan['height'], plan['padding_columns'], 0)
        if not meta['positive_pairs'] or not meta['negative_pairs']:
            raise ValueError('Smoke samples require both binding terms')
        support.append(dict(id=row['id'], objects=len(row['objects']), **meta))
    result = dict(version=VERSION, train=len(rows), val=plan['val_count'], missing=missing,
        class_features=list(load_features().shape), smoke_support=support, replicas=2, global_batch=4,
        old_World_loaded=False, fresh_Adam=True, GPU=False, hashes_computed=False,
        note='Existing cache paths/representative headers checked; not all polished object pixels visually certified')
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'CPU_CHECK.json', result)
    print(result, flush=True)
    return result
