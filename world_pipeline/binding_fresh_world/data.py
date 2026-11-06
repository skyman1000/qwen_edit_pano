"""Fresh initialization, unchanged pack, and strict resumable experiment contracts."""
from pathlib import Path
import torch
from safetensors.torch import load_file

from ..build_polished_world import ROOT, read, atomic_json
from ..object_world.data import DEFAULT_PACK, load_pack
from ..object_world.cache import check_condition, check_posterior
from ..binding_refine_world.data import stat, optimizer_step, restore, smoke_rows
from ..binding_refine_world.supervision import POLICY, sample_binding

VERSION = 'id_A_fresh_dual_instance_binding_v1'
OUTPUT = ROOT/'qwen_edit_pano/outputs/object_world_binding_fresh_dual_v1'
INITIAL = ROOT/'qwen_edit_pano/outputs/object_world_id_ab_A_v1/initial_world.safetensors'
INITIAL_CONFIG = INITIAL.parent/'config.json'


def inputs():
    plan, records = load_pack(DEFAULT_PACK)
    old = read(INITIAL_CONFIG)
    if (old['group'] != 'A' or old['binding_weight'] != 0 or old['smoke']
            or old['category_encoding'] != 'learned_ID_42_slots'
            or old['blocks'] != [20, 40, 59] or old['width'] != 256
            or old['pano_lora_scale'] != .5 or old['pack_plan'] != plan):
        raise ValueError('Original untrained A initialization / pack contract changed')
    pano = read(Path(plan['base_checkpoint'])/'pano_config.json')
    if any(pano[k] != plan[k] for k in ('base_snapshot', 'height', 'padding_columns', 'prompt')):
        raise ValueError('Frozen Panorama and existing caches differ')
    architecture = read(Path(plan['base_snapshot'])/'transformer/config.json')
    if not architecture['zero_cond_t'] or architecture['num_layers'] <= 59:
        raise ValueError('Expected original Edit architecture')
    if (plan['train_count'], plan['val_count'], plan['Local'], plan['target']) != (
            7390, 923, 'single_original_face2', 'polished_ERP'):
        raise ValueError('Unexpected training data')
    return plan, [r for r in records if r['split'] == 'train']


def configuration(plan, smoke=False):
    return dict(version=VERSION, group='binding', smoke=smoke, source_step=0,
        pack=str(DEFAULT_PACK), pack_plan=stat(DEFAULT_PACK/'PLAN.json'),
        pack_samples=stat(DEFAULT_PACK/'samples.jsonl'), initial_world=stat(INITIAL),
        initial_config=stat(INITIAL_CONFIG),
        pack_contract={k:v for k,v in plan.items() if k != 'sources'},
        base_checkpoint=plan['base_checkpoint'], base_snapshot=plan['base_snapshot'],
        height=plan['height'], padding_columns=plan['padding_columns'], prompt=plan['prompt'],
        blocks=[20,40,59], width=256, binding_block=40, category_encoding='learned_ID_42_slots',
        pano_lora_scale=.5, region_weight=.25, coverage_weight=.01, concentration_weight=.001,
        binding_ramp_steps=200, policy=POLICY, lr=1e-4, warmup_steps=100, seed=0,
        accumulation=4, replicas=2, microbatch=1, local_accumulation=2,
        betas=[.9,.999], weight_decay=1e-5, adam_eps=1e-6, clip_norm=1., save_every=500,
        forward='original A: additive fusion, learned global, unrestricted object cross-attention',
        loss='flow+.5cube+.5yaw+.25region+.01*ramp*positive_CE+.001*ramp*reliable_other_BCE',
        sampling='EpochSampler from micro-index zero; fixed seed per global micro-index',
        initialization='exact saved UNTRAINED A tensors; fresh Adam; never A10000 / B10500 / smoke',
        activation_storage='cpu', world_precision='FP32 masters/moments; BF16 autocast, cast cache disabled',
        train_count=7390, val_count=923, caption=False, spatial_routing=False,
        hashes_computed=False, automatic_evaluation=False)


def load_initial(world):
    state = load_file(str(INITIAL), device='cpu')
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError('Nonfinite initialization')
    if any(float(v) != 0 for k,v in state.items() if k.endswith('.gate')):
        raise ValueError('Fresh gates must be exactly zero')
    world.load_state_dict(state, strict=True)


def resume_step(checkpoint, config):
    if checkpoint is None:
        return 0
    checkpoint = Path(checkpoint)
    meta = read(checkpoint/'COMPLETE.json')
    step = meta.get('step')
    if (meta.get('version') != VERSION or type(step) is not int or step < 1
            or meta.get('optimizer_step') != step
            or checkpoint.name != f'checkpoint-step{step:06d}'
            or read(checkpoint/'config.json') != config
            or any(not (checkpoint/n).is_file() for n in ('world.safetensors','optimizer.pt'))):
        raise ValueError('Resume requires a complete checkpoint from this exact fresh experiment')
    return step


def latest(folder, config):
    found = []
    for marker in Path(folder).glob('run-*/checkpoint-step*/COMPLETE.json'):
        path = marker.parent
        step = resume_step(path, config)
        found.append((step, str(path), path))
    return max(found)[2] if found else None


def check(root):
    plan, rows = inputs()
    missing = {key:0 for key in ('condition_cache','posterior_cache','region_cache','object_region_supervision')}
    for row in rows:
        for key in missing:
            path = row[key]['path'] if key == 'object_region_supervision' else row[key]
            missing[key] += not Path(path).is_file()
    if any(missing.values()):
        raise ValueError('Missing existing caches: '+str(missing))
    # Validate the actual stored initialization against original A's seed-zero
    # construction. RefinedWorld constructs/discards branches, so merely setting
    # the same seed on that class would NOT reproduce A's initial tensors.
    from ..id_binding_world.runtime import WorldBranch as OriginalWorld
    architecture = read(Path(plan['base_snapshot'])/'transformer/config.json')
    hidden = architecture['num_attention_heads']*architecture['attention_head_dim']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        expected = OriginalWorld(hidden)
    actual = load_file(str(INITIAL), device='cpu')
    if actual.keys() != expected.state_dict().keys() or not all(
            torch.equal(v, actual[k]) for k,v in expected.state_dict().items()):
        raise ValueError('Stored initial World is not original untrained A seed-zero initialization')
    representatives = []
    for row in smoke_rows(rows):
        check_condition(row['condition_cache'])
        check_posterior(row['posterior_cache'], plan['height'])
        _, meta = sample_binding(row, plan['height'], plan['padding_columns'], 0)
        if not meta['positive_pairs'] or not meta['negative_pairs']:
            raise ValueError('Smoke samples need positive and reliable-negative support')
        representatives.append(dict(id=row['id'], objects=len(row['objects']), **meta))
    result = dict(version=VERSION, train=len(rows), val=plan['val_count'], missing=missing,
        exact_untrained_A_initial_tensors=True, fresh_optimizer_step=0, replicas=2, global_batch=4,
        smoke_support=representatives, initial_world=stat(INITIAL), GPU=False, hashes_computed=False,
        note='Existing cache presence + representative headers/support; no new data/embeddings/caches')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'CPU_CHECK.json', result)
    print(result, flush=True)
    return result
