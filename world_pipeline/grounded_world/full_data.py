"""Full training distribution, unchanged C architecture/loss; CPU preparation."""
import argparse
from pathlib import Path

from ..build_polished_world import ROOT, read, atomic_json
from ..object_world.data import DEFAULT_PACK
from .data import inputs, contract, ensure_support, check_rows, source_meta
from .runtime import VERSION as ARCHITECTURE
from .supervision import POLICY
from .category_contrast import EXPERIMENT

VERSION = 'grounded_full_distribution_C_v1'
SOURCE = ROOT/'qwen_edit_pano/outputs/object_world_category_contrast_v1/train/C/run-1793084255954055482/checkpoint-step000600'
OUTPUT = ROOT/'qwen_edit_pano/outputs/object_world_full_C_v1'
SUPPORT = ROOT/'qwen_edit_pano/data/grounded_world_full_C_v1_support'


def full_inputs():
    plan, allrows, _, features = inputs()
    train = [r for r in allrows if r['split'] == 'train']
    if len(train) != plan['train_count'] or len(train) != 7390:
        raise ValueError('This experiment is bound to all 7390 current training images')
    return plan, allrows, train, features


def file_stat(path):
    path = Path(path); s = path.stat()
    return dict(path=str(path.resolve()), size=s.st_size, mtime_ns=s.st_mtime_ns)


def data_contract(plan, rows):
    # Source identities are already checked by load_pack. Sizes/times detect
    # ordinary edits without a content hash or repeated whole-file CRC scan.
    return dict(pack_contract=contract(plan), manifest=file_stat(DEFAULT_PACK/'samples.jsonl'),
                train_ids=[r['id'] for r in rows], supervision_policy=POLICY,
                region_sources=[file_stat(r['object_region_supervision']['path']) for r in rows])


def configuration(plan, rows, smoke=False):
    c = read(SOURCE/'config.json'); marker = read(SOURCE/'COMPLETE.json')
    if (marker != dict(version=EXPERIMENT, step=600) or c['group'] != 'C' or c['smoke']
            or c['architecture_version'] != ARCHITECTURE or c['category_contrast_weight'] != 0
            or c['pack_contract'] != contract(plan) or c['region_weight'] != .25
            or c['binding_weight'] != .01 or c['pano_lora_scale'] != .5
            or c['blocks'] != [20,40,59] or c['width'] != 256 or c['binding_block'] != 40):
        raise ValueError('Use the reviewed real C600 source, without changing its base/architecture/loss')
    names = ('seed','blocks','width','binding_block','region_weight','binding_weight','pano_lora_scale',
             'lr','adam_eps','betas','weight_decay','clip_norm','activation_storage','frame','category_encoding')
    return dict(version=VERSION, architecture_version=ARCHITECTURE, group='Full-C', smoke=bool(smoke),
        pack=str(DEFAULT_PACK), support=str(SUPPORT), initial_checkpoint=str(SOURCE), initial_optimizer_step=600,
        **{k:c[k] for k in names}, **data_contract(plan, rows),
        accumulation=4, maximum_new_steps=10000, save_every=100,
        category_contrast_weight=0., initialization='C600 weights AND FP32 Adam moments; no World reinitialization',
        sampling='full dataset index starts at zero; noise/binding RNG indices continue after C600',
        loss='flow + 0.5 cube + 0.5 yaw + 0.25 object_region + 0.01 instance_binding',
        schedule='constant inherited lr=1e-4; requested stop step is not part of the optimizer schedule',
        step_meaning='new optimizer steps on all 7390 images; optimizer state begins at inherited step600',
        automatic_evaluation=False, automatic_extension=False, hashes_computed=False)


def require_prepared(plan, rows):
    expected=data_contract(plan, rows)
    if not (SUPPORT/'PREPARED_FULL.json').is_file():
        raise ValueError('Run the CPU prepare stage first: bash qwen_edit_pano/world_pipeline/run_object_world_full.sh prepare')
    marker=read(SUPPORT/'PREPARED_FULL.json')
    if marker['version'] != VERSION or marker['data'] != expected:
        raise ValueError('Full-data preparation is missing/stale; run the CPU prepare stage')
    # Validate per-sample metadata rather than accepting a completion count.
    for row in rows:
        meta=read(SUPPORT/(row['id']+'.json'))
        if meta['source'] != source_meta(row, plan) or not (SUPPORT/(row['id']+'.safetensors')).is_file():
            raise ValueError('Missing/stale full-data binding support: '+row['id'])
    return marker


def prepare():
    plan,_,rows,_=full_inputs();config=configuration(plan,rows)
    check_rows(rows,plan);SUPPORT.mkdir(parents=True,exist_ok=True)
    counts=[]
    for i,row in enumerate(rows,1):
        summary=ensure_support(row,plan,SUPPORT);counts.append(summary['eligible_instances'])
        if i % 100 == 0 or i == len(rows):
            print(dict(stage='CPU_full_binding_labels',completed=i,total=len(rows)),flush=True)
    result=dict(version=VERSION,data=data_contract(plan,rows),train=len(rows),val=plan['val_count'],
        eligible_binding_instances=sum(counts),zero_auxiliary_label_images=sum(n==0 for n in counts),
        zero_auxiliary_label_ids=[r['id'] for r,n in zip(rows,counts) if n==0],
        maximum_objects=max(len(r['objects']) for r in rows),all_caches_present=True,
        initial_checkpoint=str(SOURCE),hashes_computed=False,cpu_only=True)
    atomic_json(SUPPORT/'PREPARED_FULL.json',result);require_prepared(plan,rows)
    print({k:v for k,v in result.items() if k != 'data'},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true')
    p.add_argument('--threads',type=int,choices=(1,2,4,8,16),default=4);a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    if a.check_only:
        plan,_,rows,_=full_inputs();configuration(plan,rows);require_prepared(plan,rows);check_rows(rows,plan)
        print(dict(version=VERSION,train=len(rows),prepared=True,hashes_computed=False),flush=True)
    else:prepare()


if __name__=='__main__':main()
