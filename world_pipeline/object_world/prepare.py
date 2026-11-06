"""CPU plan: reuse native conditions/posteriors and pool instance loss support."""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path

from ..build_polished_world import atomic_json, read, read_rows, ROOT
from .data import DEFAULT_DATA, DEFAULT_PACK, DEFAULT_BASE, dataset, identity, write_rows, atomic_tensors


def legacy_conditions(records, base, cache_root, paired_manifest):
    # Compatibility with existing filenames ONLY. cache_key hashes a short key
    # string containing the already-stored digest, never reads/hashes any file.
    # No hash verification or new source/content digest is performed here.
    from qwen_edit_pano.paired_data import cache_key, CONDITIONING
    config=read(cache_root/'cache_config.json')
    if any(config[k]!=base[k] for k in ('base_snapshot','prompt')) or config['conditioning']!=CONDITIONING:
        raise ValueError('Existing native condition cache uses another model/prompt')
    old={row['id']:row for row in read_rows(paired_manifest)}
    present={p.name for p in cache_root.iterdir() if p.suffix=='.safetensors'}
    found={}
    for row in records:
        if row['split']!='train' or row['id'] not in old:
            continue
        previous=old[row['id']]
        if identity(row)!=identity(previous):
            raise ValueError('Original Local/target/alignment differs: '+row['id'])
        name=cache_key(previous,False)+'.safetensors'
        if name in present:
            found[row['id']]=str((cache_root/name).resolve())
    return found


def legacy_posteriors(records, base, source):
    if source is None or not (source/'config.json').is_file():
        return {}
    config=read(source/'config.json')
    if config['base_snapshot']!=base['base_snapshot'] or config['height']!=base['height'] or config['augmentation']!='none':
        raise ValueError('Existing posteriors use another VAE/resolution/augmentation')
    old={r['id']:r['paired'] for r in read_rows(Path(config['bundle'])/'samples.jsonl') if r['split']=='train'}
    found={}
    for row in records:
        if row['split']=='train' and row['id'] in old:
            if identity(row)!=identity(old[row['id']]):
                raise ValueError('Existing posterior target identity/alignment differs')
            path=source/'target_posteriors'/(row['id']+'.safetensors')
            if path.is_file():
                found[row['id']]=str(path.resolve())
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=DEFAULT_PACK)
    p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    p.add_argument('--condition-cache',type=Path,default=ROOT/'qwen_edit_pano/cache/paired_full_v1')
    p.add_argument('--paired-train',type=Path,default=ROOT/'qwen_edit_pano/data/paired_full_v1/train.jsonl')
    p.add_argument('--posterior-source',type=Path,default=ROOT/'qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1')
    p.add_argument('--max-objects',type=int,default=256)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    for key in ('data','output','base','condition_cache','paired_train','posterior_source'):
        setattr(args,key,getattr(args,key).resolve())
    if not args.output.is_relative_to(ROOT/'qwen_edit_pano/data') or args.output in (args.data,ROOT/'qwen_edit_pano/data'):
        p.error('Use a separate training pack below qwen_edit_pano/data')
    with ExitStack() as stack:
        source_lock=stack.enter_context((args.data/'BUILD.lock').open('r'))
        fcntl.flock(source_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        records,names=dataset(args.data)
        maximum=max(len(row['objects']) for row in records)
        if maximum>args.max_objects:
            raise ValueError('Capacity smaller than dataset maximum')
        base=read(args.base/'pano_config.json')
        if read(args.base/'COMPLETE.json').get('completed_epochs',0)<1:
            raise ValueError('Use a completed paired Panorama LoRA epoch')
        conditions=legacy_conditions(records,base,args.condition_cache,args.paired_train)
        posteriors=legacy_posteriors(records,base,args.posterior_source)
        train=[row for row in records if row['split']=='train']
        smoke_ids=list(dict.fromkeys([train[0]['id'],max(train,key=lambda row:len(row['objects']))['id']]))
        plan=dict(version='object_world_pack_v1',data=str(args.data),base_checkpoint=str(args.base),
            base_snapshot=base['base_snapshot'],height=base['height'],padding_columns=base['padding_columns'],
            prompt=base['prompt'],names=names,max_objects=args.max_objects,maximum_objects=maximum,
            train_count=len(train),val_count=len(records)-len(train),smoke_ids=smoke_ids,
            augmentation='none',Local='single_original_face2',target='polished_ERP',spatial_bias=False,
            region_policy='equal_instance_pixel_mass_area_pool_mean1_v1',
            semantic_policy='frozen_Qwen2_5_VL_last_hidden_category_tokens_mean_v1',
            file_content_hashes_computed=False,sources=[identity(row) for row in records])
        for row in records:
            sid=row['id']
            row['condition_cache']=conditions.get(sid,str(args.output/'conditions'/(sid+'.safetensors')))
            row['posterior_cache']=posteriors.get(sid,str(args.output/'posteriors'/(sid+'.safetensors')))
            row['region_cache']=str(args.output/'regions'/(sid+'.safetensors'))
        report=dict(train=len(train),val=len(records)-len(train),maximum_objects=maximum,
            max_objects=args.max_objects,reused_native_conditions=len(conditions),reused_posteriors=len(posteriors),
            new_native_conditions=len(train)-len(conditions),new_posteriors=len(train)-len(posteriors),
            file_content_hashes_computed=False,GPU=False)
        print(report,flush=True)
        if args.preflight:
            print('PREFLIGHT PASSED; no training pack or model loaded.',flush=True);return
        args.output.mkdir(parents=True,exist_ok=True)
        lock=stack.enter_context((args.output/'PACK.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=args.output/'PLAN.json'
        if config.exists() and read(config)!=plan:
            raise ValueError('Existing pack plan differs; use a new output')
        if not config.exists():
            atomic_json(config,plan)
        manifest=args.output/'samples.jsonl'
        if manifest.exists() and read_rows(manifest)!=records:
            raise ValueError('Existing pack samples differ; use a new output')
        write_rows(manifest,records)
        import numpy as np
        import torch
        if args.threads<1:raise ValueError('CPU threads must be positive')
        torch.set_num_threads(args.threads)
        from safetensors.torch import load_file
        from .runtime import region_weights
        for index,row in enumerate(train):
            path=Path(row['region_cache'])
            if path.is_file():
                weights=load_file(str(path))['weights']
                if weights.shape!=(base['height']//8,base['height']//4) or not torch.isfinite(weights).all() or (weights<0).any():
                    raise ValueError('Invalid existing region cache: '+row['id'])
                continue
            with np.load(row['object_region_supervision']['path'],allow_pickle=False) as arrays:
                instance=arrays['instance'];valid=arrays['object_loss_valid']
                if instance.shape!=(512,1024) or valid.shape!=instance.shape or valid.dtype!=bool:
                    raise ValueError('Unexpected source instance mask')
                weights=region_weights(instance,valid,[int(o['instance_id']) for o in row['objects']],
                                       (base['height']//8,base['height']//4))
            atomic_tensors(path,dict(weights=weights))
            if (index+1)%100==0 or index+1==len(train):
                print(dict(stage='CPU_region_weights',completed=index+1,total=len(train)),flush=True)
        atomic_json(args.output/'PREPARED.json',report)
        print('PREPARED. Next: GPU class semantics and missing target posterior caches.',flush=True)


if __name__=='__main__':
    main()
