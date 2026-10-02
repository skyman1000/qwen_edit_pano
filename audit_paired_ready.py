"""CPU-only full cache preflight; does not load a model or approve visual geometry."""
import argparse
import collections
import json
from pathlib import Path
from .common import ROOT, atomic_json, check_ids, read_jsonl, sha256, versions, require_local_output
from .paired_data import CONDITIONING, cache_key, paired_rows, local_image, target_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    require_local_output(args.output,'audit')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    train=paired_rows(args.data/'train.jsonl','train')
    val=paired_rows(args.data/'val.jsonl','val')
    heldout=read_jsonl(args.data/'heldout.jsonl')
    check_ids(heldout)
    for key in ['id','scene_id','source_view_id']:
        if {r[key] for r in train} & {r[key] for r in heldout}:
            raise ValueError('Train/heldout overlap: '+key)
    if not {r['id'] for r in val} <= {r['id'] for r in heldout}:
        raise ValueError('Validation rows absent from heldout')
    report=json.loads((args.data/'pair_report.json').read_text())
    for split,rows in [('train',train),('val',val)]:
        if report['counts'][split]!=len(rows) or report[split+'_sha256']!=sha256(args.data/(split+'.jsonl')):
            raise ValueError('Pair manifest/report mismatch')
    cfg=json.loads((args.cache/'cache_config.json').read_text())
    complete=json.loads((args.cache/'complete.json').read_text())
    if cfg['conditioning']!=CONDITIONING or cfg['versions']!=versions():
        raise ValueError('Cache conditioning/environment mismatch')
    if cfg['manifest_sha256']!=sha256(args.data/'train.jsonl') or complete['config_sha256']!=sha256(args.cache/'cache_config.json'):
        raise ValueError('Cache hashes mismatch')
    if cfg['cache_sources']!={n:sha256(Path(__file__).parent/n) for n in ('cache_paired.py','paired_data.py')}:
        raise ValueError('Cache source changed')
    if complete['count']!=2*len(train):
        raise ValueError('Wrong completion count')
    model_cfg=json.loads((Path(cfg['base_snapshot'])/'transformer/config.json').read_text())
    expected={cache_key(r,flip)+'.safetensors' for r in train for flip in (False,True)}
    actual={f.name for f in args.cache.glob('*.safetensors')}
    if expected!=actual:
        raise ValueError(f'Cache file set differs: missing={len(expected-actual)}, extra={len(actual-expected)}')
    # Validate all Arrow file stats; decode one actual pair per building below.
    pointers={r['image']['arrow_file']:r['image'] for r in train+val}
    for name,pointer in pointers.items():
        stat=Path(name).stat()
        if stat.st_size!=pointer['size_bytes'] or stat.st_mtime_ns!=pointer['mtime_ns']:
            raise ValueError('Arrow source changed: '+name)
    seen=set()
    for row in train+val:
        if row['scene_id'] not in seen:
            local_image(row)
            target_image(row)
            seen.add(row['scene_id'])
    count=0
    total_bytes=0
    lengths=collections.Counter()
    for row in train:
        for flip in (False,True):
            path=args.cache/(cache_key(row,flip)+'.safetensors')
            with safe_open(str(path),framework='pt',device='cpu') as f:
                if set(f.keys())!={'embeddings','mask','reference','ref_shape'}:
                    raise ValueError('Wrong tensor keys: '+str(path))
                e,m,r,s=[f.get_tensor(n) for n in ('embeddings','mask','reference','ref_shape')]
                if (e.ndim!=2 or e.shape[0]<1 or e.shape[1]!=model_cfg['joint_attention_dim']
                    or m.shape!=(e.shape[0],) or m.dtype!=torch.bool or not m.any()
                    or e.dtype!=torch.bfloat16 or r.dtype!=torch.bfloat16
                    or s.tolist()!=[1,64,64] or tuple(r.shape)!=(4096,model_cfg['in_channels'])):
                    raise ValueError('Invalid tensor shape/dtype: '+str(path))
                if not torch.isfinite(e).all() or not torch.isfinite(r).all():
                    raise ValueError('Nonfinite cached values: '+str(path))
                lengths[int(e.shape[0])]+=1
            count+=1
            total_bytes+=path.stat().st_size
        if count%1000==0:
            print(f'Checked {count}/{len(expected)} cache tensors',flush=True)
    result=dict(status='passed',train_rows=len(train),val_rows=len(val),
                train_houses=len({r['scene_id'] for r in train}),val_houses=len({r['scene_id'] for r in val}),
                heldout_rows=len(heldout),cache_files=count,cache_bytes=total_bytes,
                all_cache_tensors_finite=True,cache_shapes_dtypes_checked=True,
                cache_versions_and_sources_match=True,split_identity_disjoint=True,
                arrow_files_stat_checked=len(pointers),buildings_with_pair_decode_checked=len(seen),
                manifest_sha256=sha256(args.data/'train.jsonl'),cache_config_sha256=sha256(args.cache/'cache_config.json'),
                embedding_lengths=dict(lengths),rejected_rows=len(report['rejected']),
                visual_geometry_approval='not_inferred',gpu_training='not_run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
