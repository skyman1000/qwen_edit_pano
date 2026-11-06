"""Reuse the previous LPIPS/SSIMc/DISTS protocol for the no-World benchmark."""
import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the audited fixed-camera projection and image loader; no generation or
# GT-world preparation code is invoked by these helpers.
from research.world_evaluation.local_reference_metrics import project, rgb

FACES = [('front',0,0),('right',90,0),('back',180,0),('left',-90,0),('up',0,90),('down',0,-90)]


def atomic_json(path,content):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(content,indent=2)+'\n')
    temporary.replace(path)


def read_paired_cache(cache, ids, arm):
    records={}
    if not cache.exists():return records
    raw=cache.read_bytes();offset=0
    lines=raw.splitlines(keepends=True)
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if index!=len(lines)-1:raise ValueError('Corrupt interior paired-metric record')
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            cache.with_name(cache.name+'.interrupted-'+stamp).write_bytes(raw)
            with cache.open('r+b') as stream:stream.truncate(offset)
            print('Recovered interrupted final paired-metric record.',flush=True)
            break
        if row['id'] not in ids or row['arm']!=arm or row['region'] not in ('local_diagnostic','cubemap'):
            raise ValueError('Paired cache sample/arm/region changed')
        values=[row]
        if row['region']=='cubemap':
            if set(row['faces'])!={name for name,_,_ in FACES}:raise ValueError('Incomplete saved cubemap faces')
            values+=list(row['faces'].values())
        if not all(math.isfinite(value[k]) for value in values for k in ('lpips','ssim','dists')):
            raise ValueError('Nonfinite paired cache score')
        records[(row['id'],row['region'])]=row
        offset+=len(line)
    if cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
        with cache.open('ab') as stream:stream.write(b'\n');stream.flush();os.fsync(stream.fileno())
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--arm',choices=('native','finetuned','gt_state','mvd','gimbal360','panodiffusion','panodiffusion_fullerp'),required=True)
    p.add_argument('--generated-size',type=int,nargs=2,default=[2048,1024],metavar=('WIDTH','HEIGHT'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    if args.generated_size[0] != 2*args.generated_size[1] or args.generated_size[1] <= 0:
        p.error('generated-size must describe a positive 2:1 ERP')
    root=args.benchmark.resolve()
    plan=json.loads((root/'plan.json').read_text())
    complete=json.loads((root/'GENERATION_COMPLETE.json').read_text())
    sources=complete['sources'][args.arm]
    if set(sources)!=set(plan['ids']):
        raise ValueError('Completed sample IDs differ from plan')
    out=root/'paired_metrics_512_cube'/args.arm
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(size=512,metrics=['lpips(alex,v0.1)','ssimc(RGB,no-downsample)','dists'],
                  input='RGB float32 [0,1]; FP32; no autocast',
                  projection='audited previous py360convert e2p; FOV90; bilinear; fixed faces; no alignment search',
                  aggregation='six faces arithmetic mean, plus per-face and five nonfront mean; not hidden-only',
                  local='fixed front versus actual Local RGB resized PIL bicubic; diagnostic only',
                  no_world_state=args.arm!='gt_state',generated_size=args.generated_size,
                  reference_size=[2048,1024])
    if args.arm=='panodiffusion_fullerp':
        protocol.update(full_gt_erp_encoded=True,local_only_conditioning=False,
                        conditioning='Full GT ERP encoded before latent single90 mask; official demo-style protocol')
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        saved=json.loads(protocol_path.read_text())
        saved.setdefault('generated_size',[2048,1024]);saved.setdefault('reference_size',[2048,1024])
        if saved!=protocol:raise ValueError('Paired metric protocol changed; use a new output')
    else:atomic_json(protocol_path,protocol)
    if args.preflight:
        for sid in plan['ids']:
            for key in ('image','gt','local'):
                if not Path(sources[sid][key]).is_file():raise FileNotFoundError(sources[sid][key])
        print(dict(ready=True,arm=args.arm,samples=len(plan['ids']),protocol=protocol));return
    inventory=[]
    for sid in plan['ids']:
        for key in ('image','gt','local'):
            path=Path(sources[sid][key]);stat=path.stat()
            inventory.append(dict(id=sid,kind=key,path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    frozen=out/'input_inventory.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=inventory:raise ValueError('Paired metric input files changed')
    else:atomic_json(frozen,inventory)
    cache=out/'per_sample.jsonl'
    records=read_paired_cache(cache,set(plan['ids']),args.arm)
    metric_names=('lpips','ssim','dists')
    pending=any((sid,region) not in records for sid in plan['ids'] for region in ('local_diagnostic','cubemap'))
    if pending:
        import torch
        import pyiqa
        torch.set_num_threads(4)
        metrics={k:pyiqa.create_metric(name,device=args.device).eval()
                 for k,name in [('lpips','lpips'),('ssim','ssimc'),('dists','dists')]}
    def score(a,b):
        def tensor(x):
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            values={k:float(m(tensor(a),tensor(b)).item()) for k,m in metrics.items()}
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite reference metric')
        return values
    with cache.open('a') as f:
        for i,sid in enumerate(plan['ids'],1):
            source=sources[sid]
            if all((sid,region) in records for region in ('local_diagnostic','cubemap')):continue
            image,gt=rgb(source['image']),rgb(source['gt'])
            if image.size!=tuple(args.generated_size) or gt.size!=(2048,1024):raise ValueError('Unexpected ERP size')
            for region in ('local_diagnostic','cubemap'):
                if (sid,region) in records:continue
                if region=='local_diagnostic':
                    local=np.asarray(rgb(source['local']).resize((512,512),Image.Resampling.BICUBIC),dtype=np.float32)/255
                    values=score(project(image,512),local);detail={}
                else:
                    detail={name:score(project(image,512,yaw,pitch),project(gt,512,yaw,pitch))
                            for name,yaw,pitch in FACES}
                    values={k:float(np.mean([v[k] for v in detail.values()])) for k in metric_names}
                row=dict(id=sid,arm=args.arm,region=region,**values,faces=detail)
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());records[(sid,region)]=row
            print(f'{args.arm} paired metrics {i}/{len(plan["ids"])}',flush=True)
    selected=[records[(sid,region)] for sid in plan['ids'] for region in ('local_diagnostic','cubemap')]
    means={}
    for region,key in [('local_diagnostic','local_diagnostic'),('cube_mean','cubemap')]:
        rows=[r for r in selected if r['region']==key]
        means[region]={k:float(np.mean([r[k] for r in rows])) for k in metric_names}
    cube=[r for r in selected if r['region']=='cubemap']
    for name,_,_ in FACES:
        means[name]={k:float(np.mean([r['faces'][name][k] for r in cube])) for k in metric_names}
    means['nonfront_mean']={k:float(np.mean([r['faces'][name][k] for r in cube for name,_,_ in FACES if name!='front'])) for k in metric_names}
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['region','n','lpips','ssim','dists']);writer.writeheader()
        for region,values in means.items():writer.writerow(dict(region=region,n=len(plan['ids']),**values))
    (out/'summary.json').write_text(json.dumps(dict(complete=True,arm=args.arm,samples=len(plan['ids']),
                                                  protocol=protocol,means=means),indent=2)+'\n')


if __name__=='__main__':
    main()
