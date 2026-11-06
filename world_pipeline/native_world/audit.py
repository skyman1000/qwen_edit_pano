"""Small CPU visual review of existing polished targets and mesh-instance labels.

It neither relabels nor filters training data. The outline is a mesh annotation,
not a detector output or a certification that polishing preserves every object.
"""
import argparse
import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..build_polished_world import ROOT, atomic_json
from ..object_world.data import target_image
from qwen_pano.geometry import perspective
from . import data as d


def build(output):
    torch.set_num_threads(2)
    plan, rows = d.inputs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    used = set()
    for category in (22, 22, 3, 3, 14, 11, 10, 6):
        candidates = [r for r in rows if r['id'] not in used and any(o['category_id']==category for o in r['objects'])]
        found = None
        for row in candidates[:80]:
            with np.load(row['object_region_supervision']['path'], allow_pickle=False) as f:
                instance, valid = f['instance'].copy(), f['object_loss_valid'].copy()
            objects = [(int((valid&(instance==int(o['instance_id']))).sum()),o)
                       for o in row['objects'] if o['category_id']==category]
            count, obj = max(objects, key=lambda x:x[0])
            if count >= 256 and obj['center_valid']:
                found = row, obj, instance, valid, count
                break
        if found is None:
            raise ValueError('No representative supported training example for category '+str(category))
        row,obj,instance,valid,count = found
        used.add(row['id'])
        x,y,z = obj['center_local_m']
        yaw=math.degrees(math.atan2(x,z));pitch=math.degrees(math.atan2(y,math.hypot(x,z)))
        rgb=target_image(row).resize((1024,512),Image.Resampling.BICUBIC)
        t=torch.from_numpy(np.array(rgb).copy()).permute(2,0,1)[None].float()/255
        support=torch.from_numpy((valid&(instance==int(obj['instance_id']))).astype(np.float32))[None,None]
        with torch.no_grad():
            crop=perspective(t,yaw,pitch,70,320)[0].permute(1,2,0).clamp(0,1).numpy()
            m=perspective(support,yaw,pitch,70,320)[0,0].numpy()>.25
        pixels=(crop*255).round().astype(np.uint8)
        overlay=pixels.copy()
        edge=m&~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1))
        overlay[edge]=[255,0,0]
        canvas=Image.new('RGB',(640,354),'white')
        canvas.paste(Image.fromarray(pixels),(0,34));canvas.paste(Image.fromarray(overlay),(320,34))
        name=plan['names'][str(category)]
        ImageDraw.Draw(canvas).text((5,5),f'{name} / ID {category} / instance {obj["instance_id"]} / mesh-label outline',fill='black')
        file=f'{len(selected):02d}_{name}.jpg';canvas.save(output/file,quality=94)
        selected.append(dict(id=row['id'],split='train',category_id=category,name=name,
            instance_id=obj['instance_id'],supported_pixels=count,file=file,yaw=yaw,pitch=pitch))
    atomic_json(output/'SAMPLES.json',dict(samples=selected,GPU=False,hashes_computed=False,
        changes_to_data=False,note='Eight selected training crops only; not full data certification'))
    body=''.join(f'<p>{html.escape(s["id"])} · {html.escape(s["name"])}</p><img src="{s["file"]}">' for s in selected)
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Polished supervision review</title>'+body)
    print(dict(output=str(output),samples=len(selected),GPU=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'research/audits/native_world_2026-11-04/supervision')
    build(p.parse_args().output)


if __name__=='__main__':main()
