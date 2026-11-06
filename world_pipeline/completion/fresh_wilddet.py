"""The SAME official WildDet3D checkpoint for both fresh 2D inventories."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image

from ..common import field,read,sha,validate_graph,write
from .box_observer import (DEFAULT_REPO,DEFAULT_CHECKPOINT,preflight,intrinsics,
    camera_aabb,prompt_pixels,bbox_iou,projection_preview)


def infer(model,data,obj,image_size):
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners
    label=obj['raw_category'].replace('_',' ');pixels=prompt_pixels(obj,image_size)
    result=model(images=data['images'].to('cuda'),intrinsics=data['intrinsics'].to('cuda')[None],
        input_hw=[data['input_hw']],original_hw=[data['original_hw']],padding=[data['padding']],
        input_boxes=[pixels],prompt_text='geometric: '+label)
    boxes,native,scores,scores2d,scores3d,_,_=result
    if len(native)!=1 or tuple(native[0].shape)!=(1,10):raise ValueError('Expected one official native box per prompt')
    box=native[0].detach().cpu().float()
    if not np.isfinite(box.numpy()).all() or np.any(box.numpy()[0,3:6]<=0):raise ValueError('Invalid native dimensions')
    corners=boxes3d_to_corners(box,AxisMode.OPENCV).numpy()[0].tolist()
    camera_aabb(box[0,:3].tolist(),corners)
    predicted=boxes[0].detach().cpu().tolist()[0]
    return dict(id=obj['track_id'],box3d_native=box[0].tolist(),corners_native=corners,
        prompt_category=label,prompt_bbox_pixels=pixels,predicted_bbox_pixels=predicted,
        prompt_prediction_iou=bbox_iou(pixels,predicted),score=float(scores[0][0]),
        score_2d=float(scores2d[0][0]),score_3d=float(scores3d[0][0]))


def export(source,records,backend):
    state=deepcopy(validate_graph(source))
    if [r['id'] for r in records]!=[o['track_id'] for o in state['objects']]:raise ValueError('WildDet instance order differs')
    for obj,r in zip(state['objects'],records):
        center,size=camera_aabb(r['box3d_native'][:3],r['corners_native'])
        obj['center_local_m']=field(center,'model_prediction','wilddet3d_native_corner_aabb')
        obj['size_aabb_local_m']=field(size,'model_prediction','wilddet3d_native_corner_max_minus_min')
    state.update(observer=backend+'_2d_plus_official_wilddet3d',scene_gt_used=False,
        observation_geometry_verified=False,camera_geometry=dict(levelled=False,gravity_verified=False,floor_verified=False))
    return validate_graph(state)


def run(root,backend,repo,checkpoint,check_experiment=None):
    if check_experiment is None:
        from .fresh_pipeline import check as check_experiment
    config=check_experiment(root);source_path=root/backend/'observer2d'/'G_2d.json'
    source=validate_graph(read(source_path));out=root/backend/'observer3d';out.mkdir(parents=True,exist_ok=False)
    readiness=preflight(repo,checkpoint)
    import torch
    from wilddet3d import build_model,preprocess
    with Image.open(root/'input.png') as im:image=im.convert('RGB')
    K=intrinsics(*image.size,config['hfov'])
    write(out/'config.json',dict(backend=backend,hfov=config['hfov'],source_sha256=sha(source_path),
        image_sha256=sha(root/'input.png'),checkpoint_sha256=sha(checkpoint),
        upstream_inference_sha256=sha(repo/'wilddet3d/inference.py'),intrinsics=K.tolist(),**readiness))
    records=[]
    try:
        if source['objects']:
            model=build_model(checkpoint=str(checkpoint.resolve()),skip_pretrained=True,device='cuda',
                use_predicted_intrinsics=False,use_depth_input_test=False)
            data=preprocess(np.array(image,dtype=np.float32),K)
            with torch.inference_mode():
                for obj in source['objects']:
                    print('WildDet3D: '+obj['track_id'],flush=True)
                    records.append(infer(model,data,obj,image.size));write(out/'native_boxes.json',records)
        else:write(out/'native_boxes.json',records)
        state=export(source,records,backend);write(out/'G_obs.json',state)
        projection_preview(image,state,K,out/'projection_boxes.png')
        summary=dict(status='completed',objects=len(records),old_numeric_geometry_reused=False,
            no_overlap_prompt_ids=[r['id'] for r in records if r['prompt_prediction_iou']==0],
            observation_geometry_verified=False)
        write(out/'summary.json',summary);return summary
    except Exception as exc:
        write(out/'summary.json',dict(status='failed',completed_raw_boxes=len(records),error=type(exc).__name__+': '+str(exc)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    p.add_argument('--backend',choices=['api','qwen']);p.add_argument('--preflight',action='store_true')
    p.add_argument('--repo',type=Path,default=DEFAULT_REPO);p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    a=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SAM3_COMPILE='0')
    if a.preflight:print(preflight(a.repo,a.checkpoint));return
    if a.root is None or a.backend is None:p.error('--root and --backend required')
    run(a.root.resolve(),a.backend,a.repo,a.checkpoint)


if __name__=='__main__':main()
