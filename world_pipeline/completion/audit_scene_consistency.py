"""Small CPU-only GT/observer coordinate audit. Never changes input graphs."""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common import read, write, sha
from .scene_geometry import camera_geometry, observation_report, projection_measurement

DEFAULT_IDS = ['zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a',
               'zsNo4HB9uLZ_ed8645386a0a4de2bfc486e0a1d02468',
               'zsNo4HB9uLZ_38e0c09ac7a748dbadea6471861b30c3']


def audit_gt(directory):
    observed=read(directory/'G_obs.json'); bridge=read(directory/'bridge.json')
    source=read(directory/'provenance.json')
    catalog_path=next(Path(p) for p in source['sources'] if p.endswith('/objects.json'))
    catalog={str(o['id']):o for o in read(catalog_path)['objects']}
    basis=np.array(bridge['basis_local_to_world']); origin=np.array(bridge['origin_world'])
    signs=np.array(list(itertools.product([-1.,1.],repeat=3)))
    records=[]
    for obj in observed['objects']:
        raw=catalog[obj['track_id']]; axes=np.array(raw['axes_world']); radii=np.array(raw['radii'])
        world_center=np.array(raw['center_world'])
        corners_world=world_center+(signs*radii)@axes
        corners=(corners_world-origin)@basis
        center=np.array(obj['center_local_m']['value']); size=np.array(obj['size_aabb_local_m']['value'])
        bbox=obj['bbox2d_visible_xyxy_norm']['value']; u,v=(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2
        proxy=np.array([(u-.5)*2*center[2],(.5-v)*2*center[2],center[2]])
        records.append(dict(id=obj['track_id'],category_id=obj['category_id']['value'],
            center_max_error_m=float(abs(center-corners.mean(0)).max()),
            size_max_error_m=float(abs(size-np.ptp(corners,axis=0)).max()),
            world_roundtrip_max_error_m=float(abs(center@basis.T+origin-world_center).max()),
            visible_center_proxy_error_at_GT_z_m=float(np.linalg.norm(proxy-center)),
            projection=projection_measurement(center.tolist(),size.tolist(),bbox,90,(512,512))))
    return dict(id=directory.name,up_local=bridge['up_local'],levelled=bridge['levelled'],
        image_up_gravity_angle_deg=math.degrees(math.acos(np.clip(bridge['up_local'][1],-1,1))),
        determinant=float(np.linalg.det(basis)),orthogonality_max_error=float(abs(basis.T@basis-np.eye(3)).max()),
        floor_plane=read(directory/'G_full.json')['room']['floor_plane_local'],
        source_catalog_sha256=sha(catalog_path),source_catalog_matches_provenance=sha(catalog_path)==source['sources'][str(catalog_path)],
        objects=records,note='Independent source-OBB corners -> camera AABB; GT is a reference, not observer input')


def iou(a,b):
    a,b=np.array(a),np.array(b)
    overlap=float(np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))))
    return overlap/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap)


def compare_prediction(predicted,gt):
    # Diagnostic matches, not a certified instance correspondence. Multiple GT
    # bed components may correspond to one visible predicted bed inventory item.
    matches=[]
    for category in {o['category_id']['value'] for o in predicted['objects']}:
        a=[o for o in predicted['objects'] if o['category_id']['value']==category]
        b=[o for o in gt['objects'] if o['category_id']['value']==category]
        if not b: continue
        costs=np.array([[1-iou(o['bbox2d_visible_xyxy_norm']['value'],p['bbox2d_visible_xyxy_norm']['value']) for p in b] for o in a])
        rows,cols=linear_sum_assignment(costs)
        for r,c in zip(rows,cols):
            x,y=a[r],b[c]
            matches.append(dict(pred_id=x['track_id'],gt_id=y['track_id'],category_id=category,
                bbox_iou=float(1-costs[r,c]),
                pred_center=x['center_local_m']['value'],gt_center=y['center_local_m']['value'],
                pred_full_camera_aabb_size=x['size_aabb_local_m']['value'],gt_full_camera_aabb_size=y['size_aabb_local_m']['value']))
    return dict(matches=matches,note='Within-category maximum-IoU assignment; inspect IoU and instance granularity before interpreting 3D differences')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction',type=Path,default=Path('qwen_edit_pano/outputs/semantic_world_run013/G_obs.json'))
    p.add_argument('--gt-root',type=Path,default=Path('qwen_edit_pano/data/gt_world_supplement_v1/samples'))
    p.add_argument('--ids',nargs='+',default=DEFAULT_IDS)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    gt=[audit_gt(args.gt_root/sid) for sid in args.ids]
    write(args.output/'gt_coordinate_audit.json',gt)
    config=read(args.prediction.parent/'config.json'); state=read(args.prediction)
    geometry=config.get('camera_geometry') or camera_geometry(camera_height=config['camera_height_prior_m'])
    write(args.output/'prediction_projection_audit.json',observation_report(state,config['hfov_deg'],tuple(config['model_image_size']),geometry))
    from PIL import Image
    # Bind the same-image comparison before using GT, with identical resizing.
    with Image.open(args.prediction.parent/'local.png') as handle: rgb=handle.convert('RGB')
    with Image.open(args.gt_root/args.ids[0]/'local_rgb.png') as handle: ref=handle.convert('RGB')
    ref.thumbnail(rgb.size)
    if ref.size!=rgb.size or ref.tobytes()!=rgb.tobytes():
        raise ValueError('First GT ID is not bound to the prediction RGB; no 3D matching performed')
    write(args.output/'same_image_gt_diagnostic.json',compare_prediction(state,read(args.gt_root/args.ids[0]/'G_obs.json')))
    print('CPU audit written; no model/GPU/download and no input mutation: '+str(args.output))


if __name__=='__main__': main()
