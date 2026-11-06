"""Qwen's same-image 2D inventory -> pretrained WildDet3D -> Adapter boxes.

No scene GT, training, mesh reconstruction, floor alignment or downloads.
Heavy dependencies are imported only for preflight/inference, not conversion.
"""
import argparse
from copy import deepcopy
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from ..common import ROOT, field, new_output, read, sha, validate_graph, write
from .geometry import CANONICAL_CATEGORIES
from .scene_geometry import camera_geometry

DEFAULT_SOURCE = ROOT/'qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json'
DEFAULT_REPO = ROOT/'qwen_edit_pano/third_party/WildDet3D'
DEFAULT_CHECKPOINT = DEFAULT_REPO/'ckpt/wilddet3d_alldata_all_prompt_v1.0.pt'
LINGBOT = 'robbyant/lingbot-depth-postrain-dc-vitl14'


def intrinsics(width, height, hfov):
    """Original RGB pixel intrinsics; official preprocess handles resize/pad."""
    if not math.isfinite(hfov) or not 1 <= hfov <= 160:
        raise ValueError('Require perspective horizontal FOV in [1,160] degrees')
    focal = width/(2*math.tan(math.radians(hfov)/2))
    return np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float32)


def camera_aabb(center_opencv, corners_opencv):
    """Actual rotated corners -> full camera-axis AABB, preserving metres.

    Native x-right/y-down/z-forward -> project x-right/y-image-up/z-forward.
    Corner decoding is delegated to vis4d, exactly as upstream visualization;
    never assume the native three dimension slots are xyz lengths.
    """
    center = np.asarray(center_opencv, dtype=float)
    corners = np.asarray(corners_opencv, dtype=float)
    if (center.shape != (3,) or corners.shape != (8, 3)
            or not np.isfinite(center).all() or not np.isfinite(corners).all()):
        raise ValueError('Need a finite camera center and eight finite corners')
    if center[2] <= 0 or not np.allclose(corners.mean(0), center, rtol=1e-4, atol=1e-4):
        raise ValueError('Native box has invalid forward depth or inconsistent center/corners')
    corners = corners * [1, -1, 1]
    lo, hi = corners.min(0), corners.max(0)
    if np.any(hi <= lo):
        raise ValueError('Degenerate 3D box')
    return ((lo+hi)/2).tolist(), (hi-lo).tolist()


def plotting_envelope(objects):
    """Legacy completion needs bounds; these bounds are NOT recovered walls.

    Enclose the predicted boxes and camera without changing any geometry.
    The y limits are display bounds, not a floor/ceiling measurement.
    """
    numeric = [o for o in objects if o['center_local_m']['valid'] and o['size_aabb_local_m']['valid']]
    centers = np.array([o['center_local_m']['value'] for o in numeric]).reshape(-1, 3)
    sizes = np.array([o['size_aabb_local_m']['value'] for o in numeric]).reshape(-1, 3)
    lo, hi = np.array([-2., -1.5, -2.]), np.array([2., 1.2, 2.])
    if len(numeric):
        lo = np.minimum(lo, (centers-sizes/2).min(0)-.5)
        hi = np.maximum(hi, (centers+sizes/2).max(0)+.5)
    return dict(x_min=float(lo[0]), x_max=float(hi[0]), floor_y=float(lo[1]),
                ceiling_y=float(hi[1]), z_min=float(lo[2]), z_max=float(hi[2]))


def export_state(source, records):
    """Preserve IDs, categories and 2D evidence; replace ALL old numeric guesses."""
    state = deepcopy(validate_graph(source))
    ids = [o['track_id'] for o in state['objects']]
    if len(records) != len(ids) or [r['id'] for r in records] != ids:
        raise ValueError('One result in source order is required for every inventory ID')
    simple = []
    for obj, record in zip(state['objects'], records):
        obj['center_local_m'] = field(None, reason='3d_detector_did_not_return_valid_geometry')
        obj['size_aabb_local_m'] = field(None, reason='3d_detector_did_not_return_valid_geometry')
        if record.get('status') == 'predicted':
            center, size = camera_aabb(record['box3d_native'][:3], record['corners_native'])
            obj['center_local_m'] = field(center, 'model_prediction', 'wilddet3d_camera_aabb_center')
            obj['size_aabb_local_m'] = field(size, 'model_prediction', 'wilddet3d_corner_max_minus_min')
            simple.append(dict(category=CANONICAL_CATEGORIES[obj['category_id']['value']],
                               position=center, size=size))
    state.update(stage='prediction', scope='observed', relations=[], ready_for_training=False,
                 observer='qwen_2d_plus_wilddet3d_geometric',
                 room_hypothesis=plotting_envelope(state['objects']),
                 room_hypothesis_method='display_envelope_only_not_room_reconstruction',
                 camera_geometry=dict(levelled=False, gravity_verified=False, floor_verified=False),
                 observation_geometry_verified=False)
    state.pop('geometry_assumptions', None)
    return validate_graph(state), simple


def load_source(path, image_path, hfov):
    source = validate_graph(read(path))
    if source.get('stage') != 'prediction' or source['scope'] != 'observed' or not source['objects']:
        raise ValueError('Use a nonempty predicted G_obs; scene GT is not an inference input')
    vocab = read(ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json')
    if source.get('vocabulary_sha256') != vocab['category_mapping_sha256']:
        raise ValueError('Vocabulary differs from Adapter vocabulary')
    config = read(path.parent/'config.json')
    if not math.isclose(config['hfov_deg'], hfov, abs_tol=1e-9):
        raise ValueError('FOV differs from source observation')
    image_path = image_path or Path(config['image'])
    with Image.open(image_path) as im:
        image = im.convert('RGB')
    small = image.copy(); small.thumbnail((768, 768))
    with Image.open(path.parent/'local.png') as im:
        old = im.convert('RGB')
    old.thumbnail((768, 768))
    if old.size != small.size or old.tobytes() != small.tobytes():
        raise ValueError('Image differs from the source 2D observation')
    for obj in source['objects']:
        if obj['category_id']['value'] not in CANONICAL_CATEGORIES or not obj['bbox2d_visible_xyxy_norm']['valid']:
            raise ValueError('Every source object needs a supported category and a 2D box')
    return source, image, image_path


def preflight(repo, checkpoint):
    """Imports and local cache checks only; no model construction or CUDA work."""
    for relative in ('wilddet3d/inference.py', 'third_party/sam3/sam3/model_builder.py',
                     'third_party/lingbot_depth/mdm/model/v2.py'):
        if not (repo/relative).is_file():
            raise ValueError(f'Missing {repo/relative}; clone official repo with submodules first')
    if not checkpoint.is_file():
        raise ValueError(f'Missing local checkpoint: {checkpoint}')
    sys.path.insert(0, str(repo.resolve()))
    modules = {name: importlib.import_module(name) for name in ('torch', 'vis4d', 'wilddet3d')}
    ops = importlib.import_module('vis4d_cuda_ops')
    if not hasattr(ops, 'ms_deform_attn_forward'):
        raise ValueError('vis4d CUDA extension is incomplete; follow upstream installation')
    from huggingface_hub import hf_hub_download
    # Upstream skip_pretrained=True still reads model_config from this file.
    cached = hf_hub_download(LINGBOT, 'model.pt', local_files_only=True)
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners
    # Small CPU execution also verifies the installed native box convention.
    native = modules['torch'].tensor([[0., 0., 3., 2., 4., 6., 1., 0., 0., 0.]])
    decoded = boxes3d_to_corners(native, AxisMode.OPENCV).numpy()[0]
    center, size = camera_aabb([0, 0, 3], decoded)
    if not np.allclose(size, [4, 6, 2]):
        raise ValueError('Installed vis4d box convention differs from verified v1.0.0')
    return dict(torch=modules['torch'].__version__, lingbot_config_checkpoint=cached,
                source_repo=str(repo.resolve()), checkpoint=str(checkpoint.resolve()),
                cpu_corner_contract_verified=True, gpu_inference_executed=False)


def prompt_pixels(obj, image_size):
    return (np.asarray(obj['bbox2d_visible_xyxy_norm']['value'])*[*image_size, *image_size]).tolist()


def bbox_iou(a, b):
    a, b = np.array(a), np.array(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union > 0 else 0.


def projection_preview(image, state, K, path):
    """Project exported camera AABBs, without changing them or fitting 2D boxes."""
    canvas = image.copy(); canvas.thumbnail((1024, 1024)); draw = ImageDraw.Draw(canvas)
    scales = np.array([canvas.width/image.width, canvas.height/image.height])
    signs = np.array(list(itertools.product([-1., 1.], repeat=3)))
    for obj in state['objects']:
        center, size = np.array(obj['center_local_m']['value']), np.array(obj['size_aabb_local_m']['value'])
        points = (center+signs*size/2)*[1, -1, 1]
        # Only draw complete forward-facing boxes; never project through z=0.
        if np.any(points[:, 2] <= .01):
            continue
        homogeneous = points@K.T
        uv = homogeneous[:, :2]/homogeneous[:, 2:]*scales
        for i, j in itertools.combinations(range(8), 2):
            if np.count_nonzero(signs[i] != signs[j]) == 1:
                draw.line([tuple(uv[i]), tuple(uv[j])], fill='lime', width=2)
        target = prompt_pixels(obj, image.size)*np.tile(scales, 2)
        draw.rectangle(target.tolist(), outline='red', width=2)
        draw.text(tuple(target[:2]), obj['track_id'], fill='red')
    canvas.save(path)


def infer_one(model, data, obj, image_size, device):
    """One official geometric+label prompt per ID, avoiding multi-prompt batching ambiguity."""
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners
    # A clock remains "clock" for the open-vocabulary detector, while the
    # unchanged Adapter vocabulary receives official mpcat40 "objects" (39).
    label = obj.get('raw_category', CANONICAL_CATEGORIES[obj['category_id']['value']]).replace('_', ' ')
    pixels = prompt_pixels(obj, image_size)
    results = model(images=data['images'].to(device), intrinsics=data['intrinsics'].to(device)[None],
                    input_hw=[data['input_hw']], original_hw=[data['original_hw']],
                    padding=[data['padding']], input_boxes=[pixels], prompt_text='geometric: '+label)
    boxes, boxes3d, scores, scores2d, scores3d, _, _ = results
    if len(boxes3d) != 1 or tuple(boxes3d[0].shape) != (1, 10):
        raise ValueError('Official geometric mode must return one 10-value box per prompt')
    native = boxes3d[0].detach().cpu().float()
    if not np.isfinite(native.numpy()).all() or np.any(native.numpy()[0, 3:6] <= 0):
        raise ValueError('Invalid native box')
    corners = boxes3d_to_corners(native, AxisMode.OPENCV).numpy()[0].tolist()
    camera_aabb(native[0, :3].tolist(), corners)
    predicted2d = boxes[0].detach().cpu().tolist()[0]
    return dict(id=obj['track_id'], status='predicted', prompt_category=label, box3d_native=native.tolist()[0],
                corners_native=corners, predicted_bbox_pixels=predicted2d, prompt_bbox_pixels=pixels,
                prompt_prediction_iou=bbox_iou(pixels, predicted2d),
                score=float(scores[0][0]), score_2d=float(scores2d[0][0]), score_3d=float(scores3d[0][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='Reuse ONLY categories/IDs/2D boxes')
    parser.add_argument('--image', type=Path, help='Default: RGB path recorded in source config.json')
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--hfov', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    # No Hugging Face download in either preflight or inference.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['SAM3_COMPILE'] = '0'
    source, image, image_path = load_source(args.source, args.image, args.hfov)
    K = intrinsics(*image.size, args.hfov)
    if args.output.exists():
        parser.error('Use a new output directory')
    readiness = preflight(args.repo, args.checkpoint)
    if args.preflight:
        print(json.dumps(dict(status='preflight_passed', objects=len(source['objects']), **readiness)))
        return
    import torch
    from wilddet3d import build_model, preprocess
    output = new_output(args.output)
    small = image.copy(); small.thumbnail((768, 768)); small.save(output/'local.png')
    write(output/'config.json', dict(protocol='qwen_2d_wilddet3d_geometric_v1', image=str(image_path.resolve()),
          image_sha256=sha(image_path), hfov_deg=args.hfov, intrinsics_original_pixels=K.tolist(),
          source=str(args.source.resolve()), source_sha256=sha(args.source),
          vocabulary_sha256=source['vocabulary_sha256'], model_image_size=list(small.size),
          camera_height_prior_m=1.5, room_height_prior_m=2.7,
          camera_geometry=camera_geometry(), diagnostic_floor_prior_only=True,
          checkpoint_sha256=sha(args.checkpoint), upstream_inference_sha256=sha(args.repo/'wilddet3d/inference.py'),
          old_numeric_geometry_reused=False, scene_gt_used=False, **readiness))
    try:
        model = build_model(checkpoint=str(args.checkpoint.resolve()), skip_pretrained=True,
                            device='cuda', use_predicted_intrinsics=False, use_depth_input_test=False)
        data = preprocess(np.array(image, dtype=np.float32), K)
        records = []
        with torch.no_grad():
            for obj in source['objects']:
                print('3D box: '+obj['track_id'], flush=True)
                record = infer_one(model, data, obj, image.size, 'cuda')
                records.append(record)
                if len(records) == 1:
                    config = read(output/'config.json')
                    config['gpu_inference_executed'] = True
                    write(output/'config.json', config)
                # Keep successful raw per-object results if a later object fails.
                write(output/'native_boxes.json', records)
        state, simple = export_state(source, records)
        write(output/'G_obs.json', state)
        write(output/'objects.json', simple)
        projection_preview(image, state, K, output/'projection_boxes.png')
        # Red: original Qwen boxes; green: detector-selected boxes. No geometry correction.
        preview = image.copy(); preview.thumbnail((1024, 1024)); draw = ImageDraw.Draw(preview)
        scale = np.array([preview.width/image.width, preview.height/image.height]*2)
        for record in records:
            for key, color in (('prompt_bbox_pixels', 'red'), ('predicted_bbox_pixels', 'lime')):
                rect = np.asarray(record[key])*scale
                draw.rectangle(rect.tolist(), outline=color, width=2)
            draw.text(tuple(np.asarray(record['prompt_bbox_pixels'][:2])*scale[:2]), record['id'], fill='red')
        preview.save(output/'instance_binding.png')
        summary = dict(status='completed', observed=len(records), numeric_boxes=len(simple),
                       gpu_inference_executed=bool(records),
                       adapter_fields=['category_id', 'center_local_m', 'size_aabb_local_m'],
                       old_numeric_geometry_reused=False, completion_executed=False,
                       mean_prompt_prediction_iou=float(np.mean([r['prompt_prediction_iou'] for r in records])),
                       no_overlap_prompt_ids=[r['id'] for r in records if r['prompt_prediction_iou'] == 0],
                       observation_geometry_verified=False, quality_status='requires_visual_and_gt_review')
        write(output/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        write(output/'summary.json', dict(status='failed', error=str(exc), completed_raw_boxes=len(read(output/'native_boxes.json'))
              if (output/'native_boxes.json').exists() else 0))
        raise


if __name__ == '__main__':
    main()
