"""Frozen Qwen semantic proposals -> frozen M3DLayout class-conditioned additions."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read, rows, write, validate_graph
from .geometry import (CATEGORY_IDS, CANONICAL_CATEGORIES, SIGNS, graph,
                       official_stats, world_object, yaw_matrix)
from .native_probe import transform
from .qwen_upstream import Qwen, DEFAULT_MODEL
from .qwen_upstream_review import pair_iou

REPO = Path('qwen_edit_pano/third_party/M3DLayout-code')
OFFSCREEN_REGIONS = ('left', 'right', 'behind_camera', 'above', 'below', 'unspecified')


def offscreen_prompt(observed, allowed, scene_text):
    # Deliberately omit noisy metric estimates and all GT from semantic inference.
    inventory = [dict(id=o['track_id'], category=CANONICAL_CATEGORIES[o['category_id']['value']],
                     bbox2d=o['bbox2d_visible_xyxy_norm']['value']) for o in observed['objects']]
    return (
        'Understand this partial RGB view and infer ONE plausible continuation OUTSIDE the image field of view. '
        'First describe only what is actually visible. Then propose additional object instances that may '
        'exist beyond the image boundaries or behind the camera. This is uncertain scene inference, not '
        'detection, renovation, or filling occluded areas INSIDE the image. '
        'Do not list visible objects again or describe objects as visible through a doorway/window. '
        'A second instance of an observed category is allowed only as a distinct off-screen hypothesis. '
        'Preserve visible empty surfaces. Use at most FOUR candidates; this is a ceiling, not a quota. '
        'Do not invent additions solely to reach four. No candidates is allowed for insufficient evidence. '
        'Do not give metric positions, sizes, or rotations. '
        'Return JSON with description (short visible-only English description) and candidates (list). '
        'Each candidate has category, evidence, relation, and region. evidence states the visible cue '
        'and commonsense reason making this unseen instance plausible, without claiming it was seen. '
        'relation is the hypothesized placement relative to visible objects. region is one of '
        'left, right, behind_camera, above, below, unspecified. These are relative to the input CAMERA; '
        'behind a bed is NOT automatically behind_camera. If the image cannot support a direction, '
        'use unspecified rather than pretend to know it. '
        'Choose category from: '+', '.join(allowed)+'. '
        'Existing predicted visible instances (may contain errors; inspect the image): '+json.dumps(inventory)+'. '
        'Optional user requirements: '+scene_text)


def validate_proposal(result, allowed, offscreen=False):
    if not isinstance(result['description'], str) or not isinstance(result['candidates'], list) or len(result['candidates']) > 4:
        raise ValueError('Expected description and at most four candidates')
    for item in result['candidates']:
        if item['category'] not in allowed or not isinstance(item['relation'], str):
            raise ValueError(f'Invalid candidate: {item}')
        if offscreen and (item['region'] not in OFFSCREEN_REGIONS or not isinstance(item['evidence'], str) or not item['evidence'].strip()):
            raise ValueError('Offscreen candidate requires evidence and a supported camera region')


def allowed_classes(stats):
    return [name for name in stats['class_labels'] if name in CATEGORY_IDS]


def make_context(observed, stats, height):
    """Fixed experimental bridge: Local axes ~ level camera, native y-up floor at -height.

    Native sizes use half extents. Local AABBs become yaw=0 boxes in the prior frame.
    No GT room alignment, depth or hidden object information is consulted.
    """
    objects, omitted = [], []
    classes = stats['class_labels']
    for obj in observed['objects']:
        name = CANONICAL_CATEGORIES[obj['category_id']['value']]
        native = 'panel_door' if name == 'door' else name
        if native not in classes:
            omitted.append(dict(id=obj['track_id'], reason='unsupported_native_category', category=name))
            continue
        if not all(obj[key]['valid'] for key in ('center_local_m', 'size_aabb_local_m')):
            omitted.append(dict(id=obj['track_id'], reason='unknown_geometry'))
            continue
        objects.append(dict(category=native,
            location=(np.asarray(obj['center_local_m']['value'])+[0, height, 0]).tolist(),
            size=(np.asarray(obj['size_aabb_local_m']['value'])/2).tolist(), rotation=0.))
    return objects, omitted


def context_tensors(objects, stats, device):
    import torch
    classes = stats['class_labels']
    labels = np.eye(len(classes), dtype=np.float32)[[classes.index(o['category']) for o in objects]]
    result = dict(class_labels=torch.as_tensor(labels[None], device=device))
    for key, field, width in [('translations', 'location', 3), ('sizes', 'size', 3), ('angles', 'rotation', 1)]:
        values = np.asarray([o[field] for o in objects], dtype=np.float32).reshape(-1, width)
        result[key] = torch.as_tensor(transform(values, key, stats)[None], dtype=torch.float32, device=device)
    return result


def decode_added(result, stats):
    # Official add_object returns START + context + exactly one new object + END.
    i = -2
    return dict(category=stats['class_labels'][int(result['class_labels'][0, i].argmax())],
        location=transform(result['translations'][0, i].detach().cpu().numpy(), 'translations', stats, True).tolist(),
        size=transform(result['sizes'][0, i].detach().cpu().numpy(), 'sizes', stats, True).tolist(),
        rotation=float(transform(result['angles'][0, i].detach().cpu().numpy(), 'angles', stats, True)[0]))


def local_box(native, height):
    center = np.asarray(native['location'], dtype=float)-[0, height, 0]
    corners = center + (SIGNS*np.asarray(native['size'])) @ yaw_matrix(native['rotation']).T
    return center, corners


def outside_view(corners, hfov, image_size):
    """Sufficient full-box exclusion test: all vertices outside one frustum plane.

    Conservative near corners; never equates a centre-only test with hidden visibility.
    """
    tx = np.tan(np.deg2rad(hfov)/2)
    ty = tx*image_size[1]/image_size[0]
    x, y, z = np.asarray(corners).T
    return bool(np.all(z < 0) or np.all(x > z*tx) or np.all(-x > z*tx)
                or np.all(y > z*ty) or np.all(-y > z*ty))


def matches_region(corners, region, hfov, image_size):
    if region == 'unspecified':
        return True  # The separate full-box offscreen test remains mandatory.
    tx = np.tan(np.deg2rad(hfov)/2)
    ty = tx*image_size[1]/image_size[0]
    x, y, z = np.asarray(corners).T
    return bool(dict(left=np.all(-x > z*tx) and x.mean() < 0,
                     right=np.all(x > z*tx) and x.mean() > 0,
                     above=np.all(y > z*ty) and y.mean() > 0,
                     below=np.all(-y > z*ty) and y.mean() < 0,
                     behind_camera=np.all(z < 0))[region])


def generate(observed, proposal, stats, network, image_size, hfov, height, attempts, capacity, device):
    import torch
    context, omitted = make_context(observed, stats, height)
    additions, audit = [], []
    for index, item in enumerate(proposal['candidates']):
        accepted = False
        for attempt in range(attempts):
            if len(observed['objects'])+len(additions) >= capacity or len(context) >= 120:
                audit.append(dict(candidate=index, reason='capacity_reached'))
                break
            # Pass CUDA one-hot explicitly; official int helper constructs a CPU tensor.
            label = torch.zeros(1, 1, len(stats['class_labels']), device=device)
            label[0, 0, stats['class_labels'].index(item['category'])] = 1
            text = proposal['description']+' Proposed unseen object: '+item['category']+'. '+item['relation']
            if 'region' in item:
                text += ' Hypothesized camera region: '+item['region']+'.'
            result = network.add_object(torch.ones(1, 1, 64, 64, device=device), label,
                boxes=context_tensors(context, stats, device), device=device, text=text)
            native = decode_added(result, stats)
            center, corners = local_box(native, height)
            reason = None
            if not np.isfinite(corners).all() or not np.isfinite(native['rotation']) or min(native['size']) <= 0:
                reason = 'invalid_geometry'
            elif not outside_view(corners, hfov, image_size):
                reason = 'not_wholly_outside_input_view'
            elif not matches_region(corners, item.get('region', 'unspecified'), hfov, image_size):
                reason = 'wrong_hypothesized_region'
            obj = None
            if reason is None:
                obj = world_object(f'hybrid_hidden_{index:04d}', native['category'], center, corners,
                                   'm3dlayout_class_conditioned_hypothesis', False)
                if any(obj['category_id']['value'] == old['category_id']['value'] and pair_iou(obj, old) >= .5
                       for old in observed['objects']+additions):
                    reason = 'same_category_duplicate'
            audit.append(dict(candidate=index, attempt=attempt, native=native,
                center_local_m=center.tolist(), accepted=reason is None, reason=reason))
            if reason is None:
                obj['hypothesized_hidden'] = True
                obj['hypothesis_scope'] = 'outside_input_fov'
                obj['proposal'] = deepcopy(item)
                additions.append(obj)
                context.append(native)  # No START/END tokens are recycled into the next call.
                accepted = True
                break
        if not accepted:
            audit.append(dict(candidate=index, outcome='no_accepted_geometry'))
    hidden = graph(observed['id'], observed['split'], 'hidden', additions, observed['vocabulary_sha256'])
    full = graph(observed['id'], observed['split'], 'full',
                 deepcopy(observed['objects'])+deepcopy(additions), observed['vocabulary_sha256'])
    validate_graph(full)
    return hidden, full, audit, omitted


def propose(args):
    stats = official_stats(args.repo)
    allowed = allowed_classes(stats)
    model = Qwen(args.model)
    failures = []
    for entry in rows(args.bundle/'inputs.jsonl'):
        d = args.output/'proposals'/entry['id']; d.mkdir(parents=True, exist_ok=True)
        observed = read(args.observer/entry['id']/'G_obs.json')
        visible = [CANONICAL_CATEGORIES[o['category_id']['value']] for o in observed['objects']]
        prompt = (
            'Describe the visible scene briefly, then propose at most FOUR additional object instances '
            'that could plausibly exist outside this camera view for a panorama from the same camera. '
            'These are hypotheses, not observations. Preserve visible objects and empty space. '
            'Do not renovate the visible image. Do not give coordinates, dimensions or rotations. '
            'Use one candidate per instance; an empty list is allowed. '
            'Return JSON with description (a short English scene description), candidates (a list of '
            'objects each with category and relation). relation is a short English hypothetical '
            'placement relative to the CAMERA or visible objects, explicitly outside the view. '
            'Allowed category names: '+', '.join(allowed)+'. '
            'Observed category inventory: '+', '.join(visible)+'. '
            'Optional user requirements: '+entry.get('scene_text', ''))
        if getattr(args, 'offscreen', False):
            prompt = offscreen_prompt(observed, allowed, entry.get('scene_text', ''))
        try:
            with Image.open(entry['local_rgb']) as image:
                result = model.ask(image.convert('RGB'), prompt, d/'proposal.txt', profile='official', max_tokens=1536)
            validate_proposal(result, allowed, getattr(args, 'offscreen', False))
            write(d/'proposal.json', result)
            write(d/'status.json', dict(status='completed', candidates=len(result['candidates'])))
            print(entry['id'], len(result['candidates']), 'semantic candidates', flush=True)
        except (ValueError, KeyError, TypeError) as exc:
            write(d/'status.json', dict(status='failed', error=str(exc))); failures.append(entry['id'])
    if failures:
        raise SystemExit(f'Proposal failures: {failures}; raw answers preserved')


def run(args):
    import torch
    from .complete import load_pretrained
    network, stats = load_pretrained(args.repo, args.weights, args.device)
    metadata = read(args.bundle/'bundle.json')
    refs = {r['id']: r for r in rows(args.bundle/'references.jsonl')}
    destination = args.output/'completion'
    destination.mkdir(parents=True, exist_ok=True)
    write(destination/'config.json', dict(conditions=['gt', 'pred'], seeds=[0],
        observer=str(args.observer), camera_height=args.camera_height, attempts=args.attempts,
        bridge='assumed_level_camera_at_0_height_0; Local AABB -> native yaw0 half extents',
        door_mapping='door -> panel_door (generic representative)',
        weights=str(args.weights), native_stats=stats, text_condition=True,
        protocol='offscreen_evidence_v1' if getattr(args, 'offscreen', False) else 'hybrid_v1',
        scope='off-camera hypotheses only; frustum exclusion does not guarantee plausible placement'))
    failures = []
    for entry in rows(args.bundle/'inputs.jsonl'):
        sid = entry['id']; pd = args.output/'proposals'/sid
        for condition in ['gt', 'pred']:
            d = destination/condition/sid/'seed0'; d.mkdir(parents=True, exist_ok=True)
            try:
                if read(pd/'status.json')['status'] != 'completed':
                    raise ValueError('Semantic proposal unavailable')
                proposal = read(pd/'proposal.json')
                validate_proposal(proposal, allowed_classes(stats), getattr(args, 'offscreen', False))
                observed = read(refs[sid]['gt_obs'] if condition == 'gt' else args.observer/sid/'G_obs.json')
                with Image.open(entry['local_rgb']) as image:
                    image_size = image.size
                torch.manual_seed(0)
                with torch.inference_mode():
                    hidden, full, audit, omitted = generate(observed, proposal, stats, network,
                        image_size, entry['hfov_deg'], args.camera_height, args.attempts,
                        metadata['max_objects'], args.device)
                write(d/'G_obs.json', observed); write(d/'G_hidden.json', hidden); write(d/'G_full.json', full)
                write(d/'sampling_audit.json', audit); write(d/'omitted_context.json', omitted)
                write(d/'semantic_proposal.json', proposal)
                (d/'hidden_plan.txt').write_text(proposal['description']+'\n'+'\n'.join(
                    item['category']+': '+item['relation'] for item in proposal['candidates']))
                write(d/'status.json', dict(status='completed', condition=condition, seed=0,
                    generated=len(hidden['objects']), proposed=len(proposal['candidates']),
                    sampled=sum('native' in item for item in audit),
                    rejected=sum('native' in item and not item['accepted'] for item in audit),
                    observed=len(observed['objects']), omitted_context=len(omitted),
                    outcome='candidate_generated' if hidden['objects'] else 'no_accepted_geometry'))
                print(condition, sid, f'{len(hidden["objects"])}/{len(proposal["candidates"])} accepted', flush=True)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                write(d/'status.json', dict(status='failed', error=str(exc))); failures.append(str(d))
    if failures:
        raise SystemExit(f'{len(failures)} cases failed; run review to include failures')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['propose', 'run'])
    p.add_argument('--bundle', type=Path, default=Path('qwen_edit_pano/data/qwen_upstream_val12_v1'))
    p.add_argument('--observer', type=Path, default=Path('qwen_edit_pano/outputs/qwen_world_simple_run001/observer'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--weights', type=Path, default=REPO/'weights/autoregressive_59000.pth')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--camera-height', type=float, default=1.5)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--offscreen', action='store_true', help='Explicit visible evidence and offscreen hypotheses')
    args = p.parse_args()
    {'propose': propose, 'run': run}[args.action](args)


if __name__ == '__main__':
    main()
