"""Fixed observed boxes -> one jointly planned set of new camera-space boxes.

The model proposes contents and geometry together. No candidate grid, floor
snapping, support solver, automatic withdrawal or geometry optimization.
"""
from copy import deepcopy
import json
import math

import numpy as np

from ..common import finite_vector
from .box_layout import frustum_margin, projected_box
from .geometry import CANONICAL_CATEGORIES

PROMPT_VERSION = 'fixed_observation_joint_camera_boxes_v2'
REVISION_VERSION = 'fixed_inventory_camera_box_projection_feedback_v1'


def prompt(observed, observed_layout, hfov, capacity):
    by_id = {o['id']: o for o in observed_layout['objects']}
    fixed = [dict(id=o['id'], category=o['category'], description=o['description'],
                  center=by_id[o['id']]['center'], size=by_id[o['id']]['size']) for o in observed]
    return (
        'Design ONE plausible complete room around this partial photograph and its FIXED visible objects. '
        f'The image covers {hfov:g} horizontal degrees; other directions are unobserved, not empty. '
        'Infer suitable unseen contents from the room purpose, style and functions already visible. '
        'This is a plausible full-scene hypothesis, not discovery of the unique true hidden room. '
        'Plan the new objects AND their positions/sizes together so that the numeric layout matches '
        'your intended arrangement. Keep only the objects under FIXED OBSERVED BOXES unchanged; return NEW instances. '
        'Any earlier independent-room demonstrations illustrate the camera coordinate and completion format; '
        'their IDs, coordinates and contents are not this room. Plan for the CURRENT photograph and boxes. '
        'Use the supplied coarse observed boxes as scale and position context. '
        'All centers are camera-local METRES: origin at camera, x right, y image-up, z forward. '
        'The camera is not gravity-levelled. Negative z is behind the camera. '
        'size means FULL camera-axis AABB lengths [size_x,size_y,size_z], not half lengths, '
        'intrinsic object width/thickness or rotated-box dimensions. No rotations or shapes are needed. '
        'Do not assume every object stands on the floor. A picture intended above furniture must '
        'actually have a higher center_y and an appropriate nearby position. '
        'Prefer additions outside the photograph. An object may instead be inside the camera FOV '
        'but hidden by a credible occluder visible in the photo; explain that hypothesis in reason. '
        'Do not use occlusion to justify covering visible empty space. Occlusion is not verified by this system. '
        'Do not move an object far away merely to force it outside the FOV; choose a coherent placement '
        'or another suitable scene hypothesis. '
        'Return ONLY one JSON object with room_type, scene_hypothesis, and additions. '
        'Each addition needs id (unique, different from observed IDs), category (allowed name), '
        'center (3 finite numbers), size (3 positive numbers), description (short object description), '
        'reason (scene-specific purpose and placement justification), and unseen_reason '
        '(outside_fov or occluded). Only listed objects commit existence; prose adds no objects. '
        'Do not output a relation graph, candidate IDs, bottom heights or room bounds. '
        f'At most {capacity} new objects, not a target count. If none are suitable, explain why in scene_hypothesis. '
        'Allowed categories: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nFIXED OBSERVED BOXES: '+json.dumps(fixed, ensure_ascii=False))


def validate_boxes(answer):
    if not isinstance(answer, dict) or not isinstance(answer.get('additions'), list):
        raise ValueError('Scene box completion requires an additions list')
    for obj in answer['additions']:
        if not isinstance(obj, dict):
            raise ValueError('Each new scene box must be an object')
        if not finite_vector(obj.get('center'), 3):
            raise ValueError(f'New box {obj.get("id")!r} needs a finite camera-local center[3]')
        if not finite_vector(obj.get('size'), 3) or min(obj['size']) <= 0:
            raise ValueError(f'New box {obj.get("id")!r} needs positive full camera-AABB size[3]')
    return answer


def projection_feedback(answer, hfov, image_size):
    """Read-only failed offscreen claims; positive margin alone is not visibility."""
    th = math.tan(math.radians(hfov)/2)
    tv = th*image_size[1]/image_size[0]
    issues = []
    for obj in answer['additions']:
        if obj['unseen_reason'] != 'outside_fov':
            continue
        center, size = np.asarray(obj['center']), np.asarray(obj['size'])
        if frustum_margin(center, size, th, tv) <= 0:
            continue
        box = projected_box(center, size, th, tv)
        issues.append(dict(id=obj['id'], issue='outside_fov_claim_not_certified',
            center_azimuth_deg=math.degrees(math.atan2(center[0], center[2])),
            projected_bbox_xyxy_norm=box.tolist() if box is not None else None,
            whole_aabb_projects_inside_photo=bool(box is not None and min(box) >= 0 and max(box) <= 1),
            note='Projection excludes occlusion reasoning; this is not a depth or semantic certificate'))
    return issues


def revise_boxes(answer, base_prompt, ask, validate, hfov, image_size):
    """One controlled geometry revision, preserving the proposed inventory.

    Unlike the historical directional reviser, deletion or relabelling as
    occluded cannot turn an offscreen failure into a passing result.
    """
    before = projection_feedback(answer, hfov, image_size)
    report = dict(protocol=REVISION_VERSION, used=bool(before), before=before,
                  after=deepcopy(before), withdrawn=[], revision_error=None)
    if not before:
        return deepcopy(answer), report
    th = math.tan(math.radians(hfov)/2)
    tv = th*image_size[1]/image_size[0]
    question = (base_prompt + '\nCONTROLLED REVISION: preserve exactly the previous new object IDs, '
        'categories and unseen_reason labels; return the full additions list, once per ID. '
        'Revise centers, full sizes and placement descriptions together into ONE coherent room. '
        'Do not delete objects, change outside_fov into occluded, or claim success in text. '
        'Use simple coarse numbers (about 2 decimal places), not spurious measured precision. '
        'The original image is supplied again. Preserve all visible objects and visible empty space. '
        'Do not blindly flip z signs: a new location must agree with its scene-specific purpose. '
        'If a reason mentions an existing object, use its ID and actual supplied location. '
        'Sizes must describe plausible full objects: an upright door needs a horizontal span as well as height; '
        'both horizontal extents being tiny would describe a pole, not a door. '
        'Use existing furniture as rough scale/height context: a floor chair should not be at wall-art height. '
        'This is a semantic prior, not a measured floor or support constraint. '
        'Camera y is image-up; no gravity or floor is measured, so do not snap furniture to a guessed floor. '
        f'PINHOLE PROJECTION: for z>0, u=0.5+x/(2*z*{th:.6g}), '
        f'v=0.5-y/(2*z*{tv:.6g}); the photo is 0<=u,v<=1. '
        f'A center is horizontally in view when z>0 and abs(x)<={th:.6g}*z. '
        'Checking only a center is insufficient: all 8 corners are center +/- full_size/2. '
        'For example at HFOV=90, center=[2,0,4], size=[1,1,1] is IN the photo, not offscreen; '
        'center=[3,0,1], size=[1,1,1] is to the right OUTSIDE the photo. '
        'These examples illustrate camera geometry, not where to put your furniture. '
        '\nPREVIOUS PROPOSAL: '+json.dumps(answer, ensure_ascii=False)
        +'\nCOMPUTED PROJECTION FEEDBACK: '+json.dumps(before, ensure_ascii=False))
    try:
        candidate = validate_boxes(ask(question, 'scene_boxes_revision', 4096))
        validate(candidate)
        original = {o['id']: o for o in answer['additions']}
        revised = {o['id']: o for o in candidate['additions']}
        if set(original) != set(revised) or len(candidate['additions']) != len(original):
            raise ValueError('Box revision must preserve every new object ID exactly once')
        if any(revised[oid][key] != obj[key] for oid, obj in original.items()
               for key in ('category', 'unseen_reason')):
            raise ValueError('Box revision cannot change categories or unseen_reason labels')
        # Preserve slot order as well as identity for a controlled comparison.
        candidate = deepcopy(candidate)
        candidate['additions'] = [deepcopy(revised[o['id']]) for o in answer['additions']]
        report['after'] = projection_feedback(candidate, hfov, image_size)
        report['numeric_revision_changed'] = any(
            revised[oid][key] != obj[key] for oid, obj in original.items() for key in ('center', 'size'))
        return candidate, report
    except (ValueError, KeyError, TypeError) as exc:
        report['revision_error'] = str(exc)
        return deepcopy(answer), report


def complete(ask, observed, observed_layout, hfov, capacity, cached=None, validate=None,
             image_size=None):
    base_prompt = prompt(observed, observed_layout, hfov, capacity)
    answer = validate_boxes(deepcopy(cached) if cached is not None else
                            ask(base_prompt, 'scene_boxes', 4096))
    if validate is not None:
        validate(answer)
    feedback = dict(used=False, before=[], after=[], withdrawn=[], revision_error=None)
    if cached is not None:
        if validate is None or image_size is None:
            raise ValueError('Cached box revision needs semantic validation and image size')
        answer, feedback = revise_boxes(answer, base_prompt, ask, validate, hfov, image_size)
    # Shared scene_objects validates IDs, categories, descriptions and capacity.
    return deepcopy(answer), dict(protocol=PROMPT_VERSION, legacy_region_ids=[],
        position_source='joint_model_camera_xyz', floor_snapping=False, candidate_grid_used=False,
        feedback=feedback,
        occlusion_verified=False, note='Raw model boxes retained; projection checks do not correct geometry')
