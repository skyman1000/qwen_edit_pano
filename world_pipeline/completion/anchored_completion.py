"""One scene hypothesis, expressed relative to fixed/explicit object centres.

This is a project implementation, not a reproduction of LayoutGPT, LayoutVLM
or Holodeck. The compiler only adds camera-frame offsets. It never invents a
floor, translates groups out of view, moves observations, or certifies quality.
"""
from copy import deepcopy
import json

import numpy as np

from ..common import finite_vector
from .category_bridge import category as adapter_category
from .geometry import CANONICAL_CATEGORIES
from .placement_program import overlap_report

PROTOCOL = 'fixed_observation_anchored_camera_offsets_v1'
CAMERA_ID = '__camera__'


def prompt(observed, layout, hfov, capacity):
    boxes = {o['id']: o for o in layout['objects']}
    fixed = [dict(id=o['id'], category=o['category'], description=o['description'],
                  center=boxes[o['id']]['center'], size=boxes[o['id']]['size']) for o in observed]
    return (
        'Plan ONE plausible complete panoramic scene extending this partial RGB photograph. '
        'Use its actual room purpose and visible furniture; unobserved directions are unknown, not empty. '
        'Treat all FIXED OBSERVED instances as evidence and keep them unchanged. '
        'Plan the missing contents and their arrangement together across the surrounding scene. '
        'Choose useful additions, not a standard reading/storage set for every photograph. '
        'Only objects listed in additions commit existence; prose must not introduce extra objects. '
        'No direction quotas: do not force left/right/back coverage, but do not default everything '
        'to one side without a scene-specific reason. No group is forced to one side of the photo. '
        'The final image is a 2:1 ERP encoding directions around the SAME camera; it is not a long flat room. '
        f'The input photograph covers {hfov:g} horizontal degrees. '
        'Coordinates and offsets are camera-local METRES: x right, y IMAGE-UP, z forward. '
        'Negative z is behind the camera. Gravity, floor, walls and room bounds are not measured. '
        'size=[sx,sy,sz] means positive FULL camera-axis AABB lengths, not half sizes or intrinsic OBB dimensions. '
        'Return JSON with room_type, scene_hypothesis, additions. Each addition has: '
        'id, category, size[3], anchor_id, offset_m[3], description, reason, unseen_reason. '
        'anchor_id names a FIXED OBSERVED object, another explicitly listed NEW object, '
        f'or "{CAMERA_ID}" (camera centre [0,0,0]). '
        'offset_m is the signed displacement from the ANCHOR CENTRE to the NEW BOX CENTRE '
        'along these same camera axes. The compiler calculates center=anchor_center+offset_m. '
        'For camera anchors offset_m is simply the new camera-space centre. '
        'References may occur in any list order, but must eventually lead to an observed object or camera. '
        'A positioning anchor is not a claim of physical support. '
        'Use full sizes to choose sensible centre spacing: equal centre coordinates do not mean beside. '
        'For touching boxes along x the centre separation is (sx_anchor+sx_new)/2; '
        'a side gap adds to that separation. Use the analogous calculation for other axes. '
        'Choose vertical offsets from the photograph and relevant existing objects; do not assume y=-1.5 is a floor. '
        'If describing an addition relative to an existing object, use that object ID as its anchor '
        'so the reference participates in the numeric placement. '
        'Prefer useful furniture outside the photograph; do not add large objects in visible empty space. '
        'unseen_reason is outside_fov, or occluded only for a credible visible occluder explained in reason. '
        'An occluded claim remains unverified. Do not use it as a workaround for an in-view object. '
        'Avoid unnecessarily distant placements, obvious interpenetration, duplicate visible objects and '
        'unsupported claims of measured walls, seat surfaces or object facing directions. '
        'Allowed category names are the existing Adapter vocabulary below. Preserve finer identity '
        '(e.g. wardrobe rather than generic cabinet) in description, without inventing category IDs. '
        f'At most {capacity} NEW objects; this is a budget, not the required count. '
        'No observed objects in additions, no independent center field, room-boundary array, '
        'executable code, or extra relation graph. Return ONLY JSON. '
        'Independent demonstrations, if supplied, illustrate this same contract, not this room. '
        '\nALLOWED CATEGORIES: '+', '.join(CANONICAL_CATEGORIES.values())
        +'\nFIXED OBSERVED: '+json.dumps(fixed, ensure_ascii=False, separators=(',', ':')))


def compile_layout(answer, observed, layout, capacity):
    if (not isinstance(answer, dict) or
            any(not isinstance(answer.get(k), str) or not answer[k].strip()
                for k in ('room_type', 'scene_hypothesis')) or
            not isinstance(answer.get('additions'), list)):
        raise ValueError('Anchored scene needs room_type, scene_hypothesis and additions')
    items = deepcopy(answer['additions'])
    if len(items) > capacity:
        raise ValueError('Addition budget exceeded; no silent truncation')
    ids = {o['id'] for o in observed}
    if CAMERA_ID in ids:
        raise ValueError('Observed ID conflicts with reserved camera anchor')
    fixed = {o['id']: o for o in layout['objects'] if o['id'] in ids}
    if set(fixed) != ids:
        raise ValueError('Every observed anchor needs fixed geometry')
    centres = {k: np.asarray(v['center'], dtype=float) for k, v in fixed.items()}
    centres[CAMERA_ID] = np.zeros(3)
    new_ids = set()
    for obj in items:
        if (not isinstance(obj, dict) or not isinstance(obj.get('id'), str) or not obj['id'].strip()
                or obj['id'] in ids | new_ids | {CAMERA_ID}):
            raise ValueError('Each addition needs a distinct new ID')
        if (not finite_vector(obj.get('size'), 3) or min(obj['size']) <= 0 or
                not finite_vector(obj.get('offset_m'), 3)):
            raise ValueError('Need finite centre offset and positive full camera-AABB size')
        if any(k in obj for k in ('center', 'region', 'along_m', 'clearance_m')):
            raise ValueError('Use one anchor/offset placement, not a second position or old region program')
        if (not isinstance(obj.get('anchor_id'), str) or
                any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in ('description', 'reason')) or
                obj.get('unseen_reason') not in ('outside_fov', 'occluded')):
            raise ValueError('Need anchor ID, description, reason and an explicit unseen_reason')
        canonical, bridge = adapter_category(obj.get('category'))
        if bridge is not None:
            obj.update(raw_category=obj['category'], category_mapping=bridge)
        obj['category'] = canonical
        new_ids.add(obj['id'])
    for obj in items:
        if obj['anchor_id'] not in ids | new_ids | {CAMERA_ID} or obj['anchor_id'] == obj['id']:
            raise ValueError(f"Unknown or self-referencing positioning anchor: {obj['id']} -> {obj['anchor_id']}")
    pending = list(items)
    resolved = []
    while pending:
        ready = [o for o in pending if o['anchor_id'] in centres]
        if not ready:
            raise ValueError('Positioning references form an unanchored cycle; no fixed camera/observation origin')
        for obj in ready:
            centre = centres[obj['anchor_id']] + np.asarray(obj['offset_m'], dtype=float)
            if not np.isfinite(centre).all():
                raise ValueError('Computed centre is not finite')
            centres[obj['id']] = centre
            resolved.append(dict(id=obj['id'], anchor_id=obj['anchor_id'], offset_m=obj['offset_m'],
                                 center=centre.tolist()))
            pending.remove(obj)
    for obj in items:
        obj['center'] = centres[obj['id']].tolist()
    return dict(room_type=answer['room_type'], scene_hypothesis=answer['scene_hypothesis'], additions=items), dict(
        protocol=PROTOCOL, legacy_region_ids=[], placements=resolved,
        position_source='explicit_anchor_centre_plus_camera_offset',
        observed_state_preserved=True, offscreen_translation=False, floor_snapping=False,
        optimizer_enabled=False, overlap_report=overlap_report(items, layout),
        note='Offsets determine coordinates only; projection, overlap, semantics and physical support are not certified')


def complete(ask, observed, layout, hfov, capacity):
    return compile_layout(ask(prompt(observed, layout, hfov, capacity), 'anchored_scene', 4096),
                          observed, layout, capacity)
