"""Pure state conversion for instance observation and hidden-only proposals."""
from copy import deepcopy
import json

import numpy as np

from ..common import field, finite_vector, validate_graph
from .geometry import (CANONICAL_CATEGORIES, CATEGORY_IDS, SIGNS, graph, observer_categories,
                       qwen_corners, world_object)

VOCAB = ', '.join(CANONICAL_CATEGORIES.values())


def category_name(value):
    names, _, _, skipped = observer_categories({'categories': [value]})
    if skipped or len(names) != 1:
        raise ValueError(f'Unknown category: {value!r}; use the provided vocabulary')
    return names[0]


def instances_from_response(response):
    result = []
    for i, item in enumerate(response['instances']):
        box = item['bbox_2d']
        if not finite_vector(box, 4) or not (0 <= box[0] < box[2] <= 1000 and
                                             0 <= box[1] < box[3] <= 1000):
            raise ValueError(f'Instance {i}: bbox_2d must be xyxy in 0..1000')
        result.append(dict(id=f'obs_{i:04d}', category=category_name(item['category']),
                           bbox_2d=box, description=str(item.get('description', ''))))
    return result


def grounded_instances(category, response, offset=0, limit=32):
    """The query supplies category; model only localizes its visible instances."""
    boxes = response['boxes'] if isinstance(response, dict) else response
    if not isinstance(boxes, list) or len(boxes) > limit:
        raise ValueError(f'{category}: expected at most {limit} 2D boxes')
    instances, duplicates, seen = [], [], set()
    for i, item in enumerate(boxes):
        box = item.get('bbox_2d') if isinstance(item, dict) else item
        if not finite_vector(box, 4) or not (0 <= box[0] < box[2] <= 1000 and
                                             0 <= box[1] < box[3] <= 1000):
            raise ValueError(f'{category} box {i}: expected xyxy in 0..1000; no clipping performed')
        key = tuple(box)
        if key in seen:
            duplicates.append(i)
            continue
        seen.add(key)
        instances.append(dict(id=f'obs_{offset+len(instances):04d}', category=category_name(category),
                              bbox_2d=box, description=f'{category} in the specified image region'))
    return instances, dict(returned=len(boxes), exact_duplicates=duplicates, at_capacity=len(boxes) == limit)


def observed_object(instance, response):
    if isinstance(response, list):
        if len(response) != 1:
            raise ValueError('Expected one instance box; multiple returned boxes have ambiguous identity')
        response = response[0]
    bbox = response['bbox_3d']
    if not finite_vector(bbox, 9) or min(bbox[3:6]) <= 0:
        raise ValueError('bbox_3d needs nine finite numbers and positive full sizes')
    # Explicit prompt convention: normalized rotations around camera x, then y, then z.
    center, corners = qwen_corners(bbox, 'normalized')
    obj = world_object(instance['id'], instance['category'], center, corners,
                       'qwen_instance_observer', True)
    obj['bbox2d_visible_xyxy_norm'] = field([v/1000 for v in instance['bbox_2d']],
                                           'qwen_instance_observer', '2d_instance_grounding')
    return obj, dict(**instance, bbox_3d=bbox, center_local=center.tolist(),
                     corners_local=corners.tolist())


def observed_without_3d(instance, reason):
    return dict(track_id=instance['id'], local_evidence='visible',
                category_id=field(CATEGORY_IDS[instance['category']], 'qwen_instance_observer'),
                center_local_m=field(None, reason=reason), size_aabb_local_m=field(None, reason=reason),
                bbox2d_visible_xyxy_norm=field([v/1000 for v in instance['bbox_2d']], 'qwen_instance_observer'),
                in_primary_room=field(None, reason='not_estimated'))


def compact_observations(observed):
    """Whitelist: never send room GT, hidden objects, visibility counts or source IDs."""
    validate_graph(observed)
    if observed['scope'] != 'observed' or any(o['local_evidence'] != 'visible' for o in observed['objects']):
        raise ValueError('Only visible objects may enter the observation prompt')
    return [dict(id=f'obs_{i:04d}', category=CANONICAL_CATEGORIES[o['category_id']['value']],
                 center_local_m=o['center_local_m']['value'],
                 size_aabb_local_m=o['size_aabb_local_m']['value'],
                 bbox2d_visible_xyxy_norm=o['bbox2d_visible_xyxy_norm']['value'])
            for i, o in enumerate(observed['objects'])]


def hidden_prompt(observed, description, scene_text, capacity):
    context = compact_observations(observed)
    return (
        'Propose ONE plausible completion of this partially observed indoor scene. '
        'The image and OBSERVED objects are evidence; off-camera structure is a hypothesis. '
        'Return ONLY NEW hidden objects, never repeat, move or replace an observed object. '
        'Hidden means not seen in this image, including occluded objects; it does not mean '
        'all objects must be behind the camera. Do not fill visibly empty space with objects. '
        'Preserve plausible passageways. Do not invent distant rooms or populate a whole building. '
        'Estimate only objects relevant to a panorama at this same camera. '
        'Coordinates are camera-local METRES: origin at camera, x=image-right, y=image-up, '
        'z=forward (negative z is behind). These axes need not be gravity aligned. '
        'size_aabb_local_m is the FULL positive xyz edge lengths of a CAMERA-AXIS AABB, '
        'not half extents or an oriented box. Do not output rotation. '
        'A null observed coordinate/size means unknown; do not output that observed instance again as hidden. '
        'Use only categories: '+VOCAB+'. '
        f'Return 0 to {capacity} new objects; this is an upper limit, not a target count. '
        'An empty hidden_objects list is allowed. Output compact valid JSON, no markdown: '
        '{"hidden_objects":[{"category":"chair","center_local_m":[1,0,-2],'
        '"size_aabb_local_m":[0.6,0.9,0.6]}]}. '
        'The numeric example is a format example, not a placement instruction.\n'
        'OBSERVED='+json.dumps(context, ensure_ascii=False)+'\n'
        'VISIBLE_DESCRIPTION='+json.dumps(description, ensure_ascii=False)+'\n'
        'OPTIONAL_SCENE_TEXT='+json.dumps(scene_text, ensure_ascii=False))


def assemble(observed, response, max_objects, max_hidden):
    validate_graph(observed)
    if observed['scope'] != 'observed' or any(o['local_evidence'] != 'visible' for o in observed['objects']):
        raise ValueError('Completion requires an observed-only graph')
    items = response['hidden_objects']
    if not isinstance(items, list) or len(items) > min(max_hidden, max_objects-len(observed['objects'])):
        raise ValueError('Hidden proposal exceeds declared object budget; no truncation performed')
    new_objects, used = [], {o['track_id'] for o in observed['objects']}
    for i, item in enumerate(items):
        center, size = item['center_local_m'], item['size_aabb_local_m']
        if not finite_vector(center, 3) or not finite_vector(size, 3) or min(size) <= 0:
            raise ValueError(f'Invalid hidden geometry at index {i}')
        track = f'proposed_hidden_{i:04d}'
        while track in used:
            track = '_'+track
        used.add(track)
        corners = np.asarray(center)+SIGNS*np.asarray(size)/2
        obj = world_object(track, category_name(item['category']), center, corners,
                           'qwen_hidden_hypothesis', False)
        # It is intended as unobserved, but visibility is not measured: keep unknown.
        obj['hypothesized_hidden'] = True
        new_objects.append(obj)
    hidden = graph(observed['id'], observed['split'], 'hidden', new_objects,
                   observed['vocabulary_sha256'])
    full = graph(observed['id'], observed['split'], 'full',
                 deepcopy(observed['objects'])+deepcopy(new_objects), observed['vocabulary_sha256'])
    full['observed_count'] = len(observed['objects'])
    validate_graph(hidden)
    validate_graph(full)
    return hidden, full


