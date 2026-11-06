"""Frozen Qwen baseline: one RGB -> observed state -> full panorama hypothesis.

No scene GT, training, implicit downloads, or claims of geometric correctness.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from ..common import ROOT, field, finite_vector, new_output, read, sha, validate_graph, write
from .geometry import CANONICAL_CATEGORIES, CATEGORY_IDS, graph, observer_categories, qwen_corners
from .qwen_upstream import DEFAULT_MODEL, Qwen, camera_text

DEFAULT_IMAGE = ROOT/'qwen_edit_pano/data/qwen_upstream_val12_v1/zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a/local.png'
DEFAULT_VOCAB = ROOT/'qwen_edit_pano/data/gt_world_supplement_v1/vocabulary.json'
AXES = ('Origin is the camera; metres; x=image-right, y=image-up, z=forward. '
        'Behind the CAMERA is negative z, distinct from behind an object. '
        'center_local_m is a 3D centre, not pixel coordinates. '
        'size_aabb_local_m contains FULL xyz CAMERA-AXIS AABB edge lengths, not half extents. '
        'Camera axes need not be gravity aligned. Use null for unavailable geometry, never fake zeros. ')
VOCAB = ', '.join(CANONICAL_CATEGORIES.values())


def observation_prompt(camera, capacity=32):
    return ('Locate distinct visible physical objects, including partially visible ones. '
            'Return ONLY a compact JSON array with category, description (at most 6 words), '
            'and bbox_2d (tight visible xyxy box, normalized to 0..1000). '
            'No 3D, distances, sizes or long explanations in this answer. '
            'A bed and its headboard form one bed instance. Never repeat the same object/box. '
            'Exclude wall/floor/ceiling surfaces. Stop and close the array after the distinct objects. '
            f'At most {capacity} objects; if more exist, prioritize the larger ones. '
            'This is a ceiling, not a target count. Canonical categories: ' + VOCAB)


def completion_prompt(observed, camera, capacity):
    context = [dict(id=o['track_id'], category=o['raw_category'], description=o['description'],
                    bbox_2d=([v*1000 for v in o['bbox2d_visible_xyxy_norm']['value']]
                             if o['bbox2d_visible_xyxy_norm']['value'] is not None else None),
                    center_local_m=o['center_local_m']['value'], size_aabb_local_m=o['size_aabb_local_m']['value'])
               for o in observed['objects']]
    return ('Infer ONE plausible complete object state for a 360-degree panorama at the same camera. '
            'Return ONLY additional distinct objects, never move, replace or repeat an observed object. '
            'Unseen content is a hypothesis, not measured truth. Unknown regions may lie outside the image '
            'field of view or be occluded inside it. Preserve the photographed scene and visible empty space. '
            'If an object is already visible but missing from OBSERVED, label its role visible_missing; '
            'if it has no visible evidence and is hypothesized, label its role unseen. '
            'Use role=null if uncertain; do not claim visible evidence for unseen content. '
            'Return a compact JSON array with ONLY role, category, description (at most 6 words), '
            'bbox_2d (0..1000 for visible_missing, otherwise null), placement '
            '(a short relative relation to the CAMERA or an observed ID), and unseen_reason '
            '(outside_fov, occluded, uncertain, or null). '
            'Do not output numeric 3D coordinates or dimensions in this list. '
            f'At most {capacity} additions; this is a ceiling, not a requested count. '
            'Do not repeat items to fill the budget or populate distant rooms. An empty array is allowed. '
            'Canonical categories: ' + VOCAB + '\nCAMERA: ' + camera
            + '\nOBSERVED (bbox_2d uses 0..1000): '
            + json.dumps(context, ensure_ascii=False))


def request(model, image, prompt, path, budget, issues, validator):
    """One bounded retry; preserve both raw attempts and every failure."""
    for attempt in range(2):
        target = path if attempt == 0 else path.with_name(path.stem+'.retry.txt')
        instruction = prompt if attempt == 0 else prompt+' The previous attempt was malformed or repetitive. Return only the requested compact JSON and finish it; do not repeat entries.'
        try:
            answer = model.ask(image, instruction, target, seed=3407+attempt, sample=True,
                               profile='official', guard_repetition=True, max_tokens=budget)
            return validator(answer)
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(dict(request=target.name, error=str(exc), attempt=attempt))
    raise ValueError(f'Both bounded attempts failed: {path}; inspect raw answers and generation metadata')


def inventory(answer, stage, capacity):
    items = object_list(answer, 'objects' if stage == 'observed' else 'additions')
    keys = []
    for item in items:
        if isinstance(item, dict):
            box = item.get('bbox_2d')
            # Exact visible duplicates are a generation failure, not new instances.
            if finite_vector(box, 4):
                key = json.dumps([item.get('category'), box])
                if key in keys:
                    raise ValueError('Repeated visible category/box in inventory; no deduplication used to manufacture success')
                keys.append(key)
        if isinstance(item, dict) and any(item.get(k) is not None for k in ('center_local_m', 'size_aabb_local_m', 'bbox_3d')):
            raise ValueError('Inventory stage unexpectedly returned geometry; separate localization from 3D queries')
    return decode_objects(answer, stage, capacity)


def visible_geometry(model, image, camera, obj, output, budget, issues):
    box = obj['bbox2d_visible_xyxy_norm']['value']
    if box is None or obj['category_id']['value'] is None:
        issues.append(dict(id=obj['track_id'], error='visible_geometry_requires_category_and_2d_instance'))
        return
    marked = image.copy()
    w, h = marked.size
    ImageDraw.Draw(marked).rectangle([box[0]*w, box[1]*h, box[2]*w, box[3]*h], outline='red', width=3)
    prompt = ('Locate ONLY the '+CANONICAL_CATEGORIES[obj['category_id']['value']]
              +' indicated by the red rectangle. The rectangle is an annotation. Use the original whole-image camera. '
              'Return a JSON list containing ONE object with bbox_3d and label. '
              'bbox_3d order: center xyz in metres, FULL OBB side lengths xyz in metres, '
              'then rotations around x/y/z DIVIDED BY PI. Camera axes x right, y DOWN, z forward. '
              'The angles are normalized by pi, not radians or degrees. Return null bbox_3d if unavailable. '
              'No list of other objects or prose. '+camera)

    def validate(answer):
        if isinstance(answer, list):
            if len(answer) != 1:
                raise ValueError('3D grounding requires exactly one identified target')
            answer = answer[0]
        if not isinstance(answer, dict) or 'bbox_3d' not in answer:
            raise ValueError('Expected bbox_3d for the single target')
        box3d = answer['bbox_3d']
        if box3d is None:
            return None
        if not finite_vector(box3d, 9) or min(box3d[3:6]) <= 0:
            raise ValueError('Invalid native 9-number box')
        return box3d

    try:
        bbox = request(model, marked, prompt, output/(obj['track_id']+'.txt'), budget, issues, validate)
        if bbox is None:
            issues.append(dict(id=obj['track_id'], error='model_reported_unknown_geometry'))
            return
        center, corners = qwen_corners(bbox, 'normalized')
        obj['native_bbox_3d'] = bbox
        obj['native_angle_unit'] = 'pi_normalized_explicit_prompt'
        obj['center_local_m'] = field(center.tolist(), 'qwen_single_image_prediction', 'native_y_down_to_project_y_up')
        obj['size_aabb_local_m'] = field((corners.max(0)-corners.min(0)).tolist(), 'qwen_single_image_prediction', 'native_obb_corners_to_full_aabb')
        if center[2] <= 0:
            issues.append(dict(id=obj['track_id'], error='visible_center_not_in_front_review_required'))
    except ValueError as exc:
        issues.append(dict(id=obj['track_id'], error=str(exc), geometry='unknown'))


def native_target(obj):
    """Use the cookbook's referring-expression route on the unmodified image."""
    box = obj['bbox2d_visible_xyxy_norm']['value']
    if box is None or obj['category_id']['value'] is None:
        raise ValueError('Native target needs a category and a visible 2D location')
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    horizontal = 'left' if x < 1/3 else 'right' if x > 2/3 else 'middle'
    vertical = 'upper' if y < 1/3 else 'lower' if y > 2/3 else 'middle'
    return dict(category=CANONICAL_CATEGORIES[obj['category_id']['value']],
                description=obj.get('description') or '',
                image_region=f'{vertical} {horizontal}', bbox_2d=box)


def native_visible_geometry(model, image, obj, output, budget, issues):
    try:
        target = native_target(obj)
    except ValueError as exc:
        issues.append(dict(id=obj['track_id'], error=str(exc), geometry='unknown'))
        return
    # Do not add a red rectangle, crop the image, prescribe a new angle format,
    # or assert that raw model coordinates already use the project's camera.
    prompt = (f'Find the {target["category"]} described as "{target["description"]}" '
              f'in the {target["image_region"]} part of this image and estimate its 3D box. '
              'Return a JSON array for this single target, using keys label and bbox_3d. '
              'Use the native 3D grounding format: '
              '[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].')

    def validate(answer):
        items = [answer] if isinstance(answer, dict) else answer
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise ValueError('Expected one native object; returned instances cannot be assigned by order')
        bbox = items[0].get('bbox_3d')
        if not finite_vector(bbox, 9) or min(bbox[3:6]) <= 0:
            raise ValueError('Expected nine finite native values and positive full dimensions')
        if not isinstance(items[0].get('label'), str) or not items[0]['label'].strip():
            raise ValueError('Native target response must retain its label for identity review')
        return items[0]

    obj['native_grounding_target'] = target
    try:
        answer = request(model, image, prompt, output/(obj['track_id']+'.txt'), budget, issues, validate)
        obj['native_bbox_3d'] = answer['bbox_3d']
        obj['native_label'] = answer['label']
        obj['native_coordinate_contract'] = 'qwen_native_camera_not_verified_against_project_camera'
        obj['native_angle_convention'] = 'raw_roll_pitch_yaw; cookbook_visualizer_interprets_values_times_pi'
    except ValueError as exc:
        issues.append(dict(id=obj['track_id'], error=str(exc), geometry='unknown'))
    # A numeric native box is evidence to inspect, not a calibrated project state.
    for key in ('center_local_m', 'size_aabb_local_m'):
        obj[key] = field(None, reason='native_to_project_camera_mapping_unverified')
    obj['geometry_quality'] = 'native_prediction_unverified'


def native_reprojection(obj, image_size, hfov):
    """Conditional diagnostic: assume native camera equals real camera.

    Does not estimate intrinsics, align to GT, certify scale, or promote geometry.
    """
    bbox = obj.get('native_bbox_3d')
    if bbox is None:
        return dict(id=obj['track_id'], status='no_native_box')
    w, h = image_size
    focal = w/(2*math.tan(math.radians(hfov)/2))
    center, corners = qwen_corners(bbox, 'normalized')
    result = dict(id=obj['track_id'], status='diagnostic_only',
                  camera_mapping_assumption='native_camera_equals_real_camera_unverified',
                  angle_assumption='cookbook_visualizer_values_times_pi',
                  target_box=obj['bbox2d_visible_xyxy_norm']['value'])
    def project(point):
        return [.5+focal/w*float(point[0]/point[2]), .5-focal/h*float(point[1]/point[2])]
    if center[2] > 0:
        result['projected_center'] = project(center)
    if min(corners[:,2]) <= 0:
        result['box_projection_status'] = 'crosses_or_behind_camera_plane'
        return result
    uv = [project(corner) for corner in corners]
    projected = [min(p[0] for p in uv), min(p[1] for p in uv),
                 max(p[0] for p in uv), max(p[1] for p in uv)]
    result['projected_box'] = projected
    target = result['target_box']
    intersection = max(0, min(projected[2],target[2])-max(projected[0],target[0])) * max(0, min(projected[3],target[3])-max(projected[1],target[1]))
    area = (projected[2]-projected[0])*(projected[3]-projected[1])+(target[2]-target[0])*(target[3]-target[1])-intersection
    result['projected_box_vs_visible_box_iou'] = intersection/area if area else 0.
    return result


def reuse_visible_inventory(path, image):
    original = validate_graph(read(path))
    if original['stage'] != 'prediction' or original['scope'] != 'observed':
        raise ValueError('Reuse requires predicted visible inventory, not a GT scene graph')
    with Image.open(path.parent/'local.png') as source:
        saved_image = source.convert('RGB')
    if saved_image.size != image.size or saved_image.tobytes() != image.tobytes():
        raise ValueError('Saved inventory image differs from this input; cannot reuse its instance identities')
    raw = []
    for obj in original['objects']:
        if obj['local_evidence'] != 'visible':
            raise ValueError('Only visible instances may enter observer verification')
        raw.append(dict(category=CANONICAL_CATEGORIES.get(obj['category_id']['value']),
                        description=obj.get('description'),
                        bbox_2d=([v*1000 for v in obj['bbox2d_visible_xyxy_norm']['value']]
                                 if obj['bbox2d_visible_xyxy_norm']['value'] is not None else None)))
    objects, issues = inventory(raw, 'observed', 160)
    for new, old in zip(objects, original['objects']):
        new['track_id'] = old['track_id']
    return objects, issues


def run_observer(model, image, entry, output, vocabulary_hash, max_tokens, max_observed,
                 reused=None, target_ids=None):
    """Observe only. Keep native predictions separate from calibrated local state."""
    issues = []
    geometry_dir = output/'geometry'
    geometry_dir.mkdir()
    try:
        if reused is None:
            objects, parsed_issues = request(model, image, observation_prompt('', max_observed),
                output/'observation.txt', min(max_tokens,2048), issues,
                lambda answer: inventory(answer, 'observed', max_observed))
        else:
            objects, parsed_issues = reused
            write(output/'reused_inventory.json', [dict(id=o['track_id'], category=o['raw_category'],
                  description=o.get('description'), bbox_2d=o['bbox2d_visible_xyxy_norm']['value']) for o in objects])
        issues.extend(parsed_issues)
        objects = deepcopy(objects)
        for obj in objects:
            # Never carry previous metric guesses into this observer-only result.
            for key in ('center_local_m', 'size_aabb_local_m'):
                obj[key] = field(None, reason='native_to_project_camera_mapping_unverified')
            obj.pop('native_bbox_3d', None)
            obj['geometry_quality'] = 'not_queried'
        selected = set(target_ids) if target_ids else {obj['track_id'] for obj in objects}
        if not selected <= {obj['track_id'] for obj in objects}:
            raise ValueError('Requested target IDs are absent from visible inventory')
        for obj in objects:
            if obj['track_id'] in selected:
                native_visible_geometry(model, image, obj, geometry_dir, min(max_tokens,512), issues)
                print(f'Native observer {obj["track_id"]}: box_returned={"native_bbox_3d" in obj}; project_geometry_verified=False', flush=True)
        state = graph('single_image', 'val', 'observed', objects, vocabulary_hash)
        state.update(quality_status='not_validated', camera_mapping_verified=False,
                     observation_verified=False, geometry_protocol='native_referring_expression',
                     queried_target_ids=sorted(selected))
        write(output/'G_obs.json', validate_graph(state))
        diagnostics = [native_reprojection(obj, image.size, entry['hfov_deg']) for obj in objects if obj['track_id'] in selected]
        write(output/'reprojection_diagnostic.json', dict(objects=diagnostics,
              geometry_accuracy_verified=False, camera_mapping_verified=False,
              note='Projection assumes raw native camera equals real camera; not a calibrated accuracy score. '
                   'Even matching image projections cannot verify metric depth or scale.'))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for obj in objects:
            if obj['track_id'] not in selected:
                continue
            box = obj['bbox2d_visible_xyxy_norm']['value']
            if box is not None:
                w,h = image.size
                draw.rectangle([box[0]*w,box[1]*h,box[2]*w,box[3]*h], outline='red', width=2)
                draw.text((box[0]*w,box[1]*h), obj['track_id'], fill='red')
        for diagnostic in diagnostics:
            if 'projected_box' in diagnostic:
                box = diagnostic['projected_box']
                draw.rectangle([box[0]*image.width,box[1]*image.height,box[2]*image.width,box[3]*image.height], outline='lime', width=2)
        overlay.save(output/'observer_reprojection.png')
        summary = dict(status='partial' if issues else 'completed', observed=len(objects),
                       queried_targets=len(selected), native_numeric_boxes=sum('native_bbox_3d' in o for o in objects),
                       project_numeric_boxes=0, observation_verified=False, camera_mapping_verified=False,
                       completion_executed=False, quality_status='not_validated')
        write(output/'issues.json', issues)
        write(output/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    except (ValueError, KeyError, TypeError) as exc:
        write(output/'issues.json', issues)
        write(output/'summary.json', dict(status='failed', stage='observation', error=str(exc), observation_verified=False))
        raise


def unseen_geometry(model, image, camera, obj, context, output, budget, issues):
    prompt = ('Place ONLY this one hypothetical unseen object in a plausible panorama layout. '
              'Return ONE JSON object with center_local_m (3 numbers or null) and '
              'size_aabb_local_m (3 positive full edge lengths or null). No object list or description. '
              + AXES+'Respect the proposed relation; do not copy another box. '
              'These values are a hypothesis.\nCAMERA: '+camera+'\nTARGET: '
              +json.dumps({k: obj.get(k) for k in ('raw_category', 'description', 'placement', 'unseen_reason')})
              +'\nEXISTING STATE: '+json.dumps([dict(id=o['track_id'], category=o['raw_category'],
                  center=o['center_local_m']['value'], size=o['size_aabb_local_m']['value']) for o in context]))
    def validate(answer):
        if isinstance(answer, list):
            if len(answer) != 1:
                raise ValueError('Expected geometry for exactly one unseen object')
            answer = answer[0]
        if not isinstance(answer, dict):
            raise ValueError('One unseen object requires one geometry dictionary')
        for key in ('center_local_m', 'size_aabb_local_m'):
            if key not in answer:
                raise ValueError('Missing geometry key '+key)
            value = answer[key]
            if value is not None and (not finite_vector(value, 3) or (key.startswith('size') and min(value) <= 0)):
                raise ValueError('Invalid proposed '+key)
        return answer
    try:
        answer = request(model, image, prompt, output/(obj['track_id']+'.txt'), budget, issues, validate)
        for key in ('center_local_m', 'size_aabb_local_m'):
            obj[key] = field(answer[key], 'qwen_single_image_prediction', 'unverified_single_object_hypothesis',
                             reason='model_reported_unknown_geometry')
    except ValueError as exc:
        issues.append(dict(id=obj['track_id'], error=str(exc), geometry='unknown'))


def object_list(answer, key):
    if isinstance(answer, list):
        return answer
    if isinstance(answer, dict) and isinstance(answer.get(key), list):
        return answer[key]
    raise ValueError(f'Expected a JSON object array or an object with a {key} array; got {type(answer).__name__}')


def decode_objects(answer, stage, limit):
    """Retain unknown fields and report malformed entries; do not fabricate geometry."""
    items = object_list(answer, 'objects' if stage == 'observed' else 'additions')
    if len(items) > limit:
        raise ValueError(f'{len(items)} objects exceed capacity {limit}; no truncation performed')
    objects, issues = [], []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(dict(index=i, field='object', error='not_an_object', raw=item))
            continue
        if not any(key in item for key in ('category', 'description', 'bbox_2d')):
            issues.append(dict(index=i, field='object', error='no_object_content', raw=item))
            continue
        track = f'{stage}_{i:04d}'
        raw_category = item.get('category')
        names, _, _, skipped = observer_categories({'categories': [raw_category]})
        cid = CATEGORY_IDS[names[0]] if names and not skipped else None
        if cid is None:
            issues.append(dict(index=i, field='category', error='unknown_or_unsupported_category', raw=raw_category))
        values = {}
        for key in ('center_local_m', 'size_aabb_local_m'):
            value = item.get(key)
            if value is not None and (not finite_vector(value, 3) or
                                     (key == 'size_aabb_local_m' and min(value) <= 0)):
                issues.append(dict(index=i, field=key, error='invalid_geometry_retained_as_unknown', raw=value))
                value = None
            values[key] = field(value, 'qwen_single_image_prediction', 'unverified_metric_estimate',
                                reason='unknown_or_invalid_model_geometry')
        box = item.get('bbox_2d')
        if box is not None:
            if finite_vector(box, 4) and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000:
                box = [v/1000 for v in box]
            else:
                issues.append(dict(index=i, field='bbox_2d', error='invalid_2d_box_retained_as_unknown', raw=box))
                box = None
        role = 'observed' if stage == 'observed' else item.get('role')
        if stage != 'observed' and role not in ('visible_missing', 'unseen'):
            issues.append(dict(index=i, field='role', error='unresolved_visibility_role', raw=role))
            role = 'unresolved'
        if role == 'unseen' and box is not None:
            issues.append(dict(index=i, field='bbox_2d', error='unseen_role_has_visible_box', raw=item.get('bbox_2d')))
        obj = dict(track_id=track, raw_category=raw_category, description=item.get('description'),
                   category_id=field(cid, 'qwen_single_image_prediction', 'existing_category_bridge',
                                     reason='unknown_or_unsupported_category'),
                   bbox2d_visible_xyxy_norm=field(box, 'qwen_single_image_prediction', 'visible_grounding',
                                                 reason='not_reported_or_invalid'),
                   in_primary_room=field(None, reason='not_estimated'),
                   local_evidence='visible' if role in ('observed', 'visible_missing') else 'unknown',
                   completion_role=role, placement=item.get('placement'), unseen_reason=item.get('unseen_reason'),
                   geometry_quality='unverified_prediction', **values)
        if role in ('observed', 'visible_missing') and values['center_local_m']['value'] is not None and values['center_local_m']['value'][2] <= 0:
            issues.append(dict(index=i, field='center_local_m', error='visible_center_not_in_front_review_required'))
        objects.append(obj)
    if items and not objects:
        raise ValueError('No structured object entries; raw answer preserved, no empty state substituted')
    return objects, issues


def run(model, image, entry, output, vocabulary_hash, max_hidden, max_tokens, max_observed=32):
    camera = camera_text(image, entry) + f'Horizontal FOV={entry["hfov_deg"]} degrees. '
    phase = 'observation'
    obs_issues, comp_issues = [], []
    geometry_dir = output/'geometry'
    geometry_dir.mkdir(exist_ok=True)
    try:
        observed, parsed_issues = request(model, image, observation_prompt(camera, max_observed),
            output/'observation.txt', min(max_tokens, 2048), obs_issues,
            lambda answer: inventory(answer, 'observed', max_observed))
        obs_issues.extend(parsed_issues)
        print(f'Observed inventory: {len(observed)} instances; querying one 3D box per instance', flush=True)
        for obj in observed:
            visible_geometry(model, image, camera, obj, geometry_dir, min(max_tokens, 512), obs_issues)
            print(f'Observed geometry {obj["track_id"]}: valid={obj["center_local_m"]["valid"]}', flush=True)
        obs = graph('single_image', 'val', 'observed', observed, vocabulary_hash)
        validate_graph(obs)
        write(output/'G_obs.json', obs)
        phase = 'completion'
        capacity = min(max_hidden, 160-len(observed))
        if capacity:
            additions, parsed_issues = request(model, image, completion_prompt(obs, camera, capacity),
                output/'completion.txt', min(max_tokens, 2048), comp_issues,
                lambda answer: inventory(answer, 'added', capacity))
            comp_issues.extend(parsed_issues)
        else:
            additions = []
        print(f'Completion inventory: {len(additions)} additions; querying geometry separately', flush=True)
        context = list(observed)
        for obj in additions:
            if obj['completion_role'] == 'visible_missing':
                visible_geometry(model, image, camera, obj, geometry_dir, min(max_tokens, 512), comp_issues)
            elif obj['completion_role'] == 'unseen':
                unseen_geometry(model, image, camera, obj, context, geometry_dir, min(max_tokens, 384), comp_issues)
            context.append(obj)
        hidden = [o for o in additions if o['completion_role'] == 'unseen']
        recovered = [o for o in additions if o['completion_role'] == 'visible_missing']
        unresolved = [o for o in additions if o['completion_role'] == 'unresolved']
        full = graph('single_image', 'val', 'full', deepcopy(observed)+deepcopy(additions), vocabulary_hash)
        full['quality_status'] = 'not_validated'
        full['observed_count'] = len(observed)
        for name, scope, values in [('G_hidden', 'hidden', hidden), ('G_visible_added', 'observed', recovered),
                                    ('G_unresolved_added', 'hidden', unresolved)]:
            state = graph('single_image', 'val', scope, values, vocabulary_hash)
            write(output/f'{name}.json', validate_graph(state))
        write(output/'G_full.json', validate_graph(full))
        issues = dict(observation=obs_issues, completion=comp_issues)
        write(output/'issues.json', issues)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for obj in observed:
            box = obj['bbox2d_visible_xyxy_norm']['value']
            if box is not None:
                w, h = image.size
                xy = [box[0]*w, box[1]*h, box[2]*w, box[3]*h]
                draw.rectangle(xy, outline='red', width=2)
                draw.text((xy[0], xy[1]), obj['track_id']+' '+str(obj['category_id']['value']), fill='red')
        overlay.save(output/'observed_overlay.png')
        summary = dict(status='partial' if obs_issues or comp_issues else 'completed',
                       observed=len(observed), unseen_proposals=len(hidden), visible_recoveries=len(recovered),
                       unresolved_additions=len(unresolved), full_objects=len(full['objects']),
                       original_observed_preserved=full['objects'][:len(observed)] == observed,
                       complete_numeric_geometry=sum(o['center_local_m']['valid'] and o['size_aabb_local_m']['valid'] for o in full['objects']),
                       observation_at_capacity=len(observed) == max_observed,
                       completion_at_capacity=len(additions) == capacity,
                       quality_status='not_validated',
                       note='valid flags mean numeric availability, not geometric accuracy; no scene GT used')
        write(output/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    except (ValueError, KeyError, TypeError) as exc:
        write(output/'issues.json', dict(observation=obs_issues, completion=comp_issues))
        write(output/'summary.json', dict(status='failed', stage=phase, error=str(exc), raw_responses_preserved=True))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, default=DEFAULT_IMAGE)
    p.add_argument('--hfov', type=float, required=True, help='Known/estimated horizontal FOV; recorded, not inferred here')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--output', type=Path)
    p.add_argument('--vocabulary', type=Path, default=DEFAULT_VOCAB)
    p.add_argument('--max-hidden', type=int, default=12)
    p.add_argument('--max-observed', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--observe-only', action='store_true', help='Native observer verification; never run completion')
    p.add_argument('--reuse-observation', type=Path, help='Reuse predicted 2D inventory only, from a saved G_obs.json')
    p.add_argument('--observed-ids', nargs='+', help='Query a specified subset of the reused visible inventory')
    a = p.parse_args()
    if not math.isfinite(a.hfov) or not 0 < a.hfov < 180 or not 0 <= a.max_hidden <= 160 or not 1 <= a.max_observed <= 160 or a.max_new_tokens < 1:
        p.error('Invalid FOV or object/token budget')
    if not a.preflight and a.output is None:
        p.error('--output is required')
    if not a.model.is_dir() or not list(a.model.glob('*.safetensors')):
        p.error('Existing local model weights not found; no download will be attempted')
    vocabulary = read(a.vocabulary)
    if {int(k): v for k, v in vocabulary['categories'].items()} != CANONICAL_CATEGORIES:
        raise ValueError('Vocabulary differs from the existing state category mapping')
    with Image.open(a.image) as source:
        image = source.convert('RGB')
    image.thumbnail((768, 768))
    if (a.reuse_observation or a.observed_ids) and not a.observe_only:
        p.error('Inventory reuse/target selection is for --observe-only')
    if a.observed_ids and not a.reuse_observation:
        p.error('--observed-ids requires --reuse-observation so instance IDs are fixed before GPU loading')
    reused = reuse_visible_inventory(a.reuse_observation, image) if a.reuse_observation else None
    if a.observed_ids and not set(a.observed_ids) <= {o['track_id'] for o in reused[0]}:
        p.error('Unknown --observed-ids in reused visible inventory')
    if a.output is not None and a.output.exists():
        p.error('Output already exists: use a new --output, never overwrite a previous run')
    if a.preflight:
        print(f'PREFLIGHT PASSED: {a.image.resolve()}, {image.size}, hfov={a.hfov}; local model/vocabulary exist; no model loaded')
        return
    output = new_output(a.output)
    image.save(output/'local.png')
    write(output/'config.json', dict(protocol='single_image_native_observer_v1' if a.observe_only else 'single_image_qwen_world_v2_bounded', input_image=str(a.image.resolve()),
          model=str(a.model.resolve()), hfov_deg=a.hfov, model_image_size=list(image.size),
          vocabulary=str(a.vocabulary.resolve()), category_mapping_sha256=vocabulary['category_mapping_sha256'],
          max_hidden=a.max_hidden, max_observed=a.max_observed, max_new_tokens=a.max_new_tokens,
          decoding=dict(profile='official', do_sample=True, temperature=.7, top_p=.8, top_k=20,
                        repetition_penalty=1., presence_penalty=1.5, seed=3407, retry_seed=3408),
          token_budgets=dict(inventory=min(a.max_new_tokens,2048), visible_box=min(a.max_new_tokens,512),
                             unseen_box=min(a.max_new_tokens,384)),
          native_3d_angle_unit='raw_native_not_reparameterized' if a.observe_only else 'pi_normalized_explicit_prompt', scene_gt_used=False,
          observe_only=a.observe_only, observed_ids=a.observed_ids,
          reused_inventory=str(a.reuse_observation.resolve()) if a.reuse_observation else None,
          reused_inventory_sha256=sha(a.reuse_observation) if a.reuse_observation else None))
    if a.observe_only:
        run_observer(Qwen(a.model), image, dict(hfov_deg=a.hfov), output,
                     vocabulary['category_mapping_sha256'], a.max_new_tokens, a.max_observed,
                     reused, a.observed_ids)
        return
    run(Qwen(a.model), image, dict(hfov_deg=a.hfov), output,
        vocabulary['category_mapping_sha256'], a.max_hidden, a.max_new_tokens, a.max_observed)


if __name__ == '__main__':
    main()
