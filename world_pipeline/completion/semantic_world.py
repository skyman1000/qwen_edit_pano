"""One image -> visible inventory -> fixed coarse observed state -> additions.

Three image requests, at most one additional completion revision, one model load. Read-only geometry checks; no optimizer.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from ..common import field, finite_vector, new_output, read, sha, validate_graph, write
from .box_layout import BoxProblem, RELATIONS, camera_layout, directional_layout, inspect_layout, projected_box, validate_layout
from .geometry import CANONICAL_CATEGORIES, CATEGORY_IDS, graph
from .completion_feedback import revise_once
from .planned_completion import CONTENT_PROMPT_VERSION, complete as complete_planned, validate_contents
from .box_completion import PROMPT_VERSION as BOX_PROMPT_VERSION, complete as complete_boxes, validate_boxes
from .layout_examples import load as load_layout_examples, messages as layout_example_messages
from .placement_program import PROTOCOL as PROGRAM_PROTOCOL, complete as complete_program
from .anchored_completion import PROTOCOL as ANCHORED_PROTOCOL, complete as complete_anchored
from .category_bridge import category as adapter_category
from .scene_geometry import camera_geometry, observation_report
from .qwen_upstream import DEFAULT_MODEL, Qwen
from .single_image_world import DEFAULT_IMAGE, DEFAULT_VOCAB, observation_prompt

VOCAB = ', '.join(CANONICAL_CATEGORIES.values())
PARAMETERIZATION = dict(direct='camera_azimuth_range_bottom_height', planned='content_then_ranked_candidates',
                        boxes='joint_camera_xyz_full_aabb', program='camera_region_group_observed_y_anchor',
                        anchored='explicit_anchor_plus_camera_xyz_offset')
AXES = ('Camera origin [0,0,0], x right, y up, z forward, all lengths METRES. '
        'Here y is IMAGE-UP, not necessarily gravity-up; the camera has not been levelled. '
        'Negative z means behind CAMERA, not behind a piece of furniture. '
        'Use FULL xyz CAMERA-AXIS AABB sizes; no half sizes, rotations or normalized 3D units. ')


class RelationValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('Invalid scene relations:\n'+'\n'.join(issues))


def visible_inventory(answer, capacity):
    items = answer if isinstance(answer, list) else answer.get('objects') if isinstance(answer, dict) else None
    if not isinstance(items, list) or not 1 <= len(items) <= capacity:
        raise ValueError('Expected a nonempty visible object list within capacity')
    objects, seen = [], set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError('Visible object must be a JSON object')
        category, bridge = adapter_category(item.get('category'))
        box = item.get('bbox_2d')
        if not finite_vector(box, 4) or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ValueError('Visible bbox_2d must be xyxy in 0..1000')
        key = (category, *box)
        if key in seen:
            raise ValueError('Repeated visible category and box')
        seen.add(key)
        description = item.get('description')
        if description is not None and not isinstance(description,str):
            raise ValueError('Visible description must be text when supplied')
        fallback = not description or not description.strip()
        obj = dict(id=f'observed_{i:04d}', category=category,
                   description=item['category'] if fallback else description,
                   bbox_2d=[v/1000 for v in box], role='observed')
        if bridge is not None:
            obj.update(raw_category=item['category'], category_mapping=bridge)
        if fallback:
            obj['description_source']='raw_category_fallback_not_visual_inference'
        objects.append(obj)
    return objects


def reuse_inventory(path, image, vocabulary_hash, capacity):
    source = validate_graph(read(path))
    if source.get('stage') != 'prediction' or source['scope'] != 'observed' or source.get('vocabulary_sha256') != vocabulary_hash:
        raise ValueError('Reuse requires a predicted G_obs with matching vocabulary')
    # Bind reused 2D evidence to this exact RGB, not an arbitrary JSON graph.
    with Image.open(path.parent/'local.png') as handle:
        old = handle.convert('RGB')
    old.thumbnail((768, 768))
    if old.size != image.size or old.tobytes() != image.tobytes():
        raise ValueError('Reused observation belongs to a different image')
    items = [dict(category=CANONICAL_CATEGORIES.get(o['category_id']['value']),
                  description=o.get('description'),
                  bbox_2d=[v*1000 for v in o['bbox2d_visible_xyxy_norm']['value']]
                  if o['bbox2d_visible_xyxy_norm']['value'] is not None else None)
             for o in source['objects']]
    objects = visible_inventory(items, capacity)
    for obj, old_obj in zip(objects, source['objects']):
        obj['id'] = old_obj['track_id']
    return objects


def observed_state_layout(state, observed):
    """Read already predicted numeric boxes without re-estimating their centers."""
    state = validate_graph(state)
    if state['scope'] != 'observed' or [o['track_id'] for o in state['objects']] != [o['id'] for o in observed]:
        raise ValueError('Fixed observed state IDs/scope differ from visible inventory')
    boxes = []
    for obj in state['objects']:
        center, size = obj['center_local_m'], obj['size_aabb_local_m']
        if not center['valid'] or not size['valid']:
            raise ValueError('Numeric G_obs required for --reuse-state; geometry is unknown: '+obj['track_id'])
        boxes.append(dict(id=obj['track_id'], center=deepcopy(center['value']),
                          size=deepcopy(size['value']), support=None))
    return validate_layout(dict(room=deepcopy(state.get('room_hypothesis')), objects=boxes), observed)


def plan_prompt(observed, max_added, hfov):
    return (
        'Plan ONE plausible complete current room around the camera from this partial photograph. '
        'This is scene imagination consistent with evidence, not a search only for missed visible objects. '
        'Preserve all observed instances below, their counts, image locations and visible empty space. '
        'Propose distinct UNSEEN furniture or useful objects in the same room, outside the photo or occluded. '
        'Do not repeat observed objects or list distant rooms. Give one restrained coherent arrangement, '
        'not a union of possible rooms. An empty additions list is allowed but not required. '
        f'At most {max_added} additions, a ceiling rather than target. '
        'Return ONLY JSON with room_type (short string) and additions (list). '
        'Each addition has id (unique string), category (allowed name), description (short), '
        'placement (one short phrase relative to camera or an observed ID), '
        'unseen_reason (outside_fov or occluded), reason (brief scene-specific justification). '
        'Do not output a relations array or enumerate observed-object pairs. '
        'The next stage will place all objects together. No numeric 3D geometry yet. '
        f'Camera horizontal FOV={hfov} degrees. Allowed categories: '+VOCAB
        +'\nObserved evidence (bbox_2d xyxy in 0..1): '+json.dumps(observed, ensure_ascii=False))


def scene_objects(answer, observed, capacity):
    """Validate object hypotheses without requiring a generated relation graph."""
    if not isinstance(answer, dict) or not isinstance(answer.get('room_type'), str) or not answer['room_type'].strip():
        raise ValueError('Plan needs a room_type string')
    additions = answer.get('additions')
    if not isinstance(additions, list) or len(additions) > capacity:
        raise ValueError('Invalid additions list or capacity exceeded')
    objects = deepcopy(observed)
    seen = {o['id'] for o in objects}
    for obj in additions:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str) or not obj['id'].strip() or obj['id'] in seen:
            raise ValueError('Addition needs a distinct ID')
        if obj.get('category') not in CANONICAL_CATEGORIES.values():
            raise ValueError('Addition category outside supplied vocabulary')
        if obj.get('unseen_reason') not in ('outside_fov', 'occluded'):
            raise ValueError('Addition needs outside_fov or occluded')
        if any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in ('description', 'reason')):
            raise ValueError('Addition requires short description and scene-specific reason')
        seen.add(obj['id'])
        objects.append(dict(id=obj['id'], category=obj['category'], description=obj['description'],
                            role='hypothesized', unseen_reason=obj['unseen_reason'], reason=obj['reason'],
                            **{key: obj[key] for key in ('azimuth_deg', 'camera_region', 'distance_m', 'bottom_height_m') if key in obj},
                            **{key: deepcopy(obj[key]) for key in ('anchor_id','offset_m','raw_category','category_mapping') if key in obj},
                            **({'placement': obj['placement']} if isinstance(obj.get('placement'), str) else {})))
    return dict(room_type=answer['room_type'], objects=objects)


def decode_plan(answer, observed, capacity):
    """Legacy relation-graph validator, retained to audit the failed experiments."""
    plan = scene_objects(answer, observed, capacity)
    objects = plan['objects']
    seen = {obj['id'] for obj in objects}
    relations = answer.get('relations')
    if not isinstance(relations, list):
        raise RelationValidationError(['relations must be a JSON list'])
    checked, keys, parents, issues = [], set(), {}, []
    # Canonical directed order is low -> high on one camera axis. Inverse
    # wording is equivalent; reversing the same directed pair is contradictory.
    order_types = dict(left_of=('x', False), right_of=('x', True),
                       in_front_of=('z', False), behind=('z', True),
                       below=('y', False), above=('y', True))
    orders = {axis: {} for axis in ('x', 'y', 'z')}
    equivalents = set()
    for index, rel in enumerate(relations):
        if not isinstance(rel, dict) or any(not isinstance(rel.get(k), str) for k in ('a', 'b', 'type')):
            issues.append(f'Relation {index} needs string a, type, b')
            continue
        if rel['a'] not in seen or rel['b'] not in seen or rel['a'] == rel['b'] or rel['type'] not in RELATIONS:
            issues.append(f'Invalid relation IDs or type at {index}: {rel}')
            continue
        key = (rel['a'], rel['type'], rel['b'])
        if key in keys:
            issues.append(f'Duplicate relation: {key}')
            continue
        keys.add(key)
        a, b, kind = rel['a'], rel['b'], rel['type']
        if kind in order_types:
            axis, reverse = order_types[kind]
            lo, hi = (b, a) if reverse else (a, b)
            equivalent = (axis, lo, hi)
            if (axis, hi, lo) in equivalents:
                issues.append(f'Contradictory {axis} order: both {lo} < {hi} and {hi} < {lo}')
            orders[axis].setdefault(lo, []).append(hi)
        elif kind == 'near':
            equivalent = ('near', *sorted((a, b)))
        else:
            equivalent = ('on_top_of', a, b)
        if equivalent in equivalents:
            issues.append(f'Equivalent relation repeated: {rel}')
        equivalents.add(equivalent)
        if rel['type'] == 'on_top_of':
            parents.setdefault(rel['a'], []).append(rel['b'])
        checked.append(dict(a=rel['a'], type=rel['type'], b=rel['b']))
    for oid, supports in parents.items():
        if len(supports) > 1:
            issues.append(f'One supported object must have a single parent: {oid} rests on {supports}')
    def check_cycles(edges, label):
        visiting, done = [], set()
        def visit(node):
            if node in visiting:
                issues.append(label+': '+' -> '.join(visiting[visiting.index(node):]+[node]))
                return
            if node in done:
                return
            visiting.append(node)
            for successor in edges.get(node, []):
                visit(successor)
            visiting.pop()
            done.add(node)
        for oid in edges:
            visit(oid)
    check_cycles(parents, 'Cyclic physical support')
    for axis, edges in orders.items():
        check_cycles(edges, f'Cyclic {axis} order')
    if issues:
        if len(relations) > 40:
            issues.insert(0, f'Advisory relation budget exceeded: {len(relations)} > 40; also simplify the graph while correcting the errors below')
        raise RelationValidationError(issues)
    return dict(**plan, relations=checked)


def repair_relations_prompt(answer, observed, issues):
    return (
        'Review the scene relations against the ORIGINAL IMAGE and the fixed object descriptions/2D boxes. '
        'The previous relation graph failed validation. Correct semantic mistakes as well as graph mistakes. '
        'Return ONLY JSON with one key relations containing the complete corrected relation list. '
        'Keep IDs and all objects unchanged; do not return new additions. Retain useful valid relations, '
        'rather than erasing the graph to pass validation. '
        'Each relation has a, type, b. Allowed types: '+', '.join(RELATIONS)+'. '
        'Return at most 40 placement-critical relations, not all object pairs. '
        'Remove repetition, inverse-equivalent statements and blanket near lists; retain useful new-object relations. '
        'Do not assert opposite orders or cyclic orders on the same camera axis. '
        'on_top_of(a,b) means a rests on b, NOT b rests on a. Each a has at most one support parent; '
        'no reciprocal support or cycles. Above in the image is not physical support. '
        'Ceiling lights are mounted on the ceiling, not supported by the bed; wall fixtures are not '
        'objects resting on furniture. Use above/below for vertical order without contact. '
        'Check instance binding using the 2D boxes: two similar objects need not share one support. '
        'If the correct support object is absent from the fixed inventory, omit that support edge '
        'rather than attach the object to an incorrect instance or invent a new ID. '
        'left/right use camera x, above/below use camera y; in_front_of means smaller camera z, '
        'behind means larger camera z, near means about 1 metre surface separation. '
        '\nVALIDATION ISSUES: '+json.dumps(issues, ensure_ascii=False)
        +'\nFIXED OBSERVED EVIDENCE: '+json.dumps(observed, ensure_ascii=False)
        +'\nPREVIOUS PLAN (objects fixed; correct relations only): '+json.dumps(answer, ensure_ascii=False))


def validated_plan(answer, observed, capacity, ask, output):
    """One feedback correction, not automatic edge pruning or relaxed checks."""
    count = len(answer['relations']) if isinstance(answer, dict) and isinstance(answer.get('relations'), list) else None
    diagnostic = dict(repair_attempted=False, initial_issues=[], final_graph_valid=False,
                      initial_relation_count=count, suggested_relation_budget=40,
                      relation_budget_exceeded=count is not None and count > 40,
                      note='Count budget is advisory; contradictory or malformed relations require correction')
    try:
        plan = decode_plan(answer, observed, capacity)
    except RelationValidationError as exc:
        diagnostic.update(repair_attempted=True, initial_issues=exc.issues)
        write(output/'relation_validation.json', diagnostic)
        print(f'Relation validation: {len(exc.issues)} issues; one image-grounded correction request', flush=True)
        reply = ask(repair_relations_prompt(answer, observed, exc.issues), 'relation_correction', 4096)
        if not isinstance(reply, dict) or set(reply) != {'relations'}:
            raise ValueError('Relation correction must contain ONLY relations; object list is fixed')
        corrected = deepcopy(answer)
        corrected['relations'] = reply['relations']
        write(output/'semantic_plan_corrected.json', corrected)
        try:
            plan = decode_plan(corrected, observed, capacity)
        except (ValueError, KeyError, TypeError) as final:
            diagnostic['final_error'] = str(final)
            write(output/'relation_validation.json', diagnostic)
            raise
        before_relations = answer['relations'] if isinstance(answer['relations'], list) else []
        before = {(r['a'], r['type'], r['b']) for r in before_relations
                  if isinstance(r, dict) and all(isinstance(r.get(k), str) for k in ('a', 'type', 'b'))}
        after = {(r['a'], r['type'], r['b']) for r in plan['relations']}
        diagnostic.update(removed_relations=sorted(before-after), added_relations=sorted(after-before))
    diagnostic.update(final_graph_valid=True, semantic_quality='requires_visual_review')
    write(output/'relation_validation.json', diagnostic)
    return plan, diagnostic


def layout_prompt(plan, hfov, image_size, camera_height=1.5, room_height=2.7):
    return (
        'Create one approximate spatial layout for ALL the planned instances. '
        'Use the photograph to anchor observed objects and common-sense sizes for missing geometry. '
        'This is a plausible world hypothesis, not precise 3D reconstruction. '+AXES
        +f'Horizontal FOV={hfov} degrees, image width/height={image_size[0]}/{image_size[1]}. '
        'For OBSERVED objects return depth_m (positive camera-forward depth) and size only, '
        'plus id and support. Do not generate observed center x/y: the code calculates them from '
        'the supplied 2D boxes and depth. size must still be FULL xyz camera-axis dimensions in metres. '
        'Respect the descriptions and placement phrases. Keep outside_fov objects outside the viewing cone; they can lie to either '
        'side, above/below, or behind camera. Do not put all new objects behind camera by default. '
        +f'For scale, ASSUME camera height {camera_height}m and room height {room_height}m; these are priors. '
        'Do not generate room bounds. Return ONLY JSON with objects, a list with exactly one entry per ID. '
        'For UNSEEN objects return center (xyz metres), size, id and support. '
        'support is ONE string: floor, wall, ceiling, free, '
        'or the ID of the object THIS object rests on). '
        'A cushion uses its bed ID as support; the bed uses floor. A desk lamp uses its nightstand ID '
        'if that nightstand is in the inventory; otherwise free means support unknown. '
        'If supported by another object, place its bottom near the support object top. '
        'Wall pictures/windows/doors use wall, suspended ceiling fixtures use ceiling, '
        'standing furniture uses floor. AABBs are axis aligned; no rotation fields. '
        'Do not add, drop or duplicate any ID. No relations array. '
        'No null geometry or long explanation; choose coarse hypotheses. '
        '\nComplete semantic plan and observed evidence: '+json.dumps(plan, ensure_ascii=False))


def observed_layout_prompt(observed, hfov, image_size, camera_height, room_height):
    return (
        'Estimate a coarse state ONLY for the visible instances in this photograph. '+AXES
        +f'Horizontal FOV={hfov} degrees, image width/height={image_size[0]}/{image_size[1]}. '
        +f'For approximate scale only, assume camera height {camera_height}m and room height {room_height}m. '
        'Return ONLY JSON {"objects":[{"id":"...","depth_m":2.5,"center_uv":[0.5,0.5],"size":[1.0,0.6,0.8]}]}. '
        'The numbers are format examples, not measurements. Include every supplied ID exactly once. '
        'depth_m is positive forward depth z in metres; size is FULL x/y/z camera-axis AABB dimensions. '
        'depth_m is NOT Euclidean distance along a normalized ray. '
        'center_uv is the image projection of the FULL 3D box center, normalized by image width/height. '
        'The center of a partially visible 2D bounding box is not generally that point; estimate it '
        'using the complete object. It may lie outside the visible fragment. '
        'Thin wall objects do not necessarily have tiny camera-x size: transform their spatial extent '
        'to camera axes. Do not output intrinsic OBB dimensions or half extents. '
        'Choose depth and dimensions together using the visible image extent and common-sense object scale. '
        'The code computes camera x/y from your center_uv and z using the supplied pinhole FOV. '
        'Do not generate new objects, support relations, 3D center xyz, room bounds or rotations. '
        'Do not adjust objects to y=-1.5: the floor and gravity direction are not measured. '
        'These are approximate hypotheses, not exact reconstruction. '
        '\nVisible instances (bbox_2d normalized xyxy): '+json.dumps(observed, ensure_ascii=False))


def completion_prompt(observed, observed_layout, hfov, max_added, camera_height, room_height):
    boxes = {box['id']: box for box in observed_layout['objects']}
    context = [dict(**obj, center=boxes[obj['id']]['center'], size=boxes[obj['id']]['size']) for obj in observed]
    return (
        'Plan ONE plausible complete current room extending OUTSIDE this partial photo. '
        'Keep the fixed observed instances unchanged and preserve visible empty space. '
        'Choose scene-appropriate unseen furniture and contents in the SAME room, not beyond room walls. '
        'This first coarse layout plans the surrounding room, not hidden objects squeezed behind visible furniture. '
        'Return only new instances, not duplicates or variants of observed ones. '
        'Choose azimuth_deg in [-180,180]: 0 is camera forward, -90 left, +90 right, +/-180 behind. '
        'Intermediate angles are allowed; the angle is around the CAMERA, not around the bed. '
        f'The photo covers approximately [{-hfov/2:g},{hfov/2:g}] horizontal degrees. '
        'Place the whole new object outside the photo, accounting for its size, not just its center. '
        'Use the visible room boundaries and furniture as context for ONE coherent room hypothesis. '
        'Do not distribute one object per direction mechanically or add decorations merely to fill space. '
        'Choose distance_m, a positive horizontal distance from the camera in metres. '
        'Choose bottom_height_m, height of the object BOTTOM above the assumed floor; '
        '0 means floor-standing. Do not output a camera y coordinate. '
        'size is positive FULL x/y/z CAMERA-AXIS AABB dimensions in metres (x right, y up, z forward). '
        'The program converts azimuth, distance and bottom height to xyz; do not output center or rotations. '
        'Coordinate-format example only (not a required object): '
        '{"id":"new_01","category":"cabinet","description":"storage cabinet",'
        '"azimuth_deg":125,"distance_m":2.8,"bottom_height_m":0,'
        '"size":[1.2,1.8,0.5],"reason":"storage appropriate for this room"}. '
        'Choose your own categories, sizes and arrangement; do not copy the example as a scene. '
        f'At most {max_added} additions, not a target; an empty list is allowed. '
        'Return ONLY JSON with room_type and additions. Each addition has id, category, description, '
        'azimuth_deg, distance_m, bottom_height_m, size, reason. '
        'No relations or support graph. '
        f'Photo horizontal FOV={hfov} degrees; assumed camera height={camera_height}m, room height={room_height}m. '
        'Allowed categories: '+VOCAB
        +'\nFIXED OBSERVED STATE: '+json.dumps(context, ensure_ascii=False))


def make_graph(objects, layout, vocabulary_hash, method, scope='full', relations=None):
    boxes = {o['id']: o for o in layout['objects']} if layout else {}
    values = []
    for obj in objects:
        box = boxes.get(obj['id'])
        values.append(dict(track_id=obj['id'], description=obj['description'],
            category_id=field(CATEGORY_IDS[obj['category']], 'qwen_semantic_prediction', 'existing_canonical_vocabulary'),
            center_local_m=field(box['center'] if box else None, 'scene_hypothesis', method, reason='not_estimated'),
            size_aabb_local_m=field(box['size'] if box else None, 'scene_hypothesis', 'camera_axis_full_sizes', reason='not_estimated'),
            bbox2d_visible_xyxy_norm=field(obj.get('bbox_2d'), 'qwen_semantic_prediction', 'visible_grounding', reason='hypothesized_unseen'),
            in_primary_room=field(None, reason='room_is_a_hypothesis'),
            local_evidence='visible' if obj['role'] == 'observed' else 'unknown',
            completion_role=obj['role'], unseen_reason=obj.get('unseen_reason'), reason=obj.get('reason'),
            layout_parameters={key: deepcopy(obj[key]) for key in ('azimuth_deg', 'camera_region', 'distance_m', 'bottom_height_m', 'anchor_id', 'offset_m') if key in obj},
            geometry_quality='hypothesized_not_reconstructed'))
    state = graph('single_image', 'inference', scope, values, vocabulary_hash)
    for source, value in zip(objects, values):
        for key in ('raw_category','category_mapping','description_source'):
            if key in source:
                value[key]=deepcopy(source[key])
        if 'category_mapping' in source:
            value['category_id']['method']=source['category_mapping']['method']
    state.update(relations=relations or [], quality_status='requires_visual_review',
                 geometry_assumptions='camera_axes_not_levelled; approximate_depth_and_full_AABB_sizes; no_physical_optimization',
                 room_hypothesis=layout['room'] if layout else None)
    return validate_graph(state)


def previews(image, objects, raw, corrected, hfov, output):
    for name, layout in (('raw', raw), ('refined', corrected)):
        problem = BoxProblem(layout, objects, [], hfov, image.size)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for i, obj in enumerate(objects):
            box = projected_box(problem.initial[i], problem.sizes[i], problem.tan_h, problem.tan_v)
            if obj['role'] == 'observed':
                target = obj['bbox_2d']
                draw.rectangle([target[0]*image.width, target[1]*image.height,
                                target[2]*image.width, target[3]*image.height], outline='red', width=2)
            if box is not None and np.all(box[2:] > 0) and np.all(box[:2] < 1):
                clipped = np.clip(box, 0, 1)*[image.width-1, image.height-1, image.width-1, image.height-1]
                color = 'lime' if obj['role'] == 'observed' else 'orange'
                draw.rectangle(clipped.tolist(), outline=color, width=2)
                draw.text(tuple(clipped[:2]), obj['id'], fill=color)
        overlay.save(output/f'projection_{name}.png')
        # The top-down x/z plot hides height errors. Show camera-y intervals
        # separately, without drawing an invented floor or gravity direction.
        heights = Image.new('RGB', (1000, 100+28*len(objects)), 'white')
        hd = ImageDraw.Draw(heights)
        low = min(0.,float((problem.initial-problem.sizes/2)[:,1].min()))-.1
        high = max(0.,float((problem.initial+problem.sizes/2)[:,1].max()))+.1
        def height_x(y):return 300+650*(y-low)/(high-low)
        hd.text((16,12),'Camera-y AABB intervals (metres); image-up is NOT measured gravity. No floor shown.',fill='black')
        for tick in np.linspace(low,high,6):
            hd.text((height_x(tick)-12,32),f'{tick:.2f}',fill='black')
        hd.line((height_x(0),48,height_x(0),heights.height-12),fill='gray')
        hd.text((height_x(0)+3,48),'y=0',fill='gray')
        for i,obj in enumerate(objects):
            cy=problem.initial[i,1];sy=problem.sizes[i,1];row=70+28*i
            color='steelblue' if obj['role']=='observed' else 'darkorange'
            hd.text((16,row-6),obj['id']+' '+obj['category'],fill=color)
            hd.line((height_x(cy-sy/2),row,height_x(cy+sy/2),row),fill=color,width=5)
            hd.ellipse((height_x(cy)-3,row-3,height_x(cy)+3,row+3),fill='black')
        heights.save(output/f'camera_y_{name}.png')
        canvas = Image.new('RGB', (900, 900), 'white')
        draw = ImageDraw.Draw(canvas)
        lo, hi = problem.lo[[0, 2]], problem.hi[[0, 2]]
        # Include out-of-room boxes so boundary violations remain visible.
        lo = np.minimum(lo, (problem.initial-problem.sizes/2)[:, [0, 2]].min(0))
        hi = np.maximum(hi, (problem.initial+problem.sizes/2)[:, [0, 2]].max(0))
        scale = 780/max(hi-lo)
        def uv(point):
            return (60+(point[0]-lo[0])*scale, 840-(point[1]-lo[1])*scale)
        def rectangle(a, b, color, width):
            u1, v1 = uv(a); u2, v2 = uv(b)
            draw.rectangle([u1, v2, u2, v1], outline=color, width=width)
        rectangle(problem.lo[[0, 2]], problem.hi[[0, 2]], 'black', 3)
        for obj, center, size in zip(objects, problem.initial, problem.sizes):
            color = 'blue' if obj['role'] == 'observed' else 'darkorange'
            rectangle((center-size/2)[[0, 2]], (center+size/2)[[0, 2]], color, 2)
            draw.text(uv(center[[0, 2]]), obj['id']+' '+obj['category'], fill=color)
        origin = uv([0, 0]); draw.ellipse([origin[0]-4, origin[1]-4, origin[0]+4, origin[1]+4], fill='black')
        depth = max(hi-lo)*.3
        for sign in (-1, 1):
            draw.line([origin, uv([sign*problem.tan_h*depth, depth])], fill='gray', width=2)
        draw.text((15, 15), f'{name}: camera dot; +z up; blue observed, orange proposed; black envelope is NOT recovered walls', fill='black')
        canvas.save(output/f'layout_{name}.png')


def run(model, image, hfov, output, vocabulary_hash, max_observed=32, max_added=8, seed=3407, reused=None,
        cached_plan=None, cached_layout=None, camera_height=1.5, room_height=2.7, fixed_observed_state=None,
        cached_completion=None, completion_mode='direct', up_local=None, observe_only=False, inventory_only=False,
        cached_contents=None, layout_examples=None):
    phase = 'observation'
    feedback = dict(used=False, before=[], after=[], withdrawn=[], revision_error=None)
    geometry = camera_geometry(up_local,camera_height)
    try:
        def ask(prompt, name, tokens):
            if name == 'room_contents' and cached_contents is not None:
                print('Fixed room contents reused; room_contents skipped', flush=True)
                write(output/'room_contents.json', cached_contents)
                return deepcopy(cached_contents)
            print('Qwen stage: '+name, flush=True)
            extra = {}
            if layout_examples and name in ('scene_boxes','anchored_scene'):
                if name == 'anchored_scene':
                    from .anchored_examples import messages
                    extra = dict(examples=messages(layout_examples))
                else:
                    extra = dict(examples=layout_example_messages(layout_examples))
            return model.ask(image, prompt, output/(name+'.txt'), seed=seed, sample=True,
                             profile='official', guard_repetition=True, max_tokens=tokens, **extra)
        observed = deepcopy(reused) if reused is not None else visible_inventory(
            ask(observation_prompt('', max_observed), 'observation', 3072), max_observed)
        write(output/'observed_inventory.json', observed)
        write(output/'G_obs.json', fixed_observed_state if fixed_observed_state is not None else
              make_graph(observed, None, vocabulary_hash, 'not_estimated', 'observed'))
        print(f'Visible inventory: {len(observed)}; no per-object 3D requests', flush=True)
        if inventory_only:
            summary = dict(status='completed', phase='inventory_only', observed=len(observed),
                           numeric_geometry_executed=False, completion_executed=False)
            write(output/'summary.json', summary)
            print(json.dumps(summary), flush=True)
            return summary
        if cached_plan is None:
            if fixed_observed_state is not None:
                observed_boxes = observed_state_layout(fixed_observed_state, observed)
                methods = sorted({o['center_local_m']['method'] for o in fixed_observed_state['objects']})
                print('Fixed numeric G_obs reused; source='+','.join(methods)+'; observed_layout skipped', flush=True)
            else:
                phase = 'observed_layout'
                observation_layout = ask(observed_layout_prompt(observed, hfov, image.size, camera_height, room_height),
                                         'observed_layout', 4096)
                observed_boxes, observer_initialization = camera_layout(observation_layout, observed, hfov, image.size, camera_height, room_height)
                write(output/'observed_geometry_initialization.json',observer_initialization)
                # Materialize the local state BEFORE requesting any completion.
                write(output/'G_obs.json', make_graph(observed, observed_boxes, vocabulary_hash,
                      'predicted_z_and_projected_center_camera_rays', 'observed'))
                new_state=read(output/'G_obs.json')
                explicit=set(observer_initialization['explicit_projected_center_ids'])
                for obj in new_state['objects']:
                    obj['center_local_m']['method']=('predicted_z_and_full_box_center_uv'
                        if obj['track_id'] in explicit else 'predicted_z_and_visible_bbox_center_proxy')
                write(output/'G_obs.json',new_state)
            observed_state=read(output/'G_obs.json')
            if fixed_observed_state is None:
                observed_state['camera_geometry']=geometry
                observed_state['geometry_assumptions']='camera frame retained; gravity/floor are explicit unverified priors'
                write(output/'G_obs.json',observed_state)
            observed_report=observation_report(observed_state,hfov,image.size,geometry)
            if completion_mode == 'anchored':
                observed_report['camera_geometry']=deepcopy(observed_state.get('camera_geometry',{}))
                observed_report['protocol']='camera_projection_only_no_floor_prior_v1'
                for record in observed_report['objects']:
                    record.pop('min_box_signed_distance_to_floor_prior_m',None)
                    record.pop('floor_collision_verified',None)
            write(output/'observation_geometry_report.json',observed_report)
            if observe_only:
                previews(image,observed,observed_boxes,observed_boxes,hfov,output)
                summary=dict(status='completed',phase='observation_only',observed=len(observed),
                    completion_executed=False,observation_geometry_verified=False,
                    gravity_verified=False,floor_verified=False,
                    visible_bbox_center_proxy_count=sum(o['center_local_m']['method']=='predicted_z_and_visible_bbox_center_proxy' for o in observed_state['objects']),
                    note='Observed hypotheses saved for projection/GT review; no scene completion or floor correction')
                write(output/'summary.json',summary)
                print(json.dumps(summary,ensure_ascii=False),flush=True)
                return summary
            phase = 'room_planning' if completion_mode == 'planned' else 'scene_completion'
            if completion_mode == 'planned':
                if cached_completion is not None:
                    raise ValueError('Planned mode starts a new room content plan; no old completion reuse')
                answer, conversion = complete_planned(ask, observed, observed_boxes, hfov, image.size,
                                                       camera_height, max_added, output, geometry)
                plan = scene_objects(answer, observed, max_added)
                write(output/'completion_conversion.json', conversion)
            elif completion_mode == 'anchored':
                phase = 'anchored_scene'
                if layout_examples:
                    write(output/'layout_examples.json', layout_examples)
                    print(f'Matched independent training examples: {len(layout_examples["examples"])}', flush=True)
                answer, conversion = complete_anchored(ask, observed, observed_boxes, hfov, max_added)
                plan = scene_objects(answer, observed, max_added)
                write(output/'completion_conversion.json', conversion)
            elif completion_mode == 'program':
                phase = 'placement_program'
                answer, conversion = complete_program(ask, observed, observed_boxes, hfov, max_added)
                plan = scene_objects(answer, observed, max_added)
                write(output/'completion_conversion.json', conversion)
            elif completion_mode == 'boxes':
                phase = 'scene_boxes'
                if layout_examples:
                    write(output/'layout_examples.json', layout_examples)
                    print(f'Training-room layout demonstrations: {len(layout_examples["examples"])}; fresh scene hypothesis', flush=True)
                if cached_completion is not None:
                    write(output/'completion_initial.json', cached_completion)
                    print('Saved new-object inventory reused; one projection-feedback revision', flush=True)
                answer, conversion = complete_boxes(ask, observed, observed_boxes, hfov, max_added,
                    cached=cached_completion, validate=lambda a: scene_objects(a, observed, max_added),
                    image_size=image.size)
                feedback = conversion['feedback']
                write(output/'completion_feedback.json', feedback)
                plan = scene_objects(answer, observed, max_added)
                write(output/'completion_conversion.json', conversion)
            else:
                phase = 'scene_completion'
                if cached_completion is None:
                    proposal = ask(completion_prompt(observed, observed_boxes, hfov, max_added, camera_height, room_height),
                                   'scene_completion', 4096)
                else:
                    proposal = deepcopy(cached_completion)
                    write(output/'scene_completion.json', proposal)
                    print('Saved completion reused; checking candidates before one revision', flush=True)
                answer, conversion = directional_layout(proposal, camera_height)
                write(output/'completion_conversion.json', conversion)
                # scene_objects checks identity/category/semantic fields. camera_layout
                # below independently checks all new numeric geometry.
                plan = scene_objects(answer, observed, max_added)
                write(output/'completion_initial.json', answer)
                answer, feedback = revise_once(answer,
                    completion_prompt(observed, observed_boxes, hfov, max_added, camera_height, room_height),
                    ask, lambda candidate: scene_objects(candidate, observed, max_added),
                    hfov, image.size, camera_height)
                write(output/'completion_feedback.json', feedback)
                plan = scene_objects(answer, observed, max_added)
            answer_layout = dict(objects=[dict(id=o['id'], depth_m=o['center'][2], size=o['size'], support=o['support'])
                                         for o in observed_boxes['objects']]+answer['additions'])
            write(output/'semantic_plan.json', answer)
            write(output/'coarse_layout.json', answer_layout)
        else:
            # Historical continuation: keep source semantics/raw values auditable.
            # New experiments should reuse only the visible inventory, not this plan.
            phase = 'semantic_plan'
            answer = cached_plan
            plan = scene_objects(answer, observed, max_added)
            phase = 'coarse_layout'
            answer_layout = cached_layout if cached_layout is not None else ask(
                layout_prompt(plan, hfov, image.size, camera_height, room_height), 'coarse_layout', 6144)
        legacy_relations = answer.get('relations')
        write(output/'planning_protocol.json', dict(protocol=ANCHORED_PROTOCOL if completion_mode == 'anchored' else PROGRAM_PROTOCOL if completion_mode == 'program' else BOX_PROMPT_VERSION if completion_mode == 'boxes' else 'content_geometry_id_relations_camera_v3' if completion_mode == 'planned' else 'fixed_observation_then_directional_additions_v7',
            completion_parameterization=PARAMETERIZATION[completion_mode] if cached_plan is None else 'historical_numeric_layout',
            observed_state_predates_completion=cached_plan is None,
            numeric_observation_reused=fixed_observed_state is not None,
            content_plan_reused=cached_contents is not None,
            historical_plan_reused=cached_plan is not None,
            generated_relation_graph_used=False,
            explicit_id_relations=answer.get('relations',[]),
            legacy_relation_count=len(legacy_relations) if isinstance(legacy_relations, list) else 0,
            note='Observed geometry is frozen. No generated support graph or physical optimizer is applied. '
                 'Historical plans retain their original semantic limitations.'))
        write(output/'scene_plan.json', plan)
        print(f'World hypothesis: {len(plan["objects"])} objects', flush=True)
        phase = 'coarse_layout'
        if completion_mode == 'anchored':
            # These centres are already in the correct camera frame. Do not
            # route them through legacy floor/room or bbox-ray initializers.
            raw = dict(room=deepcopy(observed_boxes['room']), objects=deepcopy(observed_boxes['objects'])+
                       [dict(id=o['id'], center=o['center'], size=o['size'], support=None) for o in answer['additions']])
            initialization = dict(protocol=ANCHORED_PROTOCOL, room_envelope='display_only',
                                  floor_prior_used_for_placement=False, observed_centers='fixed_G_obs')
        else:
            raw, initialization = camera_layout(answer_layout, plan['objects'], hfov, image.size, camera_height, room_height)
        if fixed_observed_state is not None:
            # camera_layout checked all additions. Reused observed boxes keep
            # their original centers, even when not exactly on bbox-center rays.
            raw['objects'][:len(observed)] = deepcopy(observed_boxes['objects'])
            initialization['observed_centers'] = 'copied_exactly_from_fixed_numeric_G_obs'
        elif cached_plan is None:
            raw['objects'][:len(observed)] = deepcopy(observed_boxes['objects'])
            initialization['observed_centers']='frozen_from_observer_before_completion; explicit full-center projection or legacy proxy'
        else:
            write(output/'G_obs.json', make_graph(observed, raw, vocabulary_hash,
                  'predicted_depth_with_2d_ray_anchors', 'observed'))
        write(output/'camera_initialization.json', initialization)
        write(output/'layout_raw.json', raw)
        phase = 'cpu_inspection'
        corrected = deepcopy(raw)
        diagnostic = inspect_layout(raw, plan['objects'], hfov, image.size)
        report = dict(protocol='read_only_projection_checks_v1', before=diagnostic, after=deepcopy(diagnostic),
                      optimizer=dict(enabled=False, success=None, nfev=0,
                                     message='Disabled: unverified support and room priors must not move observed states'),
                      observed_state_preserved=True, variables='none; all model hypotheses preserved',
                      quality_status='requires_visual_review',
                      note='Not comparable to historical optimizer warning totals; no physics certification')
        if completion_mode == 'program':
            report['placement_construction'] = deepcopy(conversion)
            report['variables'] = 'Model group/edge decisions compiled; observed boxes unchanged; no optimization'
        elif completion_mode == 'anchored':
            report['placement_construction'] = deepcopy(conversion)
            report['variables'] = 'Explicit centre offsets compiled without group translation; no optimization'
        write(output/'support_diagnostic.json', dict(applied=False, accepted=[], warnings=[],
              note='Not used: whole-object AABB tops are not measured support surfaces; source suggestions remain in raw replies'))
        if completion_mode=='planned' and cached_plan is None:
            support_checks=[r for r in conversion.get('relation_diagnostic',{}).get('relations',[]) if r['type']=='on']
            write(output/'support_diagnostic.json',dict(applied=bool(support_checks),
                accepted=[r for r in support_checks if r['status']=='satisfied'],
                warnings=[r for r in support_checks if r['status']!='satisfied'],
                source='estimated_support_offsets_and_explicit_up_prior',
                note='Dependent child candidates follow the selected parent; no AABB-top or mesh-support certification'))
        write(output/'layout_refined.json', corrected)
        write(output/'geometry_report.json', report)
        completion_method = ('fixed_anchor_plus_camera_offset' if completion_mode == 'anchored' else 'camera_group_edge_placement_compiler' if completion_mode == 'program' else 'joint_model_camera_xyz' if completion_mode == 'boxes' else 'content_then_ranked_candidate_placement' if completion_mode == 'planned' else 'camera_azimuth_range_bottom_height_compiler') if cached_plan is None else 'historical_unseen_xyz_hypothesis'
        full = make_graph(plan['objects'], corrected, vocabulary_hash, completion_method)
        full['scene_hypothesis'] = answer.get('scene_hypothesis')
        full['camera_geometry']=geometry
        if completion_mode == 'anchored':
            full['camera_geometry']=deepcopy(fixed_observed_state.get('camera_geometry', {}))
            full['geometry_assumptions']='fixed camera axes; no floor, room-boundary or gravity prior used for placement'
        full['relations']=deepcopy(answer.get('relations',[]))
        full['object_commitment']=dict(authoritative_object_ids=[o['track_id'] for o in full['objects']],
               prose_adds_objects=False,encoder_uses='category_id,center_local_m,size_aabb_local_m,valids,mask',
               relation_metadata_consumed_by_encoder=False)
        # Preserve every observed state field, including provenance, byte-for-byte
        # at the JSON value level. Completion only appends new object records.
        full['objects'][:len(observed)] = deepcopy(read(output/'G_obs.json')['objects'])
        validate_graph(full)
        write(output/'G_full_raw.json', full)
        write(output/'G_full.json', full)
        write(output/'G_hidden.json', make_graph(plan['objects'][len(observed):], raw, vocabulary_hash,
              completion_method, 'hidden'))
        phase = 'previews'
        previews(image, plan['objects'], raw, corrected, hfov, output)
        if completion_mode == 'anchored':
            from .batch_world_review import interactive_boxes
            interactive_boxes(full,hfov,image.size,output/'scene_3d.html')
        summary = dict(status='completed', observed=len(observed),
            hypothesized=len(plan['objects'])-len(observed), full_objects=len(plan['objects']),
            completion_outcome='objects_added' if len(plan['objects']) > len(observed) else 'no_new_objects',
            optimizer_success=report['optimizer']['success'],
            optimizer_enabled=False, observed_state_preserved=True,
            explicit_id_relations_used=bool(answer.get('relations',[])),
            numeric_observation_reused=fixed_observed_state is not None,
            content_plan_reused=cached_contents is not None,
            diagnostic_protocol=report['protocol'],
            observation_geometry_verified=False, scene_semantics_verified=False,
            legacy_direction_fallback_count=len(conversion['legacy_region_ids']) if cached_plan is None else None,
            completion_parameterization=PARAMETERIZATION[completion_mode] if cached_plan is None else 'historical_numeric_layout',
            relation_correction_used=False, generated_relation_graph_used=False,
            cached_layout_postprocessed=cached_layout is not None,
            support_suggestion_warnings=0,
            geometry_warnings_before=len(report['before']['warnings']),
            geometry_warnings_after=len(report['after']['warnings']),
            quality_status='requires_visual_review', panorama_generated=False,
            observed_at_capacity=len(observed) == max_observed,
            additions_at_capacity=len(plan['objects'])-len(observed) == max_added,
            outside_fov_verified=sum(o.get('outside_fov_verified', False) for o in diagnostic['objects']),
            outside_fov_failed=sum(o.get('outside_fov_verified') is False for o in diagnostic['objects']),
            occlusion_unverified=sum(o.get('unseen_reason') == 'occluded' for o in plan['objects']),
            completion_mode=completion_mode,
            suspected_aabb_overlap_pairs=len(conversion['overlap_report']['pairs']) if completion_mode in ('program','anchored') else None,
            layout_example_count=len(layout_examples['examples']) if layout_examples else 0,
            content_plan_count=conversion['planned_count'] if completion_mode == 'planned' and cached_plan is None else None,
            placed_plan_count=conversion['placed_count'] if completion_mode == 'planned' and cached_plan is None else None,
            relations_satisfied=conversion.get('relation_diagnostic',{}).get('satisfied',0) if cached_plan is None else 0,
            relations_violated=conversion.get('relation_diagnostic',{}).get('violated',0) if cached_plan is None else 0,
            relations_unverified=conversion.get('relation_diagnostic',{}).get('unverified',0) if cached_plan is None else 0,
            completion_revision_used=feedback['used'],
            completion_numeric_revision_changed=feedback.get('numeric_revision_changed'),
            completion_conflicts_before=len(feedback['before']),
            completion_conflicts_after=len(feedback['after']),
            completion_withdrawn=len(feedback['withdrawn']),
            completion_revision_error=feedback['revision_error'],
            note='Completed means artifacts written, not correct semantics, geometry or panorama control')
        if summary['outside_fov_failed'] or feedback['revision_error']:
            summary['quality_status'] = 'needs_completion_revision'
        elif summary['relations_violated']:
            summary['quality_status'] = 'needs_relation_review'
        elif summary['suspected_aabb_overlap_pairs']:
            summary['quality_status'] = 'needs_aabb_overlap_review'
        write(output/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        write(output/'summary.json', dict(status='failed', phase=phase, error=str(exc), raw_responses_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE)
    parser.add_argument('--hfov', type=float, required=True)
    parser.add_argument('--completion-mode', choices=['direct', 'planned', 'boxes', 'program', 'anchored'], default='direct',
                        help='anchored: continuous object-relative offsets; program: historical region groups; boxes: model xyz; planned/direct: historical baselines')
    parser.add_argument('--observe-only',action='store_true',help='Save G_obs and diagnostics before any scene completion')
    parser.add_argument('--inventory-only', action='store_true', help='Only Qwen categories/2D boxes; skip numerical 3D and completion')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--vocabulary', type=Path, default=DEFAULT_VOCAB)
    parser.add_argument('--reuse-observation', type=Path, help='Reuse same-image 2D inventory only, never old 3D')
    parser.add_argument('--reuse-state', type=Path, help='Reuse complete numeric G_obs unchanged; request only a new completion')
    parser.add_argument('--reuse-contents', type=Path, help='With planned mode and --reuse-state, reuse a saved room_contents.json; only estimate geometry/placement')
    parser.add_argument('--reuse-completion', type=Path, help='With --reuse-state, check/revise once: scene_boxes.json in boxes mode, scene_completion.json in direct mode')
    parser.add_argument('--layout-examples', type=Path, help='Training demonstrations: old boxes bank or anchored pool, matched to the selected mode; no target-scene GT')
    parser.add_argument('--continue-from', type=Path, help='Reuse same-image observation and object hypotheses only; re-estimate layout, never reuse old relation graph')
    parser.add_argument('--postprocess-only', action='store_true', help='CPU only: finish --continue-from using its saved coarse_layout.json')
    parser.add_argument('--camera-height', type=float, default=1.5, help='Explicit hypothesized camera height in metres, not measured')
    parser.add_argument('--up-local',type=float,nargs=3,help='Optional gravity-up direction in camera axes; never levels or rotates G_obs')
    parser.add_argument('--room-height', type=float, default=2.7, help='Explicit hypothesized room height in metres, not measured')
    parser.add_argument('--max-observed', type=int, default=32)
    parser.add_argument('--max-additions', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.completion_mode in ('boxes','program','anchored') and (not args.reuse_state or args.observe_only or args.inventory_only):
        parser.error('--completion-mode boxes/program/anchored requires a fixed --reuse-state')
    if args.layout_examples and (args.completion_mode not in ('boxes','anchored') or args.reuse_completion or args.continue_from):
        parser.error('--layout-examples uses a fresh boxes/anchored hypothesis, not old completion revision')
    if args.reuse_contents and (args.completion_mode != 'planned' or not args.reuse_state or args.observe_only or args.inventory_only):
        parser.error('--reuse-contents requires --completion-mode planned and --reuse-state')
    if args.inventory_only and (args.observe_only or args.reuse_state or args.continue_from or args.reuse_completion or args.postprocess_only):
        parser.error('--inventory-only needs only a current image or --reuse-observation')
    camera_geometry(args.up_local,args.camera_height)
    if (args.completion_mode in ('planned','program','anchored') and (args.continue_from or args.reuse_completion)) or (args.completion_mode == 'boxes' and args.continue_from):
        parser.error('Use a new scene hypothesis; omit --continue-from/--reuse-completion')
    if args.up_local and args.completion_mode=='direct' and not args.observe_only:
        parser.error('--up-local placement requires --completion-mode planned; direct is the historical baseline')
    if args.observe_only and (args.continue_from or args.reuse_completion):
        parser.error('--observe-only uses the current observer, not old completed plans')
    if not math.isfinite(args.hfov) or not 1 <= args.hfov <= 160:
        parser.error('Use a perspective RGB with horizontal FOV between 1 and 160 degrees')
    if not 1 <= args.max_observed <= 48 or not 0 <= args.max_additions <= 12:
        parser.error('First-version limits: 1..48 observed and 0..12 additions')
    if args.output.exists():
        parser.error('Output exists; use a new directory')
    if not args.postprocess_only and (not args.model.is_dir() or not list(args.model.glob('*.safetensors'))):
        parser.error('Existing local model weights required; no downloads')
    if not (math.isfinite(args.camera_height) and math.isfinite(args.room_height)
            and 0 < args.camera_height < args.room_height):
        parser.error('Require 0 < --camera-height < --room-height; both are recorded assumptions')
    if args.postprocess_only and not args.continue_from:
        parser.error('--postprocess-only requires --continue-from with saved coarse_layout.json')
    vocab = read(args.vocabulary)
    if {int(k): v for k, v in vocab['categories'].items()} != CANONICAL_CATEGORIES:
        parser.error('Vocabulary differs from current Encoder contract')
    with Image.open(args.image) as handle:
        image = handle.convert('RGB')
    image.thumbnail((768, 768))
    if sum(p is not None for p in (args.continue_from, args.reuse_observation, args.reuse_state)) > 1:
        parser.error('Choose only one of --continue-from, --reuse-observation or --reuse-state')
    reuse_path = args.continue_from/'G_obs.json' if args.continue_from else args.reuse_state or args.reuse_observation
    reused = reuse_inventory(reuse_path, image, vocab['category_mapping_sha256'], args.max_observed) if reuse_path else None
    fixed_observed_state = None
    if args.reuse_state:
        source_config = read(args.reuse_state.parent/'config.json')
        if not math.isclose(source_config['hfov_deg'], args.hfov, abs_tol=1e-9):
            parser.error('Fixed observed state FOV differs from requested FOV')
        fixed_observed_state = read(args.reuse_state)
        observed_state_layout(fixed_observed_state, reused)
    layout_examples = None
    if args.layout_examples:
        if args.completion_mode == 'anchored':
            from .anchored_examples import select
            layout_examples = select(args.layout_examples, vocab['category_mapping_sha256'], args.hfov,
                                     sha(args.image), reused)
        else:
            layout_examples = load_layout_examples(args.layout_examples, vocab['category_mapping_sha256'], args.hfov,
                                                  sha(args.image))
    cached_completion = None
    cached_contents = None
    contents_prompt_version = CONTENT_PROMPT_VERSION
    if args.reuse_contents:
        # A plan is meaningful only for the exact observation it was given.
        if read(args.reuse_contents.parent/'G_obs.json') != fixed_observed_state:
            parser.error('Saved room contents were conditioned on a different numeric G_obs')
        cached_contents = validate_contents(read(args.reuse_contents), reused, args.max_additions)
        contents_prompt_version = read(args.reuse_contents.parent/'config.json').get(
            'completion_prompt_version', 'unknown_saved_prompt')
    if args.reuse_completion:
        if fixed_observed_state is None or args.postprocess_only:
            parser.error('--reuse-completion requires --reuse-state and a model revision run')
        source = args.reuse_completion.parent
        reuse_inventory(source/'G_obs.json', image, vocab['category_mapping_sha256'], args.max_observed)
        if read(source/'G_obs.json') != fixed_observed_state:
            parser.error('Saved completion was conditioned on a different numeric observed state')
        previous = read(source/'config.json')
        if previous.get('completion_mode', 'direct') != args.completion_mode:
            parser.error('Saved completion mode differs from --completion-mode')
        assumptions = [('hfov_deg', args.hfov)]
        if args.completion_mode != 'boxes':
            assumptions += [('camera_height_prior_m', args.camera_height), ('room_height_prior_m', args.room_height)]
        for key, value in assumptions:
            if not math.isclose(previous[key], value, abs_tol=1e-9):
                parser.error('Saved completion has a different camera/room assumption: '+key)
        cached_completion = read(args.reuse_completion)
        compiled = validate_boxes(cached_completion) if args.completion_mode == 'boxes' else directional_layout(cached_completion, args.camera_height)[0]
        scene_objects(compiled, reused, args.max_additions)
    cached_plan, cached_layout = None, None
    if args.continue_from:
        previous = read(args.continue_from/'config.json')
        if not math.isclose(previous['hfov_deg'], args.hfov, abs_tol=1e-9):
            parser.error('Continuation FOV differs from saved run')
        cached_plan = read(args.continue_from/'semantic_plan.json')
        plan = scene_objects(cached_plan, reused, args.max_additions)
        if args.postprocess_only:
            cached_layout = read(args.continue_from/'coarse_layout.json')
            camera_layout(cached_layout, plan['objects'], args.hfov, image.size, args.camera_height, args.room_height)
    if args.preflight:
        print(f'PREFLIGHT PASSED: image={image.size}, reused={len(reused) if reused is not None else 0}, numeric_state_reused={fixed_observed_state is not None}, cached_objects_only={cached_plan is not None}, content_plan_reused={cached_contents is not None}, training_layout_examples={len(layout_examples["examples"]) if layout_examples else 0}; no model loaded')
        return
    output = new_output(args.output)
    image.save(output/'local.png')
    write(output/'config.json', dict(protocol=ANCHORED_PROTOCOL if args.completion_mode == 'anchored' else PROGRAM_PROTOCOL if args.completion_mode == 'program' else BOX_PROMPT_VERSION if args.completion_mode == 'boxes' else 'qwen_content_layout_v2' if args.completion_mode == 'planned' else 'qwen_directional_completion_v7', completion_mode=args.completion_mode, image=str(args.image.resolve()),
          image_sha256=sha(args.image), model=str(args.model.resolve()), hfov_deg=args.hfov,
          model_image_size=list(image.size), seed=args.seed, max_observed=args.max_observed,
          max_additions=args.max_additions, vocabulary_sha256=vocab['category_mapping_sha256'],
          reuse_observation=str(reuse_path.resolve()) if reuse_path else None,
          reuse_observation_sha256=sha(reuse_path) if reuse_path else None,
          numeric_observation_reused=fixed_observed_state is not None,
          observer_geometry_protocol=source_config.get('observer_geometry_protocol', source_config.get('protocol', 'unknown_saved_observer')) if fixed_observed_state is not None else 'forward_z_projected_full_center_camera_AABB_v1',
          observed_numeric_methods=sorted({o['center_local_m']['method'] for o in fixed_observed_state['objects']}) if fixed_observed_state is not None else [],
          layout_examples=str(args.layout_examples.resolve()) if args.layout_examples else None,
          layout_examples_sha256=sha(args.layout_examples) if args.layout_examples else None,
          training_gt_examples_used=bool(layout_examples and layout_examples['examples']), target_scene_gt_used=False,
          completion_prompt_version=ANCHORED_PROTOCOL if args.completion_mode == 'anchored' else PROGRAM_PROTOCOL if args.completion_mode == 'program' else BOX_PROMPT_VERSION if args.completion_mode == 'boxes' else contents_prompt_version if args.completion_mode == 'planned' else 'camera_azimuth_range_bottom_height_v2',
          reused_completion=str(args.reuse_completion.resolve()) if args.reuse_completion else None,
          reused_completion_sha256=sha(args.reuse_completion) if args.reuse_completion else None,
          reused_contents=str(args.reuse_contents.resolve()) if args.reuse_contents else None,
          reused_contents_sha256=sha(args.reuse_contents) if args.reuse_contents else None,
          continued_from=str(args.continue_from.resolve()) if args.continue_from else None,
          reused_plan_sha256=sha(args.continue_from/'semantic_plan.json') if args.continue_from else None,
          reused_layout_sha256=sha(args.continue_from/'coarse_layout.json') if args.postprocess_only else None,
          postprocess_only=args.postprocess_only,
          camera_height_prior_m=None if args.completion_mode == 'anchored' else args.camera_height,
          room_height_prior_m=None if args.completion_mode == 'anchored' else args.room_height,
          camera_geometry=deepcopy(fixed_observed_state.get('camera_geometry',{})) if args.completion_mode == 'anchored' else camera_geometry(args.up_local,args.camera_height),
          observe_only=args.observe_only,
          inventory_only=args.inventory_only,
          scene_gt_used=False, profile='official', generated_relation_graph_used=False,
          optimizer='disabled',
          geometry_assumptions='fixed camera axes; no floor, room-boundary or gravity prior used for placement' if args.completion_mode == 'anchored' else 'camera axes not levelled; working gravity/floor are unverified priors; no GT at inference'))
    if args.continue_from:
        for stem in (('semantic_plan', 'coarse_layout') if args.postprocess_only else ('semantic_plan',)):
            for suffix in ('.txt', '.json', '.request.json', '.generation.json'):
                source = args.continue_from/(stem+suffix)
                if source.is_file():
                    shutil.copy2(source, output/source.name)
    if args.reuse_contents:
        for suffix in ('.txt', '.json', '.request.json', '.generation.json'):
            source = args.reuse_contents.parent/('room_contents'+suffix)
            if source.is_file():
                shutil.copy2(source, output/source.name)
    try:
        model = None if args.postprocess_only else Qwen(args.model)
    except Exception as exc:
        write(output/'summary.json', dict(status='failed', phase='model_loading', error=str(exc)))
        raise
    run(model, image, args.hfov, output, vocab['category_mapping_sha256'],
        args.max_observed, args.max_additions, args.seed, reused, cached_plan,
        cached_layout, args.camera_height, args.room_height, fixed_observed_state, cached_completion, args.completion_mode,args.up_local,args.observe_only,args.inventory_only,cached_contents,layout_examples)


if __name__ == '__main__':
    main()
