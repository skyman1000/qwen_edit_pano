"""Small CPU box-layout solver. Fixed full sizes; camera-frame positions only.

    Projection remains in camera axes. The legacy room envelope is a prior,
    not a measured gravity-aligned room. No weights, GT, meshes or downloads.
"""
from copy import deepcopy
import itertools
import math

import numpy as np
from scipy.optimize import least_squares

from ..common import finite_vector

RELATIONS = ('left_of', 'right_of', 'in_front_of', 'behind', 'above', 'below', 'near', 'on_top_of')
SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def directional_layout(answer, camera_height):
    """Compile a coarse camera-relative hypothesis, without repairing old xyz.

    Continuous azimuth removes the old three-ray restriction. Legacy region
    responses remain readable, with their fallback recorded explicitly.
    This does not reconstruct room walls, object support or occlusion.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get('additions'), list):
        raise ValueError('Directional completion requires an additions list')
    if not math.isfinite(camera_height) or camera_height <= 0:
        raise ValueError('Positive camera height prior required')
    directions = {'camera-left': (-1., 0.), 'camera-right': (1., 0.), 'behind-camera': (0., -1.)}
    converted = deepcopy(answer)
    ignored_centers = []
    legacy_ids = []
    for obj in converted['additions']:
        if not isinstance(obj, dict):
            raise ValueError('Addition must be an object')
        if 'azimuth_deg' in obj:
            angle = obj['azimuth_deg']
            if type(angle) not in (int, float) or not math.isfinite(angle) or not -180 <= angle <= 180:
                raise ValueError('azimuth_deg must be finite in [-180, 180]')
            dx, dz = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        else:
            region = obj.get('camera_region')
            if not isinstance(region, str) or region not in directions:
                raise ValueError('Require azimuth_deg or a legacy camera_region')
            dx, dz = directions[region]
            legacy_ids.append(obj.get('id'))
        distance, bottom, size = obj.get('distance_m'), obj.get('bottom_height_m'), obj.get('size')
        if type(distance) not in (int, float) or not math.isfinite(distance) or distance <= 0:
            raise ValueError('Positive finite distance_m required')
        if type(bottom) not in (int, float) or not math.isfinite(bottom) or bottom < 0:
            raise ValueError('Finite nonnegative bottom_height_m required; 0 means floor-standing')
        if not finite_vector(size, 3) or min(size) <= 0:
            raise ValueError('Positive FULL xyz sizes required')
        if obj.get('unseen_reason', 'outside_fov') != 'outside_fov':
            raise ValueError('This directional baseline plans outside-photo objects, not front-view occlusion')
        if 'center' in obj:
            ignored_centers.append(obj.get('id'))
        obj['center'] = [dx*distance, bottom+size[1]/2-camera_height, dz*distance]
        obj['unseen_reason'] = 'outside_fov'
        obj['placement'] = f"azimuth {math.degrees(math.atan2(dx, dz)):g}deg; horizontal distance {distance:g}m"
    report = dict(protocol='camera_azimuth_range_bottom_height_v2',
                  legacy_region_ids=legacy_ids,
                  center_xz='[distance*sin(azimuth), distance*cos(azimuth)]',
                  region_bearings_deg={'camera-left': -90, 'camera-right': 90, 'behind-camera': 180},
                  camera_height_prior_m=camera_height,
                  center_y='bottom_height_m + full_size_y/2 - camera_height_prior_m',
                  model_center_fields_ignored=ignored_centers,
                  note='New coarse directional hypothesis, not an xyz repair or LayoutGPT reproduction; '
                       'large boxes can still cross FOV and are checked without moving them')
    return converted, report


def camera_layout(answer, objects, hfov, image_size, camera_height=1.5, room_height=2.7):
    """Convert camera-forward z and projected-center rays to camera coordinates.

    Reusing old responses is explicit: visible center.z is a depth hypothesis,
    its x/y and the model's room coordinates are not used. This is a new coarse
    initialization, NOT a proven conversion of arbitrary model coordinates.
    No level camera is required for pinhole projection. Envelope floor/ceiling
    numbers below are legacy camera-axis plotting priors, not measured planes.
    """
    if not (math.isfinite(camera_height) and math.isfinite(room_height)
            and 0 < camera_height < room_height):
        raise ValueError('Require 0 < hypothesized camera height < hypothesized room height')
    # The outer JSON wrapper is not geometry. Accept the two unambiguous
    # representations, then apply the same instance and numeric checks.
    items = answer if isinstance(answer, list) else answer.get('objects') if isinstance(answer, dict) else None
    if not isinstance(items, list):
        raise ValueError('Layout response must be an object array or a dictionary with an objects array')
    if len(items) != len(objects):
        raise ValueError(f'Layout returned {len(items)} entries for {len(objects)} planned instances; every ID is required once')
    ids = [obj['id'] for obj in objects]
    if any(not isinstance(obj, dict) or not isinstance(obj.get('id'), str) for obj in items):
        raise ValueError('Layout object requires an ID')
    if len({obj['id'] for obj in items}) != len(items) or {obj['id'] for obj in items} != set(ids):
        raise ValueError('Layout IDs differ from plan; no objects added or dropped')
    by_id = {obj['id']: obj for obj in items}
    tan_h = math.tan(math.radians(hfov)/2)
    tan_v = tan_h*image_size[1]/image_size[0]
    boxes, reused_depths, explicit_center_ids, visible_proxy_ids = [], [], [], []
    for obj in objects:
        item = by_id[obj['id']]
        size = item.get('size')
        if not finite_vector(size, 3) or min(size) <= 0:
            raise ValueError('Positive FULL xyz sizes required: '+obj['id'])
        if obj['role'] == 'observed':
            if 'depth_m' in item:
                depth = item['depth_m']
            else:
                if not finite_vector(item.get('center'), 3):
                    raise ValueError('Observed object requires depth_m: '+obj['id'])
                depth = item['center'][2]
                reused_depths.append(obj['id'])
            if type(depth) not in (int, float) or not math.isfinite(depth) or depth <= 0:
                raise ValueError('Observed depth must be positive and finite: '+obj['id'])
            if 'center_uv' in item:
                if not finite_vector(item['center_uv'],2):
                    raise ValueError('center_uv must be the projected FULL 3D box center, normalized uv: '+obj['id'])
                u,v=item['center_uv']; explicit_center_ids.append(obj['id'])
            else:
                x1, y1, x2, y2 = obj['bbox_2d']
                u, v = (x1+x2)/2, (y1+y2)/2
                visible_proxy_ids.append(obj['id'])
            center = [(u-.5)*2*tan_h*depth, (.5-v)*2*tan_v*depth, depth]
        else:
            if not finite_vector(item.get('center'), 3):
                raise ValueError('Unseen hypothesis requires center xyz: '+obj['id'])
            center = item['center']
        boxes.append(dict(id=obj['id'], center=center, size=size, support=item.get('support')))
    centers = np.array([obj['center'] for obj in boxes])
    sizes = np.array([obj['size'] for obj in boxes])
    # This is a working room envelope, not recovered architecture. Keep >=2m
    # of unobserved space around the camera and enclose proposed boxes in x/z.
    lo = np.minimum(-2., (centers-sizes/2).min(0)[[0, 2]]-.5)
    hi = np.maximum(2., (centers+sizes/2).max(0)[[0, 2]]+.5)
    room = dict(x_min=float(lo[0]), x_max=float(hi[0]), floor_y=-camera_height,
                ceiling_y=room_height-camera_height, z_min=float(lo[1]), z_max=float(hi[1]))
    layout = validate_layout(dict(room=room, objects=boxes), objects)
    report = dict(observed_centers='bbox_center_ray_times_predicted_depth; approximate_visible_center',
                  unseen_centers='unverified_model_hypotheses', sizes='unverified_model_full_AABB_sizes',
                  reused_legacy_depth_ids=reused_depths,
                  explicit_projected_center_ids=explicit_center_ids,
                  visible_bbox_center_proxy_ids=visible_proxy_ids,
                  model_room_coordinates_used=False,
                  proposed_model_room=answer.get('room') if isinstance(answer, dict) else None,
                  response_container='array' if isinstance(answer, list) else 'objects_wrapper',
                  input_object_count=len(items),
                  camera_height_m=camera_height, room_height_m=room_height,
                  room_method='hypothesis_envelope_enclosing_boxes_plus_0.5m; minimum_2m_each_horizontal_direction',
                  assumptions='camera axes not levelled; supplied FOV; envelope only; heights are priors, not measurements',
                  quality_status='requires_visual_review')
    return layout, report


def validate_layout(answer, objects):
    if not isinstance(answer, dict) or not isinstance(answer.get('room'), dict):
        raise ValueError('Layout needs room and objects')
    room = answer['room']
    for key in ('x_min', 'x_max', 'floor_y', 'ceiling_y', 'z_min', 'z_max'):
        if type(room.get(key)) not in (float, int) or not math.isfinite(room[key]):
            raise ValueError('Invalid room scalar: '+key)
    if not (room['x_min'] < 0 < room['x_max'] and room['z_min'] < 0 < room['z_max']
            and room['floor_y'] < 0 < room['ceiling_y']):
        raise ValueError('Hypothesized room must contain camera origin with floor below it')
    items = answer.get('objects')
    if not isinstance(items, list) or len(items) != len(objects):
        raise ValueError('Layout must place every planned instance exactly once')
    ids = [o['id'] for o in objects]
    if any(not isinstance(o, dict) or not isinstance(o.get('id'), str) for o in items):
        raise ValueError('Layout object requires an ID')
    if len({o['id'] for o in items}) != len(items) or {o['id'] for o in items} != set(ids):
        raise ValueError('Layout IDs differ from plan; no objects added or dropped')
    by_id = {o['id']: o for o in items}
    ordered = []
    for oid in ids:
        obj = by_id[oid]
        if not finite_vector(obj.get('center'), 3) or not finite_vector(obj.get('size'), 3) or min(obj['size']) <= 0:
            raise ValueError('Need finite center and positive FULL xyz size: '+oid)
        # Support is auxiliary; retain malformed/unknown suggestions for reporting.
        ordered.append(dict(id=oid, center=obj['center'], size=obj['size'], support=obj.get('support')))
    return dict(room={k: room[k] for k in ('x_min', 'x_max', 'floor_y', 'ceiling_y', 'z_min', 'z_max')}, objects=ordered)


def support_constraints(layout):
    """Use unambiguous per-object support suggestions, report unresolved ones.

    No model graph correction, direction reversal, or automatic parent choice.
    Invalid optional support leaves position unconstrained by support, not valid.
    """
    objects = {obj['id']: obj for obj in layout['objects']}
    parents, warnings = {}, []
    for oid, obj in objects.items():
        parent = obj['support']
        if not isinstance(parent, str):
            warnings.append(dict(id=oid, proposed_support=parent, issue='unresolved_support_suggestion'))
            continue
        if parent in ('floor', 'wall', 'ceiling', 'free'):
            continue
        if parent not in objects or parent == oid:
            warnings.append(dict(id=oid, proposed_support=parent, issue='unresolved_support_suggestion'))
        else:
            parents[oid] = parent
    cyclic = set()
    for oid in parents:
        path, node = [], oid
        while node in parents and node not in path:
            path.append(node)
            node = parents[node]
        if node in path:
            cyclic.update(path[path.index(node):])
    for oid in sorted(cyclic):
        warnings.append(dict(id=oid, proposed_support=parents[oid], issue='cyclic_support_suggestion_not_applied'))
    relations = [dict(a=oid, type='on_top_of', b=parent)
                 for oid, parent in parents.items() if oid not in cyclic]
    return relations, warnings


def frustum_margin(center, size, tan_h, tan_v):
    """Smallest whole-AABB separation from a frustum half-space (metres).

    <=0 certifies outside this camera's ideal viewing cone. Positive values do
    not prove visibility, since occlusion is not modeled here.
    """
    x, y, z = center
    sx, sy, sz = size/2
    return min(x+tan_h*z+sx+tan_h*sz, -x+tan_h*z+sx+tan_h*sz,
               y+tan_v*z+sy+tan_v*sz, -y+tan_v*z+sy+tan_v*sz,
               z+sz+.05)


def projected_box(center, size, tan_h, tan_v):
    corners = center+SIGNS*size/2
    if np.min(corners[:, 2]) <= .01:
        return None
    uv = np.column_stack((.5+corners[:, 0]/corners[:, 2]/(2*tan_h),
                          .5-corners[:, 1]/corners[:, 2]/(2*tan_v)))
    return np.r_[uv.min(0), uv.max(0)]


def inspect_layout(layout, objects, hfov, image_size):
    """Read-only image checks, without treating guessed boxes as physics.

    An AABB does not specify support surfaces or occupied mesh volume. The room
    envelope is also a hypothesis. None of these are grounds to move evidence.
    """
    tan_h = math.tan(math.radians(hfov)/2)
    tan_v = tan_h*image_size[1]/image_size[0]
    by_id = {o['id']: o for o in layout['objects']}
    warnings, measurements = [], []
    for obj in objects:
        box = by_id[obj['id']]
        center, size = np.array(box['center']), np.array(box['size'])
        record = dict(id=obj['id'], role=obj['role'])
        if obj['role'] == 'observed':
            projected = projected_box(center, size, tan_h, tan_v)
            target = np.array(obj['bbox_2d'])
            overlap = None if projected is None else np.minimum(projected[2:], target[2:])-np.maximum(projected[:2], target[:2])
            record['projection_overlaps_evidence'] = bool(overlap is not None and min(overlap) > 0)
            record['projected_box'] = projected.tolist() if projected is not None else None
            # Containment is more informative than any-overlap, but neither
            # certifies metric depth. A partial visible box need not equal a full box.
            intersection = 0. if overlap is None else float(np.prod(np.maximum(overlap, 0)))
            record['visible_bbox_coverage'] = intersection / float(np.prod(target[2:]-target[:2]))
            if not record['projection_overlaps_evidence']:
                warnings.append(dict(id=obj['id'], issue='observed_projection_no_overlap'))
        elif obj['unseen_reason'] == 'outside_fov':
            margin = float(frustum_margin(center, size, tan_h, tan_v))
            record.update(outside_fov_verified=margin <= 0, frustum_margin_m=margin)
            if margin > 0:
                warnings.append(dict(id=obj['id'], issue='outside_fov_not_satisfied', metres=margin))
        else:
            warnings.append(dict(id=obj['id'], issue='occlusion_not_verified'))
        measurements.append(record)
    return dict(warnings=warnings, objects=measurements,
                note='Read-only projection checks; not semantic, metric-depth, support or collision certification')


class BoxProblem:
    def __init__(self, layout, objects, relations, hfov, image_size):
        self.layout, self.objects, self.relations = layout, objects, relations
        self.initial = np.array([o['center'] for o in layout['objects']], dtype=float)
        self.sizes = np.array([o['size'] for o in layout['objects']], dtype=float)
        self.index = {o['id']: i for i, o in enumerate(objects)}
        room = layout['room']
        self.lo = np.array([room['x_min'], room['floor_y'], room['z_min']])
        self.hi = np.array([room['x_max'], room['ceiling_y'], room['z_max']])
        self.tan_h = math.tan(math.radians(hfov)/2)
        self.tan_v = self.tan_h*image_size[1]/image_size[0]
        self.parents = {self.index[r['a']]: self.index[r['b']] for r in relations if r['type'] == 'on_top_of'}
        pairs = []
        for i, j in itertools.combinations(range(len(objects)), 2):
            # Wall/ceiling fixtures may overlap curtains/windows or their mounts.
            if any(layout['objects'][k]['support'] in ('wall', 'ceiling') for k in (i, j)):
                continue
            if self.is_supported_by(i, j) or self.is_supported_by(j, i):
                continue
            pairs.append((i, j))
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)

    def is_supported_by(self, i, j):
        visited = set()
        while i in self.parents and i not in visited:
            visited.add(i)
            i = self.parents[i]
            if i == j:
                return True
        return False

    def collision_depths(self, centers):
        if not len(self.pairs):
            return np.empty(0)
        i, j = self.pairs.T
        overlap = (self.sizes[i]+self.sizes[j])/2-np.abs(centers[i]-centers[j])
        return np.maximum(0, overlap.min(axis=1))

    def residuals(self, flat, grouped=False):
        p = np.asarray(flat).reshape(-1, 3)
        s = self.sizes
        blocks = dict(prior=(.15*(p-self.initial)).ravel().tolist(),
                      boundary=(3*np.maximum(0, self.lo+s/2-p)).ravel().tolist()
                      +(3*np.maximum(0, p+s/2-self.hi)).ravel().tolist(),
                      support=[], image=[], outside_fov=[], relations=[], collision=[])
        for i, obj in enumerate(self.objects):
            support = self.layout['objects'][i]['support']
            if i in self.parents:
                j = self.parents[i]
                blocks['support'].append(5*(p[i, 1]-p[j, 1]-(s[i, 1]+s[j, 1])/2))
                # Contain the supported object's center in its parent's footprint.
                blocks['support'].extend(3*np.maximum(0, np.abs(p[i, [0, 2]]-p[j, [0, 2]])-s[j, [0, 2]]/2))
            elif support == 'floor':
                blocks['support'].append(5*(p[i, 1]-self.lo[1]-s[i, 1]/2))
            elif support == 'ceiling':
                blocks['support'].append(5*(p[i, 1]-self.hi[1]+s[i, 1]/2))
            elif support == 'wall':
                distances = np.r_[p[i, [0, 2]]-s[i, [0, 2]]/2-self.lo[[0, 2]],
                                  self.hi[[0, 2]]-p[i, [0, 2]]-s[i, [0, 2]]/2]
                blocks['support'].append(2*distances[np.argmin(np.abs(distances))])
            if obj['role'] == 'observed':
                target = np.array(obj['bbox_2d'])
                # Complete-box projection may contain a smaller visible fragment.
                corners = p[i]+SIGNS*s[i]/2
                z = np.maximum(corners[:, 2], .05)
                uv = np.column_stack((.5+corners[:, 0]/z/(2*self.tan_h),
                                      .5-corners[:, 1]/z/(2*self.tan_v)))
                box = np.r_[uv.min(0), uv.max(0)]
                blocks['image'].extend(4*np.maximum(0, np.r_[box[:2]-target[:2], target[2:]-box[2:]]))
                uvcenter = np.array([.5+p[i, 0]/max(p[i, 2], .05)/(2*self.tan_h),
                                     .5-p[i, 1]/max(p[i, 2], .05)/(2*self.tan_v)])
                # Weak center cue, not an equality between full and visible boxes.
                blocks['image'].extend(np.maximum(0, target[:2]-.1-uvcenter))
                blocks['image'].extend(np.maximum(0, uvcenter-target[2:]-.1))
                blocks['image'].append(5*max(0, .05-(p[i, 2]-s[i, 2]/2)))
            elif obj['unseen_reason'] == 'outside_fov':
                blocks['outside_fov'].append(3*max(0, frustum_margin(p[i], s[i], self.tan_h, self.tan_v)))
        for relation in self.relations:
            i, j = self.index[relation['a']], self.index[relation['b']]
            kind = relation['type']
            if kind == 'on_top_of':
                continue
            if kind == 'near':
                gap = np.maximum(0, np.abs(p[i]-p[j])-(s[i]+s[j])/2)
                error = max(0, np.linalg.norm(gap)-1.)
            else:
                axis = 0 if kind in ('left_of', 'right_of') else 1 if kind in ('above', 'below') else 2
                sign = 1 if kind in ('left_of', 'in_front_of', 'below') else -1
                error = max(0, .1+sign*(p[i, axis]-p[j, axis]))
            blocks['relations'].append(2*error)
        blocks['collision'] = (3*self.collision_depths(p)).tolist()
        if grouped:
            return {k: dict(max_abs=float(np.max(np.abs(v))) if len(v) else 0.,
                            squared_sum=float(np.sum(np.square(v)))) for k, v in blocks.items()}
        return np.concatenate([np.asarray(v, dtype=float) for v in blocks.values()])

    def diagnose(self, centers):
        warnings = []
        for i, obj in enumerate(self.objects):
            oid = obj['id']
            violation = max(float(np.max(self.lo+self.sizes[i]/2-centers[i])),
                            float(np.max(centers[i]+self.sizes[i]/2-self.hi)))
            if violation > .1:
                warnings.append(dict(id=oid, issue='outside_hypothesized_room', metres=violation))
            if obj['role'] == 'observed':
                projected = projected_box(centers[i], self.sizes[i], self.tan_h, self.tan_v)
                target = np.array(obj['bbox_2d'])
                overlap = None if projected is None else np.minimum(projected[2:], target[2:])-np.maximum(projected[:2], target[:2])
                if overlap is None or min(overlap) <= 0:
                    warnings.append(dict(id=oid, issue='observed_projection_no_overlap'))
            elif obj['unseen_reason'] == 'outside_fov':
                margin = frustum_margin(centers[i], self.sizes[i], self.tan_h, self.tan_v)
                if margin > .05:
                    warnings.append(dict(id=oid, issue='outside_fov_not_satisfied', metres=float(margin)))
            else:
                warnings.append(dict(id=oid, issue='occlusion_not_verified_by_box_solver'))
        for (i, j), depth in zip(self.pairs, self.collision_depths(centers)):
            if depth > .1:
                warnings.append(dict(ids=[self.objects[i]['id'], self.objects[j]['id']],
                                     issue='box_collision', penetration_metres=float(depth)))
        residuals = self.residuals(centers.ravel(), grouped=True)
        for group in ('support', 'relations', 'image'):
            if residuals[group]['max_abs'] > .5:
                warnings.append(dict(issue='remaining_constraint_residual', group=group,
                                     weighted_max=residuals[group]['max_abs']))
        return dict(weighted_residuals=residuals, warnings=warnings,
                    note='Soft constraints and coarse boxes; no semantic or reconstruction certification')


def refine(layout, objects, relations, hfov, image_size, max_nfev=120):
    problem = BoxProblem(layout, objects, relations, hfov, image_size)
    before = problem.diagnose(problem.initial)
    result = least_squares(problem.residuals, problem.initial.ravel(), max_nfev=max_nfev,
                           ftol=1e-5, xtol=1e-5, gtol=1e-5)
    if not np.isfinite(result.x).all():
        raise ValueError('Layout optimizer returned nonfinite values')
    corrected = deepcopy(layout)
    centers = result.x.reshape(-1, 3)
    for obj, center in zip(corrected['objects'], centers):
        obj['center'] = center.tolist()
    report = dict(before=before, after=problem.diagnose(centers),
                  optimizer=dict(success=bool(result.success), message=str(result.message), nfev=result.nfev),
                  variables='positions_only; sizes_and_room_fixed', quality_status='requires_visual_review')
    return corrected, report
