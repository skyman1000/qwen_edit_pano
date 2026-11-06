"""Small ID-based relation contract; not a free-form relation-graph optimizer."""
import math
import numpy as np

from .scene_geometry import half_extent

TYPES = ('near', 'left_of', 'right_of', 'in_front_of', 'behind', 'above', 'below', 'on', 'face_to')


def validate_relations(contents, observed):
    relations = contents.get('relations', [])
    if not isinstance(relations, list):
        raise ValueError('relations must be a list of ID references')
    observed_ids = {o['id'] for o in observed}
    new_ids = {o['id'] for o in contents['additions']}
    known = observed_ids | new_ids
    seen, parents = set(), {}
    for rel in relations:
        if not isinstance(rel, dict) or any(not isinstance(rel.get(k), str) for k in ('subject', 'type', 'target')):
            raise ValueError('Relation needs subject/type/target strings')
        a, kind, b = rel['subject'], rel['type'], rel['target']
        if kind not in TYPES:
            raise ValueError(f'Unsupported relation type {kind!r}; allowed: {", ".join(TYPES)}')
        if a not in new_ids:
            raise ValueError(f'Relation subject must reference a new object ID: {a!r}')
        if b not in known:
            raise ValueError(f'Relation needs an existing/planned target ID: {b!r}')
        if a == b:
            raise ValueError(f'Relation subject and target must be different IDs: {a!r}')
        key = (a, kind, b)
        if key in seen:
            continue  # Harmless duplicate assertions are not an inference failure.
        seen.add(key)
        if kind == 'near':
            gap = rel.get('max_gap_m', 1.)
            if type(gap) not in (int, float) or not math.isfinite(gap) or gap <= 0:
                raise ValueError('near max_gap_m must be finite and positive')
        if kind == 'on':
            if a in parents and parents[a] != b:
                raise ValueError('One supported object must reference one support parent')
            parents[a] = b
    # Only a dependency cycle in support makes candidate construction impossible.
    for start in parents:
        path = set(); node = start
        while node in parents:
            if node in path:
                raise ValueError('Support dependency cycle')
            path.add(node); node = parents[node]
    return relations


def ordered_objects(contents):
    items = contents['additions']
    parents = {r['subject']:r['target'] for r in contents.get('relations', []) if r['type']=='on'}
    remaining = {o['id']:o for o in items}; result=[]
    while remaining:
        ready = [o for o in remaining.values() if parents.get(o['id']) not in remaining]
        if not ready:
            raise ValueError('Support dependency cycle')
        for obj in ready:
            result.append(obj); del remaining[obj['id']]
    return result


def supported_pair(a, b, relations):
    return any(r['type']=='on' and {r['subject'], r['target']} == {a,b} for r in relations)


def check_relation(rel, boxes, geometry):
    if rel['subject'] not in boxes or rel['target'] not in boxes:
        return dict(**rel, status='pending')
    a, b = boxes[rel['subject']], boxes[rel['target']]
    ca, cb = np.array(a['center']), np.array(b['center'])
    delta=ca-cb; kind=rel['type']; value=None
    if kind=='near':
        gap = np.maximum(0, np.abs(delta)-(np.array(a['size'])+b['size'])/2)
        value=float(np.linalg.norm(gap)); passed=value <= rel.get('max_gap_m',1.)
    elif kind in ('left_of','right_of','in_front_of','behind','above','below'):
        axis=0 if kind in ('left_of','right_of') else 1 if kind in ('above','below') else 2
        value=float(delta[axis]); passed=value < 0 if kind in ('left_of','in_front_of','below') else value > 0
    elif kind=='on':
        # A measured/estimated support plane offset is required. Never equate
        # the parent's whole AABB top with the seat/table surface.
        offset=b.get('support_surface_offset_m')
        if offset is None:
            return dict(**rel, status='unverified', reason='support_surface_unknown')
        up=np.array(geometry['up_local'])
        vertical=float(np.dot(delta,up)-offset-half_extent(a['size'],up))
        horizontal=delta-up*np.dot(delta,up)
        value=float(np.linalg.norm(horizontal)+abs(vertical)); passed=value <= .05
    else:
        return dict(**rel, status='unverified', reason='object_orientation_not_in_current_AABB_state')
    return dict(**rel, status='satisfied' if passed else 'violated', value_m=value)


def relation_report(contents, boxes, geometry):
    measured=[check_relation(r,boxes,geometry) for r in contents.get('relations',[])]
    return dict(protocol='explicit_id_relations_v1', relations=measured,
                satisfied=sum(r['status']=='satisfied' for r in measured),
                violated=sum(r['status']=='violated' for r in measured),
                unverified=sum(r['status'] in ('pending','unverified') for r in measured),
                note='Directions are camera-relative; near uses AABB surface gap; '
                     'support offsets/up are hypotheses unless separately measured. '
                     'face_to cannot be certified from camera-axis AABBs.')
