"""Compact numeric requests, aggregate missing-field errors, strict box proxies."""
from copy import deepcopy
import json
import math

import numpy as np

from ..common import finite_vector
from . import assembly_modular_geometry as base, assembly_parametric as parametric, assembly_layout as groups

PARAMETERIZATION = 'compact_numeric_and_global_box_contracts_v6'


def layout_prompt(case, observed, world, plane):
    prompt = base.layout_prompt(case, observed, world, plane)
    # Rewrite only the JSON template. Required numbers have no apparent default.
    start = prompt.index('TEMPLATE: ')+len('TEMPLATE: '); end = prompt.index('\nFIXED SEMANTIC WORLD:',start)
    template = json.loads(prompt[start:end]); contracts = {o['id']:o for o in world['objects']}
    for group in template['groups']:
        group['range_m'] = None
        for obj in group['objects']:
            obj['yaw_deg'] = None; obj['offset_m'] = [None,None,None]
            kind = contracts[obj['id']]['attachment']['kind']
            if kind not in ('seat','shelf'): obj.pop('surface_fraction',None)
            if kind != 'hang': obj.pop('hanging_drop_m',None)
    return (prompt[:start]+json.dumps(template)+prompt[end:]+
        '\nV6 NUMERIC RESPONSE: Every required null in TEMPLATE must be replaced with a finite number. '
        'Return numbers for every range_m, full_size_m, yaw_deg and all three offset_m components, even zero. '
        'Only unused attachment parameters may be omitted. Return only the JSON object. '
        'Never treat null as a prediction or leave copied placeholders. No count, object identity or host may change.')


def layout_answer(value, world, hfov):
    errors = []; expected = groups.grouped(world); contracts = {o['id']:o for o in world['objects']}
    rows = value.get('groups') if isinstance(value,dict) else None
    if not isinstance(rows,list): raise ValueError('groups must be a list; all required numbers must be filled')
    gids = [g['id'] for g in rows if isinstance(g,dict) and isinstance(g.get('id'),str)]
    if len(gids)!=len(rows) or len(gids)!=len(set(gids)) or set(gids)!=set(expected):
        errors.append('Copy every group ID exactly once: '+str(list(expected)))
    seen = []
    for group in rows:
        if not isinstance(group,dict): continue
        gid = str(group.get('id')); distance = group.get('range_m')
        if group.get('region') not in (None,'unspecified','center','centre','left','right','rear'): errors.append(gid+'.region: left/right/rear/unspecified required')
        if type(distance) not in (int,float) or not math.isfinite(distance) or distance<=0: errors.append(gid+'.range_m: finite positive number required')
        objects = group.get('objects')
        if not isinstance(objects,list): errors.append(gid+'.objects: list required'); continue
        local_ids = [o['id'] for o in objects if isinstance(o,dict) and isinstance(o.get('id'),str)]
        if gid in expected and set(local_ids)!={o['id'] for o in expected[gid]}: errors.append(gid+': wrong object membership')
        for obj in objects:
            if not isinstance(obj,dict): errors.append(gid+': object record required'); continue
            oid = obj.get('id'); prefix = str(oid); seen.append(prefix)
            if not isinstance(oid,str) or oid not in contracts: errors.append(prefix+': unknown ID'); continue
            dims = obj.get('full_size_m')
            if not finite_vector(dims,3) or min(dims)<=0: errors.append(prefix+'.full_size_m: three finite positive FULL dimensions required')
            if not finite_vector(obj.get('offset_m'),3): errors.append(prefix+'.offset_m: all three finite numbers required')
            yaw = obj.get('yaw_deg')
            if type(yaw) not in (int,float) or not math.isfinite(yaw): errors.append(prefix+'.yaw_deg: finite number required')
            kind = contracts[oid]['attachment']['kind']
            if kind in ('seat','shelf'):
                f = obj.get('surface_fraction')
                if type(f) not in (int,float) or not math.isfinite(f) or not 0<=f<=1: errors.append(prefix+'.surface_fraction: number in [0,1] required')
            if kind=='hang':
                d = obj.get('hanging_drop_m')
                if type(d) not in (int,float) or not math.isfinite(d) or d<0: errors.append(prefix+'.hanging_drop_m: nonnegative number required')
    if len(seen)!=len(set(str(i) for i in seen)) or set(str(i) for i in seen)!=set(contracts): errors.append('Copy every object ID exactly once')
    if errors: raise ValueError('All numeric contract errors (repair all, preserve inventory): '+json.dumps(errors))
    return parametric.layout_answer(value,world,hfov)


def compile_layout(case, observed, world, layout, plane, *, repairer=None, refinement_score=None, include_default_preview=False):
    result = base.compile_layout(case,observed,world,layout,plane,repairer=repairer,refinement_score=refinement_score,
        include_default_preview=include_default_preview); review = result['review']
    collisions = review['program_solver']['final_overlaps']
    review.update(geometry_parameterization=PARAMETERIZATION,
        actual_functional_affordance='not_certified_by_AABB_geometry',
        collision_policy='All nonallowed residual box overlaps block completed export; inventory retained in diagnostic candidate')
    if collisions:
        review['geometry_contract_passed'] = False; review['layout_proxy_passed'] = False
        review['hard_violations'] += len(collisions)
    boxes = {o['id']:o for o in result['items']}
    boxes.update({o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']})
    gaps = []
    for relation in world['relations']:
        if relation['kind']!='near': continue
        a,b = (boxes.get(relation[k]) for k in ('source','target'))
        if a is None or b is None: continue
        gap = np.maximum(0,np.abs(np.asarray(a['center'])-np.asarray(b['center']))-(np.asarray(a['size'])+np.asarray(b['size']))/2)
        gaps.append(dict(source=relation['source'],target=relation['target'],AABB_surface_gap_m=float(np.linalg.norm(gap)),
            centre_distance_m=float(np.linalg.norm(np.asarray(a['center'])-np.asarray(b['center']))),
            usability_verified=False,scope='Metric distance report; no universal near threshold or human-reach certification'))
    review['functional_proximity_diagnostics'] = gaps
    return result
