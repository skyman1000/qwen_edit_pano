"""Finite conditional structural scene programs, not independent class counts.

C is inferred by a frozen visual inverse model. Given C and frozen observation
metadata, finite group-mode mechanisms jointly select instances/variants/sizes/
hosts/relative offsets. Explicit uniform noises implement the categorical SCM.
This is a conditional hypothesis model, NOT an identified RGB likelihood model.
"""
from copy import deepcopy
import itertools
import math
import re

import numpy as np

from ..common import finite_vector

PROTOCOL = 'conditional_joint_scene_program_v1'
KEY = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,39}\Z')
SUPPORTS = {'floor', 'object', 'unspecified', 'wall', 'ceiling'}


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' needs a nonempty string')
    return value.strip()


def registry_answer(value, case, configurations):
    if not isinstance(value, dict): raise ValueError('Need observed registry and configurations')
    rows, configs = value.get('instances'), value.get('configurations')
    expected = {o['id'] for o in case['context']}
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
            or len(rows) != len(expected) or {r.get('id') for r in rows} != expected):
        raise ValueError('Observed registry must cover every supplied ID exactly once')
    normalized = []
    for row in rows:
        aff = row.get('affordances', [])
        if not isinstance(aff, list) or any(a not in ('top','seat') for a in aff):
            raise ValueError('affordances must be a subset of [top,seat]; [] allowed')
        if type(row.get('on_floor')) is not bool: raise ValueError('on_floor must be boolean')
        normalized.append(dict(id=row['id'], variant=text(row.get('variant'),'observed variant'),
                               affordances=sorted(set(aff)), on_floor=row['on_floor']))
    if not isinstance(configs, list) or not 1 <= len(configs) <= configurations:
        raise ValueError('Configuration list outside requested computational budget')
    out = []
    for i, config in enumerate(configs):
        if not isinstance(config, dict): raise ValueError('Configuration must be an object')
        refs = config.get('evidence_ids', [])
        if not isinstance(refs, list) or any(r not in expected for r in refs):
            raise ValueError('Configuration evidence_ids must refer to supplied observations')
        out.append(dict(id=f'c{i}', description=text(config.get('description'),'configuration'),
                        explanation=text(config.get('explanation'),'configuration explanation'), evidence_ids=refs))
    if len({r['description'] for r in out}) != len(out): raise ValueError('Duplicate configurations')
    return dict(instances=normalized, configurations=out)


def dependency_order(objects, observed_ids):
    bykey = {o['key']: o for o in objects}; pending = set(bykey); order = []
    dependencies = {}
    for key, obj in bykey.items():
        targets = [obj['anchor'], obj['support']['target']]
        if any(t == key for t in targets): raise ValueError('Object cannot refer to itself')
        if any(t is not None and t != 'camera' and t not in bykey and t not in observed_ids for t in targets):
            raise ValueError('Unknown anchor/host; use a local object key or supplied observed ID')
        dependencies[key] = {t for t in targets if t in bykey}
    while pending:
        ready = sorted(k for k in pending if not (dependencies[k] & pending))
        if not ready: raise ValueError('Cycle in combined placement/support mechanism')
        order.extend(ready); pending.difference_update(ready)
    return order


def program_answer(value, case, config, max_groups=3, max_modes=2, max_new=24):
    if not isinstance(value, dict) or not isinstance(value.get('groups'), list):
        raise ValueError('Need groups array')
    if not 1 <= len(value['groups']) <= max_groups: raise ValueError('Group count outside computational budget')
    observed_ids = {o['id'] for o in case['context']}; groups = []; maximum = 0
    for gi, group in enumerate(value['groups']):
        if not isinstance(group, dict): raise ValueError('Each group needs name, parents and modes')
        parents = group.get('parents', [])
        if (not isinstance(parents, list) or len(set(parents)) != len(parents)
                or any(type(p) is not int or not 0 <= p < gi for p in parents)):
            raise ValueError('parents are distinct zero-based indices of EARLIER groups')
        modes = group.get('modes')
        if not isinstance(modes, list) or not 1 <= len(modes) <= max_modes:
            raise ValueError('Need 1..max_modes nonempty joint alternatives per group')
        normalized = [dict(id=0, description='No additional off-camera instance from this group; existing observations retained.', objects=[], order=[])]
        for mi, mode in enumerate(modes, 1):
            if not isinstance(mode, dict) or not isinstance(mode.get('objects'), list) or not mode['objects']:
                raise ValueError('Model modes must be nonempty; the program inserts the empty alternative')
            if len(mode['objects']) > max_new: raise ValueError('Mode exceeds max_new computational budget')
            objects = []
            for obj in mode['objects']:
                if not isinstance(obj, dict): raise ValueError('Object must be a JSON record')
                key = obj.get('key')
                if not isinstance(key, str) or not KEY.fullmatch(key) or key in observed_ids or key == 'camera':
                    raise ValueError('Local object key must be a short identifier distinct from observations/camera')
                size, offset = obj.get('size_m'), obj.get('offset_m')
                if not finite_vector(size,3) or min(size) <= 0 or not finite_vector(offset,3):
                    raise ValueError('Need positive FULL camera-axis size_m and finite camera-axis offset_m')
                support = obj.get('support')
                if not isinstance(support, dict) or support.get('kind') not in SUPPORTS:
                    raise ValueError('Need support.kind in floor/object/unspecified/wall/ceiling')
                target = support.get('target'); kind = support['kind']; anchor = obj.get('anchor')
                if not isinstance(anchor, str): raise ValueError('anchor must be camera, observed ID or a local key')
                if kind == 'object':
                    if not isinstance(target, str) or target != anchor:
                        raise ValueError('For object support, anchor must equal the supporting target')
                elif target is not None: raise ValueError('Non-object support target must be null')
                objects.append(dict(key=key, name=text(obj.get('name'),'object name'),
                    category=text(obj.get('category'),'ordinary category noun'),
                    variant=text(obj.get('variant'),'specific object variant'), size_m=list(size),
                    anchor=anchor, offset_m=list(offset), support=dict(kind=kind,target=target)))
            if len({o['key'] for o in objects}) != len(objects): raise ValueError('Duplicate local object key')
            order = dependency_order(objects, observed_ids)
            normalized.append(dict(id=mi, description=text(mode.get('description'),'joint mode'), objects=objects, order=order))
        maximum += max(len(m['objects']) for m in normalized)
        groups.append(dict(id=f'g{gi}', name=text(group.get('name'),'group name'),
            explanation=text(group.get('explanation'),'group explanation'), parents=sorted(parents), modes=normalized))
    if maximum > max_new: raise ValueError('Largest joint program exceeds max_new; do not truncate any objects')
    return dict(protocol=PROTOCOL, configuration=deepcopy(config), groups=groups,
                maximum_new_instances=maximum, conditional_on_frozen_observations=True)


def states(group):
    return [dict(id=str(m['id']), description=str(m['id'])) for m in group['modes']]


def table_key(parent_values):
    return ','.join(map(str,parent_values)) if parent_values else 'root'


def parameterize(program, registry, scorer, save):
    """Only declared parents and observation metadata enter a mechanism query.

    Observation metadata is an explicit fixed context parent. These distributions
    are NOT image-independent priors. The mode answer event is an integer + EOS.
    """
    tables = {}
    for gi, group in enumerate(program['groups']):
        table = {}
        domains = [range(len(program['groups'][p]['modes'])) for p in group['parents']]
        for pv in itertools.product(*domains):
            parents = dict(configuration=program['configuration']['description'],
                observed_context=registry['instances'], group=group['name'],
                alternatives=[dict(index=m['id'], joint_recipe=m) for m in group['modes']],
                parent_groups={program['groups'][p]['id']:dict(name=program['groups'][p]['name'],
                               mode=program['groups'][p]['modes'][v]['description'])
                               for p,v in zip(group['parents'],pv)})
            record = scorer.score('Choose the integer index of one plausible EXISTING off-camera group recipe. '
                'The zero alternative is allowed. Existing observed objects can already satisfy roles. '
                'Do not choose extra equipment merely because it would be useful.', parents, states(group))
            distribution = [record['probabilities'][str(i)] for i in range(len(group['modes']))]
            validate_distribution(distribution)
            table[table_key(pv)] = distribution; save(f'{gi}_{table_key(pv)}', record)
        tables[group['id']] = table
    return tables


def validate_distribution(probabilities):
    if (not probabilities or any(type(p) not in (int,float) or not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities),1.,abs_tol=1e-8)):
        raise ValueError('Mechanism must be finite, nonnegative and normalized')


def enumerate_assignments(program, tables, intervention=None):
    intervention = intervention or {}
    groups = program['groups']
    if any(type(i) is not int or not 0 <= i < len(groups) or type(v) is not int
           or not 0 <= v < len(groups[i]['modes']) for i,v in intervention.items()):
        raise ValueError('Intervention outside finite mode domain')
    rows = []
    for values in itertools.product(*(range(len(g['modes'])) for g in groups)):
        probability = 1.
        for gi, group in enumerate(groups):
            if gi in intervention:
                probability *= float(values[gi] == intervention[gi])
            else:
                distribution = tables[group['id']][table_key([values[p] for p in group['parents']])]
                validate_distribution(distribution)
                if len(distribution) != len(group['modes']): raise ValueError('Incomplete mode CPT')
                probability *= distribution[values[gi]]
        rows.append(dict(modes=list(values), probability=probability))
    if not math.isclose(sum(r['probability'] for r in rows),1.,abs_tol=1e-8):
        raise ValueError('Joint conditional mechanism did not normalize')
    return rows


def sample_modes(program, tables, seed, intervention=None):
    """Replayable inverse-CDF structural equations with independent U_g."""
    rng = np.random.default_rng(seed); noises = rng.random(len(program['groups'])); selected = []
    intervention = intervention or {}
    # Validate intervention domain through the same finite model contract.
    if intervention: enumerate_assignments(program,tables,intervention)
    for gi, group in enumerate(program['groups']):
        ps = tables[group['id']][table_key([selected[p] for p in group['parents']])]
        validate_distribution(ps)
        selected.append(intervention[gi] if gi in intervention else
            min(int(np.searchsorted(np.cumsum(ps),noises[gi],side='right')),len(ps)-1))
    # JSON object keys are strings. Keep the in-memory replay record canonical
    # as well, instead of changing key types only when written/read from disk.
    return dict(modes=selected,noise=noises.tolist(),seed=seed,
                intervention={str(k):v for k,v in intervention.items()})


def instantiate(program, modes, case, observed, plane):
    if len(modes) != len(program['groups']): raise ValueError('Need one mode per group')
    old = {o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    objects = []; bodies = {}; placements = {}; groups = {}; order = []; unresolved = []
    for gi,(group,mi) in enumerate(zip(program['groups'],modes)):
        if type(mi) is not int or not 0 <= mi < len(group['modes']): raise ValueError('Unknown group mode')
        mode = group['modes'][mi]; keys = {o['key']:f'sp_g{gi:02d}_{o["key"]}' for o in mode['objects']}
        def target(key): return keys.get(key,key)
        bykey = {o['key']:o for o in mode['objects']}
        for key in mode['order']:
            obj = bykey[key]; oid = keys[key]; anchor = target(obj['anchor']); support = deepcopy(obj['support'])
            if support['target'] is not None: support['target'] = target(support['target'])
            ref = [0.,0.,0.] if anchor == 'camera' else (old[anchor]['center'] if anchor in old else placements[anchor])
            center = [a+b for a,b in zip(ref,obj['offset_m'])]; size = deepcopy(obj['size_m'])
            if support['kind'] == 'floor':
                if plane['floor_y_m'] is None:
                    support = dict(kind='unspecified',target=None); unresolved.append(oid)
                else: center[1] = plane['floor_y_m']+size[1]/2
            elif support['kind'] == 'object':
                host = old.get(support['target'])
                hs = host['size'] if host else bodies[support['target']]['size']
                hc = host['center'] if host else placements[support['target']]
                center[1] = hc[1]+(hs[1]+size[1])/2
            else: unresolved.append(oid)
            placements[oid] = center; groups[oid] = group['id']; order.append(oid)
            bodies[oid] = dict(size=size, center_y_m=center[1], reason=obj['variant'])
            objects.append(dict(id=oid,name=obj['name'],object_type=obj['category'],variant=obj['variant'],
                reason=group['explanation']+' / '+mode['description'],
                placement='Original camera off-frustum; executable relative scene program.',support=support))
            bodies[oid]['anchor'] = anchor; bodies[oid]['offset_m'] = deepcopy(obj['offset_m'])
    plan = dict(id=program['configuration']['id']+'_'+('_'.join(map(str,modes))),
        scene_type='conditional existing world hypothesis',scene_hypothesis=program['configuration']['description'],
        additions=objects,placement_dependency_order=order,
        relations=[dict(source=o['id'],target=o['support']['target'],kind='supported_by',
                        reason='Declared support-position mechanism, not proof of existence causality.')
                   for o in objects if o['support']['kind']=='object'])
    return plan,bodies,groups,placements,unresolved
