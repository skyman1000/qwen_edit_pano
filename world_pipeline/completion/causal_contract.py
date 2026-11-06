"""Executable *hypothesized* selection mechanisms, not identified physical causality.

All observed instances are fixed inputs. Boolean factor gates and conditional
instance selection describe one proposed surrounding world. Intervening on a
gate tests this declared selector, not what would disappear in a real room.
"""
from copy import deepcopy
import math

KINDS = {'context', 'activity', 'preference', 'environment'}
RELATIONS = {'left_of', 'right_of', 'above', 'below', 'in_front_of', 'behind',
             'near', 'supported_by', 'on_top_of', 'inside'}


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}: need a nonempty string')
    return value


def keys(value, required, optional, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label}: need an object')
    if set(required) - value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f'{label}: required keys {sorted(required)}, optional {sorted(optional)}; '
                         f'got {sorted(value)}')


def refs(value, allowed, label):
    if (not isinstance(value, list) or any(not isinstance(v, str) for v in value)
            or len(set(value)) != len(value) or set(value) - set(allowed)):
        raise ValueError(f'{label}: need distinct known IDs from {sorted(allowed)}')
    return value


def order(parents):
    """Topological order, rejecting cycles instead of silently cutting edges."""
    pending = {k: set(v) & parents.keys() for k, v in parents.items()}
    result = []
    while pending:
        ready = sorted(k for k, v in pending.items() if not v)
        if not ready:
            raise ValueError('Dependency cycle: '+', '.join(sorted(pending)))
        result.extend(ready)
        for k in ready:
            del pending[k]
        for v in pending.values():
            v.difference_update(ready)
    return result


def validate_factors(answer, observed_ids):
    keys(answer, {'scene_type', 'scene_hypothesis', 'factors'}, set(), 'context')
    text(answer['scene_type'], 'scene_type'); text(answer['scene_hypothesis'], 'scene_hypothesis')
    if not isinstance(answer['factors'], list):
        raise ValueError('factors: need an array')
    result, seen = deepcopy(answer), set(observed_ids)
    for f in result['factors']:
        keys(f, {'id', 'kind', 'statement', 'assumed', 'evidence_ids', 'reason'}, {'parents'}, 'factor')
        fid = text(f['id'], 'factor.id')
        if fid in seen or f['kind'] not in KINDS or type(f['assumed']) is not bool:
            raise ValueError('factor: duplicate ID, invalid kind or non-boolean assumed')
        seen.add(fid)
        text(f['statement'], 'factor.statement'); text(f['reason'], 'factor.reason')
        refs(f['evidence_ids'], observed_ids, fid+'.evidence_ids')
        f.setdefault('parents', [])
    ids = {f['id'] for f in result['factors']}
    for f in result['factors']:
        refs(f['parents'], ids, f['id']+'.parents')
    order({f['id']: f['parents'] for f in result['factors']})
    return result


def validate_candidates(answer, factors, observed_ids, max_candidates=160):
    keys(answer, {'additions', 'relations'}, set(), 'contents')
    if not isinstance(answer['additions'], list) or not isinstance(answer['relations'], list):
        raise ValueError('additions and relations must be arrays')
    if len(answer['additions']) > max_candidates:
        raise ValueError('Candidate safety limit exceeded; no truncation')
    result = deepcopy(answer)
    fids = {f['id'] for f in factors['factors']}
    seen = set(observed_ids) | fids
    for obj in result['additions']:
        keys(obj, {'id', 'name', 'object_type', 'reason', 'placement', 'grounds'}, {'requires'}, 'addition')
        for k in ('id', 'name', 'object_type', 'reason', 'placement'):
            text(obj[k], 'addition.'+k)
        if obj['id'] in seen:
            raise ValueError('New instance IDs must be unique and disjoint from observations/factors')
        seen.add(obj['id'])
        # OR of AND clauses. [[]] explicitly means an unconditional scene prior.
        grounds = obj['grounds']
        if not isinstance(grounds, list) or not grounds:
            raise ValueError(obj['id']+'.grounds: need nonempty OR of AND clauses; [[]] is unconditional')
        for clause in grounds:
            refs(clause, fids, obj['id']+'.grounds clause')
        obj.setdefault('requires', [])
    nids = {o['id'] for o in result['additions']}
    for obj in result['additions']:
        refs(obj['requires'], set(observed_ids) | nids, obj['id']+'.requires')
    order({o['id']: o['requires'] for o in result['additions']})
    seen_edges = set()
    for index, r in enumerate(result['relations']):
        label = f'relations[{index}]'
        keys(r, {'source', 'target', 'kind', 'reason'}, {'max_distance_m'}, 'relation')
        allowed = set(observed_ids) | nids
        for endpoint in ('source', 'target'):
            if not isinstance(r[endpoint], str) or r[endpoint] not in allowed:
                raise ValueError(f'{label}.{endpoint}={r[endpoint]!r}: unknown instance; '
                                 f'allowed IDs={sorted(allowed)}')
        if r['source'] == r['target']:
            raise ValueError(f'{label}: self edge on {r["source"]!r}')
        if not isinstance(r['kind'], str) or r['kind'] not in RELATIONS:
            raise ValueError(f'{label}.kind={r["kind"]!r}: unsupported relation; '
                             f'allowed kinds={sorted(RELATIONS)}')
        if r['source'] not in nids and r['target'] not in nids:
            raise ValueError('Relations in this experiment must involve a new instance')
        text(r['reason'], 'relation.reason')
        signature = (r['source'], r['target'], r['kind'])
        if signature in seen_edges:
            raise ValueError('Duplicate relation')
        seen_edges.add(signature)
        if 'max_distance_m' in r:
            d = r['max_distance_m']
            if r['kind'] != 'near' or type(d) not in (float, int) or not math.isfinite(d) or d <= 0:
                raise ValueError('max_distance_m: positive finite centre-distance threshold for near only')
    # These are placement dependencies, NOT additional existence mechanisms.
    placement_order(result['additions'], result['relations'])
    return result


def placement_order(objects, relations):
    parents = {o['id']: [] for o in objects}
    for r in relations:
        if r['kind'] in ('supported_by', 'on_top_of', 'inside') and r['source'] in parents:
            parents[r['source']].append(r['target'])
    return order(parents)


def execute(factors, contents, observed_ids, interventions=None):
    """Evaluate immutable proposals under optional do(gate=bool) diagnostics.

    An intervention replaces the factor's entire equation, cutting its parents.
    Observations are never intervention targets and never removed.
    """
    interventions = interventions or {}
    by_factor = {f['id']: f for f in factors['factors']}
    if set(interventions) - by_factor.keys() or any(type(v) is not bool for v in interventions.values()):
        raise ValueError('Interventions may only assign booleans to hypothesis factor IDs')
    values, traces = {}, []
    for fid in order({k: f['parents'] for k, f in by_factor.items()}):
        f = by_factor[fid]
        value = (interventions[fid] if fid in interventions else
                 f['assumed'] and all(values[p] for p in f['parents']))
        values[fid] = value
        traces.append(dict(id=fid, active=value, intervened=fid in interventions,
                           assumed_gate=f['assumed'], parents=f['parents']))
    objects = {o['id']: o for o in contents['additions']}
    active, instances = set(observed_ids), []
    for oid in order({k: o['requires'] for k, o in objects.items()}):
        obj = objects[oid]
        passed = [i for i, c in enumerate(obj['grounds']) if all(values[fid] for fid in c)]
        missing = [rid for rid in obj['requires'] if rid not in active]
        selected = bool(passed) and not missing
        if selected:
            active.add(oid)
        instances.append(dict(id=oid, selected=selected, satisfied_ground_clauses=passed,
                              missing_required_instances=missing))
    # Explicit relations with an inactive endpoint are kept in the trace; they
    # cannot silently disappear from an otherwise active object's intended support.
    relations, inactive_relations = [], []
    for r in contents['relations']:
        (relations if {r['source'], r['target']} <= active else inactive_relations).append(deepcopy(r))
    selected_objects = [deepcopy(o) for o in contents['additions'] if o['id'] in active]
    return dict(factor_values=values, factor_trace=traces, instance_trace=instances,
                selected_ids=[o['id'] for o in selected_objects],
                selected_additions=selected_objects, relations=relations,
                inactive_relations=inactive_relations,
                placement_order=placement_order(selected_objects, relations))


def diagnostics(factors, contents, observed_ids, factual):
    results = []
    for f in factors['factors']:
        value = not factual['factor_values'][f['id']]
        alternative = execute(factors, contents, observed_ids, {f['id']: value})
        a, b = set(factual['selected_ids']), set(alternative['selected_ids'])
        results.append(dict(factor=f['id'], do_value=value, removed=sorted(a-b), added=sorted(b-a),
                            resulting_factor_values=alternative['factor_values']))
    return dict(kind='declared_selector_sensitivity_only', frozen_candidate_pool=True,
                causal_effect_estimated=False, causal_gain_validated=False, results=results,
                note='These are consequences of declared Boolean rules, not evidence of real-world '
                     'causal truth, physical object disappearance or better completion.')


def typed_graph(factors, contents, observed_ids, factual):
    nodes = [dict(id=i, type='observed_instance', selected=True) for i in observed_ids]
    nodes += [dict(id=f['id'], type='hypothesized_'+f['kind'], label=f['statement'],
                   selected=factual['factor_values'][f['id']]) for f in factors['factors']]
    nodes += [dict(id=o['id'], type='hypothesized_instance', label=o['name'],
                   selected=o['id'] in factual['selected_ids']) for o in contents['additions']]
    edges = []
    for f in factors['factors']:
        edges += [dict(source=i, target=f['id'], type='evidence_for_not_cause') for i in f['evidence_ids']]
        edges += [dict(source=i, target=f['id'], type='hypothesized_AND_gate') for i in f['parents']]
    for o in contents['additions']:
        for index, clause in enumerate(o['grounds']):
            edges += [dict(source=i, target=o['id'], type='selection_ground', OR_clause=index)
                      for i in clause]
        edges += [dict(source=i, target=o['id'], type='declared_existence_requirement') for i in o['requires']]
    edges += [dict(source=r['source'], target=r['target'], type='spatial_'+r['kind'], reason=r['reason'])
              for r in contents['relations']]
    return dict(nodes=nodes, edges=edges, identified_causal_graph=False,
                equations='factor = assumed_gate AND parents; instance = OR(AND grounds) AND required_instances',
                note='Spatial/evidence edges are not causal edges. Only dependency subgraphs must be acyclic.')
