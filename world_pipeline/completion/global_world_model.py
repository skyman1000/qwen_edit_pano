"""Global conditional program SCM; no providers, scene GT, or fitted parameters.

H selects an uncertain envelope/use organization. B selects ONE joint realization
across all uses, with shared instance IDs. E is ownership and S is installation.
The finite support and elicited probabilities are model-relative, not a calibrated
visual posterior. The proposed domain is filtered by explicit executable box
checks before estimation/sampling; all discarded candidates remain in its audit.
"""
from copy import deepcopy
import itertools
import json
import math
import re

import numpy as np

from ..common import FRAME, field, validate_graph
from .geometry import graph, CANONICAL_CATEGORIES
from .category_bridge import category
from .world_scm import forward, validate as validate_scm

PROTOCOL = 'global_conditional_program_scm_v3'
SUPPORTS = ('floor', 'ceiling', 'wall', 'free', 'top', 'inside')


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value.strip()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value, n, name, positive=False):
    if (not isinstance(value, list) or len(value) != n or
            not all(number(v) for v in value) or (positive and min(value) <= 0)):
        raise ValueError(name + ': finite vector of length ' + str(n) + ' required')
    return list(value)


def strings(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(name + ' must be a list of nonempty strings')
    if len(value) != len(set(value)):
        raise ValueError(name + ' contains duplicates')
    return list(value)


def observation_packet(observed, grounding):
    validate_graph(observed)
    if (observed['scope'] != 'observed' or observed.get('stage') != 'prediction'
            or observed.get('scene_gt_used') is not False):
        raise ValueError('Require predicted observed state, never GT or completed state')
    for obj in observed['objects']:
        if obj['local_evidence'] != 'visible' or any(not obj[k]['valid'] for k in (
                'center_local_m', 'size_aabb_local_m', 'bbox2d_visible_xyxy_norm')):
            raise ValueError('Every frozen observation needs visible 2D and metric geometry')
    on_floor = {r['id']: r['on_floor'] for r in grounding.get('instances', [])}
    return dict(frame=FRAME, objects=[dict(
        id=o['track_id'], category=o.get('raw_category', o.get('description')),
        description=o.get('description'), reported_category=o.get('reported_category'),
        adapter_category_id=o['category_id']['value'],
        center_m=o['center_local_m']['value'], full_camera_aabb_m=o['size_aabb_local_m']['value'],
        visible_bbox=o['bbox2d_visible_xyxy_norm']['value'], on_floor=on_floor.get(o['track_id']))
        for o in observed['objects']], geometry_is_prediction=True,
        evidence_description=grounding.get('evidence_description', ''),
        camera_gravity_and_room_boundaries_measured=False)


SPACE_DIRECTIONS = ('left', 'right', 'rear', 'front', 'below', 'above')


def observed_envelope(packet):
    """Necessary numeric containment bounds, not recovered walls or gravity."""
    lo = np.array([-.01, -.01, -.01]);hi = -lo
    for obj in packet['objects']:
        c = np.asarray(obj['center_m']);s = np.asarray(obj['full_camera_aabb_m'])
        lo = np.minimum(lo, c-s/2);hi = np.maximum(hi, c+s/2)
    return dict(lo=lo.tolist(), hi=hi.tolist(), scope='predicted_extent_and_camera_containment_only')


def compile_space(space, packet):
    if not isinstance(space, dict) or not isinstance(space.get('extra_clearance_m'), dict):
        raise ValueError('space.extra_clearance_m requires left/right/rear/front/below/above; absolute room coordinates are not the v2 interface')
    extra = space['extra_clearance_m']
    for direction in SPACE_DIRECTIONS:
        if not number(extra.get(direction)) or extra[direction] < 0:
            raise ValueError('extra_clearance_m.'+direction+' must be finite nonnegative metres, not a placeholder')
    base = observed_envelope(packet);lo = base['lo'];hi = base['hi']
    room = dict(x_min=lo[0]-extra['left'], x_max=hi[0]+extra['right'],
        z_min=lo[2]-extra['rear'], z_max=hi[2]+extra['front'],
        floor_y=lo[1]-extra['below'], ceiling_y=hi[1]+extra['above'])
    floor_contacts = [dict(id=o['id'], residual_m=o['center_m'][1]-o['full_camera_aabb_m'][1]/2-room['floor_y'])
                      for o in packet['objects'] if o.get('on_floor') is True]
    return room, dict(base_envelope=base, extra_clearance_m={k:extra[k] for k in SPACE_DIRECTIONS},
        numeric_containment_compiled=True, walls_measured=False, gravity_verified=False,
        support_floor=dict(value=room['floor_y'], method='conservative_hypothesis_below_all_predicted_extents',
            measured=False, observed_on_floor_residuals=floor_contacts),
        caution='The numeric envelope is not recovered architecture. Frozen predicted floor objects may float relative to this uncertain support plane.')


def hypotheses_prompt(packet, camera, count):
    return ('Propose alternative GLOBAL spatial and functional hypotheses for the SAME existing indoor world, '
        'compatible with the photograph and frozen observations. Do not redesign the room or infer biography. '
        'Return {"hypotheses":[{"description":"...","assumptions":["..."],'
        '"functions":["..."],"space":{"extra_clearance_m":{'
        '"left":"estimate metres","right":"estimate metres","rear":"estimate metres",'
        '"front":"estimate metres","below":"estimate metres","above":"estimate metres"}}}]}. '
        'Use between one and ' + str(count) + ' genuinely different hypotheses, no filler duplicates. '
        'Replace all six estimate fields with actual finite NONNEGATIVE numbers; zero is legal. '
        'Predict uncertain ADDITIONAL clearance beyond the supplied base envelope, not absolute coordinates. '
        'The photograph only limits its camera frustum: unpictured sides and behind-camera space are UNKNOWN, '
        'not walls at the crop boundary. Estimate the spatial extent of each hypothesized existing room, '
        'including any assumed unpictured space. Zero clearance is a substantive restrictive hypothesis, '
        'not a substitute for unknown. Do not repeat the same extent across different spatial hypotheses. '
        'The program computes containment exactly; do not calculate observed box extrema or return a room field. '
        'The result is a conservative numeric workspace, not measured walls. Its bottom is an uncertain '
        'support-plane proxy, not a recovered real floor; consider frozen on_floor evidence and residuals. '
        'Coordinates are camera x-right, y-IMAGE-UP, z-forward in METRES. Behind camera is z<0. '
        'Horizontal floor/ceiling assumes approximately level camera; explicitly record this uncertain approximation. '
        'Functions can coexist, be absent, share observed resources, and have alternative realizations. '
        'Do not return hidden object inventories, probabilities or resident history at this stage. '
        'No object-count, category, function or rear-coverage target. '
        '\nCAMERA: ' + json.dumps(camera) + '\nBASE_ENVELOPE: '+json.dumps(observed_envelope(packet))+
        '\nFROZEN_OBSERVATIONS: ' + json.dumps(packet))


def hypotheses_answer(value, packet, maximum):
    rows = value.get('hypotheses') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError('Return one to the budgeted number of hypotheses')
    result = []
    for i, row in enumerate(rows):
        room, audit = compile_space(row.get('space'), packet)
        result.append(dict(id='h' + str(i), description=text(row.get('description'), 'description'),
            assumptions=strings(row.get('assumptions'), 'assumptions'),
            functions=strings(row.get('functions'), 'functions'), space=deepcopy(row['space']),
            room=room, spatial_audit=audit))
    return result


def programs_prompt(hypothesis, packet, alternatives, maximum_objects):
    """Historical v1/v2 full-schema prompt; v3 uses global_world_proposals instead."""
    return ('Provide JOINT complete-world realization alternatives under this fixed global hypothesis. '
        'This is a program domain for subsequent conditional sampling, not the final world draw. '
        'All functions are realized together. Reuse one instance across functions; do not duplicate observed objects. '
        'Each alternative jointly owns all its listed NEW objects, plus the unchanged frozen observation. '
        'An empty additions list is legal. Ordinary possessions/decor need not be task-essential. '
        'Return {"programs":[{"description":"...","assumptions":["..."],'
        '"objects":[{"id":"n0","category":"ordinary noun","variant":"specific description",'
        '"size_m":[1.2,0.75,0.6],"support":{"kind":"floor","host":null},'
        '"visibility":"outside_fov","opaque":true,"reason":"why a distinct instance"}],'
        '"activities":[{"function":"...","uses":["obs_000","n0"],"realization":"..."}],'
        '"relations":[{"kind":"near","source":"n0","target":"obs_000","distance_m":2}],'
        '"access_zones":[{"owner":"n0","offset_m":[0,0,1],"size_m":[0.6,1.6,0.6],'
        '"purpose":"operation or approach space"}]}]}. '
        'Use one to ' + str(alternatives) + ' alternatives; at most ' + str(maximum_objects) +
        ' new instances per alternative is a COMPUTATION LIMIT, never a target. No required objects or rear quota. '
        'NEW IDs must start with n, e.g. n0. support.kind is floor/ceiling/wall/free/top/inside. '
        'Only top/inside use a host, referencing an observed ID or another new ID; no cycles. '
        'size_m is intrinsic FULL [width,height,depth] in metres, frozen BEFORE layout; never shrink to fit. '
        'Do not return positions/yaw at this stage. visibility is outside_fov or occluded. '
        'Do not use occluded to excuse a visible addition; later ray checks will test it. '
        'opaque describes whether a box proxy may occlude rays; glass/mirrors must not be opaque occluders. '
        'Relations: near requires positive distance_m (maximum horizontal centre distance); '
        'facing requires no distance field and refers to source NEW object local +z toward target; '
        'separate requires positive distance_m (minimum horizontal footprint gap). '
        'A service/activity link does not automatically imply near, facing or support. '
        'Activities explicitly bind functions to unique existing/new instances; uses may be empty for no equipment. '
        'access_zones are optional reserved operation/approach boxes in owner yaw-local axes; '
        'they are geometric usability proxies, not simulated humans. '
        '\nHYPOTHESIS: ' + json.dumps(hypothesis) + '\nFROZEN_OBSERVATIONS: ' + json.dumps(packet))


def _programs_answer_strict(value, hypothesis, packet, maximum, maximum_objects):
    rows = value.get('programs') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError('One to the budgeted number of joint programs required')
    observed = {o['id'] for o in packet['objects']}
    result = []
    for index, raw in enumerate(rows):
        p = deepcopy(raw)
        p['description'] = text(p.get('description'), 'program description')
        p['assumptions'] = strings(p.get('assumptions'), 'program assumptions')
        objects = p.get('objects')
        if not isinstance(objects, list) or len(objects) > maximum_objects:
            raise ValueError('objects must be a bounded list, empty allowed')
        ids = [o.get('id') for o in objects]
        if any(not isinstance(v, str) or not re.fullmatch(r'n[A-Za-z0-9_]*', v) for v in ids) or len(set(ids)) != len(ids):
            raise ValueError('New IDs must be unique n-prefixed identifiers')
        byid = {o['id']: o for o in objects}
        allids = observed | set(ids)
        for o in objects:
            for k in ('category', 'variant'):
                o[k] = text(o.get(k), k)
            if o['category'].casefold() in ('ordinary noun', '...', 'category', 'object name'):
                raise ValueError('Category is a schema placeholder, not a specific object noun')
            # Narrative justification is not an installation/ownership variable.
            # Missing prose stays explicitly missing; never fabricate an explanation.
            if o.get('reason') is not None:o['reason'] = text(o['reason'], 'reason')
            else:o['reason'] = None
            o['size_m'] = vector(o.get('size_m'), 3, 'size_m', positive=True)
            if type(o.get('opaque')) is not bool or o.get('visibility') not in ('outside_fov', 'occluded'):
                raise ValueError('Explicit opaque boolean and hidden visibility hypothesis required')
            s = o.get('support', {})
            if s.get('kind') not in SUPPORTS:
                raise ValueError('Unknown support kind')
            hosted = s['kind'] in ('top', 'inside')
            if (hosted and (s.get('host') not in allids or s['host'] == o['id'])) or (not hosted and s.get('host') is not None):
                raise ValueError('Invalid support host')
            o['support'] = dict(kind=s['kind'], host=s.get('host'))
            chain = {o['id']}
            host = s.get('host')
            while host in byid:
                if host in chain:
                    raise ValueError('Support graph contains a cycle')
                chain.add(host)
                host = byid[host].get('support', {}).get('host')
        activities = p.get('activities')
        if not isinstance(activities, list):
            raise ValueError('activities list required, possibly empty')
        for a in activities:
            a['function'] = text(a.get('function'), 'function')
            a['realization'] = text(a.get('realization'), 'realization')
            if set(strings(a.get('uses'), 'uses')) - allids:
                raise ValueError('Unknown activity instance reference')
        relations = p.get('relations')
        if not isinstance(relations, list):
            raise ValueError('relations list required')
        for r in relations:
            if (r.get('kind') == 'on' and r.get('source') in byid and
                    byid[r['source']]['support'] == dict(kind='top', host=r.get('target'))):
                # Same stated dependency in two encodings, not a new relation.
                r['kind'] = 'support_top'
                continue
            if r.get('kind') not in ('near', 'separate', 'facing') or r.get('source') not in allids or r.get('target') not in allids or r['source'] == r['target']:
                raise ValueError('Invalid typed relation')
            if r['kind'] == 'facing' and r['source'] not in byid:
                raise ValueError('Observed yaw unknown; facing source must be NEW')
            if r['kind'] != 'facing' and (not number(r.get('distance_m')) or r['distance_m'] <= 0):
                raise ValueError('Positive distance_m required')
            if r.get('measure', 'center') not in ('center', 'footprint_gap'):
                raise ValueError('Unknown distance measure')
        p['support_relation_evidence'] = [r for r in relations if r['kind'] == 'support_top']
        p['relations'] = [r for r in relations if r['kind'] != 'support_top']
        zones = p.get('access_zones')
        if not isinstance(zones, list):
            raise ValueError('access_zones list required')
        for z in zones:
            if z.get('owner') not in byid:
                raise ValueError('Access-zone owner must be NEW; observed orientation unknown')
            if z.get('base') == 'floor':vector(z.get('offset_xz_m'), 2, 'offset_xz_m')
            else:vector(z.get('offset_m'), 3, 'offset_m')
            vector(z.get('size_m'), 3, 'zone size', positive=True)
            text(z.get('purpose'), 'zone purpose')
        p['id'] = hypothesis['id'] + '_b' + str(index)
        p['hypothesis'] = hypothesis['id']
        result.append(p)
    return result


def programs_answer(value, hypothesis, packet, maximum, maximum_objects):
    """Validate independent alternatives without discarding valid siblings."""
    rows = value.get('programs') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError('One to the budgeted number of joint programs required')
    valid = [];errors = []
    for i, row in enumerate(rows):
        try:
            p = _programs_answer_strict(dict(programs=[row]), hypothesis, packet, 1, maximum_objects)[0]
            p['id'] = hypothesis['id']+'_b'+str(i)
            valid.append(p)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(dict(alternative=i, error=str(exc)))
    if not valid:raise ValueError('No valid joint program: '+json.dumps(errors))
    for p in valid:p['proposal_rejections'] = deepcopy(errors)
    return valid


def choice_prompt(kind, options, packet, hypothesis=None):
    return ('Choose ONE plausible existing world state under the supplied conditions. '
        'This is a conditional mechanism sample, not a vote for what COULD exist. '
        'Use spatial resources, shared objects and alternative functional realizations jointly. '
        'No count/density/sector target. Return only {"choice":"allowed alias"}. '
        '\nMECHANISM: ' + kind + '\nHYPOTHESIS: ' + json.dumps(hypothesis) +
        '\nFROZEN_OBSERVATIONS: ' + json.dumps(packet) + '\nOPTIONS: ' + json.dumps(options))


def empirical(counts, states):
    valid = sum(counts.values())
    if not valid:
        raise ValueError('No valid choices; no uniform/empty-world fallback')
    return [(counts.get(s, 0) + 1/len(states))/(valid + 1) for s in states]


def compile_model(hypotheses, programs, recognition, conditional):
    proposed = [h['id'] for h in hypotheses]
    prior = dict(zip(proposed, empirical(recognition, proposed)))
    hs = [h for h in proposed if programs.get(h) and sum(conditional.get(h, {}).values())]
    if not hs:
        raise ValueError('No executable hypothesis mechanism; inference failed before world sampling')
    mass = sum(prior[h] for h in hs)
    q = [prior[h]/mass for h in hs]
    nodes = [dict(id='H', parents=[], states=hs, cpt={'root': q})]
    allprograms = {p['id']: p for h in hs for p in programs[h]}
    states = list(allprograms)
    cpt = {}
    for h in hs:
        allowed = [p['id'] for p in programs.get(h, [])]
        law = dict(zip(allowed, empirical(conditional[h], allowed)))
        cpt[h] = [law.get(s, 0.) for s in states]
    nodes.append(dict(id='B', parents=['H'], states=states, cpt=cpt))
    registry = {}
    for pid, p in allprograms.items():
        for obj in p['objects']:
            oid = pid + '__' + obj['id']
            registry[oid] = dict(program=pid, object=deepcopy(obj))
            nodes.append(dict(id='E_' + oid, parents=['B'], states=['0', '1'],
                cpt={s: [float(s != pid), float(s == pid)] for s in states}))
    pending = dict(registry)
    installed = set()
    while pending:
        for oid, row in list(pending.items()):
            host = row['object']['support']['host']
            newids = {o['id'] for o in allprograms[row['program']]['objects']}
            hostid = row['program'] + '__' + host if host in newids else None
            if hostid and hostid not in installed:
                continue
            parents = ['E_' + oid] + (['S_' + hostid] if hostid else [])
            nodes.append(dict(id='S_' + oid, parents=parents, states=['0', '1'],
                cpt={'|'.join(v): [float(not all(x == '1' for x in v)), float(all(x == '1' for x in v))]
                     for v in itertools.product(('0', '1'), repeat=len(parents))}))
            installed.add(oid)
            del pending[oid]
            break
        else:
            raise ValueError('Unresolved support dependency')
    return validate_scm(dict(protocol=PROTOCOL, nodes=nodes,
        hypotheses=[h for h in hypotheses if h['id'] in hs],
        programs=allprograms, registry=registry,
        mechanism_coverage=dict(proposed_hypotheses=proposed, executable_hypotheses=hs,
            excluded_hypotheses=[h for h in proposed if h not in hs],
            original_elicited_law=prior, retained_elicited_mass=mass,
            excluded_elicited_mass=1.-mass,
            policy='explicit_conditioning_on_executable_domain; operational_failures_are_not_world_states'),
        factorization='q(H|O,D) p(B_joint|H,O,D); B->E; E,S_host->S; H,B,S->X; X,C->visibility',
        probability_scope='elicited_law_conditioned_on_executable_proxy_compatible_domain_before_semantic_audit',
        real_causal_structure_identified=False, probabilities_calibrated=False))


def sample_worlds(model, count, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(count):
        noise = {n['id']: float(rng.random()) for n in model['nodes']}
        state = forward(model, noise)
        draws.append(dict(index=i + 1, state=state, noise=noise,
            hypothesis=state['H'], program=state['B']))
    return draws


def intervention_diagnostics(model, sample):
    """Explicit model-relative structural effects, without claiming solved dynamics."""
    def summary(state):
        return dict(program=state['B'], owned=[oid for oid in model['registry'] if state['E_'+oid] == '1'],
            installed=[oid for oid in model['registry'] if state['S_'+oid] == '1'])
    functional = {h['id']: summary(forward(model, sample['noise'], {'H': h['id']})) for h in model['hypotheses']}
    removals = []
    selected = model['programs'].get(sample['program'])
    if selected:
        hosts = {o['support']['host'] for o in selected['objects']}
        for obj in selected['objects']:
            if obj['id'] in hosts:
                oid = selected['id']+'__'+obj['id']
                removals.append(dict(do={'E_'+oid: '0'}, after=summary(forward(model, sample['noise'], {'E_'+oid: '0'}))))
    return dict(functional_interventions=functional, host_removals=removals,
        same_exogenous_noise=True, geometry_recomputed=False,
        scope='assumed_finite_program_SCM_only; not_identified_physical_counterfactuals')


def export_graph(observed, hypothesis, program, layout):
    objects = deepcopy(observed['objects'])
    mapping = {o['id']: program['id'] + '__' + o['id'] for o in program['objects']}
    for obj in program['objects']:
        cid = None
        try:
            canonical, _ = category(obj['category'])
            cid = next(k for k, v in CANONICAL_CATEGORIES.items() if v == canonical)
        except ValueError:
            pass  # Keep open vocabulary; never drop or silently use a catch-all.
        pos = layout[obj['id']]
        objects.append(dict(track_id=mapping[obj['id']], raw_category=obj['category'],
            description=obj['variant'], category_id=field(cid, 'model_prediction', 'official_category_bridge', 'unmapped_label'),
            center_local_m=field(pos['center_m'], 'model_prediction', 'global_program_constrained_layout'),
            size_aabb_local_m=field(pos['aabb_size_m'], 'model_prediction', 'yaw_rotated_frozen_dimensions'),
            bbox2d_visible_xyxy_norm=field(None, reason='hidden_hypothesis'),
            in_primary_room=field(None, reason='hypothesized_envelope_not_measured_room'),
            local_evidence='unobserved', intrinsic_size_m=obj['size_m'], yaw_deg=pos['yaw_deg'],
            support=dict(kind=obj['support']['kind'], host=mapping.get(obj['support']['host'], obj['support']['host'])),
            visibility_hypothesis=obj['visibility'], camera_region=obj.get('camera_region'),
            opaque_proxy=obj['opaque'], reason=obj['reason']))
    state = graph(observed['id'], observed['split'], 'full', objects, observed['vocabulary_sha256'])
    def bind(row):
        return {k: mapping.get(v, v) if k in ('source', 'target', 'owner') else v for k, v in row.items()}
    state.update(scene_gt_used=False, model_variant=PROTOCOL, room_hypothesis=deepcopy(hypothesis['room']),
        room_hypothesis_method='compiled_conservative_workspace_not_measured_architecture',
        hypothesis=deepcopy(hypothesis), assumptions=program['assumptions'],
        relations=[bind(r) for r in program['relations']],
        activities=[dict(a, uses=[mapping.get(v, v) for v in a['uses']]) for a in program['activities']],
        access_zones=[bind(z) for z in program['access_zones']],
        validation_scope='camera_AABB_and_reserved_space_proxies_not_mesh_physics_or_true_posterior')
    if state['objects'][:len(observed['objects'])] != observed['objects']:
        raise AssertionError('Frozen observation mutated')
    return validate_graph(state)
