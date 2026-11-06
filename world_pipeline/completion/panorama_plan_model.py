"""Whole-room hypotheses -> shared spatial plan -> conditional inventories.

An assumed, LM-mechanism generative model, not identified real-world causality.
Planning is independent of metric acceptance. Spatial relations are not causal edges.
"""
from copy import deepcopy
import json
import math
import re

from ..common import finite_vector

VERSION = 'panorama_plan_first_v1'
REGIONS = ('front', 'left', 'right', 'back', 'centre')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    require(isinstance(value, str) and bool(value.strip()), 'Nonempty text required')
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', value),
            'IDs must start with a letter and contain letters, digits, _ or -')
    return value


def unique(rows):
    require(isinstance(rows, list), 'Expected an array')
    ids = [identifier(r['id']) for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate IDs')
    return set(ids)


def bounds(value):
    require(finite_vector(value, 4) and value[0] < value[1] and value[2] < value[3],
            'bounds_xz_m must be [xmin,xmax,zmin,zmax], finite increasing pairs')


def context(case):
    return json.dumps(dict(hfov=case['hfov'], observed=case['context']), ensure_ascii=False)


TASK = (
    'Infer a plausible EXISTING whole indoor world surrounding the SAME camera from ONE partial RGB. '
    'The target is G_full = immutable G_obs plus unobserved instances. This is not decorating a corner '
    'and not a test of whether hidden objects are visually proved. Unseen does not mean empty. '
    'Account for the photographed area AND the rest of the room as one coherent organization. '
    'Do not duplicate any visible object, including visible objects missed by the detector. '
    'An extension of a visible activity can continue outside the image. Distinguish assumptions from evidence. '
    'No mandatory category, minimum object count or requirement to populate every direction. '
    'Empty space, circulation and room boundaries are legitimate parts of the plan. '
    'Do not infer the rest of the building or a household biography. '
)


def hypotheses_prompt(case, count):
    return TASK + (
        f'Propose exactly {count} alternative whole-room organizations, not alternatives for a single object. '
        'Vary plausible spatial topology / functional coexistence / extent where uncertain; do not merely '
        'rename the same scene or exaggerate object counts. Return {"hypotheses":[{"id":"h0",'
        '"room_type":"...","organization":"concrete whole-room description",'
        '"assumptions":["..."],"evidence_ids":["existing observed ID"]}]}. '
        'No new inventory or metric poses yet.\nFIXED_OBSERVATION: ' + context(case))


def hypotheses_answer(value, case, count):
    rows = value['hypotheses']; unique(rows)
    require(len(rows) == count, 'Return the requested number of whole-room alternatives')
    old = {o['id'] for o in case['context']}
    for h in rows:
        text(h['room_type']); text(h['organization'])
        require(isinstance(h['assumptions'], list) and h['assumptions'], 'State assumptions')
        for a in h['assumptions']: text(a)
        require(isinstance(h['evidence_ids'], list) and set(h['evidence_ids']) <= old,
                'Evidence IDs must be frozen observed IDs')
    return deepcopy(rows)


def blueprint_prompt(case, h):
    return TASK + (
        'Realize CURRENT_H as a WHOLE-ROOM spatial and functional blueprint before enumerating hidden instances. '
        'Each zone describes one coexisting area, not a competing role alternative. Several zones may share '
        'a function. Assign every observed ID to exactly one primary zone. Zones may extend beyond the image. '
        'Explain how each zone is furnished or intentionally open in contents_intent. Include ordinary '
        'supporting belongings in this intent where appropriate, not only one landmark piece. '
        'Account for all surrounding directions in sector_review; this is an explanation, not an occupancy quota. '
        'Specify circulation between zones using IDs. Note visibly missed detections in visible_unmodeled '
        'as text, never turn them into new hidden instances. '
        'Use rough camera-local metres X right,Y up,Z forward (back is negative Z). The room envelope '
        'and zone bounds are APPROXIMATE HYPOTHESES for planning/visualization, not measured geometry or hard gates. '
        'Camera [0,0,0] belongs inside the envelope. floor_y_m is negative under an assumed level floor. '
        'Return {"room_envelope":{"bounds_xz_m":[xmin,xmax,zmin,zmax],"floor_y_m":number,'
        '"height_m":number,"basis":"assumptions and visible scale cues"},'
        '"zones":[{"id":"z0","function":"free text","region":"front|left|right|back|centre",'
        '"bounds_xz_m":[xmin,xmax,zmin,zmax],"observed_ids":[],"contents_intent":"..."}],'
        '"circulation":[{"from":"z0","to":"z1","reason":"..."}],'
        '"sector_review":{"front":"...","left":"...","right":"...","back":"..."},'
        '"visible_unmodeled":["visible structure missing from G_obs, if any"]}. '
        '\nCURRENT_H: ' + json.dumps(h, ensure_ascii=False) + '\nFIXED_OBSERVATION: ' + context(case))


def blueprint_answer(value, case):
    room = value['room_envelope']; bounds(room['bounds_xz_m']); text(room['basis'])
    f, ht = room['floor_y_m'], room['height_m']
    require(type(f) in (int, float) and math.isfinite(f) and f < 0, 'Finite negative hypothesized floor')
    require(type(ht) in (int, float) and math.isfinite(ht) and ht > -f, 'Ceiling must be above camera')
    b = room['bounds_xz_m']
    require(b[0] <= 0 <= b[1] and b[2] <= 0 <= b[3], 'Camera must be in hypothesized room envelope')
    zs = unique(value['zones']); require(zs, 'Whole-room blueprint needs a zone, including observed area')
    refs = []
    for z in value['zones']:
        text(z['function']); text(z['contents_intent']); bounds(z['bounds_xz_m'])
        require(z['region'] in REGIONS, 'Unknown camera region')
        require(isinstance(z['observed_ids'], list), 'observed_ids must be an array')
        refs.extend(z['observed_ids'])
    require(len(refs) == len(set(refs)) and set(refs) == {o['id'] for o in case['context']},
            'Assign every frozen observed ID to exactly one primary zone; no omissions or new IDs')
    require(isinstance(value['circulation'], list), 'circulation must be an array')
    for r in value['circulation']:
        require(r['from'] in zs and r['to'] in zs, 'Unknown circulation zone'); text(r['reason'])
    for direction in ('front', 'left', 'right', 'back'): text(value['sector_review'][direction])
    require(isinstance(value['visible_unmodeled'], list), 'visible_unmodeled must be an array')
    for s in value['visible_unmodeled']: text(s)
    return deepcopy(value)


def inventory_prompt(case, h, plan, zone, earlier):
    return TASK + (
        'Instantiate the hidden inventory of REQUESTED_ZONE in the fixed WHOLE-ROOM BLUEPRINT. '
        'All zones coexist, including zones with the same function. Use the complete plan and already '
        'allocated objects to avoid duplicate furniture and inconsistent capacities. '
        'Choose furniture AND ordinary associated contents as a coherent instance set, with actual hosts. '
        'Do not fill the token budget by repeating small objects. Do not turn an unfurnished passage into furniture. '
        'The correct number follows this specific hypothesized setting, not a quota. '
        'Return {"objects":[{"id":"unique_new_id","category":"specific ordinary noun",'
        '"description":"...","placement":"qualitative position around camera and in zone",'
        '"support":{"kind":"floor|top|inside|wall|ceiling|free","host":null},'
        '"reason":"why this instance belongs in this proposed world"}],"empty_reason":"reason if empty, else empty string"}. '
        'top and inside require a host ID in this list, earlier inventory, or observed inventory; '
        'other support kinds require host=null. Furniture cannot rest on tiny belongings. '
        'A support is an installation dependency, not evidence the object must exist. '
        'Do not invent numeric layout or rename observed instances. '
        '\nCURRENT_H: ' + json.dumps(h, ensure_ascii=False) + '\nBLUEPRINT: ' + json.dumps(plan, ensure_ascii=False) +
        '\nREQUESTED_ZONE: ' + json.dumps(zone, ensure_ascii=False) + '\nEARLIER_INVENTORY: ' +
        json.dumps(earlier, ensure_ascii=False) + '\nFIXED_OBSERVATION: ' + context(case))


def inventory_answer(value, case, zone, earlier):
    rows = value['objects']; ids = unique(rows)
    old = {o['id'] for o in case['context']} | {o['id'] for o in earlier}
    require(not ids & old, 'New IDs must not redeclare observed or already allocated instances')
    if not rows: text(value['empty_reason'])
    result = deepcopy(rows)
    for o in result:
        for k in ('category', 'description', 'placement', 'reason'): text(o[k])
        a = o['support']; kind = a['kind']; host = a['host']
        require(kind in ('floor', 'top', 'inside', 'wall', 'ceiling', 'free'), 'Unknown support kind')
        require((kind in ('top', 'inside') and isinstance(host, str) and host in old | ids and host != o['id'])
                or (kind not in ('top', 'inside') and host is None), 'Invalid or missing support host')
        o['zone_id'] = zone['id']
    ordered(result, old)  # Explicit cycle check; do not silently cut dependencies.
    return result


def ordered(objects, available):
    result = []; known = set(available); todo = list(objects)
    while todo:
        ready = [o for o in todo if o['support']['host'] is None or o['support']['host'] in known]
        require(ready, 'Support dependency cycle or missing host')
        for o in ready: result.append(o); known.add(o['id']); todo.remove(o)
    return result


def installation(objects, observed_ids, unavailable=()):
    """do(S_host=0): propagate installation, never erase owned child inventory."""
    blocked = set(unavailable)
    require(blocked <= set(observed_ids) | {o['id'] for o in objects}, 'Unknown intervention ID')
    states = {oid: oid not in blocked for oid in observed_ids}
    for o in ordered(objects, observed_ids):
        host = o['support']['host']
        states[o['id']] = o['id'] not in blocked and (host is None or states[host])
    return dict(inventory_ids=[o['id'] for o in objects], installed=states,
                intervention=dict(unavailable_hosts=sorted(blocked)),
                scope='model_relative_installation_only_no_claim_about_ownership_or_replacement')


def review_prompt(case, h, plan, objects):
    return TASK + (
        'Audit this already proposed whole world. Judge coherent global organization, duplication of visible '
        'objects, fulfilment of the stated zone intents, ordinary support contents, excessive repeated items, '
        'and circulation. Do not demand a specific furniture category or a minimum object count. '
        'Do not reject merely because hidden items are not visible, or because metric placement is approximate. '
        'Return {"plausibility":"plausible|needs_review","reason":"...",'
        '"zones":[{"id":"each zone exactly once","assessment":"..."}],'
        '"issues":[{"kind":"...","ids":["existing observed/new/zone IDs"],"reason":"..."}]}. '
        'This audit does not edit the inventory and is not independent ground truth. '
        '\nH: ' + json.dumps(h, ensure_ascii=False) + '\nPLAN: ' + json.dumps(plan, ensure_ascii=False) +
        '\nINVENTORY: ' + json.dumps(objects, ensure_ascii=False) + '\nOBSERVATION: ' + context(case))


def review_answer(value, plan, objects, case):
    require(value['plausibility'] in ('plausible', 'needs_review'), 'Unknown review verdict')
    text(value['reason'])
    require(unique(value['zones']) == {z['id'] for z in plan['zones']}, 'Review every zone once')
    for z in value['zones']: text(z['assessment'])
    allowed = {z['id'] for z in plan['zones']} | {o['id'] for o in objects} | {o['id'] for o in case['context']}
    require(isinstance(value['issues'], list), 'issues must be an array')
    for i in value['issues']:
        text(i['kind']); text(i['reason'])
        require(isinstance(i['ids'], list) and set(i['ids']) <= allowed, 'Unknown audit reference')
    return deepcopy(value)


def metric_prompt(case, plan, objects, requested, placed):
    return (
        'Give ROUGH plausible 3D boxes for EXACT REQUESTED_IDS in this frozen full-world plan. '
        'Keep every identity, category and support; no new inventory or object deletion. '
        'Use the already planned zones and all other objects jointly. No demand for millimetric precision. '
        'Camera axes: X right,Y up,Z forward; negative Z behind camera. All coordinates are ABSOLUTE metres. '
        'Objects inferred as hidden should stay outside the original photograph. Do not move them far away '
        'or shrink furniture to satisfy this. A zone can extend beyond the image, but being on the left '
        'of the image does not itself mean outside its viewing cone. '
        'Return {"objects":[{"id":"exact requested ID","center_m":[x,y,z],'
        '"full_size_m":[intrinsic_width,height,depth],"yaw_deg":number}]}. '
        'For floor objects Y will be set to hypothesized floor+height/2. For top objects Y will be '
        'set to host top+height/2; XZ must lie on that host, respecting its orientation and siblings. '
        'inside uses the predicted centre inside its host. Wall/ceiling mounting is a hypothesis. '
        '\nPLAN: ' + json.dumps(plan, ensure_ascii=False) + '\nALL_INVENTORY: ' + json.dumps(objects, ensure_ascii=False) +
        '\nREQUESTED_IDS: ' + json.dumps([o['id'] for o in requested]) + '\nALREADY_PLACED: ' + json.dumps(placed) +
        '\nOBSERVATION: ' + context(case))


def metric_answer(value, requested):
    rows = value['objects']
    require(unique(rows) == {o['id'] for o in requested}, 'Return every requested pose ID exactly once')
    for r in rows:
        require(finite_vector(r['center_m'], 3), 'Finite centre required')
        require(finite_vector(r['full_size_m'], 3) and min(r['full_size_m']) > 0, 'Positive full sizes required')
        require(type(r['yaw_deg']) in (int, float) and math.isfinite(r['yaw_deg']), 'Finite yaw required')
    return deepcopy(rows)
