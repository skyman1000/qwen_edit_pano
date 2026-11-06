"""Small typed requests for local 8B models, following the v13 stage separation.

Inventories precede and freeze dimensions; relative camera/host coordinates are
compiled, never guessed by a transport adapter. Valid independent rows survive
bounded missing-row requests. No object-count/category/sector quota or fallback
inventory. The JSON journals remain the source for every model-valued field.
"""
from copy import deepcopy
import json
import re

import numpy as np

from ..common import write
from . import global_world_model as model

BATCH = 6
REGIONS = ('rear', 'left', 'right', 'outside_fov', 'occluded')


def inventory_prompt(h, packet, alternatives, maximum):
    return ('Infer JOINT inventories of the SAME EXISTING world beyond this photograph under the fixed '
        'global hypothesis. Infer what may already exist; this is not a request to furnish, improve or '
        'redesign the room. All its functions coexist in each alternative; reuse shared and observed '
        'instances. Predict a complete plausible hidden inventory, not one suggested improvement. '
        'Sparse or empty inventories are legal when justified. No category/count/rear quota. '
        'Return a JSON object with programs (array). Each program has description (string), '
        'assumptions (array of strings), objects (array), activities (array). '
        'Each NEW object has id (unique n-prefixed string), category (specific ordinary object noun), '
        'variant (specific description), reason (why a distinct hidden instance is hypothesized). '
        'Each activity has function (string), uses (array of exact observed/new IDs), realization (string). '
        'Do NOT output dimensions, support, coordinates, distances, access zones or probabilities yet. '
        'At most '+str(alternatives)+' distinct JOINT alternatives and at most '+str(maximum)+
        ' new instances each are computation limits, not targets. '
        '\nHYPOTHESIS: '+json.dumps(h)+'\nFROZEN_OBSERVATIONS: '+json.dumps(packet))


def inventory_answer(value, h, packet, maximum, max_objects):
    rows = value.get('programs') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError('One to the budgeted number of inventory alternatives required')
    valid = [];errors = [];observed = {o['id'] for o in packet['objects']}
    for i, raw in enumerate(rows):
        try:
            p = dict(id=h['id']+'_b'+str(i), hypothesis=h['id'],
                description=model.text(raw.get('description'), 'description'),
                assumptions=model.strings(raw.get('assumptions', []), 'assumptions'))
            objects = raw.get('objects')
            if not isinstance(objects, list) or len(objects) > max_objects:raise ValueError('Bounded objects array required')
            ids = [o.get('id') for o in objects]
            if any(not isinstance(v, str) or not re.fullmatch(r'n[A-Za-z0-9_]*', v) for v in ids) or len(set(ids)) != len(ids):
                raise ValueError('Unique n-prefixed NEW IDs required')
            p['objects'] = []
            for o in objects:
                noun = model.text(o.get('category'), 'category')
                if noun.casefold() in ('ordinary noun', '...', 'object', 'category', 'object name'):
                    raise ValueError('Specific object category required, not a schema placeholder')
                p['objects'].append(dict(id=o['id'], category=noun,
                    variant=model.text(o.get('variant'), 'variant'),
                    reason=model.text(o['reason'], 'reason') if o.get('reason') is not None else None))
            activities = raw.get('activities', [])
            if not isinstance(activities, list):raise ValueError('activities array required')
            p['activities'] = []
            for a in activities:
                uses = model.strings(a.get('uses'), 'uses')
                if set(uses) - (observed | set(ids)):raise ValueError('Unknown activity ID')
                p['activities'].append(dict(function=model.text(a.get('function'), 'function'), uses=uses,
                    realization=model.text(a.get('realization'), 'realization')))
            valid.append(p)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:errors.append(dict(alternative=i, error=str(exc)))
    if not valid:raise ValueError('No valid inventory: '+json.dumps(errors))
    for p in valid:p['inventory_rejections'] = deepcopy(errors)
    return valid


def typed_rows(value, expected, validator):
    """Only lossless containers/exact dictionary keys may supply identity."""
    if not isinstance(value, dict):raise ValueError('JSON object required')
    if 'objects' in value:rows = value['objects']
    elif set(value).issubset(expected):rows = value
    else:raise ValueError('Use objects array or exact-ID dictionary')
    if isinstance(rows, dict):
        candidates = []
        for oid, row in rows.items():
            if not isinstance(row, dict):raise ValueError('Object record must be a dictionary')
            row = deepcopy(row)
            if 'id' in row and row['id'] != oid:raise ValueError('Conflicting key and explicit ID')
            row['id'] = oid;candidates.append(row)
        rows = candidates
    if not isinstance(rows, list):raise ValueError('objects array or exact-ID dictionary required')
    valid = {};errors = [];conflicts = set()
    for row in rows:
        oid = row.get('id') if isinstance(row, dict) else None
        try:
            if not isinstance(oid, str) or oid not in expected:raise ValueError('Unrequested or missing exact ID')
            clean = validator(row)
            if oid in valid and valid[oid] != clean:conflicts.add(oid);raise ValueError('Conflicting duplicate object rows')
            valid[oid] = clean
        except (ValueError, TypeError, KeyError, AttributeError) as exc:errors.append(dict(id=oid, error=str(exc)))
    for oid in conflicts:valid.pop(oid, None)
    return dict(rows=valid, errors=errors)


def fill_rows(calls, name, image, objects, prompt_for, validator, dest):
    accepted = {};audit = []
    for batch in range(0, len(objects), BATCH):
        requested = [o['id'] for o in objects[batch:batch+BATCH]]
        errors = []
        for attempt in range(2):
            missing = [oid for oid in requested if oid not in accepted]
            if not missing:break
            prompt = prompt_for(missing, accepted)+'\nPREVIOUS_ROW_ERRORS: '+json.dumps(errors)
            stage = name+'_batch_'+str(batch//BATCH)+'_pass_'+str(attempt)
            answer = calls.ask(stage, image, prompt,
                lambda v: typed_rows(v, set(missing), validator), tokens=4096)
            accepted.update(answer['rows']);errors = list(answer['errors'])
            errors += [dict(id=oid, error='Requested row omitted') for oid in missing if oid not in accepted]
            audit.append(dict(stage=stage, requested=missing, errors=errors))
            write(dest/(name+'_progress.json'), dict(accepted=accepted, audit=audit,
                identity_invented=False, valid_rows_preserved=True))
        unresolved = [oid for oid in requested if oid not in accepted]
        if unresolved:raise ValueError('Bounded typed-row budget exhausted for '+name+': '+json.dumps(unresolved))
    return accepted


def materialize(calls, image, h, inventory, packet, dest, camera):
    """Freeze semantic identity -> size/support -> relative plan -> optional constraints."""
    p = deepcopy(inventory);pid = p['id'];objects = p['objects']
    allids = {o['id'] for o in objects} | {o['id'] for o in packet['objects']}
    def dimension_row(row):
        size = model.vector(row.get('size_m'), 3, 'size_m', positive=True)
        code = row.get('support')
        if not isinstance(code, str):raise ValueError('support is one typed string, not a kind/host dictionary')
        parts = code.split(':', 1);kind = parts[0];host = parts[1] if len(parts) == 2 else None
        if kind not in model.SUPPORTS:raise ValueError('Unknown support kind')
        if kind in ('top', 'inside'):
            if host not in allids or host == row['id']:raise ValueError('Exact non-self host ID required after colon')
        elif host is not None:raise ValueError('Only top/inside have an object host; floor/wall/ceiling/free are hostless')
        if type(row.get('opaque')) is not bool:raise ValueError('opaque boolean required')
        return dict(size_m=size, support=dict(kind=kind, host=host), opaque=row['opaque'])
    def dimensions_prompt(missing, accepted):
        return ('Estimate intrinsic FULL object dimensions and support for ONLY these frozen NEW IDs. '
            'Return a JSON object with objects (array); each row has the requested id, '
            'size_m (three positive numbers: width,height,depth in metres), support (one string), '
            'opaque (boolean; false for transparent/open structures). '
            'support is floor, ceiling, wall or free for hostless installation, or top:HOST_ID / inside:HOST_ID '
            'for an explicit observed/new host. Wall means a hypothesized room wall, not an object host. '
            'Never make up hosts, change categories, resize to fit a crop, or supply layout yet. '
            'Estimate realistic size for each category/variant independently of available space. '
            '\nEXACT_REQUESTED_IDS: '+json.dumps(missing)+'\nFROZEN_INVENTORY: '+json.dumps(p)+
            '\nACCEPTED_READ_ONLY: '+json.dumps(accepted)+'\nOBSERVATIONS: '+json.dumps(packet))
    dims = fill_rows(calls, 'dimensions_'+pid, None, objects, dimensions_prompt, dimension_row, dest)
    for o in objects:o.update(dims[o['id']])
    write(dest/(pid+'_frozen_dimensions.json'), dict(objects=objects, dimensions_frozen=True))
    # Detect cycles before a coordinate request or geometry invocation.
    byid = {o['id']:o for o in objects}
    for o in objects:
        chain = set();oid = o['id']
        while oid in byid:
            if oid in chain:raise ValueError('Support graph contains a cycle')
            chain.add(oid);oid = byid[oid]['support']['host']
    def coordinate_row(row):
        o = byid[row['id']];kind = o['support']['kind']
        if row.get('region') not in REGIONS:raise ValueError('Explicit camera region required')
        if not model.number(row.get('yaw_deg')):raise ValueError('Finite absolute yaw_deg required')
        clean = dict(region=row['region'], yaw_deg=row['yaw_deg'])
        if kind in ('top', 'inside'):
            clean['host_offset_uv'] = model.vector(row.get('host_offset_uv'), 2, 'host_offset_uv')
            if max(abs(v) for v in clean['host_offset_uv']) > 1:raise ValueError('host_offset_uv must be in [-1,1]')
            if kind == 'inside':
                f = row.get('inside_height_fraction')
                if not model.number(f) or not 0 <= f <= 1:raise ValueError('inside_height_fraction in [0,1] required')
                clean['inside_height_fraction'] = f
        else:
            if row.get('anchor') not in {'camera'} | {o['id'] for o in packet['objects']}:
                raise ValueError('Root anchor must be camera or an exact observed ID')
            clean.update(anchor=row['anchor'], offset_xz_m=model.vector(row.get('offset_xz_m'), 2, 'offset_xz_m'))
            if kind in ('wall', 'free'):
                if not model.number(row.get('height_m')):raise ValueError('Finite camera-Y height_m required')
                clean['height_m'] = row['height_m']
        return clean
    def coordinates_prompt(missing, accepted):
        contracts = []
        for oid in missing:
            kind = byid[oid]['support']['kind']
            fields = ['id', 'region', 'yaw_deg']
            if kind in ('top', 'inside'):fields += ['host_offset_uv'] + (['inside_height_fraction'] if kind == 'inside' else [])
            else:fields += ['anchor', 'offset_xz_m'] + (['height_m'] if kind in ('wall', 'free') else [])
            contracts.append(dict(id=oid, required_fields=fields))
        return ('Plan relative placement of ONLY the requested frozen instances as ONE existing-world inventory. '
            'Return a JSON object with objects (array), using each exact requested ID and its required fields. '
            'Frame is camera x-right, y-image-up, z-forward. Positive yaw rotates local +z toward +x. '
            'region is rear, left, right, outside_fov (any hidden direction), or occluded (must pass ray tests). '
            'rear/left/right describe camera-relative sectors outside the original frustum, not image quadrants. '
            'Do not label an in-view addition hidden; no sector coverage quota. '
            'For roots, anchor is camera or an exact observed ID; offset_xz_m is [dx,dz] in camera metres '
            'relative to that anchor. Floor/ceiling Y is compiled; wall/free height_m is absolute camera Y. '
            'For hosted items, host_offset_uv is [u,v] within [-1,1] in host intrinsic horizontal axes; '
            'translation is [u*host_width/2,v*host_depth/2] and compiler applies host yaw. '
            'Top Y is compiled; inside_height_fraction is the bottom-to-top fraction of host free vertical extent. '
            'Keep fixed hosts/sizes; consider all other objects and previously accepted placements together. '
            '\nCONTRACTS: '+json.dumps(contracts)+'\nFROZEN_PROGRAM: '+json.dumps(p)+
            '\nHYPOTHESIS: '+json.dumps(h)+'\nCAMERA: '+json.dumps(camera)+
            '\nACCEPTED_READ_ONLY: '+json.dumps(accepted)+'\nOBSERVATIONS: '+json.dumps(packet))
    coords = fill_rows(calls, 'coordinates_'+pid, image, objects, coordinates_prompt, coordinate_row, dest)
    from .global_world_geometry import rotation, aabb_size
    centres = {o['id']:np.asarray(o['center_m'], dtype=float) for o in packet['objects']}
    sizes = {o['id']:np.asarray(o['full_camera_aabb_m']) for o in packet['objects']}
    intrinsic = dict(sizes);yaws = {oid:0. for oid in centres};pending = list(byid);proposal = {}
    while pending:
        for oid in list(pending):
            o = byid[oid];row = coords[oid];s = o['support'];host = s['host'];kind = s['kind']
            if host and host not in centres:continue
            size = np.asarray(aabb_size(o['size_m'], row['yaw_deg']))
            if host:
                u,v = row['host_offset_uv'];offset = np.array([u*intrinsic[host][0]/2,0,v*intrinsic[host][2]/2])
                c = centres[host]+rotation(yaws[host])@offset
                if kind == 'top':c[1] = centres[host][1]+(sizes[host][1]+size[1])/2
                else:c[1] = centres[host][1]+(row['inside_height_fraction']-.5)*max(0., sizes[host][1]-size[1])
            else:
                anchor = np.zeros(3) if row['anchor'] == 'camera' else centres[row['anchor']]
                c = anchor + np.array([row['offset_xz_m'][0], 0., row['offset_xz_m'][1]])
                c[1] = (h['room']['floor_y']+size[1]/2 if kind == 'floor' else
                    h['room']['ceiling_y']-size[1]/2 if kind == 'ceiling' else row['height_m'])
            o.update(visibility='occluded' if row['region'] == 'occluded' else 'outside_fov', camera_region=row['region'])
            proposal[oid] = dict(center_m=c.tolist(), yaw_deg=row['yaw_deg'])
            centres[oid] = c;sizes[oid] = size;intrinsic[oid] = np.asarray(o['size_m']);yaws[oid] = row['yaw_deg']
            pending.remove(oid);break
        else:raise ValueError('Unresolved support dependency')
    write(dest/(pid+'_relative_plan.json'), dict(coordinates=coords, compiled_proposal=proposal,
        coordinate_frame_transport='camera_or_intrinsic_host_axes', inventory_changed=False))
    p.update(relations=[], access_zones=[])
    if objects:
        prompt = ('Specify ONLY necessary geometric relations/access spaces for this FIXED existing-world program. '
            'Return a JSON object with relations (array) and access_zones (array); both may be empty. '
            'Do not add spatial constraints just because objects participate in an activity. '
            'A relation has kind near/separate/facing, source and target exact IDs. '
            'For near/separate use measure=footprint_gap and positive distance_m: maximum/minimum '
            'horizontal box-footprint CLEAR GAP, not centre distance. Facing has no distance/measure '
            'and its source must be NEW; local +z faces the target within sixty degrees. '
            'Support is already fixed: do not repeat on/in relations. '
            'Access zones are only for an explicit operation needing free space, not every object. '
            'A zone has owner (NEW ID), base=floor, offset_xz_m ([dx,dz] in owner yaw axes), '
            'size_m (full positive width,height,depth), purpose (string). Compiler grounds its bottom '
            'on the hypothesized floor. Do not put an access space inside its owning object. '
            'Check consistency with frozen dimensions, camera region and room extent before asserting constraints. '
            '\nFIXED_PROGRAM: '+json.dumps(p)+'\nRELATIVE_PLAN: '+json.dumps(coords)+
            '\nHYPOTHESIS: '+json.dumps(h)+'\nOBSERVATIONS: '+json.dumps(packet))
        def constraints_answer(value):
            if not isinstance(value, dict):raise ValueError('Object with relations/access_zones required')
            candidate = deepcopy(p)
            candidate['relations'] = value.get('relations');candidate['access_zones'] = value.get('access_zones')
            return model.programs_answer(dict(programs=[candidate]), h, packet, 1, len(objects))[0]
        checked = calls.ask('constraints_'+pid, None, prompt, constraints_answer, tokens=4096)
        p.update(relations=checked['relations'], access_zones=checked['access_zones'],
            support_relation_evidence=checked.get('support_relation_evidence', []))
    # Final common contract, keeping the original alternative identity.
    checked = model.programs_answer(dict(programs=[p]), h, packet, 1, len(objects))[0]
    checked['id'] = pid
    checked['initial_placement'] = proposal
    checked['proposal_interface'] = 'inventory_then_frozen_dimensions_then_typed_relative_plan'
    return checked
