"""Lossless coordinate-envelope normalization; no SCM or geometry changes.

Only identity/container syntax may be normalized. Every numeric field must
already be present, finite, and valid for the immutable attachment contract.
"""
from copy import deepcopy
import json
import math
import traceback

from ..common import read,write
from . import assembly_organized_geometry as geometry
from .assembly_named_geometry import row_answer,contract,capture_legacy
from .assembly_experiment import SeededProvider,stage_seed
from .world_scm_runner import ask_valid

PARAMETERIZATION=geometry.PARAMETERIZATION
compile_layout=geometry.compile_layout
annotate_state=geometry.annotate_state
INTERFACE='typed_coordinates_lossless_envelopes_v3'
BATCH=6


def response(value,group,members):
    """Decode equivalent containers, retaining independent valid typed rows.

    An exact outer group key or object key may supply its own identifier, but
    it cannot override a conflicting explicit ID. No coordinate or range is
    invented. A conflicting duplicate typed row remains unresolved.
    """
    gid=group['key'];contracts={o['id']:o for o in members};actions=[];errors=[]
    if not isinstance(value,dict):raise ValueError('Coordinate response must be a JSON object')
    if set(value)=={gid} and isinstance(value[gid],dict):value=value[gid];actions.append('unwrap_exact_group_key')
    if 'group_id' in value and value['group_id']!=gid:raise ValueError('Conflicting group_id; expected '+gid)
    if 'group_id' not in value:actions.append('group_bound_to_exact_request')
    payload={k:v for k,v in value.items() if k not in ('group_id','range_m')};candidates=[]
    def collect(part):
        if isinstance(part,list):candidates.extend(deepcopy(part));return
        if isinstance(part,dict):
            for oid,row in part.items():
                if oid not in contracts or not isinstance(row,dict):raise ValueError('Unknown keyed object or invalid record: '+str(oid))
                row=deepcopy(row)
                if 'id' in row and row['id']!=oid:raise ValueError('Conflicting key and object id: '+oid)
                if 'id' not in row:row['id']=oid;actions.append('object_id_from_exact_key:'+oid)
                candidates.append(row)
            return
        raise ValueError('objects must be a list or an exact-ID dictionary')
    if not payload:pass  # Range-only replies carry no object records.
    elif set(payload)=={'objects'}:collect(payload['objects'])
    elif set(payload).issubset({'roots','hosted'}):
        for label in ('roots','hosted'):
            if label in payload:collect(payload[label])
        actions.append('flatten_roots_and_hosted')
    elif set(payload).issubset(contracts):collect(payload)
    else:raise ValueError('Use objects records; unexpected envelope keys '+json.dumps(sorted(set(payload)-{'objects','roots','hosted'})))
    valid={};conflicts=set()
    for row in candidates:
        oid=row.get('id') if isinstance(row,dict) else None
        if not isinstance(oid,str) or oid not in contracts:errors.append(dict(id=oid,error='Unknown or missing exact object ID'));continue
        try:clean=row_answer(row,contracts[oid])
        except ValueError as exc:errors.append(dict(id=oid,error=str(exc)));continue
        if oid in valid and clean!=valid[oid]:conflicts.add(oid);errors.append(dict(id=oid,error='Conflicting valid coordinate rows'))
        elif oid in valid:actions.append('deduplicate_identical_typed_row:'+oid)
        else:valid[oid]=clean
    for oid in conflicts:valid.pop(oid,None)
    distance=value.get('range_m')
    if distance is not None and (not geometry.number(distance) or distance<=0):errors.append(dict(id=gid,error='range_m must be finite positive metres'));distance=None
    return dict(rows=valid,range_m=distance,actions=actions,errors=errors)


def capture(comp,dest,world,index):
    accepted,ranges,sources,rejected=capture_legacy(comp,dest,world,index);audit=[]
    groups=world['groups']
    for j,g in enumerate(groups):
        members=[o for o in world['objects'] if o['group']==g['key']]
        files=list((comp/'calls').glob(f'layout_{index:02d}_named_group_{j:02d}_pass_*.json'))
        files+=list((dest/'named_recovery_calls_v2').glob(f'layout_{index:02d}_named_group_{j:02d}_pass_*.json'))
        files=sorted(files,key=lambda p:(str(p.parent),p.stem.replace('_repair',''),p.stem.endswith('_repair')))
        for path in files:
            if path.name.endswith(('.request.json','.generation.json','.validation.json','.response.json','.transport_error.json')):continue
            try:result=response(read(path),g,members)
            except ValueError as exc:rejected.append(dict(source=str(path),error=str(exc)));continue
            added=[]
            for oid,row in result['rows'].items():
                if oid not in accepted:accepted[oid]=row;sources[oid]=str(path);added.append(oid)
                elif accepted[oid]!=row:result['errors'].append(dict(id=oid,error='Later answer differs from preserved valid row; original retained'))
            if g['key'] not in ranges and result['range_m'] is not None:ranges[g['key']]=result['range_m'];sources['range:'+g['key']]=str(path)
            audit.append(dict(source=str(path),added_ids=added,actions=result['actions'],errors=result['errors']))
    return accepted,ranges,sources,rejected,audit


def placement(case,observed,world,plane,dimensions,board,ask,dest,index,reuse=None):
    accepted,ranges,sources,rejected,audit=deepcopy(reuse or ({},{},{},[],[]));initially_kept=sorted(accepted);completed=[]
    def save():
        write(dest/'coordinate_progress_v3.json',dict(interface=INTERFACE,accepted=accepted,ranges=ranges,sources=sources,
            initially_reused_ids=initially_kept,rejected_legacy_rows=rejected,response_audit=audit,inventory_changed=False,dimensions_changed=False))
    save()
    for j,g in enumerate(world['groups']):
        gid=g['key'];members=[o for o in world['objects'] if o['group']==gid]
        missing=[o['id'] for o in members if o['id'] not in accepted]
        batches=[missing[k:k+BATCH] for k in range(0,len(missing),BATCH)]
        # Independent range-only request if all coordinates were already valid.
        if not batches and gid not in ranges:batches=[[]]
        for k,batch in enumerate(batches):
            errors=[]
            for attempt in range(2):
                wanted=[oid for oid in batch if oid not in accepted];need_range=gid not in ranges
                if not wanted and not need_range:break
                requested=[dict(object_id=o['id'],attachment=o['attachment'],required_named_fields=contract(o)) for o in members if o['id'] in wanted]
                prompt=('Fill only the requested missing numeric fields of this FIXED world. '
                    'Return ONE flat JSON object with group_id, objects (a LIST of named object records), '
                    'and range_m only if requested. Do not use a group-key wrapper or split roots/hosted lists. '
                    'Every requested ID once. If objects are not requested return objects: []. '
                    'Each object record must contain its id, yaw_deg and exactly the coordinate fields listed for that ID. '
                    'Never use null, guess a new ID, omit a requested object, change hosts/sizes/regions, or return dimensions. '
                    'root_offset_m is [dx,dy,dz] in camera axes relative to the group origin, not absolute XYZ. '
                    'host_offset_uv is [u,v] in intrinsic host axes; dx=u*host_width/2 and dz=v*host_depth/2. '
                    'Predicted hosts use their yaw; observed hosts use camera AABB axes. yaw_deg is absolute camera-Y degrees. '
                    'Floor/top Y is derived; shelf/seat require surface_fraction in [0,1]; hang requires nonnegative hanging_drop_m. '
                    'Use complete world context and accepted coordinates for coherent placement and same-host siblings. '
                    '\nREQUEST: '+json.dumps(dict(group_id=gid,object_ids=wanted,range_required=need_range,frozen_range_m=ranges.get(gid)))+
                    '\nREQUESTED_OBJECT_CONTRACTS: '+json.dumps(requested)+
                    '\nPREVIOUS_FIELD_ERRORS: '+json.dumps(errors)+
                    '\nACCEPTED_COORDINATES_READ_ONLY: '+json.dumps(accepted)+
                    '\nFROZEN_DIMENSIONS: '+json.dumps(dimensions)+'\nFIXED_WORLD: '+json.dumps(world,ensure_ascii=False)+
                    '\nFROZEN_OBSERVATIONS: '+json.dumps(observed,ensure_ascii=False)+'\nLEVEL_PRIOR: '+json.dumps(plane)+
                    '\nALREADY_PREDICTED_GROUPS: '+json.dumps(completed))
                stage=f'layout_{index:02d}_coordinates_v3_group_{j:02d}_batch_{k:02d}_pass_{attempt}'
                result=ask(stage,board,prompt,lambda v:response(v,g,members));errors=list(result['errors'])
                for oid,row in result['rows'].items():
                    if oid in accepted:
                        if accepted[oid]!=row:errors.append(dict(id=oid,error='Frozen valid coordinates cannot be overwritten'))
                    elif oid in wanted:accepted[oid]=row;sources[oid]=stage
                    else:errors.append(dict(id=oid,error='Unrequested coordinate row not applied'))
                if need_range and result['range_m'] is not None:ranges[gid]=result['range_m'];sources['range:'+gid]=stage
                elif not need_range and result['range_m'] is not None and result['range_m']!=ranges[gid]:errors.append(dict(id=gid,error='Frozen range cannot be overwritten'))
                errors+=[dict(id=oid,error='Requested object omitted') for oid in wanted if oid not in accepted]
                if gid not in ranges:errors.append(dict(id=gid,error='Positive numeric range_m still required'))
                audit.append(dict(stage=stage,requested_ids=wanted,actions=result['actions'],errors=errors));save()
            unresolved=[oid for oid in batch if oid not in accepted]
            if unresolved or gid not in ranges:raise ValueError('Coordinate transport budget exhausted; unresolved '+json.dumps(dict(group=gid,objects=unresolved,range_missing=gid not in ranges))+'; valid rows preserved')
        completed.append(dict(id=gid,range_m=ranges[gid],objects=[accepted[o['id']] for o in members]))
    raw=dict(groups=completed);layout=geometry.placement_answer(raw,world,observed,dimensions,case['hfov'])
    write(dest/'coordinate_stage_v3.json',dict(interface=INTERFACE,raw_placement=raw,initially_reused_ids=initially_kept,
        sources=sources,response_audit=audit,inventory_changed=False,dimensions_frozen=True))
    return layout


def infer_layout(case,observed,world,plane,board,ask,dest,index):
    dims=ask(f'layout_{index:02d}_dimensions',None,geometry.dimensions_prompt(world),lambda v:geometry.dimensions_answer(v,world))
    write(dest/'dimension_stage.json',dict(dimensions=dims,dimensions_frozen=True))
    return placement(case,observed,world,plane,dims,board,ask,dest,index)


def recover_layouts(root,branch,backend,provider):
    from .assembly_organized_repair import materialize
    from .assembly_world import page
    comp=root/branch/backend/'completion';summary=read(comp/'summary.json');config=read(root/branch/'experiment.json')
    history=comp/'summary_before_coordinate_transport_v3.json'
    if history.exists():raise FileExistsError('This bounded v3 coordinate recovery has already run')
    write(history,summary);outcomes=[]
    for failure in list(summary['failures']):
        i=failure['world'];dest=comp/f'world_{i:02d}'
        if (dest/'G_full.json').exists():continue
        try:
            if not str(failure['stage']).startswith('layout_') or not (dest/'dimension_stage.json').exists() or (dest/'execution.json').exists():raise ValueError('Only pre-compiler coordinate failures with saved dimensions are eligible')
            case=read(comp/'case.json');observed=read(comp/'G_obs.json');world=read(dest/'semantic_world.json');plane=read(comp/'support_plane.json');dims=read(dest/'dimension_stage.json')['dimensions']
            calls=dest/'coordinate_transport_calls_v3';calls.mkdir()
            if hasattr(provider,'max_calls'):provider.max_calls=max(provider.max_calls,4*sum(max(1,math.ceil(sum(o['group']==g['key'] for o in world['objects'])/BATCH)) for g in world['groups']))
            def ask(name,image,prompt,validator):
                print(json.dumps(dict(backend=backend,stage=name)),flush=True)
                return ask_valid(SeededProvider(provider,stage_seed(config['seed'],name)),None,prompt,calls/(name+'.txt'),validator,config['max_tokens'])
            reuse=capture(comp,dest,world,i)
            write(dest/'coordinate_reuse_v3.json',dict(reused_ids=sorted(reuse[0]),ranges=reuse[1],sources=reuse[2],rejected_legacy_rows=reuse[3],response_audit=reuse[4]))
            layout=placement(case,observed,world,plane,dims,None,ask,dest,i,reuse)
            entry=dict(branch=branch,backend=backend,world=dest.name,index=i,initial_reply=f'{dest.name}/coordinate_stage_v3.json')
            passed,record=materialize(root,entry,layout,position_note='仅规范化坐标回复容器并补问缺失字段；原库存、因果机制、尺寸与已合法原始坐标保留，使用原数值编译器。')
            if passed:
                summary['worlds'].append(record);summary['worlds'].sort(key=lambda w:w['folder']);summary['failures']=[f for f in summary['failures'] if f['world']!=i]
            else:
                summary['blocked_worlds'].append(record)
                review=read(dest/'numeric_review.json')
                current=dict(world=i,stage='compile_'+str(i),proposed_instances=len(world['objects']),
                    error='Numeric geometry contract failed; residual overlap pairs='+str(len(review['program_solver']['final_overlaps'])),original_failure=failure)
                summary['failures']=[current if f['world']==i else f for f in summary['failures']]
            result=dict(world=i,status='exported' if passed else 'blocked_numeric',original_failure=failure,reused_coordinate_ids=sorted(reuse[0]))
        except Exception as exc:
            result=dict(world=i,status='failed',error=type(exc).__name__+': '+str(exc),original_failure=failure)
            (dest/'coordinate_transport_error_v3.txt').write_text(traceback.format_exc())
        outcomes.append(result);write(dest/'coordinate_transport_result_v3.json',result)
    summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else 'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',
        world_state_ready=bool(summary['worlds']),coordinate_transport_recovery=outcomes)
    write(comp/'summary.json',summary);page(comp,summary);return summary
