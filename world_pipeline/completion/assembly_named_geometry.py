"""Named numeric fields, immutable valid rows, bounded coordinate-only recovery.

This is a response interface change. Inventory, SCM, sizes and the physical
compiler are the original v13 implementations. Ambiguous old rows are rejected,
never truncated or reinterpreted as another coordinate system.
"""
from copy import deepcopy
import json
from pathlib import Path
import traceback

from ..common import read,write
from . import assembly_organized_geometry as geometry
from .assembly_io_recovery import columns
from .assembly_experiment import SeededProvider,stage_seed
from .world_scm_runner import ask_valid

PARAMETERIZATION=geometry.PARAMETERIZATION
compile_layout=geometry.compile_layout
annotate_state=geometry.annotate_state
INTERFACE='grouped_named_numeric_fields_v2'


def contract(obj):
    a=obj['attachment'];fields=dict(id='exact ID',yaw_deg='finite absolute degrees')
    if not a['host']:fields['root_offset_m']='three finite relative metres [dx,dy,dz]'
    else:
        fields['host_offset_uv']='two finite dimensionless offsets [u,v]'
        if a['kind'] in ('seat','shelf'):fields['surface_fraction']='finite in [0,1]'
        elif a['kind']=='hang':fields['hanging_drop_m']='finite nonnegative metres'
        elif a['kind']!='top':fields['host_offset_y_m']='finite signed metres'
    return fields


def row_answer(row,obj):
    fields=contract(obj)
    if not isinstance(row,dict) or set(row)!=set(fields) or row.get('id')!=obj['id']:
        raise ValueError(obj['id']+' requires exactly named fields '+json.dumps(fields))
    if not geometry.number(row['yaw_deg']):raise ValueError(obj['id']+'.yaw_deg must be finite')
    key='host_offset_uv' if obj['attachment']['host'] else 'root_offset_m'
    if not geometry.vector(row[key],2 if key=='host_offset_uv' else 3):raise ValueError(obj['id']+'.'+key+': wrong coordinate vector length or nonfinite value')
    for key in ('surface_fraction','hanging_drop_m','host_offset_y_m'):
        if key in row and not geometry.number(row[key]):raise ValueError(obj['id']+'.'+key+' must be finite')
    if 'surface_fraction' in row and not 0<=row['surface_fraction']<=1:raise ValueError('surface_fraction must be in [0,1]')
    if 'hanging_drop_m' in row and row['hanging_drop_m']<0:raise ValueError('hanging_drop_m must be nonnegative')
    return deepcopy(row)


def legacy_row(row,obj):
    names=columns(obj)
    if not isinstance(row,list) or len(row)!=len(names) or row[0]!=obj['id']:
        raise ValueError(obj['id']+': expected '+str(len(names))+' columns '+json.dumps(names)+'; ambiguous extra columns cannot be removed')
    if any(not geometry.number(v) for v in row[1:]):raise ValueError('Legacy row must contain finite numbers')
    result=dict(id=row[0],yaw_deg=row[1])
    if not obj['attachment']['host']:result['root_offset_m']=row[2:5]
    else:
        result['host_offset_uv']=row[2:4]
        if len(names)==5:result[names[-1]]=row[4]
    return row_answer(result,obj)


def capture_legacy(comp,dest,world,index):
    """Keep the first unambiguous valid row, with its exact source provenance."""
    accepted={};ranges={};sources={};rejected=[]
    paths=list((comp/'calls').glob(f'layout_{index:02d}_numeric_group_*.json'))
    paths+=list((dest/'compact_recovery_calls').glob(f'layout_{index:02d}_numeric_group_*.json'))
    # Sort original before correction; never silently replace a valid prediction.
    paths=sorted(paths,key=lambda p:(str(p.parent),p.stem.replace('_repair',''),p.stem.endswith('_repair')))
    for path in paths:
        if path.name.endswith(('.request.json','.validation.json','.generation.json','.response.json','.transport_error.json')):continue
        value=read(path)
        if not isinstance(value,dict):continue
        gid=value.get('group_id');members={o['id']:o for o in world['objects'] if o['group']==gid}
        if not members or not isinstance(value.get('placements'),list):continue
        if gid not in ranges and geometry.number(value.get('range_m')) and value['range_m']>0:ranges[gid]=value['range_m'];sources['range:'+gid]=str(path)
        rows=value['placements'];ids=[r[0] for r in rows if isinstance(r,list) and r and isinstance(r[0],str)]
        for row in rows:
            if not isinstance(row,list) or not row or not isinstance(row[0],str) or row[0] not in members or ids.count(row[0])!=1:continue
            oid=row[0]
            if oid in accepted:continue
            try:accepted[oid]=legacy_row(row,members[oid]);sources[oid]=str(path)
            except ValueError as exc:rejected.append(dict(source=str(path),id=oid,error=str(exc)))
    return accepted,ranges,sources,rejected


def patch_answer(value,group,members,missing,need_range):
    required={'group_id','roots','hosted'}|({'range_m'} if need_range else set())
    if not isinstance(value,dict) or set(value)!=required or value['group_id']!=group['key']:raise ValueError('Return exactly keys '+str(sorted(required))+' for '+group['key'])
    if need_range and (not geometry.number(value['range_m']) or value['range_m']<=0):raise ValueError('range_m must be finite positive metres')
    expected={o['id']:o for o in members if o['id'] in missing};rows=[]
    for label,hosted in (('roots',False),('hosted',True)):
        part=value[label]
        if not isinstance(part,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in part):raise ValueError(label+' must be object records with exact IDs')
        for r in part:
            if r['id'] not in expected or bool(expected[r['id']]['attachment']['host'])!=hosted:raise ValueError(label+': wrong or frozen ID '+r['id'])
        rows+=part
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Return each requested missing ID exactly once: '+json.dumps(list(expected)))
    valid={};errors=[]
    for r in rows:
        try:valid[r['id']]=row_answer(r,expected[r['id']])
        except ValueError as exc:errors.append(dict(id=r['id'],error=str(exc)))
    return dict(rows=valid,range_m=value.get('range_m'),errors=errors)


def placement(case,observed,world,plane,dimensions,board,ask,dest,index,reuse=None):
    accepted,ranges,sources,rejected=deepcopy(reuse or ({},{},{},[]));completed=[]
    initially_kept=sorted(accepted)
    for j,group in enumerate(world['groups']):
        members=[o for o in world['objects'] if o['group']==group['key']];gid=group['key']
        for attempt in range(2):
            missing=[o['id'] for o in members if o['id'] not in accepted];need_range=gid not in ranges
            if not missing and not need_range:break
            required={o['id']:contract(o) for o in members if o['id'] in missing}
            prompt=('Predict actual numeric coordinates for the requested missing objects in this fixed 360-degree world. '
                'Return group_id, roots (records for unhosted objects), hosted (records for attached objects)'+(', range_m (finite positive horizontal camera distance). ' if need_range else '. Do not return range_m; it is frozen. ')+
                'Each record uses exactly the NAMED fields in REQUIRED_FIELDS_BY_ID. Return empty lists where appropriate. '
                'Do not output row arrays, null, placeholders, dimensions, or extra fields. No new, removed or renamed objects. '
                'yaw_deg is absolute camera-Y yaw. root_offset_m is [dx,dy,dz] from the group origin in camera axes, NOT absolute XYZ. '
                'host_offset_uv is [u,v] in intrinsic host axes, dx=u*host_width/2 and dz=v*host_depth/2; zero is host centre. '
                'Predicted hosts use their yaw; observed hosts use camera AABB axes. Floor/top Y is derived; '
                'shelf/seat surface_fraction is in [0,1]; hanging_drop_m is nonnegative. '
                'Use the full inventory for coherent placement and arrange same-host siblings jointly. '
                'Preserve sampled original-camera regions, frozen hosts and sizes, and all accepted coordinates. '
                '\nGROUP_ID: '+gid+'\nREQUIRED_FIELDS_BY_ID: '+json.dumps(required)+
                '\nFROZEN_GROUP_RANGE: '+json.dumps(ranges.get(gid))+
                '\nACCEPTED_COORDINATES_READ_ONLY: '+json.dumps(accepted)+
                '\nFROZEN_DIMENSIONS: '+json.dumps(dimensions)+'\nFIXED_WORLD: '+json.dumps(world,ensure_ascii=False)+
                '\nFROZEN_OBSERVATIONS: '+json.dumps(observed,ensure_ascii=False)+'\nLEVEL_PRIOR: '+json.dumps(plane)+
                '\nALREADY_PREDICTED_GROUPS: '+json.dumps(completed))
            stage=f'layout_{index:02d}_named_group_{j:02d}_pass_{attempt}'
            result=ask(stage,board,prompt,lambda v:patch_answer(v,group,members,missing,need_range))
            for oid,row in result['rows'].items():accepted[oid]=row;sources[oid]=stage
            if need_range:ranges[gid]=result['range_m'];sources['range:'+gid]=stage
            write(dest/'named_numeric_progress_v2.json',dict(accepted=accepted,ranges=ranges,sources=sources,initially_reused_ids=initially_kept,
                rejected_legacy_rows=rejected,last_errors=result['errors'],inventory_changed=False,dimensions_changed=False))
        missing=[o['id'] for o in members if o['id'] not in accepted]
        if missing or gid not in ranges:raise ValueError('Named numeric coordinate budget exhausted for '+gid+'; unresolved IDs '+json.dumps(missing)+'; accepted coordinates preserved')
        completed.append(dict(id=gid,range_m=ranges[gid],objects=[accepted[o['id']] for o in members]))
    raw=dict(groups=completed);layout=geometry.placement_answer(raw,world,observed,dimensions,case['hfov'])
    write(dest/'named_numeric_stage_v2.json',dict(interface=INTERFACE,raw_placement=raw,dimensions_frozen=True,inventory_changed=False,
        initially_reused_ids=initially_kept,sources=sources,rejected_legacy_rows=rejected))
    return layout


def infer_layout(case,observed,world,plane,board,ask,dest,index):
    dims=ask(f'layout_{index:02d}_dimensions',None,geometry.dimensions_prompt(world),lambda v:geometry.dimensions_answer(v,world))
    write(dest/'dimension_stage.json',dict(dimensions=dims,dimensions_frozen=True))
    return placement(case,observed,world,plane,dims,board,ask,dest,index)


def recover_layouts(root,branch,backend,provider):
    from .assembly_organized_repair import materialize
    from .assembly_world import page
    comp=root/branch/backend/'completion';summary=read(comp/'summary.json');config=read(root/branch/'experiment.json')
    history=comp/'summary_before_named_recovery_v2.json'
    if history.exists():raise FileExistsError('This bounded named-coordinate recovery was already attempted; inspect preserved errors')
    write(history,summary);outcomes=[]
    for failure in list(summary['failures']):
        i=failure['world'];dest=comp/f'world_{i:02d}'
        if (dest/'G_full.json').exists():continue
        try:
            if not str(failure['stage']).startswith('layout_') or not (dest/'dimension_stage.json').exists() or (dest/'execution.json').exists():raise ValueError('Only pre-compiler coordinate failures with saved sizes are eligible')
            case=read(comp/'case.json');observed=read(comp/'G_obs.json');world=read(dest/'semantic_world.json');plane=read(comp/'support_plane.json');dims=read(dest/'dimension_stage.json')['dimensions']
            calls=dest/'named_recovery_calls_v2';calls.mkdir()
            if hasattr(provider,'max_calls'):provider.max_calls=max(provider.max_calls,4*len(world['groups']))
            def ask(name,image,prompt,validator):
                print(json.dumps(dict(backend=backend,stage=name)),flush=True)
                return ask_valid(SeededProvider(provider,stage_seed(config['seed'],name)),None,prompt,calls/(name+'.txt'),validator,config['max_tokens'])
            reuse=capture_legacy(comp,dest,world,i)
            layout=placement(case,observed,world,plane,dims,None,ask,dest,i,reuse)
            # Existing exporter reads this file for provenance but does not parse
            # its contents; it compiles the strictly validated layout above.
            entry=dict(branch=branch,backend=backend,world=dest.name,index=i,initial_reply=f'{dest.name}/named_numeric_stage_v2.json')
            passed,record=materialize(root,entry,layout,position_note='仅修复无效坐标字段；原库存、机制概率、尺寸及已合法坐标固定。数值通过不等于语义认证。')
            if passed:
                summary['worlds'].append(record);summary['worlds'].sort(key=lambda w:w['folder']);summary['failures']=[f for f in summary['failures'] if f['world']!=i]
            else:summary['blocked_worlds'].append(record)
            result=dict(world=i,status='exported' if passed else 'blocked_numeric',original_failure=failure,reused_coordinate_ids=sorted(reuse[0]))
        except Exception as exc:
            result=dict(world=i,status='failed',error=type(exc).__name__+': '+str(exc),original_failure=failure)
            (dest/'named_recovery_error_v2.txt').write_text(traceback.format_exc())
        outcomes.append(result);write(dest/'named_recovery_result_v2.json',result)
    summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else 'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',
        world_state_ready=bool(summary['worlds']),named_coordinate_recovery=outcomes)
    write(comp/'summary.json',summary);page(comp,summary);return summary
