"""Bounded visible-inventory chunks and compact numeric rows; SCM unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import traceback

from PIL import Image,ImageDraw

from ..common import read,write
from . import assembly_organized_geometry as geometry
from .fresh_observer import PROMPT,answer,state_from_answer
from .assembly_experiment import SeededProvider,stage_seed
from .world_scm_runner import ask_valid

PARAMETERIZATION=geometry.PARAMETERIZATION
compile_layout=geometry.compile_layout
annotate_state=geometry.annotate_state


def visible_key(obj):return (obj['name'].strip().lower(),tuple(obj['bbox_1000']))


TRANSPORT_PAGE_SIZE=8
INVENTORY_LIMIT=64


class InventoryBudgetError(ValueError):
    """A computation limit cannot be repaired by deleting visual evidence."""


def chunk_answer(value,accepted):
    if not isinstance(value,dict) or type(value.get('done')) is not bool:raise ValueError('objects, evidence_description and boolean done required')
    if isinstance(value.get('objects'),list) and len(value['objects'])>INVENTORY_LIMIT:
        raise InventoryBudgetError('Visible inventory exceeded the existing 64-instance computation budget; original reply retained, no truncation')
    clean=answer(value)
    # Eight is a requested transport page size, not a visual validity test.
    # Every row still needs the same schema, identity and total-budget checks.
    if not clean['objects'] and not value['done']:raise ValueError('An empty response must explicitly finish with done=true')
    old={visible_key(o) for o in accepted};seen=set()
    for obj in clean['objects']:
        key=visible_key(obj)
        if key in old or key in seen:raise ValueError('Repeated exact visible instance; do not relist accepted objects')
        seen.add(key)
    if len(accepted)+len(clean['objects'])>INVENTORY_LIMIT:raise InventoryBudgetError('Visible inventory exceeded the existing 64-instance computation budget; no object dropped to fit')
    return dict(clean,done=value['done'])


def observe_chunked(provider,root,backend='qwen',max_tokens=8192):
    config=read(root/'experiment.json');out=root/backend/'observer2d';out.mkdir(parents=True,exist_ok=False)
    with Image.open(root/'input.png') as im:image=im.convert('RGB')
    accepted=[];descriptions=[];calls=0;rejected_nonempty_inventory=False;transport=[]
    try:
        for index in range(8):
            prompt=(PROMPT+'\nINCREMENTAL VISIBLE INVENTORY: return at most EIGHT additional distinct visible instances in this response. '
                'Do not continue repetitive patterns or invent objects to fill a budget. '
                'Use a concrete ordinary noun for category, such as sofa or lamp, rather than furniture or decor. '
                'Add boolean done: true if no other identifiable visible instances remain, false if another chunk is needed. '
                'Return objects, evidence_description (one short sentence), done. Empty objects with done=true is legal. '
                'Previously accepted objects are read-only and must not be repeated. No hidden objects or surface inventory.\nACCEPTED_VISIBLE_OBJECTS: '+json.dumps(accepted,ensure_ascii=False))
            value=None
            for attempt in range(2):
                path=out/f'chunk_{index:02d}_{attempt}.txt';calls+=1
                raw=None
                try:
                    raw=provider.ask(image,prompt,path,seed=3407+index*2+attempt,max_tokens=min(max_tokens,2048),profile='completion',sample=True)
                    value=chunk_answer(raw,accepted)
                    if not accepted and not value['objects'] and rejected_nonempty_inventory:
                        raise ValueError('Contradictory empty inventory after a rejected nonempty visible-object response; no observation graph exported. Resolve the original duplicate/identity errors instead of erasing the inventory.')
                    break
                except InventoryBudgetError as exc:
                    write(path.with_suffix('.validation.json'),dict(error=str(exc),error_kind='inventory_budget',stochastic_retry=False))
                    raise
                except ValueError as exc:
                    parsed_nonempty=False;repeated=False
                    if isinstance(raw,dict):
                        try:
                            records=answer(raw)['objects'];parsed_nonempty=bool(records);rejected_nonempty_inventory|=parsed_nonempty
                            keys=[visible_key(o) for o in records];repeated=len(set(keys))!=len(keys) or bool(set(keys)&{visible_key(o) for o in accepted})
                        except ValueError:pass
                    write(path.with_suffix('.validation.json'),dict(error=str(exc),error_kind='duplicate_instance' if repeated else 'schema_or_incomplete_reply'))
                    if parsed_nonempty and repeated:
                        # Do not discard valid boxes because some entries were
                        # repeated. Reconcile a complete image inventory in a
                        # separate task, rather than asking for more "done".
                        from .assembly_observer_recovery import reconcile
                        return reconcile(provider,root,out,backend,image,config,accepted,raw,calls,max_tokens)
                    if attempt:raise
                    # Do not paste a truncated/repetitive rejected answer into
                    # the next context: the original failure repeated it verbatim.
                    prompt+='\nFORMAT CORRECTION: '+str(exc)[:600]+' Return a compact current chunk only, close the JSON, and explicitly set done.'
                    if parsed_nonempty:
                        prompt+=' Preserve the previously proposed visible instances while repairing the reported field error. Return object records, not workflow or acceptance status.'
            start=len(accepted);count=len(value['objects'])
            transport.append(dict(chunk=index,reply_objects=count,requested_page_size=TRANSPORT_PAGE_SIZE,
                soft_page_overflow=count>TRANSPORT_PAGE_SIZE,
                pages=[dict(object_indices=list(range(start+k,start+min(k+TRANSPORT_PAGE_SIZE,count))))
                       for k in range(0,count,TRANSPORT_PAGE_SIZE)]))
            accepted+=value['objects'];descriptions.append(value['evidence_description'])
            write(out/'accepted_chunks.json',dict(objects=accepted,done=value['done'],chunks=index+1))
            write(out/'transport_audit.json',dict(responses=transport,inventory_limit=INVENTORY_LIMIT,
                instances_truncated=False,extra_inference_for_page_overflow=False))
            if value['done']:break
        else:raise ValueError('Visible inventory did not finish within eight chunks; no partial G_2d exported')
        value=dict(objects=accepted,evidence_description=' '.join(dict.fromkeys(descriptions)))
        state,mapping=state_from_answer(value,config,backend)
        write(out/'G_2d.json',state);write(out/'labels.json',mapping)
        write(out/'grounding.json',dict(instances=[dict(id=o['track_id'],variant=r['name'],on_floor=r['on_floor']) for o,r in zip(state['objects'],accepted)],evidence_description=value['evidence_description']))
        preview=image.copy();draw=ImageDraw.Draw(preview)
        for o in state['objects']:
            b=o['bbox2d_visible_xyxy_norm']['value'];box=[b[0]*image.width,b[1]*image.height,b[2]*image.width,b[3]*image.height]
            draw.rectangle(box,outline='#287db5',width=2);draw.text(tuple(box[:2]),o['track_id']+' '+o['description'],fill='#287db5')
        preview.save(out/'boxes2d.png');summary=dict(status='completed',objects=len(accepted),backend=backend,request_interface='incremental_visible_chunks_lossless_v2',
            actual_calls=calls,maximum_calls=16,chunk_output_budget=2048,inventory_limit_is_computational=True,partial_inventory_exported=False,
            transport_pages=sum(len(r['pages']) for r in transport),soft_page_overflows=sum(r['soft_page_overflow'] for r in transport),
            instances_truncated=False,quality_verified=False)
        write(out/'summary.json',summary);return summary
    except Exception as exc:
        usage=read(out/'reconciliation_usage.json') if (out/'reconciliation_usage.json').exists() else dict(actual_calls=calls)
        write(out/'summary.json',dict(status='failed',error=type(exc).__name__+': '+str(exc),**usage,partial_inventory_exported=False));raise


def columns(obj):
    attachment=obj['attachment'];kind=attachment['kind']
    if not attachment['host']:return ['id','yaw_deg','root_dx_m','root_dy_m','root_dz_m']
    result=['id','yaw_deg','host_u','host_v']
    if kind in ('seat','shelf'):result+=['surface_fraction']
    elif kind=='hang':result+=['hanging_drop_m']
    elif kind!='top':result+=['host_offset_y_m']
    return result


def group_answer(value,group,world,observed,dimensions,hfov):
    if not isinstance(value,dict) or set(value)!={'group_id','range_m','placements'} or value['group_id']!=group['key']:raise ValueError('Return only group_id, range_m and placements for '+group['key'])
    if not geometry.number(value['range_m']) or value['range_m']<=0:raise ValueError('range_m must be finite positive metres')
    objects=[o for o in world['objects'] if o['group']==group['key']];contracts={o['id']:o for o in objects};rows=value['placements']
    if not isinstance(rows,list) or any(not isinstance(r,list) or not r or not isinstance(r[0],str) for r in rows):raise ValueError('placements must be numeric row arrays beginning with an exact object ID')
    ids=[r[0] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(contracts):raise ValueError('Every fixed group member ID once required: '+json.dumps(list(contracts)))
    converted=[]
    for row in rows:
        obj=contracts[row[0]];names=columns(obj)
        if len(row)!=len(names) or any(not geometry.number(v) for v in row[1:]):raise ValueError(row[0]+' requires finite numeric columns '+json.dumps(names)+'; no nulls or strings')
        objrow=dict(id=row[0],yaw_deg=row[1]);a=obj['attachment']
        if not a['host']:objrow['root_offset_m']=row[2:5]
        else:
            objrow['host_offset_uv']=row[2:4]
            if len(names)==5:objrow[names[-1]]=row[4]
        converted.append(objrow)
    raw=dict(id=group['key'],range_m=value['range_m'],objects=converted)
    subset=dict(world,groups=[group],objects=objects)
    geometry.placement_answer(dict(groups=[raw]),subset,observed,dimensions,hfov)
    return raw


def placement(case,observed,world,plane,dimensions,board,ask,dest,index):
    completed=[]
    for j,group in enumerate(world['groups']):
        objects=[o for o in world['objects'] if o['group']==group['key']]
        prompt=('Predict numeric placement for this fixed group as part of the ENTIRE fixed world. '
            'Return exactly three keys: group_id (the supplied ID), range_m (finite positive horizontal camera distance), '
            'placements (an array of rows). Every row starts with its exact ID followed only by the numeric columns specified below. '
            'There is no blank output template to copy. Estimate actual numbers; never output null or placeholder text. '
            'Do not add/drop objects, change dimensions, hosts, or the protected original-camera region. '
            'yaw is absolute camera-Y degrees. Root dx,dy,dz are camera-axis displacements from the group origin, not absolute observed coordinates. '
            'Host u,v are dimensionless relative offsets in intrinsic host axes, dx=u*host_width/2 and dz=v*host_depth/2. '
            'Observed hosts use camera AABB axes; predicted hosts use their yaw. '
            'Floor/top Y is derived; shelf/seat fraction must be in [0,1], hanging_drop_m nonnegative. '
            'Arrange same-host siblings together. Use the entire inventory and completed groups for coherent relative placement. '
            '\nGROUP_ID: '+group['key']+'\nROW_COLUMNS_BY_ID: '+json.dumps({o['id']:columns(o) for o in objects})+
            '\nFROZEN_DIMENSIONS: '+json.dumps(dimensions)+'\nFIXED_WORLD: '+json.dumps(world,ensure_ascii=False)+
            '\nFROZEN_OBSERVATIONS: '+json.dumps(observed,ensure_ascii=False)+'\nLEVEL_PRIOR: '+json.dumps(plane)+
            '\nALREADY_PREDICTED_GROUPS: '+json.dumps(completed))
        raw=ask(f'layout_{index:02d}_numeric_group_{j:02d}',board,prompt,lambda v,g=group:group_answer(v,g,world,observed,dimensions,case['hfov']))
        completed.append(raw)
    raw=dict(groups=completed);layout=geometry.placement_answer(raw,world,observed,dimensions,case['hfov'])
    write(dest/'compact_numeric_stage.json',dict(interface='grouped_numeric_rows_v1',raw_placement=raw,dimensions_frozen=True,inventory_changed=False))
    return layout


def infer_layout(case,observed,world,plane,board,ask,dest,index):
    dims=ask(f'layout_{index:02d}_dimensions',None,geometry.dimensions_prompt(world),lambda v:geometry.dimensions_answer(v,world))
    write(dest/'dimension_stage.json',dict(dimensions=dims,dimensions_frozen=True))
    return placement(case,observed,world,plane,dims,board,ask,dest,index)


def recover_layouts(root,branch,backend,provider):
    """Repair preserved failed placement only; never re-sample SCM or sizes."""
    from .assembly_organized_repair import inputs,materialize
    from .assembly_world import page
    comp=root/branch/backend/'completion';summary=read(comp/'summary.json');config=read(root/branch/'experiment.json')
    history=comp/'summary_before_compact_recovery.json'
    if history.exists():raise FileExistsError('This fixed-budget coordinate recovery was already attempted')
    write(history,summary);outcomes=[]
    for failure in list(summary['failures']):
        i=failure['world'];dest=comp/f'world_{i:02d}'
        if (dest/'G_full.json').exists():continue
        entry=dict(branch=branch,backend=backend,world=dest.name,index=i,initial_reply=f'calls/layout_{i:02d}_placement.json')
        try:
            if not str(failure['stage']).startswith('layout_') or not (dest/'dimension_stage.json').exists() or (dest/'execution.json').exists():raise ValueError('Only saved pre-compiler layout failures with frozen dimensions are eligible')
            _,_,case,observed,world,plane,dims,_=inputs(root,entry);calls=dest/'compact_recovery_calls';calls.mkdir()
            def ask(name,image,prompt,validator):
                selected=SeededProvider(provider,stage_seed(config['seed'],name))
                return ask_valid(selected,None,prompt,calls/(name+'.txt'),validator,config['max_tokens'])
            layout=placement(case,observed,world,plane,dims,None,ask,dest,i);passed,record=materialize(root,entry,layout)
            if passed:
                summary['worlds'].append(record);summary['worlds'].sort(key=lambda w:w['folder']);summary['failures']=[f for f in summary['failures'] if f['world']!=i]
            else:summary['blocked_worlds'].append(record)
            result=dict(world=i,status='exported' if passed else 'blocked_numeric',original_failure=failure)
        except Exception as exc:
            result=dict(world=i,status='failed',error=type(exc).__name__+': '+str(exc),original_failure=failure)
            (dest/'compact_recovery_error.txt').write_text(traceback.format_exc())
        outcomes.append(result);write(dest/'compact_recovery_result.json',result)
    summary.update(status='worlds_exported_needs_review' if summary['worlds'] and not summary['failures'] else 'partial_layouts_needs_review' if summary['worlds'] else 'no_world_exported',
        world_state_ready=bool(summary['worlds']),coordinate_recovery=outcomes)
    write(comp/'summary.json',summary);page(comp,summary);return summary
