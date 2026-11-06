"""Conditional layout approximation for fixed externally sampled instances."""
from copy import deepcopy
import json

from ..common import write
from .conditional_world_geometry import compile_layout
from .frozen_causal_world import ask_validated
from .surrounding_world import bearing_prompt


def instantiate(sample,case,domain):
    observed={o['id'] for o in case['context']};objects=[];counter=0
    for row in sample['counts']:
        for ordinal in range(row['outside']):
            while True:
                counter+=1;oid=f'scm_new_{counter:04d}'
                if oid not in observed:break
            objects.append(dict(id=oid,name=row['object_type']+f' ({ordinal+1})',object_type=row['object_type'],
                reason='Externally sampled outside-count instance under the declared generative semantic model.',
                placement='Outside the original input frustum; geometry pending.',support=dict(kind='unspecified',target=None)))
    b=next(s['description'] for s in domain['backgrounds'] if s['id']==sample['B'])
    k=next(s['description'] for s in domain['configurations'] if s['id']==sample['K'])
    return dict(id='sample_'+str(sample['seed']),scene_type='hypothesized indoor world',
        scene_hypothesis='Sampled background: '+b+'; sampled furnishing configuration: '+k,
        additions=objects,relations=[],placement_dependency_order=[o['id'] for o in objects],
        semantic_selection_source='external_posterior_count_sampler_not_LM_inventory_selection')


def support_answer(value,plan,case):
    rows=value.get('objects') if isinstance(value,dict) else None
    expected={o['id'] for o in plan['additions']};old={o['id'] for o in case['context']}
    if (not isinstance(rows,list) or any(not isinstance(o,dict) or not isinstance(o.get('id'),str) for o in rows) or len(rows)!=len(expected)
        or {o.get('id') for o in rows}!=expected):raise ValueError('Support assignment must cover every fixed sampled ID exactly once')
    byid={o['id']:o for o in rows};out=deepcopy(plan)
    for obj in out['additions']:
        row=byid[obj['id']];s=row.get('support')
        if not isinstance(s,dict):raise ValueError('Need typed support object')
        s=deepcopy(s);kind=s.get('kind');target=s.get('target')
        if kind in ('wall','ceiling'):
            if target is not None:raise ValueError('Unmeasured wall/ceiling target must be null')
            obj['support_declaration']=deepcopy(s);kind='unspecified'
        if kind not in ('floor','object','unspecified'):raise ValueError('Support kinds floor/object/unspecified/wall/ceiling only')
        if kind=='object':
            if not isinstance(target,str) or target not in old|expected or target==obj['id']:raise ValueError('Unknown/self support reference')
        elif target is not None:raise ValueError('Non-object support target must be null')
        if any(key in row for key in ('name','object_type','category')):raise ValueError('Layout may not change sampled identity')
        placement=row.get('placement')
        if not isinstance(placement,str) or not placement.strip():raise ValueError('Need qualitative off-camera placement')
        obj.update(placement=placement,support=dict(kind=kind,target=target))
    byid={o['id']:o for o in out['additions']};pending=set(expected);order=[]
    while pending:
        ready=sorted(i for i in pending if byid[i]['support']['target'] not in pending)
        if not ready:raise ValueError('Cyclic support dependencies')
        order+=ready;pending.difference_update(ready)
    out['placement_dependency_order']=order
    out['relations']=[dict(source=o['id'],target=o['support']['target'],kind='supported_by',
        reason='Declared layout dependency, not object-existence causal edge.') for o in out['additions'] if o['support']['kind']=='object']
    return out


def solve_batches(model,image,case,observed,plan,mappings,plane,dest,options,batch_size=6):
    ids=plan['placement_dependency_order'];raw_byid={};items=[];review=None;batches=[]
    if not ids:return compile_layout({'objects':[]},plan,mappings,observed,plane,case['hfov'],case['image_size'])
    for start in range(0,len(ids),batch_size):
        current=ids[start:start+batch_size];prefix_ids=set(ids[:start+len(current)])
        prefix=deepcopy(plan);prefix['additions']=[o for o in plan['additions'] if o['id'] in prefix_ids]
        prefix['placement_dependency_order']=ids[:start+len(current)]
        prefix['relations']=[r for r in plan['relations'] if r['source'] in prefix_ids]
        local=deepcopy(plan);local['additions']=[o for o in plan['additions'] if o['id'] in current]
        fixed=None;last=None;attempts=[]
        for attempt in range(2):
            question=bearing_prompt(case,local,plane)
            question+='\nPlace ONLY these batch IDs: '+json.dumps(current)+'. Previously placed supporting instances are fixed: '+json.dumps(items)
            if last is not None:
                question+='\nCorrect current batch bearing/range ONLY. Keep every size fixed. '+json.dumps(dict(fixed_sizes=fixed,checks=last[1]))
            def validate(v):
                if not isinstance(v,dict) or set(v)!={'objects'} or not isinstance(v['objects'],list):raise ValueError('Need exactly objects array')
                if (any(not isinstance(o,dict) or not isinstance(o.get('id'),str) for o in v['objects']) or len(v['objects'])!=len(current)
                    or {o.get('id') for o in v['objects']}!=set(current)):
                    raise ValueError('Each batch ID required exactly once')
                candidate={**raw_byid,**{o['id']:o for o in v['objects']}}
                combined={'objects':[candidate[i] for i in prefix['placement_dependency_order']]}
                sizes={i:raw_byid[i]['size'] for i in raw_byid}
                if fixed is not None:sizes.update(fixed)
                solved=compile_layout(combined,prefix,mappings,observed,plane,case['hfov'],case['image_size'],
                    sizes if len(sizes)==len(prefix_ids) else None)
                return candidate,solved
            try:
                candidate,solved=ask_validated(model,image,question,dest/f'batch_{start:04d}_{attempt}.txt',validate,
                    {**options,'seed':options['seed']+start*3+attempt*2})
            except ValueError as exc:
                if last is None:raise
                attempts.append(dict(attempt=attempt,schema_valid=False,error=str(exc)));break
            last=solved;raw_byid=candidate
            fixed={i:raw_byid[i]['size'] for i in current}
            attempts.append(dict(attempt=attempt,schema_valid=True,contract_passed=solved[1]['geometry_contract_passed']))
            if solved[1]['geometry_contract_passed']:break
        items,review=last;batches.append(dict(ids=current,attempts=attempts))
        write(dest/'layout_progress.json',dict(placed_ids=[o['id'] for o in items],required_ids=ids,batches=batches))
    # Compile the full raw answer once; no coordinate projection or object deletion.
    answer={'objects':[raw_byid[i] for i in ids]};write(dest/'geometry_raw.json',answer)
    items,review=compile_layout(answer,plan,mappings,observed,plane,case['hfov'],case['image_size'])
    write(dest/'layout_batches.json',batches)
    return items,review
