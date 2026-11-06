"""Auditable whole-surroundings proposals; groups are NOT causal edges.

Each alternative is expanded within itself, never unioned with another world.
An explicit edit ledger prevents an unconstrained second generation silently
discarding instances. Frozen-VLM judgements remain hypotheses, not verification.
"""
from copy import deepcopy
import json

from ..common import write
from .causal_world_completion import visible_inventory
from .conditional_world_core import worlds_answer
from .frozen_causal_core import text
from .frozen_causal_world import ask_validated

BROAD={'furniture','object','objects','item','other','decoration','appliance','lighting'}
STRUCTURE={'wall','walls','floor','ceiling','room','empty space'}


def blueprint_answer(value,case):
    groups=value.get('groups') if isinstance(value,dict) else None
    if not isinstance(groups,list) or len(groups)>8:
        raise ValueError('groups must be an array within computational budget 8; [] is allowed')
    known={o['id'] for o in case['context']}; result=[]
    for g in groups:
        if not isinstance(g,dict):raise ValueError('Group must be an object')
        refs=g.get('evidence_ids')
        if not isinstance(refs,list) or any(i not in known for i in refs):raise ValueError('Unknown evidence ID')
        result.append(dict(id=text(g.get('id'),'group.id'),
            **{k:text(g.get(k),'group.'+k) for k in ('description','off_camera_region','assumption')},
            evidence_ids=refs))
    if len({g['id'] for g in result})!=len(result):raise ValueError('Duplicate group ID')
    return dict(groups=result,extent_assumption=text(value.get('extent_assumption'),'extent_assumption'),
                groups_are_causal_variables=False)


def normalize(value,case,context_id,groups,budget,capacity):
    if not isinstance(value,dict) or not isinstance(value.get('worlds'),list):
        raise ValueError('Need worlds array')
    raw=deepcopy(value);byid={g['id'] for g in groups}
    for w in raw.get('worlds',[]):
        if not isinstance(w,dict) or not isinstance(w.get('additions'),list):raise ValueError('World needs additions array')
        for o in w.get('additions',[]):
            if not isinstance(o,dict):raise ValueError('Addition must be an object')
            if o.get('group_id') not in byid:raise ValueError('Addition must reference a proposed group')
            support=o.get('support')
            if isinstance(support,dict) and support.get('kind') in ('wall','ceiling'):
                if support.get('target') is not None:
                    raise ValueError('Wall/ceiling attachment has no measured support instance; target must be null')
                # v3's unspecified branch already denotes wall/hanging support.
                # Preserve the exact declaration. Do not infer floor contact or
                # turn an "against wall" bookshelf into measured wall support.
                o['support_declaration']=deepcopy(support)
                o['support_normalization']='attachment_preserved_geometry_support_unmodeled'
                o['support']=dict(kind='unspecified',target=None)
            kind=text(o.get('object_type'),'object_type').casefold().replace('_',' ').strip()
            name=text(o.get('name'),'name')
            if kind in STRUCTURE or name.casefold().strip() in STRUCTURE:
                raise ValueError('Room envelope is an extent assumption, not a movable object instance')
            if kind in BROAD:o['object_type']=name  # lossless open-name fallback, not a 37-class mapping
    plans=worlds_answer(raw,case,context_id,budget,capacity)
    for w,source in zip(plans,raw['worlds']):
        w['surroundings_groups']=deepcopy(groups)
        for obj,original in zip(w['additions'],source['additions']):
            obj['group_id']=original['group_id']
            for key in ('support_declaration','support_normalization'):
                if key in original:obj[key]=deepcopy(original[key])
    return plans


def as_raw(plan):
    return dict(description=plan['scene_hypothesis'],additions=deepcopy(plan['additions']))


def expand_answer(value,plan,case,context_id,capacity):
    if not isinstance(value,dict) or not isinstance(value.get('additions'),list):
        raise ValueError('Expansion needs additions array, including [] for no further proposals')
    groups=plan['surroundings_groups'];expected={g['id'] for g in groups}
    coverage=value.get('coverage')
    if (not isinstance(coverage,list) or any(not isinstance(r,dict) for r in coverage) or
        len(coverage)!=len(expected) or {r.get('group_id') for r in coverage}!=expected):
        raise ValueError('Explain consideration of every group; coverage is not an occupancy quota')
    for row in coverage:text(row.get('reason'),'coverage.reason')
    text(value.get('stop_reason'),'stop_reason')
    original=as_raw(plan);ids={o['id'] for o in original['additions']}
    if any(not isinstance(o,dict) or o.get('id') in ids for o in value['additions']):raise ValueError('Expansion must use new IDs')
    original['additions']+=deepcopy(value['additions'])
    # Keep existing IDs stable even for alternative w2, w3 (normalizer owns IDs).
    wi=int(plan['id'].rsplit('_w',1)[1]);prefix=[dict(description='unused',additions=[]) for _ in range(wi-1)]
    result=normalize({'worlds':prefix+[original]},case,context_id,groups,wi,capacity)[wi-1]
    if 'extent_assumption' in plan:result['extent_assumption']=plan['extent_assumption']
    if result['additions'][:len(plan['additions'])]!=plan['additions']:
        raise ValueError('Expansion changed an existing instance or its structured reference')
    return result


def review_answer(value,plan,case,context_id,capacity):
    rows=value.get('decisions') if isinstance(value,dict) else None
    source={o['id']:o for o in plan['additions']}
    if (not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows) or
        len(rows)!=len(source) or {r.get('id') for r in rows}!=set(source)):
        raise ValueError('Review must decide each instance exactly once; no silent disappearance')
    byid={r['id']:r for r in rows};kept=[];ledger=[]
    observed={o['id'] for o in case['context']}
    for oid,obj in source.items():
        r=byid[oid];action=r.get('action');reason=text(r.get('reason'),'review.reason')
        if action not in ('retain','revise','withdraw'):raise ValueError('Unknown review action')
        if action=='withdraw':
            if r.get('grounds') not in ('already_visible','same_instance','contradicts_evidence','incompatible_world','architecture_not_object'):
                raise ValueError('Withdrawal needs a substantive ground; uncertainty alone is not contradiction')
            refs=r.get('evidence_ids')
            if not isinstance(refs,list) or any(i not in observed|set(source) or i==oid for i in refs):
                raise ValueError('Invalid review evidence IDs')
            if r['grounds']=='same_instance' and not refs:raise ValueError('same_instance needs a distinct instance reference')
            if r.get('replacement') is not None:raise ValueError('Withdraw cannot conceal a replacement')
        elif action=='revise':
            replacement=r.get('replacement')
            if not isinstance(replacement,dict) or replacement.get('id')!=oid:
                raise ValueError('Revision needs a complete replacement with the same ID')
            kept.append(deepcopy(replacement))
        else:
            if r.get('replacement') is not None:raise ValueError('Retain cannot change instance content')
            kept.append(deepcopy(obj))
        ledger.append(dict(id=oid,action=action,reason=reason,grounds=r.get('grounds'),evidence_ids=r.get('evidence_ids',[])))
    # Re-normalize after a deletion; support edges to removed parents fail explicitly.
    wi=int(plan['id'].rsplit('_w',1)[1]);prefix=[dict(description='unused',additions=[]) for _ in range(wi-1)]
    revised=normalize({'worlds':prefix+[dict(description=plan['scene_hypothesis'],additions=kept)]},
        case,context_id,plan['surroundings_groups'],wi,capacity)[wi-1]
    revised['review_ledger']=ledger
    if 'extent_assumption' in plan:revised['extent_assumption']=plan['extent_assumption']
    revised['pre_review_to_export_id']={old['id']:new['id'] for old,new in zip(kept,revised['additions'])}
    return revised


def build_catalog(model,image,case,contexts,plane,folder,options,per_context,capacity,mode='expanded'):
    """Budget controls number of alternative worlds, never target object count."""
    catalog=[];counts=[]
    from .conditional_world import worlds_prompt,review_prompt
    for ci,c in enumerate(contexts):
        cid=c['id'];base=options['seed']+10+ci*100
        if mode=='legacy':
            raw=ask_validated(model,image,worlds_prompt(case,c,plane,per_context),folder/f'contents_{cid}.txt',
                lambda v:worlds_answer(v,case,cid,per_context,capacity),{**options,'seed':base})
            revised=ask_validated(model,image,review_prompt(case,c,plane,raw,per_context),folder/f'review_{cid}.txt',
                lambda v:worlds_answer(v,case,cid,per_context,capacity),{**options,'seed':base+2})
            write(folder/f'reviewed_{cid}.json',revised)
            for w in revised:
                before=next((r for r in raw if r['id']==w['id']),None)
                counts.append(dict(id=w['id'],initial=len(before['additions']) if before else None,
                                   expanded=None,reviewed=len(w['additions'])))
            catalog+=revised;continue
        common=('Infer one possible EXISTING surrounding world from this local RGB and fixed inventory. '
            'The task is a full 360-degree object state around this original camera. Unseen space is unknown, '
            'not empty because the visible patch is tidy. All new instances must be outside the original '
            'viewing cone, distinct from anything visible in RGB (including detector omissions). '
            'Keep G_obs fixed. No target count and no requirement to occupy every direction or function. '
            'Ordinary optional objects may exist without being indispensable. Separate assumptions from evidence. '
            '\nCONFIGURATION: '+json.dumps(c,ensure_ascii=False)+
            '\nOBSERVED: '+json.dumps(visible_inventory(case),ensure_ascii=False))
        blueprint=ask_validated(model,image,common+
            '\nDescribe a coherent possible organization of the unphotographed portion of this SAME room. '
            'Use functional/spatial groups at room scale, not individual accessories, and state unknown extent. '
            'Groups decompose one world; they are not causal claims, mandatory rooms, or a wishlist. '
            'Return {"extent_assumption":string,"groups":array}. Each group: id,description,off_camera_region,'
            'assumption (strings),evidence_ids (observed IDs or []). Up to 8 groups is a computation limit; '
            '[] is valid for a hypothesis with no further object-bearing groups.',folder/f'blueprint_{cid}.txt',
            lambda v:blueprint_answer(v,case),{**options,'seed':base})
        write(folder/f'blueprint_{cid}.normalized.json',blueprint)
        groups=blueprint['groups']
        schema=(' Each addition: id,name,object_type,group_id,reason,placement,support:{kind,target}. '
            'object_type is a specific ordinary noun (armchair, cabinet, etc.), never just furniture/object. '
            'Support kinds floor, object (target is another valid observed/new ID), unspecified (target null). '
            'Room envelope wall/floor/ceiling belongs in extent assumptions, not the object inventory. '
            'Consider large furniture, secondary objects and objects on NEW off-camera supports jointly; '
            'none is mandatory. Different instances of the same category are allowed. '
            'Do not complete visible surfaces or place new objects inside the original viewing cone. '
            'Reason records why each instance is compatible with this configuration, with uncertainty, '
            'not proof it exists or a requirement that a room is functional. No metric geometry yet.')
        initial=ask_validated(model,image,common+'\nSURROUNDING ORGANIZATION: '+json.dumps(blueprint)+
            f'\nPropose up to {per_context} alternative COMPLETE surrounding inventories under this organization. '
            'Return {"worlds":array}; each world has description and additions (array of ALL new instances). '
            'Each alternative describes the entire unphotographed portion, not a single fragment.'+schema,
            folder/f'contents_{cid}.txt',lambda v:normalize(v,case,cid,groups,per_context,capacity),
            {**options,'seed':base+2})
        for wi,plan in enumerate(initial):
            plan['extent_assumption']=blueprint['extent_assumption']
            write(folder/f'initial_{plan["id"]}.json',plan)
            stem=plan['id'];expanded=ask_validated(model,image,common+
                '\nCURRENT WHOLE-WORLD HYPOTHESIS: '+json.dumps(plan,ensure_ascii=False)+
                '\nCheck overlooked surroundings at both furniture and secondary-object scales within THIS SAME '
                'hypothesis. Propose missing instances only when coherent together with the current inventory. '
                'Return {"additions":array,"coverage":array,"stop_reason":string}. Each coverage entry '
                'has group_id and reason explaining whether that group was considered; this does not require '
                'adding an object. [] additions is valid. Use new IDs. Do not replace/repeat current instances '
                'or concatenate alternative worlds. For additions use the exact per-instance fields below.'+schema,
                folder/f'expand_{stem}.txt',lambda v:expand_answer(v,plan,case,cid,capacity),
                {**options,'seed':base+10+wi*4})
            reviewed=ask_validated(model,image,common+
                '\nREVIEW WORLD: '+json.dumps(expanded,ensure_ascii=False)+
                '\nReturn {"decisions":array}, exactly one decision per new instance: id,action '
                '(retain/revise/withdraw),reason,grounds,evidence_ids,replacement. Retain uses replacement null. '
                'Revise uses a complete replacement object with SAME id and all original schema fields; '
                'correct support or placement before withdrawing a coherent object. Withdraw uses replacement '
                'null and grounds already_visible/same_instance/contradicts_evidence/incompatible_world/'
                'architecture_not_object, with concrete explanation. same_instance needs a referenced '
                'observed/new ID distinct from itself. Other visible-but-undetected evidence may be described '
                'with evidence_ids=[]. Uncertainty or lack of necessity alone is not a contradiction. '
                'Judge joint plausibility, not minimal functionality; do not regenerate a smaller world '
                'silently. This is model critique, not independent semantic verification.',
                folder/f'review_{stem}.txt',lambda v:review_answer(v,expanded,case,cid,capacity),
                {**options,'seed':base+12+wi*4})
            write(folder/f'expanded_{stem}.json',expanded);write(folder/f'reviewed_{stem}.json',reviewed)
            counts.append(dict(id=stem,initial=len(plan['additions']),expanded=len(expanded['additions']),
                reviewed=len(reviewed['additions']),withdrawn=sum(d['action']=='withdraw' for d in reviewed['review_ledger'])))
            catalog.append(reviewed)
    # Preserve alternatives: category count equality cannot establish world equivalence.
    if not any(not w['additions'] for w in catalog):
        catalog.append(dict(id='empty',scene_type='hypothesized indoor scene',scene_hypothesis='No additional distinct off-camera objects.',
            additions=[],relations=[],placement_dependency_order=[],proposal_context_id=None))
    write(folder/'proposal_coverage.json',dict(mode=mode,stage_counts=counts,
        total_catalog_worlds=len(catalog),category_based_deduplication=False,alternatives_unioned=False,
        object_count_is_quality_metric=False))
    return catalog
