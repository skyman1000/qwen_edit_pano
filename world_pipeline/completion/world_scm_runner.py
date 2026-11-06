"""Manual API/local execution: causal semantic world hypotheses -> fixed G_obs + new boxes."""
import argparse
from copy import deepcopy
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
import traceback

from PIL import ImageDraw

from ..common import new_output,read,sha,validate_graph,validate_new_output,write
from .causal_world_completion import check_hashes,visible_inventory
from .conditional_world_geometry import support_plane
from .frozen_causal_world import load_case
from .off_fov_completion import validate_world
from .scene_completion import export_states
from .scene_program import SUPPORTS,program_answer
from .scene_program_world import observation_board,solve_assignment
from .semantic_layout_visual import render
from .world_api import APIRequestError,DEFAULT_API_BASE,DEFAULT_API_MODEL,SiliconFlowJSON
from .world_scm import PROTOCOL,abduct,distribution,enumerate_model,forward,independent_groups,mechanism_strengths,validate as validate_model
from .world_scm_spec import build_model,domain_answer,geometry_answer,grounding_answer

INTERFACE_PROTOCOL='typed_mechanisms_v5'


def source_hashes():
    from .scene_program_world import code_hashes
    return {**code_hashes(),**{str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('world_*.py')}}


def load_checkpoint(output,allow_transport_updates=False,domain_only=False):
    """Read a saved model, or just its validated domain, without provider calls.

    Continuation permits this fix's client/orchestration changes. Semantic SCM,
    domain/geometry compiler sources and input hashes must still match. Replay
    retains its stronger full-source requirement.
    domain_only discards all old CPTs/layouts, including failed responses; it
    reuses only the bound grounding and proposed domain from immutable outputs.
    """
    output=Path(output).resolve();config=read(output/'config.json')
    if config.get('protocol')!=PROTOCOL:raise ValueError('Checkpoint protocol differs')
    manifest=read(output/'artifact_manifest.json')
    actual={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    if actual!=manifest:raise ValueError('Checkpoint artifact manifest changed')
    permitted={str(Path(__file__).resolve()),str(Path(__file__).with_name('world_api.py').resolve())} if allow_transport_updates else set()
    check_hashes({p:h for p,h in config['code_sha256'].items() if p not in permitted})
    case=config['case'];check_hashes(case['source_sha256']);folder=output/case['id']
    grounding=grounding_answer(read(folder/'grounding.json'),case)
    if grounding!=read(folder/'grounding.json'):raise ValueError('Saved grounding normalization differs')
    domain=read(folder/'domain.json');model=None
    # Bind the saved domain to a successful preserved final answer. This keeps
    # original identity, alternatives and budgets fixed during recovery.
    raw_domain=folder/'calls/domain_repair.json'
    if not raw_domain.exists():raw_domain=folder/'calls/domain.json'
    bindings=[]
    rebuilt=domain_response(read(raw_domain),case,config['budgets'],bindings=bindings)
    if rebuilt!=domain:raise ValueError('Saved semantic domain differs from preserved answer')
    if (folder/'domain_binding.json').exists() and read(folder/'domain_binding.json')['bindings']!=bindings:
        raise ValueError('Saved domain reference binding differs from preserved answer')
    if not domain_only:
        model=validate_model(read(folder/'causal_model.json'))
        posterior=enumerate_model(model,{'E':'signature'})
        if posterior!=read(folder/'posterior.json'):raise ValueError('Saved posterior does not replay under unchanged semantic mechanisms')
        if rank_worlds(posterior,domain)!=read(folder/'ranked_semantic_worlds.json'):
            raise ValueError('Saved world ranking differs')
    return dict(output=output,folder=folder,config=config,case=case,grounding=grounding,domain=domain,model=model,domain_bindings=bindings,
        provenance=dict(source_output=str(output),manifest_sha256=sha(output/'artifact_manifest.json'),
            reused_semantic_checkpoint=not domain_only,reused_domain_checkpoint=domain_only,changed_transport_sources=[p for p,h in config['code_sha256'].items()
                if p in permitted and sha(p)!=h]))


def ask_valid(provider,image,prompt,path,validator,max_tokens=8192):
    for attempt in range(2):
        dest=path if not attempt else path.with_name(path.stem+'_repair.txt')
        try:
            answer=provider.ask(image,prompt,dest,max_tokens=max_tokens,profile='completion',sample=True,seed=3407+attempt)
            if not isinstance(answer,dict):
                raise ValueError('The top-level JSON must be an object with the named fields in the request; '
                    f'received {type(answer).__name__}. Do not return a standalone list of descriptions.')
            return validator(answer)
        except ValueError as exc:
            write(dest.with_suffix('.validation.json'),dict(error=str(exc)))
            if attempt:raise
            if dest.exists():prompt+='\nPREVIOUS REJECTED ANSWER (for correction, not an output template):\n'+dest.read_text()
            prompt+='\nCORRECTION REQUIRED: The previous answer FAILED validation. Return the COMPLETE corrected JSON object, '
            prompt+='not an explanation or a partial patch. Copy every required instance ID exactly; '
            prompt+='do not repeat the rejected answer unchanged. Exact interface error: '+str(exc)


def require_instance_ids(value,array_name,expected):
    """Identity is supplied by the program; omissions cannot become inference.

    Report the actual differences so one interface retry can address them.
    No prefix guessing, positional binding, missing-row fabrication or dropping.
    """
    rows=value.get(array_name) if isinstance(value,dict) else None
    if not isinstance(rows,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
        raise ValueError(json.dumps(dict(field=array_name,error='Expected an array of records with string id',required_ids=expected)))
    actual=[r['id'] for r in rows]
    missing=[i for i in expected if i not in actual]
    unknown=[i for i in actual if i not in expected]
    duplicates=sorted({i for i in actual if actual.count(i)>1})
    if missing or unknown or duplicates:
        raise ValueError(json.dumps(dict(field=array_name,required_count=len(expected),returned_count=len(actual),
            missing_ids=missing,unknown_ids=unknown,duplicate_ids=duplicates,required_ids=expected)))


def grounding_response(value,case):
    require_instance_ids(value,'instances',[o['id'] for o in case['context']])
    return grounding_answer(value,case)


def domain_response(value,case,budgets,bindings=None):
    """Bind one support target to the existing compiler's kind/target pair.

    A surface token or an exact instance reference already determines the kind.
    Legacy pairs remain supported, including a host category used as kind ONLY
    when it exactly matches the category of that specifically referenced host.
    Never infer a host from a noun, relocate an object, or change its semantics.
    Append an audit trace only after the complete domain passes validation.
    """
    def object_at(v,path):
        if not isinstance(v,dict):raise ValueError(path+' must be a JSON object')
    def records_at(v,path):
        if not isinstance(v,list):raise ValueError(path+' must be a JSON array of objects')
        for i,row in enumerate(v):object_at(row,f'{path}[{i}]')
        return v
    object_at(value,'domain (required top-level fields: contexts, organizations, groups)')
    for name in ['contexts','organizations','groups']:
        records_at(value.get(name),name)
    bound=deepcopy(value);changes=[];errors=[]
    observed={o['id']:o.get('category') for o in case['context']}
    surfaces=SUPPORTS-{'object'}
    for gi,group in enumerate(bound['groups']):
        path=f'groups[{gi}]'
        parents=group.get('parents',[])
        if not isinstance(parents,list) or any(type(p) is not int for p in parents):
            raise ValueError(path+'.parents must be an array of integer group indices')
        for mi,mode in enumerate(records_at(group.get('modes'),path+'.modes')):
            mp=path+f'.modes[{mi}]'
            existing=mode.get('observed_role_ids',[])
            if not isinstance(existing,list) or any(not isinstance(i,str) for i in existing):
                raise ValueError(mp+'.observed_role_ids must be an array of existing ID strings')
            objects=records_at(mode.get('objects'),mp+'.objects')
            keys=[o.get('key') for o in objects]
            if any(not isinstance(k,str) for k in keys) or len(set(keys))!=len(keys):
                raise ValueError(mp+'.objects need distinct string keys')
            hosts={**observed,**{o['key']:o.get('category') for o in objects}}
            for oi,obj in enumerate(objects):
                op=mp+f'.objects[{oi}]';allowed={k:v for k,v in hosts.items() if k!=obj['key']}
                def fail(message):
                    raise ValueError(message)
                def from_target(target):
                    if not isinstance(target,str):fail('support_target must be one string')
                    if target in surfaces and target in allowed:
                        fail('Ambiguous surface/local-key token; rename the local key or use an explicit legacy support pair')
                    if target in surfaces:return dict(kind=target,target=None)
                    if target in allowed:return dict(kind='object',target=target)
                    fail('Unknown support_target; choose one exact surface token or instance reference')
                def from_legacy(support):
                    object_at(support,op+'.support')
                    kind=support.get('kind');target=support.get('target')
                    if not isinstance(kind,str) or (target is not None and not isinstance(target,str)):
                        fail('support needs string kind and null or string target')
                    if kind in surfaces:
                        if target is not None:fail('support.target must be null for '+kind+'; a location anchor is separate')
                        return dict(kind=kind,target=None)
                    if target not in allowed:fail('Object support requires one exact observed ID or another local key in this same mode')
                    if kind=='object':return dict(kind=kind,target=target)
                    category=allowed[target]
                    if isinstance(category,str) and kind.strip().casefold()==category.strip().casefold():
                        return dict(kind='object',target=target)
                    fail('Invalid support kind: '+repr(kind)+'; it is neither a primitive kind nor the exact referenced host category')
                try:
                    has_target='support_target' in obj;has_legacy='support' in obj
                    if not has_target and not has_legacy:fail('Missing support_target')
                    target_support=from_target(obj['support_target']) if has_target else None
                    legacy_support=from_legacy(obj['support']) if has_legacy else None
                    if has_target and has_legacy and target_support!=legacy_support:
                        fail('support_target and legacy support disagree; return one consistent support target')
                    canonical=target_support if has_target else legacy_support
                    if has_target or canonical!=obj.get('support'):
                        changes.append(dict(path=op,source={k:deepcopy(obj[k]) for k in ['support_target','support'] if k in obj},
                            canonical_support=canonical,rule='exact_support_target' if has_target else 'exact_referenced_host_category'))
                    obj['support']=canonical;obj.pop('support_target',None)
                except ValueError as exc:
                    errors.append(dict(path=op,error=str(exc),received={k:obj[k] for k in ['support_target','support'] if k in obj},
                        allowed_surface_tokens=sorted(surfaces),allowed_instance_targets=list(allowed)))
    if errors:raise ValueError(json.dumps(dict(field='domain.support',errors=errors),ensure_ascii=False))
    normalized=domain_answer(bound,case,max_groups=budgets['groups'],max_new=budgets['max_new'])
    if bindings is not None:bindings.extend(changes)
    return normalized


def geometry_records(domain,state,case):
    """Materialize the exact wire identities and per-instance anchor domains."""
    result=[];observed=[o['id'] for o in case['context']]
    for group in domain['groups']:
        mode=group['modes'][int(state[group['id']])]
        for obj in mode['objects']:
            qualify=lambda key:group['id']+'_'+key if key in {o['key'] for o in mode['objects']} else key
            anchors=([qualify(obj['support']['target'])] if obj['support']['kind']=='object' else
                ['camera']+observed+[qualify(o['key']) for o in mode['objects'] if o['key']!=obj['key']])
            result.append(dict(id=group['id']+'_'+obj['key'],group=group['id'],local_key=obj['key'],
                name=obj['name'],variant=obj['variant'],placement_hint=obj['placement_hint'],
                support={**obj['support'],'target':qualify(obj['support']['target'])},allowed_anchor_values=anchors))
    return result


def observed_geometry(case):
    """Relative metric proposals require the actual frozen anchor coordinates.

    The semantic inventory intentionally excludes noisy depth. Layout must not
    mistake that inventory for a numeric 3D context or invent host coordinates.
    """
    observed=validate_graph(read(case['state']))
    return [dict(id=o['track_id'],center_m=o['center_local_m']['value'],
        full_AABB_size_m=o['size_aabb_local_m']['value']) for o in observed['objects']]


def geometry_response(value,domain,state,case):
    records=geometry_records(domain,state,case)
    require_instance_ids(value,'objects',[r['id'] for r in records])
    expected={r['id']:r for r in records}
    bound=deepcopy(value);bindings={}
    for row in bound['objects']:
        allowed=expected[row['id']]['allowed_anchor_values']
        group=expected[row['id']]['group'];wire=row.get('anchor')
        local={r['local_key']:r['id'] for r in records if r['group']==group}
        # Exact qualified identities are canonical. Exact local keys from the
        # SAME group remain accepted for saved pre-v3 replies. Never strip an
        # arbitrary prefix, select a suffix, cross groups or bind by position.
        canonical=wire if wire in allowed else local.get(wire) if isinstance(wire,str) else None
        if canonical not in allowed:
            raise ValueError(json.dumps(dict(id=row['id'],field='anchor',received=wire,
                allowed_values=allowed,error='Choose ONE exact value; do not concatenate values or use type prefixes')))
        inverse={full:key for key,full in local.items()}
        row['anchor']=inverse.get(canonical,canonical)
        if row['anchor']!=wire:bindings[row['id']]=dict(wire_anchor=wire,canonical_anchor=canonical,compiler_anchor=row['anchor'])
    normalized,converted=geometry_answer(bound,domain,state,case)
    for oid,binding in bindings.items():converted[oid]['reference_binding']=binding
    # Reference cycles and support/anchor mismatches belong to response repair,
    # before starting any numerical layout. Keep the existing compiler rules.
    program_answer(normalized,case,dict(id='interface_check',description='Fixed selected instances'),
        max_groups=max(1,len(domain['groups'])),max_modes=1,max_new=len(records))
    return normalized,converted


def grounding_prompt(case):
    template=dict(instances=[dict(id=o['id'],variant=None,on_floor=None) for o in case['context']],evidence_description=None)
    return ('Read the original partial indoor photo and the ID-bound crops. Keep all original observed identities and geometry. '
        'Return ONE JSON object by filling the null fields in the complete template below. '
        'Return exactly '+str(len(case['context']))+' instance records: EVERY supplied observed ID exactly once, '
        'including sofas, cushions, supported objects and lights. Do not select a subset. '
        'Copy each id literally. Fill variant with a specific noun, on_floor with a JSON boolean, '
        'and evidence_description with one joint description covering the observed scene and relationships. '
        'on_floor means the whole object rests directly on the floor, rather than being supported by another object. '
        'Describe observed facts only, no hidden inventory, redesign, furniture moves or absence claims outside the image. '
        'The evidence is ONE jointly observed semantic signature, not independent object confidence scores.\n'
        'REQUIRED OUTPUT TEMPLATE: '+json.dumps(template,ensure_ascii=False)+'\n'
        'FROZEN OBSERVED INVENTORY: '+json.dumps(visible_inventory(case),ensure_ascii=False))


def domain_prompt(case,grounding,budgets):
    return ('Infer plausible EXISTING contents of the unphotographed surroundings of this same room. '
        'The input is a partial camera view; the target is the surrounding 360-degree world, including space beside and behind the camera. '
        'Keep all observed objects fixed. Do not redesign the visible arrangement or complete visible object shapes. '
        'The top-level JSON MUST be ONE OBJECT with ALL THREE fields: contexts, organizations, groups. '
        'Each field is an array. Return the entire object, not only scene descriptions. '
        'contexts (1..3) [{"description":"a possible room use / occupant preference that implies distinct unseen surroundings"}]; '
        'organizations (1..3) [{"description":"one possible overall arrangement of visible AND surrounding unseen areas"}]; '
        'These are alternative explanations of this same partial view, not a list of coexisting visible zones or decorative attributes. '
        'groups (0..'+str(budgets['groups'])+') [{"name":...,"mechanism_reason":"why this generative group depends on context/organization '
        'and declared parents, as a hypothesis", "parents":[earlier zero-based group indices], '
        '"modes":[{"description":"one optional complete joint object assembly", "observed_role_ids":[existing IDs satisfying roles], '
        '"objects":[{"key":"short_local_key","name":"ordinary specific object name","category":"ordinary noun",'
        '"variant":"specific type","identity":"new_unobserved","distinct_instance_reason":"why this is a separate unseen instance, '
        'not the photographed object relocated","placement_hint":"where it could already exist OUTSIDE the photographed view",'
        '"support_target":"floor"}]}]}]. '
        'Each group has 1..2 NONEMPTY alternatives; code inserts an optional absent state with equal descriptive treatment. '
        'support_target is ONE literal string: floor, wall, ceiling, unspecified, an exact observed instance ID, '
        'or an exact host key within the SAME mode. The program derives support kind; do not output a support.kind field. '
        'A category noun such as sofa does not identify a host instance. Support is physical support, not proximity or a location anchor. '
        'Using a photographed host is allowed only if placement_hint explains an unphotographed part that could hold the new object. '
        'No metric positions or sizes in this semantic stage. '
        'Largest combined assembly at most '+str(budgets['max_new'])+' instances: capacity, not a desired count. '
        'Explain why objects may already exist in the surrounding world; adding symmetry or improving the pictured decor is not evidence. '
        'Reuse visible objects via observed_role_ids; do not add unseen duplicates merely to finish a photographed arrangement. '
        'Optional furnishings, storage and personal decor are allowed; '
        'do not reduce a world to minimally necessary equipment. No object or direction quotas. '
        'Independent new instances of an observed category are allowed when explicitly distinct. '
        'Parents are generative decision dependencies, not mere co-occurrence or unsupported physical causality. '
        'The model is an assumption, not a discovered true causal graph. No GT full scene is available.\n'
        'OBSERVATION: '+json.dumps(grounding,ensure_ascii=False)+'\nFROZEN OBSERVED SEMANTICS: '+json.dumps(visible_inventory(case),ensure_ascii=False))


def mechanism_response(value,payload,bindings=None):
    """Compile explicitly keyed probabilities to the existing ordered CPT wire.

    Quoted JSON number literals can be decoded without changing their values;
    missing states, incorrect sums and prose percentages cannot be repaired by
    normalization, padding, dropping states or inventing an `other` state.
    """
    states=[s['id'] for s in payload['states']];parents=list(payload['declared_parent_domains'])
    expected=payload['required_given_rows'];records=value.get('rows') if isinstance(value,dict) else None
    def fail(path,message,received):
        raise ValueError(json.dumps(dict(variable=payload['variable'],path=path,error=message,received=received,
            required_state_ids=states,required_state_count=len(states),required_given_rows=expected),ensure_ascii=False))
    if not isinstance(records,list) or len(records)!=len(expected):
        fail('rows','Return exactly one record for each requested parent assignment',records)
    result=[];changes=[];seen=[]
    def vector(values,path):
        if not isinstance(values,list) or len(values)!=len(states):
            fail(path,'Need exactly one probability per declared state, in the stated order; do not add states',values)
        numbers=[]
        for i,v in enumerate(values):
            candidate=v
            if isinstance(v,str) and re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',v):
                candidate=float(v)
            try:valid=type(candidate) in (int,float) and math.isfinite(candidate) and 0<=candidate<=1
            except OverflowError:valid=False
            if not valid:fail(path+f'[{i}]','Expected a finite JSON number in [0,1], without percent signs or prose',v)
            if isinstance(v,str):changes.append(dict(path=path+f'[{i}]',source=v,number=candidate,rule='JSON_number_literal'))
            numbers.append(candidate)
        try:distribution(numbers,len(states))
        except ValueError:
            fail(path,'Probabilities must sum to 1 within 1e-5; '+('the only declared state must have probability 1' if len(states)==1
                else 'do not normalize arbitrary weights or invent missing states'),dict(values=numbers,sum=sum(numbers)))
        return numbers
    for i,row in enumerate(records):
        path=f'rows[{i}]'
        if not isinstance(row,dict):fail(path,'Expected a JSON object',row)
        given=row.get('given')
        if (not isinstance(given,dict) or set(given)!=set(parents) or given not in expected or given in seen):
            fail(path+'.given','Use each exact requested parent assignment once, with no extra parents',given)
        seen.append(given);has_map='p_by_state' in row;has_list='p' in row
        if not has_map and not has_list:fail(path,'Missing p_by_state',row)
        mapped=legacy=None
        if has_map:
            mapping=row['p_by_state']
            if not isinstance(mapping,dict) or set(mapping)!=set(states):
                fail(path+'.p_by_state','Keys must be exactly the declared state IDs, with no missing or extra states',mapping)
            mapped=vector([mapping[s] for s in states],path+'.p_by_state (order '+','.join(states)+')')
        if has_list:legacy=vector(row['p'],path+'.p')
        if has_map and has_list and mapped!=legacy:fail(path,'p_by_state and p disagree',row)
        ps=mapped if has_map else legacy
        if has_map:changes.append(dict(path=path+'.p_by_state',rule='exact_state_ID_order',state_order=states))
        result.append(dict(given=deepcopy(given),p=ps))
    if bindings is not None:bindings.extend(changes)
    return dict(rows=result)


def mechanism_prompt(payload):
    wire=deepcopy(payload);states=[s['id'] for s in payload['states']]
    wire['state_order']=states;wire['state_count']=len(states)
    wire['response_schema']=dict(rows=[dict(given=r,p_by_state={s:None for s in states}) for r in payload['required_given_rows']])
    return ('Specify a stochastic GENERATIVE mechanism as subjective conditional probabilities for a finite causal model. '
        'This is prior/likelihood elicitation, not choose-the-best-answer confidence and not calibrated real-world frequency. '
        'Return the complete JSON object in response_schema. Replace EVERY null in p_by_state with a JSON NUMBER, '
        'without quotation marks; copy all state-ID keys and given assignments exactly. Each row sums to 1. '
        'There are exactly '+str(len(states))+' states: '+json.dumps(states)+'. No additional or omitted states or parents. '
        'Use only the declared parent values and this fixed domain; no actual photograph is supplied. '
        'For E, the described outcome is the event whose likelihood is evaluated, not a fact to condition on twice. '
        'For hidden group states, optional absence and nonempty assemblies are competing possible existing states; '
        'unobserved is not absent, useful is not required. Do not make probabilities extreme merely from certainty about your answer.\n'+
        json.dumps(wire,ensure_ascii=False))


def selected_semantics(domain,state):
    return [dict(group=g['id'],name=g['name'],mode=g['modes'][int(state[g['id']])]) for g in domain['groups']]


def layout_prompt(domain,state,case,plane):
    records=geometry_records(domain,state,case)
    template=dict(objects=[dict(id=r['id'],width_m=None,height_m=None,depth_m=None,yaw_deg=None,
        anchor=r['allowed_anchor_values'][0] if len(r['allowed_anchor_values'])==1 else None,
        offset_m=[None,None,None]) for r in records])
    return ('Give metric attributes for the FIXED selected new unseen instances. '
        'Return ONE JSON object by filling the null fields in the complete template below. '
        'Return exactly '+str(len(records))+' object records. Copy each complete id literally, including its group prefix. '
        'For each anchor choose ONE literal string from that record\'s allowed_anchor_values. '
        'Do not add type prefixes or concatenate alternatives. Keep already filled template fields unchanged. '
        'All dimensions must be positive numbers; yaw_deg and all three offset_m entries must be finite numbers. '
        'No objects added, removed, renamed or relocated from the observations. '
        'Width and depth are intrinsic upright footprint dimensions, height is vertical thickness/height (a rug is thin in HEIGHT). '
        'Yaw rotates intrinsic footprint about CAMERA Y; the code converts to FULL camera AABB dimensions. '
        'This uses an explicit unmeasured camera-Y-level prior, not measured gravity. '
        'Offsets use original CAMERA x-right,y-image-up,z-forward metres relative to anchor centre. '
        'Negative z is behind camera. Use the complete prefixed ID when anchoring to another new instance in the same group. '
        'For object support, anchor MUST equal the fixed semantic support target. Support heights are computed externally. '
        'Infer reasonable positions in the surrounding unphotographed room; new full boxes must be OUTSIDE the original input view. '
        'Do not place items on visible surfaces and merely rename them unseen. No measured walls/room perimeter are available. '
        'A relative solver may refine placement while keeping sizes and observed state fixed.\n'
        'REQUIRED OUTPUT TEMPLATE: '+json.dumps(template,ensure_ascii=False)+
        '\nFIXED INSTANCE CONTRACTS: '+json.dumps(records,ensure_ascii=False)+
        '\nSELECTED GENERATIVE STATE: '+json.dumps(state)+'\nJOINT ASSEMBLIES: '+json.dumps(selected_semantics(domain,state),ensure_ascii=False)+
        '\nFROZEN OBSERVATIONS: '+json.dumps(visible_inventory(case),ensure_ascii=False)+
        '\nFROZEN OBSERVED CAMERA GEOMETRY (predicted metres, not GT; use for observed anchors): '+
        json.dumps(observed_geometry(case),ensure_ascii=False)+
        '\nCAMERA AND LEVEL PRIOR: '+json.dumps(dict(hfov=case['hfov'],image_size=case['image_size'],plane=plane)))


def rank_worlds(posterior,domain):
    merged={}
    for row in posterior['rows']:
        signature=tuple(row['state'][g['id']] for g in domain['groups'])
        entry=merged.setdefault(signature,dict(probability=0.,state=row['state'],representative_latent_mass=row['probability'],contributors=[]))
        if row['probability']>entry['representative_latent_mass']:
            entry.update(state=row['state'],representative_latent_mass=row['probability'])
        entry['probability']+=row['probability'];entry['contributors'].append(row)
    return sorted(merged.values(),key=lambda w:(-w['probability'],tuple(w['state'][g['id']] for g in domain['groups'])))


def model_graph(model,folder):
    nodes=model['nodes'];edges=[[p,n['id']] for n in nodes for p in n['parents']]
    write(folder/'causal_graph.json',dict(nodes=nodes,edges=edges,assumed=True,evidence={'E':'signature'},
        semantics='generative semantic decision mechanisms, not identified physical forces'))
    body=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="400"><rect width="100%" height="100%" fill="white"/>'
          '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0 L7 3 L0 6" fill="none" stroke="#777"/></marker></defs>']
    positions={'C':(80,90),'S':(300,90),'E':(300,280)}
    for i,n in enumerate(n for n in nodes if n['id'].startswith('g')):positions[n['id']]=(560+i%2*230,80+i//2*130)
    body+=['<text x="20" y="25">Assumed generative semantic SCM; E=positive local signature is conditioned on</text>']
    for a,b in edges:
        x,y=positions[a];u,v=positions[b];body.append(f'<path d="M{x+140} {y+22} L{u} {v+22}" stroke="#777" fill="none" marker-end="url(#arrow)"/>')
    labels={'C':'C: context','S':'S: zone organization','E':'E: visible signature'}
    for name,(x,y) in positions.items():body.append(f'<rect x="{x}" y="{y}" width="140" height="45" fill="#ddebf8" stroke="#678"/><text x="{x+5}" y="{y+28}">{escape(labels.get(name,name+": assembly"))}</text>')
    body.append('</svg>');(folder/'causal_graph.svg').write_text(''.join(body))


def execute(provider,case,output,budgets,worlds=3,seed=3407,max_tokens=8192,replay=None,checkpoint=None):
    observed=validate_graph(read(case['state']));check_hashes(case['source_sha256'])
    folder=output/case['id'];folder.mkdir();original,board=observation_board(case,observed,folder/'observation_board.png')
    write(folder/'G_obs.json',observed);write(folder/'input.json',case);original.save(folder/'local.png')
    calls=folder/'calls';calls.mkdir();stage='grounding';exports=[];failures=[];mechanism_sources={}
    record=dict(protocol=PROTOCOL,response_interface=INTERFACE_PROTOCOL,id=case['id'],world_state_ready=False,observed_instances=len(observed['objects']),
        generation_profile=getattr(provider,'generation_profile',None),
        execution_kind=getattr(provider,'execution_kind','local_Qwen' if provider is not None else 'cpu_replay'),
        scene_gt_used=False,adapter_connected=False,panorama_generated=False,training_performed=False,
        semantics_verified=False,causal_gain_validated=False,RGB_likelihood_defined=False)
    if replay is not None:record['replayed_execution_kind']=read(replay/'summary.json').get('execution_kind')
    if checkpoint is not None:
        record.update(checkpoint['provenance'],reused_layout_responses=[])
        for path in (checkpoint['folder']/'calls').iterdir():
            prefixes=('grounding.','grounding_repair.','domain.','domain_repair.')
            if checkpoint['model'] is not None:prefixes+=('mechanism_',)
            if path.is_file() and path.name.startswith(prefixes):
                shutil.copyfile(path,calls/path.name)
    try:
        def ask(name,image,prompt,validator):
            nonlocal stage
            stage=name;print(json.dumps(dict(stage=name)),flush=True)
            completed_layout=(checkpoint is not None and checkpoint['model'] is not None and name.startswith('layout_') and
                (checkpoint['folder']/('world_'+name.split('_')[1])/'execution.json').exists() and
                (checkpoint['folder']/('world_'+name.split('_')[1])/'G_full.json').exists())
            if completed_layout:
                for stem in (name+'_repair',name):
                    cached=checkpoint['folder']/'calls'/(stem+'.json')
                    if not cached.exists():continue
                    try:normalized=validator(read(cached))
                    except (ValueError,KeyError,TypeError):continue
                    for path in cached.parent.glob(stem+'.*'):shutil.copyfile(path,calls/path.name)
                    record['reused_layout_responses'].append(name)
                    print(json.dumps(dict(stage=name,event='reused_completed_layout_response')),flush=True)
                    return normalized
            if replay is not None:
                raw=replay/'calls'/(name+'_repair.json')
                if not raw.exists():raw=replay/'calls'/(name+'.json')
                value=read(raw);write(calls/(name+'.json'),value)
                return validator(value)
            return ask_valid(provider,image,prompt,calls/(name+'.txt'),validator,max_tokens)
        def infer_mechanism(name,payload,validator):
            nonlocal stage
            ids=[s['id'] for s in payload['states']];query='mechanism_'+name
            source=dict(state_ids=ids,parent_assignments=len(payload['required_given_rows']))
            mechanism_sources[name]=source
            if len(ids)==1:
                # On a finite one-state domain there is no unknown parameter.
                # This says nothing about whether the proposed domain is complete.
                stage=query
                answer=dict(rows=[dict(given=r,p=[1.]) for r in payload['required_given_rows']])
                result=validator(answer)
                source.update(kind='mathematical_single_state',model_request_needed=False,
                    scope='conditional_on_proposed_finite_domain_not_real_world_certainty')
                write(calls/(query+'.deterministic.json'),dict(**source,answer=answer))
                print(json.dumps(dict(stage=query,event='single_state_exact',state_ids=ids,model_called=False)),flush=True)
                return result
            source.update(kind='frozen_model_response',model_request_needed=True,response_validated=False)
            def bind(value):
                bindings=[]
                result=validator(mechanism_response(value,payload,bindings=bindings))
                write(calls/(query+'.binding.json'),dict(state_order=ids,bindings=bindings,
                    scope='Reference/type conversion only; arbitrary weights are not normalized'))
                source['response_validated']=True
                return result
            return ask(query,None,mechanism_prompt(payload),bind)
        model=None
        if checkpoint is not None:
            grounding=checkpoint['grounding'];domain=checkpoint['domain'];model=checkpoint['model']
            bindings=checkpoint.get('domain_bindings',[])
            print(json.dumps(dict(stage='semantic_checkpoint_loaded' if model is not None else 'domain_checkpoint_loaded',
                semantic_model_reused=model is not None,domain_reused=True)),flush=True)
        else:
            bindings=[]
            grounding=ask('grounding',board,grounding_prompt(case),lambda v:grounding_response(v,case))
            write(folder/'grounding.json',grounding)
            domain=ask('domain',board,domain_prompt(case,grounding,budgets),lambda v:domain_response(v,case,budgets,bindings=bindings))
        write(folder/'grounding.json',grounding)
        write(folder/'domain.json',domain)
        write(folder/'domain_binding.json',dict(interface=INTERFACE_PROTOCOL,bindings=bindings,
            scope='Deterministic reference conversion only; no semantic or geometric repair'))
        if model is None:
            evidence=dict(description=grounding['evidence_description'],camera_hfov=case['hfov'],
                          image_size=case['image_size'],outside_absence_asserted=False)
            model=build_model(domain,evidence,infer_mechanism)
        write(folder/'causal_model.json',model);model_graph(model,folder)
        write(folder/'mechanism_strengths.json',dict(strengths=mechanism_strengths(model),
            scope='CPT_sensitivity_in_assumed_model; not_validated_real_world_causal_effect'))
        stage='exact_semantic_inference';prior=enumerate_model(model);posterior=enumerate_model(model,{'E':'signature'})
        write(folder/'prior.json',prior);write(folder/'posterior.json',posterior)
        record.update(latent_domain_sizes={name:len(next(n for n in model['nodes'] if n['id']==name)['states']) for name in ['C','S']},
            max_hidden_marginal_update_TV=max((sum(abs(prior['marginals'][g['id']][s]-p) for s,p in
                posterior['marginals'][g['id']].items())/2 for g in domain['groups']),default=0.))
        ablation=independent_groups(model);write(folder/'independent_ablation_model.json',ablation)
        write(folder/'independent_ablation_posterior.json',enumerate_model(ablation,{'E':'signature'}))
        noise=abduct(model,{'E':'signature'},seed)
        interventions=[]
        for node in model['nodes']:
            if node['id']=='E':continue
            for value in node['states']:
                do={node['id']:value}
                interventions.append(dict(do=do,interventional_marginals=enumerate_model(model,do=do)['marginals'],
                    same_abducted_noise_prediction=forward(model,noise['noise'],do)))
        write(folder/'mechanism_queries.json',dict(abduction=noise,interventions=interventions,
            scope='model_relative_mechanism_diagnostics; not_validation_of_real_world_causal_effects'))
        ranked=rank_worlds(posterior,domain)
        write(folder/'ranked_semantic_worlds.json',ranked)
        plane=support_plane(observed,[o['id'] for o in grounding['instances'] if o['on_floor']]);write(folder/'support_plane.json',plane)
        expected=sum(w['probability']*sum(len(g['modes'][int(w['state'][g['id']])]['objects']) for g in domain['groups']) for w in ranked)
        record.update(semantic_inference_complete=True,semantic_worlds=len(ranked),
            expected_new_instances_in_proposed_domain=expected,no_geometric_renormalization=True,
            latent_prior={n:prior['marginals'][n] for n in ['C','S']},latent_posterior={n:posterior['marginals'][n] for n in ['C','S']},
            observation_probability_in_assumed_model=posterior['evidence_probability'])
        for index,world in enumerate(ranked[:worlds],1):
            state=world['state'];dest=folder/f'world_{index:02d}';dest.mkdir()
            try:
                nonempty=any(state[g['id']]!='0' for g in domain['groups'])
                if nonempty:
                    raw,converted=ask(f'layout_{index:02d}',board,layout_prompt(domain,state,case,plane),
                        lambda v:geometry_response(v,domain,state,case))
                    config=dict(id='semantic_world',description='; '.join(next(c['description'] for c in domain[k] if c['id']==state[n])
                        for k,n in [('contexts','C'),('organizations','S')]))
                    program=program_answer(raw,case,config,max_groups=max(1,budgets['groups']),max_modes=1,max_new=budgets['max_new'])
                else:
                    program=dict(configuration=dict(id='empty',description='No additional object in the proposed finite domain'),groups=[]);converted={}
                solved=solve_assignment(case,observed,plane,program,[1]*len(program['groups']),dict(region_keep=3,beam_width=4,rounds=8))
                if not solved['review']['geometry_contract_passed']:raise ValueError('Selected complete assembly has no valid off-frustum numeric export')
                full,_,unmapped=export_states(observed,solved['plan'],solved['items'],solved['mappings'],solved['review'])
                full.update(completion_protocol=PROTOCOL,adapter_connected=False,world_state_ready=True,
                    world_state_ready_scope='numeric_export_not_semantic_certification',semantic_model_weight=world['probability'],
                    causal_gain_validated=False,target='outside_input_frustum',declared_support_plane=plane)
                semantic_metadata={f'sp_g{gi:02d}_'+obj['key']:dict(variant=obj['variant'],identity=obj['identity'],
                    distinct_instance_reason=obj['distinct_instance_reason'],generative_group=g['id'])
                    for gi,(g,m) in enumerate((g,g['modes'][int(state[g['id']])]) for g in domain['groups'] if state[g['id']]!='0')
                    for obj in m['objects']}
                for obj in full['objects'][len(observed['objects']):]:obj.update(semantic_metadata[obj['track_id']])
                validate_world(observed,full,solved['items'],len(observed['objects'])+budgets['max_new'])
                write(dest/'G_full.json',full);hidden=deepcopy(full);hidden.update(scope='hidden',objects=full['objects'][len(observed['objects']):])
                write(dest/'G_out.json',hidden);write(dest/'numeric_review.json',solved['review'])
                write(dest/'execution.json',dict(semantic_world=world,geometry_conversions=converted,solved=solved))
                h=dict(folder=dest.name,new_instances=len(solved['items']),semantic_model_weight=world['probability'],unmapped_new=len(unmapped),
                    observed_state_preserved=True,remaining_AABB_overlap_pairs=len(solved['review']['program_solver']['final_overlaps']),
                    status='candidate_needs_semantic_review' if solved['review']['layout_proxy_passed'] else 'layout_needs_review')
                visual=dest/'world';visual.mkdir();render(full,original,case['hfov'],visual,h,
                    position_note='生成语义机制与证据后验；原观测固定。几何为相机Y水平假设下的近似布局。')
                page=visual/'index.html';page.write_text(page.read_text().replace(
                    '红色候选保留在G_full_candidates中，不进入Adapter子集G_full。',
                    '红色未映射对象完整保留在G_full/G_out，本阶段不连接Adapter。'))
                from PIL import Image
                image_path=visual/'layout_xz.png'
                with Image.open(image_path) as handle:panel=handle.convert('RGB')
                painter=ImageDraw.Draw(panel);painter.rectangle((0,0,panel.width,32),fill='white')
                painter.text((10,10),'Camera XZ boxes; Y-level floor PRIOR; gravity and walls unmeasured.',fill='black');panel.save(image_path)
                exports.append(h)
            except (ValueError,FileNotFoundError,APIRequestError) as exc:
                failures.append(dict(world=index,semantic_model_weight=world['probability'],error=str(exc),
                    failure_kind='transport' if isinstance(exc,APIRequestError) else 'layout_or_response',
                    stage=stage,transport_details=exc.details if isinstance(exc,APIRequestError) else None))
                (dest/'error_traceback.txt').write_text(traceback.format_exc())
        if exports:
            for name in ['G_full.json','G_out.json']:shutil.copyfile(folder/exports[0]['folder']/name,folder/name)
        expected=sum(w['probability']*sum(len(g['modes'][int(w['state'][g['id']])]['objects']) for g in domain['groups']) for w in ranked)
        record.update(status='worlds_exported_needs_review' if exports and not failures else 'partial_layouts_needs_review' if exports else 'no_world_exported',
            world_state_ready=bool(exports),worlds=exports,layout_failures=failures,semantic_inference_complete=True,
            expected_new_instances_in_proposed_domain=expected,semantic_worlds=len(ranked),
            exported_semantic_mass=sum(h['semantic_model_weight'] for h in exports),
            failed_layout_semantic_mass=sum(h['semantic_model_weight'] for h in failures),
            unattempted_semantic_mass=sum(w['probability'] for w in ranked[worlds:]),
            no_geometric_renormalization=True,representative_world=exports[0]['folder'] if exports else None,
            latent_prior={n:prior['marginals'][n] for n in ['C','S']},latent_posterior={n:posterior['marginals'][n] for n in ['C','S']},
            observation_probability_in_assumed_model=posterior['evidence_probability'])
        check_hashes(case['source_sha256'])
    except Exception as exc:
        record.update(status='failed',failed_stage=stage,error=f'{type(exc).__name__}: {exc}',world_state_ready=False)
        (folder/'error_traceback.txt').write_text(traceback.format_exc())
    if provider is not None:record.update(api_calls=getattr(provider,'calls',None),api_usage=getattr(provider,'usage',None))
    write(folder/'mechanism_execution.json',dict(queries=mechanism_sources,model_reused=bool(checkpoint and checkpoint['model'] is not None)))
    write(folder/'summary.json',record);write(output/'summary.json',record)
    body=['<!doctype html><meta charset="utf-8"><title>CauPano causal world</title><h1>局部观测 → 周围世界假设</h1>',
        '<p>生成语义机制 + 联合正观测事件的精确有限域后验。概率为假设模型内的主观权重；未验证现实因果或语义质量。2:1图是状态方向图。</p>',
        '<img src="local.png" width="500"><img src="causal_graph.svg" width="1000">',
        '<p><a href="posterior.json">证据后验</a> | <a href="causal_model.json">完整条件机制</a> | <a href="mechanism_queries.json">机制干预与模型内溯因</a></p>']
    for h in exports:
        body.append('<p>'+escape(h['folder'])+f'：新增 {h["new_instances"]}，有限模型权重 {h["semantic_model_weight"]:.5f} '
            f'<a href="{h["folder"]}/world/index.html">3D与2:1方向图</a> | <a href="{h["folder"]}/G_full.json">G_full</a></p>')
    body.append('<pre>'+escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre>');(folder/'index.html').write_text(''.join(body))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><a href="'+case['id']+'/index.html">打开因果语义世界与可视化</a>')
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps(record,ensure_ascii=False),flush=True);return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--case',type=Path)
    parser.add_argument('--backend',choices=['siliconflow','local','fixture'],default='siliconflow')
    parser.add_argument('--generation-profile',choices=['legacy','matched_v1'],default='legacy')
    parser.add_argument('--api-model',default=os.environ.get('SILICONFLOW_MODEL',DEFAULT_API_MODEL))
    parser.add_argument('--api-base',default=os.environ.get('SILICONFLOW_BASE_URL',DEFAULT_API_BASE))
    parser.add_argument('--key-env',default='SILICONFLOW_API_KEY');parser.add_argument('--thinking',choices=['auto','on','off'],default='auto')
    parser.add_argument('--local-model',type=Path);parser.add_argument('--max-api-calls',type=int,default=32)
    parser.add_argument('--timeout',type=int,default=600);parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--stream',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--groups',type=int,default=4);parser.add_argument('--max-new',type=int,default=24)
    parser.add_argument('--worlds',type=int,default=3);parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--check-api',action='store_true')
    parser.add_argument('--replay',type=Path,help='Existing output to replay completely on CPU, with no API or local model')
    parser.add_argument('--continue-from',type=Path,help='Reuse a completed semantic model; provider only for missing layouts, NEW output required')
    parser.add_argument('--continue-domain',type=Path,help='Reuse validated grounding/domain only; recompute every CPT and layout')
    args=parser.parse_args()
    if args.check_api:
        if args.preflight or args.replay or args.continue_from or args.continue_domain:parser.error('--check-api is an explicit GET /models action, separate from offline actions')
        print(json.dumps(SiliconFlowJSON(args.api_model,args.api_base,args.key_env,max_calls=1,timeout=args.timeout).check_model()));return
    if args.output is None:parser.error('--output NEW_DIRECTORY is required')
    try:args.output=validate_new_output(args.output)
    except (ValueError,OSError) as exc:parser.error(str(exc))
    if not 1<=args.groups<=4 or not 1<=args.max_new<=48 or not 1<=args.worlds<=8 or not 1<=args.max_api_calls<=100:
        parser.error('Invalid computational budgets')
    if not 512<=args.max_tokens<=16384 or not 1<=args.timeout<=900 or not 0<=args.seed<2**31:parser.error('Invalid token/timeout/seed budget')
    replay_folder=None;checkpoint=None
    continuation=args.continue_from or args.continue_domain
    if continuation and (args.replay or args.case or (args.continue_from and args.continue_domain)):
        parser.error('Use only one continuation source, without --replay or --case')
    if continuation:
        checkpoint=load_checkpoint(continuation,allow_transport_updates=True,domain_only=bool(args.continue_domain))
        previous=checkpoint['config'];case=checkpoint['case']
        args.groups=previous['budgets']['groups'];args.max_new=previous['budgets']['max_new']
    if args.replay:
        previous=read(args.replay/'config.json');check_hashes(previous['code_sha256'])
        manifest=read(args.replay/'artifact_manifest.json')
        if {str(p.relative_to(args.replay)):sha(p) for p in args.replay.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}!=manifest:
            parser.error('Replay artifact manifest changed')
        case=previous['case'];args.groups=previous['budgets']['groups'];args.max_new=previous['budgets']['max_new']
        replay_folder=args.replay/case['id'];check_hashes(case['source_sha256'])
    elif not continuation:case=load_case(args.case)
    if args.backend=='local':
        from .qwen_upstream import DEFAULT_MODEL
        if not args.replay and not (args.local_model or DEFAULT_MODEL).is_dir():parser.error('Local cached model directory is missing')
    if args.preflight:
        from .world_generation import recipe
        print(json.dumps(dict(status='causal_world_preflight_passed',id=case['id'],observed=len(case['context']),output=str(args.output),
            response_interface=INTERFACE_PROTOCOL,generation_profile=args.generation_profile,
            common_recipe=recipe() if args.generation_profile=='matched_v1' else None,
            backend=args.backend,api_model=args.api_model if args.backend=='siliconflow' else None,
            local_model=str(args.local_model or DEFAULT_MODEL) if args.backend=='local' else None,
            key_present=bool(os.environ.get(args.key_env)) if args.backend=='siliconflow' else False,
            semantic_checkpoint_reused=bool(checkpoint and checkpoint['model'] is not None),
            domain_checkpoint_reused=bool(checkpoint and args.continue_domain),stream=args.stream,timeout_seconds=args.timeout,
            network_started=False,model_loaded=False,gpu_started=False)));return
    if args.replay:provider=None
    elif args.backend=='siliconflow':provider=SiliconFlowJSON(args.api_model,args.api_base,args.key_env,args.timeout,args.max_api_calls,args.thinking,
        stream=args.stream,generation_profile=args.generation_profile)
    elif args.backend=='local':
        from .qwen_upstream import DEFAULT_MODEL,Qwen
        if args.generation_profile=='matched_v1':
            from .world_generation import MatchedLocalJSON
            provider=MatchedLocalJSON(args.local_model or DEFAULT_MODEL)
        else:provider=Qwen(args.local_model or DEFAULT_MODEL)
    else:
        from .world_scm_fixture import FixtureProvider
        provider=FixtureProvider(case)
    output=new_output(args.output);budgets=dict(groups=args.groups,max_new=args.max_new)
    write(output/'config.json',dict(protocol=PROTOCOL,response_interface=INTERFACE_PROTOCOL,case=case,budgets=budgets,code_sha256=source_hashes(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        execution_kind='cpu_fixture_not_model_prediction' if args.backend=='fixture' and not args.replay else 'cpu_replay' if args.replay else args.backend))
    record=execute(provider,case,output,budgets,args.worlds,args.seed,args.max_tokens,replay_folder,checkpoint)
    if not record['world_state_ready']:raise SystemExit('World completion failed; inspect preserved raw responses and summary.json')


if __name__=='__main__':main()
