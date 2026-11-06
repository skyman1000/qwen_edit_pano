"""Joint content mechanism; distinguish object inventory from installation.

This is an ASSUMED conditional generative SCM, not an identified causal model
of the real room. Reference probabilities are a sampling policy, not an image
posterior. Removing a host invalidates installation without erasing ownership.
"""
from copy import deepcopy
import itertools
import json

import numpy as np

from . import assembly_model as base
from .scene_program import text
from .world_scm import validate,key,forward

PROTOCOL='conditional_inventory_installation_scm_v2'
INTERFACE='joint_contents_single_attachment_v3'
hypotheses_prompt=base.hypotheses_prompt
hypotheses_answer=base.hypotheses_answer
furniture_prompt=base.furniture_prompt
furniture_answer=base.furniture_answer


def contents_prompt(case,furniture):
    groups={g['key']:[o['key'] for o in g['objects']] for g in furniture['groups']}
    schema={**base.OBJECT_SCHEMA,'group':next(iter(groups),'g_free')}
    return ('Propose 1..2 alternative COMPLETE JOINT inventories of NEW everyday contents for this existing surrounding world. '
        'Infer plausible hidden belongings from the specific configuration and observed evidence, not a minimum functional room design. '
        'Furniture and all photographed observations ALREADY EXIST; objects contains ONLY the additional content delta. '
        'Do not copy furniture records. All added instances are distinct, outside the ORIGINAL input view. '
        'A lamp shade, handle, chair leg or other integral part is described in its parent variant, not a new independent object. '
        'Do not invent small items to fill every group: one alternative covers the WHOLE world, may omit groups, and may be empty. '
        'Return {"alternatives":[{"description":"one joint hidden contents hypothesis","objects":['+json.dumps(schema)+']} ]}. '
        'Capacity is at most 12 new content instances per alternative, with no minimum. '
        'category is a concrete noun (book, mug, basket), not everyday belonging or decoration. '
        'attachment is the SINGLE source of physical installation; there is NO separate support-relation list to fill. '
        'attachment.kind: floor/top/seat/shelf/hang/wall/ceiling/free. '
        'top/seat/shelf/hang require an exact furniture key, an observed ID, or another new object key IN THE SAME alternative. '
        'The other kinds require host=null. All furniture in this configuration is available; the program derives a hosted item group from its host. '
        'For an unhosted item optionally set group to an existing group ID or g_free; omitted means independent surroundings g_free. '
        'affordances is any subset of top,seat,shelf,hanging and lists surfaces this object offers to OTHER objects, usually []; '
        'a geometrical top is not a usable tabletop. '
        'Books, mugs and plants do not ordinarily rest on floor lamp bases or picture frames. Respect concrete variants. '
        'No added large furniture, numeric geometry, alternative IDs or probability tables. '
        '\nFURNITURE GROUP IDS: '+json.dumps(groups)+
        '\nDECLARED PARENTS (whole configuration, furniture variants, host affordances, fixed observed identities): '+
        json.dumps(dict(furniture=furniture,observed=case['context']),ensure_ascii=False))


def contents_answer(value,case,furniture):
    rows=value.get('alternatives');observed={o['id'] for o in case['context']}
    hosts={o['key']:g['key'] for g in furniture['groups'] for o in g['objects']}
    if not isinstance(rows,list) or not 1<=len(rows)<=2:
        raise ValueError('Return alternatives: 1..2 COMPLETE JOINT content deltas for the whole configuration, not per-group bundles')
    result=[];errors=[]
    for i,row in enumerate(rows):
        path=f'alternatives[{i}]'
        try:
            if not isinstance(row,dict):raise ValueError(path+' must be an object')
            objects=base.objects_answer(row.get('objects'),observed,hosts,max_objects=12,path=path+'.objects',host_scope='THIS CONFIGURATION/ALTERNATIVE')
            original={o['key']:o for o in row['objects']};assigned=dict(hosts)
            assigned.update({oid:'g_obs' for oid in observed})
            for obj in base.order_objects(objects,set(hosts)|observed):
                host=obj['attachment']['host']
                if host:gid=assigned[host]
                else:
                    gid=original[obj['key']].get('group','g_free')
                    if not isinstance(gid,str) or gid not in set(hosts.values())|{'g_free'}:
                        raise ValueError(path+'.objects['+obj['key']+'].group for an unhosted item must be '+str(sorted(set(hosts.values())|{'g_free'})))
                obj['group']=gid;assigned[obj['key']]=gid
            # Robust reading of unsolicited extra relations. A repeated support
            # assertion is audited against attachment; it never rewrites it.
            rel,bindings=base.relations_answer(row.get('relations',[]),set(hosts)|observed|{o['key'] for o in objects},objects,path+'.relations')
            result.append(dict(description=text(row.get('description'),path+'.description'),objects=objects,
                relations=rel,relation_bindings=bindings))
        except ValueError as exc:errors.append(str(exc))
    if errors:raise ValueError(json.dumps(dict(joint_content_contract_errors=errors),ensure_ascii=False))
    return result


def build_model(hypotheses,furnitures,contents):
    nodes=[];registry={}
    def add(name,states,parents,mechanism):
        domains={n['id']:n['states'] for n in nodes};table={}
        for values in itertools.product(*(domains[p] for p in parents)):
            table[key(values)]=mechanism(dict(zip(parents,values)),states)
        nodes.append(dict(id=name,states=states,parents=parents,cpt=table))
    def one(states,wanted):return [float(s==wanted) for s in states]
    add('Z',[z['id'] for z in hypotheses],[],lambda g,s:[1/len(s)]*len(s))
    def furniture_law(g,states):
        active=[fid for fid in states if furnitures[fid]['hypothesis']['id']==g['Z']]
        return [float(fid in active)/len(active) for fid in states]
    add('F',list(furnitures),['Z'],furniture_law)
    for fid,f in furnitures.items():
        tid='T_'+fid;alts=contents[fid]
        add(tid,['off']+[str(i) for i in range(len(alts))],['F'],
            lambda g,s,fid=fid:one(s,'off') if g['F']!=fid else [0.]+[1/(len(s)-1)]*(len(s)-1))
        furniture=[dict(deepcopy(o),group=g['key']) for g in f['groups'] for o in g['objects']]
        furniture_ids={o['key']:f'{fid}_{o["group"]}_base_{o["key"]}' for o in furniture}
        for ai,objects in [(-1,furniture)]+[(i,a['objects']) for i,a in enumerate(alts)]:
            local={o['key']:(furniture_ids[o['key']] if ai<0 else f'{fid}_{o["group"]}_c{ai}_{o["key"]}') for o in objects}
            external={o['attachment']['host'] for o in objects if o['attachment']['host'] not in local}
            for obj in base.order_objects(objects,external):
                oid=local[obj['key']];host=obj['attachment']['host'];host_id=local.get(host) or furniture_ids.get(host)
                eid='E_'+oid;sid='S_'+oid;host_node='S_'+host_id if host_id else None
                def exists(g,s,fid=fid,ai=ai,tid=tid):
                    return one(s,'1' if g['F']==fid and (ai<0 or g[tid]==str(ai)) else '0')
                add(eid,['0','1'],['F']+([tid] if ai>=0 else []),exists)
                def installed(g,s,eid=eid,hn=host_node):return one(s,'1' if g[eid]=='1' and (hn is None or g[hn]=='1') else '0')
                add(sid,['0','1'],[eid]+([host_node] if host_node else []),installed)
                rec=deepcopy(obj);rec.update(id=oid,presence_node=eid,installation_node=sid,
                    furniture_state=fid,content_alternative=ai)
                rec['attachment']['host']=host_id or host;registry[oid]=rec
    return validate(dict(protocol=PROTOCOL,nodes=nodes,component_registry=registry,hypotheses=deepcopy(hypotheses),
        furniture_states=deepcopy(furnitures),contents_mechanisms=deepcopy(contents),
        distribution_scope='uniform_reference_law_conditional_on_image_proposed_finite_grammar',
        installation_node_scope='declared_host_dependency_satisfied_not_physical_verification',
        RGB_likelihood_defined=False,calibrated_posterior_defined=False,real_causal_structure_identified=False,
        intervention_scope='inventory_removal_invalidates_installation_but_does_not_erase_other_owned_objects'))


def active_world(model,state):
    objects=[deepcopy(o) for o in model['component_registry'].values() if state[o['presence_node']]=='1']
    f=model['furniture_states'][state['F']];choice=state.get('T_'+state['F'],'off')
    mapping={o['key']:o['id'] for o in objects}
    if len(mapping)!=len(objects):raise ValueError('Duplicate active identity')
    relations=deepcopy(f['relations'])
    if choice!='off':relations+=deepcopy(model['contents_mechanisms'][state['F']][int(choice)]['relations'])
    available=set(mapping)|set(mapping.values())
    inventory_keys={o['key'] for o in model['component_registry'].values()}
    relations=[{**r,'source':mapping.get(r['source'],r['source']),'target':mapping.get(r['target'],r['target'])} for r in relations
        if all(r[k] not in inventory_keys or r[k] in available for k in ('source','target'))]
    groups=deepcopy(f['groups']);gids={g['key'] for g in groups}
    for obj in objects:
        obj['installation_ready']=state[obj['installation_node']]=='1'
        if obj['group'] not in gids:
            groups.append(dict(key=obj['group'],function='Independent surrounding contents or fixed observed host.',objects=[]));gids.add(obj['group'])
    return dict(configuration=f['description'],hypothesis=f['hypothesis'],objects=objects,groups=groups,relations=relations,
        structural_state=deepcopy(state),unresolved_installations=[o['id'] for o in objects if not o['installation_ready']])


def sample_worlds(model,seed=3407,count=3):
    rng=np.random.default_rng(seed);worlds=[];seen=set()
    for trial in range(128):
        noise={n['id']:float(rng.random()) for n in model['nodes']};world=active_world(model,forward(model,noise))
        signature=tuple(sorted(o['id'] for o in world['objects']))
        if signature in seen:continue
        seen.add(signature);world.update(noise=noise,seed=seed,trial=trial);worlds.append(world)
        if len(worlds)>=count:break
    return worlds


def intervention_diagnostics(model,world):
    rows=[];before={o['id'] for o in world['objects']}
    for obj in world['objects']:
        if not any(o['attachment']['host']==obj['id'] for o in world['objects']):continue
        do={obj['presence_node']:'0'};after=active_world(model,forward(model,world['noise'],do));ids={o['id'] for o in after['objects']}
        rows.append(dict(do=do,host=obj['id'],before=sorted(before),after=sorted(ids),removed=sorted(before-ids),
            unresolved_installations=after['unresolved_installations'],same_exogenous_noise=True,model_called=False,
            geometry_not_resolved=True))
    return dict(queries=rows,scope=model['intervention_scope'],causal_gain_validated=False)
