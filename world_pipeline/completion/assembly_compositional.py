"""v12: optional complete functional programs, with fresh spatial choices.

Restores the productive v5 grammar grain, not its variant-count-dependent
existence prior. A program may contain several same-function groups/instances.
No global multi-select response, count target, training, or post-sample pruning.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np

from ..common import read,write,sha
from . import assembly_modular as modular,assembly_panorama as panorama
from . import assembly_population as population
from .world_scm import validate,forward,key

PROTOCOL=modular.PROTOCOL
INTERFACE='optional_complete_functional_spatial_programs_v12'
MODEL_VARIANT='compositional_functional_program_scm_v12'
KIND='compositional_functional_program'
profile=modular.profile
load_domain=modular.load_domain
semantic_diagnostics=modular.semantic_diagnostics
hypotheses_prompt=modular.hypotheses_prompt
hypotheses_answer=modular.hypotheses_answer


def furniture_prompt(case,hypothesis,channel='surrounding'):
    text=modular.furniture_prompt(case,hypothesis,channel)
    text=text.replace('Each group is ONE optional, self-contained functional assembly,',
        'Each group is one internally complete part of a functional program,')
    text=text.replace('Multiple groups of the same function will be interpreted as alternatives, not automatically combined.',
        'Within ONE alternative, ALL groups of the same function coexist and form its COMPLETE functional program. Different alternatives describe different possible programs.')
    return text+('\nV12 COMPLETE PROGRAM CONTRACT: Jointly propose substantial, evidence-compatible existing assemblies '
        'with their ordinary contents at appropriate scales, rather than a catalog of isolated anchor nouns. '
        'A program can contain several separate same-function groups at different anchors. '
        'Keep all new host dependencies internal to their group; observed references remain fixed. '
        'H is non-exhaustive, not an additional-zone prerequisite. No category, activity, object count or camera sector is required. '
        'Do not copy already photographed instances. No metric dimensions or probabilities; camera plans are sampled later.')


def furniture_answer(value,case,hypothesis):
    return modular.furniture_answer(value,case,hypothesis)


def revalidate_program(program,case):
    """Revalidate a saved semantic program, never reuse a CPT or metric layout."""
    raw=dict(alternatives=[dict(description=program['description'],groups=[],relations=deepcopy(program['relations']))])
    for g in program['groups']:
        raw['alternatives'][0]['groups'].append(dict(function=g['function'],reason=g.get('reason') or program['description'],
            observed_refs=g.get('observed_refs',[]),new_objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],
                reason=o['distinct_instance_reason'],placement=o['placement_hint'],attachment=deepcopy(o['attachment']),
                affordances=o['affordances']) for o in g['objects']]))
    # Canonical greater/smaller-Z aliases must pass the same typed contract.
    for r in raw['alternatives'][0]['relations']:r['kind']={'in_front_of':'farther_than','behind':'nearer_than'}.get(r['kind'],r['kind'])
    h=profile(program['hypothesis']['id'],program['hypothesis']['functions'])
    clean=modular.furniture_answer(raw,case,h)[0]
    original={o['key']:o for g in program['groups'] for o in g['objects']}
    names={o['id']:o['description'] for o in case['context']}
    for i,g in enumerate(clean['groups']):
        # Parser can add an observed-host group: semantic_group identifies its
        # original group rather than relying on a row's numeric position.
        semantic=next((o.get('semantic_group') for o in g['objects']),None)
        origin=int(semantic[1:]) if semantic and semantic[1:].isdigit() else min(i,len(program['groups'])-1)
        source=program['groups'][origin]
        g.update(layout_region=source.get('layout_region','unspecified'),spatial_reason=source.get('spatial_reason',source.get('reason',program['description'])),
            functional_links=deepcopy(source.get('functional_links',[])))
        for o in g['objects']:
            o['support_model']=original[o['key']].get('support_model','rigid_box')
            if o['attachment']['host'] in names:o['observed_host_name']=names[o['attachment']['host']]
    clean['catalog_provenance']=deepcopy(program.get('catalog_provenance',{}))
    assert {o['key'] for g in clean['groups'] for o in g['objects']}==set(original)
    return clean


def spatial_choices(programs,case):
    choices={};rejections=[]
    observed={o['id'] for o in case['context']}
    for fid,p in programs.items():
        if fid=='f_null':continue
        domains=[]
        for g in p['groups']:
            r=g.get('layout_region','unspecified')
            anchored=any(o['attachment']['host'] in observed for o in g['objects'])
            domains.append([r] if r!='unspecified' else ['unspecified'] if anchored else ['left','right','rear'])
        for regions in itertools.product(*domains):
            trial=deepcopy(p);plan={g['key']:r for g,r in zip(trial['groups'],regions)}
            for g in trial['groups']:g['layout_region']=plan[g['key']]
            try:panorama.check_region_order(trial,case)
            except ValueError as exc:rejections.append(dict(module=fid,regions=plan,error=str(exc)));continue
            cid=fid+'_p'+str(sum(v['module']==fid for v in choices.values()))
            choices[cid]=dict(module=fid,regions=plan)
    represented={v['module'] for v in choices.values()}
    if represented!=set(programs)-{'f_null'}:raise ValueError('A complete program has no legal camera plan; inspect rather than delete it')
    return choices,rejections


def positive_context(case):
    observed=read(case['state'])
    return dict(frame=observed['frame'],hfov=case['hfov'],image_size=case['image_size'],
        objects=[dict(id=o['track_id'],name=o['description'],predicted_center_m=o['center_local_m']['value'],
            predicted_full_AABB_m=o['size_aabb_local_m']['value']) for o in observed['objects']],
        uncertainty='Predicted boxes only; unseen space, walls, support surfaces and resident history are unknown')


def program_menu(domain,role):
    return [dict(id=fid,description=domain['programs'][fid]['description'],groups=domain['programs'][fid]['groups'],
        relations=domain['programs'][fid]['relations']) for fid in domain['slots'][role][1:]]


def presence_prompt(domain,h,role):
    return ('Sample whether this possible existing world contains AN ADDITIONAL HIDDEN FUNCTIONAL PROGRAM of the specified role. '
        'The program can contain furniture AND ordinary belongings jointly, including multiple activity groups or repeated categories. '
        'It can serve observed anchors without a second independent activity zone. '
        'H is uncertain and non-exhaustive: a missing function tag is not evidence of no possessions of that role. '
        'Do not test whether hidden objects are visible or whether another object is strictly needed. '
        'Existing photographed equipment does not establish absence of other possessions. '
        'This is optional generative sampling: presence and absence are both legal; no room density, count or rear target. '
        'Return one JSON object with exactly the key present and a JSON boolean value; no inventory list or explanation. '
        '\nH: '+json.dumps(h,ensure_ascii=False)+'\nROLE: '+role+
        '\nFIXED_POSITIVE_CONTEXT: '+json.dumps(domain['positive_context'],ensure_ascii=False)+
        '\nALTERNATIVE_COMPLETE_PROGRAMS: '+json.dumps(program_menu(domain,role),ensure_ascii=False))


def presence_answer(value):
    if not isinstance(value,dict) or set(value)!={'present'} or type(value['present']) is not bool:raise ValueError('Return exactly present with a JSON boolean')
    return '1' if value['present'] else '0'


def variant_prompt(domain,h,role,aliases):
    return ('The additional hidden functional program of ROLE is PRESENT in this conditional mechanism. '
        'Sample ONE COMPLETE INVENTORY AND CAMERA PLAN from the allowed choices. '
        'The existence decision is already made; do not omit contents, decide absence again, or choose only one object. '
        'All objects of the chosen program coexist, including all groups of the same function. '
        'Other functional roles are sampled separately and can coexist; no role is mandatory. '
        'left/right/rear refer to ORIGINAL CAMERA coordinates. rear is Z<0, not behind a sofa. '
        'Facing or serving a sofa does not imply a camera-depth order or near relation. '
        'Use the COMPLETE spatial reasons, placements and fixed predicted observation geometry. '
        'No count or sector coverage target. Return exactly {"choice":"one allowed alias"}. '
        '\nH: '+json.dumps(h,ensure_ascii=False)+'\nROLE: '+role+
        '\nFIXED_POSITIVE_CONTEXT: '+json.dumps(domain['positive_context'],ensure_ascii=False)+
        '\nPROGRAMS: '+json.dumps(program_menu(domain,role),ensure_ascii=False)+
        '\nALLOWED_CHOICES: '+json.dumps([dict(choice=alias,**domain['spatial_choices'][cid]) for alias,cid in aliases.items()],ensure_ascii=False))


def raw(votes):
    counts=Counter(v for v in votes if v is not None)
    if not counts:raise ValueError('No valid draws in a required mechanism row; no empty/default fallback')
    return dict(counts=dict(counts),valid_draws=sum(counts.values()),invalid_draws=sum(v is None for v in votes))


def predictive(row,states,concentration,prior):
    if sum(row['counts'].values())!=row['valid_draws'] or row['valid_draws']<1 or any(type(n) is not int or n<0 for n in row['counts'].values()):raise ValueError('Invalid raw counts')
    if set(row['counts'])-set(states) or abs(sum(prior.values())-1)>1e-9:raise ValueError('Predictive support mismatch')
    return [(row['counts'].get(s,0)+concentration*prior[s])/(row['valid_draws']+concentration) for s in states]


def build_model(hypotheses,programs,unused=None):
    return population.build_model(hypotheses,programs,signature_fields=('layout_region','functional_links'))


def compile_model(base,domain,table,policy='functional'):
    if policy not in ('functional','function_free'):raise ValueError('Unknown policy')
    m=deepcopy(base);hs=[h['id'] for h in domain['profiles']];q=table['recognition']['probabilities'];alpha=table['concentration'];eta=table['variant_concentration']
    if set(q)!=set(hs):raise ValueError('H support mismatch')
    nodes=[dict(id='H',parents=[],states=hs,cpt={'root':[q[h] for h in hs]}),
        dict(id='J',parents=['H'] if policy=='functional' else [],states=hs,
             cpt={h:[float(v==h) for v in hs] for h in hs} if policy=='functional' else {'root':[q[h] for h in hs]})]
    ar={};vr={};state_role={};group_nodes={}
    for role,programs in domain['slots'].items():
        allowed=[cid for cid,v in domain['spatial_choices'].items() if v['module'] in programs[1:]]
        prior={cid:1/(len(programs)-1)/sum(v['module']==domain['spatial_choices'][cid]['module'] for v in domain['spatial_choices'].values()) for cid in allowed}
        ar[role]={h:predictive(table['presence'][h+'|'+role],['0','1'],alpha,{'0':.5,'1':.5}) for h in hs}
        vr[role]={h:predictive(table['variants'][h+'|'+role],allowed,eta,prior) for h in hs}
        nodes.append(dict(id='A_'+role,parents=['J'],states=['0','1'],cpt=ar[role]))
        nodes.append(dict(id='M_'+role,parents=['J','A_'+role],states=['off']+allowed,
            cpt={key([h,a]):([1.]+[0.]*len(allowed) if a=='0' else [0.]+vr[role][h]) for h in hs for a in ('0','1')}))
        for fid in programs[1:]:
            state_role[fid]=role
            for group in domain['programs'][fid]['groups']:
                rid='R_'+fid+'_'+group['key'];group_nodes[fid+'_'+group['key']]=rid
                nodes.append(dict(id=rid,parents=['M_'+role],states=['off']+list(panorama.REGIONS),
                    cpt={s:[float(r==('off' if s=='off' or domain['spatial_choices'][s]['module']!=fid else domain['spatial_choices'][s]['regions'][group['key']])) for r in ['off']+list(panorama.REGIONS)] for s in ['off']+allowed}))
    for node in base['nodes']:
        if node['id'].startswith('E_'):
            o=next(o for o in base['component_registry'].values() if o['presence_node']==node['id']);fid=o['furniture_state'];r=state_role[fid]
            states=next(n['states'] for n in nodes if n['id']=='M_'+r)
            nodes.append(dict(id=node['id'],parents=['M_'+r],states=['0','1'],cpt={s:[float(not(s!='off' and domain['spatial_choices'][s]['module']==fid)),float(s!='off' and domain['spatial_choices'][s]['module']==fid)] for s in states}))
        elif node['id'].startswith('S_'):nodes.append(deepcopy(node))
    m.update(nodes=nodes,profiles=domain['profiles'],slots=domain['slots'],state_role=state_role,spatial_choices=domain['spatial_choices'],group_region_nodes=group_nodes,
        activation_rows=ar,variant_rows=vr,mechanism_policy=policy,mechanism_source=table['source'],empirical_tables=table,
        model_variant=MODEL_VARIANT,response_interface=INTERFACE,training_performed=False,
        maximum_hidden_instances=sum(max(sum(len(g['objects']) for g in domain['programs'][f]['groups']) for f in fs[1:]) for fs in domain['slots'].values()),
        distribution_scope='optional_complete_functional_programs_with_fresh_spatial_predictives_not_calibrated_world_posterior',
        generation_factorization='q(H|O_positive,D) product_role p(A_role|H,O_positive,D) p(V_role,R_role|A_role,H,O_positive,D); V->E->S<-S_host; (O,H,V,R)->X',
        sampling_policy='IID_all_empty_duplicates_and_failures_retained',conditional_independence_assumed='Complete functional programs independent given H; groups/contents inside each program jointly selected',
        evidence_delivery='Fixed positive observation geometry; text-only local/API mechanisms and optional matched layout RGB',
        candidate_support=dict(candidate_instances=len(m['component_registry']),nonempty_programs=len(state_role),all_nonempty_proposals_missing=False),
        calibrated_posterior_defined=False,real_causal_structure_identified=False,RGB_likelihood_defined=False)
    return validate(m)


def prepare_model(base,case,config,out):
    bundle=Path(config['sdscene_bundle']);d=read(bundle/'domain.json');source='pooled' if config['mechanism_source']=='pooled' else config['execution_backend']
    t=read(bundle/'mechanisms'/(source+'.json'))
    if t['kind']!=KIND or t['domain_sha256']!=sha(bundle/'domain.json'):raise ValueError('Frozen program domain mismatch')
    m=compile_model(base,d,t,config['mechanism_policy'])
    if out is not None:
        write(out/'empirical_mechanisms.json',t);write(out/'probability_audit.json',marginals(m))
        write(out/'functional_probability_interventions.json',{h['id']:marginals(m,h['id']) for h in m['profiles']})
        (out/'probability_audit.html').write_text('<meta charset="utf-8"><h1>完整功能程序概率</h1><p>存在、内部装配和方位均进入实际采样；概率未校准。</p><pre>'+escape(json.dumps(marginals(m),ensure_ascii=False,indent=2))+'</pre>')
    return m


def marginals(m,do_profile=None):
    q=dict(zip(m['nodes'][0]['states'],m['nodes'][0]['cpt']['root']))
    hprob=dict(q)
    if do_profile is not None:
        if do_profile not in q:raise ValueError('Unknown H')
        hprob={h:float(h==do_profile) for h in q}
        if m['mechanism_policy']=='functional':q={h:float(h==do_profile) for h in q}
    counts={fid:Counter(o['category'].lower().replace('_',' ').strip() for g in m['furniture_states'][fid]['groups'] for o in g['objects']) for fid in m['state_role']}
    cats={c for v in counts.values() for c in v};cp=dict.fromkeys(cats,0.);ce=dict(cp);assembly=dict.fromkeys(counts,0.);activation=dict.fromkeys(m['slots'],0.);pmf={};regions=dict.fromkeys(panorama.REGIONS,0.)
    for h,qh in q.items():
        local={0:1.};absent=dict.fromkeys(cats,1.);missing=dict.fromkeys(panorama.REGIONS,1.)
        for role,fs in m['slots'].items():
            on=m['activation_rows'][role][h][1];activation[role]+=qh*on
            states=next(n['states'][1:] for n in m['nodes'] if n['id']=='M_'+role);nlaw={0:1-on};present=dict.fromkeys(cats,0.);rprob=dict.fromkeys(panorama.REGIONS,0.)
            for s,p in zip(states,m['variant_rows'][role][h]):
                v=m['spatial_choices'][s];fid=v['module'];p*=on;n=sum(counts[fid].values());nlaw[n]=nlaw.get(n,0)+p;assembly[fid]+=qh*p
                for c,nc in counts[fid].items():present[c]+=p;ce[c]+=qh*p*nc
                for r in set(v['regions'].values()):rprob[r]+=p
            for c in cats:absent[c]*=1-present[c]
            for r in regions:missing[r]*=1-rprob[r]
            nxt={}
            for n,p in local.items():
                for k,v in nlaw.items():nxt[n+k]=nxt.get(n+k,0)+p*v
            local=nxt
        for c in cats:cp[c]+=qh*(1-absent[c])
        for r in regions:regions[r]+=qh*(1-missing[r])
        for n,p in local.items():pmf[str(n)]=pmf.get(str(n),0)+qh*p
    objects={oid:assembly[o['furniture_state']] for oid,o in m['component_registry'].items()}
    return dict(profile_probabilities=hprob,functional_context_mixture_probabilities=q,assembly_probabilities=assembly,activation_probabilities=activation,object_presence=objects,
        category_presence=cp,expected_category_counts=ce,hidden_count_probabilities=pmf,empty_probability=pmf.get('0',0),
        expected_hidden_instances=sum(objects.values()),any_declared_region_presence=regions,probability_scope=m['distribution_scope'],probabilities_calibrated=False,do_profile=do_profile)


def active_world(m,state):
    mapped=deepcopy(state)
    for role in m['slots']:mapped['M_'+role]='none' if state['M_'+role]=='off' else m['spatial_choices'][state['M_'+role]]['module']
    world=modular.active_world(m,mapped);owned={o['id'] for o in world['objects']};spatial={};links=[];unresolved=[]
    for group in world['groups']:
        rid=m['group_region_nodes'][group['key']];r=state[rid];group['layout_region']=r;spatial[group['key']]=dict(region=r,region_node=rid)
        fid=next(o['furniture_state'] for o in world['objects'] if o['group']==group['key']) if any(o['group']==group['key'] for o in world['objects']) else None
        mapping={o['key']:o['id'] for o in m['component_registry'].values() if o['furniture_state']==fid}
        for link in group.get('functional_links',[]):
            row=dict(link,source=mapping.get(link['source'],link['source']),target=mapping.get(link['target'],link['target']))
            (links if row['source'] in owned and (row['target'] in owned or row['target'].startswith('obs_')) else unresolved).append(row)
    world.update(structural_state=deepcopy(state),spatial_program=spatial,functional_links=links,unresolved_functional_links=unresolved)
    return world


def sample_worlds(m,seed=3407,count=4):
    if type(count) is not int or count<1:raise ValueError('Positive count required')
    rng=np.random.default_rng(seed);law=marginals(m);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in m['nodes']};w=active_world(m,forward(m,noise))
        for o in w['objects']:o['presence_probability']=law['object_presence'][o['id']]
        w.update(noise=noise,seed=seed,trial=trial,probability_scope=m['distribution_scope'],mechanism_source=m['mechanism_source']);worlds.append(w)
    return worlds


def intervention_diagnostics(m,w):
    rows=[]
    for o in w['objects']:
        if any(v['attachment']['host']==o['id'] for v in w['objects']):
            after=active_world(m,forward(m,w['noise'],{o['presence_node']:'0'}));rows.append(dict(do={o['presence_node']:'0'},after_inventory=[v['id'] for v in after['objects']],unresolved_installations=after['unresolved_installations']))
    return dict(queries=rows,functional_probabilities={h['id']:marginals(m,h['id']) for h in m['profiles']},same_exogenous_noise=True,
        model_called=False,geometry_not_resolved=True,scope='Assumed program SCM; not identified physical counterfactuals')
