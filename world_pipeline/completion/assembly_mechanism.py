"""Inference-only reference shrinkage and a direct joint-inventory control.

The graph is assumed, not identified from RGB. No weights or scene-data priors
are trained. Reference draws are D-only; direct draws see O,D but no sampled H.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

from ..common import read,write,sha
from . import assembly_realization as v7,assembly_sdscene as sd
from .world_scm import validate,forward,key

PROTOCOL=v7.PROTOCOL
INTERFACE='reference_shrunk_realization_v8'
MODEL_VARIANT='reference_shrunk_functional_scene_scm_v8'
KIND = 'reference_shrunk_realization'
MODES=v7.MODES
FUNCTIONS=v7.FUNCTIONS
profile=v7.profile
hypotheses_prompt=v7.hypotheses_prompt
hypotheses_answer=v7.hypotheses_answer
recognition_prompt=v7.recognition_prompt
furniture_prompt=v7.furniture_prompt
activation_prompt=v7.activation_prompt
activation_answer=v7.activation_answer
variant_prompt=v7.variant_prompt
build_model=v7.build_model
load_domain=v7.load_domain
active_world=v7.active_world
semantic_diagnostics=v7.semantic_diagnostics
intervention_diagnostics=v7.intervention_diagnostics


def furniture_answer(value,case,hypothesis):
    """Keep legal COMPLETE alternatives, never pieces of an illegal one.

    A reply with any legal alternative is accepted without an extra retry.
    If none is legal, the caller's existing single schema-repair budget applies.
    """
    rows=value.get('alternatives') if isinstance(value,dict) else None
    if not isinstance(rows,list) or not 1<=len(rows)<=2:
        raise ValueError('alternatives must contain 1..2 complete hidden-world deltas')
    accepted=[];rejected=[]
    for i,alt in enumerate(rows):
        try:
            programs=v7.furniture_answer(dict(alternatives=[deepcopy(alt)]),case,hypothesis)
        except (ValueError,KeyError,TypeError,AttributeError) as exc:
            rejected.append(dict(alternative=i,error=type(exc).__name__+': '+str(exc)));continue
        for program in programs:
            program['proposal_alternative_index']=i;accepted.append(program)
    if not accepted:
        raise ValueError('No legal complete alternative: '+json.dumps(rejected,ensure_ascii=False))
    audit=dict(policy='accept_all_legal_complete_alternatives_in_first_usable_reply',
        accepted_indices=[p['proposal_alternative_index'] for p in accepted],rejected=rejected,
        objects_salvaged_from_invalid_alternatives=False)
    for program in accepted:program['proposal_validation']=deepcopy(audit)
    return accepted


def reference_prompt(domain,order):
    menu=[]
    for c in order:
        nouns=set();anchors=set()
        for s in domain['slots'][c]:
            if s=='none':continue
            for g in domain['programs'][s]['groups']:
                nouns.update(o['category'] for o in g['objects']);anchors.update(g.get('observed_refs',[]))
        menu.append(dict(id=c,**domain['channel_definitions'][c],possible_object_nouns=sorted(nouns),observed_anchor_ids=sorted(anchors)))
    return ('Sample ONE joint set of existing hidden realization channels from FIXED_DOMAIN_D only. '
        'No particular latent use profile is supplied. Do not infer or condition on one specific profile. '
        'observed_extension serves existing anchors; new_group is an additional arrangement. Both may coexist. '
        'Any subset including empty is allowed. No object-count goal or minimum furnishing requirement. '
        'Return exactly {"active_channels":[exact channel IDs]}. '
        '\nAVAILABLE_REALIZATION_CHANNELS: '+json.dumps(menu,ensure_ascii=False)+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'],ensure_ascii=False))


def direct_prompt(domain,order):
    """Neutral option IDs; no H, recognition votes, or mechanism answers."""
    canonical=list(domain['slots']);menu=[]
    for c in order:
        options=[dict(id='none',meaning='No hidden assembly in this slot')]
        for s in domain['slots'][c]:
            if s=='none':continue
            p=domain['programs'][s]
            options.append(dict(id=s,objects=[dict(key=o['key'],category=o['category'],variant=o['variant'],attachment=o['attachment'])
                for g in p['groups'] for o in g['objects']],relations=[{k:r[k] for k in ('source','target','kind')} for r in p['relations']]))
        menu.append(dict(slot='k'+str(canonical.index(c)),options=options))
    return ('From the attached LOCAL OBSERVATION and fixed candidate domain, sample ONE possible complete hidden inventory. '
        'Choose one exact option for EVERY neutral slot, jointly accounting for coexistence, redundancy and observed anchors. '
        'none is legal in every slot; all-none is legal. Choose a plausible existing world, not everything possible or a minimum furnishing plan. '
        'No new identities, no count target. Return exactly {"choices":{"k0":"option ID",...}}, with every supplied slot once. '
        '\nNEUTRAL_SLOT_DOMAIN: '+json.dumps(menu,ensure_ascii=False)+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'],ensure_ascii=False))


def direct_answer(value,domain):
    slots=list(domain['slots']);wanted={'k'+str(i) for i in range(len(slots))}
    if not isinstance(value,dict) or set(value)!={'choices'} or not isinstance(value['choices'],dict) or set(value['choices'])!=wanted:
        raise ValueError('Return choices with every exact neutral slot once: '+str(sorted(wanted)))
    values=[]
    for i,c in enumerate(slots):
        v=value['choices']['k'+str(i)]
        if not isinstance(v,str) or v not in domain['slots'][c]:raise ValueError('Invalid option in k'+str(i))
        values.append(v)
    return json.dumps(values,separators=(',',':'))


def raw_activation(draws,channels):
    valid=[a for a in draws if isinstance(a,str) and len(a)==len(channels) and set(a)<=set('01')]
    if not valid:raise ValueError('No valid activation draws; not absence or uniform fallback')
    return dict(counts=dict(sorted(Counter(valid).items())),valid_draws=len(valid),invalid_draws=len(draws)-len(valid),
        scope='Raw counts; conditional predictive uses frozen reference base, not uniform-mask smoothing')


def raw_direct(draws,domain):
    valid=[]
    for d in draws:
        if not isinstance(d,str):continue
        try:
            values=json.loads(d)
            if not isinstance(values,list) or len(values)!=len(domain['slots']):continue
            if any(not isinstance(v,str) or v not in states for v,states in zip(values,domain['slots'].values())):continue
        except (ValueError,TypeError):continue
        valid.append(json.dumps(values,separators=(',',':')))
    if not valid:raise ValueError('No valid direct inventory draws; not an empty-world inference')
    return dict(counts=dict(sorted(Counter(valid).items())),valid_draws=len(valid),invalid_draws=len(draws)-len(valid))


def _positive(x,name):
    if type(x) not in (int,float) or not math.isfinite(x) or x<=0:raise ValueError(name+' must be finite and positive')


def _count_row(row):
    counts=row['counts'];k=row['valid_draws']
    if type(k) is not int or k<1 or any(type(n) is not int or n<0 for n in counts.values()) or sum(counts.values())!=k:
        raise ValueError('Inconsistent or unavailable raw mechanism counts')
    return counts,k


def activation_rows(activation,reference,channels,alpha):
    _positive(alpha,'concentration');rc,rk=_count_row(reference);eta=reference['concentration'];_positive(eta,'reference concentration')
    rows=[_count_row(r) for r in activation.values()]
    patterns=sorted(set(rc)|{a for c,k in rows for a in c})
    if any(not isinstance(a,str) or len(a)!=len(channels) or set(a)-set('01') for a in patterns):raise ValueError('Invalid joint mask')
    components={'prior':[.5]*len(channels)}|{a:[float(v) for v in a] for a in patterns}
    q0={a:rc.get(a,0)/(rk+eta) for a in patterns};q0['prior']=eta/(rk+eta)
    ar={};rr={}
    for h,row in activation.items():
        counts,k=_count_row(row)
        rr[h]={j:alpha*q0[j]/(k+alpha) for j in components}
        ar[h]={j:rr[h][j]+(0 if j=='prior' else counts.get(j,0)/(k+alpha)) for j in components}
    return components,ar,rr


def _direct_components(table,slots,hs,alpha):
    rc,rk=_count_row(table['reference']);eta=table['reference']['concentration'];_positive(eta,'reference concentration')
    dc,k=_count_row(table['direct']);components={'prior':[.5]*len(slots)};weights={'prior':alpha/(k+alpha)*eta/(rk+eta)}
    variants={'prior':{r:{v:1/(len(states)-1) for v in states if v!='none'} for r,states in slots.items()}}
    reference=dict(weights)
    for mask,n in sorted(rc.items()):
        if len(mask)!=len(slots) or set(mask)-set('01'):raise ValueError('Invalid reference mask')
        j='b_'+mask;components[j]=[float(v) for v in mask];weights[j]=alpha/(k+alpha)*n/(rk+eta);reference[j]=weights[j]
        variants[j]=deepcopy(variants['prior'])
    for i,(raw,n) in enumerate(sorted(dc.items())):
        values=json.loads(raw)
        if len(values)!=len(slots) or any(v not in states for v,states in zip(values,slots.values())):raise ValueError('Direct row/domain mismatch')
        j='d_'+str(i);components[j]=[float(v!='none') for v in values];weights[j]=n/(k+alpha);reference[j]=0.
        variants[j]={r:({v:float(v==chosen) for v in states if v!='none'} if chosen!='none' else deepcopy(variants['prior'][r]))
            for (r,states),chosen in zip(slots.items(),values)}
    return components,{h:deepcopy(weights) for h in hs},{h:deepcopy(reference) for h in hs},variants


def prepare_model(base,case,config,out):
    bundle=Path(config['sdscene_bundle']);domain=read(bundle/'domain.json');source=config['mechanism_source']
    if source=='per_backend':source=config['execution_backend']
    table=read(bundle/'mechanisms'/(source+'.json'))
    if table.get('kind')!=config.get('mechanism_table_kind',KIND) or table['domain_sha256']!=sha(bundle/'domain.json'):raise ValueError('v8 mechanism/domain binding differs')
    alpha=table['concentration'];_positive(alpha,'concentration')
    m=deepcopy(base);hs=[h['id'] for h in domain['profiles']];slots=domain['slots'];hp=table['recognition']['probabilities']
    if set(hp)!=set(hs) or not slots:raise ValueError('v8 recognition or slot domain differs')
    policy=config.get('mechanism_policy','functional')
    if policy not in ('functional','function_free','direct'):raise ValueError('Unknown v8 policy')
    component_variants={}
    if policy=='direct':
        components,ar,rr,component_variants=_direct_components(table,slots,hs,alpha);vr={}
    else:
        if set(table['activation'])!=set(hs):raise ValueError('Activation H domain differs')
        components,ar,rr=activation_rows(table['activation'],table['reference'],list(slots),alpha)
        vr={r:{h:table['variants'][h+'|'+r]['probabilities'] for h in hs} for r in slots}
        if any(set(row)!=set(slots[r])-{'none'} for r in slots for row in vr[r].values()):raise ValueError('Conditional variant domain differs')
        if policy=='function_free':
            for i,r in enumerate(slots):
                weights={h:hp[h]*sum(ar[h][j]*components[j][i] for j in components) for h in hs};total=sum(weights.values())
                if total<=0:raise ValueError('Activation has no support for a legal channel')
                average={v:sum(weights[h]*vr[r][h][v] for h in hs)/total for v in slots[r] if v!='none'}
                vr[r]={h:deepcopy(average) for h in hs}
            for rows in (ar,rr):
                average={j:sum(hp[h]*rows[h][j] for h in hs) for j in components}
                for h in hs:rows[h]=deepcopy(average)
    js=list(components);functional=policy=='functional'
    nodes=[dict(id='H',parents=[],states=hs,cpt={'root':[hp[h] for h in hs]}),
        dict(id='J',parents=['H'] if functional else [],states=js,
            cpt={h:[ar[h][j] for j in js] for h in hs} if functional else {'root':[ar[hs[0]][j] for j in js]})]
    state_slot={}
    for i,(r,states) in enumerate(slots.items()):
        nodes.append(dict(id='A_'+r,parents=['J'],states=['0','1'],cpt={j:[1-components[j][i],components[j][i]] for j in js}))
        rows={}
        parents=['J','A_'+r] if policy=='direct' else ['H','A_'+r] if functional else ['A_'+r]
        contexts=js if policy=='direct' else hs if functional else [hs[0]]
        for context in contexts:
            probs=component_variants[context][r] if policy=='direct' else vr[r][context]
            for bit in ('0','1'):
                ix=key([context,bit] if len(parents)==2 else [bit])
                rows[ix]=[float(s=='none') if bit=='0' else 0. if s=='none' else probs[s] for s in states]
        nodes.append(dict(id='M_'+r,parents=parents,states=states,cpt=rows));state_slot.update({s:r for s in states if s!='none'})
    registry=m['component_registry'];by_node={o['presence_node']:o for o in registry.values()}
    for node in base['nodes']:
        if node['id'].startswith('E_'):
            obj=by_node[node['id']];fid=obj['furniture_state'];r=state_slot[fid]
            nodes.append(dict(id=node['id'],parents=['M_'+r],states=['0','1'],cpt={s:[float(s!=fid),float(s==fid)] for s in slots[r]}))
        elif node['id'].startswith('S_'):nodes.append(deepcopy(node))
    m.update(nodes=nodes,profiles=domain['profiles'],slots=slots,state_slot=state_slot,joint_components=components,
        activation_rows=ar,reference_rows=rr,variant_rows=vr,component_variant_rows=component_variants,
        mechanism_source=source,activation_policy=policy,mechanism_policy=policy,model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        empirical_tables=table,channel_definitions=domain['channel_definitions'],
        maximum_hidden_instances=sum(max(sum(len(g['objects']) for g in domain['programs'][s]['groups']) for s in states if s!='none') for states in slots.values()),
        distribution_scope='finite_domain_reference_shrunk_conditional_inventory_not_calibrated_world_probability',
        generation_factorization=('p_direct(V_joint|O,D); V->E->S<-S_host; H retained for matched metric context' if policy=='direct' else
            'q(H|O,D) p_ref_shrunk(A_joint|H,D) product_s p(V_s|H,A_s,D); V->E->S<-S_host; (O,H,V)->X'),
        conditional_independence_assumed='Channels independent given H,J; mixture represents joint activation and direct joint inventory. Finite D is incomplete.',
        joint_mixture_scope='J is a numeric mixture representation, not a physical cause or biography',
        sampling_policy='IID_all_null_duplicates_failures_retained',RGB_likelihood_defined=False,count_likelihood_defined=False,
        calibrated_posterior_defined=False,real_causal_structure_identified=False,training_performed=False,
        evidence_delivery=('RGB_at_direct_inventory_and_layout; no_H_in_direct_inventory; H_retained_in_metric' if policy=='direct' else
            'RGB_at_proposal_recognition_layout; reference_D_only; conditional_activation_and_variants_H_D_only'),
        candidate_support=dict(candidate_instances=len(registry),nonempty_programs=len(state_slot),all_nonempty_proposals_missing=not registry))
    if policy=='function_free':m['generation_factorization']='q(H|O,D) p_m(A_joint|D) product_s p_m(V_s|A_s,D); H inventory edges removed, H metric retained'
    validate(m)
    if out is not None:
        write(out/'empirical_mechanisms.json',table);write(out/'probability_audit.json',marginals(m))
        laws={h:marginals(m,h) for h in hs};write(out/'functional_probability_interventions.json',laws)
        diagnostic=dict(policy=policy,probabilities_calibrated=False,training_performed=False,
            reference_base=table['reference'],mechanism_uncertainty=uncertainty(m),
            matched_profile_effects=[dict(p,activation_delta={c:laws[p['with_role']]['activation_probabilities'][c]-laws[p['without']]['activation_probabilities'][c] for c in slots})
                for p in domain.get('matched_profile_pairs',[])],causal_gain_validated=False)
        write(out/'mechanism_diagnostics.json',diagnostic);probability_page(m,out/'probability_audit.html')
        if policy=='functional':
            write(out/'uniform_reference_ablation.json',dict(scope='Same D, q(H), raw conditional counts and variants; only activation base replaced. No quality gain claimed.',
                reference_shrunk=marginals(m),uniform_base=uniform_reference_ablation(m)))
    return m


def _variants(m,h,j,r):
    return m['component_variant_rows'][j][r] if m['mechanism_policy']=='direct' else m['variant_rows'][r][h]


def marginals(m,do_profile=None):
    hs=[h['id'] for h in m['profiles']]
    if do_profile is not None and do_profile not in hs:raise ValueError('Unknown profile')
    q=dict(zip(hs,m['nodes'][0]['cpt']['root']))
    if do_profile is not None:q={h:float(h==do_profile) for h in hs}
    counts={s:Counter(o['category'].lower().replace('_',' ').strip() for g in m['furniture_states'][s]['groups'] for o in g['objects']) for s in m['state_slot']}
    cats=sorted({c for v in counts.values() for c in v});cp=dict.fromkeys(cats,0.);ce=dict(cp)
    activation=dict.fromkeys(m['slots'],0.);assembly=dict.fromkeys(m['state_slot'],0.);pmf={};mixture=dict.fromkeys(m['joint_components'],0.)
    reference_n=0.;uniform_n=0.
    for h in hs:
        if not q[h]:continue
        for j,pj in m['activation_rows'][h].items():
            if not pj:continue
            w=q[h]*pj;mixture[j]+=w;local={0:1.};absent=dict.fromkeys(cats,1.);conditional_n=0.
            for i,r in enumerate(m['slots']):
                on=m['joint_components'][j][i];activation[r]+=w*on;nlaw={0:1-on};present=dict.fromkeys(cats,0.)
                for s,p in _variants(m,h,j,r).items():
                    prob=on*p;assembly[s]+=w*prob;n=sum(counts[s].values());nlaw[n]=nlaw.get(n,0)+prob;conditional_n+=prob*n
                    for c,nc in counts[s].items():present[c]+=prob;ce[c]+=w*prob*nc
                for c in cats:absent[c]*=1-present[c]
                nxt={}
                for n,p in local.items():
                    for z,t in nlaw.items():nxt[n+z]=nxt.get(n+z,0)+p*t
                local=nxt
            reference_n+=q[h]*m['reference_rows'][h][j]*conditional_n
            if j=='prior':uniform_n+=w*conditional_n
            for c in cats:cp[c]+=w*(1-absent[c])
            for n,p in local.items():pmf[str(n)]=pmf.get(str(n),0)+w*p
    objects={oid:assembly[o['furniture_state']] for oid,o in m['component_registry'].items()}
    return dict(profile_probabilities=q,activation_probabilities=activation,joint_activation_mixture=mixture,
        assembly_probabilities=assembly,object_presence=objects,category_presence=cp,expected_category_counts=ce,
        hidden_count_probabilities=pmf,empty_probability=pmf.get('0',0),expected_hidden_instances=sum(objects.values()),
        expected_hidden_from_reference=reference_n,expected_hidden_from_uniform_base=uniform_n,
        expected_hidden_from_activation_smoothing=reference_n,activation_prior_component_probability=mixture.get('prior',0),
        do_profile=do_profile,probabilities_calibrated=False,probability_scope=m['distribution_scope'],
        conditional_independence_assumed=m['conditional_independence_assumed'])


def uncertainty(m):
    if m['mechanism_policy']!='functional':return dict(scope='No applied H-conditional Beta intervals for this policy; original rows remain in empirical_mechanisms.json')
    t=m['empirical_tables'];alpha=t['concentration'];rc,rk=_count_row(t['reference']);eta=t['reference']['concentration'];result=[]
    for i,c in enumerate(m['slots']):
        pi=(sum(n for a,n in rc.items() if a[i]=='1')+eta/2)/(rk+eta)
        for h,row in t['activation'].items():
            counts,k=_count_row(row);n=sum(v for a,v in counts.items() if a[i]=='1');a=n+alpha*pi;b=k-n+alpha*(1-pi)
            result.append(dict(profile=h,channel=c,on_votes=n,valid_draws=k,reference_mean=pi,mean=a/(a+b),
                reference_contribution=alpha*pi/(k+alpha),conditional_interval_95=beta.ppf([.025,.975],a,b).tolist(),
                zero_conditional_votes=n==0))
    return dict(rows=result,scope='IID valid LM replies, conditional on fixed reference base. Excludes reference uncertainty, prompt bias, domain gaps, q(H) uncertainty; not real-world calibration.')


def uniform_reference_ablation(m):
    if m['mechanism_policy']!='functional':raise ValueError('Uniform-base comparison requires functional model')
    u=deepcopy(m);alpha=m['empirical_tables']['concentration'];node=next(n for n in u['nodes'] if n['id']=='J')
    for h,row in m['empirical_tables']['activation'].items():
        counts,k=_count_row(row)
        u['activation_rows'][h]={j:alpha/(k+alpha) if j=='prior' else counts.get(j,0)/(k+alpha) for j in u['joint_components']}
        u['reference_rows'][h]={j:alpha/(k+alpha) if j=='prior' else 0. for j in u['joint_components']}
        node['cpt'][h]=[u['activation_rows'][h][j] for j in node['states']]
    validate(u)
    return marginals(u)


def sample_worlds(m,seed=3407,count=4):
    if type(count) is not int or count<1:raise ValueError('Positive world count required')
    rng=np.random.default_rng(seed);law=marginals(m);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in m['nodes']};world=active_world(m,forward(m,noise))
        for obj in world['objects']:obj['presence_probability']=law['object_presence'][obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,mechanism_source=m['mechanism_source'],probability_scope=m['distribution_scope']);worlds.append(world)
    return worlds


def probability_page(m,path):
    path.write_text('<!doctype html><meta charset="utf-8"><h1>v8：免训练参考收缩与直接库存对照</h1>'
        '<p>原始票数、参考机制和均匀尾部分开记录。没有最低数量；概率未完成真实世界校准。</p>'
        '<p><a href="mechanism_diagnostics.json">机制来源与条件不确定性</a> · <a href="functional_probability_interventions.json">逐 H 干预</a></p><pre>'+
        escape(json.dumps(dict(policy=m['mechanism_policy'],law=marginals(m)),ensure_ascii=False,indent=2))+'</pre>')
