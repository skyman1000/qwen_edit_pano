"""Observed-anchored extensions and new groups have separate existence paths.

Sparse empirical/product-Bernoulli mixture is exactly a symmetric Dirichlet
predictive over all joint masks, without enumerating their exponential domain.
It is a numerical representation, not an inferred physical mixture cause.
"""
from collections import Counter
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path

import numpy as np

from ..common import read,write,sha
from . import assembly_activation as previous, assembly_modular as modular
from .world_scm import forward,validate,key

PROTOCOL=previous.PROTOCOL
INTERFACE='anchored_extensions_and_new_groups_v7'
MODEL_VARIANT='anchored_realization_scene_scm_v7'
MODES=('observed_extension','new_group')
FUNCTIONS=previous.FUNCTIONS
profile=previous.profile
hypotheses_prompt=previous.hypotheses_prompt
hypotheses_answer=previous.hypotheses_answer
recognition_prompt=previous.recognition_prompt
build_model=previous.build_model
load_domain=previous.load_domain
active_world=modular.active_world
semantic_diagnostics=previous.semantic_diagnostics
variant_prompt=previous.variant_prompt


def furniture_prompt(case,hypothesis,channel='surrounding'):
    from .assembly_taxonomy import vocabulary
    prompt=previous.furniture_prompt(case,hypothesis,channel).replace(
        'Multiple groups of the same function will be interpreted as alternatives, not automatically combined.',
        'Groups with the same function AND realization mode are alternatives; different modes may coexist.')
    prompt=prompt.replace('including smaller assemblies and versions without optional belongings when plausible;',
        'including genuinely different coexisting furniture and belonging inventories when plausible;')
    prompt=prompt.replace('The dataset label table is '+json.dumps(vocabulary()['categories'],ensure_ascii=False)+'. ','')
    return prompt+('\nV7 REQUIRED GROUP FIELDS: Every group has realization_mode equal to observed_extension or new_group. '
        'observed_extension means unseen objects serving one or more EXISTING observed anchors, without assuming an extra independent activity zone. '
        'It may include floor furniture such as a side table beside a visible sofa, as well as supported belongings. '
        'Specify nonempty observed_refs and at least one explicit relation or attachment to those exact observed IDs. '
        'new_group means a distinct additional functional arrangement. It need not be unconnected to the visible room. '
        'Both channels may coexist for the same function; no channel is mandatory. Do not infer absence from visible sufficiency. '
        'The surrounding/belongings query label does not determine realization_mode: choose it explicitly for each group. '
        'Full object sizes and counts remain uncertain; do not minimize or maximize them. '
        'category is the concrete noun, e.g. table lamp, book or pen; never the coarse labels lighting, furniture or objects. '
        'The executor computes dataset_category_label separately; do not put that coarse label in category. '
        'Do not describe unmeasured walls as observed. Use the existing complete schema otherwise.')


def furniture_answer(value,case,hypothesis):
    for alt in value.get('alternatives',[]):
        for group in alt.get('groups',[]):
            if group.get('realization_mode') not in MODES:raise ValueError('Each group.realization_mode must be observed_extension or new_group')
            if group['realization_mode']=='observed_extension' and not group.get('observed_refs'):
                raise ValueError('observed_extension requires nonempty exact observed_refs')
    programs=previous.furniture_answer(value,case,hypothesis)
    for raw,program in zip(value['alternatives'],programs):
        declared={'g'+str(i):g for i,g in enumerate(raw['groups'])}
        for group in program['groups']:
            declaration=declared[group['assembly_key']];mode=declaration['realization_mode']
            group['realization_mode']=mode
            if mode!='observed_extension':continue
            anchors=set(declaration['observed_refs']);objects=group['objects'];allkeys={o['key'] for g in program['groups']
                if g['assembly_key']==group['assembly_key'] for o in g['objects']}
            attached=any(o['attachment']['host'] in anchors for o in objects)
            related=any((r['source'] in allkeys and r['target'] in anchors) or (r['target'] in allkeys and r['source'] in anchors) for r in program['relations'])
            if not attached and not related:raise ValueError('observed_extension must explicitly relate/attach to its declared observed anchors')
    return programs


def activation_answer(value,channels):
    if not isinstance(value,dict) or set(value)!={'active_channels'}:raise ValueError('Return exactly {"active_channels": [zero or more exact channel IDs]}')
    return previous.activation_answer(dict(active_roles=value['active_channels']),channels)


def activation_prompt(domain,h,order):
    menu=[]
    for channel in order:
        definition=domain['channel_definitions'][channel];nounsets=set();anchors=set()
        for state in domain['slots'][channel]:
            if state=='none':continue
            p=domain['programs'][state];nounsets.update(o['category'] for g in p['groups'] for o in g['objects'])
            anchors.update(a for g in p['groups'] for a in g.get('observed_refs',[]))
        # Set-valued summaries carry semantics, without listing variant counts.
        menu.append(dict(id=channel,**definition,possible_object_nouns=sorted(nounsets),observed_anchor_ids=sorted(anchors)))
    return ('Sample ONE joint set of existing hidden realization channels, conditioned on H and this fixed domain. '
        'An observed_extension channel concerns unseen possessions/furniture serving visible anchors. '
        'It does NOT assert another independent activity area. A new_group channel concerns an additional arrangement. '
        'These channels can coexist, including for the same function. Visible equipment sufficiency neither requires nor excludes unseen possessions. '
        'Choose a plausible existing world, not a minimum furnishing plan or a list of everything conceivable. '
        'No count target, resident biography, measured room area, or known absent region is supplied. '
        'Any subset including the empty set is allowed. Do not select channels just because they are available. '
        'Return only {"active_channels":[exact channel IDs]}. '
        '\nCURRENT_PARENT_H: '+json.dumps(profile(h['id'],h['functions']))+
        '\nAVAILABLE_REALIZATION_CHANNELS: '+json.dumps(menu,ensure_ascii=False)+
        '\nFIXED_COMMON_OBSERVED_INVENTORY: '+json.dumps(domain['common_inventory'],ensure_ascii=False))


def estimate_activation(draws,channels,concentration=1.):
    if not math.isfinite(concentration) or concentration<=0:raise ValueError('Positive concentration required')
    valid=[v for v in draws if isinstance(v,str) and len(v)==len(channels) and set(v)<=set('01')]
    if not valid:raise ValueError('No valid joint activation draws; no empty/uniform fallback')
    counts=dict(sorted(Counter(valid).items()));k=len(valid)
    return dict(counts=counts,valid_draws=k,invalid_draws=len(draws)-k,concentration=concentration,
        empirical_component_weights={s:n/(k+concentration) for s,n in counts.items()},prior_weight=concentration/(k+concentration),
        prior='product_Bernoulli_half_over_all_channels',scope='exact_symmetric_Dirichlet_predictive_over_all_joint_masks')


def prepare_model(base,case,config,out):
    bundle=Path(config['sdscene_bundle']);domain=read(bundle/'domain.json');source=config['mechanism_source']
    if source=='per_backend':source=config['execution_backend']
    table=read(bundle/'mechanisms'/(source+'.json'))
    if table['domain_sha256']!=sha(bundle/'domain.json') or table['kind']!='anchored_realization':raise ValueError('Frozen v7 mechanism/domain mismatch')
    m=deepcopy(base);profiles=domain['profiles'];hs=[h['id'] for h in profiles];slots=domain['slots'];channels=list(slots)
    hp=table['recognition']['probabilities'];patterns=sorted({a for row in table['activation'].values() for a in row['counts']});js=['prior']+patterns
    if set(hp)!=set(hs):raise ValueError('Recognition profile domain differs')
    components={'prior':[.5]*len(channels)}|{a:[float(x) for x in a] for a in patterns}
    ar={h:{j:table['activation'][h]['prior_weight'] if j=='prior' else table['activation'][h]['empirical_component_weights'].get(j,0.) for j in js} for h in hs}
    vr={r:{h:table['variants'][h+'|'+r]['probabilities'] for h in hs} for r in channels}
    for r in channels:
        if any(set(row)!=set(slots[r])-{'none'} for row in vr[r].values()):raise ValueError('Variant domain differs')
    policy=config.get('realization_policy','functional')
    if policy not in ('functional','function_free'):raise ValueError('Unknown realization policy')
    def on(h,i):return sum(ar[h][j]*components[j][i] for j in js)
    if policy=='function_free':
        average={j:sum(hp[h]*ar[h][j] for h in hs) for j in js}
        for i,r in enumerate(channels):
            weights={h:hp[h]*on(h,i) for h in hs};total=sum(weights.values())
            row={v:sum(weights[h]*vr[r][h][v] for h in hs)/total for v in slots[r] if v!='none'}
            vr[r]={h:deepcopy(row) for h in hs}
        ar={h:deepcopy(average) for h in hs}
    nodes=[dict(id='H',parents=[],states=hs,cpt={'root':[hp[h] for h in hs]})]
    nodes.append(dict(id='J',parents=['H'] if policy=='functional' else [],states=js,
        cpt={h:[ar[h][j] for j in js] for h in hs} if policy=='functional' else {'root':[ar[hs[0]][j] for j in js]}))
    state_slot={}
    for i,r in enumerate(channels):
        nodes.append(dict(id='A_'+r,parents=['J'],states=['0','1'],cpt={j:[1-components[j][i],components[j][i]] for j in js}))
        states=slots[r];rows={}
        for h in (hs if policy=='functional' else [hs[0]]):
            for bit in ('0','1'):rows[key([h,bit] if policy=='functional' else [bit])]=[
                float(s=='none') if bit=='0' else 0. if s=='none' else vr[r][h][s] for s in states]
        nodes.append(dict(id='M_'+r,parents=['H','A_'+r] if policy=='functional' else ['A_'+r],states=states,cpt=rows))
        state_slot.update({s:r for s in states if s!='none'})
    for node in base['nodes']:
        if node['id'].startswith('E_'):
            obj=next(o for o in m['component_registry'].values() if o['presence_node']==node['id']);fid=obj['furniture_state'];r=state_slot[fid]
            nodes.append(dict(id=node['id'],parents=['M_'+r],states=['0','1'],cpt={s:[float(s!=fid),float(s==fid)] for s in slots[r]}))
        elif node['id'].startswith('S_'):nodes.append(deepcopy(node))
    maximum=sum(max(sum(len(g['objects']) for g in domain['programs'][s]['groups']) for s in states if s!='none') for states in slots.values())
    m.update(nodes=nodes,profiles=profiles,slots=slots,state_slot=state_slot,joint_components=components,
        activation_rows=ar,variant_rows=vr,activation_policy=policy,mechanism_source=source,model_variant=MODEL_VARIANT,response_interface=INTERFACE,
        channel_definitions=domain['channel_definitions'],empirical_tables=table,maximum_hidden_instances=maximum,
        distribution_scope='joint_anchored_extensions_and_new_groups_given_fixed_observations_and_domain',
        sampling_policy='IID_all_samples_including_null_duplicates_and_failures_preserved',
        generation_factorization='q(H|O,D) p(A_channels_joint|H,D) product_s p(V_s|H,A_s,D); V->E->S<-S_host; (O,H,V)->X',
        conditional_independence_assumed='Variants independent given H and own channel; activation joint; finite domain is not full-world truth',
        joint_mixture_scope='Exact predictive parameterization; J is not an inferred physical scene cause',
        RGB_likelihood_defined=False,count_likelihood_defined=False,calibrated_posterior_defined=False,real_causal_structure_identified=False,
        candidate_support=dict(candidate_instances=len(m['component_registry']),nonempty_programs=len(state_slot),all_nonempty_proposals_missing=not m['component_registry']))
    if policy=='function_free':m['generation_factorization']='q(H|O,D) p_m(A_channels_joint) product_s p_m(V_s|A_s,D); inventory H edges removed, metric H retained'
    validate(m)
    if out is not None:
        write(out/'empirical_mechanisms.json',table);write(out/'probability_audit.json',marginals(m))
        interventions={h:marginals(m,h) for h in hs}
        write(out/'functional_probability_interventions.json',interventions)
        write(out/'mechanism_diagnostics.json',dict(matched_profile_effects=[dict(pair,
            activation_delta={r:interventions[pair['with_role']]['activation_probabilities'][r]-interventions[pair['without']]['activation_probabilities'][r] for r in channels})
            for pair in domain.get('matched_profile_pairs',[])],causal_gain_validated=False))
        probability_page(m,out/'probability_audit.html')
    return m


def marginals(m,do_profile=None):
    hs=[h['id'] for h in m['profiles']]
    if do_profile is not None and do_profile not in hs:raise ValueError('Unknown profile')
    q=dict(zip(hs,m['nodes'][0]['cpt']['root']))
    if do_profile is not None:q={h:float(h==do_profile) for h in hs}
    cats=set();counts={}
    for fid in m['state_slot']:
        counts[fid]=Counter(o['category'].lower().replace('_',' ').strip() for g in m['furniture_states'][fid]['groups'] for o in g['objects']);cats.update(counts[fid])
    cp=dict.fromkeys(cats,0.);ce=dict.fromkeys(cats,0.);activation=dict.fromkeys(m['slots'],0.);assembly=dict.fromkeys(m['state_slot'],0.);pmf={};mixture=dict.fromkeys(m['joint_components'],0.);prior_expected=0.
    for h in hs:
        if q[h]==0:continue
        for j,pj in m['activation_rows'][h].items():
            if pj==0:continue
            w=q[h]*pj;mixture[j]+=w;local={0:1.};absent=dict.fromkeys(cats,1.)
            for i,(r,states) in enumerate(m['slots'].items()):
                on=m['joint_components'][j][i];activation[r]+=w*on;nlaw={0:1-on};presence=dict.fromkeys(cats,0.)
                for fid,p in m['variant_rows'][r][h].items():
                    prob=on*p;assembly[fid]+=w*prob;n=sum(counts[fid].values());nlaw[n]=nlaw.get(n,0.)+prob
                    if j=='prior':prior_expected+=w*prob*n
                    for c,nc in counts[fid].items():presence[c]+=prob;ce[c]+=w*prob*nc
                for c in cats:absent[c]*=1-presence[c]
                nxt={}
                for n,p in local.items():
                    for z,t in nlaw.items():nxt[n+z]=nxt.get(n+z,0.)+p*t
                local=nxt
            for c in cats:cp[c]+=w*(1-absent[c])
            for n,p in local.items():pmf[str(n)]=pmf.get(str(n),0.)+w*p
    objects={oid:assembly[o['furniture_state']] for oid,o in m['component_registry'].items()}
    return dict(profile_probabilities=q,activation_probabilities=activation,joint_activation_mixture=mixture,
        assembly_probabilities=assembly,object_presence=objects,category_presence=cp,expected_category_counts=ce,
        hidden_count_probabilities=pmf,empty_probability=pmf.get('0',0.),expected_hidden_instances=sum(objects.values()),
        expected_hidden_from_activation_smoothing=prior_expected,activation_prior_component_probability=mixture.get('prior',0.),
        do_profile=do_profile,probabilities_calibrated=False,probability_scope=m['distribution_scope'],conditional_independence_assumed=m['conditional_independence_assumed'])


def sample_worlds(m,seed=3407,count=4):
    if type(count) is not int or count<1:raise ValueError('Positive world count required')
    rng=np.random.default_rng(seed);law=marginals(m);worlds=[]
    for trial in range(count):
        noise={n['id']:float(rng.random()) for n in m['nodes']};world=active_world(m,forward(m,noise))
        for obj in world['objects']:obj['presence_probability']=law['object_presence'][obj['id']]
        world.update(noise=noise,seed=seed,trial=trial,mechanism_source=m['mechanism_source'],probability_scope=m['distribution_scope']);worlds.append(world)
    return worlds


def intervention_diagnostics(m,world):return previous.intervention_diagnostics(m,world)


def probability_page(m,path):
    law=marginals(m)
    path.write_text('<!doctype html><meta charset="utf-8"><h1>v7 可见区域延伸与新功能组：分别存在、联合采样</h1>'
        '<p>不设新增数目标。概率未经真实场景校准。J 是联合分布的紧凑表示，不是生活历史。</p>'
        '<p><a href="probability_audit.json">实际类别/实例/数量概率</a> · <a href="functional_probability_interventions.json">逐 H 数值干预</a> · '
        '<a href="mechanism_diagnostics.json">匹配 H 作用</a></p><pre>'+escape(json.dumps(dict(channels=m['channel_definitions'],law=law),ensure_ascii=False,indent=2))+'</pre>')
