"""Image-conditioned mechanism model, not an identified physical causal model.

O=(RGB, fixed G_obs) is common conditioning context throughout. There is no
invented P(E|Z), unconditional room prior, or scored catch-all inventory state.
The explicit conditional mechanism assumption is Z -> H inside M_O.
"""
from collections import Counter
from copy import deepcopy
import json
import math
import re

from .causal_probability import CausalGraph
from .frozen_causal_core import normalized_probabilities, text


def aggregate_rotations(rounds, state_ids):
    n = len(state_ids)
    if len(rounds) != n or len(set(state_ids)) != n:
        raise ValueError('Need one complete cyclic assignment per finite state')
    visits = {s:[] for s in state_ids}; aligned = []
    for row in rounds:
        if set(row['state_ids']) != set(state_ids) or len(row['log_scores']) != n:
            raise ValueError('Incomplete rotation')
        aligned.append(dict(zip(row['state_ids'], normalized_probabilities(row['log_scores']))))
        for position, sid in enumerate(row['state_ids']):
            visits[sid].append(position)
    if any(sorted(v) != list(range(n)) for v in visits.values()):
        raise ValueError('Every state must visit every option position exactly once')
    means = {sid:sum(row['log_scores'][row['state_ids'].index(sid)] for row in rounds)/n for sid in state_ids}
    probs = dict(zip(state_ids, normalized_probabilities([means[s] for s in state_ids])))
    tv = max((.5*sum(abs(a[s]-b[s]) for s in state_ids) for a in aligned for b in aligned),default=0.)
    return dict(probabilities=probs, mean_log_scores=means,
                permutation_max_total_variation=tv,
                pool='balanced_cyclic_log_pool_not_arithmetic_average', calibrated=False)


class VisualMechanismScorer:
    """One visual prefill per rotation, gathering all single-token option logits.

    A balanced log pool cancels additive label/position terms, but cannot
    guarantee removal of content/order interaction. Disagreement stays visible.
    No EOS/long answer scoring: the variable is explicitly a single option code.
    """
    def __init__(self, qwen, image, observed_inventory):
        self.qwen=qwen; self.image=image; self.observed=deepcopy(observed_inventory)

    def score(self, description, parents, states):
        if not 1 <= len(states) <= 20: raise ValueError('Need 1..20 finite states')
        if len(states)==1:
            return dict(probabilities={states[0]['id']:1.},rounds=[],single_state=True,
                        calibrated=False,permutation_max_total_variation=0.)
        q=self.qwen; tok=q.processor.tokenizer
        labels=[chr(65+i) for i in range(len(states))]
        ids=[tok.encode(label,add_special_tokens=False) for label in labels]
        if any(len(i)!=1 for i in ids): raise ValueError('Option codes must each be one token')
        rounds=[]
        for offset in range(len(states)):
            ordered=states[offset:]+states[:offset]
            prompt=('Select a compatible value of the specified mechanism, conditioned on this SAME local photograph '
                'and its fixed observed inventory. Predict an existing surrounding world, not improvements '
                'or minimum functional equipment. Unphotographed space is unknown. The finite alternatives '
                'are hypotheses, not an exhaustive room distribution. Use the stated parent configuration '
                'when provided; do not redefine it. Respond with one option code.\nMECHANISM: '+description+
                '\nOBSERVED: '+json.dumps(self.observed,ensure_ascii=False)+
                '\nPARENTS: '+json.dumps(parents,ensure_ascii=False)+'\nOPTIONS:\n'+
                '\n'.join(label+': '+s['description'] for label,s in zip(labels,ordered)))
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
            prefix=q.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prefix+='The selected option is:\n'
            encoded=q.processor(text=[prefix],images=[self.image],return_tensors='pt').to(q.model.device)
            encoded.pop('token_type_ids',None)
            with q.torch.inference_mode():
                result=q.model(**encoded,use_cache=False,logits_to_keep=1)
                logp=result.logits[0,-1].float().log_softmax(-1)
                values=[float(logp[i[0]].item()) for i in ids]
            rounds.append(dict(offset=offset,state_ids=[s['id'] for s in ordered],
                labels=labels,option_token_ids=[i[0] for i in ids],log_scores=values,
                allowed_option_token_mass=sum(math.exp(v) for v in values),prompt=prompt,
                assistant_prefix='The selected option is:\n'))
            del result,encoded
        return {**aggregate_rotations(rounds,[s['id'] for s in states]),'rounds':rounds,
                'source':'frozen_Qwen_visual_parent_conditioned_single_token',
                'conditioning_evidence_used_once':True,'training_performed':False}


def contexts_answer(value, case, budget=3):
    contexts=value.get('contexts') if isinstance(value,dict) else None
    anchors=value.get('floor_anchor_ids') if isinstance(value,dict) else None
    ids={o['id'] for o in case['context']}
    if not isinstance(contexts,list) or not 1<=len(contexts)<=budget:
        raise ValueError('Need nonempty contexts within proposal budget')
    if not isinstance(anchors,list) or any(not isinstance(x,str) or x not in ids for x in anchors) or len(set(anchors))!=len(anchors):
        raise ValueError('floor_anchor_ids must reference distinct observed instances')
    result=[]
    for i,c in enumerate(contexts):
        if not isinstance(c,dict): raise ValueError('Context must be an object')
        evidence=c.get('evidence_ids',[])
        if not isinstance(evidence,list) or any(not isinstance(x,str) or x not in ids for x in evidence):
            raise ValueError('Unknown context evidence ID')
        result.append(dict(id=f'z{i+1}',description=text(c.get('description'),'context.description'),
            reason=text(c.get('reason'),'context.reason'),evidence_ids=evidence))
    if len({c['description'].casefold() for c in result})!=len(result): raise ValueError('Duplicate contexts')
    return dict(contexts=result,floor_anchor_ids=anchors,
                context_scope='hypotheses_abduced_from_joint_RGB_and_inventory_not_unconditional_prior')


def worlds_answer(value, case, context_id, budget=2, capacity=256):
    worlds=value.get('worlds') if isinstance(value,dict) else None
    if not isinstance(worlds,list) or not 1<=len(worlds)<=budget: raise ValueError('Need complete worlds array within budget')
    observed={o['id'] for o in case['context']}; result=[]
    for wi,w in enumerate(worlds):
        if not isinstance(w,dict) or not isinstance(w.get('additions'),list): raise ValueError('World needs additions')
        if len(w['additions'])+len(observed)>capacity: raise ValueError('Capacity exceeded; no truncation')
        aliases={}; additions=[]
        for oi,o in enumerate(w['additions']):
            if not isinstance(o,dict): raise ValueError('Addition must be an object')
            alias=text(o.get('id'),'addition.id')
            if alias in observed or alias in aliases: raise ValueError('Duplicate ID or observed-instance reuse')
            aliases[alias]=f'{context_id}_w{wi+1}_n{oi+1}'
            if aliases[alias] in observed:raise ValueError('Program-owned new ID collides with observed instance')
        for o in w['additions']:
            support=o.get('support')
            if not isinstance(support,dict) or support.get('kind') not in ('floor','object','unspecified'):
                raise ValueError('Each object needs typed support: floor/object/unspecified')
            target=support.get('target')
            if support['kind']=='object':
                if not isinstance(target,str) or target not in observed|aliases.keys() or target==o['id']:
                    raise ValueError('Object support needs a different existing instance ID')
                target=aliases.get(target,target)
            elif target is not None: raise ValueError('Non-object support target must be null')
            fields={k:text(o.get(k),'addition.'+k) for k in ('name','object_type','reason','placement')}
            # Structured references carry identity. Check old-style prose IDs so
            # stale n7/observed_0006 references cannot evade schema validation.
            prose_aliases={a:b for a,b in aliases.items() if re.fullmatch(r'(?:n\d+|z\d+_w\d+_n\d+)',a)}
            pattern=r'\b(?:'+'|'.join(re.escape(a) for a in sorted(prose_aliases,key=len,reverse=True))+r')\b'
            for field in fields:
                if prose_aliases:fields[field]=re.sub(pattern,lambda m:prose_aliases[m.group(0)],fields[field])
                for match in re.findall(r'\b(?:observed_\d+|n\d+|z\d+_w\d+_n\d+)\b',fields[field]):
                    if match not in observed|set(aliases.values()): raise ValueError('Dangling prose instance reference: '+match)
            additions.append(dict(id=aliases[o['id']],**fields,
                support=dict(kind=support['kind'],target=target)))
        # A supporting new object's position is an actual computation parent.
        byid={o['id']:o for o in additions}; pending=set(byid); order=[]
        while pending:
            ready=sorted(i for i in pending if byid[i]['support']['target'] not in pending)
            if not ready: raise ValueError('Cyclic support dependencies')
            order+=ready;pending.difference_update(ready)
        relations=[dict(source=o['id'],target=o['support']['target'],kind='supported_by',
                        reason='Declared placement support, not evidence of object existence.')
                   for o in additions if o['support']['kind']=='object']
        result.append(dict(id=f'{context_id}_w{wi+1}',scene_type='hypothesized indoor scene',
            scene_hypothesis=text(w.get('description'),'world.description'),
            additions=additions,relations=relations,placement_dependency_order=order,
            proposal_context_id=context_id))
    return result


def inventory_key(world):
    # A coarse Adapter superclass is not an instance's semantic identity.
    # In the actual pilot wall, armchair and ottoman all arrived as "furniture".
    def noun(o):
        category=o['object_type'].casefold().replace('_',' ').strip()
        return o.get('name',category).casefold().strip() if category in (
            'furniture','object','item','other','objects','decoration','appliance','lighting') else category
    return tuple(sorted(Counter(noun(o) for o in world['additions']).items()))


def conditional_model(contexts, worlds, scorer, save):
    zstates=[dict(id=c['id'],description=c['description']) for c in contexts]
    hstates=[dict(id=w['id'],description=json.dumps(dict(inventory=dict(inventory_key(w)),
        instances=[{k:o[k] for k in ('name','object_type','placement','support') if k in o}
                   for o in w['additions']],hypothesis=w['scene_hypothesis']),ensure_ascii=False)) for w in worlds]
    z=scorer.score('Configuration hypothesis compatible with the local observations.',{},zstates)
    save('context_given_observations',z);rows=[]; signals={}
    direct=scorer.score('Existing off-camera inventory compatible with the observations without an explicit configuration assumption.',{},hstates)
    save('direct_inventory_given_observations',direct)
    for c in contexts:
        values=scorer.score('Existing off-camera inventory under this configuration, conditioned on the supplied observations.',
                            {'configuration':c['description']},hstates)
        save('inventory_given_'+c['id'],values)
        rows.append(dict(when=[c['id']],probabilities=values['probabilities']))
        signals[c['id']]=values.get('permutation_max_total_variation')
    spec=dict(nodes=[dict(id='Z',states=[s['id'] for s in zstates],parents=[],
        cpt=[dict(when=[],probabilities=z['probabilities'])]),
        dict(id='H',states=[s['id'] for s in hstates],parents=['Z'],cpt=rows)],
        parameter_source='frozen_visual_conditional_mechanisms_uncalibrated',
        model_scope='conditional_M_O_with_O_fixed_not_identified_physical_SCM',
        observation_likelihood_defined=False,coverage_probability=None,
        finite_domain_is_exhaustive=False,identified_from_data=False)
    model=CausalGraph(spec)
    marginal=model.query('H')
    regimes={s['id']:model.query('H',interventions={'Z':s['id']}) for s in zstates}
    values=[v['probabilities'] for v in regimes.values()]
    sensitivity=max((.5*sum(abs(a[s]-b[s]) for s in a) for a in values for b in values),default=0.)
    comparison=.5*sum(abs(marginal['probabilities'][s]-direct['probabilities'][s]) for s in marginal['probabilities'])
    return spec,dict(context_given_observations=z['probabilities'],world_mixture=marginal,
        direct_inventory_baseline=direct['probabilities'],
        mechanism_signal=dict(configuration_sensitivity_tv=sensitivity,mixture_vs_direct_tv=comparison,
            note='Matched-catalog ranking diagnostics, not demonstrated causal benefit.'),
        context_regimes=regimes,regime_kind='configuration_mechanism_sensitivity_at_fixed_O',
        actual_world_causal_effect_identified=False,causal_gain_validated=False,
        context_permutation_tv=z.get('permutation_max_total_variation'),
        inventory_permutation_tv=signals,coverage_probability=None,
        probability_scope='relative_weights_inside_proposed_catalog_only',
        single_configuration=len(contexts)==1)
