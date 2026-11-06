"""Parent-only frozen LM mechanisms; images are never used for these scores."""
from collections import Counter
from copy import deepcopy
import json
import math

from ..common import write
from .conditional_world_core import aggregate_rotations
from .frozen_causal_core import text


def noun(value):
    value=text(value,'object_type').casefold().replace('_',' ')
    return ' '.join(value.split())


def domain_answer(value,budget=8):
    if not isinstance(value,dict):raise ValueError('Domain must be an object')
    out={}
    for field,prefix in (('backgrounds','b'),('configurations','k')):
        source=value.get(field)
        if not isinstance(source,list) or not 2<=len(source)<=3:raise ValueError('Need 2..3 distinct '+field)
        states=[dict(id=prefix+str(i+1),description=text(s,'state description')) for i,s in enumerate(source)]
        if len({s['description'].casefold() for s in states})!=len(states):raise ValueError('Duplicate state')
        out[field]=states
    types=value.get('object_types')
    if not isinstance(types,list) or not 1<=len(types)<=budget:raise ValueError('Object catalog exceeds computational budget')
    types=[noun(t) for t in types]
    if len(set(types))!=len(types) or any(t in ('wall','floor','ceiling','object','furniture') for t in types):
        raise ValueError('Need distinct specific object nouns, no room envelope/superclasses')
    out['object_types']=types
    return out


def registration_answer(value,case,domain):
    if not isinstance(value,dict) or not isinstance(value.get('instances'),list):raise ValueError('Need instances array')
    ids={o['id'] for o in case['context']};rows=value['instances']
    if any(not isinstance(o,dict) or not isinstance(o.get('id'),str) for o in rows):raise ValueError('Instance needs a string ID')
    if len(rows)!=len(ids) or {o.get('id') for o in rows}!=ids:raise ValueError('Register each observed instance exactly once')
    normalized={o['id']:noun(o.get('object_type')) for o in rows}
    if any(t in ('wall','floor','ceiling','object','furniture') for t in normalized.values()):raise ValueError('Need specific observed object noun')
    anchors=value.get('floor_anchor_ids')
    if (not isinstance(anchors,list) or any(not isinstance(i,str) for i in anchors) or len(set(anchors))!=len(anchors)
        or any(i not in ids for i in anchors)):
        raise ValueError('Invalid floor anchor IDs')
    types=list(domain['object_types'])
    for oid in [o['id'] for o in case['context']]:
        if normalized[oid] not in types:types.append(normalized[oid])
    if len(types)>24:raise ValueError('Finite noun domain exceeds computational budget; no truncation')
    return dict(instances=[dict(id=o['id'],object_type=normalized[o['id']]) for o in case['context']],
        observed_counts=dict(Counter(normalized.values())),object_types=types,floor_anchor_ids=anchors,
        observer_state_changed=False,finite_vocabulary_instantiated_for_case=True,
        vocabulary_exhaustive=False)


class TextMechanismScorer:
    """A finite single-code mechanism, balanced across all cyclic positions.

    This defines a normalized model mechanism, not an empirical frequency.
    The answer variable is one code token; no full-sequence likelihood claim.
    """
    def __init__(self,qwen):self.qwen=qwen

    def score(self,description,parents,states):
        if not 1<=len(states)<=20:raise ValueError('Need 1..20 finite mechanism states')
        if len(states)==1:return dict(probabilities={states[0]['id']:1.},rounds=[],calibrated=False)
        q=self.qwen;tok=q.processor.tokenizer;labels=[chr(65+i) for i in range(len(states))]
        token_ids=[tok.encode(s,add_special_tokens=False) for s in labels]
        if any(len(i)!=1 for i in token_ids):raise ValueError('Each state code must be one token')
        rounds=[]
        for offset in range(len(states)):
            ordered=states[offset:]+states[:offset]
            prompt=('Parameterize the specified generative mechanism of a hypothetical existing indoor world. '
                'Use ONLY the declared parent states below and general knowledge. No photograph, detector '
                'summary or observed counts are available. Predict existing object counts, not required '
                'equipment or improvements. Optional objects can be absent; repeated instances are possible. '
                'This finite domain is not exhaustive or calibrated. Select one state code.\nMECHANISM: '+description+
                '\nPARENTS: '+json.dumps(parents,ensure_ascii=False)+'\nSTATES:\n'+
                '\n'.join(a+': '+s['description'] for a,s in zip(labels,ordered)))
            messages=[{'role':'user','content':[{'type':'text','text':prompt}]}]
            prefix=q.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+'The selected option is:\n'
            encoded=q.processor(text=[prefix],return_tensors='pt').to(q.model.device)
            encoded.pop('token_type_ids',None)
            with q.torch.inference_mode():
                result=q.model(**encoded,use_cache=False,logits_to_keep=1)
                logp=result.logits[0,-1].float().log_softmax(-1)
                scores=[float(logp[i[0]].item()) for i in token_ids]
            rounds.append(dict(state_ids=[s['id'] for s in ordered],log_scores=scores,prompt=prompt,
                allowed_code_mass=sum(math.exp(s) for s in scores)))
            del encoded,result
        return dict(**aggregate_rotations(rounds,[s['id'] for s in states]),rounds=rounds,
            parents=deepcopy(parents),description=description,states=deepcopy(states),image_used=False,
            source='frozen_parent_only_text_single_code_balanced_log_pool')


def count_states(object_type,max_count,contract='sentence_v1'):
    if contract not in ('sentence_v1','json_integer_v1'):raise ValueError('Unknown count event contract')
    return [dict(id=str(n),description=(str(n) if contract=='json_integer_v1' else
        f'Exactly {n} distinct {object_type} instances in the entire room, including all camera directions.'))
        for n in range(max_count+1)]


def parameterize(domain,types,max_count,scorer,folder):
    folder.mkdir();bs=domain['backgrounds'];ks=domain['configurations'];diagnostics={}
    contract=getattr(scorer,'count_event_contract','sentence_v1')
    def score(name,description,parents,states):
        result=scorer.score(description,parents,states)
        probabilities=result.get('probabilities',{})
        ids=[s['id'] for s in states]
        if (set(probabilities)!=set(ids) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in probabilities.values())
            or not math.isclose(sum(probabilities.values()),1.,abs_tol=1e-8)):
            raise ValueError('Invalid frozen mechanism: '+name)
        write(folder/(name+'.json'),dict(**result,declared_parents=parents,state_catalog=states))
        diagnostics[name]=result.get('permutation_max_total_variation')
        print(json.dumps(dict(stage='generative_mechanism',mechanism=name,permutation_tv=diagnostics[name],
            scoring_source=result.get('source','synthetic_test'))),flush=True)
        return [float(probabilities[i]) for i in ids]
    p=dict(B=score('B','Joint intended-use, room-configuration and household-preference background.',{},bs),K={},N={})
    for b in bs:
        p['K'][b['id']]=score('K_'+b['id'],'Furnishing configuration pattern.',{'B':b['description']},ks)
        p['N'][b['id']]={}
        for k in ks:
            p['N'][b['id']][k['id']]={}
            for ti,t in enumerate(types):
                states=count_states(t,max_count,contract)
                field='Whole-room number of instances of object type '+t+'.'
                if contract=='json_integer_v1':
                    field+=f' Include all camera directions. Answer with one JSON integer between 0 and {max_count}, inclusive.'
                p['N'][b['id']][k['id']][t]=score(f'N_{b["id"]}_{k["id"]}_t{ti:02d}',
                    field,
                    {'B':b['description'],'K':k['description']},states)
    return dict(backgrounds=bs,configurations=ks,object_types=types,max_count=max_count,mechanisms=p,
        count_event_contract=contract,
        graph='B -> K; B,K -> N_t; N_t,F,D -> O_t',
        nuisance_priors=dict(view_fraction='Uniform(0,1)',recording_rate='Uniform(0,1)',independent=True),
        count_independence='N_t mutually independent conditional on B,K; shared parents induce marginal dependence',
        parameter_source='frozen_parent_only_LM_mechanisms_uncalibrated',
        finite_domain_exhaustive=False,tail_mass_estimated=False,
        domain_scope='case_instantiated_noun_vocabulary_no_observed_count_values_in_prior_mechanisms',
        image_used_for_generative_parameters=False,joint_metric_posterior_defined=False),diagnostics
