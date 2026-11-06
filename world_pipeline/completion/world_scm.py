"""Finite generative semantic SCM: exact conditioning, interventions and noise abduction.

Mechanisms are elicited subjective tables, not API token probabilities. Domain
selection may use the photograph; all inference is conditional on that domain.
"""
from copy import deepcopy
import itertools
import math

import numpy as np

PROTOCOL='generative_semantic_world_scm_v1'


def key(values):return '|'.join(values) if values else 'root'


def distribution(values,n):
    if (not isinstance(values,list) or len(values)!=n or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values)):
        raise ValueError('Conditional probabilities must be a finite nonnegative list of the exact state count')
    total=sum(values)
    if not math.isclose(total,1.,abs_tol=1e-5):raise ValueError('Conditional probabilities must sum to 1')
    return [float(v/total) for v in values]


def validate(model):
    seen={}
    for node in model['nodes']:
        name=node['id'];states=node['states'];parents=node['parents']
        if name in seen or not states or len(set(states))!=len(states) or any(not isinstance(s,str) or '|' in s for s in states):
            raise ValueError('SCM IDs/states must be unique strings without table separators')
        if len(set(parents))!=len(parents) or any(p not in seen for p in parents):raise ValueError('SCM parents must precede child')
        expected={key(v) for v in itertools.product(*(seen[p]['states'] for p in parents))}
        if set(node['cpt'])!=expected:raise ValueError('Missing or extra parent assignments for '+name)
        for values in node['cpt'].values():distribution(values,len(states))
        seen[name]=node
    return model


def enumerate_model(model,evidence=None,do=None):
    validate(model);evidence=evidence or {};do=do or {};nodes=model['nodes'];byid={n['id']:n for n in nodes}
    for name,value in {**evidence,**do}.items():
        if name not in byid or value not in byid[name]['states']:raise ValueError('Unknown evidence/intervention state')
    if any(name in do and do[name]!=v for name,v in evidence.items()):raise ValueError('Evidence contradicts intervention')
    rows=[];mass=0.
    for values in itertools.product(*(n['states'] for n in nodes)):
        state=dict(zip(byid,values))
        if any(state[k]!=v for k,v in evidence.items()):continue
        probability=1.
        for node,value in zip(nodes,values):
            name=node['id']
            if name in do:probability*=float(value==do[name])
            else:probability*=node['cpt'][key([state[p] for p in node['parents']])][node['states'].index(value)]
        if probability>0:rows.append(dict(state=state,joint_mass=probability));mass+=probability
    if mass<=0:raise ValueError('Evidence has zero mass in the specified semantic model')
    for row in rows:row['probability']=row['joint_mass']/mass
    marginals={n['id']:{s:sum(r['probability'] for r in rows if r['state'][n['id']]==s) for s in n['states']} for n in nodes}
    return dict(rows=rows,evidence_probability=mass,marginals=marginals,evidence=evidence,intervention=do,
                inference_scope='exact_within_elicited_finite_semantic_model_and_selected_domain')


def forward(model,noise,do=None):
    validate(model);do=do or {};byid={n['id']:n for n in model['nodes']}
    if any(k not in byid or v not in byid[k]['states'] for k,v in do.items()):raise ValueError('Invalid intervention')
    state={}
    for node in model['nodes']:
        name=node['id'];u=noise[name]
        if not 0<=u<1:raise ValueError('SCM noise must lie in [0,1)')
        ps=node['cpt'][key([state[p] for p in node['parents']])]
        idx=min(int(np.searchsorted(np.cumsum(ps),u,side='right')),len(ps)-1)
        state[name]=do[name] if name in do else node['states'][idx]
    return state


def abduct(model,evidence,seed):
    """Draw state ~ p(V|E), then U_i uniformly inside that state's CDF interval."""
    posterior=enumerate_model(model,evidence);rng=np.random.default_rng(seed)
    row=posterior['rows'][rng.choice(len(posterior['rows']),p=[r['probability'] for r in posterior['rows']])]
    state=row['state'];noise={}
    for node in model['nodes']:
        ps=node['cpt'][key([state[p] for p in node['parents']])];idx=node['states'].index(state[node['id']])
        lower=sum(ps[:idx]);noise[node['id']]=min(float(lower+ps[idx]*rng.random()),float(np.nextafter(1.,0.)))
    if forward(model,noise)!=state:raise RuntimeError('Noise abduction did not reproduce sampled factual state')
    return dict(factual_state=state,noise=noise,seed=seed,
        scope='model_relative_abduction; categorical_inverse_CDF_coupling_assumed; not_identified_photo_counterfactual')


def parent_rows(nodes,parents):
    byid={n['id']:n for n in nodes}
    return [dict(zip(parents,values)) for values in itertools.product(*(byid[p]['states'] for p in parents))]


def mechanism_strengths(model):
    """Maximum CPT total-variation change per declared parent, other parents fixed.

    A model diagnostic over its declared rows, not an estimated physical effect.
    Zero-prior parent assignments can contribute to this maximum.
    """
    byid={n['id']:n for n in model['nodes']};result=[]
    for node in model['nodes']:
        for parent in node['parents']:
            other=[p for p in node['parents'] if p!=parent];maximum=0.
            for row in parent_rows(model['nodes'],other):
                probabilities=[node['cpt'][key([{**row,parent:state}[p] for p in node['parents']])]
                               for state in byid[parent]['states']]
                for a,b in itertools.combinations(probabilities,2):maximum=max(maximum,sum(abs(x-y) for x,y in zip(a,b))/2)
            result.append(dict(parent=parent,child=node['id'],max_CPT_total_variation=maximum))
    return result


def independent_groups(model):
    """Delete group-to-group dependencies, preserving p(g|C,S) exactly.

    This compares joint structure at fixed conditional marginals and requires no
    additional API calls. It is an ablation, not a causal effect estimate.
    """
    result=deepcopy(model)
    for c in result['nodes'][0]['states']:
        s_node=next(n for n in result['nodes'] if n['id']=='S')
        for s in s_node['states']:
            conditional=enumerate_model(model,do={'C':c,'S':s})
            for node in result['nodes']:
                if node['id'].startswith('g'):
                    node['cpt'][key([c,s])]=[conditional['marginals'][node['id']][v] for v in node['states']]
    for node in result['nodes']:
        if node['id'].startswith('g'):
            node['parents']=['C','S']
            node['cpt']={k:v for k,v in node['cpt'].items() if k.count('|')==1}
    result['ablation']='group_edges_removed_conditional_marginals_preserved'
    return validate(result)
