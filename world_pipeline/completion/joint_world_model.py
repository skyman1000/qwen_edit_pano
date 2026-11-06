"""Semantic H -> complete all-scale delta; v13 identity/support validators.

No metric gate before semantic generation; no role-wise program splitting.
"""
from collections import Counter
from copy import deepcopy
import json
import math
import numpy as np

from . import assembly_model as typed, assembly_delta as inventory
from .world_scm import validate,forward

VARIANT='joint_approximate_world_valid_support_v4'
LIMIT=24  # Existing v13 whole-program transport bound, never a target.

def finite(v):return type(v) in (int,float) and math.isfinite(v)
def text(v):
    if not isinstance(v,str) or not v.strip():raise ValueError('Nonempty descriptive string required')
    return v.strip()

PLACEHOLDERS={'whole surrounding-world hypothesis','coexisting activities','uncertain hidden-world assumptions',
    'fill with a concrete surrounding-world hypothesis','fill with the concrete activity or storage area',
    'evidence and uncertainty','evidence-compatible reason for this separate unseen instance','qualitative surrounding position'}

def concrete(value):
    """Reject exact prompt stencils, never judge whether a category is required."""
    if isinstance(value,dict):
        for v in value.values():concrete(v)
    elif isinstance(value,list):
        for v in value:concrete(v)
    elif isinstance(value,str) and value.strip().casefold() in PLACEHOLDERS:
        raise ValueError('Replace copied schema placeholder with a concrete hypothesis: '+value)

def hypotheses_prompt(case):
    return ('Infer 1..2 concrete hypotheses about the WHOLE surrounding 360-degree indoor world from this partial RGB. '
        'The camera observes only one sector. Distinguish photographed facts from assumptions about the unseen space. '
        'Describe how its activities, ownership/use and spatial organization may continue outside the view, not just the pictured objects. '
        'Several worlds can explain the same photograph. Lack of visual evidence outside the view is not evidence of an empty surrounding world. '
        'Do not design a new room or require extra objects to meet a checklist. No category, count or direction quota. '
        'Return JSON {"hypotheses":[{"description":"","functions":[],"assumptions":[]}]}. '
        'Fill description and list entries with concrete content; do not copy schema instructions as content. '
        'No coordinates, dimensions or object inventory in this stage.\nFIXED_OBSERVED: '+json.dumps(case['context'],ensure_ascii=False))

def hypotheses_answer(value,case=None,require_concrete=False):
    if require_concrete:concrete(value)
    rows=value.get('hypotheses')
    if not isinstance(rows,list) or not 1<=len(rows)<=3:raise ValueError('hypotheses needs 1..3 semantic records')
    result=[]
    for i,r in enumerate(rows):
        if not isinstance(r,dict):raise ValueError('Each hypothesis is an object')
        h=dict(id='h'+str(i),description=text(r.get('description')))
        for key in ('functions','assumptions'):
            if not isinstance(r.get(key),list):raise ValueError(key+' must be an array')
            h[key]=[text(x) for x in r[key]]
        result.append(h)
    return result

def programs_prompt(case,h):
    obj=dict(key='',category='',variant='',reason='',placement='',attachment=dict(kind='',host=None),affordances=[])
    group=dict(function='',reason='',observed_refs=[],new_objects=[obj])
    template=dict(alternatives=[dict(description='',groups=[group],relations=[])])
    return ('Propose 1..2 concrete COMPLETE hidden-world deltas under the supplied whole-world hypothesis. '
        'This is abductive completion of an existing surrounding world from a partial view. '
        'FIXED_OBSERVED already exists and must be preserved. new_objects contains ONLY distinct unseen instances, '
        'never copied or renamed visible objects, nor clearly visible detector omissions moved out of view. '
        'Include coexisting furniture, possessions and supported contents as appropriate to this particular hypothesis. '
        'All new hosts and their contents belong to the SAME alternative; do not split the inventory by scale or function. '
        'Outside-view objects do not need to be visible or logically necessary for the photographed activity. '
        'Infer plausible assumptions; no minimum inventory, mandatory category or sector quota. '
        'An empty delta is allowed if you explicitly hypothesize genuinely empty unseen space; not merely because it is unobserved. '
        'Capacity per alternative: up to 4 groups and 24 new objects, never a target. '
        'Return this JSON structure, replacing every empty string with concrete content, without copying instructions into data: '+json.dumps(template)+'. '
        'groups/new_objects may be empty for your own explicit absence hypothesis. Keys must be unique short identifiers. '
        'observed_refs uses exact observed IDs. category is the concrete noun; variant includes that noun and its concrete type. '
        'reason explains this distinct unseen instance and the assumptions supporting it; placement is qualitative in original-camera axes. '
        'attachment.kind is floor/top/seat/shelf/hang/wall/ceiling/free. '
        'For floor/wall/ceiling/free, host MUST be null: nearby objects are spatial references, not the mounting surface. '
        'For top/seat/shelf/hang, host is one exact new key in this alternative or observed ID. '
        'Do not substitute an observed cabinet or sink ID for a wall. Floor-standing furniture near a wall uses floor. '
        'affordances lists surfaces it offers to others: subset of top,seat,shelf,hanging. '
        'Hosted objects inherit their host placement group. Use OBSERVED_SUPPORT_DOMAINS for observed-host compatibility; '
        'an empty candidate list forbids that wholly outside-frustum contact hypothesis. '
        +typed.RELATION_CONTRACT+'in_front_of means greater camera Z; behind means smaller camera Z. '
        'No metric values yet.\nHYPOTHESIS: '+json.dumps(h,ensure_ascii=False)+
        '\nOBSERVED_SUPPORT_DOMAINS: '+json.dumps(case.get('observed_support_domains',{}),ensure_ascii=False)+
        '\nFIXED_OBSERVED: '+json.dumps(case['context'],ensure_ascii=False))

def failed_program(h,error):
    return dict(description='Proposal unresolved; this is not an empty world.',hypothesis=deepcopy(h),
        groups=[],relations=[],proposal_failed=True,proposal_error=error)

def programs_answer(value,case,h,require_concrete=False):
    rows=value.get('alternatives')
    if not isinstance(rows,list) or not 1<=len(rows)<=2:raise ValueError('Need 1..2 complete alternatives using groups/new_objects')
    valid=[];failed=[]
    for i,alt in enumerate(rows):
        errors=[];p=None
        if require_concrete:
            try:concrete(alt)
            except ValueError as exc:errors.append(str(exc))
        try:
            p=inventory.furniture_answer(dict(alternatives=[alt]),case,h)[0]
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(type(exc).__name__+': '+str(exc))
        if errors:failed.append(dict(alternative=i,error='; '.join(errors)))
        else:p['source_alternative']=i;valid.append(p)
    if not valid:raise ValueError('No legal complete delta: '+json.dumps(failed,ensure_ascii=False))
    # Retain legal siblings and explicit failure states, rather than erase one
    # sibling's valid inventory when a different alternative is malformed.
    return dict(valid=valid,failed=failed)

def has_valid_program(domain):
    return any(not p.get('proposal_failed',False) for p in domain['programs'].values())

def predictive(votes,states):
    if not votes:raise ValueError('No valid empirical choices; no default probability row')
    counts=Counter(votes)
    return dict(counts=dict(counts),valid_draws=len(votes),states=list(states),
        probabilities=[(counts[s]+1/len(states))/(len(votes)+1) for s in states],concentration=1.)

def reference_program_law(choices):
    """Explicit finite generative reference law, not a visibility/necessity vote.

    This changes the declared model. It is not posterior recalibration or
    conditioning on a successful/nonempty draw; failure and genuine empty
    proposal states stay in the same law.
    """
    states=list(choices)
    if not states:raise ValueError('No program states')
    return dict(states=states,probabilities=[1./len(states)]*len(states),counts={},valid_draws=0,
        law='uniform_reference_over_complete_LM_proposals_including_explicit_absence_and_unresolved_failure',
        estimated_from_RGB=False,posterior_claimed=False)


def compile_model(hypotheses,programs,tables):
    m=inventory.build_model(hypotheses,programs)
    z=next(n for n in m['nodes'] if n['id']=='Z');z['cpt']={'root':tables['recognition']['probabilities']}
    f=next(n for n in m['nodes'] if n['id']=='F')
    for h in hypotheses:
        row=tables['programs'][h['id']];prob=dict(zip(row['states'],row['probabilities']))
        f['cpt'][h['id']]=[prob.get(s,0.) for s in f['states']]
    m.update(model_variant=VARIANT,empirical_tables=deepcopy(tables),maximum_hidden_instances=max((sum(len(g['objects']) for g in p['groups']) for p in programs.values()),default=0),
        generation_factorization='q(H|RGB,O,D) reference(F|H,D); F->zone_instances,E; S_i=E_i AND S_host; (H,O,F,hosts)->metric_pose',
        distribution_scope='finite_proposed_world_reference_model_not_an_identified_posterior',
        sampling_policy='IID_duplicates_empty_and_failed_hypothesis_draws_retained',
        no_role_factorization=True,probabilities_calibrated=False)
    return validate(m)


def available_proposal_model(raw):
    """Condition the declared finite law on an executable semantic proposal.

    Generation failures are coverage diagnostics, not physical empty worlds.
    Preserve their original reference mass and original SCM for audit; never
    call the conditioned law the posterior of the actual unseen room.
    """
    model=deepcopy(raw);z=next(n for n in model['nodes'] if n['id']=='Z');f=next(n for n in model['nodes'] if n['id']=='F')
    valid={fid for fid,p in model['furniture_states'].items() if not p.get('proposal_failed',False)}
    original=list(z['cpt']['root']);weights=[];rows=[]
    for h,p_h in zip(z['states'],original):
        row=f['cpt'][h];mass=sum(p for fid,p in zip(f['states'],row) if fid in valid)
        weights.append(p_h*mass)
        rows.append(dict(hypothesis=h,original_H_weight=p_h,available_F_mass=mass,
            unresolved_F_states=[fid for fid,p in zip(f['states'],row) if p>0 and fid not in valid]))
        if mass>0:f['cpt'][h]=[p/mass if fid in valid else 0. for fid,p in zip(f['states'],row)]
    retained=sum(weights)
    if retained<=0:raise ValueError('All semantic proposals unresolved; no available world sampling support')
    z['cpt']['root']=[w/retained for w in weights]
    coverage=dict(condition='semantic_proposal_available_not_nonempty_or_layout_success',
        retained_reference_mass=retained,excluded_reference_mass=max(0.,1.-retained),hypotheses=rows,
        conditioned_H_weights=dict(zip(z['states'],z['cpt']['root'])),real_world_posterior_claimed=False)
    model.update(proposal_coverage=coverage,sampling_policy='IID_on_available_semantic_proposals_duplicates_and_genuine_empty_retained_no_layout_resampling')
    return validate(model),coverage


def sample_worlds(model,seed,count):
    rng=np.random.default_rng(seed);worlds=[]
    for i in range(count):
        noise={n['id']:float(rng.random()) for n in model['nodes']};w=inventory.active_world(model,forward(model,noise))
        program=model['furniture_states'][w['structural_state']['F']]
        w.update(noise=noise,seed=seed,trial=i,proposal_failed=program.get('proposal_failed',False),
            zone_decisions=deepcopy(program.get('zone_decisions',[])))
        worlds.append(w)
    return worlds
