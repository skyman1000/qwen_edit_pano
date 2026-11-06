"""Executable finite semantic SCM and censored-count observation inference.

B -> K; (B,K) -> N_t. Unknown view fraction F and recording rate D have
declared independent Uniform(0,1) priors. For an observed category,
O_t | N_t,F,D ~ Binomial(N_t,F*D). Categories absent from the supplied
summary are censored, not observations of zero. Given an unrecorded instance,
P(outside | unrecorded,F,D)=(1-F)/(1-F*D).

Gauss-Legendre quadrature integrates the polynomial finite-count likelihood.
The auxiliary quadrature node is not a measured visibility/detection rate.
This is NOT a metric/camera-calibrated likelihood or an identified real SCM.
"""
from copy import deepcopy
import math
import numpy as np


def distribution(values,length):
    a=np.asarray(values,dtype=float)
    if (a.shape!=(length,) or not np.isfinite(a).all() or (a<0).any() or
        not math.isclose(float(a.sum()),1.,abs_tol=1e-8)):
        raise ValueError('Mechanism must be finite, normalized and cover every state')
    return a


def validate(spec,observed):
    if not isinstance(spec,dict):raise ValueError('Need semantic SCM specification')
    bs=spec['backgrounds'];ks=spec['configurations'];types=spec['object_types'];m=spec['max_count']
    for states in (bs,ks):
        if not isinstance(states,list) or not states or len({s['id'] for s in states})!=len(states):
            raise ValueError('Need distinct nonempty finite state catalogs')
    if (not isinstance(types,list) or not types or len(types)>24 or len(set(types))!=len(types)
        or any(not isinstance(t,str) or not t for t in types) or type(m) is not int or not 1<=m<=16):
        raise ValueError('Invalid finite object/count domain')
    if not isinstance(observed,dict) or any(t not in types or type(k) is not int or not 0<=k<=m for t,k in observed.items()):
        raise ValueError('Observed category/count outside declared domain; increase domain, never clip counts')
    p=spec['mechanisms'];bids=[b['id'] for b in bs];kids=[k['id'] for k in ks]
    if set(p['K'])!=set(bids) or set(p['N'])!=set(bids):raise ValueError('Incomplete background mechanisms')
    distribution(p['B'],len(bs))
    for b in bids:
        distribution(p['K'][b],len(ks))
        if set(p['N'][b])!=set(kids):raise ValueError('Incomplete configuration mechanisms')
        for k in kids:
            if set(p['N'][b][k])!=set(types):raise ValueError('Incomplete object mechanisms')
            for t in types:distribution(p['N'][b][k][t],m+1)
    return spec


def logsumexp(values,axis=None):
    a=np.asarray(values,dtype=float);maximum=np.max(a,axis=axis,keepdims=True)
    safe=np.where(np.isfinite(maximum),maximum,0.)
    with np.errstate(divide='ignore',invalid='ignore'):
        result=safe+np.log(np.exp(a-safe).sum(axis=axis,keepdims=True))
    if axis is None:return float(result.reshape(-1)[0])
    return np.squeeze(result,axis=axis)


def log_binomial(k,m,q):
    result=np.full((len(q),m+1),-np.inf)
    for n in range(k,m+1):
        result[:,n]=(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                      +k*np.log(q)+(n-k)*np.log1p(-q))
    return result


def infer(spec,observed,interventions=None):
    """Exact marginal count inference within the declared finite polynomial model.

    do(B)/do(K) replace the corresponding prior/mechanism. Ordinary observation
    factors stay in the product. No RGB or GT is supplied to this function.
    """
    validate(spec,observed);interventions=interventions or {}
    if set(interventions)-{'B','K'}:raise ValueError('Supported interventions are B and K')
    bids=[b['id'] for b in spec['backgrounds']];kids=[k['id'] for k in spec['configurations']]
    for n,ids in (('B',bids),('K',kids)):
        if n in interventions and interventions[n] not in ids:raise ValueError('Unknown intervention state')
    m=spec['max_count'];types=spec['object_types'];p=spec['mechanisms']
    # Covers joint total/outside-count polynomials, including censored categories.
    order=(m*len(types)+2)//2+1
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    f,d=np.meshgrid(x,x,indexing='ij');wf,wd=np.meshgrid(w,w,indexing='ij')
    f=f.ravel();d=d.ravel();quadrature=(wf*wd).ravel();q=f*d
    logs={t:log_binomial(k,m,q) for t,k in observed.items()}
    log_joint=[];regimes=[]
    with np.errstate(divide='ignore'):
        for bi,b in enumerate(bids):
            pb=float(p['B'][bi]) if 'B' not in interventions else float(b==interventions['B'])
            for ki,k in enumerate(kids):
                pk=float(p['K'][b][ki]) if 'K' not in interventions else float(k==interventions['K'])
                row=np.full(len(f),math.log(pb*pk) if pb*pk>0 else -np.inf)
                row+=np.log(quadrature)
                for t in observed:
                    row+=logsumexp(logs[t]+np.log(np.asarray(p['N'][b][k][t]))[None,:],axis=1)
                log_joint.append(row);regimes.append((b,k))
    log_joint=np.asarray(log_joint);evidence_logp=logsumexp(log_joint)
    if not math.isfinite(evidence_logp):raise ValueError('Observation has zero probability in finite model; no fallback world invented')
    joint=np.exp(log_joint-evidence_logp);bk=joint.sum(axis=1)
    pb={b:float(sum(bk[i] for i,r in enumerate(regimes) if r[0]==b)) for b in bids}
    pk={k:float(sum(bk[i] for i,r in enumerate(regimes) if r[1]==k)) for k in kids}
    return dict(spec=deepcopy(spec),observed=dict(observed),interventions=dict(interventions),
        joint=joint,regimes=regimes,f=f,d=d,q=q,logs=logs,
        evidence_log_probability=evidence_logp,background_posterior=pb,configuration_posterior=pk,
        quadrature_order=order)


def count_distribution(result,t):
    spec=result['spec'];m=spec['max_count'];answer=np.zeros(m+1)
    for ri,(b,k) in enumerate(result['regimes']):
        prior=np.asarray(spec['mechanisms']['N'][b][k][t])
        if t in result['observed']:
            with np.errstate(divide='ignore'):logp=result['logs'][t]+np.log(prior)[None,:]
            normal=logsumexp(logp,axis=1)
            valid=np.isfinite(normal)
            conditional=np.zeros_like(logp);conditional[valid]=np.exp(logp[valid]-normal[valid,None])
            answer+=(result['joint'][ri,:,None]*conditional).sum(axis=0)
        else:answer+=result['joint'][ri].sum()*prior
    return answer.tolist()


def draw(result,seed):
    rng=np.random.default_rng(seed);spec=result['spec'];m=spec['max_count']
    flat=result['joint'].ravel();index=int(rng.choice(len(flat),p=flat/flat.sum()))
    ri,qi=np.unravel_index(index,result['joint'].shape);b,k=result['regimes'][ri]
    f=float(result['f'][qi]);d=float(result['d'][qi]);q=f*d;rows=[]
    for t in spec['object_types']:
        prior=np.asarray(spec['mechanisms']['N'][b][k][t]);recorded=result['observed'].get(t)
        if recorded is not None:
            with np.errstate(divide='ignore'):lp=result['logs'][t][qi]+np.log(prior)
            probabilities=np.exp(lp-logsumexp(lp))
            n=int(rng.choice(m+1,p=probabilities/probabilities.sum()))
            outside=int(rng.binomial(n-recorded,(1-f)/(1-q)))
            inside_unrecorded=n-recorded-outside
        else:
            n=int(rng.choice(m+1,p=prior));outside=int(rng.binomial(n,1-f))
            inside_unrecorded=n-outside
        rows.append(dict(object_type=t,total=n,observed=recorded,
            outside=outside,inside_not_exported=inside_unrecorded,observation_censored=recorded is None))
    return dict(seed=seed,B=b,K=k,counts=rows,
        integration_auxiliary=dict(index=int(qi),view_fraction=f,recording_rate=d,
            interpretation='quadrature_auxiliary_not_measured_parameters_or_continuous_posterior_draw'),
        new_instances=sum(r['outside'] for r in rows),joint_metric_posterior_defined=False)


def completion_diagnostics(result):
    """Marginal moments and empty-world probability; does not resample a world.

    Conditional on B,K,F,D the type counts/partitions are independent. Integrate
    their conditional moments and the product of their zero-outside probabilities
    against the *joint* posterior, rather than multiplying marginal probabilities.
    Censored types use F**N for P(no outside); no zero detector count is invented.
    """
    spec=result['spec'];ns=np.arange(spec['max_count']+1);joint=result['joint']
    f,d=result['f'],result['d'];outside_given_unrecorded=(1-f)/(1-result['q'])
    zero_all=np.ones_like(joint);rows={}
    for t in spec['object_types']:
        recorded=result['observed'].get(t);stats=dict(expected_total=0.,expected_outside=0.,
            expected_inside_not_exported=0.,probability_zero_outside=0.)
        remaining=ns-(recorded or 0)
        for ri,(b,k) in enumerate(result['regimes']):
            prior=np.asarray(spec['mechanisms']['N'][b][k][t])
            if recorded is not None:
                with np.errstate(divide='ignore'):lp=result['logs'][t]+np.log(prior)[None,:]
                normal=logsumexp(lp,axis=1);valid=np.isfinite(normal)
                probs=np.zeros_like(lp);probs[valid]=np.exp(lp[valid]-normal[valid,None])
                remaining_mean=probs@remaining
                outside_mean=remaining_mean*outside_given_unrecorded
                # Impossible N < O have zero mass; avoid negative exponents.
                zero=(probs*(1-outside_given_unrecorded[:,None])**np.maximum(remaining,0)[None,:]).sum(axis=1)
            else:
                probs=np.broadcast_to(prior,(len(f),len(ns)))
                remaining_mean=probs@ns;outside_mean=remaining_mean*(1-f)
                zero=(probs*f[:,None]**ns[None,:]).sum(axis=1)
            weights=joint[ri];stats['expected_total']+=float(weights@(probs@ns))
            stats['expected_outside']+=float(weights@outside_mean)
            stats['expected_inside_not_exported']+=float(weights@(remaining_mean-outside_mean))
            stats['probability_zero_outside']+=float(weights@zero);zero_all[ri]*=zero
        rows[t]=dict(recorded_count=recorded,observation_censored=recorded is None,**stats)
    marginal=joint.sum(axis=0)
    return dict(by_type=rows,expected_total_instances=sum(r['expected_total'] for r in rows.values()),
        supplied_observed_instances=sum(result['observed'].values()),
        expected_outside_instances=sum(r['expected_outside'] for r in rows.values()),
        expected_inside_not_exported_instances=sum(r['expected_inside_not_exported'] for r in rows.values()),
        probability_zero_outside=float((joint*zero_all).sum()),
        nuisance_posterior_means=dict(view_fraction=float(marginal@f),recording_rate=float(marginal@d),
            product=float(marginal@(f*d))),
        interpretation='declared_model_diagnostic_not_measured_camera_coverage_or_calibrated_counts',
        sampler_or_generated_worlds_changed=False)


def trace(result):
    spec=result['spec'];prior=infer(spec,{},result['interventions'])
    counts={t:count_distribution(result,t) for t in spec['object_types']}
    return dict(background_prior=prior['background_posterior'],background_posterior=result['background_posterior'],
        configuration_prior=prior['configuration_posterior'],configuration_posterior=result['configuration_posterior'],
        evidence_log_probability=result['evidence_log_probability'],
        completion_diagnostics=completion_diagnostics(result),
        count_posteriors=counts,upper_count_boundary_probability={t:p[-1] for t,p in counts.items()},
        background_total_variation=.5*sum(abs(result['background_posterior'][b]-prior['background_posterior'][b]) for b in prior['background_posterior']),
        quadrature_order=result['quadrature_order'],count_domain_tail_probability_estimated=False,
        summary_observation_likelihood='censored_shared_view_and_recording_binomial',
        view_and_detection_separately_identified=False,camera_pose_likelihood_defined=False,
        probabilities_calibrated=False,causal_gain_validated=False,joint_metric_posterior_defined=False)
