"""Fit count-mechanism parameters from audited complete independent scenes.

JSONL: scene_id, split=train/val, roles={six audited binary labels},
full_counts={all count families}, optional views=[{all matched observed counts}].
These are annotation proxies, not observed resident intentions or causal labels.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ..common import read,write,new_output,sha
from . import assembly_population as pop


def fit(records,base=None,regularization=1.):
    p=pop.parameters(base);seen=set()
    if not math.isfinite(regularization) or regularization<=0:raise ValueError('regularization must be positive')
    for row in records:
        if not isinstance(row.get('scene_id'),str) or not row['scene_id'] or row['scene_id'] in seen:
            raise ValueError('Require one record per independent scene; repeated/cross-split scene_id rejected')
        seen.add(row['scene_id'])
        if row.get('split') not in ('train','val'):raise ValueError('split must be train/val')
        if set(row.get('roles',{}))!=set(pop.ROLES) or any(type(v) is not int or v not in (0,1) for v in row['roles'].values()):
            raise ValueError('All six audited functional-organization proxy labels are required')
        for counts in [row.get('full_counts',{})]+row.get('views',[]):
            if set(counts)!=set(pop.FAMILIES) or any(type(v) is not int or v<0 for v in counts.values()):
                raise ValueError('All count families require nonnegative integer counts')
        if any(any(v[k]>row['full_counts'][k] for k in pop.FAMILIES) for v in row.get('views',[])):
            raise ValueError('View counts must be matched true instances, not unverified detector false positives')
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    if not train:raise ValueError('No training scenes')
    for role in pop.ROLES:p['role_priors'][role]=(1+sum(r['roles'][role] for r in train))/(2+len(train))
    x=np.asarray([[1.]+[r['roles'][role] for role in pop.ROLES] for r in train],float)
    report={};dispersion=p['dispersion']
    for family in pop.FAMILIES:
        y=np.asarray([r['full_counts'][family] for r in train],float)
        initial=np.asarray([p['base_means'][family]]+[p['role_loadings'][r].get(family,0.) for r in pop.ROLES])
        def objective(theta):
            mean=x@theta;base=theta[0];gamma=np.clip((mean-base)/mean,0.,1-1e-12)
            bg=gammaln(y+dispersion)-gammaln(dispersion)-gammaln(y+1)+dispersion*np.log(dispersion/(dispersion+base))+y*np.log(base/(dispersion+base))
            fg=y*np.log(mean)-mean-gammaln(y+1)
            ll=np.logaddexp(np.log1p(-gamma)+bg,np.log(np.maximum(gamma,1e-300))+fg)
            return float(-ll.sum()+regularization*np.sum((theta-initial)**2))
        result=minimize(objective,initial,method='L-BFGS-B',bounds=[(.001,None)]+[(0.,None)]*len(pop.ROLES))
        if not result.success:raise ValueError('Count fit failed for '+family+': '+str(result.message))
        p['base_means'][family]=float(result.x[0])
        for i,role in enumerate(pop.ROLES):p['role_loadings'][role][family]=float(result.x[i+1])
        report[family]=dict(penalized_nll=float(result.fun),train_scenes=len(train))
    # Absolute scale and all rate coefficients cannot be identified separately.
    p['population_scales']=[1.];p['scale_weights']=[1.]
    coverage=[]
    for row in train:
        total=sum(row['full_counts'].values())
        if total and row.get('views'):
            # One statistic per independent scene, so many crops do not dominate.
            coverage.append(float(np.mean([sum(v.values())/total for v in row['views']])))
    if coverage:
        modes=sorted(set(float(np.clip(q,.01,.99)) for q in np.quantile(coverage,[.25,.5,.75])))
        p['report_probabilities']=modes;p['report_weights']=[1.]*len(modes)
    losses=[]
    for row in val:
        h={r:str(row['roles'][r]) for r in pop.ROLES}
        losses.append(-sum(pop._log_nb(row['full_counts'][k],pop._mean(p,h,0,k,False),dispersion)+
            pop._log_count_ratio(row['full_counts'][k],p,h,0,k) for k in pop.FAMILIES))
    p['provenance']='audited_complete_scene_count_MAP_functional_proxy_labels_NOT_CAUSALLY_IDENTIFIED'
    return pop.parameters(p),dict(train_scenes=len(train),val_scenes=len(val),family_fits=report,
        population_scale_fixed_to_one=True,dispersion='fixed_reference',report_process_fitted=bool(coverage),
        report_process_scope='approximate_matched_count_coverage_not_geometric_visibility',
        validation_full_count_nll_given_audited_roles=float(np.mean(losses)) if losses else None,
        hidden_inventory_calibrated=False,causal_structure_identified=False,
        scene_splits={r['scene_id']:r['split'] for r in records})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parameters',type=Path)
    parser.add_argument('--regularization',type=float,default=1.);a=parser.parse_args()
    records=[json.loads(s) for s in a.annotations.read_text().splitlines() if s.strip()]
    p,report=fit(records,read(a.parameters) if a.parameters else None,a.regularization)
    out=new_output(a.output);write(out/'population_parameters.json',p)
    report['annotation_sha256']=sha(a.annotations);write(out/'fit_report.json',report)
    print(out/'population_parameters.json')


if __name__=='__main__':main()
