"""Fit explicit cue mechanisms from audited functional labels, never biography.

Each independent scene: scene_id, split (train/val), roles (all six binary),
cue_reports (cue -> 0/1/null), optional needs [{role,active:0/1}].
Zero is an explicitly reviewed report outcome under the configured observation
process, NOT an automatically inferred absence from a partial image.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import assembly_functional as f
from ..common import new_output, read, write, sha


def validate_records(rows):
    scenes=set()
    for row in rows:
        if not isinstance(row.get('scene_id'),str) or not row['scene_id'] or row['scene_id'] in scenes:
            raise ValueError('One record per independent scene; repeated/cross-split scene_id rejected')
        scenes.add(row['scene_id'])
        if row.get('split') not in ('train','val'):raise ValueError('split must be train/val')
        if set(row.get('roles',{}))!=set(f.ROLES) or any(type(v) is not int or v not in (0,1) for v in row['roles'].values()):
            raise ValueError('Every role needs an audited binary functional-organization label')
        if not isinstance(row.get('cue_reports'),dict) or not set(row['cue_reports'])<=set(f.PATTERNS):
            raise ValueError('Unknown cue_reports vocabulary')
        if any(v is not None and (type(v) is not int or v not in (0,1)) for v in row['cue_reports'].values()):
            raise ValueError('Cue outcomes must be reviewed 0/1 or unobserved null')
        for need in row.get('needs',[]):
            if need.get('role') not in f.ROLES or type(need.get('active')) is not int or need['active'] not in (0,1):
                raise ValueError('Needs require role and audited binary active label')
    if not any(r['split']=='train' for r in rows):raise ValueError('No training scenes')
    return rows


def fit(rows, base=None, regularization=5.):
    validate_records(rows);p=f.parameters(base);train=[r for r in rows if r['split']=='train']
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('regularization must be positive')
    for role in f.ROLES:
        # Beta(1,1) smoothing, training scenes only.
        p['role_priors'][role]=(1+sum(r['roles'][role] for r in train))/(2+len(train))
    fitting={}
    # cue_leak and detector/coverage parameters stay fixed. Trying to learn all
    # these simultaneously from binary reports is generally not identifiable.
    for cue,strengths in p['cue_strengths'].items():
        records=[r for r in train if r['cue_reports'].get(cue) is not None]
        if not records:
            fitting[cue]=dict(fitted=False,reason='no explicit report labels');continue
        roles=list(strengths);initial=np.asarray([strengths[r] for r in roles])
        x=np.asarray([[r['roles'][role] for role in roles] for r in records]);y=np.asarray([r['cue_reports'][cue] for r in records])
        def objective(weights):
            presence=1-(1-p['cue_leak'])*np.prod(1-x*weights,axis=1)
            prob=np.clip(f._report_probability(p,presence),1e-9,1-1e-9)
            return float(-np.sum(y*np.log(prob)+(1-y)*np.log(1-prob))+
                         regularization*np.sum((weights-initial)**2))
        result=minimize(objective,initial,method='L-BFGS-B',bounds=[(.001,.999)]*len(roles))
        if not result.success:raise ValueError('Cue fit failed: '+cue+' '+result.message)
        p['cue_strengths'][cue]=dict(zip(roles,map(float,result.x)))
        fitting[cue]=dict(fitted=True,train_reports=len(records),penalized_nll=float(result.fun),
            varying_parents=[r for j,r in enumerate(roles) if len(set(x[:,j]))>1])
    for state,name in ((0,'inactive'),(1,'active')):
        labels=[n['active'] for r in train for n in r.get('needs',[]) if r['roles'][n['role']]==state]
        if labels:
            center=p['need_activation'][name]
            p['need_activation'][name]=(regularization*center+sum(labels))/(regularization+len(labels))
        fitting['need_'+name]=dict(train_labels=len(labels),fitted=bool(labels))
    p['provenance']='audited_scene_supervised_MAP_and_smoothed_counts_NOT_CAUSALLY_IDENTIFIED'
    report=dict(independent_train_scenes=len(train),independent_val_scenes=sum(r['split']=='val' for r in rows),
        fitting=fitting,coverage_recall_false_positive_and_leak='fixed_not_identifiable_jointly_from_binary_reports',
        strategy_weights='unchanged_explicit_reference_weights',
        calibrated=False,causal_structure_identified=False)
    val=[r for r in rows if r['split']=='val'];loss=[];squared=[]
    for row in val:
        h={r:str(row['roles'][r]) for r in f.ROLES}
        for cue,y in row['cue_reports'].items():
            if y is None:continue
            prob=float(f._report_probability(p,f._cue_presence(p,cue,h)))
            loss.append(-(y*math.log(prob)+(1-y)*math.log(1-prob)));squared.append((prob-y)**2)
    report['validation_cue_report_metrics_given_audited_roles']=dict(
        n=len(loss),nll=float(np.mean(loss)) if loss else None,brier=float(np.mean(squared)) if squared else None,
        scope='cue_report_prediction_not_hidden_inventory_calibration')
    return f.parameters(p),report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parameters',type=Path);parser.add_argument('--regularization',type=float,default=5.)
    a=parser.parse_args();rows=[json.loads(line) for line in a.annotations.read_text().splitlines() if line.strip()]
    p,report=fit(rows,read(a.parameters) if a.parameters else None,a.regularization)
    out=new_output(a.output);write(out/'functional_parameters.json',p)
    report['annotation_sha256']=sha(a.annotations);report['scene_splits']={r['scene_id']:r['split'] for r in rows}
    write(out/'fit_report.json',report);print(out/'functional_parameters.json')


if __name__=='__main__':main()
