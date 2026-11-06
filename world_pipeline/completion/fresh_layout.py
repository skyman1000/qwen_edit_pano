"""Joint layout with declared relations as soft objectives, fixed observed boxes.

The search algorithm is shared with the existing scene-program compiler. A
private function namespace substitutes the objective without mutating that
module or changing the behaviour of historical experiments.
"""
from copy import deepcopy
import types

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from . import scene_program_geometry as base
from .scm_joint_geometry import system, linear_feasible


def relation_objective(ids, old, relations, weight=1.):
    index={oid:i for i,oid in enumerate(ids)}
    available=set(index)|set(old)
    relevant=[r for r in relations if r['source'] in available and r['target'] in available]
    def point(oid,x):
        return x[2*index[oid]:2*index[oid]+2] if oid in index else np.array(old[oid]['center'])[[0,2]]
    def evaluate(x):
        value=0.;gradient=np.zeros_like(x)
        for r in relevant:
            a,b,k=r['source'],r['target'],r['kind'];delta=point(a,x)-point(b,x);g=np.zeros(2)
            if k=='near':
                # Soft attraction, not an invented hard maximum distance.
                value+=weight*float(delta@delta);g=2*weight*delta
            elif k in ('left_of','right_of','in_front_of','behind'):
                axis=0 if k in ('left_of','right_of') else 1
                sign=1 if k in ('right_of','in_front_of') else -1
                hinge=min(0.,sign*delta[axis]);value+=weight*hinge**2
                g[axis]=2*weight*hinge*sign
            if a in index:gradient[2*index[a]:2*index[a]+2]+=g
            if b in index:gradient[2*index[b]:2*index[b]+2]-=g
        return value,gradient
    return evaluate


def compile_program(case,observed,plan,bodies,groups,placements,relations=(),relation_weight=1.,**search):
    if not np.isfinite(relation_weight) or relation_weight<0:raise ValueError('Invalid relation weight')
    def solve(ids,plan,bodies,old,heights,case,regions,separations=()):
        A,b=system(ids,plan,bodies,old,heights,case,regions,separations)
        if A is None:return None
        matrix,target,energy,start=base.relative_quadratic(ids,bodies,old)
        rel=relation_objective(ids,old,relations,relation_weight)
        if np.min(A@start-b,initial=0.) < -1e-8:
            start=linear_feasible(A,b)
            if start is None:return None
        fit=minimize(lambda x:float(x@matrix@x-2*target@x)+rel(x)[0],start,
            jac=lambda x:2*(matrix@x-target)+rel(x)[1],constraints=LinearConstraint(A,b,np.inf),
            method='SLSQP',options=dict(maxiter=200,ftol=1e-10))
        valid=np.isfinite(fit.x).all() and np.min(A@fit.x-b,initial=0.)>=-1e-7
        x=fit.x if valid else start
        return dict(x=x,energy=energy(x)+rel(x)[0],converged=bool(fit.success and valid),
            residual_min=float(np.min(A@x-b,initial=0.)),separations=tuple(separations),regions=dict(regions))
    # Function globals are copied. No monkey patch and no process-global state.
    namespace={**base.__dict__,'solve':solve}
    compiler=types.FunctionType(base.compile_program.__code__,namespace,
        argdefs=base.compile_program.__defaults__)
    raw,review=compiler(case,observed,plan,bodies,groups,placements,**search)
    review.update(method='joint_program_soft_relation_compiler_v1',relation_weight=relation_weight,
        declared_relations=deepcopy(list(relations)),relations_are_soft=True)
    return raw,review


def review_relations(relations,items,observed):
    boxes={o['track_id']:o['center_local_m']['value'] for o in observed['objects']}
    boxes.update({o['id']:o['center'] for o in items});rows=[]
    for r in relations:
        a,b=np.array(boxes[r['source']]),np.array(boxes[r['target']]);d=a-b;k=r['kind'];ok=None
        if k=='requires':ok=True  # Presence implication, not inferred physical cause.
        elif k in ('left_of','right_of','above','below','in_front_of','behind'):
            axis={'left_of':0,'right_of':0,'above':1,'below':1,'in_front_of':2,'behind':2}[k]
            sign=1 if k in ('right_of','above','in_front_of') else -1
            ok=bool(sign*d[axis]>1e-7)
        rows.append(dict(**r,delta_camera_m=d.tolist(),distance_m=float(np.linalg.norm(d)),
            ordering_or_presence_satisfied=ok,metric_near_threshold_defined=False))
    return rows
