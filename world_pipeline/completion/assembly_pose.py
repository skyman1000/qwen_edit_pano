"""Fitted, model-relative support pose equations and local interventions.

This exports the actual selected layout's dependency mechanism. It does not
infer real-world causation, simulate dynamics, or silently re-optimize a do().
"""
from copy import deepcopy
import numpy as np


def fit(world,items,observed):
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    boxes={o['id']:o for o in items};nodes=[];pending=dict(boxes)
    while pending:
        ready=[o for o in world['objects'] if o['id'] in pending and
            (o['attachment']['host'] is None or o['attachment']['host'] in old or o['attachment']['host'] not in pending)]
        if not ready:raise ValueError('Cannot fit pose equations: missing host or cycle')
        for obj in ready:
            oid=obj['id'];b=pending.pop(oid);host=obj['attachment']['host'];kind=obj['attachment']['kind']
            node=dict(id=oid,host=host,kind=kind,size=list(b['size']),installation_node=obj.get('installation_node'),
                parents=['P_'+host] if host else [],equation='parent_center_plus_camera_relative_offset' if host else 'root_pose_parameter')
            if host:
                parent=(old|boxes)[host];node['offset_camera_m']=(np.array(b['center'])-parent['center']).tolist()
                # Height equation distinguishes a supported child from a
                # generic spatial edge. Interior levels are fitted hypotheses.
                if kind=='top':node['height_equation']='parent_top_plus_child_half_height'
                elif kind in ('seat','shelf'):
                    node['height_equation']='parent_fraction_plus_child_half_height'
                    node['surface_fraction']=(b['center'][1]-b['size'][1]/2-parent['center'][1]+parent['size'][1]/2)/parent['size'][1]
                elif kind=='hang':node['height_equation']='parent_center_minus_drop_minus_child_half_height'
                else:node['height_equation']='fixed_relative_offset'
            else:node['root_center_m']=list(b['center'])
            nodes.append(node)
    return dict(protocol='fitted_support_pose_equations_v1',nodes=nodes,observed=deepcopy(old),
        scope='conditional_on_selected_inventory_and_fitted_metric_parameters',
        real_physical_mechanisms_identified=False,optimization_reexecuted_under_do=False,
        world_context_and_noise_held_fixed=True,parameters_are_fitted_hypotheses=True)


def forward(model,do=None):
    do=do or {};known={n['id'] for n in model['nodes']}
    if not set(do)<=known:raise ValueError('Pose interventions require exact predicted object IDs; observations stay fixed')
    state=deepcopy(model['observed'])
    for n in model['nodes']:
        oid=n['id'];host=n['host']
        if oid in do:
            center=np.asarray(do[oid],dtype=float)
            if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('do(P) must be a finite 3D centre')
        elif host:
            parent=state[host];center=np.array(parent['center'])+n['offset_camera_m']
            if n['kind']=='top':center[1]=parent['center'][1]+(parent['size'][1]+n['size'][1])/2
            elif n['kind'] in ('seat','shelf'):
                center[1]=parent['center'][1]-parent['size'][1]/2+n['surface_fraction']*parent['size'][1]+n['size'][1]/2
        else:center=np.array(n['root_center_m'])
        state[oid]=dict(center=center.tolist(),size=deepcopy(n['size']))
    return {n['id']:state[n['id']] for n in model['nodes']}


def diagnostics(model,items):
    before=forward(model);expected={o['id']:o['center'] for o in items};queries=[]
    error=max((float(np.max(np.abs(np.array(before[k]['center'])-v))) for k,v in expected.items()),default=0.)
    if error>1e-8:raise ValueError('Fitted pose mechanism does not replay the selected layout')
    for n in model['nodes']:
        oid=n['id']
        if not any(r['host']==oid for r in model['nodes']):continue
        # Explicit diagnostic sensitivity input; not a placement distance prior.
        delta=np.array([.5,0.,0.]);position=(np.array(before[oid]['center'])+delta).tolist()
        after=forward(model,{oid:position})
        affected=[k for k in before if not np.allclose(before[k]['center'],after[k]['center'],rtol=0,atol=1e-8)]
        queries.append(dict(do={'P_'+oid:position},diagnostic_translation_m=delta.tolist(),changed_objects=affected,
            before={k:before[k] for k in affected},after={k:after[k] for k in affected},
            observed_objects_unchanged=True,model_called=False,layout_reoptimized=False,
            physical_counterfactual_validity_verified=False))
    return dict(exact_layout_replay_error_m=error,queries=queries,causal_gain_validated=False,
        scope='assumed_kinematic_dependency_sensitivity_not_physical_dynamics')
