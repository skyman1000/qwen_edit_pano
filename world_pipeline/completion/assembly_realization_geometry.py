"""Declared ordering is a constraint; near is a primary surface-gap objective."""
from copy import deepcopy
from functools import partial
import math
import numpy as np

from . import assembly_activation_geometry as base, assembly_spatial as spatial

PARAMETERIZATION='relation_constrained_support_trees_v7'
layout_answer=base.layout_answer


def layout_prompt(case,observed,world,plane):
    return base.layout_prompt(case,observed,world,plane)+(
        '\nV7: The region is an initial numeric hypothesis, not an observed room boundary. '
        'The solver may compare other hidden directions to satisfy the declared relations. '
        'farther_than and nearer_than refer to camera Z, not distance from a wall. '
        'Propose sensible initial furniture positions; final optimization prioritizes declared ordering and surface proximity '
        'while preserving all instances, dimensions, attachment ownership and fixed observed boxes.')


def support_tree_groups(case,observed,world,layout):
    """Functional ownership does not force all furniture onto one frustum face."""
    revised={};numeric=deepcopy(world);numeric['groups']=[];contracts={o['id']:o for o in world['objects']};roots={}
    for oid in contracts:
        root=oid
        while contracts[root]['attachment']['host'] in contracts:root=contracts[root]['attachment']['host']
        roots[oid]=root
    old={o['track_id']:o['center_local_m']['value'] for o in observed['objects']};tx=math.tan(math.radians(case['hfov'])/2)
    neighbours={oid:set() for oid in set(contracts)|set(old)}
    for oid,obj in contracts.items():
        host=obj['attachment']['host']
        if host in neighbours:neighbours[oid].add(host);neighbours[host].add(oid)
    for relation in world['relations']:
        if relation['kind']=='near':
            a,b=relation['source'],relation['target'];neighbours[a].add(b);neighbours[b].add(a)
    mapping={};groups={g['key']:g for g in world['groups']}
    for root in sorted(set(roots.values())):
        members={oid for oid,r in roots.items() if r==root};original=contracts[root]['group'];gid=original+'__tree_'+root
        source=layout[original];group=deepcopy(source);group['id']=gid;group['objects']=[o for o in group['objects'] if o['id'] in members]
        anchor_ids=set();todo=list(members);seen=set()
        while todo:
            node=todo.pop()
            if node in seen:continue
            seen.add(node)
            if node in old:anchor_ids.add(node);continue
            todo.extend(neighbours[node]-seen)
        initial=source['region_domain'];order=list(dict.fromkeys(initial+['left','right','rear']))
        if anchor_ids:
            x,_,z=np.mean([old[a] for a in sorted(anchor_ids)],axis=0)
            distance=dict(left=max(0,x,(x+tx*z)/math.sqrt(1+tx*tx)),
                right=max(0,-x,(tx*z-x)/math.sqrt(1+tx*tx)),rear=max(0,z))
            order=sorted(order,key=lambda r:(distance[r],order.index(r)))
        group['region_domain']=order;revised[gid]=group
        numeric['groups'].append(dict(deepcopy(groups[original]),key=gid))
        for obj in numeric['objects']:
            if obj['id'] in members:obj['group']=gid;mapping[obj['id']]=dict(semantic_group=original,support_tree=root,numeric_group=gid)
    return numeric,revised,mapping


def compile_layout(case,observed,world,layout,plane):
    numeric,revised,mapping=support_tree_groups(case,observed,world,layout)
    old={o['track_id']:dict(center=o['center_local_m']['value'],size=o['size_aabb_local_m']['value']) for o in observed['objects']}
    def score(solved):
        r=solved['review'];s=r['program_solver'];boxes={o['id']:o for o in solved['items']}|old;near=0.
        for relation in world['relations']:
            if relation['kind']!='near':continue
            a,b=(boxes[relation[k]] for k in ('source','target'))
            gap=np.maximum(0,np.abs(np.asarray(a['center'])-np.asarray(b['center']))-(np.asarray(a['size'])+np.asarray(b['size']))/2)
            near+=float(gap@gap)
        return (r['hard_violations']+sum(v['ordering_or_presence_satisfied'] is False for v in r['declared_relation_review']),
            s.get('contact_and_frustum_linear_violations',0),r['attachment_contact_failures'],len(s['final_overlaps']),
            sum(v['volume_m3'] for v in s['final_overlaps']),near,
            sum(float(np.asarray(d)@np.asarray(d)) for d in s.get('local_adjustments_XZ_m',{}).values()))
    result=base.compile_layout(case,observed,numeric,revised,plane,
        repairer=partial(spatial.articulated_repair,hard_ordering=True,surface_proximity=True),refinement_score=score,include_default_preview=True)
    r=result['review'];r.update(geometry_parameterization=PARAMETERIZATION,
        declared_XZ_ordering_hard_constraints=True,near_optimization='lexicographic_surface_gap_then_displacement_without_distance_cutoff',
        initial_region_domains={k:g['region_domain'] for k,g in layout.items()},
        semantic_to_support_tree_groups=mapping,semantic_ownership_changed=False,
        region_scope='Revisable LM initialization; all final boxes still outside the original frustum',
        geometry_search_complete=False)
    if r['program_solver'].get('contact_and_frustum_linear_violations',0):
        r['geometry_contract_passed']=False;r['layout_proxy_passed']=False
    return result
