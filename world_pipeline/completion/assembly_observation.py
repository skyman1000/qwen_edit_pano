"""Necessary visibility conditions on fixed observed support envelopes.

These are camera/AABB tests, not a learned observation likelihood or recovered
support surfaces. A nonempty region is a possibility, never a verified site.
"""
import itertools
import math


def host_domains(case, observed):
    tx=math.tan(math.radians(case['hfov'])/2)
    ty=tx*case['image_size'][1]/case['image_size'][0]
    normals={'left':(-1,0,-tx),'right':(1,0,-tx),
             'above':(0,1,-ty),'below':(0,-1,-ty),'rear':(0,0,-1)}
    result={}
    for obj in observed['objects']:
        c=obj['center_local_m']['value'];s=obj['size_aabb_local_m']['value']
        bounds=[(c[i]-s[i]/2,c[i]+s[i]/2) for i in range(3)]
        kinds={}
        for kind in ('top','seat','shelf'):
            # A necessary contact site. seat/shelf scan the whole host height
            # because an AABB does not tell us the actual interior surface.
            ys=(bounds[1][1],) if kind=='top' else bounds[1]
            points=list(itertools.product(bounds[0],ys,bounds[2]))
            kinds[kind]=[name for name,n in normals.items()
                if max(sum(a*b for a,b in zip(p,n)) for p in points)>1e-6]
        result[obj['track_id']]=dict(contact_region_candidates=kinds,
            scope='necessary_point_contact_condition_only',support_affordance_verified=False,
            nonempty_is_not_a_feasibility_certificate=True)
    return result


def check_observed_attachments(objects, domains):
    """Reject a contradictory *proposal*, without editing its inventory."""
    errors=[]
    for obj in objects:
        a=obj['attachment'];host=a['host'];kind=a['kind']
        if host in domains and kind in ('top','seat','shelf'):
            if not domains[host]['contact_region_candidates'][kind]:
                errors.append(dict(object=obj['key'],host=host,kind=kind,
                    reason='The entire possible contact envelope is inside the original input frustum. '
                    'A wholly outside-frustum addition cannot touch it. Reconsider this whole hidden-world hypothesis; '
                    'do not move the observed host or silently reassign this object.'))
    if errors:
        import json
        raise ValueError(json.dumps(dict(incompatible_observed_support_hypotheses=errors),ensure_ascii=False))
