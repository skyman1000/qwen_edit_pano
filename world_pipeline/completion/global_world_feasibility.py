"""Bounded disjunctive AABB feasibility seed for the global layout solver.

Fixed yaw/dimensions/support/region, frozen observed boxes. Enumerates separation
halfspaces through MILP rather than hoping local gradients escape intersections.
Near circles have a conservative polygon approximation; separate footprint gap
uses a conservative axis witness. A failed seed is not a proof of nonlinear
infeasibility. Final acceptance always uses the independent numeric/ray audits.
"""
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def plus(*terms):
    coefficients = {};constant = 0.
    for a,c in terms:
        constant += c
        for k,v in a.items():coefficients[k] = coefficients.get(k, 0.)+v
    return {k:v for k,v in coefficients.items() if v},constant


def times(term, weight):
    return {k:weight*v for k,v in term[0].items()},weight*term[1]


def constant(value):return {},float(value)


def seed_layout(ids, specs, obs, sizes, room, proposal, initial, lower, upper,
                pairs, relations, zones, camera, rotation, time_limit=15.):
    lows = list(lower.ravel());highs = list(upper.ravel());integrality = [0]*len(lows);cost = [0.]*len(lows)
    rows = [];row_lows = []
    centres = {i:[constant(v) for v in o['center_m']] for i,o in obs.items()}
    centres.update({i:[({3*k+j:1.},0.) for j in range(3)] for k,i in enumerate(ids)})
    lo = np.array([room['x_min'],room['floor_y'],room['z_min']])
    hi = np.array([room['x_max'],room['ceiling_y'],room['z_max']])

    def variable(low, high, integer=False, weight=0.):
        k = len(lows);lows.append(float(low));highs.append(float(high));integrality.append(int(integer));cost.append(weight)
        return {k:1.},0.

    def add(margin):
        rows.append(margin[0]);row_lows.append(-margin[1])

    def limits(margin):
        a,c = margin
        return (c+sum(v*(lows[k] if v>=0 else highs[k]) for k,v in a.items()),
                c+sum(v*(highs[k] if v>=0 else lows[k]) for k,v in a.items()))

    def disjunction(branches):
        # Discard impossible branches; avoid binary variables for fixed separated pairs.
        intervals = [(b,limits(b)) for b in branches]
        if any(low >= 0 for _,(low,high) in intervals):return
        possible = [(b,low) for b,(low,high) in intervals if high >= 0]
        if len(possible) == 1:add(possible[0][0]);return
        binaries = []
        for b,low in possible:
            indicator = variable(0,1,True);binaries.append(indicator)
            big_m = max(0.,-low)+1e-7
            add(plus(b, constant(big_m), times(indicator,-big_m)))
        add(plus(*binaries,constant(-1.)))  # Empty disjunction explicitly infeasible.

    def difference(a,b,axis):return plus(centres[a][axis],times(centres[b][axis],-1.))

    def separation(ca,sa,cb,sb,gap=0.):
        branches = []
        for j in range(3):
            delta = plus(ca[j],times(cb[j],-1.));radius = (sa[j]+sb[j])/2+gap
            branches += [plus(times(delta,sign),constant(-radius)) for sign in (-1.,1.)]
        disjunction(branches)

    for i in ids:
        c = centres[i];size = sizes[i];support = specs[i]['support'];kind = support['kind'];host = support['host']
        if kind in ('floor','ceiling','top'):
            y = constant(lo[1]+size[1]/2) if kind=='floor' else constant(hi[1]-size[1]/2)
            if kind=='top':y = plus(centres[host][1],constant((sizes[host][1]+size[1])/2))
            residual = plus(c[1],times(y,-1.));add(residual);add(times(residual,-1.))
        if kind in ('top','inside'):
            for j in ([0,2] if kind=='top' else [0,1,2]):
                d = difference(i,host,j);extent = (sizes[host][j]-size[j])/2
                add(plus(constant(extent),d));add(plus(constant(extent),times(d,-1.)))
        if kind=='wall':
            branches = []
            for j in (0,2):
                branches += [plus(constant(.08+lo[j]+size[j]/2),times(c[j],-1.)),
                             plus(c[j],constant(.08-hi[j]+size[j]/2))]
            disjunction(branches)
        separation(c,size,[constant(0.)]*3,[0.,0.,0.],.05)
        if camera and specs[i]['visibility']=='outside_fov':
            th = math.tan(math.radians(camera['hfov'])/2);tv = th*camera['image_size'][1]/camera['image_size'][0]
            normals = np.array([[0,0,1],[1,0,th],[-1,0,th],[0,1,tv],[0,-1,tv]],float)
            plane_limits = [.01,0.,0.,0.,0.];branches = []
            for n,limit in zip(normals,plane_limits):
                branches.append(plus(constant(limit-np.abs(n)@size/2-.02),
                    *[times(c[j],-n[j]) for j in range(3)]))
            region = specs[i].get('camera_region','outside_fov')
            if region in ('rear','left','right'):
                k = {'rear':0,'left':1,'right':2}[region];add(branches[k])
                if region in ('left','right'):
                    add(plus(times(c[0],-1. if region=='left' else 1.),constant(-size[0]/2)))
            else:disjunction(branches)
    for a,b in pairs:separation(centres[a],sizes[a],centres[b],sizes[b])
    for relation in relations:
        a,b = relation['source'],relation['target']
        dx = difference(a,b,0);dz = difference(a,b,2);kind = relation['kind']
        if kind=='facing':
            # delta points from source to target; camera-Y positive yaw convention.
            f = rotation(proposal[a]['yaw_deg'])[:,2]
            forward = plus(times(dx,-f[0]),times(dz,-f[2]))
            across = plus(times(dx,-f[2]),times(dz,f[0]))
            add(plus(times(forward,math.sqrt(3)),across));add(plus(times(forward,math.sqrt(3)),times(across,-1.)))
        elif kind=='near':
            distance = relation['distance_m']
            if relation.get('measure','center')=='footprint_gap':
                gaps = []
                for axis,delta in ((0,dx),(2,dz)):
                    bound = max(abs(v) for v in limits(delta))+(sizes[a][axis]+sizes[b][axis])/2+distance
                    gap = variable(0,bound);gaps.append(gap)
                    for sign in (-1.,1.):
                        add(plus(gap,times(delta,-sign),constant((sizes[a][axis]+sizes[b][axis])/2)))
                dx,dz = gaps
            for angle in np.arange(16)*2*math.pi/16:
                add(plus(constant(distance*math.cos(math.pi/16)),times(dx,-math.cos(angle)),times(dz,-math.sin(angle))))
        else:
            branches = []
            for axis,delta in ((0,dx),(2,dz)):
                radius = (sizes[a][axis]+sizes[b][axis])/2+relation['distance_m']
                branches += [plus(times(delta,sign),constant(-radius)) for sign in (-1.,1.)]
            disjunction(branches)
    for zone in zones:
        owner = zone['owner'];rot = rotation(proposal[owner]['yaw_deg']);size = np.abs(rot)@np.asarray(zone['size_m'])
        if zone.get('base')=='floor':
            offset = rot@np.array([zone['offset_xz_m'][0],0.,zone['offset_xz_m'][1]])
            c = [plus(centres[owner][j],constant(offset[j])) for j in range(3)]
            c[1] = constant(lo[1]+size[1]/2)
        else:
            offset = rot@np.asarray(zone['offset_m'])
            c = [plus(centres[owner][j],constant(offset[j])) for j in range(3)]
        for j in range(3):
            add(plus(c[j],constant(-size[j]/2-lo[j])));add(plus(times(c[j],-1.),constant(hi[j]-size[j]/2)))
        # A usable approach box must not intersect even its owning object.
        for other in centres:separation(c,size,centres[other],sizes[other])
    for k,target in enumerate(initial.ravel()):
        residual = ({k:1.},-target);bound = max(abs(v) for v in limits(residual))+1.
        error = variable(0,bound,weight=1.)
        add(plus(error,residual));add(plus(error,times(residual,-1.)))
    rr = [];cc = [];vv = []
    for r,a in enumerate(rows):
        for col,value in a.items():rr.append(r);cc.append(col);vv.append(value)
    matrix = coo_matrix((vv,(rr,cc)),shape=(len(rows),len(lows))).tocsc()
    answer = milp(np.array(cost),integrality=np.array(integrality),bounds=Bounds(lows,highs),
        constraints=LinearConstraint(matrix,np.array(row_lows),np.inf),
        options=dict(time_limit=time_limit,node_limit=5000,mip_rel_gap=.02))
    vector = None if answer.x is None else np.asarray(answer.x[:3*len(ids)])
    audit = dict(status=int(answer.status),message=str(answer.message),solution_found=vector is not None,
        variables=len(lows),binaries=sum(integrality),constraints=len(rows),time_limit_s=time_limit,
        node_limit=5000,scope='conservative_linear_seed; not_full_nonlinear_infeasibility_certificate',
        inventory_changed=False,dimensions_changed=False,observations_changed=False)
    return vector,audit
