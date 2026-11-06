"""Closest translation into a declared off-FOV region, for numerical diagnostics.

No arbitrary translation radius. This is a tiny convex projection, not causal
inference, semantic correction or permission to move an invalid scene proposal.
It is NOT automatically wired into the existing LLM completion pilot.
"""
import itertools
import math

import numpy as np

from ..common import finite_vector
from .off_fov_geometry import region_margin
from .scene_completion import view_check


def closest_region_center(center,size,region,hfov,image_size,epsilon=1e-6):
    if (not finite_vector(center,3) or not finite_vector(size,3) or min(size)<=0
            or not 0<hfov<180 or len(image_size)!=2 or min(image_size)<=0
            or not math.isfinite(epsilon) or epsilon<=0):
        raise ValueError('Invalid geometry/camera')
    c=np.asarray(center,dtype=float);s=np.asarray(size,dtype=float)
    tx=math.tan(math.radians(hfov)/2);ty=tx*image_size[1]/image_size[0]
    if region=='behind_camera':
        A=np.array([[0,0,-1.]]);b=np.array([s[2]/2+epsilon])
    elif region in ('left','right','above','below'):
        axis=0 if region in ('left','right') else 1
        sign=1 if region in ('right','above') else -1
        tangent=tx if axis==0 else ty
        first=np.zeros(3);first[axis]=sign
        second=first.copy();second[2]=-tangent
        A=np.array([first,second]);b=np.array([s[axis]/2+epsilon,(s[axis]+tangent*s[2])/2+epsilon])
    else:raise ValueError('Unknown region')
    candidates=[]
    for count in range(len(b)+1):
        for active in itertools.combinations(range(len(b)),count):
            if not active:p=c.copy()
            else:
                active_A=A[list(active)];active_b=b[list(active)]
                try:lagrange=np.linalg.solve(active_A@active_A.T,active_b-active_A@c)
                except np.linalg.LinAlgError:continue
                if np.any(lagrange < -1e-9):continue
                p=c+active_A.T@lagrange
            if np.all(A@p>=b-1e-10):candidates.append(p)
    if not candidates:raise ArithmeticError('Convex projection failed')
    best=min(candidates,key=lambda p:float(np.sum((p-c)**2)))
    return dict(center=best.tolist(),translation_m=(best-c).tolist(),
                translation_norm_m=float(np.linalg.norm(best-c)),
                input_frustum_intersects=view_check(best.tolist(),size,hfov,image_size)['intersects_input_box_proxy'],
                region_margin_m=region_margin(best.tolist(),size,region,hfov,image_size),
                semantic_validity_verified=False,diagnostic_candidate_only=True)
