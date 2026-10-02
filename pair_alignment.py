"""Offline target-frame registration. Never used to provide GT to the generator."""
import sys
from .common import ROOT


def front(image, size=128):
    import numpy as np
    vendor = str(ROOT/'benchmark_assets/PanFusion/external')
    if vendor not in sys.path:
        sys.path.insert(0,vendor)
    import py360convert
    return py360convert.e2p(np.asarray(image).astype('float32'),(90,90),0,0,(size,size),mode='bilinear')


def apply_alignment(image, alignment):
    import numpy as np
    from PIL import Image
    pixels = np.array(image)
    if alignment['mirror']:
        pixels = pixels[:, ::-1]
    shift = round(alignment['roll_fraction'] * pixels.shape[1])
    return Image.fromarray(np.roll(pixels,shift,axis=1))


def estimate_alignment(local, target):
    import numpy as np
    from PIL import Image
    # Compare central part to reduce JPEG/projection edge differences.
    def descriptor(x):
        x = np.asarray(x,dtype=np.float32)[8:-8,8:-8]
        x = x - x.mean(axis=(0,1),keepdims=True)
        norm = np.linalg.norm(x)
        return (x/max(float(norm),1e-8)).reshape(-1)
    desired = descriptor(local.resize((64,64),Image.Resampling.BILINEAR))
    small = np.asarray(target.resize((512,256),Image.Resampling.BILINEAR))
    candidates = []
    for mirror in (False,True):
        pixels = small[:,::-1] if mirror else small
        for shift in range(0,512,4):
            projected = front(np.roll(pixels,shift,axis=1),64)
            score = float(np.dot(desired,descriptor(projected)))
            candidates.append((score,mirror,shift))
    best = max(candidates)
    distinct = [s for s,m,x in candidates if m != best[1] or min((x-best[2])%512,(best[2]-x)%512)>24]
    # Refine to one pixel on the 512-wide registration grid (~0.7 degree).
    pixels = small[:,::-1] if best[1] else small
    refined = [(float(np.dot(desired,descriptor(front(np.roll(pixels,shift,axis=1),64)))),best[1],shift%512)
               for shift in range(best[2]-3,best[2]+4)]
    best = max(refined)
    return dict(method='central_rgb_ncc_yaw_mirror_v1',mirror=best[1],roll_fraction=best[2]/512,
                score=best[0],margin=best[0]-max(distinct),registration_width=512)
