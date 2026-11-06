"""Camera-local full AABBs -> fractional ERP token support. No image/GT input."""
import math
import torch
from torch.nn import functional as F

POLICY = dict(version='camera_aabb_supersampled_erp_v1', supersample=4, object_chunk=8,
    smoothing='separable_121_token_kernel_periodic_x_replicated_y',
    overlap='coverage / max(1, sum_object_coverage)',
    camera_inside='omit_local_routing_keep_legacy_context', occlusion='not_inferred_from_coarse_boxes',
    global_token='zero_spatial_support', geometry_gradients=False)


@torch.no_grad()
def erp_support(geometry, mask, height, width, padding=0, supersample=4):
    """Return [1,padded_queries,objects+global], with original input slot order.

    Camera x right/y image up/z forward; width=2*height. AABB dimensions are
    FULL lengths. Fractional coverage uses uniform ERP subpixel rays (not sphere
    area weights). Projection has no learnable parameters and consumes no masks.
    """
    if (geometry.ndim != 3 or geometry.shape[0] != 1 or geometry.shape[-1] != 8 or
        mask.shape != geometry.shape[:2] or width != 2*height or height < 1 or
        padding < 0 or padding > width or supersample < 1):
        raise ValueError('Invalid camera geometry/token grid')
    device=geometry.device
    with torch.autocast(device.type, enabled=False):
        g=geometry.float();count=g.shape[1]
        if not torch.isfinite(g).all():raise ValueError('Nonfinite input geometry')
        valid=mask[0].bool() & (g[0,:,6]>.5) & (g[0,:,7]>.5) & (g[0,:,3:6]>0).all(-1)
        valid[0]=False
        centers=g[0,:,:3];half=g[0,:,3:6]/2
        inside=(centers.abs()<=half).all(-1)
        valid &= ~inside
        coverage=torch.zeros(count,height,width,device=device,dtype=torch.float32)
        # x/y subdivisions are exactly pixel-centred in the unpadded ERP grid.
        yy=(torch.arange(height*supersample,device=device,dtype=torch.float32)+.5)/(height*supersample)
        xx=(torch.arange(width*supersample,device=device,dtype=torch.float32)+.5)/(width*supersample)
        lat=math.pi/2-yy*math.pi;lon=xx*(2*math.pi)-math.pi
        rays=torch.stack((lat.cos()[:,None]*lon.sin()[None,:],
            lat.sin()[:,None].expand(-1,width*supersample),lat.cos()[:,None]*lon.cos()[None,:]),-1).reshape(-1,3)
        parallel=rays.abs()<1e-7
        inv=torch.where(parallel,torch.ones_like(rays),rays).reciprocal()
        indices=valid.nonzero().flatten()
        for selected in indices.split(POLICY['object_chunk']):
            low=(centers[selected]-half[selected])[:,None,:]
            high=(centers[selected]+half[selected])[:,None,:]
            a=low*inv;b=high*inv
            # Parallel rays hit a slab only if the origin is within that slab.
            within=(low<=0)&(high>=0)
            near=torch.where(parallel,torch.where(within,-torch.inf,torch.inf),torch.minimum(a,b)).amax(-1)
            far=torch.where(parallel,torch.where(within,torch.inf,-torch.inf),torch.maximum(a,b)).amin(-1)
            hits=(far>=near.clamp_min(0)) & (far>0)
            coverage[selected]=hits.reshape(-1,height,supersample,width,supersample).float().mean((2,4))
        if count:
            kernel=torch.tensor([1.,2.,1.],device=device)
            kernel=(kernel[:,None]*kernel[None,:])[None,None]/16
            extended=F.pad(coverage[:,None],(1,1,0,0),mode='circular')
            extended=F.pad(extended,(0,0,1,1),mode='replicate')
            coverage=F.conv2d(extended,kernel)[:,0]
        weights=coverage/coverage.sum(0,keepdim=True).clamp_min(1)
        if padding:weights=F.pad(weights,(padding,padding,0,0),mode='circular')
        weights=weights.flatten(1).T[None].contiguous()
        if not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite routing')
        return weights,dict(valid_boxes=int(valid.sum()),camera_inside_boxes=int((inside&mask[0]& (g[0,:,6]>.5)&(g[0,:,7]>.5)).sum()),
            routed_boxes=int((coverage.flatten(1).sum(-1)>0).sum()),queries=weights.shape[1],
            covered_query_fraction=float((weights.sum(-1)>0).float().mean()),maximum_overlap_weight=float(weights.sum(-1).max()))
