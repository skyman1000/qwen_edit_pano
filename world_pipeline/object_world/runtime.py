"""Semantic object tokens, unchanged Fourier geometry and target-only attention."""
import torch
from torch import nn
from torch.nn import functional as F

from ..fourier_world.runtime import GeometryFeatures
from ..scaled_world.runtime import WorldBranch as BaseBranch

VERSION = 'polished_single_face_semantic_region_world_v1'


class SemanticEncoder(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        if features.ndim != 2 or features.shape[0] != 42 or not torch.isfinite(features).all():
            raise ValueError('Expected finite 42-slot frozen semantic features')
        self.register_buffer('class_features', F.normalize(features.detach().float(), dim=-1))
        self.semantic = nn.Linear(features.shape[1], width, bias=False)
        self.geometry = nn.Sequential(GeometryFeatures(), nn.Linear(392, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.empty = nn.Parameter(torch.randn(1, 1, width)*.02)

    def forward(self, category, geometry, mask):
        semantic=self.semantic(self.class_features[category])
        spatial=self.geometry(geometry)
        with torch.no_grad():
            # Report pre-fusion scales, excluding the separately learned global.
            if mask[:,1:].any():
                s=semantic[:,1:][mask[:,1:]].float().square().mean().sqrt()
                g=spatial[:,1:][mask[:,1:]].float().square().mean().sqrt()
                self.feature_diagnostics=dict(semantic_rms=float(s),geometry_rms=float(g),
                                              semantic_to_geometry=float(s/g.clamp_min(1e-12)))
            else:
                self.feature_diagnostics=dict(semantic_rms=0.,geometry_rms=0.,semantic_to_geometry=0.)
        tokens = self.norm(semantic+spatial)
        tokens = torch.cat([self.empty.expand(tokens.shape[0], -1, -1), tokens[:, 1:]], 1)
        return tokens, mask


class WorldBranch(BaseBranch):
    def __init__(self, hidden, features, blocks=(20, 40, 59), width=256):
        super().__init__(hidden, blocks, width)
        self.encoder = SemanticEncoder(features, width)


def region_weights(instance, valid, object_ids, size):
    """Equal weight per represented instance, fractional area on the latent grid.

    Construct pixel mass before area pooling. Even a subcell object is retained;
    a large object cannot dominate merely because it covers more pixels.
    """
    ids = torch.as_tensor(instance, dtype=torch.long)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if ids.ndim != 2 or valid.shape != ids.shape:
        raise ValueError('Expected matching 2D instance and loss-valid maps')
    represented = valid & torch.isin(ids, torch.as_tensor(list(object_ids), dtype=torch.long))
    mass = torch.zeros(ids.shape, dtype=torch.float32)
    if represented.any():
        _, inverse, counts = torch.unique(ids[represented], return_inverse=True, return_counts=True)
        mass[represented] = 1./(len(counts)*counts[inverse].float())
        mass *= ids.numel()
    weights = F.interpolate(mass[None, None], size=size, mode='area')[0, 0]
    # area pooling is mass preserving for the 512->128 default; explicitly
    # normalize for other supported integer sizes too.
    if represented.any():
        weights /= weights.mean()
    return weights


def object_region_loss(prediction, target, weights):
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 1 or prediction.shape[0]!=1:
        raise ValueError('Region weighting supports one sample: expected matching 1,C,1,H,W velocity tensors')
    if weights.shape != prediction.shape[-2:] or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid latent object weights')
    error = (prediction.float()-target.float()).square().mean(dim=(0, 1, 2))
    return (error*weights.float()).mean()
