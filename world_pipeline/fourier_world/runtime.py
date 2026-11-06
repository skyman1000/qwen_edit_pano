"""LACONIC-inspired scalar geometry PE; not a full LACONIC reproduction.

Keep the six raw meter coordinates plus validity bits, append masked sin/cos
features. No sequence-slot PE, new annotations, yaw wrapping, or image RoPE changes.
"""
import torch
from torch import nn
from ..scaled_world.runtime import WorldBranch as BaselineWorldBranch

ENCODING = dict(version='local_aabb_fourier32_raw_v1',
                frequencies=32, min_frequency=1., max_frequency=128.,
                coordinate_scale_m=1., include_raw=True,
                phase='coordinate_m / 1m * 128**(j/31); no 2pi',
                missing='zero raw and all sin/cos; retain validity bits',
                trig_precision='float32', input_dimension=392)


class GeometryFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', 128. ** (torch.arange(32).float()/31))

    def forward(self, geometry):
        # Keep high-frequency phase computations outside BF16 autocast.
        with torch.autocast(device_type=geometry.device.type, enabled=False):
            g = geometry.float()
            valid = g[..., 6:8] > .5
            scalar_valid = valid.repeat_interleave(3, dim=-1)
            xyz_size = torch.where(scalar_valid, g[..., :6], 0.)
            phase = xyz_size[..., None] * self.frequencies.float()
            pe = torch.cat((phase.sin(), phase.cos()), dim=-1)
            pe = torch.where(scalar_valid[..., None], pe, 0.)
            return torch.cat((xyz_size, valid.float(), pe.flatten(-2)), dim=-1)


class WorldBranch(BaselineWorldBranch):
    def __init__(self, hidden, blocks=(20,40,59), width=256):
        super().__init__(hidden, blocks, width)
        # Preserve identical category/attention initialization versus the baseline.
        # Extra encoder initialization must not perturb the surrounding RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.encoder.geometry = nn.Sequential(
                GeometryFeatures(), nn.Linear(392, width), nn.SiLU(), nn.Linear(width, width))
