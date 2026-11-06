"""Explicit precision and reproducible sample / loss RNG policy for oracle v2."""
import random
import torch
from ..oracle_memory import training_autocast
from ..oracle_model import WorldBranch as OriginalWorldBranch


class WorldBranch(OriginalWorldBranch):
    """Keep native pipeline precision; use BF16 autocast only within the World branch.

    State dict keys/parameters match the original branch. Training and inference
    use the same policy even when only training has an outer autocast context.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for branch in self.branches.values():
            original = branch.forward
            def forward(hidden, *values, _original=original, **options):
                with training_autocast(hidden.device.type):
                    return _original(hidden, *values, **options)
            branch.forward = forward

    def set_world(self, tensors):
        if tensors is None:
            self.current = None
        else:
            with training_autocast(tensors[0].device.type):
                self.current = self.encoder(*tensors)


def seed_step(seed, step):
    # panorama_loss uses Python random.choice for the yaw angle.
    random.seed(seed + step)
    torch.manual_seed(seed + step)


def learning_rate(lr, warmup, step):
    return lr * min(1., (step + 1) / max(1, warmup))


class EpochSampler:
    """Every sample once per shuffled pass; random access supports exact resume.

    The stream is independent of posterior/noise/timestep random draws. A batch
    may straddle two passes; no examples are dropped at pass boundaries.
    """
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Empty training set')
        self.size, self.seed = size, seed
        self.epoch, self.permutation = None, None

    def index(self, microbatch):
        epoch, position = divmod(microbatch, self.size)
        if self.epoch != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.permutation = torch.randperm(self.size, generator=generator).tolist()
            self.epoch = epoch
        return self.permutation[position]
