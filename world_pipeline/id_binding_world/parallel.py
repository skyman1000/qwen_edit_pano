"""Replicated World gradients and rank-local samples; frozen base is not reduced.

World is invoked through checkpointed Transformer hooks rather than one module
forward, so synchronization is explicit after local gradient accumulation.
"""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def micro_indices(step, accumulation, rank=0, replicas=1):
    if accumulation < 1 or replicas < 1 or accumulation % replicas or not 0 <= rank < replicas:
        raise ValueError('Global accumulation must divide evenly over replicas')
    local = accumulation // replicas
    return range(step * accumulation + rank * local, step * accumulation + (rank + 1) * local)


def seed_microbatch(seed, index):
    from ..scaled_world.runtime import seed_step
    seed_step(seed * 1000003, index)


def average_gradients(module):
    """Average local-mean gradients, including an explicit unused-grad check."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    params = list(module.parameters())
    present = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.int32)
    dist.all_reduce(present)
    count = dist.get_world_size()
    if ((present != 0) & (present != count)).any():
        raise RuntimeError('Ranks disagree on which World parameters received gradients')
    for parameter in params:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(count)


def check_replicas(module):
    """One-time raw tensor equality check; no content hashes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    mismatch = torch.zeros((), device=next(module.parameters()).device, dtype=torch.int32)
    for parameter in module.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        mismatch.copy_(torch.maximum(mismatch, torch.ne(parameter.detach(), reference).any().to(torch.int32)))
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    if mismatch.item():
        raise RuntimeError('World parameters differ across ranks')


class Parallel:
    def __init__(self):
        self.size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.device = torch.device('cuda', self.local_rank)
        torch.cuda.set_device(self.device)
        if self.size > 1:
            dist.init_process_group('nccl', timeout=timedelta(minutes=20))

    @property
    def main(self):
        return self.rank == 0

    def barrier(self):
        if self.size > 1:
            dist.barrier()

    def mean(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if self.size > 1:
            dist.all_reduce(tensor)
            tensor.div_(self.size)
        return float(tensor)

    def any(self, value):
        tensor = torch.tensor(int(bool(value)), device=self.device, dtype=torch.int32)
        if self.size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def gather(self, value):
        if self.size == 1:
            return [value]
        result = [None] * self.size
        dist.all_gather_object(result, value)
        return result

    def close(self):
        if self.size > 1:
            dist.destroy_process_group()
