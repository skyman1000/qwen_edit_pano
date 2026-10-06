"""Precision and activation storage policies; no weight/dataset/precision changes."""
from contextlib import nullcontext
import torch


def training_autocast(device_type):
    # Shared FP32 world weights are reused across blocks and smoke no-grad calls.
    # Cached BF16 casts can have a different autograd history on recomputation.
    return torch.autocast(device_type,dtype=torch.bfloat16,cache_enabled=False)


class ActivationStorage:
    """Offload differentiable non-leaf saved activations, never frozen weights.

    Copies are synchronous and preserve dtype. Test device='cpu' exercises hooks
    without a GPU, but cannot measure VRAM savings or CUDA transfer behavior.
    """
    def __init__(self,enabled=True,device_type='cuda'):
        self.enabled=enabled
        self.device_type=device_type
        self.count=0
        self.bytes=0

    def pack(self,tensor):
        if tensor.device.type==self.device_type and tensor.requires_grad and not tensor.is_leaf:
            self.count+=1;self.bytes+=tensor.numel()*tensor.element_size()
            return (tensor.device,tensor.detach().to(device='cpu',copy=True))
        return tensor

    @staticmethod
    def unpack(saved):
        if isinstance(saved,tuple):
            device,tensor=saved
            return tensor.to(device=device)
        return saved

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack) if self.enabled else nullcontext()
