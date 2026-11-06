"""Optimization-only local calibration; the spatial v1 forward stays unchanged."""
import torch
from .runtime import VERSION,LEGACY_VERSION

POLICY=dict(version='local_only_calibration_v1',trainable='branches.*.local.*',
    frozen='encoder and legacy attention, including legacy gates',
    adam_eps=1e-8,initial_optimizer='fresh; never import the joint-training optimizer',
    forward='unchanged spatial v1, including two BF16 writebacks',
    objective='unchanged A objective, region 0.25, binding 0',
    purpose='short optimization/capacity experiment; not a new loss ablation')


def initialization(config,complete,local_only=False):
    """Return cumulative ancestry step and native sampling offset, not resume state."""
    if config['version']!=complete['version'] or config.get('group')!='A':
        raise ValueError('Incomplete or incompatible A initialization checkpoint')
    if not local_only:
        if config['version']!=LEGACY_VERSION:
            raise ValueError('Joint spatial/control training initializes from legacy A')
        return complete['step'],0
    if (config['version']!=VERSION or config.get('spatial_mode')!='spatial'
            or config.get('binding_weight')!=0 or 'optimization_policy' in config
            or config.get('smoke') or complete['step']<1):
        raise ValueError('Local calibration initializes from a completed, jointly trained spatial v1 checkpoint')
    return config['initial_step']+complete['step'],complete['step']


def configure(world):
    if not world.enabled:raise ValueError('Local calibration requires active spatial routing')
    world.requires_grad_(False)
    for branch in world.branches.values():branch.local.requires_grad_(True)
    names=[name for name,p in world.named_parameters() if p.requires_grad]
    if not names or any('.local.' not in name for name in names):
        raise AssertionError('Unexpected trainable parameters in local calibration')
    if any(p.dtype!=torch.float32 for p in world.parameters() if p.requires_grad):
        raise ValueError('Local calibration requires FP32 master weights for Adam eps 1e-8')
    return dict(trainable_names=names,trainable_parameters=sum(p.numel() for p in world.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in world.parameters() if not p.requires_grad))


def frozen_snapshot(world):
    return {name:value.detach().cpu().clone() for name,value in world.state_dict().items() if '.local.' not in name}


def check_frozen(world,reference):
    current=world.state_dict()
    if set(reference)!={name for name in current if '.local.' not in name}:
        raise AssertionError('Frozen parameter/buffer names changed')
    if any(not torch.equal(value,current[name].detach().cpu()) for name,value in reference.items()):
        raise AssertionError('Local calibration changed a frozen encoder/legacy tensor')
    return dict(checked_tensors=len(reference),all_frozen_tensors_exact=True)
