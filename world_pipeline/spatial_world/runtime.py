"""Keep legacy attention, add zero-gated state-routed local residuals.

Checkpoint recomputation captures tokens AND routing as immutable state. No
branch reads a mutable latest sample during backward. No spatial attention bias.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ..id_binding_world.runtime import IDEncoder, BindingAttention, VERSION as LEGACY_VERSION
from ..object_world.runtime import object_region_loss
from ..oracle_memory import training_autocast
from .projection import erp_support

VERSION='polished_single_face_id_region_spatial_world_v1'


class LocalResidual(nn.Module):
    def __init__(self,hidden,width):
        super().__init__()
        self.norm=nn.LayerNorm(hidden);self.object_norm=nn.LayerNorm(width)
        self.query=nn.Linear(hidden,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.out=nn.Linear(width,hidden,bias=False)
        self.gate=nn.Parameter(torch.zeros(()));self.diagnostics={}

    def forward(self,target,tokens,routing):
        with training_autocast(target.device.type):
            value=self.value(self.object_norm(tokens.float()))
            routed=torch.bmm(routing.to(value.dtype),value)
            query=self.query(self.norm(target.float())).sigmoid()
            update=self.out(routed*query).float()
            scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
            delta=torch.tanh(self.gate)*scale*update
            with torch.no_grad():
                self.diagnostics=dict(gate=float(self.gate.tanh()),delta_rms=float(delta.square().mean().sqrt()),
                    relative_delta_rms=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-12)))
            return delta


class CombinedAttention(BindingAttention):
    def __init__(self,hidden,width):
        super().__init__(hidden,width)
        self.local=LocalResidual(hidden,width)

    def forward(self,hidden,tokens,mask,boundary,routing=None):
        changed,aux=super().forward(hidden,tokens,mask,boundary)
        if routing is not None:
            legacy_delta=self.diagnostics['effective_delta_rms']
            delta=self.local(hidden[:,:boundary],tokens,routing)
            target=(changed[:,:boundary].float()+delta).to(changed.dtype)
            changed=torch.cat((target,changed[:,boundary:]),1)
            with torch.no_grad():
                # Keep the existing recorder truthful for the combined output,
                # without allocating another full FP32 hidden-sized difference.
                mass=target.new_zeros((),dtype=torch.float32)
                for start in range(0,boundary,512):
                    mass+=(target[:,start:start+512].float()-hidden[:,start:min(start+512,boundary)].float()).square().sum()
                self.diagnostics=dict(self.diagnostics,legacy_effective_delta_rms=legacy_delta,
                    effective_delta_rms=float((mass/target.numel()).sqrt()))
        self.completed_diagnostics=dict(self.diagnostics)
        return changed,aux


class WorldBranch(nn.Module):
    def __init__(self,hidden,blocks=(20,40,59),width=256,height=1024,padding=1,enabled=True):
        super().__init__()
        if height%16 or height<16:raise ValueError('Height must divide the packed VAE scale')
        self.encoder=IDEncoder(width)
        self.branches=nn.ModuleDict({str(i):CombinedAttention(hidden,width) for i in blocks})
        self.height=height;self.padding=padding;self.enabled=enabled
        self.current=None;self.boundary=None;self.routing_meta={}
        if not enabled:
            for branch in self.branches.values():branch.local.requires_grad_(False)

    def set_world(self,tensors):
        if tensors is None:self.current=None;self.routing_meta={};return
        with training_autocast(tensors[0].device.type):tokens,mask=self.encoder(*tensors)
        routing=None
        if self.enabled:
            routing,self.routing_meta=erp_support(tensors[1],mask,self.height//16,self.height//8,self.padding)
        self.current=(tokens,mask,routing)

    def clear_context(self):self.current=None;self.routing_meta={}

    def set_supervision(self,supervision):
        if supervision is not None:raise ValueError('This experiment does not add attention CE')

    def diagnostics(self):return {k:getattr(b,'completed_diagnostics',b.diagnostics) for k,b in self.branches.items()}

    def spatial_diagnostics(self):return {k:b.local.diagnostics for k,b in self.branches.items()}

    def attach(self,transformer):
        if getattr(transformer,'_oracle_world_attached',False):raise ValueError('World already attached')
        if max(map(int,self.branches))>=len(transformer.transformer_blocks):raise ValueError('Invalid blocks')
        transformer._oracle_world_attached=True
        def before(module,args,kwargs):
            shapes=kwargs['img_shapes']
            expected=(1,self.height//16,self.height//8+2*self.padding)
            if any(s!=shapes[0] for s in shapes) or tuple(shapes[0][0])!=expected or len(shapes[0])<2:
                raise ValueError('Spatial ERP grid/padding differs from checkpoint')
            self.boundary=math.prod(expected)
            if kwargs['hidden_states'].shape[1]!=sum(math.prod(s) for s in shapes[0]):raise ValueError('Target/reference boundary differs')
        transformer.register_forward_pre_hook(before,with_kwargs=True)
        selected={}
        for key,branch in self.branches.items():
            block=transformer.transformer_blocks[int(key)];original=block.forward
            def compute(args,kwargs,state,_original=original,_branch=branch):
                text,hidden=_original(*args,**kwargs)
                if state is not None:hidden,_=_branch(hidden,*state)
                return text,hidden
            def context():
                if self.current is None:return None
                tokens,mask,routing=self.current
                return tokens,mask,self.boundary,routing
            def wrapped(*args,_compute=compute,_context=context,**kwargs):return _compute(args,kwargs,_context())
            block.forward=wrapped;selected[block]=(compute,context)
        native=getattr(transformer,'_gradient_checkpointing_func',None)
        if native is not None:
            def with_world(function,*inputs,**kwargs):
                if function not in selected:return native(function,*inputs,**kwargs)
                compute,context=selected[function];state=context()
                def run(*packed):return compute(packed,kwargs,state)
                return checkpoint(run,*inputs,use_reentrant=False)
            transformer._gradient_checkpointing_func=with_world


def load_world(world,state,config):
    """Explicitly distinguish legacy warm start from strict new-version resume."""
    if config['version']==LEGACY_VERSION:
        if config.get('group')!='A':raise ValueError('Warm start only the selected legacy A group')
        result=world.load_state_dict(state,strict=False)
        expected={k for k in world.state_dict() if '.local.' in k}
        if result.unexpected_keys or set(result.missing_keys)!=expected:raise ValueError('Legacy warm-start parameter mismatch')
        return 'legacy_shared_weights_only'
    if config['version']!=VERSION:raise ValueError('Wrong World checkpoint version')
    world.load_state_dict(state,strict=True)
    return 'strict_spatial_checkpoint'
