"""ID+Fourier World; optional train-only instance coverage supervision.

Pure checkpoint functions return auxiliary scalars as explicit outputs. Collection
happens outside checkpoint recomputation, never in a mutable observer hook.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..fourier_world.runtime import GeometryFeatures
from ..oracle_memory import training_autocast
from ..object_world.runtime import object_region_loss
from .supervision import attention_coverage_loss

VERSION='polished_single_face_id_region_binding_world_v1'


class IDEncoder(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.category=nn.Embedding(42,width)
        self.geometry=nn.Sequential(GeometryFeatures(),nn.Linear(392,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
        self.empty=nn.Parameter(torch.randn(1,1,width)*.02)

    def forward(self,category,geometry,mask):
        categorical=self.category(category);spatial=self.geometry(geometry)
        with torch.no_grad():
            valid=mask[:,1:]
            c=categorical[:,1:][valid].float().square().mean().sqrt() if valid.any() else categorical.new_zeros(())
            g=spatial[:,1:][valid].float().square().mean().sqrt() if valid.any() else spatial.new_zeros(())
            self.feature_diagnostics=dict(category_rms=float(c),geometry_rms=float(g),category_to_geometry=float(c/g.clamp_min(1e-12)))
        tokens=self.norm(categorical+spatial)
        return torch.cat((self.empty.expand(tokens.shape[0],-1,-1),tokens[:,1:]),1),mask


class BindingAttention(nn.Module):
    """Same forward/parameter keys as original TargetAttention; no masked routing."""
    def __init__(self,hidden,width,heads=8):
        super().__init__()
        if width%heads:raise ValueError('Width must divide heads')
        self.heads=heads
        self.norm=nn.LayerNorm(hidden);self.world_norm=nn.LayerNorm(width)
        self.q=nn.Linear(hidden,width,bias=False);self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False);self.out=nn.Linear(width,hidden,bias=False)
        self.gate=nn.Parameter(torch.zeros(()));self.diagnostics={}

    def forward(self,hidden,tokens,mask,boundary,supervision=None):
        if not 0<boundary<hidden.shape[1]:raise ValueError('Invalid target/reference boundary')
        with training_autocast(hidden.device.type):
            target=hidden[:,:boundary];b,n,_=target.shape
            world=self.world_norm(tokens.float())
            def heads(x):return x.reshape(b,-1,self.heads,x.shape[-1]//self.heads).transpose(1,2)
            q,k,v=map(heads,(self.q(self.norm(target.float())),self.k(world),self.v(world)))
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
            update=self.out(attended.transpose(1,2).reshape(b,n,-1)).float()
            scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
            changed=(target.float()+torch.tanh(self.gate)*scale*update).to(target.dtype)
            # Auxiliary reads the SAME Q/K used by the normal SDPA forward.
            loss=attention_coverage_loss(q,k,mask,supervision) if supervision is not None else update.new_zeros(())
            with torch.no_grad():
                self.diagnostics=dict(gate=float(torch.tanh(self.gate)),
                    effective_delta_rms=float((changed.float()-target.float()).square().mean().sqrt()))
            return torch.cat((changed,hidden[:,boundary:]),1),loss


class WorldBranch(nn.Module):
    def __init__(self,hidden,blocks=(20,40,59),width=256,binding_block=40):
        super().__init__()
        if binding_block not in blocks:raise ValueError('Binding layer must be injected')
        self.encoder=IDEncoder(width)
        self.branches=nn.ModuleDict({str(i):BindingAttention(hidden,width) for i in blocks})
        self.binding_block=binding_block;self.current=None;self.boundary=None
        self.supervision=None;self._losses=[]

    def set_world(self,tensors):
        if tensors is None:self.current=None
        else:
            with training_autocast(tensors[0].device.type):self.current=self.encoder(*tensors)

    def set_supervision(self,supervision):self.supervision=supervision

    def take_binding_loss(self):
        if len(self._losses)!=1:raise ValueError(f'Expected one binding scalar, got {len(self._losses)}')
        loss=self._losses[0];self._losses=[]
        return loss

    def clear_context(self):
        self.current=None;self.supervision=None;self._losses=[]

    def diagnostics(self):return {k:b.diagnostics for k,b in self.branches.items()}

    def attach(self,transformer):
        if getattr(transformer,'_oracle_world_attached',False):raise ValueError('World already attached')
        if max(map(int,self.branches))>=len(transformer.transformer_blocks):raise ValueError('Invalid blocks')
        transformer._oracle_world_attached=True
        def before(module,args,kwargs):
            shapes=kwargs['img_shapes']
            if any(s!=shapes[0] for s in shapes) or len(shapes[0])<2:raise ValueError('Expected target/reference shapes')
            self.boundary=math.prod(shapes[0][0]);self._losses=[]
            if kwargs['hidden_states'].shape[1]!=sum(math.prod(s) for s in shapes[0]):raise ValueError('Boundary mismatch')
        transformer.register_forward_pre_hook(before,with_kwargs=True)
        selected={}
        for key,branch in self.branches.items():
            block=transformer.transformer_blocks[int(key)];original=block.forward
            # This function is pure with respect to all runtime World/label state.
            def compute(args,kwargs,state,labels,_original=original,_branch=branch):
                text,hidden=_original(*args,**kwargs)
                if state is None:return text,hidden,hidden.new_zeros((),dtype=torch.float32)
                hidden,aux=_branch(hidden,*state,labels)
                return text,hidden,aux
            def context(_key=key):
                state=None if self.current is None else (*self.current,self.boundary)
                labels=self.supervision if int(_key)==self.binding_block and state is not None else None
                return state,labels
            def wrapped(*args,_compute=compute,_context=context,**kwargs):
                state,labels=_context();text,hidden,aux=_compute(args,kwargs,state,labels)
                if labels is not None:self._losses.append(aux)
                return text,hidden
            block.forward=wrapped;selected[block]=(compute,context)
        native_checkpoint=getattr(transformer,'_gradient_checkpointing_func',None)
        if native_checkpoint is not None:
            def with_world(function,*inputs,**kwargs):
                if function not in selected:return native_checkpoint(function,*inputs,**kwargs)
                compute,context=selected[function];state,labels=context()
                # Immutable tuples capture this microbatch, surviving clear_context.
                # Collection is OUTSIDE the pure function called during backward.
                def run(*packed):return compute(packed,kwargs,state,labels)
                text,hidden,aux=checkpoint(run,*inputs,use_reentrant=False)
                if labels is not None:self._losses.append(aux)
                return text,hidden
            transformer._gradient_checkpointing_func=with_world
