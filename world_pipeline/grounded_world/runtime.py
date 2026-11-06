"""Semantic + geometry tokens and learned region-restricted attention.

The null key/value is exactly zero, not an inherited global-style token. Mesh
labels only read probabilities for an auxiliary loss; they never mask attention.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..fourier_world.runtime import GeometryFeatures
from ..oracle_memory import training_autocast
from ..spatial_world.projection import erp_support

VERSION='polished_single_face_semantic_grounded_world_v2'


class ObjectEncoder(nn.Module):
    def __init__(self,features,width):
        super().__init__()
        if features.ndim!=2 or features.shape[0]!=42 or not torch.isfinite(features).all():
            raise ValueError('Expected finite cached category-name features')
        self.register_buffer('class_features',F.normalize(features.detach().float(),dim=-1))
        self.semantic=nn.Sequential(nn.Linear(features.shape[1],width,bias=False),nn.LayerNorm(width))
        self.geometry=nn.Sequential(GeometryFeatures(),nn.Linear(392,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.fusion=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.LayerNorm(width))

    def forward(self,category,geometry,mask):
        semantic=self.semantic(self.class_features[category]);spatial=self.geometry(geometry)
        value=self.fusion(torch.cat((semantic,spatial),-1))
        # No trainable empty/global value. Object slot numbers are not features.
        value=torch.cat((value.new_zeros(value.shape[0],1,value.shape[-1]),value[:,1:]),1)
        with torch.no_grad():
            chosen=mask[:,1:]
            self.feature_diagnostics=dict(semantic_rms=float(semantic[:,1:][chosen].float().square().mean().sqrt()) if chosen.any() else 0.,
                geometry_rms=float(spatial[:,1:][chosen].float().square().mean().sqrt()) if chosen.any() else 0.)
        return value,mask


def probabilities_loss(q,k,allowed,supervision):
    """FP32 readout of the SAME Q/K and allowed candidates as forward SDPA."""
    query_indices,pair_queries,slots,weights=supervision
    if not len(slots):return q[...,:1,:1].float().sum()*0+k[...,:1,:1].float().sum()*0
    if (q.shape[0]!=1 or query_indices.ndim!=1 or pair_queries.shape!=slots.shape or weights.shape!=slots.shape
            or query_indices.min()<0 or query_indices.max()>=q.shape[-2] or pair_queries.min()<0
            or pair_queries.max()>=len(query_indices) or slots.min()<1 or slots.max()>=k.shape[-2]
            or not torch.isfinite(weights).all() or (weights<0).any()
            or not torch.isclose(weights.sum(),weights.new_tensor(1.))):
        raise ValueError('Invalid binding labels')
    if not allowed[0,query_indices[pair_queries],slots].all():
        raise ValueError('Binding target is unreachable in the actual forward; do not train against -inf')
    with torch.autocast(device_type=q.device.type,enabled=False):
        logits=(q[0,:,query_indices].float()@k[0].float().transpose(-1,-2))/math.sqrt(q.shape[-1])
        logits=logits.masked_fill(~allowed[0,query_indices][None],-torch.inf)
        logp=logits.log_softmax(-1)
        # Gather only finite, allowed targets before aggregation across heads.
        target=logp[:,pair_queries,slots]
        return -((torch.logsumexp(target,0)-math.log(q.shape[1]))*weights.float()).sum()


class ObjectAttention(nn.Module):
    def __init__(self,hidden,width,heads=8):
        super().__init__()
        if width%heads:raise ValueError('Attention width must divide heads')
        self.heads=heads;self.norm=nn.LayerNorm(hidden);self.world_norm=nn.LayerNorm(width)
        self.q=nn.Linear(hidden,width,bias=False);self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False);self.out=nn.Linear(width,hidden,bias=False)
        self.gate=nn.Parameter(torch.zeros(()));self.diagnostics={}

    def forward(self,hidden,tokens,mask,boundary,candidates,supervision=None):
        if not 0<boundary<hidden.shape[1] or candidates.shape!=(hidden.shape[0],boundary,tokens.shape[1]):
            raise ValueError('Invalid target/reference or candidate dimensions')
        with training_autocast(hidden.device.type):
            target=hidden[:,:boundary];b,n,_=target.shape
            def heads(x):return x.reshape(b,-1,self.heads,x.shape[-1]//self.heads).transpose(1,2)
            world=self.world_norm(tokens.float())
            q=heads(self.q(self.norm(target.float())));k=self.k(world);v=self.v(world)
            # LayerNorm bias must not turn the null into a learned constant value.
            k=torch.cat((k.new_zeros(b,1,k.shape[-1]),k[:,1:]),1)
            v=torch.cat((v.new_zeros(b,1,v.shape[-1]),v[:,1:]),1)
            k,v=heads(k),heads(v)
            allowed=candidates&mask[:,None,:];allowed=allowed.clone();allowed[:,:,0]=True
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None],dropout_p=0.)
            update=self.out(attended.transpose(1,2).reshape(b,n,-1)).float()
            scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
            delta=self.gate.tanh()*scale*update
            changed=(target.float()+delta).to(target.dtype)
            auxiliary=probabilities_loss(q,k,allowed,supervision) if supervision is not None else delta.new_zeros(())
            with torch.no_grad():
                mass=changed.new_zeros((),dtype=torch.float32)
                for start in range(0,n,512):mass+=(changed[:,start:start+512].float()-target[:,start:start+512].float()).square().sum()
                self.diagnostics=dict(gate=float(self.gate.tanh()),effective_delta_rms=float((mass/target.numel()).sqrt()),
                    nominal_relative_delta_rms=float(delta.square().mean().sqrt()/target.float().square().mean().sqrt().clamp_min(1e-12)),
                    candidate_query_fraction=float(candidates[:,:,1:].any(-1).float().mean()))
            return torch.cat((changed,hidden[:,boundary:]),1),auxiliary


class WorldBranch(nn.Module):
    def __init__(self,hidden,features,blocks=(20,40,59),width=256,height=1024,padding=1,binding_block=40):
        super().__init__()
        if height<16 or height%16 or binding_block not in blocks:raise ValueError('Invalid grid/binding block')
        self.encoder=ObjectEncoder(features,width)
        self.branches=nn.ModuleDict({str(i):ObjectAttention(hidden,width) for i in blocks})
        self.height=height;self.padding=padding;self.binding_block=binding_block
        self.current=None;self.boundary=None;self.supervision=None;self._losses=[];self.routing_meta={}

    def set_world(self,tensors):
        if tensors is None:self.current=None;self.routing_meta={};return
        category,geometry,mask=tensors
        # Geometry has no learned gradients. CPU projection is shared with label
        # preparation and inference, avoiding CPU/CUDA boundary disagreements.
        route,meta=erp_support(geometry.detach().cpu(),mask.detach().cpu(),self.height//16,self.height//8,self.padding)
        candidates=(route>0).to(category.device)
        with training_autocast(category.device.type):tokens,mask=self.encoder(*tensors)
        self.current=(tokens,mask,candidates);self.routing_meta=meta

    def set_supervision(self,labels):self.supervision=labels
    def take_binding_loss(self):
        if len(self._losses)!=1:raise ValueError('Expected one explicit binding scalar')
        loss=self._losses[0];self._losses=[];return loss
    def clear_context(self):self.current=None;self.supervision=None;self._losses=[];self.routing_meta={}
    def diagnostics(self):return {k:b.diagnostics for k,b in self.branches.items()}

    def attach(self,transformer):
        if getattr(transformer,'_oracle_world_attached',False):raise ValueError('A World adapter is already attached')
        if max(map(int,self.branches))>=len(transformer.transformer_blocks):raise ValueError('Invalid injection blocks')
        transformer._oracle_world_attached=True
        def before(module,args,kwargs):
            shapes=kwargs['img_shapes'];expected=(1,self.height//16,self.height//8+2*self.padding)
            if any(s!=shapes[0] for s in shapes) or tuple(shapes[0][0])!=expected or len(shapes[0])<2:
                raise ValueError('Expected the exact padded target ERP and reference')
            self.boundary=math.prod(expected);self._losses=[]
            if kwargs['hidden_states'].shape[1]!=sum(math.prod(s) for s in shapes[0]):raise ValueError('Target/reference mismatch')
        transformer.register_forward_pre_hook(before,with_kwargs=True);selected={}
        for key,branch in self.branches.items():
            block=transformer.transformer_blocks[int(key)];original=block.forward
            def compute(args,kwargs,state,labels,_original=original,_branch=branch):
                text,hidden=_original(*args,**kwargs)
                if state is None:return text,hidden,hidden.new_zeros((),dtype=torch.float32)
                hidden,aux=_branch(hidden,*state,labels)
                return text,hidden,aux
            def context(_key=key):
                state=None if self.current is None else (self.current[0],self.current[1],self.boundary,self.current[2])
                labels=self.supervision if int(_key)==self.binding_block and state is not None else None
                return state,labels
            def wrapped(*args,_compute=compute,_context=context,**kwargs):
                state,labels=_context();text,hidden,aux=_compute(args,kwargs,state,labels)
                if labels is not None:self._losses.append(aux)
                return text,hidden
            block.forward=wrapped;selected[block]=(compute,context)
        native=getattr(transformer,'_gradient_checkpointing_func',None)
        if native is not None:
            def with_world(function,*inputs,**kwargs):
                if function not in selected:return native(function,*inputs,**kwargs)
                compute,context=selected[function];state,labels=context()
                def run(*packed):return compute(packed,kwargs,state,labels)
                text,hidden,aux=checkpoint(run,*inputs,use_reentrant=False)
                if labels is not None:self._losses.append(aux)
                return text,hidden
            transformer._gradient_checkpointing_func=with_world
