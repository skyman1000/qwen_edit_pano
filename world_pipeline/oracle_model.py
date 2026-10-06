"""Object-only oracle feature contract and Edit target-only residual attention.

No GT IDs, source flags, room state, or Observer predictions become model features.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import validate_graph

FEATURE_VERSION = 'objects_local_aabb_masked_v1'


def tensorize(graph, max_objects=128, condition='full'):
    validate_graph(graph)
    if condition not in ('full', 'observed', 'constant'):
        raise ValueError('Unknown condition')
    objects = graph['objects']
    if condition == 'observed':
        objects = [o for o in objects if o['local_evidence']=='visible']
    if condition == 'constant':
        objects = []  # Constant token count/mask; no sample object-count leakage.
    if len(objects) > max_objects:
        raise ValueError(f'{len(objects)} objects exceeds {max_objects}; never silently truncate')
    # Slot zero is an always-valid learned empty/global token, even for empty scenes.
    category=torch.zeros(1,max_objects+1,dtype=torch.long)
    geometry=torch.zeros(1,max_objects+1,8)
    mask=torch.zeros(1,max_objects+1,dtype=torch.bool); mask[:,0]=True
    for index,o in enumerate(objects,1):
        category[0,index]=o['category_id']['value'] or 0
        for offset,key in ((0,'center_local_m'),(3,'size_aabb_local_m')):
            field=o[key]
            if field['valid']:
                geometry[0,index,offset:offset+3]=torch.tensor(field['value'])
                geometry[0,index,6+offset//3]=1
        mask[0,index]=True
    return category,geometry,mask


class WorldEncoder(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.category=nn.Embedding(42,width)
        self.geometry=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
        self.empty=nn.Parameter(torch.randn(1,1,width)*.02)

    def forward(self,category,geometry,mask):
        tokens=self.norm(self.category(category)+self.geometry(geometry.float()))
        tokens=torch.cat([self.empty.expand(tokens.shape[0],-1,-1),tokens[:,1:]],1)
        return tokens,mask


class TargetAttention(nn.Module):
    def __init__(self,hidden,width,heads=8):
        super().__init__()
        if width%heads:
            raise ValueError('width must divide heads')
        self.heads=heads
        self.norm=nn.LayerNorm(hidden)
        self.world_norm=nn.LayerNorm(width)
        self.q=nn.Linear(hidden,width,bias=False)
        self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.out=nn.Linear(width,hidden,bias=False)
        self.gate=nn.Parameter(torch.zeros(()))
        self.diagnostics={}

    def forward(self,hidden,tokens,mask,boundary):
        if not 0 < boundary < hidden.shape[1]:
            raise ValueError('Expected target then nonempty reference tokens')
        target=hidden[:,:boundary]
        b,n,_=target.shape; m=tokens.shape[1]
        world=self.world_norm(tokens.float())
        def split(x):
            return x.reshape(b,-1,self.heads,x.shape[-1]//self.heads).transpose(1,2)
        q,k,v=split(self.q(self.norm(target.float()))),split(self.k(world)),split(self.v(world))
        attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
        update=self.out(attended.transpose(1,2).reshape(b,n,-1)).float()
        scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
        delta=torch.tanh(self.gate)*scale*update
        changed=(target.float()+delta).to(target.dtype)
        with torch.no_grad():
            self.diagnostics=dict(gate=float(torch.tanh(self.gate)),
                effective_delta_rms=float((changed.float()-target.float()).square().mean().sqrt()))
        return torch.cat([changed,hidden[:,boundary:]],1)


class WorldBranch(nn.Module):
    def __init__(self,hidden,blocks=(20,40,59),width=256):
        super().__init__()
        self.encoder=WorldEncoder(width)
        self.branches=nn.ModuleDict({str(i):TargetAttention(hidden,width) for i in blocks})
        self.current=None
        self.boundary=None

    def set_world(self,tensors):
        self.current=None if tensors is None else self.encoder(*tensors)

    def attach(self,transformer):
        if getattr(transformer,'_oracle_world_attached',False):
            raise ValueError('World already attached')
        if any(int(i)>=len(transformer.transformer_blocks) for i in self.branches):
            raise ValueError('Invalid block index')
        transformer._oracle_world_attached=True
        # Install AFTER circular pre-hook: shapes now include padded target columns.
        def before(module,args,kwargs):
            shapes=kwargs['img_shapes']
            if any(s!=shapes[0] for s in shapes) or len(shapes[0])<2:
                raise ValueError('Expected uniform target/reference shapes')
            self.boundary=math.prod(shapes[0][0])
            if kwargs['hidden_states'].shape[1] != sum(math.prod(s) for s in shapes[0]):
                raise ValueError('World target boundary mismatch')
        transformer.register_forward_pre_hook(before,with_kwargs=True)
        selected=set()
        for key,branch in self.branches.items():
            block=transformer.transformer_blocks[int(key)]
            original=block.forward
            def wrapped(*args,_original=original,_branch=branch,oracle_state=None,**kwargs):
                text,hidden=_original(*args,**kwargs)
                state=oracle_state
                if state is None and self.current is not None:
                    state=(*self.current,self.boundary)
                if state is not None:
                    hidden=_branch(hidden,*state)
                return text,hidden
            block.forward=wrapped
            selected.add(block)
        original_checkpoint=getattr(transformer,'_gradient_checkpointing_func',None)
        if original_checkpoint is not None:
            def with_world(function,*inputs,**kwargs):
                if function not in selected or self.current is None:
                    return original_checkpoint(function,*inputs,**kwargs)
                tokens,mask=self.current
                boundary=self.boundary
                def run(*packed):
                    return function(*packed[:-2],oracle_state=(packed[-2],packed[-1],boundary))
                return checkpoint(run,*inputs,tokens,mask,use_reentrant=False)
            transformer._gradient_checkpointing_func=with_world

    def diagnostics(self):
        return {k:b.diagnostics for k,b in self.branches.items()}
