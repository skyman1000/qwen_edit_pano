"""Original ID A forward and parameter keys; no new spatial routing or features."""
import torch
from torch import nn
from torch.nn import functional as F
from ..oracle_memory import training_autocast
from ..id_binding_world.runtime import WorldBranch as OriginalWorld, BindingAttention
from .supervision import binding_terms


class RefinedAttention(BindingAttention):
    def forward(self,hidden,tokens,mask,boundary,supervision=None):
        if not 0<boundary<hidden.shape[1]:raise ValueError('Invalid target/reference boundary')
        with training_autocast(hidden.device.type):
            target=hidden[:,:boundary];b,n,_=target.shape;world=self.world_norm(tokens.float())
            def heads(x):return x.reshape(b,-1,self.heads,x.shape[-1]//self.heads).transpose(1,2)
            q,k,v=map(heads,(self.q(self.norm(target.float())),self.k(world),self.v(world)))
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
            update=self.out(attended.transpose(1,2).reshape(b,n,-1)).float()
            scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
            changed=(target.float()+torch.tanh(self.gate)*scale*update).to(target.dtype)
            aux=torch.stack(binding_terms(q,k,mask,supervision)) if supervision is not None else update.new_zeros(2)
            with torch.no_grad():
                self.diagnostics=dict(gate=float(torch.tanh(self.gate)),
                    effective_delta_rms=float((changed.float()-target.float()).square().mean().sqrt()))
            return torch.cat((changed,hidden[:,boundary:]),1),aux


class WorldBranch(OriginalWorld):
    def __init__(self,hidden,blocks=(20,40,59),width=256,binding_block=40):
        super().__init__(hidden,blocks,width,binding_block)
        self.branches=nn.ModuleDict({str(i):RefinedAttention(hidden,width) for i in blocks})
