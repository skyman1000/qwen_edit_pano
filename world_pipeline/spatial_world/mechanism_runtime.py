"""Inference-only local ablations; v1 training/runtime/checkpoints stay unchanged."""
import torch
from torch.nn import functional as F
from ..oracle_memory import training_autocast
from .runtime import WorldBranch,CombinedAttention

MODES=('native_v1','measured_v1','local_off','single_cast')


class MeasuredAttention(CombinedAttention):
    def __init__(self,hidden,width):
        super().__init__(hidden,width)
        self.measure_mode='native_v1';self.measurement={}

    def forward(self,hidden,tokens,mask,boundary,routing=None):
        if torch.is_grad_enabled():raise RuntimeError('Mechanism runtime is forward-only; do not use for training')
        self.measurement={}
        if self.measure_mode in ('native_v1','local_off'):
            return super().forward(hidden,tokens,mask,boundary,routing)
        if self.measure_mode not in ('measured_v1','single_cast') or routing is None:
            raise ValueError('Measured mode needs active local routing')
        if not 0<boundary<hidden.shape[1]:raise ValueError('Invalid target/reference boundary')
        # Exact same projections, shapes and operation order as BindingAttention.
        # measured_v1 is checked against the unmodified native_v1 prediction.
        with training_autocast(hidden.device.type):
            target=hidden[:,:boundary];b,n,_=target.shape
            world=self.world_norm(tokens.float())
            def heads(x):return x.reshape(b,-1,self.heads,x.shape[-1]//self.heads).transpose(1,2)
            q,k,v=map(heads,(self.q(self.norm(target.float())),self.k(world),self.v(world)))
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
            update=self.out(attended.transpose(1,2).reshape(b,n,-1)).float()
            scale=target.detach().float().square().mean(-1,keepdim=True).sqrt()
            legacy_fp32=target.float()+torch.tanh(self.gate)*scale*update
            legacy=legacy_fp32.to(target.dtype)
            delta=self.local(target,tokens,routing)
            if self.measure_mode=='measured_v1':changed=(legacy.float()+delta).to(target.dtype)
            else:changed=(legacy_fp32+delta).to(target.dtype)
            actual_mass=target.new_zeros((),dtype=torch.float32)
            nominal_mass=actual_mass.clone();target_mass=actual_mass.clone()
            changed_elements=torch.zeros((),device=target.device,dtype=torch.int64)
            supported_elements=torch.zeros_like(changed_elements);changed_supported=torch.zeros_like(changed_elements)
            support=(routing.sum(-1)>0)
            for start in range(0,boundary,512):
                end=min(start+512,boundary)
                actual_mass+=(changed[:,start:end].float()-legacy[:,start:end].float()).square().sum()
                nominal_mass+=delta[:,start:end].float().square().sum()
                target_mass+=target[:,start:end].float().square().sum()
                different=changed[:,start:end]!=legacy[:,start:end]
                supported=support[:,start:end,None].expand_as(different)
                changed_elements+=different.sum();supported_elements+=supported.sum()
                changed_supported+=(different&supported).sum()
            count=target.numel();nominal=(nominal_mass/count).sqrt();actual=(actual_mass/count).sqrt()
            self.measurement=dict(hidden_dtype=str(target.dtype),mode=self.measure_mode,
                nominal_delta_rms=float(nominal),actual_delta_rms=float(actual),
                nominal_relative_rms=float((nominal_mass/target_mass.clamp_min(1e-30)).sqrt()),
                actual_relative_rms=float((actual_mass/target_mass.clamp_min(1e-30)).sqrt()),
                actual_to_nominal_rms=float(actual/nominal.clamp_min(1e-30)),
                changed_element_fraction=float(changed_elements/count),
                changed_supported_element_fraction=float(changed_supported/supported_elements.clamp_min(1)),
                supported_element_fraction=float(supported_elements/count),
                note='Actual change is relative to rounded legacy output; RMS ratio may exceed 1 due to quantized jumps. Not semantic accuracy.')
            return torch.cat((changed,hidden[:,boundary:]),1),changed.new_zeros((),dtype=torch.float32)


class MechanismWorld(WorldBranch):
    def __init__(self,hidden,blocks=(20,40,59),width=256,height=1024,padding=1):
        super().__init__(hidden,blocks,width,height,padding)
        self.branches=torch.nn.ModuleDict({str(i):MeasuredAttention(hidden,width) for i in blocks})

    def set_measure_mode(self,mode):
        if mode not in MODES:raise ValueError('Unknown mechanism mode')
        self.enabled=mode!='local_off'
        for branch in self.branches.values():branch.measure_mode=mode;branch.measurement={}

    def measurements(self):return {key:dict(branch.measurement) for key,branch in self.branches.items() if branch.measurement}
