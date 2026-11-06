"""Read-only Legacy diagnostics; never used by the training forward path."""
import math
import torch
from torch.nn import functional as F

from ..oracle_memory import training_autocast

MODES=('off','global_only','objects_only','correct','yaw90','class_cycle','mean_objects','order_permute')


def condition_variant(tensors, mode, class_ids):
    if mode not in MODES:raise ValueError('Unknown diagnostic condition')
    if mode=='off':return None
    category,geometry,mask=(t.clone() for t in tensors)
    if category.shape[0]!=1 or not mask[:,1:].any():raise ValueError('Diagnostics require one nonempty scene')
    if mode=='global_only':mask[:,1:]=False
    elif mode=='objects_only':mask[:,0]=False
    elif mode=='yaw90':
        # Rigid 90-degree yaw of the STATE only. AABB x/z side lengths swap.
        # Permuting object rows would preserve the geometry set and is not a
        # reliable test of geometry use when category signals are weak.
        old=geometry.clone();cv=(old[...,6]>.5)&mask;sv=(old[...,7]>.5)&mask
        geometry[...,0]=torch.where(cv,old[...,2],old[...,0])
        geometry[...,2]=torch.where(cv,-old[...,0],old[...,2])
        geometry[...,3]=torch.where(sv,old[...,5],old[...,3])
        geometry[...,5]=torch.where(sv,old[...,3],old[...,5])
    elif mode=='class_cycle':
        ids=sorted(int(i) for i in class_ids)
        if len(ids)<2:raise ValueError('Need at least two task classes')
        source=category.clone()
        for i,cid in enumerate(ids):category[(source==cid)&mask]=ids[(i+1)%len(ids)]
    elif mode=='order_permute':
        order=torch.cat((torch.zeros(1,device=category.device,dtype=torch.long),
                         torch.arange(category.shape[1]-1,0,-1,device=category.device)))
        category=category[:,order];geometry=geometry[:,order];mask=mask[:,order]
    return category,geometry,mask


def set_condition(world,tensors,mode,class_ids):
    variant=condition_variant(tensors,mode,class_ids)
    world.set_world(variant)
    if mode=='mean_objects':
        tokens,mask=world.current;tokens=tokens.clone()
        valid=mask[:,1:]
        mean=tokens[:,1:][valid].mean(0)
        tokens[:,1:]=torch.where(valid[...,None],mean,tokens[:,1:])
        world.current=tokens,mask


def common_ratio(values):
    """Norm(mean token)/RMS token norm. Not an attention probability."""
    return float(values.float().mean(0).norm()/values.float().square().sum(-1).mean().sqrt().clamp_min(1e-12))


class AttentionProbe:
    """Observe a bounded subset of REAL image queries; never replace SDPA output."""
    def __init__(self,world):
        self.active=False;self.records={};self.query_indices=None;self.expected_pairs=[]
        self.handles=[]
        for key,branch in world.branches.items():
            self.handles.append(branch.register_forward_pre_hook(self._hook(key)))

    def reset(self,query_indices=None,expected_pairs=()):
        self.records={};self.query_indices=query_indices;self.expected_pairs=expected_pairs

    def _hook(self,key):
        @torch.no_grad()
        def collect(branch,args):
            if not self.active:return
            hidden,tokens,mask,boundary=args
            target=hidden[:,:boundary]
            if self.query_indices is None:
                qi=torch.linspace(0,boundary-1,min(boundary,64),device=hidden.device).long()
            else:qi=torch.as_tensor(self.query_indices,device=hidden.device,dtype=torch.long)
            if qi.max()>=boundary:raise ValueError('Diagnostic query index exceeds padded target boundary')
            with training_autocast(hidden.device.type):
                source=branch.world_norm(tokens.float())
                q=branch.q(branch.norm(target[:,qi].float()))
                k=branch.k(source);v=branch.v(source)
                def heads(t):return t.reshape(1,-1,branch.heads,t.shape[-1]//branch.heads).transpose(1,2)
                qh,kh,vh=map(heads,(q,k,v))
                logits=(qh.float()@kh.float().transpose(-1,-2))/math.sqrt(qh.shape[-1])
                logits=logits.masked_fill(~mask[:,None,None,:],-torch.inf)
                probabilities=logits.softmax(-1)
                attended=probabilities@vh.float()
                update=branch.out(attended.transpose(1,2).reshape(1,len(qi),-1).to(v.dtype)).float()[0]
            probs=probabilities[0].mean(0)
            count=int(mask.sum());objects=mask[0].clone();objects[0]=False
            entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)
            values=v[0,objects].float()
            rec=dict(valid_tokens=count,queries=len(qi),global_attention_mass=float(probs[:,0].mean()),
                entropy_of_head_mean_normalized=float(entropy.mean()/math.log(count)) if count>1 else 0.,
                update_query_variation_ratio=float((update-update.mean(0)).square().mean().sqrt()/update.square().mean().sqrt().clamp_min(1e-12)),
                object_value_common_ratio=common_ratio(values) if len(values) else None,
                raw_update_rms=float(update.square().mean().sqrt()))
            if self.expected_pairs:
                mass=torch.stack([probs[qpos,slot] for qpos,slot in self.expected_pairs])
                rec.update(gt_query_object_count=len(mass),gt_object_attention_mean=float(mass.mean()),
                    gt_object_attention_over_uniform=float(mass.mean()*count),
                    gt_object_top1_fraction=float(torch.tensor([int(probs[qpos].argmax())==slot for qpos,slot in self.expected_pairs]).float().mean()))
            self.records[key]=rec
        return collect


def compare_gradients(first,second):
    if set(first)!=set(second):raise ValueError('Checkpoint paths produced different gradient parameter sets')
    diff=sum(float((first[k].float()-second[k].float()).square().sum()) for k in first)
    power=sum(float(first[k].float().square().sum()) for k in first)
    return dict(relative_l2_difference=math.sqrt(diff/max(power,1e-30)),
                maximum_absolute_difference=max(float((first[k]-second[k]).abs().max()) for k in first),
                parameters_with_gradients=len(first))
