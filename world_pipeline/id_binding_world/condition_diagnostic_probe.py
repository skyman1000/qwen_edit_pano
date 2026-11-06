"""Read ACTUAL projection outputs; hooks return None and never alter attention."""
import math

import torch


class ActualAttentionProbe:
    def __init__(self,world):
        self.active=False;self.spec=None;self.pending={};self.records={};self.handles=[]
        for key,branch in world.branches.items():
            for name in ('q','k','v','out'):
                self.handles.append(getattr(branch,name).register_forward_hook(self._projection(key,name)))
            self.handles.append(branch.register_forward_hook(self._finish(key)))

    def reset(self,spec,active=True):
        self.spec=spec;self.active=active;self.pending={};self.records={}

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.pending={}

    def _projection(self,key,name):
        def collect(module,args,output):
            if not self.active:return
            entry=self.pending.setdefault(key,{})
            if name in ('q','out'):
                qi=torch.as_tensor(self.spec['queries'],device=output.device,dtype=torch.long)
                if int(qi.max())>=output.shape[1]:raise ValueError('Probe query exceeds padded target boundary')
                entry[name]=output[:,qi].detach()
            else:entry[name]=output.detach()
        return collect

    def _finish(self,key):
        @torch.no_grad()
        def collect(branch,args,output):
            if not self.active:return
            values=self.pending.pop(key);mask=args[2]
            q,k,v=(values[n] for n in ('q','k','v'))
            heads=branch.heads
            def split(x):return x.reshape(1,-1,heads,x.shape[-1]//heads).transpose(1,2).float()
            with torch.autocast(q.device.type,enabled=False):
                logits=split(q)@split(k).transpose(-1,-2)/math.sqrt(q.shape[-1]//heads)
                logits=logits.masked_fill(~mask[:,None,None,:],-torch.inf)
                per_head=logits.softmax(-1)[0];probs=per_head.mean(0)
                entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)
                count=int(mask.sum());rows=[]
                for obj in self.spec['objects']:
                    pairs=[p for p,s in self.spec['pairs'] if s==obj['slot']]
                    if not pairs:continue
                    idx=torch.tensor(pairs,device=probs.device);slot=obj['slot']
                    rows.append(dict(instance_id=obj['instance_id'],category_id=obj['category_id'],
                        probability=float(probs[idx,slot].mean()),probability_over_uniform=float(probs[idx,slot].mean()*count),
                        top1_fraction=float((probs[idx].argmax(-1)==slot).float().mean()),
                        per_head_probability_over_uniform=(per_head[:,idx,slot].mean(-1)*count).cpu().tolist()))
                update=values['out'].float()[0];objects=v[0,mask[0]].float()[1:]
                self.records[key]=dict(valid_tokens=count,queries=len(self.spec['queries']),
                    global_attention_mass=float(probs[:,0].mean()),
                    head_mean_entropy_normalized=float(entropy.mean()/math.log(count)),
                    object_value_common_ratio=float(objects.mean(0).norm()/objects.square().sum(-1).mean().sqrt().clamp_min(1e-12)),
                    update_query_variation_ratio=float((update-update.mean(0)).square().mean().sqrt()/update.square().mean().sqrt().clamp_min(1e-12)),
                    gt_instance_probability_over_uniform=sum(r['probability_over_uniform'] for r in rows)/len(rows),
                    gt_instance_top1_fraction=sum(r['top1_fraction'] for r in rows)/len(rows),per_object=rows,
                    note='FP32 diagnostic softmax reconstructed from actual native BF16 Q/K outputs; not semantic generation accuracy.')
        return collect
