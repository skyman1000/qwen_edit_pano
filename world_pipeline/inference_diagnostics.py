"""Read-only hook reporting adapter residual relative to target activation scale."""
import math


class RelativeResidualRecorder:
    def __init__(self,world):
        self.records={}
        self.handles=[]
        for key,branch in world.branches.items():
            def collect(module,args,output,key=key):
                target=args[0][:,:args[3]]
                target_rms=float(target.float().square().mean().sqrt())
                delta=module.diagnostics['effective_delta_rms']
                self.records.setdefault(key,[]).append(dict(target_rms=target_rms,
                    effective_delta_rms=delta,relative_delta_rms=delta/max(target_rms,1e-12)))
            self.handles.append(branch.register_forward_hook(collect))

    def reset(self):self.records={}

    def summary(self):
        result={}
        for key,records in self.records.items():
            ratios=[r['relative_delta_rms'] for r in records]
            if not all(math.isfinite(x) for x in ratios):raise ValueError('Nonfinite adapter diagnostics')
            result[key]=dict(calls=len(records),relative_min=min(ratios),relative_max=max(ratios),
                relative_mean=sum(ratios)/len(ratios),last=records[-1])
        return result
