"""CPU check -> isolated two-step dual smoke -> fresh or own-formal resume."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from ..build_polished_world import ROOT, read, atomic_json
from . import data as d


def smoke_valid(checkpoint, config):
    if checkpoint is None or not (checkpoint/'SMOKE_COMPLETE.json').is_file():
        return False
    meta = read(checkpoint/'SMOKE_COMPLETE.json')
    required = ('both_ranks_passed','zero_gate_baseline_exact','labels_do_not_change_forward',
        'zero_gate_connector_key_gradients','encoder_gradient','base_frozen',
        'replica_parameters_equal','fresh_initialization','both_binding_terms','value_generation_gradient')
    return (meta.get('version') == d.VERSION and meta.get('config') == config
        and meta.get('optimizer_step') == 2 and all(meta.get(k) is True for k in required)
        and read(checkpoint/'COMPLETE.json')['step'] == 2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=d.OUTPUT)
    p.add_argument('--stage',choices=('all','check','smoke'),default='all')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--stop-file',type=Path)
    a=p.parse_args()
    if not 2<=a.steps<=10000:p.error('Total steps must be between 2 and 10000')
    root=a.root.resolve();allowed=ROOT/'qwen_edit_pano/outputs'
    if root==allowed or not root.is_relative_to(allowed):
        raise ValueError('Use a separate experiment root below qwen_edit_pano/outputs')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,_=d.inputs()
        # Reject a foreign/changed run before producing new CPU check output.
        config=d.configuration(plan,False);formal=d.latest(root/'train',config)
        if formal and d.resume_step(formal,config)>=a.steps:
            print(dict(already_complete=str(formal),target=a.steps));return
        d.check(root)
        if a.stage=='check':return
        def stopped():
            return (root/'STOP').exists() or a.stop_file is not None and a.stop_file.exists()
        if stopped():
            print('Stop file exists; no training started.',flush=True);return
        def run(smoke, resume=None):
            out=root/('smoke' if smoke else 'train')/f'run-{time.time_ns()}'
            command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc-per-node=2','-m','qwen_edit_pano.world_pipeline.native_world.train',
                '--output',str(out),'--steps',str(a.steps),'--manual-stop-file',str(root/'STOP')]
            if smoke:command+=['--smoke']
            if resume:command+=['--resume',str(resume)]
            if a.stop_file:command+=['--stop-file',str(a.stop_file)]
            subprocess.run(command,check=True)
        smoke_config=d.configuration(plan,True)
        smoke=d.latest(root/'smoke',smoke_config)
        if not smoke_valid(smoke,smoke_config):
            run(True)
            if stopped():return
            smoke=d.latest(root/'smoke',smoke_config)
        if not smoke_valid(smoke,smoke_config):
            raise ValueError('Two-step dual GPU smoke did not pass; inspect logs')
        if a.stage=='smoke' or stopped():return
        # Formal initialization is independent of the trained smoke and old Worlds.
        run(False,formal)
        formal=d.latest(root/'train',config)
        if formal and d.resume_step(formal,config)>=a.steps:
            atomic_json(root/'FINISHED.json',dict(version=d.VERSION,target=a.steps,
                checkpoint=str(formal),automatic_evaluation=False,automatic_extension=False))
        elif not stopped():
            raise ValueError('Trainer exited without reaching target or a stop request')


if __name__=='__main__':main()
