"""User-run orchestration: CPU check, separate smoke, finite warm-start/resume."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from ..build_polished_world import ROOT,read,atomic_json
from . import data as d


def latest(folder,config):
    found=[]
    for marker in Path(folder).glob('run-*/checkpoint-step*/COMPLETE.json'):
        path=marker.parent;meta=read(marker)
        if meta.get('version')!=d.VERSION:raise ValueError('Foreign checkpoint in experiment')
        if read(path/'config.json')!=config:raise ValueError('Existing checkpoint configuration changed; use a new root')
        if type(meta.get('step')) is not int or path.name!=f"checkpoint-step{meta['step']:06d}":
            raise ValueError('Checkpoint step/name mismatch')
        if all((path/k).is_file() for k in ('world.safetensors','optimizer.pt')):
            if meta.get('optimizer_step')!=meta['step']:raise ValueError('Invalid optimizer marker')
            found.append((meta['step'],str(path),path))
    return max(found)[2] if found else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',choices=('control','binding'))
    p.add_argument('--stage',choices=('all','smoke','check'),default='all')
    p.add_argument('--additional-steps',type=int,default=1000);p.add_argument('--root',type=Path,default=d.OUTPUT)
    a=p.parse_args()
    if not 1<=a.additional_steps<=1000:raise ValueError('Use a finite 1..1000 additional-step target')
    root=a.root.resolve()
    allowed=ROOT/'qwen_edit_pano/outputs'
    if root==allowed or not root.is_relative_to(allowed):raise ValueError('Use a new directory below qwen_edit_pano/outputs')
    group=root/a.group;group.mkdir(parents=True,exist_ok=True)
    with (group/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d.check(group)
        if a.stage=='check':return
        plan,_,old=d.inputs()
        def run(smoke,resume=None):
            directory=group/('smoke' if smoke else 'train')/f'run-{time.time_ns()}'
            command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.binding_refine_world.train',
                '--group',a.group,'--output',str(directory),'--additional-steps',str(a.additional_steps)]
            if smoke:command+=['--smoke']
            if resume:command+=['--resume',str(resume)]
            command+=['--stop-file',str(group/'STOP')]
            subprocess.run(command,check=True)
            return directory
        if (group/'STOP').exists():raise ValueError('STOP exists; move it before continuing')
        smoke_config=d.configuration(plan,old,a.group,True)
        smoke=latest(group/'smoke',smoke_config)
        if smoke is None or read(smoke/'COMPLETE.json')['step']!=10002:
            run(True);smoke=latest(group/'smoke',smoke_config)
        if smoke is None or read(smoke/'COMPLETE.json')['step']!=10002 or not (smoke/'SMOKE_COMPLETE.json').is_file():
            raise ValueError('Two-step GPU smoke did not complete; inspect STOP/logs')
        if (group/'STOP').exists():return
        marker=read(smoke/'SMOKE_COMPLETE.json')
        if (marker['config']!=smoke_config or not marker['base_frozen'] or not marker['forward_unchanged_with_labels']
                or a.group=='binding' and not marker['both_binding_terms']):
            raise ValueError('Smoke configuration/result mismatch')
        if a.stage=='smoke':return
        config=d.configuration(plan,old,a.group,False);checkpoint=latest(group/'train',config)
        target=10000+a.additional_steps
        if checkpoint and read(checkpoint/'COMPLETE.json')['step']>=target:
            print(dict(already_complete=str(checkpoint),target=target));return
        # Formal training always starts from A10000 or this arm's formal
        # checkpoint, never the two smoke optimizer steps.
        run(False,checkpoint);checkpoint=latest(group/'train',config)
        if checkpoint and read(checkpoint/'COMPLETE.json')['step']>=target:
            atomic_json(group/'FINISHED.json',dict(version=d.VERSION,target=target,checkpoint=str(checkpoint),
                automatic_evaluation=False,automatic_extension=False))


if __name__=='__main__':main()
