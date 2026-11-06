"""User-run C/D jobs; source B200, exact arm recovery, no automatic evaluation."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from ..build_polished_world import read
from .data import inputs
from . import category_contrast as cc
from .continue_train import configuration


def latest(root):
    found=[]
    for marker in Path(root).glob('run-*/checkpoint-step*/COMPLETE.json'):
        path=marker.parent
        if not all((path/n).is_file() for n in ('world.safetensors','optimizer.pt','config.json')):continue
        m=read(marker)
        if m['version']!=cc.EXPERIMENT:raise ValueError('Unexpected continuation version')
        found.append((m['step'],path))
    return max(found,key=lambda x:(x[0],str(x[1]))) if found else (200,None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('preflight','smoke','train','evaluate'))
    p.add_argument('--group',choices=('C','D'));p.add_argument('--root',type=Path,default=cc.OUTPUT);args=p.parse_args();args.root=args.root.resolve()
    if args.stage=='evaluate':
        subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.grounded_world.continue_evaluate','--root',str(args.root)],check=True);return
    prepared=cc.prepare();plan,_,rows,_=inputs()
    if args.stage=='preflight':
        print({k:v for k,v in prepared.items() if k not in ('sources','targets')},flush=True);return
    if args.stage=='train' and not args.group:p.error('Choose --group C or --group D; separate GPUs/jobs')
    if args.stage=='train':
        for group in ('C','D'):
            _,path=latest(args.root/'smoke'/group)
            if path is None or not (path/'SMOKE_COMPLETE.json').is_file():raise ValueError('Run the C/D smoke first')
            if read(path/'config.json')!=configuration(plan,rows,prepared,group,True):raise ValueError('Smoke contract changed')
            m=read(path/'SMOKE_COMPLETE.json')
            if m['base_trainable'] or (group=='D' and not m['actual_value_output_gradient_verified']):raise ValueError('Smoke did not verify the new training path')
    for group in ([args.group] if args.group else ['C','D']):
        stage='smoke' if args.stage=='smoke' else 'train';root=args.root/stage/group;root.mkdir(parents=True,exist_ok=True)
        expected=configuration(plan,rows,prepared,group,stage=='smoke');target=expected['continuation_target']
        with (root/'.launcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);step,resume=latest(root)
            if resume and read(resume/'config.json')!=expected:raise ValueError('Saved group/data/config changed; use a new output root')
            if step>=target:print(dict(stage=stage,group=group,already_complete=str(resume)),flush=True);continue
            out=root/('run-'+str(time.time_ns()));command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.grounded_world.continue_train',
                '--group',group,'--output',str(out)]
            if stage=='smoke':command.append('--smoke')
            if resume:command+=['--resume',str(resume)]
            print(dict(stage=stage,group=group,source=str(resume or cc.INITIAL),target_step=target,output=str(out)),flush=True)
            subprocess.run(command,check=True)
    print('COMPLETE. No automatic evaluation, extension or full-data training.',flush=True)


if __name__=='__main__':main()
