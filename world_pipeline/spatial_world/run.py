"""User-run short entry points; automatic saved-checkpoint recovery, no eval."""
import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from ..build_polished_world import ROOT,read
from .runtime import VERSION

DEFAULT_ROOT=ROOT/'qwen_edit_pano/outputs/object_world_spatial_v1'
CALIBRATION_INITIAL=DEFAULT_ROOT/'pilot/spatial/run-1792923761168994486/checkpoint-step000200'


def latest(root):
    completed=[]
    if root.exists():
        for marker in root.glob('run-*/checkpoint-step*/COMPLETE.json'):
            directory=marker.parent
            if all((directory/name).is_file() for name in ('config.json','world.safetensors','optimizer.pt')):
                meta=read(marker);config=read(directory/'config.json')
                if meta['version']!=VERSION or config['version']!=VERSION:raise ValueError('Unexpected checkpoint version in run root')
                completed.append((meta['step'],directory))
    return max(completed,key=lambda x:(x[0],str(x[1]))) if completed else (0,None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','smoke','pilot','full','diagnose','probe','mechanism',
        'calibrate','calibrate-preflight','calibrate-diagnose','calibrate-probe'))
    p.add_argument('--mode',choices=('spatial','control'),default='spatial')
    p.add_argument('--gpus',type=int,choices=(1,2),default=1)
    p.add_argument('--steps',type=int)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();args.root=args.root.resolve()
    if args.stage.startswith('calibrate'):
        if args.mode!='spatial':p.error('Local calibration requires --mode spatial')
        root=args.root/'calibrate/spatial'
        initial=args.checkpoint or CALIBRATION_INITIAL
        if args.stage in ('calibrate-diagnose','calibrate-probe'):
            checkpoint=args.checkpoint
            if checkpoint is None:_,checkpoint=latest(root)
            if checkpoint is None:raise ValueError('No completed local calibration checkpoint')
            source=Path(read(checkpoint/'config.json')['initial_checkpoint'])
            if args.stage=='calibrate-probe':
                if args.gpus!=1 or args.steps is not None:p.error('calibrate-probe uses one GPU and fixed 28 generation steps')
                subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.state_probe',
                    '--checkpoint',str(checkpoint),'--reference-checkpoint',str(source),'--cases',
                    'baseline','correct','move_tv','mirror_tv','remove_tv','hidden_chair_to_plant','add_floor_plant'],check=True);return
            subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.diagnose',
                '--checkpoints',str(source),str(checkpoint)],check=True);return
        steps=100 if args.steps is None else args.steps
        if steps<=20:raise ValueError('Calibration target must exceed the 20-step warmup')
        common=['--mode','spatial','--local-only-calibration','--init-checkpoint',str(initial),
            '--steps',str(steps),'--train-limit','32','--warmup-steps','20','--save-every','25',
            '--accumulation','4','--lr','0.0001','--region-weight','0.25','--activation-storage','cpu']
        if args.stage=='calibrate-preflight':
            subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.train',*common,'--preflight'],check=True);return
        root.mkdir(parents=True,exist_ok=True)
        with (root/'.launcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            saved,checkpoint=latest(root)
            if saved>=steps:print(f'Already complete: {checkpoint}; target={steps}. No training/evaluation performed.');return
            output=root/('run-'+str(time.time_ns()))
            command=[sys.executable,'-u']
            if args.gpus==2:command+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2']
            command+=['-m','qwen_edit_pano.world_pipeline.spatial_world.train',*common,'--output',str(output)]
            if checkpoint:command+=['--resume',str(checkpoint)]
            print(dict(stage='local_calibration',gpus=args.gpus,resume=str(checkpoint),initial=str(initial),target_steps=steps,output=str(output)),flush=True)
            subprocess.run(command,check=True)
            print('CALIBRATION COMPLETE. No automatic evaluation or image generation.',flush=True)
        return
    if args.stage=='mechanism':
        if args.mode!='spatial' or args.gpus!=1 or args.steps is not None:
            p.error('mechanism is a single-GPU, forward-only check of the Spatial checkpoint; no --mode control/--steps')
        command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.mechanism']
        if args.checkpoint:command+=['--checkpoint',str(args.checkpoint)]
        subprocess.run(command,check=True);return
    if args.stage in ('diagnose','probe'):
        checkpoint=args.checkpoint
        if checkpoint is None:
            _,checkpoint=latest(args.root/'full'/args.mode)
            if checkpoint is None:_,checkpoint=latest(args.root/'pilot'/args.mode)
        if checkpoint is None:raise ValueError('No completed pilot/full checkpoint')
        module='diagnose' if args.stage=='diagnose' else 'state_probe'
        subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.'+module,'--checkpoint',str(checkpoint)],check=True)
        return
    stage='pilot' if args.stage=='preflight' else args.stage
    if stage=='smoke' and args.steps not in (None,2):raise ValueError('Smoke always uses exactly two steps')
    steps=args.steps if args.steps is not None else (2 if stage=='smoke' else 200 if stage=='pilot' else 2000)
    if steps<1:raise ValueError('Positive steps required')
    root=args.root/stage/args.mode
    common=['--mode',args.mode,'--steps',str(steps),'--train-limit','0' if stage=='full' else '32',
        '--warmup-steps','100' if stage=='full' else '20','--save-every','2' if stage=='smoke' else '50',
        '--accumulation','4','--lr','0.0001','--region-weight','0.25','--activation-storage','cpu']
    if stage=='smoke':common.append('--smoke')
    if args.stage=='preflight':
        subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.spatial_world.train',*common,'--preflight'],check=True);return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved,checkpoint=latest(root)
        if saved>=steps:print(f'Already complete: {checkpoint}; target={steps}. No training/evaluation performed.');return
        output=root/('run-'+str(time.time_ns()))
        command=[sys.executable,'-u']
        if args.gpus==2:command+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2']
        command+=['-m','qwen_edit_pano.world_pipeline.spatial_world.train',*common,'--output',str(output)]
        if checkpoint:command+=['--resume',str(checkpoint)]
        print(dict(stage=stage,mode=args.mode,gpus=args.gpus,resume=str(checkpoint),target_steps=steps,output=str(output)),flush=True)
        subprocess.run(command,check=True)
        print('TRAINING COMPLETE. No automatic evaluation or image generation.',flush=True)


if __name__=='__main__':main()
