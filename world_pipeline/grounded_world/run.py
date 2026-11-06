"""Short user-run entries. A/B are independent single-GPU jobs with recovery."""
import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from ..build_polished_world import read
from .data import DEFAULT_ROOT,DEFAULT_PACK,DEFAULT_PANEL,DEFAULT_SUPPORT,inputs,smoke_rows
from .runtime import VERSION


def latest(root):
    found=[]
    for marker in Path(root).glob('run-*/checkpoint-step*/COMPLETE.json'):
        p=marker.parent
        if not all((p/n).is_file() for n in ('world.safetensors','optimizer.pt','config.json')):continue
        meta=read(marker);config=read(p/'config.json')
        if meta['version']!=VERSION or config['version']!=VERSION:raise ValueError('Unexpected checkpoint version')
        found.append((meta['step'],p))
    return max(found,key=lambda p:(p[0],str(p[1]))) if found else (0,None)


def checkpoints(root,groups=('A','B')):
    result={}
    for group in groups:
        step,path=latest(Path(root)/'pilot'/group)
        if path is None:raise ValueError('Missing completed pilot checkpoint for '+group)
        result[group]=path
    if len({read(p/'COMPLETE.json')['step'] for p in result.values()})!=1:raise ValueError('Compare equal checkpoint steps in A/B')
    configs={g:read(p/'config.json') for g,p in result.items()}
    common=[{k:v for k,v in c.items() if k not in ('group','binding_weight')} for c in configs.values()]
    if any(c!=common[0] for c in common):raise ValueError('A/B settings differ beyond binding loss')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','smoke','pilot','diagnose','render','evaluate'))
    p.add_argument('--group',choices=('A','B'));p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=p.parse_args();args.root=args.root.resolve()
    if args.stage=='preflight':
        subprocess.run([sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.grounded_world.data'],check=True);return
    if args.stage in ('diagnose','render','evaluate'):
        groups=[args.group] if args.group else ['A','B'];paths=checkpoints(args.root,groups)
        for stage in (('diagnose','render') if args.stage=='evaluate' else (args.stage,)):
            command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.grounded_world.evaluate',stage,'--root',str(args.root)]
            for group,path in paths.items():command+=['--checkpoint',group,str(path)]
            subprocess.run(command,check=True)
        return
    if args.stage=='pilot' and args.group is None:p.error('Choose --group A or --group B; one GPU per arm')
    from .train import configuration
    plan,_,selected,_=inputs()
    def expected(group,smoke):
        settings=SimpleNamespace(pack=DEFAULT_PACK,panel=DEFAULT_PANEL,support=DEFAULT_SUPPORT,group=group,smoke=smoke,seed=0)
        return configuration(plan,settings,smoke_rows(selected) if smoke else selected)
    groups=[args.group] if args.group else ['A','B'];target=2 if args.stage=='smoke' else 200
    if args.stage=='pilot':
        for group in ('A','B'):
            _,smoke=latest(args.root/'smoke'/group)
            if smoke is None or not (smoke/'SMOKE_COMPLETE.json').is_file():
                raise ValueError('Run the two-arm smoke first: bash .../run_object_world_grounded.sh smoke')
            if read(smoke/'config.json')!=expected(group,True):raise ValueError('Smoke configuration/data changed; use a fresh experiment root')
            check=read(smoke/'SMOKE_COMPLETE.json')
            if not check['zero_gate_panorama_exact'] or not check['semantic_gradient'] or not check['value_output_gradient']:
                raise ValueError('Smoke did not verify the actual trainable paths')
    for group in groups:
        root=args.root/args.stage/group;root.mkdir(parents=True,exist_ok=True)
        with (root/'.launcher.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);saved,resume=latest(root)
            if resume and read(resume/'config.json')!=expected(group,args.stage=='smoke'):
                raise ValueError('Saved run belongs to different data/configuration; use a fresh experiment root')
            if saved>=target:print(dict(stage=args.stage,group=group,already_complete=str(resume)),flush=True);continue
            out=root/('run-'+str(time.time_ns()))
            command=[sys.executable,'-u','-m','qwen_edit_pano.world_pipeline.grounded_world.train','--group',group,
                '--output',str(out),'--steps',str(target)]
            if args.stage=='smoke':command.append('--smoke')
            if resume:command+=['--resume',str(resume)]
            print(dict(stage=args.stage,group=group,target=target,resume=str(resume),output=str(out)),flush=True)
            subprocess.run(command,check=True)
    print('STAGE COMPLETE. No automatic training extension or evaluation.',flush=True)


if __name__=='__main__':main()
