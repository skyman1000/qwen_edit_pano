"""User-launched preparation/cache/smoke/fresh training with exact resume."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import torch
from ..build_polished_world import read,atomic_json
from . import data as d


def latest(directory,expected):
    found=[]
    for marker in Path(directory).glob('run-*/checkpoint-step*/COMPLETE.json'):
        path=marker.parent;meta=read(marker)
        if not all((path/name).is_file() for name in ('config.json','world.safetensors','optimizer.pt')):continue
        if meta['version']!=d.VERSION or read(path/'config.json')!=expected or meta['step']!=meta['optimizer_step']:
            raise ValueError('Different experiment/configuration in resume directory')
        found.append((meta['step'],path))
    return max(found,key=lambda item:(item[0],str(item[1]))) if found else (0,None)


def workers(module,gpus,args):
    command=[sys.executable,'-u']
    if gpus==2:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    command+=['-m',__package__+'.'+module,*args]
    subprocess.run(command,check=True)


def cache(root,gpus,smoke):
    missing=d.check(root,smoke)
    if missing:
        args=['--root',str(root)]
        if smoke:args.append('--smoke')
        workers('cache',gpus,args)
    if d.check(root,smoke):raise ValueError('Cache workers did not finish all requested samples')
    _,allrows,_,_,_=d.inputs(root)
    count=len(d.smoke_rows(allrows)) if smoke else sum(r['split']=='train' for r in allrows)
    atomic_json(root/('CACHE_SMOKE_COMPLETE.json' if smoke else 'CACHE_COMPLETE.json'),dict(version=d.VERSION,
        samples=count,caption_normalization='scene_caption_direction_neutral_v1',hashes_computed=False))


def train(root,gpus,steps,smoke):
    plan,rows,_,prepared,_=d.inputs(root);expected=d.configuration(root,plan,rows,prepared,smoke)
    target=2 if smoke else steps
    if not 0<target<=expected['maximum_steps']:raise ValueError('Training target must be 1..10000')
    smoke_initial=None
    if not smoke:
        smoke_config=d.configuration(root,plan,rows,prepared,True)
        saved,checkpoint=latest(root/'smoke'/str(gpus),smoke_config)
        if saved!=2:raise ValueError('Finish two-step smoke with this GPU count first')
        marker=read(checkpoint/'SMOKE_COMPLETE.json')
        wanted=[r['id'] for r in d.smoke_rows(rows)]
        if (marker['replicas']!=gpus or marker['ids']!=wanted or marker['binding_weight']!=0.
                or not all(marker[k] for k in ('zero_gate_equivalent','semantic_gradient','value_output_gradient','caption_condition'))
                or marker['base_trainable']!=0):raise ValueError('Smoke does not certify this training configuration')
        smoke_initial=checkpoint.parent/'initial_world.safetensors'
    directory=root/('smoke/'+str(gpus) if smoke else 'train');directory.mkdir(parents=True,exist_ok=True)
    saved,resume=latest(directory,expected)
    if saved>=target:
        print(dict(stage='already_complete',checkpoint=str(resume),target=target),flush=True);return
    args=['--root',str(root),'--output',str(directory/('run-'+str(time.time_ns()))),'--steps',str(target),
          '--stop-file',str(root/'STOP')]
    if smoke:args.append('--smoke')
    if resume:args+=['--resume',str(resume)]
    elif smoke_initial is not None:args+=['--smoke-initial',str(smoke_initial)]
    print(dict(stage='smoke' if smoke else 'train',gpus=gpus,start=saved,target=target,resume=str(resume) if resume else None,
        fresh_World_and_Adam=resume is None,smoke_weights_inherited=False,binding=False,global_batch=4,
        full_train_passes=target*4/7390,automatic_evaluation=False),flush=True)
    workers('train',gpus,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','cache','smoke','train','all'))
    parser.add_argument('--root',type=Path,default=d.OUTPUT);parser.add_argument('--gpus',type=int,choices=(1,2),default=2)
    parser.add_argument('--steps',type=int,default=2000)
    a=parser.parse_args();torch.set_num_threads(4);root=a.root.resolve()
    if not root.is_relative_to(d.ROOT/'qwen_edit_pano/outputs') or root==d.ROOT/'qwen_edit_pano/outputs':
        parser.error('Use an isolated experiment directory below qwen_edit_pano/outputs')
    if not 1<=a.steps<=10000:parser.error('Use target steps 1..10000')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage in ('prepare','all'):d.prepare(root)
        if a.stage=='check':
            if d.check(root):raise SystemExit('Caption caches missing; run cache/all on allocated GPU(s)')
            return
        if a.stage=='cache':cache(root,a.gpus,False)
        elif a.stage=='smoke':cache(root,a.gpus,True);train(root,a.gpus,2,True)
        elif a.stage=='train':
            if d.check(root):raise ValueError('Caption caches missing')
            train(root,a.gpus,a.steps,False)
        elif a.stage=='all':
            # Three samples first, then real two-step smoke; fail before full caching.
            cache(root,a.gpus,True);train(root,a.gpus,2,True)
            cache(root,a.gpus,False);train(root,a.gpus,a.steps,False)
    print('STAGE COMPLETE. No automatic evaluation or target extension.',flush=True)


if __name__=='__main__':main()
