"""Manual Full-C preparation, GPU smoke, bounded training and evaluation."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from ..build_polished_world import read
from . import full_data as fd


def latest(root):
    candidates=[]
    for marker in Path(root).glob('run-*/checkpoint-step*/COMPLETE.json'):
        path=marker.parent
        if not all((path/n).is_file() for n in ('world.safetensors','optimizer.pt','config.json')):continue
        meta=read(marker)
        if meta['version']!=fd.VERSION:raise ValueError('Unexpected Full-C checkpoint version')
        candidates.append((meta['step'],path))
    return max(candidates,key=lambda x:(x[0],str(x[1]))) if candidates else (0,None)


def evaluation(root, stage, preflight=False):
    from . import evaluate as ev
    from .data import EVAL_PANEL
    from .continue_evaluate import single_variant
    from .category_contrast import OUTPUT as CATEGORY_OUTPUT
    from .runtime import VERSION as ARCHITECTURE
    plan,allrows,rows,features=fd.full_inputs();byid={r['id']:r for r in allrows}
    step,path=latest(root/'train')
    if path is None or read(path/'config.json')!=fd.configuration(plan,rows):raise ValueError('No matching real Full-C checkpoint')
    if step<=0:raise ValueError('Use a positive completed Full-C checkpoint')
    # Reuse the exact reviewed development cases and one-object interventions,
    # rather than changing the task or selecting images based on new scores.
    old=list((CATEGORY_OUTPUT/'evaluation').glob('*/render/PROTOCOL.json'))
    if len(old)!=1:raise ValueError('Select one unambiguous reviewed C/D rendering protocol')
    previous=read(old[0]);panel=read(EVAL_PANEL);configs={'C600':read(fd.SOURCE/'config.json'),'Full-C':read(path/'config.json')}
    if previous['checkpoints']['C']!=str(fd.SOURCE):raise ValueError('Previous C reference changed')
    for c in panel['cases']:
        row=byid[c['id']]
        if row['split']!=c['split'] or (c['id'] in configs['Full-C']['train_ids']) != (c['split']=='train'):
            raise ValueError('Full-C validation/training panel overlap')
        from ..object_world.data import identity
        if c['source']!=identity(row):raise ValueError('Evaluation image source changed')
    for sid,target in previous['single_object_render']['targets'].items():single_variant(byid[sid],target)
    paths={'C600':fd.SOURCE,'Full-C':path}
    import torch
    from safetensors.torch import load_file
    from .runtime import WorldBranch
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features)
    for checkpoint in paths.values():
        state=load_file(str(checkpoint/'world.safetensors'),device='cpu')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite evaluation checkpoint')
        world.load_state_dict(state,strict=True)
    del world,state
    protocol=dict(version=fd.VERSION,architecture_version=ARCHITECTURE,checkpoints={g:str(p) for g,p in paths.items()},
        steps={'C600':600,'Full-C':step},configs=configs,panel=panel,sigmas=[.2,.5,.8],modes=list(ev.MODES),
        input_objects={c['id']:byid[c['id']]['objects'] for c in panel['cases']},
        single_object_render=previous['single_object_render'],
        note='Same C architecture/loss/base; different training distribution and budget. Not a matched A/B loss ablation or generated-object certification.')
    out=root/'evaluation'/('new-step'+str(step)+'-'+path.parent.name)/stage
    fn=ev.diagnose if stage=='diagnose' else ev.render
    if preflight:fn(plan,byid,features,protocol,out,paths,True);return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fn(plan,byid,features,protocol,out,paths,False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','check','cuda-check','smoke','train','evaluate','diagnose','render'))
    p.add_argument('--gpus',type=int,choices=(1,2),default=1);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--root',type=Path,default=fd.OUTPUT);p.add_argument('--preflight',action='store_true',help='evaluation stages only, CPU checks without generation')
    p.add_argument('--prepare-threads',type=int,choices=(1,2,4,8,16),default=4,help='CPU prepare/check stages only')
    a=p.parse_args();a.root=a.root.resolve()
    if a.preflight and a.stage not in ('evaluate','diagnose','render'):
        p.error('--preflight applies to evaluation only; use check for CPU data checks')
    if a.stage=='cuda-check':
        from ..cuda_check import require_cuda
        require_cuda('manual_full_C_cuda_check');return
    if a.stage in ('prepare','check'):
        command=[sys.executable,'-u','-m',__package__+'.full_data','--threads',str(a.prepare_threads)]
        if a.stage=='check':command.append('--check-only')
        subprocess.run(command,check=True);return
    if a.stage in ('evaluate','diagnose','render'):
        for stage in (('diagnose','render') if a.stage=='evaluate' else (a.stage,)):
            # A separate process releases the diagnostic model before rendering.
            command=[sys.executable,'-u','-m',__package__+'.full_run',stage,'--root',str(a.root)]
            if a.preflight:command.append('--preflight')
            if a.stage=='evaluate':subprocess.run(command,check=True)
            else:evaluation(a.root,stage,a.preflight)
        return
    import torch
    torch.set_num_threads(4)
    plan,_,rows,_=fd.full_inputs();fd.require_prepared(plan,rows)
    smoke=a.stage=='smoke';expected=fd.configuration(plan,rows,smoke);target=2 if smoke else a.steps
    if not 0 < target <= 10000:raise ValueError('Use 1..10000 new full-data steps')
    if not smoke:
        _,checkpoint=latest(a.root/'smoke'/str(a.gpus))
        if checkpoint is None or read(checkpoint/'config.json')!=fd.configuration(plan,rows,True):
            raise ValueError('Run the smoke with the same --gpus first')
        marker=read(checkpoint/'SMOKE_COMPLETE.json')
        if (marker['replicas']!=a.gpus or marker['maximum_objects']!=max(len(r['objects']) for r in rows) or marker['base_trainable']!=0
                or not marker['semantic_gradient'] or not marker['value_output_gradient']):
            raise ValueError('Smoke does not cover this GPU count/max object count')
    directory=a.root/('smoke/'+str(a.gpus) if smoke else 'train');directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);saved,resume=latest(directory)
        if resume is not None and read(resume/'config.json')!=expected:raise ValueError('Saved Full-C settings/data changed')
        if saved>=target:print(dict(already_complete=str(resume),new_full_step=saved),flush=True);return
        command=[sys.executable,'-u']
        if a.gpus>1:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node='+str(a.gpus)]
        command+=['-m',__package__+'.full_train','--output',str(directory/('run-'+str(time.time_ns()))),'--steps',str(target)]
        if smoke:command.append('--smoke')
        if resume is not None:command+=['--resume',str(resume)]
        print(dict(stage=a.stage,train=len(rows),gpus=a.gpus,global_batch=4,new_step_start=saved,new_step_target=target,
            resume=str(resume) if resume else None,initial_C600=str(fd.SOURCE),automatic_evaluation=False),flush=True)
        subprocess.run(command,check=True)
    print('STAGE COMPLETE. No automatic evaluation or extension.',flush=True)


if __name__=='__main__':main()
