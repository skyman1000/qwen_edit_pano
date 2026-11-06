"""Explicit matched C/D evaluation, fixed 6 val images and 20 ERP generations."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file
from ..build_polished_world import read
from ..object_world.data import tensorize,identity
from ..spatial_world.controls import fully_outside
from .data import inputs,EVAL_PANEL
from .runtime import WorldBranch
from .continue_train import configuration
from .continue_run import latest
from . import category_contrast as cc
from . import evaluate as ev
from .supervision import prepare as sparse_targets


def single_variant(row,target):
    state=tensorize(row);cat,geometry,mask=(t.clone() for t in state);slot=target['slot']
    if int(cat[0,slot])!=target['category_id'] or row['objects'][slot-1]['instance_id']!=target['instance_id']:
        raise ValueError('Single-object rendering target changed')
    if not bool(fully_outside(geometry)[0,slot]):raise ValueError('Rendering target conflicts with Local')
    if target['mode']=='wrong_category':cat[0,slot]=target['wrong_category_id']
    elif target['mode']=='wrong_position':
        original=geometry.clone();geometry[0,slot,0]+=1. if float(geometry[0,slot,0])>=0 else -1.
        if not bool(fully_outside(geometry)[0,slot]):geometry[0,slot,:3]=original[0,slot,:3];geometry[0,slot,2]-=1.
        if not bool(fully_outside(geometry)[0,slot]):raise ValueError('Cannot move selected object without conflicting with Local')
    else:raise ValueError('Invalid single-object edit')
    return (cat,geometry,mask),dict(operation=target['mode'],changed_slots=[slot],instance_id=target['instance_id'],
        original_category=target['category_id'],new_category=int(cat[0,slot]),
        original_center=state[1][0,slot,:3].tolist(),new_center=geometry[0,slot,:3].tolist(),scope='one_supported_fully_offscreen_object')


def protocol(paths,root,prepared=None):
    prepared=prepared or cc.prepare();plan,allrows,rows,features=inputs();byid={r['id']:r for r in allrows}
    configs={g:read(path/'config.json') for g,path in paths.items()};steps={};initial=None
    original=load_file(str(cc.INITIAL/'world.safetensors'))
    for group,path in paths.items():
        if configs[group]!=configuration(plan,rows,prepared,group,False):raise ValueError('Continuation evaluation contract changed')
        meta=read(path/'COMPLETE.json')
        if meta['version']!=cc.EXPERIMENT or meta['step']<=200:raise ValueError('Use a complete new continuation checkpoint')
        steps[group]=meta['step'];run=path.parent
        while not (run/'initial_world.safetensors').is_file():
            run=Path(read(run/'run_plan.json')['resume']).parent
        saved=load_file(str(run/'initial_world.safetensors'))
        if saved.keys()!=original.keys() or not all(torch.equal(v,original[k]) for k,v in saved.items()):
            raise ValueError('C/D did not start from the actual B200 weights')
        if initial is not None and not all(torch.equal(v,initial[k]) for k,v in saved.items()):raise ValueError('C/D initial tensors differ')
        initial=saved
    if len(set(steps.values()))!=1:raise ValueError('Compare equal C/D step counts')
    panel=read(EVAL_PANEL)
    for case in panel['cases']:
        row=byid[case['id']]
        if case['source']!=identity(row) or row['split']!=case['split']:raise ValueError('Panel source/split changed')
        if (case['id'] in configs['C']['train_ids'])!=(case['split']=='train'):raise ValueError('Train/val panel mismatch')
    p=dict(version=cc.EXPERIMENT,architecture_version=ev.VERSION,checkpoints={g:str(path) for g,path in paths.items()},steps=steps,configs=configs,
        panel=panel,sigmas=[.2,.5,.8],modes=list(ev.MODES),same_actual_initial_tensors=True,
        input_objects={c['id']:byid[c['id']]['objects'] for c in panel['cases']},
        note='Same B200 source, fixed main samples/noise, region0.25/binding0.01; D alone adds category contrast. Diagnostic permutations are stress tests, not single-object generation.')
    ids=ev.render_scenes(p,byid);targets={}
    for i,sid in enumerate(ids):
        row=byid[sid];entries=prepared['targets'].get(sid)
        if entries is None:
            items,_=sparse_targets(row,plan['height'],plan['padding_columns'],plan['max_objects']);entries=cc.target_entries(row,items)
        if not entries:raise ValueError('No sufficiently supported offscreen rendering object')
        mode='wrong_category' if i%2==0 else 'wrong_position'
        candidates=entries
        if mode=='wrong_position':
            movable=[r for r in entries if r['category_id'] in (3,5,6,8,10,11,13,14,19,20,22,28,33,35,38)]
            if movable:candidates=movable
        selected=sorted(candidates,key=lambda r:(r['category_id']!=22,-r['pixels'],r['slot']))[0]
        chosen=cc.select_target([selected],plan['names'],20261027+i);targets[sid]=dict(**chosen,mode=mode)
        single_variant(row,targets[sid])
    p['single_object_render']=dict(ids=ids,targets=targets,
        selection='same two train/two of existing six val; sufficiently supported offscreen objects, prefer TV; single category/position edit in both C/D, no hidden fallback')
    tag='-'.join(g+'-'+path.parent.name+'-'+path.name for g,path in paths.items())
    return plan,byid,features,p,Path(root)/'evaluation'/tag


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=cc.OUTPUT)
    p.add_argument('--stage',choices=('diagnose','render'));p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.stage is None:
        for stage in ('diagnose','render'):
            command=[sys.executable,'-u','-m',__package__+'.continue_evaluate','--root',str(args.root),'--stage',stage]
            if args.preflight:command.append('--preflight')
            subprocess.run(command,check=True)
        return
    torch.set_num_threads(4);paths={g:latest(args.root/'train'/g)[1] for g in ('C','D')}
    if any(path is None for path in paths.values()):raise ValueError('Both trained C/D checkpoints are required')
    plan,byid,features,p,out=protocol(paths,args.root)
    arch=read(Path(plan['base_snapshot'])/'transformer/config.json');w=WorldBranch(arch['num_attention_heads']*arch['attention_head_dim'],features)
    for path in paths.values():
        state=load_file(str(path/'world.safetensors'))
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite continuation checkpoint')
        w.load_state_dict(state,strict=True)
    del state,w
    fn=ev.diagnose if args.stage=='diagnose' else ev.render;directory=out/args.stage
    if args.preflight:fn(plan,byid,features,p,directory,paths,True);return
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fn(plan,byid,features,p,directory,paths,False)


if __name__=='__main__':main()
