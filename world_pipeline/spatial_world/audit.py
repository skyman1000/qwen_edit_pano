"""CPU review of real input projection/control contracts and measured old speed."""
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ..build_polished_world import ROOT,read,atomic_json
from ..object_world.data import load_pack,DEFAULT_PACK,tensorize
from .data import select_train,DEFAULT_INITIAL
from .projection import erp_support,POLICY
from .runtime import WorldBranch,load_world
from .controls import condition_variant,fully_outside

OUT=ROOT/'research/audits/spatial_world_review_2026-10-25'


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    plan,records=load_pack(DEFAULT_PACK);byid={r['id']:r for r in records}
    panel=read(ROOT/'research/audits/id_condition_diagnostic_2026-10-25/PANEL.json')
    selected=select_train(records,32,0)
    cases={c['id']:c for c in panel['cases']}
    ids=list(dict.fromkeys([r['id'] for r in selected]+list(cases)))
    rows=[];selected_instances=[];control_rows=[];started=time.monotonic()
    for sid in ids:
        row=byid[sid];category,g,mask=tensorize(row,plan['max_objects'])
        weights,meta=erp_support(g,mask,plan['height']//16,plan['height']//8,plan['padding_columns'])
        assert float(weights[:,:,0].sum())==0
        assert float(weights.sum(-1).max())<=1.000001
        assert torch.isfinite(weights).all()
        grid=weights[0].reshape(plan['height']//16,plan['height']//8+2*plan['padding_columns'],-1)
        pad=plan['padding_columns']
        if pad:
            torch.testing.assert_close(grid[:,:pad],grid[:,-2*pad:-pad],rtol=0,atol=0)
            grid=grid[:,pad:-pad]
        rows.append(dict(id=sid,split=row['split'],**meta))
        if sid not in cases:continue
        with np.load(row['object_region_supervision']['path'],allow_pickle=False) as a:
            instance=a['instance'];valid=a['object_loss_valid']
        for obj in cases[sid]['objects']:
            slot=next(i for i,o in enumerate(row['objects'],1) if o['instance_id']==obj['instance_id'])
            mass=F.interpolate(torch.tensor(valid&(instance==int(obj['instance_id']))).float()[None,None],size=grid.shape[:2],mode='area')[0,0]
            recall=float((mass*(grid[:,:,slot]>0)).sum()/mass.sum())
            inside=bool((g[0,slot,:3].abs()<=g[0,slot,3:6]/2).all())
            selected_instances.append(dict(id=sid,instance_id=obj['instance_id'],name=obj['name'],
                support_recall=recall,camera_inside_aabb=inside,interpretation='broad geometric support, not predicted visible segmentation'))
        for mode in ('wrong_category','wrong_position'):
            altered,change=condition_variant((category,g,mask),mode)
            keep=~fully_outside(g)[0];keep[0]=True
            for a,b in zip(altered,(category,g,mask)):torch.testing.assert_close(a[:,keep],b[:,keep],rtol=0,atol=0)
            assert fully_outside(altered[1][0,change['changed_slots']]).all()
            control_rows.append(dict(id=sid,**change))
    a=read(Path(plan['base_snapshot'])/'transformer/config.json')
    world=WorldBranch(a['num_attention_heads']*a['attention_head_dim'])
    state=load_file(str(DEFAULT_INITIAL/'world.safetensors'),device='cpu');load_world(world,state,read(DEFAULT_INITIAL/'config.json'))
    assert all(torch.equal(value,world.state_dict()[key]) for key,value in state.items())
    timing=[]
    for path in (ROOT/'qwen_edit_pano/outputs/object_world_id_ab_A_v1/train_log.jsonl',
                 DEFAULT_INITIAL.parent/'train_log.jsonl'):
        logs=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tail=logs[-201:];durations=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(tail,tail[1:]) if b['step']==a['step']+1]
        timing.append(dict(source=str(path),gpus=logs[-1].get('distributed_world_size',1),
            median_seconds_per_step=statistics.median(durations),mean_seconds_per_step=statistics.mean(durations),
            old_peak_allocated_gpu_gib=logs[-1]['peak_gpu_gib'],new_model_runtime_measured=False))
    result=dict(train_images=32,val_images_in_optimizer=0,reviewed_input_states=len(ids),projection_policy=POLICY,
        projection_checks=rows,selected_instances=selected_instances,offscreen_controls=control_rows,
        selected_total=len(selected_instances),selected_with_support=sum(r['support_recall']>0 for r in selected_instances),
        selected_fallback_camera_inside=sum(r['camera_inside_aabb'] for r in selected_instances),
        legacy_shared_weights_exact=True,
        additional_parameters=sum(p.numel() for k,p in world.named_parameters() if '.local.' in k),
        total_world_parameters=sum(p.numel() for p in world.parameters()),historical_timing=timing,
        cpu_seconds=time.monotonic()-started,gpu_jobs_run=0,downloads=0,hashes_computed=False)
    atomic_json(OUT/'CPU_AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('projection_checks','selected_instances','offscreen_controls')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
