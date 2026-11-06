"""CPU summaries of paired fixed-noise records; no semantic success claims."""
import html
import math
from pathlib import Path

import numpy as np

from ..build_polished_world import read,atomic_json


def report(output,protocol):
    rows=[];paired=[];groups={}
    for step in protocol['steps']:
        for case in protocol['panel']['cases']:
            for schedule in protocol['schedule']:
                directory=output/'records'/f'{step:06d}'/case['id']/f"{schedule['index']:04d}"
                found=[]
                for mode in protocol['modes']:
                    path=directory/(mode+'.json')
                    if not path.is_file():raise ValueError('Diagnostic incomplete: '+str(path))
                    row=read(path)
                    expected=dict(step=step,id=case['id'],split=case['split'],mode=mode,scheduler_index=schedule['index'],
                                  sigma=schedule['sigma'],latent_seed=case['latent_seed'])
                    if any(row[k]!=v for k,v in expected.items()):raise ValueError('Record identity differs: '+str(path))
                    if not all(math.isfinite(row[k]) for k in ('total','object_region','relative_prediction_change')):
                        raise ValueError('Nonfinite diagnostic record')
                    if len(row['selected_object_errors'])!=len(case['objects']):raise ValueError('Instance diagnostic count differs')
                    row['selected_object_mean']=float(np.mean(row['selected_object_errors']))
                    found.append(row);rows.append(row)
                correct=next(r for r in found if r['mode']=='correct')
                repeat=next(r for r in found if r['mode']=='repeat_correct')
                for row in found:
                    changed=set(row.get('perturbation',{}).get('changed_slots',[]))
                    selected=[i for i,o in enumerate(row.get('selected_objects',[])) if o['slot'] in changed]
                    changed_gap=(float(np.mean([row['selected_object_errors'][i]-correct['selected_object_errors'][i] for i in selected]))
                                 if selected else None)
                    pair=dict(step=step,id=case['id'],split=case['split'],sigma=schedule['sigma'],mode=row['mode'],
                        total_gap=row['total']-correct['total'],region_gap=row['object_region']-correct['object_region'],
                        selected_gap=row['selected_object_mean']-correct['selected_object_mean'],
                        relative_total_gap_pct=(row['total']/max(correct['total'],1e-20)-1)*100,
                        relative_region_gap_pct=(row['object_region']/max(correct['object_region'],1e-20)-1)*100,
                        relative_prediction_change=row['relative_prediction_change'],
                        changed_selected_instances=len(selected),changed_selected_gap=changed_gap,
                        repeat_prediction_floor=repeat['relative_prediction_change'])
                    paired.append(pair)
                    groups.setdefault((step,case['split'],schedule['sigma'],row['mode']),[]).append(pair)
    summary=[]
    for key,values in groups.items():
        step,split,sigma,mode=key
        summary.append(dict(step=step,split=split,sigma=sigma,mode=mode,samples=len(values),
            **{metric:float(np.mean([r[metric] for r in values])) for metric in
               ('total_gap','region_gap','selected_gap','relative_total_gap_pct','relative_region_gap_pct','relative_prediction_change','repeat_prediction_floor')},
            fraction_correct_lower_total=float(np.mean([r['total_gap']>0 for r in values]))))
    attention=[]
    for step in protocol['steps']:
        for split in ('train','val'):
            for schedule in protocol['schedule']:
                selected=[r for r in rows if r['step']==step and r['split']==split and r['sigma']==schedule['sigma'] and r['mode']=='correct']
                for block in selected[0]['attention']:
                    attention.append(dict(step=step,split=split,sigma=schedule['sigma'],block=block,
                        **{k:float(np.mean([r['attention'][block][k] for r in selected])) for k in
                           ('global_attention_mass','gt_instance_probability_over_uniform','gt_instance_top1_fraction',
                            'head_mean_entropy_normalized','object_value_common_ratio','update_query_variation_ratio')}))
    atomic_json(output/'RESULTS.json',rows);atomic_json(output/'PAIRED_GAPS.json',paired)
    atomic_json(output/'SUMMARY.json',dict(paired_summary=summary,correct_attention=attention,
        interpretation='Positive wrong/off minus correct loss means correct is better on this fixed target/noise. Check repeat floor. Not generation success; small train/val groups have different image difficulty.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sigmas=[r['sigma'] for r in protocol['schedule']];steps=protocol['steps'];modes=['wrong_category','wrong_position','world_off']
    for metric,filename,label in [('relative_total_gap_pct','DIAGNOSTIC.png','Full objective gap (%)'),
                                  ('relative_region_gap_pct','REGION_GAPS.png','Object-region loss gap (%)')]:
        fig,axes=plt.subplots(2,len(sigmas),figsize=(5*len(sigmas),8),squeeze=False)
        for row,split in enumerate(('train','val')):
            for col,sigma in enumerate(sigmas):
                ax=axes[row,col]
                for m,mode in enumerate(modes):
                    values=[next(r[metric] for r in summary if r['step']==step and r['split']==split and r['sigma']==sigma and r['mode']==mode) for step in steps]
                    ax.bar(np.arange(len(steps))+(m-1)*.24,values,width=.23,label=mode)
                ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(steps)),[str(s) for s in steps])
                ax.set_title(f'{split}: sigma={sigma:.3f}');ax.set_xlabel('Checkpoint step');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
                if row==0 and col==0:ax.legend(fontsize=8)
        fig.suptitle('Paired (wrong/off - correct) / correct: positive favors correct\nTeacher-forced diagnostic only; inspect repeat controls before interpreting small gaps')
        fig.tight_layout();fig.savefig(output/filename,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,block in zip(axes,('20','40','59')):
        for split in ('train','val'):
            values=[float(np.mean([r['gt_instance_probability_over_uniform'] for r in attention if r['step']==step and r['split']==split and r['block']==block])) for step in steps]
            ax.plot(steps,values,marker='o',label=split)
        ax.axhline(1,color='black',ls='--');ax.set_title('Block '+block);ax.set_xlabel('Checkpoint step')
        ax.set_ylabel('GT-instance probability / uniform');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Actual Q/K readout under correct state (sigma-averaged); not semantic accuracy')
    fig.tight_layout();fig.savefig(output/'ATTENTION.png',dpi=150);plt.close(fig)
    def table(entries,keys):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in entries)+'</table>'
    cases=[dict(c,training_exposures={str(s):protocol.get('training_exposures',{}).get(str(s),{}).get(c['id']) for s in protocol['steps']}) for c in protocol['panel']['cases']]
    body=['<!doctype html><meta charset="utf-8"><title>固定噪声条件诊断</title>',
        '<style>body{font-family:system-ui;margin:24px;max-width:1500px}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccc}p{line-height:1.7}</style>',
        '<h1>固定噪声条件诊断：训练图与6张新World验证图</h1>',
        '<p>正值：错误状态/关闭World的误差高于正确状态。先检查repeat_correct的数值底噪；小于或接近底噪的差异不要解读。此处输入含GT加噪潜变量，不是从纯噪声自由生成，也不是编辑成功率。</p>',
        '<p>训练图与验证图的难度不同，重点比较各图内部的配对差值。注意力读出来自实际Q/K，但注意力对应不等于正确生成对象。验证图仅保证不在扫描到的历史World诊断中；此前的全景baseline曝光在PANEL.json中记录。</p>',
        '<img src="DIAGNOSTIC.png"><img src="REGION_GAPS.png"><img src="ATTENTION.png">',
        '<h2>配对差值</h2>',table(summary,['step','split','sigma','mode','samples','relative_total_gap_pct','relative_region_gap_pct','relative_prediction_change','repeat_prediction_floor']),
        '<h2>固定样本</h2>',table(cases,['id','split','scene_id','prior_world_diagnostic','training_exposures']),
        '<p>读法：训练图也无类别/位置差距，优先查条件接口/监督；训练图有差距而验证图无差距，优先查泛化和布局多样性；类别有差距而位置无差距，优先查几何绑定。样本量小，不自动判定唯一原因。</p>']
    (output/'report.html').write_text('\n'.join(body))
    atomic_json(output/'COMPLETE.json',dict(version=protocol['version'],condition_records=len(rows),
        train=protocol['panel']['train_count'],val=protocol['panel']['val_count'],generated_panoramas=0,optimizer_steps=0,
        note='Numeric diagnostic complete; generation grounding/generalization not certified.'))
    print(dict(stage='diagnostic_complete',records=len(rows),output=str(output),training_updates=0),flush=True)
