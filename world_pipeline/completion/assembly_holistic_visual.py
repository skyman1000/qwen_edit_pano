"""Actual v11 inventory, probability, support and 360-degree comparison."""
import argparse
from html import escape
import json
import math
import os
from pathlib import Path

from ..common import read,write
from .assembly_taxonomy import annotate
from .assembly_pilot_visual import shared_bounds,footprint
from .batch_world_review import interactive_boxes
from .semantic_layout_visual import erp_map


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def report(root,baseline_root=None,out=None,*,current_label='v11',gallery_experiments=None,baseline_experiments=None):
    root=Path(root).resolve();out=Path(out).resolve() if out else root/'visual_v11_001'
    if out.exists():raise FileExistsError('Use a new visual --out directory')
    out.mkdir(parents=True);run=read(root/'run.json');sources=[(current_label,root)]
    baselines=[] if baseline_root is None else baseline_root if isinstance(baseline_root,(list,tuple)) else [baseline_root]
    for baseline in baselines:
        baseline_root=Path(baseline).resolve();old=read(baseline_root/'run.json')
        if any(old[k]!=run[k] for k in ('RGB_sha256','G_obs_sha256')):raise ValueError('Comparison observation differs')
        label='v9.1' if old['version']==9 and str(old.get('proposal_interface_revision','')).startswith('v9.1') else 'v'+str(old['version'])
        sources.insert(0,(label,baseline_root))
    branches=[];gallery=[]
    for version,base in sources:
        experiments=run['experiments'] if base==root else (baseline_experiments or ['end_to_end'])
        for exp in experiments:
            config=read(base/exp/'experiment.json')
            for backend in ('qwen','api'):
                c=base/exp/backend/'completion';summary=read(c/'summary.json') if (c/'summary.json').exists() else dict(status='not_run')
                row=dict(version=version,experiment=exp,backend=backend,status=summary['status'],worlds=[],probabilities=read(c/'probability_audit.json') if (c/'probability_audit.json').exists() else None)
                for i in range(read(base/'run.json')['worlds']):
                    w=c/f'world_{i+1:02d}';sem=read(w/'semantic_world.json') if (w/'semantic_world.json').exists() else {}
                    source=w/'G_full.json';complete=source.exists()
                    if not complete:source=w/'G_full_candidates.json'
                    wr=dict(world=w.name,sampled_count=len(sem.get('objects',[])),exported=complete,objects=[],status='exported' if complete else 'blocked_or_failed')
                    if source.exists():
                        state=read(source);obs=read(base/exp/backend/'observer3d/G_obs.json');contracts={o['id']:o for o in sem['objects']}
                        if state['objects'][:len(obs['objects'])]!=obs['objects']:raise ValueError('Changed observed state')
                        if {o['track_id'] for o in state['objects'][len(obs['objects']):]}!=set(contracts):raise ValueError('Inventory changed')
                        for o in state['objects'][len(obs['objects']):]:
                            x,y,z=o['center_local_m']['value'];wr['objects'].append(dict(id=o['track_id'],name=o['description'],rear=z<0,
                                root=contracts[o['track_id']]['attachment']['host'] is None,azimuth_deg=math.degrees(math.atan2(x,z))))
                        numeric=read(w/'numeric_review.json');wr['support_projection_review']=numeric.get('support_projection_review',[])
                        if exp in (gallery_experiments or ('end_to_end','layout_control','rgb_inventory')):
                            gallery.append(dict(id=f'{version}_{exp}_{backend}_{w.name}',row=wr,state=annotate(state,sem),config=config))
                    row['worlds'].append(wr)
                branches.append(row)
    bounds=shared_bounds([g['state'] for g in gallery]);write(out/'review.json',dict(branches=branches,shared_bounds=bounds,
        training_performed=False,causal_gain_validated=False,scope='Counts/directions describe samples, not calibrated completeness or semantic correctness'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for backend,ax in zip(('qwen','api'),axes):
        for r in branches:
            if r['backend']!=backend or r['experiment'] not in ('end_to_end','rgb_inventory'):continue
            vals=[o['azimuth_deg'] for w in r['worlds'] for o in w['objects'] if o['root']]
            ax.hist(vals,bins=np.arange(-180,181,15),histtype='step',linewidth=2,label=r['version']+'/'+r['experiment'])
        ax.axvspan(-run.get('hfov',90)/2,run.get('hfov',90)/2,color='blue',alpha=.1)
        ax.set(title=backend,xlim=(-180,180),xlabel='Original-camera azimuth, forward=0',ylabel='Placement-root records');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'angles.png',dpi=160);plt.close(fig)
    parts=['<!doctype html><meta charset="utf-8"><title>联合功能全景世界对比 v11</title><style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px}img{max-width:100%}iframe{width:100%;height:520px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{border:1px solid #ddd;padding:10px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}</style>',
        '<h1>v11：联合库存与相机空间组织</h1><p>真实样本，原顺序保留空集/重复/失败。蓝：观测；橙：原名在数据集类别表内；紫：原名表外但已映射；红：未映射。图片是状态框，不是 RGB 全景。后方为原相机中心 Z&lt;0，不要求每个角度有家具。</p>',
        '<p><a href="review.json">全部逐世界记录与概率</a></p><img src="angles.png">',
        table(['版本/分支','后端','四次新增','E[N]','P(empty)','后方/全部记录','导出/计划'],[[r['version']+'/'+r['experiment'],r['backend'],[w['sampled_count'] for w in r['worlds']],
            round(r['probabilities']['expected_hidden_instances'],5) if r['probabilities'] else None,round(r['probabilities']['empty_probability'],6) if r['probabilities'] else None,
            str(sum(o['rear'] for w in r['worlds'] for o in w['objects']))+'/'+str(sum(len(w['objects']) for w in r['worlds'])),str(sum(w['exported'] for w in r['worlds']))+'/'+str(len(r['worlds']))] for r in branches]),
        '<p>跨版本候选域不同，差异不是单个箭头的因果效应。RGB 对照仅改变选择阶段图像输入，本地初始 draw seed 相同；服务端 API 未传 seed。H-cut 保留联合库存/空间边缘分布，不能用平均物体数检验 H 的作用。</p>']
    for r in branches:
        parts.append('<details><summary>'+r['version']+'/'+r['experiment']+'/'+r['backend']+' 概率与支撑</summary><pre>'+escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<div class="grid">')
    for g in gallery:
        dest=out/g['id'];dest.mkdir();write(dest/'G_display.json',g['state'])
        labels=footprint(g['state'],bounds,g['config']['hfov'],dest/'layout_xz.png',g['id']);write(dest/'plot_labels.json',labels)
        interactive_boxes(g['state'],g['config']['hfov'],g['config']['image_size'],dest/'scene_3d.html',title=g['id'],scene_bounds=bounds)
        erp_map(g['state'],g['config']['hfov'],g['config']['image_size'],dest/'erp_state_map.png')
        parts.append('<article><b>'+g['id']+' / '+g['row']['status']+'</b><img loading="lazy" src="'+g['id']+'/layout_xz.png"><img loading="lazy" src="'+g['id']+'/erp_state_map.png"><details><summary>旋转 3D 与物体信息</summary><iframe loading="lazy" src="'+g['id']+'/scene_3d.html"></iframe></details></article>')
    parts.append('</div>');(out/'index.html').write_text(''.join(parts).replace('<h1>v11：','<h1>'+current_label+'：').replace('全景世界对比 v11</title>','全景世界对比 '+current_label+'</title>'))
    (root/'index.html').write_text('<meta charset="utf-8"><h1>'+current_label+' 实际补全对比</h1><a href="'+os.path.relpath(out/'index.html',root)+'">打开所有世界、概率、后方与支撑对比</a>')
    return out/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,action='append');p.add_argument('--out',type=Path)
    a=p.parse_args();print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__':main()
