"""Versioned, equal-scale galleries of actual pilot worlds; CPU only."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import read, write, sha, validate_graph
from .assembly_review import STYLE, world_card, link
from .batch_world_review import interactive_boxes
from .semantic_layout_visual import erp_map, color

LABELS={'baseline':'普通联合补全','joint':'联合场景程序','needs':'需求与实现机制',
    'prior':'相同候选：不使用观测似然','posterior':'相同候选：功能后验采样',
    'joint_reference':'同候选：均匀联合库存参考','population_posterior':'完整库存：功能与计数观测后验',
    'population_no_function':'同候选：去除功能计数机制'}
LABELS.update(qwen='本地 Qwen',api='API 模型')
LABELS.update(functional='功能条件机制',function_free='切断 H→装配的条件依赖')


def contact_proxies(state, semantic):
    """Report footprint overlap, without claiming actual surface or stability."""
    boxes={o['track_id']:o for o in state['objects']};rows=[]
    for obj in semantic['objects']:
        host=obj['attachment']['host'];kind=obj['attachment']['kind']
        if kind not in ('top','seat','shelf') or host not in boxes:continue
        child=boxes[obj['id']];parent=boxes[host]
        c=np.asarray(child['center_local_m']['value']);s=np.asarray(child['size_aabb_local_m']['value'])
        p=np.asarray(parent['center_local_m']['value']);t=np.asarray(parent['size_aabb_local_m']['value'])
        overlap=np.maximum(0,np.minimum(c+s/2,p+t/2)-np.maximum(c-s/2,p-t/2))
        rows.append(dict(child=obj['id'],host=host,kind=kind,
            child_footprint_on_host_AABB_fraction=float(overlap[0]*overlap[2]/(s[0]*s[2])),
            scope='AABB_projection_proxy_not_surface_or_stability_verification'))
    return rows


def shared_bounds(states):
    lo=np.zeros(3);hi=np.zeros(3)
    for state in states:
        for obj in state['objects']:
            c=obj['center_local_m']['value'];s=obj['size_aabb_local_m']['value']
            if c is None or s is None:continue
            lo=np.minimum(lo,np.asarray(c)-np.asarray(s)/2);hi=np.maximum(hi,np.asarray(c)+np.asarray(s)/2)
    return {axis:[float(lo[i]-.5),float(hi[i]+.5)] for i,axis in enumerate(('x','y','z'))}


def footprint(state, bounds, hfov, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(7,7));ax.set_aspect('equal')
    ax.set_xlim(bounds['x']);ax.set_ylim(bounds['z']);ax.grid(alpha=.18)
    ax.set_xlabel('camera X / metres');ax.set_ylabel('camera Z / metres')
    ax.set_title(title+'\nShared axes; AABB footprints, no inferred room boundary',fontsize=10)
    names=[];new_index=0
    for obj in state['objects']:
        c=obj['center_local_m']['value'];s=obj['size_aabb_local_m']['value']
        if c is None or s is None:continue
        visible=obj['local_evidence']=='visible'
        ax.add_patch(Rectangle((c[0]-s[0]/2,c[2]-s[2]/2),s[0],s[2],
            edgecolor=color(obj),facecolor=color(obj),alpha=.10 if visible else .23,lw=1.3))
        if visible:label=obj['track_id'].replace('obs_','O')
        else:
            new_index+=1;label=f'N{new_index:02d}'
            names.append(dict(label=label,id=obj['track_id'],category=obj.get('raw_category'),
                description=obj.get('description'),center=c,size=s))
        ax.annotate(label,(c[0],c[2]),fontsize=7,color=color(obj),xytext=(2,2),textcoords='offset points')
    ray=min(1.5,max(bounds['z'][1],1.))
    t=np.tan(np.deg2rad(hfov/2))
    for sign in (-1,1):ax.plot([0,sign*t*ray],[0,ray],color='#8492a6',lw=1)
    ax.scatter([0],[0],c='black',s=15);ax.annotate('camera',(0,0),fontsize=8)
    fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)
    return names


def build(root, backend='qwen', needs_root=None):
    root=Path(root).resolve();pilot=read(root/'pilot.json');methods=pilot['methods']
    branches=[];hashes=set();images=set();configs=[];states=[]
    for method in methods:
        base=Path(needs_root).resolve() if method=='needs' and needs_root is not None else root/method
        config=read(base/'experiment.json');configs.append(config)
        if method=='needs' and needs_root is not None and config.get('completion_model')!='needs':
            raise ValueError('--needs-root must identify a needs experiment, not another method')
        images.add(sha(base/'input.png'));hashes.add(sha(base/backend/'observer3d/G_obs.json'))
        observed=read(base/backend/'observer3d/G_obs.json');completion=base/backend/'completion'
        summary=read(completion/'summary.json') if (completion/'summary.json').is_file() else dict(status='not_run',worlds=[])
        worlds=[]
        for record in summary.get('worlds',[])+summary.get('blocked_worlds',[]):
            folder=completion/record['folder'];state=validate_graph(read(folder/record.get('state_file','G_full.json')))
            if state['objects'][:len(observed['objects'])]!=observed['objects']:
                raise ValueError('Observed objects changed in '+str(folder))
            states.append(state)
            worlds.append(dict(record=record,folder=folder,state=state,semantic=read(folder/'semantic_world.json')))
        branches.append(dict(method=method,base=base,completion=completion,summary=summary,worlds=worlds,observed=observed))
    if len(hashes)!=1 or len(images)!=1:raise ValueError('Methods must share exactly the same RGB and G_obs')
    if len({(c['seed'],c['hfov'],tuple(c['image_size']),c['worlds']) for c in configs})!=1:
        raise ValueError('Method seed/camera/requested world count differs')
    index=1
    while (root/f'visual_compare_{index:03d}').exists():index+=1
    out=root/f'visual_compare_{index:03d}';out.mkdir()
    bounds=shared_bounds(states or [branches[0]['observed']]);records=[];figures={}
    functional=methods==['prior','posterior']
    population=pilot.get('comparison')=='population_v3_fixed_domain'
    backend_pair=pilot.get('comparison','').startswith('sdscene_backend_')
    heading=('完整库存 v3：联合参考 / 功能后验 / 去除功能机制' if population else
        '相同候选目录：先验 / 功能后验实际生成对比' if functional else '实际补全物体：普通补全 / 联合程序 / 需求机制')
    if backend_pair:
        heading='Qwen / API：相同库存的布局能力对照' if pilot['comparison'].endswith('layout_control') else 'Qwen / API：识别与机制参与的端到端对照'
        if pilot['comparison'].endswith('function_free'):heading='Qwen / API：切断 H→装配后的生成对照'
    if pilot.get('comparison','').startswith('modular_edge_ablation_'):
        heading='功能条件生成 / 切断 H→装配：'+pilot['comparison'].split('_')[-1]
    parts=['<!doctype html><meta charset="utf-8"><title>实际物体世界对比</title>',STYLE,
        '<style>.branches{grid-template-columns:repeat(3,minmax(0,1fr))}.branch{padding:10px}iframe.scene{height:460px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}table{font-size:12px;width:100%}@media(max-width:1100px){.branches{grid-template-columns:1fr}}</style>',
        '<h1>'+heading+'</h1>',
        '<style>.branches{grid-template-columns:repeat('+str(len(methods))+',minmax(0,1fr))}</style>',
        '<p>各列使用相同观测与相同坐标范围。蓝色为观测，橙色为新增，红色为未映射新增。每个候选按本方法采样顺序展示，跨方法没有真值配对。</p>',
        '<p>这是世界状态与 AABB 可视化；没有生成 RGB 全景。没有 G_full 的分支明确显示失败。</p>',
        '<img class="input" alt="局部观测 RGB" src="'+link(root/methods[0]/'input.png',out)+'">',
        '<p><a href="synchronized_3d.html">打开全部实际世界的三维对比（统一坐标，可同步旋转）</a> | <a href="review.json">数据与几何代理诊断</a></p>',
        '<div class="branches">']
    ablation_check=dict(applicable=functional or population,verified=False)
    if functional:
        model_paths={method:root/method/backend/'completion/causal_model.json' for method in methods}
        missing=[method for method,path in model_paths.items() if not path.is_file()]
        ablation_check['missing_model_branches']=missing
        if missing:
            parts.insert(-1,'<p class="missing">概率对照尚未建立：'+escape(', '.join(missing))+
                ' 未产生候选模型。下方显示原始失败原因；本页不表示生成成功或概率收益。</p>')
        else:
            prior_model=read(model_paths['prior']);posterior_model=read(model_paths['posterior'])
            if prior_model['scene_program_states']!=posterior_model['scene_program_states'] or prior_model['functional_parameters']!=posterior_model['functional_parameters']:
                raise ValueError('Probability ablation must share the exact candidate grammar and numerical parameters')
            ablation_check['verified']=True
        audit=root/'posterior'/backend/'completion/probability_audit.html'
        if audit.is_file():
            parts.insert(-1,'<p><a href="'+link(audit,out)+'">数值概率、证据来源与逐需求干预</a></p>')
    if population:
        paths={m:root/m/backend/'completion/causal_model.json' for m in methods}
        missing=[m for m,path in paths.items() if not path.is_file()]
        ablation_check['missing_model_branches']=missing
        if missing:
            parts.insert(-1,'<p class="missing">对照未全部建立：'+escape(', '.join(missing))+'。下方保留真实失败记录。</p>')
        else:
            models=[read(path) for path in paths.values()];base_model=models[0]
            if any(m['scene_program_states']!=base_model['scene_program_states'] or
                   m['population_parameters']!=base_model['population_parameters'] or
                   m['observed_counts']!=base_model['observed_counts'] for m in models[1:]):
                raise ValueError('Population comparison must share exact programs, count observations and parameters')
            ablation_check['verified']=True
        audit=root/'population_posterior'/backend/'completion/probability_audit.html'
        if audit.is_file():parts.insert(-1,'<p><a href="'+link(audit,out)+'">完整库存概率、零新增概率与参数</a></p>')
    for branch in branches:
        method=branch['method'];summary=branch['summary'];parts.append('<section class="branch"><h2>'+escape(LABELS.get(method,method))+'</h2>')
        parts.append('<p>'+escape(summary['status'])+'；新增实例：'+escape(str(summary.get('proposed_sample_counts',[])))+'</p>')
        if not branch['worlds']:
            parts.append('<div class="missing">没有实际三维补全结果。<p>'+escape(summary.get('error','未运行或未完成'))+'</p></div>')
            for path in sorted((branch['completion']/'calls').glob('furniture_*.json')):
                if path.name.endswith(('.request.json','.validation.json')):continue
                parts.append('<details><summary>'+escape(path.name)+' 原始提议</summary><pre>'+escape(path.read_text())+'</pre></details>')
        for world in branch['worlds']:
            record=world['record'];visual=out/method/record['folder'];visual.mkdir(parents=True)
            title=method+' / '+record['folder']
            figure=interactive_boxes(world['state'],configs[0]['hfov'],configs[0]['image_size'],visual/'scene_3d.html',title=title,scene_bounds=bounds)
            figures[(method,record['folder'])]=figure
            erp_map(world['state'],configs[0]['hfov'],configs[0]['image_size'],visual/'erp_state_map.png')
            names=footprint(world['state'],bounds,configs[0]['hfov'],visual/'layout_xz.png',title)
            proxies=contact_proxies(world['state'],world['semantic'])
            parts.append(world_card(world['folder'],record,out,visual_override=visual))
            parts.append('<details open><summary>新增物体：N 编号对应上图</summary><table><tr><th>图中编号</th><th>实际提议名称</th><th>完整尺寸 m</th></tr>')
            for item in names:
                parts.append('<tr><td>'+item['label']+'</td><td>'+escape(item['description'] or item['category'] or item['id'])+'</td><td>'+escape(str(np.round(item['size'],3).tolist()))+'</td></tr>')
            parts.append('</table></details>')
            records.append(dict(method=method,world=record['folder'],objects=names,contact_proxies=proxies,
                original_state=str(world['folder']/record.get('state_file','G_full.json')),
                original_state_sha256=sha(world['folder']/record.get('state_file','G_full.json'))))
        parts.append('</section>')
    parts.append('</div>')
    if any(row['child_footprint_on_host_AABB_fraction']<.5 for record in records for row in record['contact_proxies']):
        parts.append('<p class="missing">部分受支撑物体只有很小一部分投影落在宿主 AABB 上。零交叠不代表稳定支撑；具体投影比例见 review.json。这是几何代理诊断。</p>')
    (out/'index.html').write_text(''.join(parts))
    write(out/'review.json',dict(source=str(root),backend=backend,shared_G_obs_sha256=next(iter(hashes)),
        shared_RGB_sha256=next(iter(images)),shared_bounds=bounds,worlds=records,
        branch_status={b['method']:b['summary']['status'] for b in branches},
        method_roots={b['method']:str(b['base']) for b in branches},probability_ablation_check=ablation_check,
        no_worlds_fabricated=True,original_states_modified=False,causal_gain_validated=False))
    _synchronized(figures,methods,bounds,out/'synchronized_3d.html')
    return out/'index.html'


def _synchronized(figures,methods,bounds,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    count=max((int(world.split('_')[-1]) for _,world in figures),default=1)
    titles=[]
    for row in range(1,count+1):
        for method in methods:
            titles.append(method+' / '+f'world_{row:02d}' if (method,f'world_{row:02d}') in figures else method+' / no exported world')
    fig=make_subplots(rows=count,cols=len(methods),specs=[[{'type':'scene'} for _ in methods] for _ in range(count)],
        subplot_titles=titles,horizontal_spacing=.025,vertical_spacing=.04)
    for row in range(1,count+1):
        for col,method in enumerate(methods,1):
            source=figures.get((method,f'world_{row:02d}'))
            if source is None:continue
            for trace in source.data:
                trace=go.Scatter3d(trace);trace.showlegend=False;fig.add_trace(trace,row=row,col=col)
    span={k:bounds[k][1]-bounds[k][0] for k in bounds}
    fig.update_scenes(xaxis=dict(title='X right (m)',range=bounds['x']),
        yaxis=dict(title='Z forward (m)',range=bounds['z']),zaxis=dict(title='Y image-up (m)',range=bounds['y']),
        aspectmode='manual',aspectratio=dict(x=span['x'],y=span['z'],z=span['y']))
    fig.update_layout(height=count*500,title='Actual exported worlds — shared axes; empty panels have no exported geometry',margin=dict(l=0,r=0,t=85,b=0))
    script="""var gd=document.getElementById('{plot_id}');var busy=false;
gd.on('plotly_relayout',function(ev){if(busy)return;var key=Object.keys(ev).find(k=>/^scene[0-9]*\\.camera$/.test(k));if(!key)return;
var update={};Object.keys(gd.layout).filter(k=>/^scene[0-9]*$/.test(k)).forEach(k=>update[k+'.camera']=ev[key]);
busy=true;Plotly.relayout(gd,update).then(()=>{busy=false;});});"""
    fig.write_html(str(output),include_plotlyjs=True,post_script=script,auto_open=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--backend',choices=['api','qwen'],default='qwen')
    parser.add_argument('--needs-root',type=Path,help='Optional new needs experiment with exactly matched observations/camera/seed')
    args=parser.parse_args()
    print(build(args.root,args.backend,args.needs_root))


if __name__=='__main__':main()
