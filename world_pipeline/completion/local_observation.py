"""Observed-only extraction entry and read-only 2D/3D diagnostic gallery.

The report reads observation/ only, never a completed or candidate world.
Missing 3D measurements stay missing; no coordinates are filled or optimized.
"""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil

from PIL import Image,ImageDraw

from ..common import ROOT,new_output,read,write,sha,validate_graph
from . import assembly_presentation_visual as visual
from .assembly_multiscene import DEFAULT_INPUTS,initialize_suite,verify_suite

KIND='local_observation_only_v1'
KITCHEN='8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'
NAMES={'8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':'休息区',
    'X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c':'客厅',
    'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6':'卫生间',
    'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a':'卧室',KITCHEN:'厨房'}


def entry_hashes():
    files=[Path(__file__).resolve(),ROOT/'qwen_edit_pano/scripts/local_observation.sh']
    return {str(p):sha(p) for p in files}


def prepare(root,inputs=DEFAULT_INPUTS):
    suite=initialize_suite(Path(root),Path(inputs),mode='single',worlds=1)
    write(Path(root)/'observation_only_run.json',dict(kind=KIND,scheduled_phases=['observer2d','observer3d'],
        observer='Qwen_2D_plus_official_WildDet3D',hidden_state_generation=False,API_used=False,
        causal_mechanisms_estimated=False,layout_solver_used=False,entry_sha256=entry_hashes()))
    return suite


def check_run(root):
    root=Path(root).resolve();meta=read(root/'observation_only_run.json')
    if meta.get('kind')!=KIND or meta.get('scheduled_phases')!=['observer2d','observer3d']:raise ValueError('Not a local-observation-only run')
    if any(not Path(p).is_file() or sha(Path(p))!=h for p,h in meta['entry_sha256'].items()):raise ValueError('Observation-only entry changed after preparation; use a new run')
    return verify_suite(root)


def visible_state(value,metric=False):
    state=validate_graph(value)
    if state.get('scope')!='observed' or state.get('stage')!='prediction' or state.get('scene_gt_used') is not False:
        raise ValueError('Require predicted observed state, not a full/hidden/GT graph')
    if any(o['local_evidence']!='visible' for o in state['objects']):raise ValueError('Observed state contains a non-visible object; no silent filtering')
    if metric and any(not o[k]['valid'] for o in state['objects'] for k in ('center_local_m','size_aabb_local_m')):
        raise ValueError('Incomplete 3D observation: missing geometry remains unknown, never filled')
    return state


def bind_observations(two,three):
    if two['id']!=three['id'] or two['frame']!=three['frame']:raise ValueError('2D/3D scene or camera-frame mismatch')
    if [o['track_id'] for o in two['objects']]!=[o['track_id'] for o in three['objects']]:raise ValueError('2D/3D instance mismatch; no object dropping')
    for a,b in zip(two['objects'],three['objects']):
        for key in ('category_id','raw_category','description','bbox2d_visible_xyxy_norm'):
            if a.get(key)!=b.get(key):raise ValueError('WildDet observation changed 2D identity or box: '+key)


def sources(primary,replacement=None):
    primary=Path(primary).resolve();suite=read(primary/'suite.json');selected={s['id']:primary for s in suite['scenes']}
    if replacement:
        replacement=Path(replacement).resolve();other=read(replacement/'suite.json')
        for r in other['scenes']:
            sid=r['id']
            if sid not in selected:raise ValueError('Replacement is not an input of the primary suite: '+sid)
            original=next(s for s in suite['scenes'] if s['id']==sid)
            a=primary/'scenes'/sid/'observation/input.png';b=replacement/'scenes'/sid/'observation/input.png'
            with Image.open(a) as ia,Image.open(b) as ib:
                if ia.size!=ib.size or ia.convert('RGB').tobytes()!=ib.convert('RGB').tobytes() or original['hfov']!=r['hfov']:
                    raise ValueError('Replacement RGB/HFOV mismatch: '+sid)
            selected[sid]=replacement
    return suite,selected


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def report(primary,out,replacement=None,renderer=None):
    primary=Path(primary).resolve();out=Path(out).resolve();suite,selected=sources(primary,replacement)
    roots=set(selected.values())|{primary}
    if any(out==r or r in out.parents for r in roots):raise ValueError('Keep reports outside source runs')
    # Snapshot only observations and metadata. No completion/ files are read.
    files={str(root/'suite.json'):sha(root/'suite.json') for root in roots}
    for sid,root in selected.items():
        obs=root/'scenes'/sid/'observation'
        for p in obs.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.txt','.png'):files[str(p)]=sha(p)
        for phase in ('observer2d','observer3d'):
            p=root/'phase_records'/(sid+'_'+phase+'_qwen.json')
            if p.is_file():files[str(p)]=sha(p)
    out=new_output(out);records=[];cards=[];instances=[]
    render=renderer or visual.render_world
    css='<style>body{font:16px/1.7 system-ui,sans-serif;margin:24px;color:#17263d}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cdd5e1;padding:8px;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.warning{padding:12px;background:#fff0cb}.small{font-size:13px;overflow-wrap:anywhere}iframe{width:100%;height:700px;border:0}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>'
    for s in suite['scenes']:
        sid=s['id'];obs=selected[sid]/'scenes'/sid/'observation';dest=out/'scenes'/sid;dest.mkdir(parents=True)
        shutil.copyfile(obs/'input.png',dest/'input.png')
        p2=obs/'qwen/observer2d/G_2d.json';p3=obs/'qwen/observer3d/G_obs.json';two=None;three=None;error=None;rows=[]
        status='not_observed';source2summary=read(p2.parent/'summary.json') if (p2.parent/'summary.json').exists() else {}
        source3summary=read(p3.parent/'summary.json') if (p3.parent/'summary.json').exists() else {}
        phases={}
        for phase in ('observer2d','observer3d'):
            path=selected[sid]/'phase_records'/(sid+'_'+phase+'_qwen.json')
            phases[phase]=read(path) if path.exists() else {}
        try:
            if p2.exists():
                two=visible_state(read(p2));shutil.copyfile(p2,dest/'G_2d.json');status='2d_only'
            if p3.exists():
                if two is None:raise ValueError('3D state without its 2D identity source')
                if source2summary.get('status')=='failed' or source3summary.get('status')=='failed':
                    raise ValueError('Saved 3D artifact belongs to a failed observation stage; not a completed observation')
                three=visible_state(read(p3),metric=True);bind_observations(two,three);shutil.copyfile(p3,dest/'G_obs.json')
                status='observed_state_exported_needs_review' if three['objects'] else 'empty_observation_needs_review'
        except ValueError as exc:error=str(exc);status='invalid_observation';three=None
        if not error and three is None:
            if source2summary.get('status')=='failed' or phases['observer2d'].get('status')=='failed':
                status='observer2d_failed';error=source2summary.get('error') or phases['observer2d'].get('error') or '2D observation failed'
            elif source3summary.get('status')=='failed' or phases['observer3d'].get('status')=='failed':
                status='observer3d_failed';error=source3summary.get('error') or phases['observer3d'].get('error') or '3D observation failed'
            elif phases['observer3d'].get('status')=='blocked_dependency':
                status='blocked_3d_dependency';error=phases['observer3d'].get('error') or '3D blocked by missing inputs: '+str(phases['observer3d'].get('missing_inputs',[]))
        grounding=read(obs/'qwen/observer2d/grounding.json') if (obs/'qwen/observer2d/grounding.json').exists() else {}
        floor={r['id']:r['on_floor'] for r in grounding.get('instances',[])}
        if two is not None:
            labels=visual.label_rows(two)
            with Image.open(obs/'input.png') as im:
                canvas=im.convert('RGB');draw=ImageDraw.Draw(canvas);occupied=[]
                for obj,label in zip(two['objects'],labels):
                    box=obj['bbox2d_visible_xyxy_norm']['value']
                    if box is None:continue
                    x1,y1,x2,y2=[v*k for v,k in zip(box,[canvas.width,canvas.height]*2)]
                    draw.rectangle((x1,y1,x2,y2),outline=visual.OBSERVED,width=3)
                    visual.put_label(draw,(x1,y1),label['label'],visual.OBSERVED,occupied,canvas.width,canvas.height,size=17)
                canvas.save(dest/'boxes2d.png')
            metric={o['track_id']:o for o in three['objects']} if three is not None else {}
            for obj,label in zip(two['objects'],labels):
                m=metric.get(obj['track_id']);center=m['center_local_m']['value'] if m else None;size=m['size_aabb_local_m']['value'] if m else None
                row=dict(scene=sid,id=obj['track_id'],label=label['label'],raw_category=obj.get('raw_category'),
                    dataset_category=label['category'],bbox2d=obj['bbox2d_visible_xyxy_norm']['value'],
                    center_m=center,full_AABB_size_m=size,on_floor=floor.get(obj['track_id']),generated=False)
                instances.append(row);rows.append([row[k] for k in ('label','raw_category','dataset_category','bbox2d','center_m','full_AABB_size_m','on_floor')])
        if three is not None:
            original_legend=visual.LEGEND
            try:
                visual.LEGEND='Blue: predicted visible objects ONLY | No hidden-state completion'
                with Image.open(obs/'input.png') as im:checks=render(three,im,s['hfov'],dest/'views')
            finally:visual.LEGEND=original_legend
            if checks.get('labels')!=len(three['objects']):raise ValueError('Renderer omitted an observed label')
            checks.update(observation_only=True,generated_instances=0,palette=[visual.OBSERVED]);write(dest/'views/visual_validation.json',checks)
            # Replace only this new report's generic world page with observation wording.
            page=dest/'views/index.html'
            if page.exists():
                text=page.read_text().replace('局部观测与完整世界状态','仅局部可见观测状态').replace('蓝：观测；橙：可映射新增；红：未映射新增。','所有对象均为蓝色的可见观测预测；没有新增对象。')
                page.write_text(text)
        notes=['这里只显示预测到的可见实例；未识别到不等于图中不存在。三维尺度、重力、地面与真实遮挡未独立核验。']
        if sid==KITCHEN:notes.append('此厨房曾漏掉中岛、橱柜、灶具/抽油烟机等。请对照原图验收；on_floor为模型预测，不会由本报告自动纠正。')
        record=dict(scene=sid,name=NAMES.get(sid,sid),status=status,observed_2d=len(two['objects']) if two is not None else None,
            observed_3d=len(three['objects']) if three is not None else None,generated_instances=0,source_run=str(selected[sid]),
            source_2d_sha256=sha(p2) if p2.exists() else None,source_3d_sha256=sha(p3) if p3.exists() else None,
            error=error,observation_quality_verified=False,notes=notes,observer2d_status=source2summary.get('status'),observer3d_status=source3summary.get('status'),
            observer2d_phase=phases['observer2d'].get('status'),observer3d_phase=phases['observer3d'].get('status'))
        records.append(record);write(dest/'observation_review.json',record)
        page='<!doctype html><meta charset="utf-8">'+css+'<h1>'+escape(record['name'])+'：仅局部观测</h1><p><a href="../../index.html">返回五图观测总览</a></p>'
        page+='<div class="warning">'+escape(' '.join(notes))+'</div><p>二维数量：'+str(record['observed_2d'])+'；三维数量：'+str(record['observed_3d'])+'；补全新增：0。</p><div class="grid"><div><h2>原局部图</h2><img src="input.png"></div>'
        if two is not None:page+='<div><h2>原图上的二维识别框</h2><img src="boxes2d.png"></div>'
        page+='</div>'
        if three is not None:
            page+='<p><a href="G_obs.json">原样G_obs</a> · <a href="views/index.html">全部观测视图</a> · <a href="views/scene_3d.html">交互3D</a></p>'
            for name,title in [('local_projection.png','观测三维框重投影'),('layout_xz.png','仅观测XZ平面图'),('direction_projections.png','仅观测六方向投影'),('scene_3d.png','仅观测3D'),('erp_overlay.png','仅观测2:1 ERP状态投影')]:
                page+='<h2>'+title+'</h2><a href="views/'+name+'"><img loading="lazy" src="views/'+name+'"></a>'
            page+='<iframe loading="lazy" src="views/scene_3d.html"></iframe>'
        else:page+='<p class="warning">没有有效的完整3D观测；不生成虚构坐标或3D图。'+escape(error or source3summary.get('error',''))+'</p>'
        if two is not None:page+='<p><a href="G_2d.json">原样G_2d</a></p>'
        page+=table(['编号+类别','原始类别','数据集类别','二维框','相机中心/m','完整AABB尺寸/m','直接落地预测'],rows)
        page+='<p class="small">来源：'+escape(str(obs))+'</p>';(dest/'index.html').write_text(page)
        folder='scenes/'+sid;preview='views/overview.png' if three is not None else 'boxes2d.png' if two is not None else 'input.png'
        cards.append('<article><h2>'+escape(record['name'])+'</h2><p>2D='+str(record['observed_2d'])+'；3D='+str(record['observed_3d'])+'；新增=0</p><a href="'+folder+'/index.html"><img loading="lazy" src="'+folder+'/'+preview+'">打开原图、二维框与观测状态</a></article>')
        print(json.dumps(dict(scene=sid,status=status,observed_2d=record['observed_2d'],observed_3d=record['observed_3d'],generated=0),ensure_ascii=False),flush=True)
    write(out/'results.json',dict(kind=KIND,records=records,hidden_state_generation=False,model_called=False,
        observation_reused=True,API_used=False,semantic_quality_certified=False))
    write(out/'instances.json',instances);write(out/'source_hashes.json',files)
    with (out/'instances.csv').open('w',newline='') as f:
        if instances:w=csv.DictWriter(f,fieldnames=list(instances[0]));w.writeheader();w.writerows(instances)
    if not instances:(out/'instances.csv').write_text('')
    page='<!doctype html><meta charset="utf-8"><title>仅局部观测状态</title>'+css+'<h1>局部RGB → 可见观测状态 G_obs</h1><p>本页不含新增状态，不做因果补全，不调用API。共享观察器为Qwen二维识别＋WildDet3D三维估计。这里只重绘已有结果，不代表重新识别或提高了检测质量。</p><p>对象全部为蓝色，标签为编号＋类别。未知周围空间留空；ERP只是局部观测状态投影，不是生成的全景RGB。框和坐标原样保留，不做布局优化。</p><p><a href="instances.csv">逐实例CSV</a> · <a href="instances.json">逐实例JSON</a> · <a href="results.json">状态及来源</a> · <a href="validation.json">核验记录</a></p>'
    page+=table(['输入','2D实例','3D实例','补全新增','状态'],[[r['name'],r['observed_2d'],r['observed_3d'],0,r['status']] for r in records])
    page+='<div class="grid">'+''.join(cards)+'</div>';(out/'index.html').write_text(page)
    write(out/'validation.json',dict(status='checking'))
    changed=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h];broken=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,v in attrs:
                if key in ('href','src') and v and not v.startswith(('http:','https:','data:','#','javascript:')) and not (current.parent/v.split('#')[0]).exists():broken.append(str(current)+': '+v)
    for current in out.rglob('*.html'):Links().feed(current.read_text())
    write(out/'validation.json',dict(source_files_unchanged=not changed,source_files_checked=len(files),changed_sources=changed,
        broken_links=broken,model_calls=0,generated_instances=0,completion_files_read=False,all_exported_instances_visible=True,
        state_geometry_modified=False,missing_3d_fabricated=False,scenes=len(records)))
    if changed or broken:raise AssertionError('Source changed or local links broken; inspect validation.json')
    return out/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    x=sub.add_parser('prepare');x.add_argument('--root',type=Path,required=True);x.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS)
    x=sub.add_parser('check');x.add_argument('--root',type=Path,required=True)
    x=sub.add_parser('report');x.add_argument('--source',type=Path,required=True);x.add_argument('--replacement',type=Path);x.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.inputs);print(a.root)
    elif a.action=='check':check_run(a.root);print('Observation-only run verified')
    else:print(report(a.source,a.out,a.replacement))


if __name__=='__main__':main()
