"""Common read-only box diagnostics and RGB-direct/full-pipeline comparison."""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import itertools
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from ..common import new_output,read,write,sha
from .direct_rgb_world import verify,pixel_sha
from .assembly_presentation_visual import render_world,label_rows,font
from .assembly_pilot_visual import shared_bounds
from .scene_completion import view_check


def measure(state,hfov,size):
    """Same box-level measurements for both methods; no correction/selection."""
    objects=state['objects'];hidden={o['track_id'] for o in objects if o['local_evidence']!='visible'}
    checks={o['track_id']:view_check(o['center_local_m']['value'],o['size_aabb_local_m']['value'],hfov,size) for o in objects}
    host_pairs=set()
    for o in objects:
        a=o.get('attachment',{})
        if a.get('host'):host_pairs.add(frozenset((o['track_id'],a['host'])))
    for r in state.get('relations',[]):
        if isinstance(r,dict) and r.get('kind') in ('on','inside','attached_to'):host_pairs.add(frozenset((r['source'],r['target'])))
    overlaps=[]
    for a,b in itertools.combinations(objects,2):
        if a['track_id'] not in hidden and b['track_id'] not in hidden:continue
        ac=np.asarray(a['center_local_m']['value']);asize=np.asarray(a['size_aabb_local_m']['value']);bc=np.asarray(b['center_local_m']['value']);bs=np.asarray(b['size_aabb_local_m']['value'])
        extent=np.minimum(ac+asize/2,bc+bs/2)-np.maximum(ac-asize/2,bc-bs/2)
        if min(extent)>1e-7:
            overlaps.append(dict(a=a['track_id'],b=b['track_id'],volume_m3=float(np.prod(extent)),declared_host_pair=frozenset((a['track_id'],b['track_id'])) in host_pairs))
    labels=label_rows(state)
    result=dict(total=len(objects),visible_claimed=len(objects)-len(hidden),hidden=len(hidden),
        hidden_rear=sum(o['track_id'] in hidden and o['center_local_m']['value'][2]<0 for o in objects),
        hidden_central_rear=sum(o['track_id'] in hidden and abs(math.degrees(math.atan2(o['center_local_m']['value'][0],o['center_local_m']['value'][2])))>135 for o in objects),
        claimed_hidden_intersects_input=sum(checks[i]['intersects_input_box_proxy'] for i in hidden),
        claimed_visible_outside_input=sum(not checks[o['track_id']]['intersects_input_box_proxy'] for o in objects if o['track_id'] not in hidden),
        camera_inside_boxes=sum(v['camera_inside_box_proxy'] for v in checks.values()),
        overlap_pairs_with_hidden=len(overlaps),overlap_declared_host_pairs=sum(p['declared_host_pair'] for p in overlaps),
        overlap_nonhost_pairs=sum(not p['declared_host_pair'] for p in overlaps),unmapped_hidden=sum(not r['visible'] and not r['mapped'] for r in labels),
        max_horizontal_center_distance_m=max((math.hypot(o['center_local_m']['value'][0],o['center_local_m']['value'][2]) for o in objects),default=0.))
    return result,dict(box_checks=checks,overlaps=overlaps,scope='Identical post-hoc AABB diagnostics; no mesh, measured room boundary or calibrated semantic accuracy')


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'


def report(root,out=None,causal=None):
    root=Path(root).resolve();config=verify(root);causal=Path(causal).resolve() if causal else None
    if out is None:
        n=1
        while (root/f'visual_direct_{n:03d}').exists():n+=1
        out=root/f'visual_direct_{n:03d}'
    out=Path(out).resolve()
    if causal and (out==causal or causal in out.parents):raise ValueError('Comparison artifacts must be outside the causal source run')
    if causal:
        ref=read(causal/'suite.json')
        if ref.get('worlds')!=1:raise ValueError('Compare against a single-output causal suite; no best-of-many reference selection')
        refs={r['id']:r for r in ref['scenes']}
        for s in config['scenes']:
            if s['id'] not in refs:continue
            r=refs[s['id']]
            with Image.open(causal/'scenes'/s['id']/'observation/input.png') as im:
                if pixel_sha(im)!=s['pixel_sha256'] or list(im.size)!=s['image_size'] or r['hfov']!=s['hfov']:raise ValueError('RGB/HFOV mismatch: '+s['id'])
    source_files={str(p):sha(p) for base in ([root]+([causal] if causal else [])) for p in base.rglob('*.json') if not p.is_relative_to(out)}
    out=new_output(out);records=[];states=[];diagnostics={};gallery=[];visual_errors=[]
    for scene in config['scenes']:
        sid=scene['id'];image=root/'scenes'/sid/'input.png'
        for method in (['direct_rgb','full_pipeline'] if causal else ['direct_rgb']):
            for backend in ('qwen','api'):
                folder=root/'scenes'/sid/backend if method=='direct_rgb' else causal/'scenes'/sid/'completion/end_to_end'/backend/'completion'
                summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
                state_path=folder/'G_full.json' if method=='direct_rgb' else folder/'world_01/G_full.json'
                row=dict(scene=sid,method=method,backend=backend,status=summary.get('status','not_run'),total=None,visible_claimed=None,hidden=None,hidden_rear=None,hidden_central_rear=None,
                    claimed_hidden_intersects_input=None,claimed_visible_outside_input=None,camera_inside_boxes=None,
                    overlap_pairs_with_hidden=None,overlap_declared_host_pairs=None,overlap_nonhost_pairs=None,unmapped_hidden=None,max_horizontal_center_distance_m=None,
                    model_calls=summary.get('model_calls') if method=='direct_rgb' else None,visibility_source='same direct model' if method=='direct_rgb' else 'shared frozen observer')
                if method=='full_pipeline' and not state_path.exists() and (folder/'world_01/G_full_candidates.json').exists():
                    state_path=folder/'world_01/G_full_candidates.json';row['status']='numeric_candidate_not_completed_world'
                if method=='full_pipeline' and sid not in refs:row['status']='not_in_reference_manifest'
                elif state_path.exists():
                    state=read(state_path);metrics,diagnostic=measure(state,scene['hfov'],scene['image_size']);row.update(metrics)
                    key=sid+'__'+method+'__'+backend;diagnostics[key]=diagnostic
                    states.append(dict(key=key,scene=sid,state=state,image=image,hfov=scene['hfov'],method=method))
                elif method=='full_pipeline' and summary.get('blocked_worlds'):row['status']='numeric_blocked_no_completed_world'
                records.append(row)
    for scene in config['scenes']:
        group=[s for s in states if s['scene']==scene['id']];bounds=shared_bounds([s['state'] for s in group])
        for s in group:
            folder=out/'worlds'/s['key']
            try:
                with Image.open(s['image']) as image:render_world(s['state'],image,s['hfov'],folder,bounds)
                if s['method']=='direct_rgb':
                    p=folder/'index.html';p.write_text('<meta charset="utf-8"><p>RGB直接法：蓝框是本模型自行声称可见的实例，未经过独立观察器核验。位置、尺寸和可见性原样保留；没有WildDet3D、因果程序或布局求解。</p>'+p.read_text())
                gallery.append((s['key'],'worlds/'+s['key']))
            except Exception as exc:visual_errors.append(dict(world=s['key'],error=type(exc).__name__+': '+str(exc),state_preserved=True))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(out/'metrics.json',records);write(out/'box_diagnostics.json',diagnostics);write(out/'visual_errors.json',visual_errors)
    inputs=Image.new('RGB',(600*len(config['scenes']),650),'white');draw=ImageDraw.Draw(inputs)
    for i,s in enumerate(config['scenes']):
        with Image.open(root/'scenes'/s['id']/'input.png') as im:tile=ImageOps.contain(im.convert('RGB'),(590,590))
        inputs.paste(tile,(600*i,45));draw.text((600*i+8,10),'Input '+str(i+1),font=font(18),fill='black')
    inputs.save(out/'inputs.png')
    explanation=('无显式因果模型、无WildDet3D、无预先给定观测库存或候选库。每图每后端一次直接生成，最多一次格式纠正。位置、尺寸与对象不会在后处理中优化、裁剪或删除。'
        '蓝色表示模型声称可见；橙色表示可映射新增；红色表示未映射新增。ERP灰色区域未观测，不是生成RGB。'
        '重叠是包围框代理，不等于网格碰撞；声明宿主关系只用于单独计数，不认证支撑正确。')
    if config.get('synthetic_fixture'):explanation='CPU 合成 fixture，不是真实模型效果。'+explanation
    if causal:explanation+=' 与完整流程的比较同时改变了观察、显式功能机制、几何来源和求解流程，属于整条流程对照，不能单独归因于因果模块；条件H断边消融仍应保留。两条流程调用预算也不同。'
    page='<!doctype html><meta charset="utf-8"><title>RGB直接生成世界基线</title><style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{border:1px solid #ccc;padding:10px}</style><h1>RGB直接生成结构化世界状态</h1><p>'+explanation+'</p><a href="metrics.csv">逐案例CSV</a> | <a href="box_diagnostics.json">详细框级诊断</a><img src="inputs.png">'
    page+=table(['输入','方法','后端','状态','可见/隐藏','后方隐藏','隐藏框入视野','非宿主重叠','格式调用'],[[r['scene'],r['method'],r['backend'],r['status'],str(r['visible_claimed'])+'/'+str(r['hidden']),r['hidden_rear'],r['claimed_hidden_intersects_input'],r['overlap_nonhost_pairs'],r['model_calls']] for r in records])
    page+='<div class="grid">'+''.join('<article><h3>'+escape(k)+'</h3><a href="'+p+'/index.html"><img loading="lazy" src="'+p+'/overview.png">全部视图与交互3D</a></article>' for k,p in gallery)+'</div>'
    if visual_errors:page+='<h2>可视化错误（原状态仍保留）</h2>'+escape(str(visual_errors))
    (out/'index.html').write_text(page)
    changed=[p for p,h in source_files.items() if not Path(p).is_file() or sha(Path(p))!=h]
    if changed:
        p=out/'index.html';p.write_text(p.read_text()+'<p>来源文件在出图期间发生更新，可能仍在生成。本报告不是稳定的最终比较快照；请等实验结束后重做只读报告。</p>')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and v and not v.startswith(('http:','https:','data:','#','javascript:')):self.paths.append(v)
    broken=[];links=0
    for p in out.rglob('*.html'):
        parser=Links();parser.feed(p.read_text());links+=len(parser.paths)
        broken += [str(p)+': '+v for v in parser.paths if not (p.parent/v.split('#')[0]).exists()]
    write(out/'validation.json',dict(source_JSON_unchanged=not changed,comparison_snapshot_stable=not changed,source_files_changed_during_review=changed,
        source_JSON_count=len(source_files),views=len(gallery),records=len(records),local_links=links,broken_links=broken,visual_errors=visual_errors,model_called=False))
    if broken:raise AssertionError('Broken local review links')
    (root/'index.html').write_text('<meta charset="utf-8"><a href="'+escape(os.path.relpath(out/'index.html',root))+'">打开RGB直接法与完整流程比较</a>')
    return out/'index.html'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--causal-suite',type=Path);a=p.parse_args();print(report(a.root,a.out,a.causal_suite))


if __name__=='__main__':main()
