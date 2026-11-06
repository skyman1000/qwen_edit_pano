"""Read-only four-column comparison of saved worlds; never calls a model.

Use an explicit primary causal suite and a later per-scene replacement suite.
Failed and empty direct responses remain in the planned experiment denominator.
"""
import argparse
from collections import Counter
import csv
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil

from PIL import Image

from ..common import new_output,read,write,sha
from .assembly_pilot_visual import shared_bounds
from .assembly_presentation_visual import render_world
from .direct_rgb_review import measure
from .direct_rgb_world import pixel_sha,verify

KITCHEN='8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029'
NAMES={'8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39':'休息区',
    'X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c':'客厅',
    'x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6':'卫生间',
    'zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a':'卧室',KITCHEN:'厨房'}
METHODS=(('causal','qwen','功能因果方法 · Qwen'),('causal','api','功能因果方法 · API'),
    ('direct','qwen','直接 RGB · Qwen'),('direct','api','直接 RGB · API'))


def href(path,out):return os.path.relpath(path,out)


def compare(primary,replacement,direct,out):
    primary=Path(primary).resolve();replacement=Path(replacement).resolve();direct=Path(direct).resolve();out=Path(out).resolve()
    roots=list(dict.fromkeys([primary,replacement,direct]))
    if any(out==r or r in out.parents for r in roots):raise ValueError('Keep the comparison outside all source runs')
    config=verify(direct);sources={};overrides={}
    for root in (primary,replacement):
        suite=read(root/'suite.json')
        if suite['worlds']!=1:raise ValueError('Use single-output suites; never select a best sample')
        for s in suite['scenes']:
            if s['id'] in sources:overrides[s['id']]=dict(previous=str(sources[s['id']]),replacement=str(root))
            sources[s['id']]=root
    # Save the JSON/input hashes before producing any new artifact.
    protected={str(p):sha(p) for root in roots for p in root.rglob('*.json')}
    for root in roots:
        for p in root.glob('scenes/*/input.png'):protected[str(p)]=sha(p)
        for p in root.glob('scenes/*/observation/input.png'):protected[str(p)]=sha(p)
    out=new_output(out);records=[];scenes=[];details={};views=[]
    for s in config['scenes']:
        sid=s['id'];source=sources.get(sid)
        if source is None:raise ValueError('Missing causal input: '+sid)
        image=direct/'scenes'/sid/'input.png';observed_image=source/'scenes'/sid/'observation/input.png'
        suite=read(source/'suite.json');ref=next(r for r in suite['scenes'] if r['id']==sid)
        with Image.open(observed_image) as im:
            if pixel_sha(im)!=s['pixel_sha256'] or list(im.size)!=s['image_size'] or ref['hfov']!=s['hfov']:raise ValueError('Input RGB/HFOV mismatch: '+sid)
        dest=out/'inputs';dest.mkdir(exist_ok=True);shutil.copyfile(image,dest/(sid+'.png'))
        obs=source/'scenes'/sid/'observation/qwen/observer2d'
        if (obs/'boxes2d.png').exists():shutil.copyfile(obs/'boxes2d.png',dest/(sid+'_boxes.png'))
        observation_note=('可见清单仍只有3个吧凳：漏掉中岛、橱柜、灶具/抽油烟机等；on_floor=false及“吧凳在中岛表面”的解释也不可靠。两路世界只是数值通过，不能称完整观测已恢复。'
            if sid==KITCHEN else '可见物体由共享Qwen识别并由WildDet3D估计三维；尚未做独立语义真值验收。')
        states=[];rows=[]
        for method,backend,title in METHODS:
            folder=source/'scenes'/sid/'completion/end_to_end'/backend/'completion' if method=='causal' else direct/'scenes'/sid/backend
            summary=read(folder/'summary.json') if (folder/'summary.json').exists() else {}
            state_path=folder/'world_01/G_full.json' if method=='causal' else folder/'G_full.json'
            row=dict(scene=sid,scene_name=NAMES.get(sid,sid),method=method,backend=backend,title=title,
                status=summary.get('status','not_run'),outcome='failed_or_missing',error=summary.get('error'),
                source_run=str(source if method=='causal' else direct),summary_link=href(folder/'summary.json',out) if (folder/'summary.json').exists() else None,
                state_link=None,view_folder=None,visible_claimed=None,hidden=None,total=None,hidden_rear=None,
                hidden_central_rear=None,claimed_hidden_intersects_input=None,claimed_visible_outside_input=None,
                overlap_nonhost_pairs=None,unmapped_hidden=None,max_horizontal_center_distance_m=None,
                numeric_contract_passed=None,observation_issue='incomplete_visible_inventory' if sid==KITCHEN and method=='causal' else None,
                semantic_quality_verified=False,model_calls=summary.get('model_calls') if method=='direct' else None)
            if state_path.exists():
                state=read(state_path);metrics,diagnostic=measure(state,s['hfov'],s['image_size']);row.update(metrics)
                row['outcome']='nonempty_world' if state['objects'] else 'empty_world'
                row['state_link']=href(state_path,out)
                if method=='causal':
                    review=read(folder/'world_01/numeric_review.json');row['numeric_contract_passed']=review['geometry_contract_passed']
                key=sid+'__'+method+'__'+backend;states.append((key,state,row));details[key]=diagnostic
            rows.append(row);records.append(row)
        bounds=shared_bounds([state for _,state,_ in states])
        for key,state,row in states:
            dest=out/'worlds'/key
            with Image.open(image) as im:render_world(state,im,s['hfov'],dest,bounds)
            row['view_folder']='worlds/'+key;views.append(key)
            (dest/'index.html').write_text('<meta charset="utf-8"><p>'+escape(row['title'])+'：'+escape(observation_note if row['method']=='causal' else '直接法的可见标记、位置、尺寸均由该模型自行预测，未做布局优化或独立观测校验。空世界仍是失败结果。')+'</p>'+(dest/'index.html').read_text())
        scenes.append(dict(id=sid,name=NAMES.get(sid,sid),image='inputs/'+sid+'.png',
            boxes='inputs/'+sid+'_boxes.png' if (out/'inputs'/(sid+'_boxes.png')).exists() else None,
            observation_note=observation_note,source_run=str(source),rows=rows))
        print(json.dumps(dict(scene=sid,rendered=len(states)),ensure_ascii=False),flush=True)
    write(out/'metrics.json',records);write(out/'box_diagnostics.json',details)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in records for k in r}));w.writeheader();w.writerows(records)
    counts={method:dict(Counter(r['outcome'] for r in records if r['method']==method)) for method in ('causal','direct')}
    write(out/'provenance.json',dict(primary_causal=str(primary),replacement_causal=str(replacement),direct=str(direct),
        scene_sources={s['id']:s['source_run'] for s in scenes},explicit_replacements=overrides,
        source_input_matched=True,best_of_many_selection=False,counts=counts,models_called=False))
    from .meeting_model_notes import build
    build(out,replacement,records)
    (out/'index.html').write_text(page(scenes,counts))
    # The page links to the audit itself; create it before checking local links.
    write(out/'validation.json',dict(status='checking'))
    changed=[p for p,h in protected.items() if not Path(p).exists() or sha(Path(p))!=h]
    broken=[];links=0
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            nonlocal links
            for k,v in attrs:
                if k in ('href','src') and v and not v.startswith(('http:','https:','data:','#','javascript:')):
                    links+=1
                    if not (current.parent/v.split('#')[0]).exists():broken.append(str(current)+': '+v)
    for current in out.rglob('*.html'):Links().feed(current.read_text())
    validation=dict(source_files_unchanged=not changed,source_files_checked=len(protected),changed_sources=changed,
        planned_cells=len(records),scenes=len(scenes),rendered_states=len(views),outcome_counts=counts,
        local_links=links,broken_links=broken,real_model_calls=0,synthetic_worlds_created=False,
        scene_input_pixels_and_HFOV_matched=True,kitchen_observation_quality_issue_retained=True)
    write(out/'source_hashes.json',protected);write(out/'validation.json',validation)
    if changed or broken:raise AssertionError('Source changed or comparison links broken; inspect validation.json')
    return out/'index.html'


def page(scenes,counts):
    data=json.dumps(scenes,ensure_ascii=False).replace('<','\\u003c')
    counttext=escape(json.dumps(counts,ensure_ascii=False))
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>组会：局部RGB到完整世界状态 · 四路比较</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#142037;font:16px/1.6 system-ui,sans-serif}header,main{padding:24px;max-width:1900px;margin:auto}h1{font-size:28px;margin:0 0 12px}h2{font-size:20px}a{color:#1d57a3}.note{background:#fff4cf;padding:15px;border-left:5px solid #d88913;margin:12px 0}.toolbar{position:sticky;top:0;background:#fff;padding:14px;z-index:2;border:1px solid #dce2ec;display:flex;flex-wrap:wrap;gap:12px;align-items:center}select,button{font:inherit;padding:6px 10px}button{cursor:pointer}section{margin:24px 0}.input{display:flex;align-items:flex-start;gap:18px}.input img{width:min(42vw,460px)}.input p{max-width:800px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:#fff;border:1px solid #dce2ec;padding:14px;min-width:0}.card h3{font-size:18px;margin:0 0 12px}.card img{width:100%;height:320px;object-fit:contain;background:#fafafa}.card iframe{width:100%;height:420px;border:0}.bad{color:#a22a25}.small{font-size:13px;overflow-wrap:anywhere}.pill{padding:3px 7px;background:#e7eef9;border-radius:5px;display:inline-block}.error{white-space:pre-wrap;overflow-wrap:anywhere;min-height:120px}.legend{display:flex;gap:18px;flex-wrap:wrap}.blue{color:#287db5}.orange{color:#de8a27}.red{color:#d44f57}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #dce2ec;padding:7px;text-align:left}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid{grid-template-columns:1fr}.input{display:block}.input img{width:100%}}</style>
<header><h1>局部 RGB → 周围360°世界状态：四路实际结果</h1><p>同一场景并排比较：功能因果方法 Qwen / API，直接 RGB Qwen / API。API：硅基流动 DeepSeek-V4.1-Flash；本地：Qwen3-VL-8B-Instruct。</p>
<div class="note">这是完整流程对照，同时改变观察器、显式概率机制、几何来源、求解和调用预算，不能把所有差异单独归因于因果模块。厨房采用最新重测，其余四图采用先前保存的成功状态；不是重新执行的一次五图实验。厨房观测漏检、直接法截断失败和空世界均明确保留。</div>
<p><a href="metrics.csv">全部20格指标 CSV</a> · <a href="metrics.json">指标 JSON</a> · <a href="provenance.json">来源与替换记录</a> · <a href="validation.json">只读核验</a> · <a href="model_explained.html">模型讲解与流程图</a></p>
<div class="legend"><span class="blue">蓝：观测/直接法自行声称可见</span><span class="orange">橙：可映射到数据集类别的新增对象</span><span class="red">红：未映射新增对象</span></div>
<p class="small">ERP是2:1状态投影，灰色区域未知，未生成隐藏RGB。每场景的平面图与3D图使用四路状态并集的统一坐标范围，不缩放对象，不删物体。非宿主重叠为统一AABB代理统计；现有数值通过不认证真实网格、通行或语义。</p>
<details><summary>完整结果计数（非空≠合理）</summary><pre>'''+counttext+'''</pre></details></header><main>
<div class="toolbar"><label>场景 <select id="scene"><option value="all">全部五图</option></select></label><label>比较视图 <select id="view"><option value="layout_xz.png">状态平面图（同尺度）</option><option value="overview.png">全部视图总览</option><option value="local_projection.png">原局部图投影</option><option value="direction_projections.png">六方向投影</option><option value="scene_3d.png">3D静态图</option><option value="scene_3d.html">交互3D（可旋转）</option><option value="erp_overlay.png">2:1 ERP叠加</option><option value="erp_state_map.png">2:1 ERP纯状态</option></select></label><button onclick="document.documentElement.requestFullscreen?.()">全屏展示</button></div><div id="scenes"></div></main>
<script>const scenes='''+data+''';const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const choice=document.querySelector('#scene'),view=document.querySelector('#view');
scenes.forEach(s=>{const o=document.createElement('option');o.value=s.id;o.textContent=s.name;choice.append(o)});
function card(r){let html='<article class="card"><h3>'+esc(r.title)+'</h3><span class="pill">'+esc(r.outcome==='nonempty_world'?'非空状态已导出':r.outcome==='empty_world'?'空世界：未有效完成':'失败：无完整状态')+'</span>';
html+='<p>可见 / 隐藏：<b>'+esc(r.visible_claimed)+' / '+esc(r.hidden)+'</b><br>后半空间隐藏：'+esc(r.hidden_rear)+'<br>隐藏框与原视野相交：'+esc(r.claimed_hidden_intersects_input)+'<br>非宿主AABB重叠：'+esc(r.overlap_nonhost_pairs)+'</p>';
if(r.observation_issue)html+='<p class="bad">共享观测仍漏掉主要可见家具，不能作为已验收的完整世界。</p>';
if(r.view_folder){const src=r.view_folder+'/'+view.value;html+=view.value.endsWith('.html')?'<iframe loading="lazy" src="'+src+'"></iframe>':'<a href="'+src+'" target="_blank"><img loading="lazy" src="'+src+'"></a>';html+='<p><a href="'+r.view_folder+'/index.html">全部视图、标签表与交互3D</a></p>';}else html+='<div class="error bad">'+esc(r.error||r.status)+'</div>';
html+='<p class="small">原始状态：'+esc(r.status)+'<br>'+(r.method==='causal'?'原数值门槛：'+esc(r.numeric_contract_passed):'无WildDet3D、显式因果模型或布局优化')+'</p>';
if(r.summary_link)html+='<a href="'+esc(r.summary_link)+'">原摘要</a> ';if(r.state_link)html+='<a href="'+esc(r.state_link)+'">原G_full</a>';return html+'</article>';}
function draw(){document.querySelector('#scenes').innerHTML=scenes.filter(s=>choice.value==='all'||choice.value===s.id).map(s=>'<section><h2>'+esc(s.name)+' <span class="small">'+esc(s.id)+'</span></h2><div class="input"><a href="'+s.image+'"><img src="'+s.image+'"></a><div><p>'+esc(s.observation_note)+'</p>'+(s.boxes?'<a href="'+s.boxes+'">查看实际2D识别框</a>':'')+'<p class="small">因果来源：'+esc(s.source_run)+'</p></div></div><div class="grid">'+s.rows.map(card).join('')+'</div></section>').join('');}choice.onchange=draw;view.onchange=draw;draw();</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-causal',type=Path,required=True);p.add_argument('--replacement-causal',type=Path,required=True)
    p.add_argument('--direct',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(compare(a.primary_causal,a.replacement_causal,a.direct,a.out))


if __name__=='__main__':main()
