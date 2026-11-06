"""Planning is visible before any metric object layout succeeds."""
from html import escape
import json
import math

from ..common import read, write


def blueprint_svg(plan, observed, hfov):
    b = plan['room_envelope']['bounds_xz_m']
    xs = [b[0], b[1], 0.]; zs = [b[2], b[3], 0.]
    for z in plan['zones']:
        q = z['bounds_xz_m']; xs += q[:2]; zs += q[2:]
    for o in observed['objects']:
        x, _, z = o['center_local_m']['value']; sx, _, sz = o['size_aabb_local_m']['value']
        xs += [x-sx/2, x+sx/2]; zs += [z-sz/2, z+sz/2]
    x0, x1, z0, z1 = min(xs)-.5, max(xs)+.5, min(zs)-.5, max(zs)+.5
    scale = min(800/(x1-x0), 620/(z1-z0))
    def point(x,z): return (50+(x-x0)*scale, 70+(z1-z)*scale)
    def rectangle(q, color, dashed=True):
        x,y=point(q[0],q[3]); w=(q[1]-q[0])*scale; h=(q[3]-q[2])*scale
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" fill-opacity=".10" stroke="{color}"'+(' stroke-dasharray="7 4"' if dashed else '')+'/>'
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 760"><rect width="920" height="760" fill="white"/>',
         '<text x="30" y="28" font-size="18">Whole-room plan: dashed = hypothesis; blue = frozen observation</text>',
         rectangle(b,'#424b57')]
    centroids={}
    colors=['#a45115','#288b66','#7951ae','#ab3a68','#207d9b']
    for i,z in enumerate(plan['zones']):
        q=z['bounds_xz_m']; x,y=point((q[0]+q[1])/2,(q[2]+q[3])/2);centroids[z['id']]=(x,y)
        out += [rectangle(q,colors[i%len(colors)]),f'<text x="{x}" y="{y}" text-anchor="middle" font-size="15">{escape(z["id"]+": "+z["function"])}</text>']
    for r in plan['circulation']:
        a,c=centroids[r['from']],centroids[r['to']]
        out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{c[0]}" y2="{c[1]}" stroke="#666" stroke-dasharray="3 5"/>')
    for o in observed['objects']:
        x,_,z=o['center_local_m']['value'];sx,_,sz=o['size_aabb_local_m']['value'];px,py=point(x,z)
        out += [rectangle([x-sx/2,x+sx/2,z-sz/2,z+sz/2],'#287db5',False),
                f'<text x="{px}" y="{py}" font-size="11" fill="#185780">{escape(o["track_id"])}</text>']
    cx,cy=point(0,0)
    for sign in (-1,1):
        px,py=point(sign*math.sin(math.radians(hfov/2)),math.cos(math.radians(hfov/2)))
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{px}" y2="{py}" stroke="#111"/>')
    out += [f'<circle cx="{cx}" cy="{cy}" r="5"/><text x="{cx+8}" y="{cy+15}">camera; +Z forward</text>', '</svg>']
    return ''.join(out)


def page(folder, hypothesis, plan, observed, hfov, objects):
    folder.mkdir(parents=True,exist_ok=True)
    if plan:
        (folder/'blueprint.svg').write_text(blueprint_svg(plan,observed,hfov))
    body='<h1>完整世界规划与实例展开</h1><p>这是场景状态假设，不是生成的全景 RGB。规划、语义审查与几何诊断分别显示。</p>'
    body+='<pre>'+escape(json.dumps(hypothesis,ensure_ascii=False,indent=2))+'</pre>'
    if plan:
        body+='<img src="blueprint.svg"><p>虚线为假设区域；蓝框为冻结观测。区域框是规划范围，不是精确墙体。</p>'
        for z in plan['zones']:
            items=[o for o in objects if o['zone_id']==z['id']]
            body+='<h2>'+escape(z['id']+' / '+z['function'])+'</h2><p>'+escape(z['contents_intent'])+'</p>'
            body+='<p>已有实例：'+escape(', '.join(z['observed_ids']) or '无')+'；新增实例：'+str(len(items))+'</p><ul>'
            for o in items:
                body+='<li>'+escape(o['id']+' '+o['category']+' — '+o['placement']+'; support='+str(o['support']))+'</li>'
            body+='</ul>'
        body+='<h2>四周空间解释与观测缺口</h2><pre>'+escape(json.dumps(dict(sectors=plan['sector_review'],visible_unmodeled=plan['visible_unmodeled']),ensure_ascii=False,indent=2))+'</pre>'
    for name in ('status.json','semantic_review.json','numeric_review.json'):
        if (folder/name).exists():body+='<h2>'+name+'</h2><pre>'+escape(json.dumps(read(folder/name),ensure_ascii=False,indent=2))+'</pre>'
    for name in ('blueprint.json','inventory.json','causal_trace.json','installation_interventions.json','G_full.json','G_full_candidate.json'):
        if (folder/name).exists():body+='<p><a href="'+name+'">'+name+'</a></p>'
    if (folder/'views/index.html').exists():body+='<a href="views/index.html"><img src="views/overview.png">原视角、平面图、六方向、ERP 与交互 3D</a>'
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:25px auto}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'+body)


def report(root, suite):
    records=[];cards=[]
    for scene in suite['scenes']:
        base=root/'scenes'/scene['id']
        if (base/'scene_status.json').exists():
            state=read(base/'scene_status.json')
            if state.get('status')=='failed':records.append(dict(scene=scene['id'],**state))
        for folder in sorted(base.glob('world_*')):
            state=read(folder/'status.json') if (folder/'status.json').exists() else dict(status='planning_in_progress')
            records.append(dict(scene=scene['id'],world=folder.name,**state))
            url=str(folder.relative_to(root))
            cards.append('<article><h2>'+escape(scene['id']+' / '+folder.name)+'</h2><pre>'+escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre><a href="'+url+'/index.html">查看规划、对象和诊断</a>'+('<img src="'+url+'/blueprint.svg">' if (folder/'blueprint.svg').exists() else '')+'</article>')
    failed=sum(r['status']=='failed' for r in records)
    result=dict(status='execution_partial' if failed else 'completed_needs_review' if len(records)==len(suite['scenes'])*suite['hypotheses'] and all(r['status'] in ('plan_complete','layout_complete') for r in records) else 'in_progress',
                execution_failures=failed,API_used=False,observer_rerun=False,model='panorama_plan_first_v1',records=records)
    write(root/'results.json',result)
    body='<h1>局部 RGB → 全景场景规划 → 完整对象图</h1><p>规划先行；不设最少数量或方位配额。每一份假设独立展开，不混合互斥世界。状态完成不代表语义正确或真实因果已识别。</p><p><a href="results.json">运行状态</a></p>'
    body+='<pre>'+escape(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:24px}article{border:1px solid #ccc;padding:16px;margin:16px 0}img{max-width:900px;width:100%}pre{white-space:pre-wrap}</style>'+body)
    return result
