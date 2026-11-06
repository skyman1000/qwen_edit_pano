"""Descriptive comparison; more instances or broader bearings are not quality scores."""
from collections import Counter
from html import escape
from pathlib import Path

from ..common import read,sha


def candidate_page(domain,path):
    """Inspect every proposed delta even if no sampled world reaches geometry."""
    page='<!doctype html><meta charset="utf-8"><title>全局假设与完整增量候选</title><style>body{font:16px/1.6 system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left;overflow-wrap:anywhere}article{margin:25px 0}code{overflow-wrap:anywhere}</style>'
    page+='<h1>全局假设与全部增量候选</h1><p>这是模型提出的语义库存与宿主依赖，不代表几何已通过或已经导出 G_full。空增量只来自模型自己的明确提案。候选选择使用每个 H 内公开声明的有限参考分布，未校准为真实世界后验。</p>'
    for h in domain['hypotheses']:
        page+='<h2>'+escape(h['id'])+'</h2><p>'+escape(h['description'])+'</p><p>假设：'+escape('; '.join(h['assumptions']))+'</p>'
        for fid,p in domain['programs'].items():
            if p['hypothesis']['id']!=h['id']:continue
            state='提案失败，库存未确定' if p.get('proposal_failed') else '完整增量候选'
            page+='<article><h3>'+escape(fid)+' · '+state+'</h3><p>'+escape(p['description'])+'</p>'
            if p.get('proposal_failed'):page+='<pre>'+escape(p.get('proposal_error',''))+'</pre>'
            else:
                objects=[(g,o) for g in p['groups'] for o in g['objects']]
                page+='<p>新增候选实例：'+str(len(objects))+'；未执行按数量或方向配额补物体。</p><table><tr><th>实例</th><th>类别 / 类型</th><th>活动组</th><th>依赖</th><th>实例假设与位置意图</th></tr>'
                for g,o in objects:
                    a=o['attachment'];host=a['host'] if a['host'] else '环境 / 无对象宿主'
                    page+='<tr>'+''.join('<td>'+escape(v)+'</td>' for v in (o['key'],o['category']+' / '+o['variant'],g['function'],a['kind']+' → '+host,o['distinct_instance_reason']+'; '+o['placement_hint']))+'</tr>'
                page+='</table>'
            page+='</article>'
    path.write_text(page)


def inventory(world):
    objects=world['objects'];roots=[o for o in objects if o['attachment']['host'] is None]
    hosts=Counter(o['attachment']['host'] for o in objects if o['attachment']['host'] is not None)
    return dict(proposed=len(objects),roots=len(roots),hosted=len(objects)-len(roots),
        active_zones=len({o['group'] for o in objects}),raw_categories=dict(Counter(o['category'] for o in objects)),
        largest_same_host_collection=max(hosts.values(),default=0),
        largest_same_host_share_of_all_objects=max(hosts.values(),default=0)/len(objects) if objects else None)


def scene_review(scene_root,baseline_root):
    out=scene_root/'joint_completion';result=dict(semantic_quality_verified=False,metrics_are_not_objective_rewards=True)
    if (out/'domain.json').exists():
        d=read(out/'domain.json');programs=[]
        for fid,p in d['programs'].items():
            programs.append(dict(id=fid,hypothesis=p['hypothesis']['id'],proposal_failed=p.get('proposal_failed',False),
                planned_zones=None,zone_decisions=p.get('zone_decisions',[]),
                **inventory(dict(objects=[dict(o,group=g['key']) for g in p['groups'] for o in g['objects']]))))
        result['candidate_programs']=programs
    result['worlds']=[]
    for p in sorted(out.glob('world_*/semantic_world.json')):
        w=read(p);row=dict(folder=p.parent.name,hypothesis=w['hypothesis']['id'],proposal_failed=w['proposal_failed'],**inventory(w))
        if (p.parent/'numeric_review.json').exists():row['review']=read(p.parent/'numeric_review.json')
        result['worlds'].append(row)
    baseline=Path(baseline_root)/'scenes'/scene_root.name
    obs=scene_root/'observation';other=baseline/'observation'
    matching=all((other/name).exists() and sha(obs/name)==sha(other/name) for name in ('input.png','qwen/observer3d/G_obs.json'))
    result['baseline_same_RGB_and_frozen_G_obs']=matching
    if matching:
        comp=baseline/'completion/end_to_end/qwen/completion'
        result['baseline_worlds']=[dict(folder=p.parent.name,**inventory(read(p))) for p in sorted(comp.glob('world_*/semantic_world.json'))]
    result['review_questions']=[
        'Does the whole global hypothesis explain the photographed room, including visible detector omissions?',
        'Which planned functions/zones coexist, which are absent, and which failed proposal generation?',
        'Are root objects and hosted quantities plausible for this assumed space, beyond total object count?',
        'Are co-located objects a useful functional cluster, and do different zones have a coherent relation?',
        'Is the floor assumption compatible with observed object bases and camera tilt uncertainty?',
        'Do alternate worlds express materially different plausible hypotheses, rather than repeated samples?']
    return result
