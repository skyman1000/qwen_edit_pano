"""Read-only experimental inputs; compare independent observers and worlds."""
from html import escape
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw

from ..common import read,write
from .fresh_observer import PROMPT
from .world_generation import SYSTEM_PROMPT


def request_record(path):
    r=read(path)
    if 'payload' in r:
        p=r['payload'];message=p['messages'][1]['content']
        return dict(stage=path.stem,prompt=next(v['text'] for v in message if v['type']=='text'),
            system=p['messages'][0]['content'],temperature=p['temperature'],top_p=p.get('top_p'),top_k=p.get('top_k'),
            max_tokens=p['max_tokens'],penalty_neutral=p.get('frequency_penalty')==0.,
            generation_profile=r.get('generation_profile'),image_sha256=r.get('canonical_image_sha256'))
    return dict(stage=path.stem,prompt=r['prompt'],system=r['system_prompt'],temperature=r['generation']['temperature'],
        top_p=r['generation']['top_p'],top_k=r['generation']['top_k'],max_tokens=r['max_tokens'],
        penalty_neutral=r['generation']['repetition_penalty']==1.,generation_profile=r['generation_profile'],
        image_sha256=r.get('canonical_image_sha256'))


def review(root):
    from .fresh_pipeline import check
    config=check(root,strict_code=False);i=1
    while (root/f'review_{i:03d}').exists():i+=1
    out=root/f'review_{i:03d}';out.mkdir();reports={};body=['<!doctype html><meta charset="utf-8"><title>Fresh RGB API/Qwen comparison</title>',
        '<style>body{font:16px/1.6 sans-serif;margin:24px}table{width:100%;table-layout:fixed}td{vertical-align:top;padding:10px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warn{background:#fff0db;padding:15px}</style>',
        '<h1>同一局部RGB → 独立2D → 同一WildDet3D → 同构SCM → 视野外世界</h1>',
        '<p class="warn">这是端到端系统对照，两侧G_obs允许不同，因此不能单独归因于补全模型。相同的是代码、提示词模板与公开参数；'
        '后续提示词中的观测、候选域和数值CPT由两端分别生成。候选按各自排名展示，ID不跨模型配对，模型内权重不作跨模型质量分数。</p>',
        '<p>蓝色=固定观测，橙色/红色=新增假设。2:1为世界状态方向图，本阶段没有生成RGB全景。</p>',
        '<p><a href="../experiment.json">固定实验配置与代码哈希</a> | <a href="review.json">对照检查结果</a></p><table><tr><th>API</th><th>Qwen3-VL-8B</th></tr><tr>']
    request_sets={}
    for backend in ('api','qwen'):
        base=root/backend;summary_path=base/'completion'/'summary.json';summary=read(summary_path) if summary_path.is_file() else {'status':'not_completed'}
        states={name:read(base/name/file) for name,file in [('observer2d','G_2d.json'),('observer3d','G_obs.json')]
            if (base/name/file).is_file()}
        requests=[request_record(p) for p in sorted(base.rglob('*.request.json'))];request_sets[backend]=requests
        checks=[dict(stage=r['stage'],shared_parameters_match=(r['system']==SYSTEM_PROMPT and r['temperature']==.4
            and r['top_p']==.9 and r['top_k']==50 and r['max_tokens']==config['max_tokens'] and r['penalty_neutral']
            and r['generation_profile']=='matched_v1')) for r in requests]
        reports[backend]=dict(summary=summary,observed_2d_count=len(states.get('observer2d',{}).get('objects',[])),
            observed_3d_count=len(states.get('observer3d',{}).get('objects',[])),request_checks=checks,
            observer3d_summary=read(base/'observer3d/summary.json') if (base/'observer3d/summary.json').is_file() else None,
            raw_labels=[o.get('raw_category') for o in states.get('observer2d',{}).get('objects',[])])
        body.append('<td><h2>'+backend+'</h2><pre>'+escape(json.dumps(reports[backend],ensure_ascii=False,indent=2))+'</pre>')
        for file,label in [('observer2d/boxes2d.png','2D识别'),('observer3d/projection_boxes.png','WildDet3D投影：红色2D/绿色3D AABB')]:
            if (base/file).is_file():body.append('<h3>'+label+'</h3><img src="../'+backend+'/'+file+'">')
        if (base/'completion/index.html').is_file():body.append('<p><a href="../'+backend+'/completion/index.html">查看所有世界、机制和关系</a></p>')
        for world in summary.get('worlds',[]):
            relative=backend+'/completion/'+world['folder'];worldpath=root/relative/'G_full.json';full=read(worldpath)
            observed=states['observer3d'];n=len(observed['objects'])
            if full['objects'][:n]!=observed['objects']:raise ValueError('Observed state changed in '+str(worldpath))
            names=[o.get('description',o.get('object_type','')) for o in full['objects'][n:]]
            body.append('<h3>'+escape(world['folder'])+f'：新增{len(names)}</h3><p>'+escape('、'.join(names))+'</p>'+
                '<a href="../'+relative+'/world/index.html">交互3D与完整检查</a> | <a href="../'+relative+'/G_full.json">世界字段</a><br>')
            for imagefile in ('erp_state_map.png','layout_xz.png'):
                if (root/relative/'world'/imagefile).is_file():
                    body.append('<img src="../'+relative+'/world/'+imagefile+'">')
        body.append('</td>')
    observers={b:next((r for r in requests if r['stage']=='answer.request'),None) for b,requests in request_sets.items()}
    pair=[observers[b] for b in ('api','qwen')]
    wild_configs=[read(root/b/'observer3d/config.json') if (root/b/'observer3d/config.json').is_file() else None for b in ('api','qwen')]
    audit=dict(protocol=config['protocol'],branches=reports,
        source_changes_since_init=config['source_changes_since_init'],
        same_2d_prompt=bool(all(pair) and all(r['prompt']==PROMPT for r in pair)),
        same_2d_canonical_image=bool(all(pair) and pair[0]['image_sha256']==pair[1]['image_sha256']),
        same_public_generation_parameters=bool(all(request_sets.values()) and all(c['shared_parameters_match'] for r in reports.values() for c in r['request_checks'])),
        same_SCM_factorization=True,same_numerical_CPT=False,observations_independently_predicted=True,
        same_WildDet3D_checkpoint_and_intrinsics=(all(wild_configs) and all(wild_configs[0][k]==wild_configs[1][k]
            for k in ('checkpoint_sha256','upstream_inference_sha256','intrinsics','image_sha256'))) if all(wild_configs) else None,
        observer_instance_ids_cross_backend_comparable=False,causal_gain_validated=False,
        interpretation='End-to-end systems comparison, not isolated completion-backend causal attribution')
    write(out/'review.json',audit);body.append('</tr></table><pre>'+escape(json.dumps({k:v for k,v in audit.items() if k!='branches'},ensure_ascii=False,indent=2))+'</pre>')
    (out/'index.html').write_text(''.join(body));return out/'index.html'
