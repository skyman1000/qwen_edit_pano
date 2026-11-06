"""Placed-object galleries; numeric/request records are folded below the views."""
from html import escape
import json
import os

from ..common import read,write,sha
from .fresh_review import request_record
from .world_generation import SYSTEM_PROMPT

STYLE='''<style>
body{font:16px/1.6 sans-serif;margin:24px;color:#172033;background:#f7f8fa}
.branches{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.branch,.world{background:white;padding:16px;border:1px solid #dce1e8;border-radius:10px;margin:12px 0}
.branch{min-width:0}img{max-width:100%;height:auto}iframe.scene{width:100%;height:600px;border:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:16px 0}
.missing{padding:22px;background:#fff2e4;color:#7b3300}.input{max-height:240px}
@media(max-width:950px){.branches{grid-template-columns:1fr}}
</style>'''


def link(path,destination):return escape(os.path.relpath(path,destination),quote=True)


def world_card(folder,record,destination,visual_override=None):
    """Only display placed boxes from a saved G_full, never raw proposals."""
    counts=(f'自由放置 {record["root_instances"]} / 依附宿主 {record["hosted_instances"]}' if record.get('counts_kind')=='placement_roots_and_hosted'
        else f'家具 {record["furniture_instances"]} / 内容 {record["content_instances"]}')
    visual=visual_override or folder/record.get('visual_folder','world');state_path=folder/record.get('state_file','G_full.json')
    parts=['<article class="world"><h3>'+escape(folder.name)+f' · 新增 {record["new_instances"]} 件（'+counts+'）</h3>']
    if record.get('status')=='blocked_numeric_diagnostic':parts.append('<p class="missing">视野外条件未满足：这是诊断候选，未导出完成世界。</p>')
    parts.append('<p>候选布局；语义与物理安装尚待审核。残留 AABB 交叠：'+str(record.get('remaining_AABB_overlap_pairs','未知'))+' 对。</p>')
    if (folder/'semantic_review.json').is_file():
        audit=read(folder/'semantic_review.json');repeats=audit['repeated_variant_on_same_host']
        if repeats:parts.append('<p>需核实同类实例：'+escape('；'.join(r['variant']+' × '+str(r['count']) for r in repeats))+ '。所有实例仍保留。</p>')
    if (visual/'scene_3d.html').is_file():
        parts.append('<iframe class="scene" loading="lazy" title="'+escape(folder.name)+' 已放置物体三维框" src="'+
            link(visual/'scene_3d.html',destination)+'"></iframe><p>拖动旋转、滚轮缩放；悬停查看物体名称与尺寸。'+
            '<a href="'+link(visual/'scene_3d.html',destination)+'">单独打开三维视图</a></p>')
    for filename,label in [('layout_xz.png','已放置物体：相机 X/Z 俯视布局'),('erp_state_map.png','完整 360° 状态方向图（2:1）')]:
        if (visual/filename).is_file():parts.append('<h4>'+label+'</h4><img loading="lazy" alt="'+label+'" src="'+link(visual/filename,destination)+'">')
    parts.append('<details><summary>状态字段、对象清单与几何诊断</summary><p>状态：'+escape(record['status'])+'</p>')
    full=read(state_path)
    parts.append('<p>'+escape('、'.join(o.get('description','') for o in full['objects']))+'</p>')
    for path,label in [(state_path,'诊断候选字段' if record.get('state_file') else 'G_full'),(visual/'index.html','完整物体视图与坐标'),
                       (folder/'numeric_review.json','几何诊断'),(folder/'host_interventions.json','宿主存在性干预'),
                       (folder/'semantic_review.json','对象语义诊断'),(folder/'pose_model.json','位置结构方程'),(folder/'pose_interventions.json','宿主位移传播')]:
        if path.is_file():parts.append('<a href="'+link(path,destination)+'">'+label+'</a>　')
    parts.append('</details></article>');return ''.join(parts)


def comparison(root,destination):
    from .assembly_experiment import check
    config=check(root,strict_code=False);shared=bool(config.get('frozen_observations'));report={}
    lines=['<section><h1>'+('固定同一 G_obs：API / Qwen 物体布局' if shared else '同一 RGB 独立观测：API / Qwen 物体布局')+'</h1>',
        '<p>蓝色为固定观测；橙色为补全；红色为未映射但保留的补全。框表示物体位置与完整 AABB 尺寸。</p>',
        '<p>这是世界状态可视化，尚未生成 RGB 全景。两侧候选按采样顺序展示，不按对象 ID 配对。</p>']
    lines.append('<p>协议：<code>'+escape(config['protocol'])+'</code></p>')
    if (root/'input.png').is_file():lines.append('<img class="input" alt="输入局部 RGB" src="'+link(root/'input.png',destination)+'">')
    lines.append('<div class="branches">');observer_hashes={};parameter_checks=[]
    for backend,label in [('api','API'),('qwen','Qwen3-VL-8B')]:
        base=root/backend;path=base/'completion/summary.json'
        summary=read(path) if path.is_file() else dict(status='not_completed',worlds=[])
        observed=read(base/'observer3d/G_obs.json') if (base/'observer3d/G_obs.json').is_file() else None
        if observed:observer_hashes[backend]=sha(base/'observer3d/G_obs.json')
        requests=[request_record(p) for p in sorted(base.rglob('*.request.json'))]
        checks=[dict(stage=r['stage'],parameters_match=(r['system']==SYSTEM_PROMPT and r['temperature']==.4 and r['top_p']==.9
            and r['top_k']==50 and r['max_tokens']==8192 and r['penalty_neutral'] and r['generation_profile']=='matched_v1')) for r in requests]
        parameter_checks+=checks
        report[backend]=dict(summary=summary,observed_names=[o.get('description') for o in observed['objects']] if observed else [],
            observer_source=config.get('frozen_observations') or config.get('observation_reuse') or backend,request_checks=checks)
        lines.append('<div class="branch"><h2>'+label+'</h2>')
        if not summary.get('worlds'):
            reason=summary.get('error') or '; '.join(f.get('error','') for f in summary.get('failures',[])) or '补全尚未完成'
            title='没有完成的 G_full；下方保留未通过的数值放置诊断。' if summary.get('blocked_worlds') else '尚无放置结果：没有生成 G_full 三维布局。'
            lines.append('<div class="missing"><strong>'+title+'</strong><p>停止阶段：'+
                escape(summary.get('failed_stage','布局/未完成'))+'</p><p>'+escape(reason)+'</p></div>')
        for w in summary.get('worlds',[])+summary.get('blocked_worlds',[]):
            folder=base/'completion'/w['folder'];full=read(folder/w.get('state_file','G_full.json'));n=len(observed['objects'])
            if full['objects'][:n]!=observed['objects']:raise ValueError('Observed state changed: '+str(folder))
            lines.append(world_card(folder,w,destination))
        lines.append('<details><summary>输入观测与运行记录</summary>')
        for name,title in [('observer2d/boxes2d.png','2D 观测'),('observer3d/projection_boxes.png','WildDet3D 投影')]:
            if (base/name).is_file():lines.append('<h4>'+title+'</h4><img loading="lazy" alt="'+title+'" src="'+link(base/name,destination)+'">')
        if (base/'completion/index.html').is_file():lines.append('<p><a href="'+link(base/'completion/index.html',destination)+'">该分支全部候选与机制</a></p>')
        lines.append('<pre>'+escape(json.dumps(report[backend],ensure_ascii=False,indent=2))+'</pre></details></div>')
    audit=dict(protocol=config['protocol'],experiment_kind=config['experiment_kind'],branches=report,
        source_changes_since_init=config['source_changes_since_init'],shared_observations_by_design=shared,
        observed_state_SHA_equal=len(observer_hashes)==2 and observer_hashes['api']==observer_hashes['qwen'],
        common_public_parameters_passed=bool(parameter_checks) and all(r['parameters_match'] for r in parameter_checks),
        same_structural_mechanism_template=True,exact_numerical_model_shared=False,causal_gain_validated=False,
        sampling_law=config['sampling_law'],worlds_paired_by_object_ID=False)
    if shared and not audit['observed_state_SHA_equal']:raise ValueError('Controlled comparison needs identical G_obs')
    lines.append('</div><details><summary>实验配置与请求核对</summary><a href="'+link(root/'experiment.json',destination)+
        '">实验配置与观测来源</a><pre>'+escape(json.dumps({k:v for k,v in audit.items() if k!='branches'},ensure_ascii=False,indent=2))+'</pre></details></section>')
    return ''.join(lines),audit


def review(root):
    index=1
    while (root/f'review_{index:03d}').exists():index+=1
    out=root/f'review_{index:03d}';out.mkdir();body,audit=comparison(root,out)
    write(out/'review.json',audit)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>物体布局对比</title>'+STYLE+body)
    return out/'index.html'


def suite(root,other):
    # New versioned page: rebuilding a review never overwrites an old report.
    first=review(root);second=review(other);target=root/'two_experiments.html';i=1
    while target.exists():target=root/f'two_experiments_{i:03d}.html';i+=1
    sections=[]
    for directory,page in ((root,first),(other,second)):
        body,_=comparison(directory,root)
        sections.append('<p><a href="'+link(page,root)+'">单独查看本组对比</a></p>'+body)
    target.write_text('<!doctype html><meta charset="utf-8"><title>两组物体布局对比</title>'+STYLE+''.join(sections))
    return target
