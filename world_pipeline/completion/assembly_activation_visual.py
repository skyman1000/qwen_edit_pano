"""CPU-only actual-result comparison with strict taxonomy coloring and failures."""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

import numpy as np

from ..common import read, write, sha, new_output
from .assembly_taxonomy import annotate, vocabulary, VOCABULARY
from .assembly_pilot_visual import shared_bounds, footprint, _synchronized
from .batch_world_review import interactive_boxes
from .semantic_layout_visual import erp_map
from .assembly_review import STYLE, link


def _plots(branches, out, current_version='v6'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    available = [b for b in branches if b['law']]
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for backend, ax in zip(('qwen','api'),axes):
        for branch in available:
            if branch['backend']!=backend: continue
            row = sorted((int(n),p) for n,p in branch['law']['hidden_count_probabilities'].items())
            ax.plot([n for n,p in row],[p for n,p in row],marker='.',label=branch['label'])
        ax.set_title(backend+' / exact model count law'); ax.set_xlabel('new object records'); ax.set_ylabel('probability')
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'count_probability_comparison.png',dpi=140); plt.close(fig)
    if current_version not in ('v6','v7','v8','v9','v10'):return
    current = [b for b in available if b['label']==current_version+'/end_to_end']
    if not current: return
    height=max(5,max(len(read(b['completion']/'causal_model.json')['profiles']) for b in current)*.28+2.5)
    fig, axes = plt.subplots(1,len(current),figsize=(8*len(current),height),squeeze=False)
    for b,ax in zip(current,axes[0]):
        model = read(b['completion']/'causal_model.json')
        if model.get('model_variant')=='context_joint_assembly_scm_v10':
            from .assembly_set import marginals
        elif model.get('model_variant')=='functional_spatial_program_scm_v9':
            from .assembly_panorama import marginals
        elif model.get('model_variant')=='reference_shrunk_functional_scene_scm_v8':
            from .assembly_mechanism import marginals
        elif model.get('model_variant')=='anchored_realization_scene_scm_v7':
            from .assembly_realization import marginals
        else:
            from .assembly_activation import marginals
        hs = model['profiles']; roles = list(model['slots'])
        matrix = np.array([[marginals(model,h['id'])['activation_probabilities'][r] for r in roles] for h in hs])
        ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        labels=[r.replace('__observed_extension',' / extension').replace('__new_group',' / new') for r in roles]
        ax.set_xticks(range(len(roles)),labels,rotation=40,ha='right',fontsize=8)
        ax.set_yticks(range(len(hs)),[h['id']+':'+(','.join(h['functions']) or 'unspecified') for h in hs],fontsize=7)
        for i in range(len(hs)):
            for j in range(len(roles)): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.55 else 'black')
        ax.set_title(b['backend']+' / P(additional role | do(H))')
    fig.tight_layout(); fig.savefig(out/'activation_interventions.png',dpi=140); plt.close(fig)


def report(root, baseline_root=None, out=None):
    root = Path(root).resolve(); external_output = out is not None
    current_version = 'v'+str(read(root/'run.json').get('version',5)) if (root/'run.json').is_file() else 'saved'
    if out is None:
        i = 1
        while (root/f'visual_{current_version}_{i:03d}').exists(): i += 1
        out = root/f'visual_{current_version}_{i:03d}'; out.mkdir()
    else: out = new_output(Path(out))
    branches = []; observed_sha = set(); rgb_sha = set(); sources = {}; states = []
    experiments=read(root/'run.json').get('experiments',['end_to_end','layout_control','function_free']) if (root/'run.json').is_file() else ['end_to_end','layout_control','function_free']
    roots = [(current_version,root,experiments)] if (root/'function_free').is_dir() else [('saved',root,['end_to_end','layout_control'])]
    if baseline_root is not None:
        base=Path(baseline_root).resolve();base_version='v'+str(read(base/'run.json').get('version',5))
        if base_version==current_version:base_version+=' baseline'
        roots.append((base_version,base,['end_to_end']))
    for version, base, experiments in roots:
        for experiment in experiments:
            for backend in ('qwen','api'):
                branch = base/experiment; config_path = branch/'experiment.json'
                if not config_path.is_file(): continue
                config = read(config_path); completion = branch/backend/'completion'; summary_path = completion/'summary.json'
                summary = read(summary_path) if summary_path.is_file() else dict(status='not_run',worlds=[],failures=[])
                obs_path = branch/backend/'observer3d/G_obs.json'; observed_sha.add(sha(obs_path)); rgb_sha.add(sha(branch/'input.png'))
                sources[str(config_path)] = sha(config_path)
                label = version+'/'+experiment; identifier = version+'_'+experiment+'_'+backend
                law_path = completion/'probability_audit.json'; law = read(law_path) if law_path.is_file() else None
                b = dict(id=identifier,label=label,backend=backend,completion=completion,config=config,summary=summary,worlds=[],law=law)
                for record in summary.get('worlds',[])+summary.get('blocked_worlds',[]):
                    folder = completion/record['folder']; state_path = folder/record.get('state_file','G_full.json')
                    if not state_path.is_file(): raise ValueError('Recorded world is missing: '+str(state_path))
                    semantic_path = folder/'semantic_world.json'; semantic = read(semantic_path)
                    full = read(state_path); observed = read(obs_path)
                    if full['objects'][:len(observed['objects'])]!=observed['objects']: raise ValueError('Observed records changed: '+str(state_path))
                    display = annotate(full,semantic); states.append(display); sources[str(state_path)] = sha(state_path); sources[str(semantic_path)] = sha(semantic_path)
                    b['worlds'].append(dict(record=record,state=display,folder=folder,source=state_path))
                branches.append(b)
    if not branches or len(observed_sha)!=1 or len(rgb_sha)!=1: raise ValueError('Comparison requires identical RGB and G_obs across all branches')
    bounds = shared_bounds(states); figures = {}; audit = []
    parts = ['<!doctype html><meta charset="utf-8"><title>因果机制与完整世界对比</title>',STYLE,
        '<h1>'+current_version+' 实际世界状态、概率与类别审核</h1>',
        '<p>蓝色：固定观测；橙色：原始类别名在数据集类别表中；紫色：原始名称不在表中、但有受支持映射；红色：类别未解析。全部物体保留。</p>',
        '<p>类别表来自项目绑定的 Matterport 对象词表（37 类，已排除部分结构类），不是 NYU40 或自行新建类别。原始名和映射结果分别列出；objects/misc 等粗类别不代表细粒度识别正确。</p>',
        '<p>框与 ERP 是世界状态可视化，尚未生成 RGB 全景。数值失败与约束阻断仍计入分母；图中候选不自动视为有效完整世界。</p>',
        '<p><a href="vocabulary.json">冻结类别表</a> · <a href="review.json">完整审核与来源</a></p>',
        '<img class="input" src="'+link(root/'end_to_end/input.png',out)+'" alt="原始局部观测">',
        '<img src="count_probability_comparison.png" alt="精确数量概率对比">']
    if current_version in ('v6','v7','v8','v9','v10') and any(b['label']==current_version+'/end_to_end' and b['law'] for b in branches): parts.append('<img src="activation_interventions.png" alt="逐H装配存在概率">')
    parts.append('<table><tr><th>分支</th><th>后端</th><th>结构采样新增记录</th><th>数值导出/请求数</th><th>诊断候选</th><th>期望新增</th><th>零新增概率</th></tr>')
    for b in branches:
        s = b['summary']; law = b['law'] or {}; requested = b['config']['worlds']
        parts.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [b['label'],b['backend'],s.get('proposed_sample_counts',[]),
            str(len(s.get('worlds',[])))+'/'+str(requested),len(s.get('blocked_worlds',[])),law.get('expected_hidden_instances'),law.get('empty_probability')])+'</tr>')
    parts.append('</table>')
    comparisons = {}
    controls = [b for b in branches if b['label']==current_version+'/layout_control']
    paths = [b['completion']/'structural_samples.json' for b in controls]
    if len(paths)==2 and all(p.is_file() for p in paths):
        equal = sha(paths[0])==sha(paths[1]); comparisons['layout_control_structural_samples_identical'] = equal
        if not equal: raise ValueError('Metric backend control used different inventories/noise')
    for backend in ('qwen','api'):
        a = next((b for b in branches if b['label']==current_version+'/end_to_end' and b['backend']==backend and b['law']),None)
        c = next((b for b in branches if b['label']==current_version+'/function_free' and b['backend']==backend and b['law']),None)
        if a and c:
            first,second = a['law'],c['law']; delta = max(abs(p-second['object_presence'][k]) for k,p in first['object_presence'].items())
            if delta>1e-8: raise ValueError('H-cut did not preserve module/object marginals')
            comparisons[backend+'_H_cut'] = dict(max_object_marginal_delta=delta,
                count_PMF_TV=sum(abs(p-second['hidden_count_probabilities'].get(n,0)) for n,p in first['hidden_count_probabilities'].items())/2,
                scope='H dependence removed; joint activation and module marginals preserved; H still influences metric stage')
    for b in branches:
        s = b['summary']; parts.append('<section class="branch"><h2>'+escape(b['label']+' / '+b['backend'])+'</h2>')
        parts.append('<p>状态：'+escape(s['status'])+'</p>')
        for filename,label in [('probability_audit.html','概率机制'),('mechanism_graph.svg','实际模型结构'),('functional_probability_interventions.json','逐H概率干预'),('mechanism_diagnostics.json','匹配功能干预与弱机制诊断')]:
            p = b['completion']/filename
            if p.is_file(): parts.append('<a href="'+link(p,out)+'">'+label+'</a>　')
        for failure in s.get('failures',[]): parts.append('<p class="missing">'+escape(json.dumps(failure,ensure_ascii=False))+'</p>')
        if s.get('error'): parts.append('<p class="missing">'+escape(s['error'])+'</p>')
        for w in b['worlds']:
            record = w['record']; dest = out/b['id']/record['folder']; dest.mkdir(parents=True)
            write(dest/'G_display.json',w['state'])
            fig = interactive_boxes(w['state'],b['config']['hfov'],b['config']['image_size'],dest/'scene_3d.html',
                title=b['label']+' / '+b['backend']+' / '+record['folder']+' / '+record['status'],scene_bounds=bounds)
            figures[(b['id'],record['folder'])] = fig
            footprint(w['state'],bounds,b['config']['hfov'],dest/'layout_xz.png',b['id']+'/'+record['folder'])
            erp_map(w['state'],b['config']['hfov'],b['config']['image_size'],dest/'erp_state_map.png')
            parts.append('<article class="world"><h3>'+escape(record['folder']+' / '+record['status'])+'</h3>')
            parts.append('<p>残留非许可 AABB 交叠：'+str(record.get('remaining_AABB_overlap_pairs'))+'；本页所有视图使用统一坐标范围。</p>')
            parts.append('<iframe class="scene" loading="lazy" src="'+link(dest/'scene_3d.html',out)+'"></iframe>')
            for filename in ('layout_xz.png','erp_state_map.png'): parts.append('<img loading="lazy" src="'+link(dest/filename,out)+'">')
            parts.append('<details open><summary>生成物体类别核对</summary><table><tr><th>实例</th><th>原始类别</th><th>在类别表中</th><th>数据集标签 / ID</th><th>映射状态</th></tr>')
            tax = []
            for obj in w['state']['objects']:
                if obj['local_evidence']=='visible': continue
                t = obj['taxonomy_review']; tax.append(dict(id=obj['track_id'],**t))
                parts.append('<tr style="color:'+obj['visual_color']+'">'+''.join('<td>'+escape(str(v))+'</td>' for v in [f'N{len(tax):02d} / '+obj['track_id'],t['raw_label'],t['raw_label_in_dataset_table'],
                    str(t['dataset_category_label'])+' / '+str(t['dataset_category_id']),t['mapping_status']])+'</tr>')
            parts.append('</table><p><a href="'+link(w['source'],out)+'">原始状态</a> · <a href="'+link(dest/'G_display.json',out)+'">附加类别标记的显示副本</a></p></details></article>')
            audit.append(dict(branch=b['id'],world=record['folder'],status=record['status'],taxonomy=tax,source=str(w['source'])))
        parts.append('</section>')
    # Small groups of synchronized panels stay legible; all use the global bounds.
    pairs = []
    for label in sorted({b['label'] for b in branches}): pairs.append([b['id'] for b in branches if b['label']==label])
    for backend in ('qwen','api'): pairs.append([b['id'] for b in branches if b['backend']==backend and (b['label'].endswith('/end_to_end') or b['label']==current_version+'/function_free')])
    links = []
    for i,methods in enumerate(pairs):
        selected = {(m,w):f for (m,w),f in figures.items() if m in methods}
        if len(methods)<2 or not selected: continue
        filename = 'synchronized_'+str(i)+'.html'; _synchronized(selected,methods,bounds,out/filename)
        links.append('<p><a href="'+filename+'">同步旋转：'+escape(' / '.join(methods))+'</a></p>')
    parts[3:3] = links
    write(out/'vocabulary.json',dict(vocabulary(),source=str(VOCABULARY),sha256=sha(VOCABULARY)))
    write(out/'review.json',dict(root=str(root),baseline_root=str(baseline_root) if baseline_root else None,
        source_sha256=sources,shared_RGB_sha256=next(iter(rgb_sha)),shared_G_obs_sha256=next(iter(observed_sha)),
        shared_bounds=bounds,comparisons=comparisons,worlds=audit,source_worlds_modified=False,actual_RGB_panorama=False,
        branch_status={b['id']:b['summary']['status'] for b in branches},causal_gain_validated=False))
    _plots(branches,out,current_version); (out/'index.html').write_text(''.join(parts))
    if not external_output:
        (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>'+current_version+' 实验对比</h1><a href="'+link(out/'index.html',root)+'">打开概率、类别和完整世界可视化对比</a>')
    return out/'index.html'


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path); p.add_argument('--out',type=Path)
    a = p.parse_args(); print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__': main()
