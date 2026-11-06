"""v13 actual completion, interface diagnosis and frozen-world comparison."""
from html import escape
from itertools import combinations
import json
from pathlib import Path

from ..common import read,write
from . import assembly_organized as scm
from .assembly_holistic_visual import report as generic_report,table


def diagnostics(root):
    d=read(root/'shared/domain.json');result={}
    for b in ('qwen','api'):
        path=root/'shared/mechanisms'/(b+'.json')
        if not path.exists():result[b]=dict(status='mechanisms_missing');continue
        t=read(path);m=scm.compile_model(d,t);conditional={h['id']:scm.marginals(m,h['id']) for h in d['profiles']};q=list(conditional)
        roles={}
        for role in d['slots']:
            rows=[t['presence'][h+'|'+role] for h in q];roles[role]=dict(
                true_draws=sum(r['counts'].get('1',0) for r in rows),valid_draws=sum(r['valid_draws'] for r in rows),
                min_presence=min(m['activation_rows'][role][h][1] for h in q),max_presence=max(m['activation_rows'][role][h][1] for h in q),
                max_variant_TV=max(.5*sum(abs(x-y) for x,y in zip(m['variant_rows'][role][h],m['variant_rows'][role][j])) for h,j in combinations(q,2)))
        p=root/'shared'/('interface_probe_'+b+'.json');probe=read(p) if p.exists() else dict(complete=False,rows={},errors=['not available'])
        previous=read(root/'shared/source_mechanisms'/(b+'.json'))
        result[b]=dict(status='available',roles=roles,conditional_laws=conditional,
            original_v12_presence=previous['presence'],same_oldH_new_interface=probe,
            context_effect_identified_in_real_world=False,probabilities_calibrated=False,
            H_cut_preserves_unconditional_inventory_by_design=True)
    write(root/'mechanism_diagnostics.json',result);return result


def report(root):
    root=Path(root).resolve();run=read(root/'run.json');diagnosis=diagnostics(root)
    page=generic_report(root,[Path(run['source_run'])],root/'visual_v13_001',current_label='v13',
        gallery_experiments=('end_to_end','layout_control','geometry_control'),baseline_experiments=('end_to_end','layout_control'))
    parts=['<section><h2>本轮优化的可核查对照</h2><p>固定 v12 全部丰富候选。主线重新估计功能组织、存在、装配及方位；尺寸冻结后联合放置同一宿主物品。没有对象数或后方配额。</p>',
        '<p>geometry_control 精确重放 v12 layout_control 的语义世界和方位，重新估计尺寸与位置；不是固定旧尺寸实验。两后端使用相同方法和配置；尺寸阶段输出不同会使后续放置输入不同。</p>',
        '<p>interface_oldH 仅改变旧 H 下的存在接口，不生成额外几何。H-cut 保留无条件离散分布，不能以平均数量相同判失败。程序概率全真或 H 无作用都会原样显示。</p>',
        '<p><a href="../mechanism_diagnostics.json">存在率与条件作用完整诊断</a></p>',
        table(['后端','角色','true/有效响应','最小存在率','最大存在率','最大条件变体 TV'],
              [[b,r,str(v['true_draws'])+'/'+str(v['valid_draws']),round(v['min_presence'],4),round(v['max_presence'],4),round(v['max_variant_TV'],4)] for b,d in diagnosis.items() if d['status']=='available' for r,v in d['roles'].items()]),
        '<details><summary>固定调用预算与范围</summary><pre>'+escape(json.dumps(read(root/'call_budget.json'),ensure_ascii=False,indent=2))+'</pre></details></section>']
    text=page.read_text().replace('RGB 对照仅改变选择阶段图像输入，本地初始 draw seed 相同；服务端 API 未传 seed。','')
    if run.get('synthetic_fixture'):text='<h1>CPU 合成 fixture，非真实模型效果</h1>'+text
    page.write_text(text+''.join(parts));return page
