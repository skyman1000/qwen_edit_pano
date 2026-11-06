"""Joint assembly-set audit, actual contents, RGB ablation and legacy gallery."""
import argparse
from html import escape
import json
from pathlib import Path

from ..common import read,write
from . import assembly_set as model
from .assembly_panorama_visual import report as previous_report


def report(root,baseline_root=None,out=None):
    root=Path(root).resolve();page=previous_report(root,baseline_root,out);dest=page.parent;details={}
    for b in ('qwen','api'):
        rows={}
        for exp in ('end_to_end','tag_only','direct_inventory','function_free'):
            p=root/exp/b/'completion/causal_model.json'
            if not p.exists():continue
            m=read(p);law=model.marginals(m)
            rows[exp]=dict(E_N=law['expected_hidden_instances'],empty_probability=law['empty_probability'],
                unit_presence=law['activation_probabilities'],expected_categories=law['expected_category_counts'],
                fixed_context_parents=m['fixed_context_parents'])
        details[b]=rows
    audit=read(dest/'review.json');audit['joint_assembly_set_review']=dict(backends=details,
        function_slot_exclusivity=False,inventory_H_cut_scope='Preserves full unconditional joint-unit inventory law; tests H response/association, not a quantity improvement',
        comparison_scope='tag_only removes RGB only at joint selection; same text D, recognition and geometry. All failures remain in the denominator.')
    write(dest/'review.json',audit)
    section='<section><h2>联合功能装配：减少人为互斥与信息瓶颈</h2><p>同一用途、不同锚点的装配可以共存。图像与 H 一起进入实际联合选择，无数量或后方配额。</p>'
    section+='<p>tag_only 只移除联合选择阶段的 RGB，保留同一 D、识别与几何。function_free 保留完整库存边缘分布，不期待平均数量变化。</p>'
    section+='<p>直接与功能联合选择的初始请求数、单次输出上限匹配；实际修复、token 使用量分别记录。共享提案域不代表各模型独立提案能力。</p><pre>'+escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></section>'
    page.write_text(page.read_text()+section);return page


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__':main()
