"""v12 actual worlds vs richer early versions; no best-of selection."""
import argparse
from html import escape
import json
import os
from pathlib import Path

from ..common import read,write
from .assembly_holistic_visual import report as generic_report


def report(root,baseline_root=None,out=None):
    root=Path(root).resolve();dest=Path(out).resolve() if out else root/'visual_v12_001'
    # Original renderer preserves all planned primary/control worlds. It only
    # reads baseline folders and writes into this new run's destination.
    page=generic_report(root,baseline_root,dest,current_label='v12')
    run=read(root/'run.json');budget=read(root/'call_budget.json') if (root/'call_budget.json').exists() else {}
    scope=('固定旧版语义候选，重新估计用途、程序存在、具体装配及相机方位；旧概率和布局未复用。不是新的自主提案能力实验。' if run['catalog_source'] else
        '从固定局部观测重新提出完整功能程序，再估计存在、装配与方位；未复用旧语义候选、概率或布局。')
    section='<section><h2>完整功能程序：候选来源与条件机制</h2><p>'+scope+'</p>'
    section+='<p>存在先验与非空备选数量分开；没有物体数、功能区数或后方比例目标。H-cut 保留完整离散边缘分布。</p><pre>'+escape(json.dumps(dict(catalog=run['catalog_source'],layout_RGB=run['layout_image_delivery'],budget=budget),ensure_ascii=False,indent=2))+'</pre></section>'
    text=page.read_text().replace('RGB 对照仅改变选择阶段图像输入，本地初始 draw seed 相同；服务端 API 未传 seed。','')
    if run.get('synthetic_fixture'):text='<meta charset="utf-8"><h1>合成 CPU fixture：不是实际模型效果</h1>'+text
    page.write_text(text+section)
    (root/'index.html').write_text('<meta charset="utf-8"><h1>v12 完整功能程序补全</h1><a href="'+os.path.relpath(page,root)+'">打开实际状态与对照</a>')
    return page


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,action='append');p.add_argument('--out',type=Path)
    a=p.parse_args();print(report(a.root,a.baseline_root,a.out))


if __name__=='__main__':main()
