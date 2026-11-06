"""Category-labelled galleries, measured result tables and a meeting slide deck."""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import math
import os
from pathlib import Path

from PIL import Image,ImageDraw,ImageOps

from ..common import ROOT,new_output,read,write,sha
from .assembly_presentation_visual import render_world,label_rows,font
from .assembly_pilot_visual import shared_bounds


def snapshot(roots,out):
    return {str(p):sha(p) for root in roots for p in root.rglob('*.json') if p.is_file() and not p.is_relative_to(out)}


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'


def csv_file(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect_run(root,scene):
    run=read(root/'run.json');metrics=[];worlds=[];effects=[]
    for branch in run['experiments']:
        config=read(root/branch/'experiment.json')
        for backend in ('qwen','api'):
            c=root/branch/backend/'completion';summary=read(c/'summary.json') if (c/'summary.json').exists() else {}
            for i in range(1,run['worlds']+1):
                w=c/f'world_{i:02d}';sem=read(w/'semantic_world.json') if (w/'semantic_world.json').exists() else None
                path=w/'G_full.json';complete=path.exists()
                if not complete:path=w/'G_full_candidates.json'
                state=read(path) if path.exists() else None;obs=read(c/'G_obs.json') if (c/'G_obs.json').exists() else None
                row=dict(scene=scene,branch=branch,backend=backend,world=w.name,status='exported' if complete else 'numeric_candidate' if state else 'failed' if summary else 'not_run',
                    observed=len(obs['objects']) if obs else None,proposed=len(sem['objects']) if sem else None,
                    placed_new=None,rear_new=None,central_rear_new=None,placement_roots=None,rear_roots=None,unmapped_new=None,
                    numeric_passed=None,overlap_pairs=None,observed_preserved=None)
                if state:
                    if obs is None or sem is None:raise ValueError('Missing observed/semantic binding: '+str(w))
                    if state['objects'][:len(obs['objects'])]!=obs['objects']:raise ValueError('Changed observations: '+str(w))
                    generated=state['objects'][len(obs['objects']):];byid={o['id']:o for o in sem['objects']}
                    if {o['track_id'] for o in generated}!=set(byid):raise ValueError('Changed inventory: '+str(w))
                    roots=[o for o in generated if byid[o['track_id']]['attachment']['host'] is None]
                    numeric=read(w/'numeric_review.json');labels=label_rows(state)[len(obs['objects']):]
                    row.update(placed_new=len(generated),rear_new=sum(o['center_local_m']['value'][2]<0 for o in generated),
                        central_rear_new=sum(abs(math.degrees(math.atan2(o['center_local_m']['value'][0],o['center_local_m']['value'][2])))>135 for o in generated),
                        placement_roots=len(roots),rear_roots=sum(o['center_local_m']['value'][2]<0 for o in roots),unmapped_new=sum(not r['mapped'] for r in labels),
                        numeric_passed=numeric['geometry_contract_passed'],overlap_pairs=len(numeric['program_solver']['final_overlaps']),observed_preserved=True)
                    worlds.append(dict(key=scene+'__'+branch+'__'+backend+'__'+w.name,state=state,image=root/branch/'input.png',config=config,row=row))
                metrics.append(row)
            model_path=c/'causal_model.json'
            if branch=='end_to_end' and model_path.exists():
                model=read(model_path)
                for role,entries in model.get('activation_rows',{}).items():
                    for h in model['profiles']:
                        effects.append(dict(scene=scene,backend=backend,H=h['id'],organization=h.get('primary_role') or 'mixed',role=role,presence_probability=entries[h['id']][1],probabilities_calibrated=False))
    return metrics,worlds,effects


def model_figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
    fig,ax=plt.subplots(figsize=(14,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
    nodes={'O':(1.5,5.5,'O+: observed RGB objects'),'H':(5.,5.5,'H: organization hypothesis'),
        'A':(8.5,5.5,'A: hidden program exists'),'V':(8.5,3.5,'V: complete program + R'),
        'E':(12.,3.5,'E: object inventory'),'S':(12.,1.5,'S: installation dependency'),
        'B':(5.,1.5,'B: frozen intrinsic sizes'),'X':(8.5,1.5,'X: joint position / yaw')}
    for k,(x,y,text) in nodes.items():
        ax.add_patch(FancyBboxPatch((x-1.48,y-.37),2.96,.74,boxstyle='round,pad=.08',facecolor='#edf3fb',edgecolor='#486a91'));ax.text(x,y,text,ha='center',va='center',fontsize=10)
    for a,b in [('O','H'),('H','A'),('H','V'),('H','X'),('A','V'),('V','E'),('E','S'),('V','B'),('B','X'),('V','X'),('S','X'),('O','X')]:
        x,y=nodes[a][:2];u,v=nodes[b][:2];dx=u-x;dy=v-y;scale=max(abs(dx)/1.57,abs(dy)/.48)
        ax.add_patch(FancyArrowPatch((x+dx/scale,y+dy/scale),(u-dx/scale,v-dy/scale),arrowstyle='-|>',mutation_scale=14,color='#49576a',linestyle='--' if (a,b)==('O','H') else '-',connectionstyle='arc3,rad=.1' if (a,b)==('O','X') else 'arc3'))
    ax.text(.2,.35,'Dashed O -> H is inference. O+, D condition the mechanisms; this is not identified physical causation.\nS also depends on the host installation. V determines camera regions R. No training or object/rear quota.',fontsize=11)
    fig.tight_layout();fig.savefig(out/'model_structure.png',dpi=160);fig.savefig(out/'model_structure.svg');plt.close(fig)


def result_figure(out,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for backend,color in (('qwen','#287db5'),('api','#e88c13')):
        rows=[r for r in metrics if r['branch']=='end_to_end' and r['backend']==backend and r['status']=='exported']
        if not rows:continue
        x=list(range(1,len(rows)+1));axes[0].plot(x,[r['placed_new'] for r in rows],'o-',label=backend,color=color)
        axes[1].plot(x,[r['rear_new'] for r in rows],'o-',label=backend,color=color)
    axes[0].set(title='Actual exported hidden inventory',xlabel='Scene/world record',ylabel='Objects')
    axes[1].set(title='Objects behind the original camera (Z < 0)',xlabel='Scene/world record',ylabel='Objects')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    fig.suptitle('Descriptive counts; not semantic accuracy or causal gain');fig.tight_layout();fig.savefig(out/'measured_results.png',dpi=160);plt.close(fig)


def finish(out,metrics,gallery,effects,input_tiles,title,pending,synthetic=False):
    csv_file(out/'world_metrics.csv',metrics);csv_file(out/'conditional_probabilities.csv',effects);write(out/'results.json',dict(metrics=metrics,effects=effects,pending_inputs=pending,synthetic_fixture=synthetic))
    model_figure(out);result_figure(out,metrics)
    contact=Image.new('RGB',(1000*max(1,len(input_tiles)),1060),'white');d=ImageDraw.Draw(contact)
    for j,(label,path) in enumerate(input_tiles):
        with Image.open(path) as im:tile=ImageOps.contain(im.convert('RGB'),(980,980))
        contact.paste(tile,(j*1000+(1000-tile.width)//2,60));d.text((j*1000+15,15),label,font=font(22),fill='black')
    contact.save(out/'test_inputs.png')
    summary=[]
    for scene,branch,backend in sorted({(r['scene'],r['branch'],r['backend']) for r in metrics}):
        rows=[r for r in metrics if (r['scene'],r['branch'],r['backend'])==(scene,branch,backend)];valid=[r for r in rows if r['status']=='exported']
        summary.append([scene,branch,backend,[r['proposed'] for r in rows],str(len(valid))+'/'+str(len(rows)),sum(r['placed_new'] for r in valid),sum(r['rear_new'] for r in valid),sum(r['overlap_pairs'] for r in valid)])
    description=('CPU 合成 fixture，不是真实模型效果。' if synthetic else '')+'灰色ERP区域未观测，未生成隐藏区域RGB。平面图为相机XZ状态投影，没有测量墙体。蓝：观测；橙：可映射新增；红：未映射新增。'
    results_table=table(['输入','分支','后端','各候选计划新增','导出/计划','已导出新增累计','后方累计','碰撞对'],summary)
    html='<!doctype html><meta charset="utf-8"><title>'+escape(title)+'</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{border:1px solid #ddd;padding:10px}</style><h1>'+escape(title)+'</h1><p>'+description+'</p><p><a href="group_meeting.html">打开组会幻灯片</a> | <a href="group_meeting.md">组会讲稿</a> | <a href="world_metrics.csv">逐世界实验表CSV</a> | <a href="conditional_probabilities.csv">实际条件概率CSV</a></p><img src="test_inputs.png">'
    if pending:html+='<p>尚未完成新模型生成的输入：'+escape(', '.join(pending))+'。它们不计作已有实验结果。</p>'
    html+=results_table+'<img src="measured_results.png"><h2>当前模型</h2><img src="model_structure.png"><div class="grid">'
    for key,folder in gallery:html+='<article><h3>'+escape(key)+'</h3><a href="'+folder+'/index.html"><img loading="lazy" src="'+folder+'/overview.png">打开原图、平面图、原视角及六方向投影、三维图、2:1 ERP</a></article>'
    html+='</div>';(out/'index.html').write_text(html)
    effect_rows=[r for r in effects if r['backend']=='api' and r['role']=='work']
    md='# 组会汇报：局部观测到合理的360°世界状态\n\n'+description+'\n\n'
    md+='## 研究问题\n单张局部RGB不能唯一确定隐藏真相。目标是在保留观测的条件下生成语义、安装依赖与空间结构相容的完整世界候选；本阶段输出物体状态框，不进行训练。\n\n'
    md+='## 当前模型\nO+为固定预测观测，D为该图候选域。q(H|O+,D)推断功能组织侧重；A_r决定额外隐藏完整程序是否存在；V_r,R_r选择完整装配与原相机空间计划；E_i为对象库存；S_i为安装依赖，取决于E_i和宿主状态；B_i尺寸先冻结，再联合生成位置与朝向X_i。\n\n'
    md+='离散分解：`q(H|O+,D) × ∏r p(A_r|H,O+,D) p(V_r,R_r|A_r,H,O+,D)`。连续阶段为条件大模型输出与确定性数值编译，当前没有完整RGB似然。不存在住户具体生活历史C变量。\n\n'
    md+='存在率：`π_r(h)=(n_present+α/2)/(K_valid+α)`，α=1；实际条件选择频数进入CPT和逆CDF采样。程序内部对象共同进入库存；几何失败保留库存而非删除对象。H之间无效应时原样保留，不能强制创造因果差异。\n\n'
    if effect_rows:md+='本报告API办公存在率（模型内未校准）：\n\n'+'\n'.join('- '+r['scene']+' / H='+r['organization']+': '+str(round(r['presence_probability'],4)) for r in effect_rows)+'\n\n'
    md+='## 实验设置与结果\nend_to_end：各后端条件概率与布局；layout_control：共享两后端原始选择计数池化的概率表与离散库存，再比较布局；function_free：切断H到计算耦合J的联系，J独立按同一q(H)抽样，保持无条件离散联合分布。它不是直接库存强基线，不能由平均数量相近判断因果收益。\n\n'
    md+='两后端为Qwen3-VL-8B-Instruct与硅基流动deepseek-ai/DeepSeek-V4.1-Flash；图内共享观测与候选、方法和配置一致，模型输出可不同。服务端不传未受支持的随机种子。每个新图重新提案；保留失败、重复、空集。\n\n'
    md+='|输入|分支|后端|计划新增|导出/计划|导出新增累计|后方累计|碰撞对|\n|---|---|---|---|---|---|---|---|\n'+'\n'.join('|'+ '|'.join(str(x) for x in r)+'|' for r in summary)+'\n\n'
    if pending:md+='待运行的新输入：'+', '.join(pending)+'。没有把待运行或CPU合成fixture当成真实新实验。\n\n'
    md+='## 能汇报的结论与边界\n现有结果能检验接口成功率、观测保持、库存保留、周围方位支持、框级碰撞与安装依赖。它们不是语义正确率或真实隐藏恢复准确率。概率来自少量冻结大模型条件采样，未校准；假设性SCM尚未识别真实物理因果。H条件独立和有限候选支持是限制。\n\n'
    md+='已知局部问题包括服务目标与空间计划不一致、个别具体形态尺寸、缺少统一资产正面、未测量墙面与房间边界。下一步用新局部输入验证，不添加物体/后方数量配额；只有定位到重复问题才修改机制。\n'
    (out/'group_meeting.md').write_text(md)
    examples=''.join('<p>'+escape(k)+'</p><img src="'+f+'/overview.png">' for k,f in gallery[:2])
    slides=[('任务与研究目标','<p>局部RGB → 固定观测 O+ → 合理的完整360°世界候选</p><p>对象类别、数量、依赖、尺寸、位置与朝向；免训练。</p><p>'+description+'</p>'),
        ('可执行的条件结构生成模型','<img src="model_structure.png"><p>q(H|O+,D) ∏ p(Aᵣ|H,O+,D) p(Vᵣ,Rᵣ|Aᵣ,H,O+,D)</p>'),
        ('方法与对照','<p>完整功能程序共同生成家具及其普通用品。条件概率实际进入抽样。</p><p>end_to_end：主线；layout_control：共享离散库存后比较布局；function_free：保边缘的 H 断边对照。</p><p>尺寸先冻结，随后联合放置；失败原样记录。</p>'),
        ('实际实验结果',results_table+'<img src="measured_results.png"><p>这是数量、方位与数值可执行性统计，不是语义准确率或因果收益证明。</p>'),
        ('输入与可视化示例',examples or '<img src="test_inputs.png"><p>新输入生成尚未执行。</p>'),
        ('已知问题与下一步','<p>服务对象和空间计划相容性、具体形态尺寸与固有轴、未测量墙面与房间边界。</p><p>先固定因果骨架与完整装配粒度，使用新图独立候选做跨输入验证。</p><p>未训练、概率未校准、未识别真实因果。新测试待完成：'+escape(', '.join(pending) or '见实际结果表')+'</p>')]
    deck='<!doctype html><meta charset="utf-8"><title>组会汇报</title><style>body{margin:0;font:24px sans-serif;background:#edf2f7}section{display:none;padding:40px;box-sizing:border-box;min-height:100vh;background:white}section.active{display:block}img{max-width:95%;max-height:65vh}table{font-size:16px;border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px}footer{position:fixed;bottom:5px;right:15px;color:#657487;font-size:15px}@media print{section{display:block;page-break-after:always}footer{display:none}}</style>'
    deck+=''.join('<section'+(' class="active"' if i==0 else '')+'><h1>'+t+'</h1>'+b+'</section>' for i,(t,b) in enumerate(slides))
    deck+='<footer>← → 翻页；浏览器打印可导出PDF</footer><script>let i=0;const s=[...document.querySelectorAll("section")];document.addEventListener("keydown",e=>{if(["ArrowRight","PageDown"," "].includes(e.key))i=Math.min(i+1,s.length-1);if(["ArrowLeft","PageUp"].includes(e.key))i=Math.max(i-1,0);s.forEach((x,j)=>x.classList.toggle("active",j===i));});</script>'
    (out/'group_meeting.html').write_text(deck)


def build(references,out,title,all_branches=False,suite_root=None):
    out=new_output(out);roots=[p for _,p,_ in references if p is not None];before=snapshot(roots,out)
    metrics=[];worlds=[];effects=[];inputs=[];pending=[];synthetic=False
    for scene,run,image in references:
        inputs.append((scene,image))
        if run is None or not (run/'run.json').exists():pending.append(scene);continue
        synthetic=synthetic or read(run/'run.json').get('synthetic_fixture',False)
        m,w,e=collect_run(run,scene);metrics+=m;worlds+=w;effects+=e
        if not any(r['status']=='exported' for r in m):pending.append(scene)
    gallery=[]
    for scene in {w['row']['scene'] for w in worlds}:
        members=[w for w in worlds if w['row']['scene']==scene and (all_branches or w['row']['branch']=='end_to_end')]
        bounds=shared_bounds([w['state'] for w in members])
        for w in members:
            folder='worlds/'+w['key']
            with Image.open(w['image']) as image:render_world(w['state'],image,w['config']['hfov'],out/folder,bounds)
            gallery.append((w['key']+' ['+w['row']['status']+']',folder))
    finish(out,metrics,gallery,effects,inputs,title,pending,synthetic)
    after=snapshot(roots,out)
    if before!=after:raise AssertionError('Visualization changed source JSON')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href') and v and not v.startswith(('http:','https:','data:','#','javascript:','mailto:')):self.paths.append(v)
    broken=[];link_count=0
    for page in out.rglob('*.html'):
        parser=Links();parser.feed(page.read_text());link_count+=len(parser.paths)
        broken.extend(str(page)+': '+v for v in parser.paths if not (page.parent/v.split('#')[0]).exists())
    if broken:raise AssertionError('Broken local visual links: '+str(broken[:10]))
    write(out/'validation.json',dict(source_JSON_count=len(before),source_JSON_unchanged=True,worlds_rendered=len(gallery),
        metric_world_records=len(metrics),pending_inputs=pending,real_model_called=False,palette='blue_observed_orange_mapped_red_unmapped',
        labels='stable_number_plus_category',ERP_ratio='2:1',source_states_modified=False,local_HTML_links=link_count,broken_links=broken))
    if suite_root is not None:(suite_root/'index.html').write_text('<meta charset="utf-8"><a href="'+escape(os.path.relpath(out/'index.html',suite_root))+'">打开多输入实验、完整可视化与组会材料</a>')
    return out/'index.html'


def report_suite(root,out=None,all_branches=False):
    from .assembly_multiscene import verify_suite
    root=Path(root).resolve();suite=verify_suite(root)
    if out is None:
        n=1
        while (root/f'presentation_{n:03d}').exists():n+=1
        out=root/f'presentation_{n:03d}'
    references=[(r['id'],root/'scenes'/r['id']/'completion',root/'scenes'/r['id']/'observation/input.png') for r in suite['scenes']]
    page=build(references,Path(out),'v13 多输入测试与组会材料',all_branches,root)
    phase_rows=[]
    for p in sorted((root/'phase_records').glob('*.json')):
        r=read(p)
        if r['status']!='complete':phase_rows.append([r['scene'],r['phase'],r['backend'],r['status'],r.get('error') or r.get('missing_inputs') or '保留有效部分提案；覆盖不完整'])
    note='<h2>阶段状态与前置依赖</h2><p>blocked_dependency 表示前置未完成，本阶段没有调用模型。部分提案保留有效方案并显示警告；不能把未执行记作空世界。</p>'+table(['输入','阶段','后端','状态','原因'],phase_rows)
    if suite.get('recovery_source'):note+='<p>本批复用旧运行成功状态、修复失败接口并新增输入，不是全部重新生成的配对能力实验。原运行：'+escape(suite['recovery_source'])+'</p>'
    page.write_text(page.read_text()+note);return page


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',type=Path);g.add_argument('--suite',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--all-branches',action='store_true');a=p.parse_args()
    if a.suite:print(report_suite(a.suite,a.out,a.all_branches))
    else:
        root=a.run.resolve();run=read(root/'run.json');branch='end_to_end' if 'end_to_end' in run['experiments'] else run['experiments'][0]
        print(build([('development_input',root,root/branch/'input.png')],a.out,'v13 已有真实结果：新版可视化与组会材料',a.all_branches))


if __name__=='__main__':main()
