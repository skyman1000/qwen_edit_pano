"""Export source-grounded meeting notes and offline vector diagrams."""
from html import escape
from pathlib import Path
import json

from ..common import read,write


def diagram(path,title,nodes,edges,subtitle):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="800" viewBox="0 0 1480 800"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#465570"/></marker></defs><rect width="1480" height="800" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#142037"><text x="35" y="44" font-size="29">'+escape(title)+'</text><text x="35" y="77" font-size="17">'+escape(subtitle)+'</text></g>']
    for a,b,dashed in edges:
        x,y,w,h,_,_=nodes[a];xx,yy,ww,hh,_,_=nodes[b]
        if a=='H' and b=='X':
            svg.append('<path d="M690 230 L690 270 L1450 270 L1450 525 L1225 525 L1225 555" fill="none" stroke="#465570" stroke-width="2" marker-end="url(#arrow)"/>')
            continue
        if yy>y+h:
            start=(x+w/2,y+h);end=(xx+ww/2,yy)
        elif xx>x+w:
            start=(x+w,y+h/2);end=(xx,yy+hh/2)
        elif xx+ww<x:
            start=(x,y+h/2);end=(xx+ww,yy+hh/2)
        else:start=(x+w/2,y+h);end=(xx+ww/2,yy)
        svg.append('<path d="M %g %g L %g %g" fill="none" stroke="#465570" stroke-width="2" marker-end="url(#arrow)" %s/>'%(*start,*end,'stroke-dasharray="7 5"' if dashed else ''))
    for x,y,w,h,lines,color in nodes.values():
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#a7b6cc"/>')
        for i,line in enumerate(lines):svg.append(f'<text x="{x+w/2}" y="{y+30+i*27}" text-anchor="middle" font-family="sans-serif" font-size="19" fill="#142037">{escape(line)}</text>')
    svg.append('</svg>');path.write_text(''.join(svg).replace('font-family="sans-serif"','font-family="Noto Sans CJK SC"').replace('O⁺','O+'))


def probability_figure(out,root,domain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    hs=[h['id'] for h in domain['profiles']];roles=list(domain['slots'])
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained',gridspec_kw={'height_ratios':[1,1.6]})
    for j,backend in enumerate(('qwen','api')):
        m=read(root/'end_to_end'/backend/'completion/causal_model.json');t=read(root/'shared/mechanisms'/(backend+'.json'))
        ax=axes[0,j];values=[t['recognition']['probabilities'][h] for h in hs]
        ax.bar(hs,values,color='#287db5' if j==0 else '#de8a27');ax.set(ylim=(0,1),ylabel='q(H | O+, D)',title=backend.upper()+': functional-organization mixture')
        for k,p in enumerate(values):ax.text(k,p+.02,f'{p:.3f}',ha='center',fontsize=10)
        ar=np.array([[m['activation_rows'][r][h][1] for r in roles] for h in hs]);ax=axes[1,j]
        image=ax.imshow(ar,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set(xticks=range(len(roles)),xticklabels=roles,yticks=range(len(hs)),yticklabels=hs,
            title='P(additional hidden role program exists | H, O+, D)',ylabel='Assumed functional organization H')
        for y in range(len(hs)):
            for x in range(len(roles)):ax.text(x,y,f'{ar[y,x]:.2f}',ha='center',va='center',color='white' if ar[y,x]>.6 else 'black')
        fig.colorbar(image,ax=ax,label='Smoothed empirical LM-choice probability')
    fig.suptitle('Actual saved kitchen mechanisms: K=4; total smoothing concentration=1; not calibrated world probabilities',fontsize=14)
    fig.savefig(out/'probability_mechanisms.png',dpi=170);fig.savefig(out/'probability_mechanisms.svg');plt.close(fig)


def build(out,causal,records):
    out=Path(out);suite=read(Path(causal)/'suite.json');sid=suite['scenes'][0]['id'];root=Path(causal)/'scenes'/sid/'completion'
    domain=read(root/'shared/domain.json');evidence={}
    for backend in ('qwen','api'):
        folder=root/'end_to_end'/backend/'completion';table=read(root/'shared/mechanisms'/(backend+'.json'))
        model=read(folder/'causal_model.json');world=read(folder/'world_01/semantic_world.json');law=read(folder/'probability_audit.json');interventions=read(folder/'functional_probability_interventions.json')
        evidence[backend]=dict(recognition=table['recognition'],main_mechanism_questions=len(table['original_parent_prompt_plan']),
            presence_h0={r:table['presence']['h0|'+r] for r in domain['slots']},
            presence_probability_ranges={r:[min(v[1] for v in model['activation_rows'][r].values()),max(v[1] for v in model['activation_rows'][r].values())] for r in domain['slots']},
            expected_hidden=law['expected_hidden_instances'],expected_hidden_under_do_H={h:v['expected_hidden_instances'] for h,v in interventions.items()},
            actual_choices={k:v for k,v in world['structural_state'].items() if k=='H' or k.startswith(('A_','M_'))},
            actual_groups=[dict(id=g['key'],role=g['function'],region=g['layout_region'],objects=sum(o['group']==g['key'] for o in world['objects'])) for g in world['groups']],
            uniform_noise_A={r:world['noise']['A_'+r] for r in domain['slots']},actual_hidden=len(world['objects']),
            source_files={name:str(folder/name) for name in ['causal_model.json','probability_audit.json','functional_probability_interventions.json','world_01/semantic_world.json']})
    write(out/'model_evidence.json',dict(scene=sid,profiles=domain['profiles'],slots=domain['slots'],spatial_choices=len(domain['spatial_choices']),
        total_candidate_instances=len(domain['base_model']['component_registry']),backends=evidence,
        sources_are_actual_saved_results=True,real_model_calls_in_this_report=0))
    probability_figure(out,root,domain)
    diagram(out/'generation_pipeline.svg','实际执行流程：局部RGB → 一份完整世界状态',{
        'I':(40,125,255,105,['局部RGB + 相机HFOV','原图与视野参数'],'#e8eef7'),
        'O':(390,125,290,105,['Qwen二维识别 → WildDet3D','固定可见状态 O⁺'],'#deeffb'),
        'D':(800,125,300,105,['两路大模型提出装配候选','程序校验 / 合并得到域 D'],'#fff0ce'),
        'T':(800,330,300,105,['Qwen 或 API 短选择 K=4','程序计票 / 平滑形成 CPT'],'#fff0ce'),
        'W':(390,330,290,105,['程序生成均匀噪声 U','按概率表抽样 H / A / V'],'#e5f5e8'),
        'E':(40,330,255,105,['程序展开对象和宿主关系','确定库存 E、安装 S、方位 R'],'#e5f5e8'),
        'X':(40,555,320,105,['大模型预测尺寸 / 初始布局','程序计算宿主坐标与支撑高度'],'#fff0ce'),
        'C':(490,555,310,105,['程序做MILP及碰撞约束搜索','固定观测、库存、尺寸与方位'],'#e5f5e8'),
        'G':(960,555,350,105,['G_full = G_obs ∪ ΔG','输出标签、平面、投影、3D、ERP'],'#deeffb')},
        [('I','O',False),('O','D',False),('D','T',False),('T','W',False),('W','E',False),('E','X',False),('X','C',False),('C','G',False)],
        '橙：大模型参与预测；绿：概率计算与数值求解。机制/几何主路径使用文字观测；照片不是场景形成原因。')
    diagram(out/'conditional_scm.svg','当前功能SCM：固定观测与候选域条件下的生成结构',{
        'O':(40,125,300,105,['O⁺：预测的可见物体及3D框','D：当前图像的候选程序域'],'#deeffb'),
        'H':(520,125,340,105,['H：候选世界功能组织假设','混合 / 某功能得到强调'],'#fff0ce'),
        'A':(1070,125,330,105,['A_r：该功能隐藏程序存在','不是观察物体存在与否'],'#e5f5e8'),
        'V':(520,330,340,105,['V_r：完整程序 + 相机空间计划','代码 M_r；无程序时 off'],'#e5f5e8'),
        'E':(40,555,300,105,['E_i：候选对象属于所选程序','拥有状态；不等同安装成功'],'#e5f5e8'),
        'R':(560,555,260,105,['R_g：left / right / rear','相对原输入相机'],'#e5f5e8'),
        'S':(40,685,340,85,['S_i = E_i AND S_host','宿主声明的安装依赖'],'#e5f5e8'),
        'X':(1040,555,370,105,['尺寸 B、朝向 Θ、位置 X','大模型初始化 + 程序约束求解'],'#fff0ce')},
        [('O','H',True),('H','A',False),('H','V',False),('H','X',False),('A','V',False),('V','E',False),('V','R',False),('E','S',False),('V','X',False),('R','X',False),('S','X',False)],
        '虚线表示由观测识别H，不是照片导致世界。O⁺、D是固定条件；图为概念结构，非从图像识别出的真实因果图。')
    q=evidence['qwen'];a=evidence['api'];lines=[]
    lines.extend(['# 组会讲解：免训练的功能因果场景补全（当前实际实现）','',
        '日期：2026-10-28。依据当前源码和保存的真实输出整理；本次只做分析和可视化，没有新GPU/API生成。',
        '', '## 1. 任务与当前结果', '',
        '输入是一张局部室内RGB和相机HFOV。输出是与已观测内容一致、覆盖周围未知空间的一份可能世界状态，而不是对看不见区域的唯一真值恢复。每件对象具有类别、米制中心、尺寸，生成对象还记录朝向、功能组、宿主/附着与方位信息。状态用于之后的全景生成；当前没有生成隐藏区域RGB。', '',
        '最新厨房两路已导出：Qwen为3个共享观测 + 15个新增，API为3 + 37个新增，原数值门槛均通过。但共享观测只保留3个吧凳，遗漏中岛、橱柜、灶具/抽油烟机等主要可见物体；错误的on_floor=false还导致没有地面锚点。因此它是跑通的状态假设，尚不是已验收的完整世界。', '',
        '四图沿用先前保存的状态，厨房用本次新观测后的状态明确替换。直接RGB基线10格中4格非空导出、2格空世界、4格失败；空/失败不计为有效完整世界。没有一张图具有四格都非空成功的结果，不能伪装成已完成的五图四路严格比较。', '',
        '| 场景 | 因果Qwen：观测+新增 | 因果API：观测+新增 | 直接Qwen | 直接API |',
        '|---|---:|---:|---|---|'])
    for sid in dict.fromkeys(r['scene'] for r in records):
        rs=[r for r in records if r['scene']==sid];cells=[]
        for r in rs:
            cells.append(str(r['visible_claimed'])+' + '+str(r['hidden']) if r['outcome']=='nonempty_world' else '空世界' if r['outcome']=='empty_world' else '失败：输出截断' if r.get('error') and ('length' in r['error'] or 'token limit' in r['error']) else '失败/缺失')
        lines.append('| '+rs[0]['scene_name']+' | '+' | '.join(cells)+' |')
    lines.extend(['','此表只表示真实保存的产物与数量，不能由“更多物体”推出“更好”。两条路线的观测器、概率机制、求解和预算同时不同，只能作为完整流程对照。', '',
        '## 2. 一句话说明当前方法', '',
        '**大模型提出和评价可能的功能装配，程序把评价变成显式条件概率，再按结构方程抽取库存、推导宿主依赖，并对预测的布局做数值求解。**', '',
        '更准确的名称是“冻结大模型驱动的、免训练的功能条件场景生成模型，带显式SCM和几何约束”。因果图是人为规定的建模假设，条件机制来自模型响应。没有从图像或数据集统计识别真实因果图，也没有通过真实干预验证因果效应。', '',
        '![实际流程](generation_pipeline.svg)', '', '![条件生成结构](conditional_scm.svg)', '',
        '## 3. 各变量究竟是什么意思', '',
        '| 变量 | 含义及实现 | 是否由大模型直接给出 |',
        '|---|---|---|',
        '| I、K | 原图、HFOV/相机内参 | 输入及程序推导内参 |',
        '| O⁺ / G_obs | 可见对象身份、2D框及预测3D AABB | Qwen识别；WildDet3D估计几何 |',
        '| D | 该图的有限候选程序、宿主关系与合法空间计划 | 两路LM提案；程序校验、去重、按功能组成程序并枚举计划 |',
        '| H | 候选世界的功能组织：混合或强调某功能 | 假设家族由程序构造；权重来自LM选择计数 |',
        '| A_r | 是否额外拥有功能r的一套隐藏程序 | 条件概率来自LM选择；最终0/1由程序采样 |',
        '| V_r / M_r | 选择哪套完整装配及相机方位计划 | LM回答候选别名；程序估计概率并采样 |',
        '| E_i | 对象i是否属于选中的程序，即拥有/存在 | 从已选程序确定性展开 |',
        '| S_i | 声明宿主依赖是否满足 | 根据E和宿主安装状态确定性计算 |',
        '| R_g | 功能组相对原相机的left/right/rear等区域 | 从V的空间计划展开 |',
        '| B_i、Θ_i、X_i | 固有尺寸、朝向、位置 | LM预测数值初值；程序计算变换和约束布局 |',
        '| U_i | 各离散节点的均匀随机数 | NumPy随机数；保存用于重放 |', '',
        'H不表示住户具体生活史，不强断言正在发生某项活动。强调烹饪不意味着只能有烹饪物品，也不意味着必须增加独立烹饪区。A_r的“缺席”只表示没有这套额外隐藏程序，不会删除原图已经看见的物体。', '',
        '代码还有一个J节点：主方法令J=H，用它给各角色机制提供共同功能上下文；function_free离散消融令J独立按相同边缘分布抽样，从而切断实际H与库存的耦合。当前single运行仅有end_to_end，没有新运行这项消融。图中为可读性省略J。', '',
        '## 4. 从一张图到世界的实际步骤', '',
        '① 观测：Qwen输出可见实例和框，重复候选转整图复核，必要时判断实例身份。WildDet3D对每个保留实例的类别及2D框估计三维框。坐标统一为X向右、Y图像向上、Z向前，后方Z<0。O⁺随后冻结，补全过程不移动或删除它。WildDet3D是冻结的专用三维检测网络，不是用于估计概率的聊天大模型。', '',
        '② 提案：两路大模型分别根据同一原图/观测提出可能用途、家具与生活物品装配、附着关系及语义放置提示。程序把合法提案合并成当前图像的共同有限域D。新图重新提案，不从训练集拟合库，也不训练网络。候选不是最终库存，候选合计数量不是每个世界的目标。', '',
        '③ 空间候选：没有固定锚点或显式区域的组，程序枚举left/right/rear计划并检验已有方向约束；固定宿主或有明确约束的组保留相应合法域。不要求后方必须出现对象，不靠配额塞物体。', '',
        '④ 机制估计：固定同一O⁺、D，两路分别调用LM估计q(H|O⁺,D)、p(A_r|H,O⁺,D)、p(V_r|A_r=1,H,O⁺,D)。主路径是文字观测；最新suite设置layout_image_delivery=false，因此不能说所有阶段都在重新看原图。原图在观察和语义提案时使用。', '',
        '⑤ 概率编译与抽样：程序将票数平滑为CPT，创建拓扑有序的离散图；对每个节点抽U，按逆CDF执行结构方程，展开完整程序中的对象、类别、宿主和方位。此步骤不请求LM重新自由生成一套对象。', '',
        '⑥ 几何：LM先只预测各选中物体的固有全尺寸；冻结尺寸后按功能组、每批最多6个缺失坐标对象，预测朝向、组距离、根物体偏移或宿主坐标。程序保留已合法数值，纠错只问缺失字段；不为修复格式重新抽库存。', '',
        '⑦ 数值编译：程序计算米制坐标、支撑高度和旋转后的相机AABB；同宿主物品做MILP联合摆放，全局再做有限预算的约束/碰撞搜索。保留观测、库存、尺寸、宿主、采样区域，不通过删物体或缩尺寸换取通过。数值失败保存完整候选，不当作“物体不存在”的新概率。', '',
        '⑧ 导出：G_full = G_obs ∪ ΔG。生成局部投影、平面、六方向投影、3D、2:1 ERP状态图及编号+类别表。图只是对象框状态，不是CAD网格和已生成的隐藏纹理。', '',
        '## 5. 概率怎么来的：不是让模型随口报一个0.8', '',
        '当前K=4：在同一个父节点条件下，用四种别名/顺序编码询问LM，让它只输出一个允许的choice；程序解码、计数。交换“有/无”的别名绑定及选项次序，减少固定词或位置偏好；不会强制两种回答各一半。变体/H选项也做配对打乱。', '',
        '若某行4次中有n次回答“存在”，程序用总浓度α=1、对称先验(1/2,1/2)：', '',
        'p̂(A_r=1 | h,O⁺,D) = (n + α/2) / (K_valid + α)。', '',
        '因此n=0,1,2,3,4分别得到0.1、0.3、0.5、0.7、0.9。0.9来自“4票存在 + 0.5先验伪计数”，不是模型直接输出0.9，也不是从真实房间中测得90%。若有效票不是4则按实际分母；整行无有效票失败，不能补一个默认概率。', '',
        'H有L个合法假设时，均匀先验给q̂(h)=(n_h+α/L)/(K_valid+α)。最新厨房L=6；Qwen对h0有3票、h3有1票，因此q(h0)=(3+1/6)/5=0.6333，q(h3)=0.2333，其余各0.0333。API四票全选h2，所以q(h2)=(4+1/6)/5=0.8333，其余各0.0333。', '',
        '对于程序/空间变体s：p̂(s|A=1,h)=(n_s+ηπ_s)/(K_valid+η)，η=1。π先在同角色的语义程序间均分，再在该程序的合法空间计划间均分，避免“计划更多的程序”仅靠枚举数量获得更高先验。', '',
        '真实例：Qwen在h0、cooking下的四票为a6_p0三次、a6_p1一次、a6_p2零次。这个角色只有一个语义程序、三个计划，因此π=1/3，得到(0.6667,0.2667,0.0667)。该程序存在率0.7；选择a6_p1的联合条件概率为0.7×0.2667≈0.1867。a6中的四个对象作为整套候选共同拥有，并非各独立抛一次硬币。', '',
        '**这些是冻结LM响应的平滑经验分布，没有世界真值校准。** K=4很小，编码变化和采样相关性也使它们不能视作4个独立真实房间。α=1、对称先验和逆CDF耦合都是明确设定的建模选择，不是从数据中学得。', '',
        '![真实机制概率：上为H权重，下为每个H的程序存在率](probability_mechanisms.png)', '',
        '## 6. 概率怎样真实决定对象', '',
        '固定O⁺和D时，离散部分分解为：', '',
        'P(H,A,V | O⁺,D) = q(H|O⁺,D) × ∏_r p(A_r|H,O⁺,D) p(V_r|A_r,H,O⁺,D)。', '',
        '当A_r=0，V_r=off；当A_r=1，按变体分布选择完整程序和空间计划。给定H，各功能角色独立，是当前近似；同一程序内部的多件对象联合展开。', '',
        '每个节点有U_i~Uniform[0,1)，X_i=F_i^{-1}(U_i | Pa_i)。代码的二元顺序是[0,1]，所以p=0.7时区间[0,0.3)对应缺席，[0.3,1)对应存在。最新Qwen的entry噪声U≈0.1516，小于缺席区间上界0.3，因此entry缺席；不是LM在最终世界阶段自由删掉该功能。', '',
        '若对象i属于程序f，则其边缘拥有概率为：', '',
        'P(E_i=1 | O⁺,D) = Σ_h q(h) p(A_r=1|h) Σ_{s:program(s)=f} p(V_r=s|A_r=1,h)。', '',
        '期望数量由程序计算E[N_hidden]=Σ_i P(E_i=1)，类别出现率、数量PMF和区域出现率也由固定的表解析合成。不是把LM解释句中的“可能”当作概率。', '',
        '宿主依赖为S_i=E_i∧S_host；根对象无宿主时S_i=E_i。干预移除宿主会使其子物体在当前安装方式下无效，但不会改写子物体原有的拥有状态。S仅表达声明依赖满足，支撑是否物理合理仍由后续几何代理另查。', '',
        '## 7. 用本次真实厨房结果串起来', '',
        '当前D含9个非空语义程序、50条候选实例、27个空间选择、5个功能角色。H由程序构造为混合加5种功能强调，共6个。每个后端的短机制提问数=K+2×|H|×|roles|×K=4+2×6×5×4=244；不包含观察、语义提案、尺寸/坐标、格式纠错和API限流重传。244不是244个世界；本次worlds=1，两路各只采样一个世界。', '',
        '两路复用同一观测与候选、相同方法和提示计划、相同结构噪声；它们分别估计自己的概率表和数值。API未发送未确认支持的seed，不能声称两个服务的内部采样随机性相同。共享候选也意味着某后端可能选择另一后端提出的程序，因此这不是完全隔离的自主提案能力比较。', '',
        '| 后端 | 实际H | 激活程序及实例数 | 新增总数 | 模型期望新增数 |',
        '|---|---|---|---:|---:|',
        '| Qwen | h0 混合 | cooking 4 + dining 5 + social 2 + storage 4；entry缺席 | 15 | '+f"{q['expected_hidden']:.4f}"+' |',
        '| API | h2 用餐强调 | cooking 4 + dining 5 + social 24 + storage 4；entry缺席 | 37 | '+f"{a['expected_hidden']:.4f}"+' |', '',
        '数量差22的直接来源是两路选择了不同social程序：2件或24件；不是程序为了让API看起来更丰富额外填物体。实际一次采样数量不必等于期望数量。', '',
        '本次还有局限：Qwen的cooking、dining、social、storage存在率在所有H下分别始终为0.7、0.9、0.9、0.9，仅entry在0.7到0.9之间变化；do(H)下期望数量约23.77至24.37。API对应期望范围约21.3033至32.33。说明当前图中Qwen对功能假设的存在机制响应很弱，不能因为图中画了H→A就断言这条边产生了强作用。不同程序/区域条件仍需分别查看。', '',
        '上面的do(H)是对已设定模型的干预分布计算：固定H后重新合成下游概率，没有真实改变房间。它能验证代码路径和表的敏感性，不能证明现实因果效应，也不能据此断言API真实因果推理优于Qwen。当前并未为每个do(H)都生成新的连续3D世界。', '',
        '## 8. 几何中的程序计算：以桌上物品为例', '',
        '大模型给出宿主桌子的尺寸和朝向，以及杯子相对宿主的u/v。程序换算：Δx=u×w_host/2，Δz=v×d_host/2，再使用宿主旋转R_host变换到相机坐标。杯子落在桌面时，中心Y由桌面顶Y+杯子高/2计算，不需要LM自由猜每个绝对坐标。', '',
        '同一桌面的多个物品由MILP选择非重叠分离关系并尽量少移动。全局碰撞搜索固定物体身份、尺寸、朝向、宿主和受保护方位；当前轮数预算min(128,max(8,2n))，不是对象数量先验。地面只能由预测的落地锚点估计，不是真实测量；最新厨房没有锚点，不能把布局的高度关系当作已验证真实地面。', '',
        '数值检查采用AABB、支撑投影、视锥和声明关系代理，未认证真实网格碰撞、人体可供性或通行。把隐藏框放到原视野之外，也不自动保证它与真实墙面或未识别的中岛相容。', '',
        '## 9. 直接RGB基线做什么，不做什么', '',
        '直接基线把相同原图和HFOV直接交给本地Qwen或API，一次生成结构化对象表：id、名称、类别、visible/outside_view、中心、固有尺寸、yaw，以及可选空间关系。最多一次格式纠正，无WildDet3D、候选程序库、H/A/V显式概率图或布局优化。程序仅校验字段、做yaw到AABB表示转换、映射类别和可视化，不调整其位置。', '',
        '无显式因果模块不代表LM内部没有常识推理。基线与完整方法不是只差一条因果边，所以不能从两者重叠数差异直接得出因果收益。尤其当前完整方法用专用三维检测和求解，直接基线没有，而且生成预算不同。应在固定O⁺、D、预算和几何流程下加入H断边等消融，再讨论功能条件机制的贡献。', '',
        '## 10. 组会上可以怎样讲，下一步做什么', '',
        '可以讲：已经实现免训练的功能条件世界状态生成；LM提出有限装配域并回答条件机制，程序统计平滑、按SCM采样和推导依赖，再用数值约束构成可检查世界。已保存五图两后端状态及直接RGB对照中的部分成功/失败例；最新厨房修复了重复记录导致的协议失败。', '',
        '不能讲：已经证明因果推理优于非因果；已从单图恢复真实因果结构；这些概率是真实世界后验；所有场景通过语义验收；厨房完整观测已经正确；所有直接基线都成功；状态ERP就是生成的RGB全景。', '',
        '建议当前先固定模型骨架，记录并修正厨房观测漏检/落地标记，并单独处理直接基线截断与空世界的公平输出接口，再补齐对照。之后在同观测/同候选/同几何/匹配预算下跑消融，观察功能机制是否改变并改善类别、数量、区域、支撑与一致性。当前没有证据要求为了数量继续改因果结构。', '',
        '可用的一分钟讲稿：“我们的目标不是猜中唯一的镜头外真值，而是生成与局部图像相容的一份可能完整世界。方法先固定可见物体的预测状态，再用功能组织变量选择完整装配和空间计划。大模型提供候选及条件选择，程序把回答计数平滑成概率，用结构方程采样对象库存并计算宿主依赖，最后预测和求解三维布局。当前证明的是机制已接入真实采样和实现能够导出状态；语义质量和因果收益仍需通过观测验收与受控消融验证。”', '',
        '## 11. 对应代码与产物', '',
        '| 功能 | 当前源码 |', '|---|---|',
        '| 多场景调度、最新机制估计 | assembly_multiscene.py：estimate、run_phase |',
        '| 功能组织、对称存在提示 | assembly_organized.py：organization_profiles、presence_prompt |',
        '| 四编码选择与请求记录 | assembly_organized_experiment.py：Requests、binary_aliases、aliases_for |',
        '| 共享候选和程序组合 | assembly_compositional_experiment.py：propose、pool |',
        '| 平滑、CPT编译、边缘概率、世界抽样 | assembly_compositional.py：predictive、compile_model、marginals、sample_worlds |',
        '| 结构方程、逆CDF、模型内干预 | world_scm.py：forward、validate |',
        '| 依赖节点与宿主干预 | assembly_population.py：build_model；assembly_modular.py：active_world、intervention_diagnostics |',
        '| 流水线输出 | assembly_world.py：_run |',
        '| 尺寸/坐标接口 | assembly_coordinate_transport.py：infer_layout、placement |',
        '| MILP和几何编译 | assembly_organized_geometry.py；assembly_spatial.py |',
        '| 直接基线 | direct_rgb_world.py |', '',
        '查看当前图的shared/domain.json了解候选，shared/mechanisms/{backend}.json了解原始票数，end_to_end/{backend}/completion/causal_model.json了解CPT，probability_audit.json了解边缘概率，functional_probability_interventions.json了解固定H后的分布。world_01/semantic_world.json保存选中程序和噪声，dimension_stage.json保存尺寸，coordinate_stage_v3.json保存坐标，numeric_review.json保存数值检查，G_full.json是最终对象状态。', '',
        '本页实际数字的机器可读来源：model_evidence.json。比较页指标及四格失败在metrics.json中。报告不写入或修改原实验产物。', '',
        '## 12. 相关工作的准确关系', '',
        '- [Human-Centric Indoor Scene Synthesis Using Stochastic Grammar](https://arxiv.org/abs/1808.08473)：使用S-AOG、功能/支撑组织及人体上下文，并学习场景分布与采样。当前仅借鉴装配结构思路，不复制其训练过程；语法不是现实因果识别的证明。',
        '- [Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image](https://arxiv.org/abs/1808.02201)：联合功能、几何、物理及合成观测差异作逆向推断。当前尚未实现其世界到图像的完整似然/analysis-by-synthesis，不能称等价复现。',
        '- [Language Models as Causal Effect Generators / SD-SCM](https://aclanthology.org/2025.emnlp-main.107/)：用户指定结构、LM定义机制，研究模型内观测/干预/反事实采样。当前是有限程序域、经验选择CPT与显式逆CDF结构噪声的借鉴实现，没有从论文直接得到室内世界的真实因果概率。', '',
        '上述论文为方法启发，真实实现和实验结论以本项目源码与保存结果为准。'])
    markdown='\n'.join(lines)+'\n';(out/'MODEL_EXPLAINED.md').write_text(markdown)
    # Render the limited Markdown used above with offline standard-library HTML.
    chunks=[];table_open=False;in_list=False
    import re
    def inline(t):
        t=escape(t);t=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',t)
        return re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2">\1</a>',t)
    for line in lines:
        if line.startswith('|'):
            if re.fullmatch(r'[| :\-]+',line):continue
            if not table_open:chunks.append('<table>');table_open=True
            chunks.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if table_open:chunks.append('</table>');table_open=False
        if line.startswith('!['):
            path=line.split('](')[1][:-1];chunks.append('<p><a href="'+path+'"><img src="'+path+'"></a></p>');continue
        if line.startswith('# '):chunks.append('<h1>'+inline(line[2:])+'</h1>')
        elif line.startswith('## '):chunks.append('<h2>'+inline(line[3:])+'</h2>')
        elif line:chunks.append('<p>'+inline(line)+'</p>')
    (out/'model_explained.html').write_text('<!doctype html><meta charset="utf-8"><title>当前因果推理模型：组会讲解</title><style>body{font:18px/1.8 system-ui,sans-serif;max-width:1180px;padding:30px;margin:auto;color:#142037}img{max-width:100%}table{border-collapse:collapse;width:100%}td{border:1px solid #ccd4df;padding:10px;font-size:16px}h2{margin-top:40px}a{color:#175aaa}</style><p><a href="index.html">四路可视化比较</a> · <a href="MODEL_EXPLAINED.md">Markdown讲稿</a> · <a href="model_evidence.json">实际概率与抽样记录</a></p>'+''.join(chunks))
