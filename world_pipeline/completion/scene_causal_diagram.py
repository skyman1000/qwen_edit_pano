"""Render a research graph specification, not a fitted/identified scene model.

The graph distinguishes historical generation, observational inversion, and
the smaller mechanism implemented in assembly_needs. No probabilities or boxes
are invented for a failed run. All output is native SVG/HTML/JSON.
"""
import argparse
from html import escape
import json
from pathlib import Path

from ..common import new_output, write


def specification():
    def node(id,label,parents,equation,meaning,status,xy):
        return dict(id=id,label=label,parents=parents,equation=equation,meaning=meaning,
            implementation_status=status,xy=xy)
    nodes=[
        node('C','C 背景与生活历史',[], 'C = f_C(U_C)',
             '资源、偏好、居住历史等联合背景。允许它们有共同原因，不声称从照片唯一识别。','latent_proposed',[30,70]),
        node('L','L 房间结构',['C'],'L = f_L(C,U_L)',
             '建筑包络、墙面、门窗。未见边界是带不确定性的假设，不能直接成为硬墙约束。','not_recovered_currently',[30,290]),
        node('H','H 预期活动组织',['C','L'],'H = f_H(C,L,U_H)',
             '房间形成时预期支持的活动与组织方式。与拍照时正在做什么、以及生成后的实际可行性分开。','text_hypotheses_only',[430,70]),
        node('R','R 需求与背景来源',['H','C'],'R = f_R(H,C,U_R)',
             '想满足的资源需求，例如显示、储存、照明；背景来源容纳收藏、闲置和装饰，不强迫所有物品有当前用途。','deterministic_given_Z_in_pilot',[830,70]),
        node('B','B 联合实现与共享角色',['R','L','C'],'B = f_B(R,L,C,U_B)',
             '可替代的联合实现、角色绑定和共享实例。观测家具可以参与实现；不同实现不是必加家具列表。','uniform_reference_catalog_strategies',[1230,70]),
        node('J','J 库存 / 类别 / 宿主',['B','C'],'(E,c,h) = f_J(B,C,U_J)',
             '完整世界的身份、存在、类别和安装地址。完整模型要能解释已观测部分；当前增量目录只覆盖新增对象。','shared_instance_union_for_hidden_delta',[1230,290]),
        node('D','D 固有尺寸',['J','B','C'],'log d_i = f_D(c_i,role_i(B),C,U_Di)',
             '尺寸同时取决于类别、角色和背景，输出分布。与随姿态变化的 camera AABB 尺寸区分。','metric_VLM_point_proposal_only',[830,290]),
        node('X','X 联合位姿',['J','D','B','L'],'X = f_X(J,D,B,L,U_X)',
             '位置与朝向的联合变量；其内部允许宿主坐标依赖。邻近、面对和碰撞是联合几何因素，不随意定向成物体因果边。','VLM_initialization_and_geometric_solver',[430,290]),
        node('K','κ 相机与视角',[],'κ = (intrinsics,extrinsics)',
             '对场景形成是外部测量条件。改变相机改变可见性和相机坐标表达，不改变固定房间中的实体配置。','fixed_camera_assumption',[30,540]),
        node('W','W 完整实体世界',['J','D','X'],'W = assemble(J,D,X)',
             '在房间坐标系中装配完整对象世界。当前 G_full 的数值坐标则表达在原相机局部坐标中。','G_obs_union_delta_in_camera_frame',[430,540]),
        node('V','V 可见性',['W','K'],'V = visibility(W,κ)',
             '视锥、遮挡与可见比例。当前只做 AABB 视锥排除，尚不做真实形状遮挡似然。','frustum_proxy_only',[830,540]),
        node('O','O 观测摘要',['W','V','K'],'O = observer(W,V,κ,U_O)',
             '折叠渲染与检测链路的证据摘要。只用 O 时不再独立乘同张 RGB 的似然；未检测不等于不存在。','G_obs_predicted_but_likelihood_unfitted',[1230,540]),
        node('A','A 实际活动可行性',['W'],'A = feasibility(W,agent)',
             '生成世界对指定 agent/活动的可达、净空、观看和操作能力。它是世界的后果；推断中可作软相容因素，不画回因果箭头。','not_physically_verified',[830,790]),
        node('G','G 相机坐标输出',['W','K'],'G = T_κ(W)',
             '转为项目所需的 x-right/y-image-up/z-forward 米制状态。相机干预时坐标会变，固定观测实验中 κ 保持不变。','current_output_representation',[1230,790])]
    spec=dict(name='role_mediated_abductive_scene_completion_research_spec_v1',nodes=nodes,
        scope='assumed_historical_SCM_design_not_identified_or_fully_implemented',
        observed_input='RGB and predicted G_obs; neither reveals the true hidden world',
        target='P(W_hidden,Z | O,κ), Z=(C,L,H,R,B)',
        reference_prior='p_theta(W,Z) ∝ p0_theta(W,Z) * phi_geom(W,L) * phi_func(W,R,B)',
        observation_update='p(W,Z | O,κ) ∝ p_theta(W,Z) * p_eta(O | W,κ)',
        factor_graph_edges_are_not_automatically_causal=True,
        RGB_likelihood_implemented=False,calibrated_posterior_implemented=False,real_causal_structure_identified=False,
        current_pilot='Z -> R_k -> B_k -> E_i -> S_i <- S_host; image-proposed domain; uniform reference law',
        required_evidence=['complete scenes with view/visibility supervision',
            'audited activity/role or program labels with alternatives',
            'held-out calibration and paired mechanism interventions'])
    validate(spec)
    return spec


def validate(spec):
    seen=set()
    for node in spec['nodes']:
        if node['id'] in seen or any(parent not in seen for parent in node['parents']):
            raise ValueError('Graph must have unique, topologically ordered nodes')
        seen.add(node['id'])
    return spec


def svg(spec):
    byid={n['id']:n for n in spec['nodes']};width=330;height=92
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 950" role="img" aria-label="假设的场景形成因果图">',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#526b89"/></marker></defs>',
        '<rect width="1600" height="950" fill="#fff"/>',
        '<text x="30" y="30" font-family="sans-serif" font-size="21" fill="#172033">场景形成的假设因果图：点击节点查看方程、父变量与实现范围</text>']
    for node in spec['nodes']:
        bx,by=node['xy']
        for parent in node['parents']:
            ax,ay=byid[parent]['xy'];dx=bx-ax;dy=by-ay
            if abs(dx)>abs(dy):
                sx=ax+(width if dx>0 else 0);sy=ay+height/2
                tx=bx+(0 if dx>0 else width);ty=by+height/2
                curve=f'M{sx},{sy} C{(sx+tx)/2},{sy} {(sx+tx)/2},{ty} {tx},{ty}'
            else:
                sx=ax+width/2;sy=ay+(height if dy>0 else 0)
                tx=bx+width/2;ty=by+(0 if dy>0 else height)
                curve=f'M{sx},{sy} C{sx},{(sy+ty)/2} {tx},{(sy+ty)/2} {tx},{ty}'
            parts.append(f'<path data-source="{parent}" data-target="{node["id"]}" d="{curve}" fill="none" stroke="#526b89" stroke-width="1.6" opacity=".35" marker-end="url(#arrow)"/>')
    for node in spec['nodes']:
        x,y=node['xy'];fill='#eaf2ff' if node['id'] not in ('O','G','A') else '#e7f4eb' if node['id']=='G' else '#fff1dd'
        parts.append(f'<g class="node" data-node="{node["id"]}" style="cursor:pointer"><title>{escape(node["meaning"])}</title>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="#6b83a2"/>'
            f'<text x="{x+14}" y="{y+35}" font-family="sans-serif" font-size="20" fill="#172033">{escape(node["label"])}</text>'
            f'<text x="{x+14}" y="{y+65}" font-family="sans-serif" font-size="13" fill="#596c83">parents: {escape(", ".join(node["parents"]) or "exogenous/root")}</text></g>')
    parts.append('<text x="30" y="915" font-family="sans-serif" font-size="17" fill="#915000">所有形成箭头均为研究假设；O 是观测；A 是世界的后果。图已验证无环，不表示因果关系已从数据识别。</text></svg>')
    return ''.join(parts)


def build(output):
    out=new_output(output);spec=specification();write(out/'graph_spec.json',spec)
    diagram=svg(spec);(out/'generation_scm.svg').write_text(diagram)
    lines=['flowchart LR']
    for n in spec['nodes']:
        lines.append('  '+n['id']+'["'+n['label']+'"]')
        lines.extend('  '+p+' --> '+n['id'] for p in n['parents'])
    (out/'generation_scm.mmd').write_text('\n'.join(lines)+'\n')
    factor='''flowchart LR
  H((活动 H)) --- prior[背景/活动先验]
  R((需求 R)) --- prior
  B((实现 B)) --- realize[替代与角色绑定因素]
  R --- realize
  W((完整世界 W)) --- realize
  W --- obs[观测/可见性因素]
  O[实际观测 O] --- obs
  K[相机 κ] --- obs
  W --- geom[尺寸/宿主/朝向/净空因素]
  L((房间结构 L)) --- geom
  W --- func[活动相容因素]
  R --- func
'''
    (out/'posterior_factor_graph.mmd').write_text(factor)
    html=['<!doctype html><meta charset="utf-8"><title>CauPano 因果推理图</title>',
        '<style>body{font:16px/1.65 sans-serif;margin:24px;color:#172033}svg{width:100%;min-width:1050px}.diagram{overflow:auto}pre{white-space:pre-wrap;background:#f5f7fa;padding:16px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}aside{padding:18px;background:#edf4ff;position:sticky;bottom:0}a{margin-right:14px}</style>',
        '<h1>用于局部观测场景补全的因果生成与逆向推断</h1>',
        '<p>研究模型设计。当前实验的 needs 分支尚未产生世界；这张图不表示该实验已成功或已经识别真实因果关系。</p>',
        '<div class="diagram">',diagram,'</div><aside id="detail">点击节点显示该变量的方程与当前实现范围。</aside>',
        '<p><a href="generation_scm.svg">独立 SVG</a><a href="generation_scm.mmd">Mermaid 生成图</a><a href="posterior_factor_graph.mmd">Mermaid 推断因子图</a><a href="graph_spec.json">完整变量定义</a></p>',
        '<h2>因果生成与观测反推</h2><p>生成方向：背景/结构 → 预期用途 → 需求 → 联合实现 → 库存、尺寸与位姿 → 可见性 → 观测。',
        '推断方向：将观测条件化，联合更新所有与证据有关的潜变量；不会把图中的因果箭头直接反转。</p>',
        '<pre>p(W_hidden,Z | O,κ) ∝ pθ(W_full,Z) · pη(O | W_full,κ)\n',
        'pθ(W,Z) ∝ p0θ(W,Z) · φgeom(W,L) · φfunc(W,R,B)\n',
        'Z=(C,L,H,R,B)；因素需要明确的可计算定义、独立数据和校准。当前尚未拟合 pη。</pre>',
        '<h2>这张客厅图的具体推理例子</h2><table><tr><th>观测或假设</th><th>应如何参与推断</th></tr>',
        '<tr><td>obs_000/001 两张沙发，obs_003 落地灯</td><td>支持会客、休闲、阅读等多个解释；无法仅据此确定电视存在。</td></tr>',
        '<tr><td>阅读需求</td><td>既有沙发和灯可能已满足座位与照明角色；可选择不增加桌椅，或提出另一种共享储存实现。</td></tr>',
        '<tr><td>影音用途假设</td><td>显示设备及其安装/观看方向作为联合实现；遥控器按具体设备依赖提出，不因有沙发就重复添加。</td></tr>',
        '<tr><td>杯子/书与桌面</td><td>尺寸、宿主接触与可达位置共同推断。极小接触投影不能作为合理支撑的充分证据。</td></tr>',
        '<tr><td>生活积累与闲置</td><td>由背景/储物来源表达，保留生活丰富性；不要求每件物品都服务当前活动。</td></tr></table>',
        '<h2>与当前 pilot 的差距</h2><p>当前代码只有 Z→R→B→E→S 的有限目录机制，R 在给定 Z 后确定、B 使用均匀参考采样，',
        '尺寸/布局仍是 VLM 提议加几何求解。完整图新增了明确的观测机制、角色条件尺寸/联合位姿、空间不确定性与实际功能后果；这些需要学习或独立验证。</p>',
        '<h2>本轮新增核验的一手资料</h2><p>',
        '<a href="https://arxiv.org/html/2510.10292v1">FactoredScenes</a>',
        '<a href="https://arxiv.org/html/2604.03696v1">FunFact</a>',
        '<a href="https://arxiv.org/html/2605.30819v1">Function2Scene</a>',
        '<a href="https://proceedings.neurips.cc/paper/2021/hash/4fc66104f8ada6257fa55f29a2a567c7-Abstract.html">3DP3</a>',
        '<a href="https://arxiv.org/abs/2510.14049">CausalVerse</a></p>']
    data=json.dumps(spec,ensure_ascii=False).replace('<','\\u003c')
    html.append('<script>const spec='+data+''';document.querySelectorAll('[data-node]').forEach(g=>g.addEventListener('click',()=>{
const n=spec.nodes.find(n=>n.id===g.dataset.node);const panel=document.getElementById('detail');panel.replaceChildren();
for(const text of [n.label,n.equation,n.meaning,'当前实现：'+n.implementation_status]){const p=document.createElement('p');p.textContent=text;panel.appendChild(p);}
document.querySelectorAll('[data-source]').forEach(e=>{const active=e.dataset.source===n.id||e.dataset.target===n.id;e.style.opacity=active?'1':'.10';e.style.strokeWidth=active?'3':'1';});
}));</script>''')
    (out/'index.html').write_text(''.join(html));return out/'index.html'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(build(args.output))


if __name__=='__main__':main()
