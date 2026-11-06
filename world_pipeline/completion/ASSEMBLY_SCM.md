# 结果诊断与条件结构场景机制

当前模型、诊断与执行指令统一见 [WORLD_COMPLETION_CURRENT.md](WORLD_COMPLETION_CURRENT.md)。下文保留历史分析；命令入口保持 `scripts/assembly_compare.sh`，旧结果不覆盖。

日期：2026-10-24。任务保持为局部 RGB → 已观测状态 → 合理的视野外对象状态 → 完整 G_full。本阶段只生成状态与可视化，未接 Adapter、未生成 RGB 全景、未训练模型。

## 2026-10-24 接口错误修复与物体布局展示

核查的两个目录为 `assembly_rgb_compare_20261024_155034_3023065`（独立观测）和 `assembly_rgb_compare_20261024_160207_3073962`（固定观测）。四个分支都在内容机制阶段失败，尚无新增物体的已放置 G_full；不能把本次失败当成“补全物体为零”的成功结果。

- 内容提示遗漏了关系格式和允许类型，而校验器要求固定枚举。Qwen 的 `beside/next_to`、API 的 `next_to/on` 因此触发同一个笼统错误。原纠错提示没有提供具体行与合法值，Qwen 原答案和纠错答案完全相同。
- 继续校验完整 Qwen 回答，还发现 g0 的某内容变体把书和杯子绑定到 g1 的落地灯。组内布局与宿主依赖不能执行这个跨组安装，所以不能只接受关系别名就宣称原回答已经修好。提示现明确列出每组的宿主 ID；校验按内容变体收集错误并一次报告具体路径、物体、宿主和合法选项，不把第一条错误掩盖的后续失败留到下一次运行。
- 独立观测 API 分支还把家具记录再次放入内容清单，错误只说身份冲突，没有指出具体 ID。内容清单现明确为 **新增内容的 delta**；现有家具只能作为引用，不重复定义。拒绝信息列明冲突 ID 与记录路径，不静默删掉重复记录。
- 对象示例原先默认 `affordances=["top"]`，容易被机械复制。真实 Qwen 回答把落地灯、相框都标为 top，并把书、杯子、植物放在那里。示例改为 `[]`，说明 affordance 是供其他物品使用的真实支撑面，区别于几何上必有的顶端。具体类别仍开放；没有新增物体数量下限或强制类别配额。该说明不能保证模型以后所有语义正确，仍需看实际输出。

接口版本为 `explicit_relations_and_content_delta_v2`；SCM 结构及采样法则未改变。两个模型共用修正后的提示模板与校验器。

`beside/next_to/adjacent_to` 只绑定为不指定左右方向的 `near`，保存原词、端点与转换动作。`on/on_top_of/supported_by` 只在与源对象已有的类型化 attachment 完全一致时作为重复支撑声明绑定；保存该声明，不生成额外的中心高度约束。否则要求模型纠错。未知 ID、自指、未知关系不做猜测、重绑定或静默删除。转换记录保存在家具/内容机制的 `relation_bindings` 中，也包含在 `causal_model.json`。

对比页现在直接嵌入离线 Plotly 三维框、XZ 俯视布局与 2:1 ERP 状态方向图。可旋转、缩放、悬停查看物体；数字、提示参数和机制放在折叠区。单分支入口也采用同一展示方式。没有布局的分支明确显示停止阶段，不拿语义提议当成已放置物体。重新制图创建版本化新页面，不覆盖之前的报告。

重跑现有两组实验（只复用已完成观测，重新调用 API / Qwen 生成全部语义机制与布局）：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
# 若当前 shell 尚未设置 API Key：
read -rsp 'SiliconFlow API Key: ' SILICONFLOW_API_KEY
export SILICONFLOW_API_KEY
bash qwen_edit_pano/scripts/assembly_compare.sh retry2 GPU3 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_155034_3023065 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_160207_3073962
```

`retry2` 分别复制原来两组实验的两端 2D/3D 观测，校验文件 SHA，保持各组原本的观测来源、模型名称和种子。不重新执行 Observer/WildDet3D，不复用旧语义机制、布局或失败答案。新目录的 `observation_reuse` 保存来源和文件哈希；所有新请求使用当前代码。最终打印新实验的物体布局对比页路径。只重跑一组可用 `retry GPU3 SOURCE_ROOT`。从 RGB 全部重做仍可用 `run2 GPU3`。

CPU 回归验证：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh test
```

测试包括关系别名、支撑声明一致性、未知 ID 拒绝、可操作的纠错信息、观测复用及原文件保留、三维框直接嵌入两组对比页、失败分支不伪造布局，以及原有宿主依赖、几何和完整导出。测试不调用 API/GPU，不构成真实语义质量或因果收益验证。

## 实际结果与历史核查

本次检查的真实实验为 `outputs/fresh_rgb_compare_20261024_132221_2406817`。数据提取见 [`actual_result_analysis.json`](../../../research/audits/assembly_model_2026-10-24/actual_result_analysis.json)。

| 实验 | 新增对象 | 判断依据 |
|---|---:|---|
| 最新 API | 13、14、13 | 提议数与导出数一致，无布局导出失败 |
| 最新 Qwen | 5、5 | 两套语义清单各 5 件，全部导出；没有对象被几何后端删掉 |
| 用户指出的 `generative_world_qwen8b_layout_run007_compare002_recovered` | 13、13、8 | 复用 DeepSeek run007 的语义机制，Qwen 生成布局；恢复阶段没有重新推理 |
| 找到的本地 Qwen `scm_world_val_living_run004` | 11、7、9 | 同一局部图；继承 run003 的本地 Qwen 计数机制，外部采样产生实例 |

上一轮只找到另一张图的 4 件 Qwen 记录，历史检索不完整，现已纠正。确实存在同图、本地 Qwen 路线生成更多对象的实验。但旧 11 件的首个世界中，一盏灯与书架中心完全重合，两株植物中心也完全重合。它证明生成数量可以更多，不能证明语义/几何更合理；旧实验还使用不同 G_obs、计数分解和采样配置，不能直接认定模型能力退化。

最新结果的主要问题有四个。

1. **清单覆盖发生在概率排序之前。** `Z→H` 中 H 是一份已经生成完的清单的确定性展开；之后再修改 CPT 只能给现有清单重排序，不能补回没有提出的宿主、物品和关系。当前 Qwen 清单从源头就是 5 件，调整 E 似然不会让它变成 13 件。
2. **描述与实例化脱节。** Qwen 的餐饮/阅读方案有两个功能区，但只列桌、两椅、书架、窗边座位；对应 `relations=[]`。另一方案仅两条左右关系，且将书架标为 wall。缺少具体宿主与日常内容的生成机制。
3. **坐标提议与任务不匹配。** Qwen 给餐桌椅的相对偏移中，椅子相对桌子 z 增加 3 米；把相对偏移当成布局说明或近似绝对位置。旧求解器随后为出视野和消除重叠逐对象移动，两个方案的最大 XZ 调整分别约 5.16 米、6.10 米。无 AABB 重叠不能保证餐桌椅仍属于一个可用区域。
4. **支撑机制过度简化。** 旧代码把对象支撑统一解释为“宿主顶面”。这会把挂钩上的袋子放到挂钩上方、把书架层板上的书放到柜顶。API 的对象多也不免疫：其所有方案仍为 `layout_needs_review`，包含未验证的壁挂/悬挂安装。

两端最新隐藏后验更新 TV 约 .029，不是因果效应或质量分数。原图参与了域提议，再让同一个模型为这张图填写主观似然后自我排序，不能当作校准的图像后验。新版去掉这个步骤。

## 方法边界与研究依据

单张局部图不能识别唯一真实因果图。这里提供的是可执行、可审计、针对任务的**假设条件 SCM**，而不是声称已经找到了真实的房间生成原因。

- [SD-SCM: Language Models as Causal Effect Generators](https://aclanthology.org/2025.emnlp-main.107/) 及其[官方代码](https://github.com/lbynum/sequence-driven-scms)支持指定结构、父变量条件机制、外部采样与干预。官方使用 LM 续写分数。本版本为保持 API/Qwen 接口一致，使用 LM 提议的有限条件规则与明确的均匀参考法则，不冒充其 token-score 实现，也不声称该论文验证了室内补全。
- [DoWhy 官方 GCM 说明](https://www.pywhy.org/dowhy/main/user_guide/gcm_based_inference/introduction.html)区分图与生成机制。新版不仅画图；实际生成表、外生噪声、宿主依赖、机制替换和下游更新均由程序执行。没有安装 DoWhy。
- [Human-Centric Indoor Scene Synthesis Using Stochastic Grammar](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Human-Centric_Indoor_Scene_CVPR_2018_paper.pdf)及其[官方实现](https://github.com/SiyuanQi-zz/human-centric-scene-synthesis)将家具、受支撑物品、功能组合与空间关系分开建模，适合借鉴场景表示。原工作有学习与 MRF/MCMC，不是本项目已实现的免训练因果框架。
- [Holodeck 官方实现](https://github.com/allenai/Holodeck)提供语义关系与布局求解分工；[DiffuScene 官方实现](https://github.com/tangjiapeng/DiffuScene)支持部分场景补全、联合对象建模。前者不等于因果识别，后者是学习式模型。本轮借鉴组织方式，没有替换 WildDet3D、下载这些模型或训练。
- [CausalStruct](https://arxiv.org/html/2509.15249v1)的关系顺序和布局优化可参考，但不能将任意关系边或几何通过当作真实因果效应；其 3DGS/SDS 流程也不是本项目的 AABB 状态任务。

## 新的模型：条件结构场景语法

固定上下文 `B=(I,G_obs,K)`，图像首先用于提出少量用途、活动、生活方式假设及条件规则。**候选域的图像条件提议是近似溯因，不是已识别的视觉贝叶斯后验。**

```text
B ──近似逆向提议──→ 条件规则域

Z：活动/生活方式 ──→ F：完整联合家具变体
                       ├──→ T_g：该变体下、宿主条件化的内容组合
                       └──→ P_i：具体家具组件是否激活
                             ↑
             T_g + P_host ───┘ → P_child

已激活对象 + 类型化安装 + 相对尺寸/姿态
             → 按家具组的几何编译 → 视锥检查 → G_full
```

所有随机选择使用保存的外生噪声与逆 CDF。有限规则域中的法则为：

```text
Z   = inverse_CDF(reference_Z, U_Z)
F   = inverse_CDF(reference_F_given_Z, U_F)
T_g = off                                      if selected F is not its parent state
      inverse_CDF(reference_bundle_given_F,U_g) otherwise
P_i = selected_by(F,T_g) AND all(P_host)
W   = fixed_G_obs UNION geometry(active installed components)
```

`reference` 当前显式采用均匀分布，表示一种透明的探索采样政策，而非对真实房间的概率估计。它不优待物体更多的方案，不要求每类独立计数，不自动给每个有效组增加一个高概率“不添加”。激活家具下的空内容组合仅在模型明确提出时存在；未选择的家具变体，其专属内容机制确定性关闭。

结构的实际含义：

- **F 联合生成核心家具及变体。** 不把各类数量独立采样后随便摆放。可见对象的角色已通过固定 G_obs 提供，不重复新增。
- **T 根据具体宿主与 affordance 生成内容。** 台灯放在哪张桌、书在哪个书架、袋子在哪个挂钩是具体实例关系；也允许地面植物等独立物体。每个空间组共享上层的活动/生活方式与家具状态，组内内容是联合组合。
- **P 有真实的程序父变量。** 改变一个宿主的激活状态，程序重新执行其依赖组件的方程；独立地面植物不会仅因“台灯和桌子常共现”而被一起删掉。保存 `host_interventions.json` 可核查这些实际执行结果，期间不调用模型。
- **安装/几何机制也区分类型。** `top` 用宿主顶面；`seat/shelf` 用声明的内部支撑层；`hang` 从锚点向下；`wall/ceiling` 是未验证安装假设；靠墙的落地书架用 `floor`，不是 wall。

P 表示“该生成世界中处于所声明安装组合的组件是否激活”。`do(P_host=0)` 是生成组合干预，不是物理世界中搬走桌子后模拟台灯下落，也不是“没有桌子的真实房间绝不可能有灯”。任意真实物体级反事实仍未识别。

模型的局限必须保留：候选覆盖受冻结 LM 能力限制；候选及其顺序、参考法则、组件依赖都是建模假设。未验证因果收益、未学习真实概率，几何也不是形状物理模拟。可以执行机制不等于已经证明因果模型真实。

## 几何改动为何必要

位置提议改为“整个空间组的方位角与距离＋组内相对布局”。方位角以原始相机为准：0 前方、±90 侧方、180 后方。组中心从原输入视锥外的方向提出，程序按完整组的框尺寸计算必要的最小距离；不会先把全部家具摆进照片，再逐个大幅挪走。

碰撞代理优化的自由变量仅为**整组 XZ 平移**，组内桌椅、宿主与物品的相对位置保持不变。组内布局不合理会如实显示在诊断中，求解器不能通过把椅子搬离餐桌来隐藏问题。左右/前后/靠近关系作为 XZ 软目标，不设凭空的“近距离必须小于多少米”。

书放进书架、垫子放在座面时，父子 AABB 有意相交；父框并不代表实心柜体。这类宿主包络交叠单独报告为形状占用未验证，不被当作普通家具互穿而推出柜外。接触与安装假设、内部组合冲突、所有相对调整都保留。

原 G_obs 的类别、中心和完整 AABB 尺寸保持不变。新增对象类别开放、未映射对象保留；任何几何失败保留整份语义样本及原始回答，不删物体来充当成功导出。这里仍只输出米制原相机坐标 `x右/y图像上/z前` 的类别、中心和完整尺寸。

## 两次有区分目的的对照

1. **端到端对照：** 同一 RGB → API/Qwen 各自 2D → 同一 WildDet3D → 各自 G_obs → 同一结构机制模板 → 各自 G_full。
2. **固定观测对照：** 复制第一组中同一份 G_obs、可见语义、RGB、相机参数给两端，再独立生成条件规则与布局。默认使用第一组 Qwen 的观测，并记录来源和 SHA；可通过 `CAUPANO_SHARED_OBSERVER=api` 选择 API 观测。

第一组比较端到端系统；第二组减少 Observer 差异对补全比较的混淆。第二组不复用第一组的家具清单、内容机制或布局。两组共四次补全，前端只在第一组生成。它们不是统计意义上的两次同分布重复，也不是普通补全与因果补全的消融；单凭它们不能证明因果收益。

默认固定 Qwen 观测时，Qwen 在两组中的输入和本地种子相同，结果可能重复；它是固定参考。API 则从自己的观测换成该参考观测。若希望两组仅做随机重复，这是另一种实验设计，不应与这里的观测控制混称。

两端共用提示词模板、JSON 校验、条件结构方程、参考采样政策、几何求解及公开参数：温度 .4、top_p .9、top_k 50、max_tokens 8192，无重复惩罚，API thinking 关闭。初次本地请求 seed=3407，接口纠错请求 3408。API 无已保证的 seed，内部视觉编码和解码不能声称一致。第一组观测不同，后续具体提示词自然不同；第二组输入观测相同，但各自规则生成后又会不同。共用框架不等于实例化节点和数值模型完全相同。

默认有限容量：1..2 个语义假设，每个假设 1..2 个家具变体，每变体至多 4 个空间组/12 件核心家具，以及整体至多 12 件内容，共最多 24 件新增。每组内容提供 1..2 个组合，可明确为空。程序保存 3 个不同结构样本，域不足时保留实际数量；没有新增数量下限。

## 执行指令

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/assembly_compare.sh preflight
read -rsp 'SiliconFlow API Key: ' SILICONFLOW_API_KEY
export SILICONFLOW_API_KEY
export SILICONFLOW_MODEL='deepseek-ai/DeepSeek-V4.1-Flash'
bash qwen_edit_pano/scripts/assembly_compare.sh run2 GPU1
```

把 GPU1 换成可用节点。默认仍为 `X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e/local.png`，HFOV=90。程序为每次对照创建全新 `outputs/assembly_rgb_compare_时间_PID/`，不覆盖旧结果。最终打印 `two_experiments.html` 的实际路径，包含两组对照入口。

仅运行第一组：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh run GPU1
```

从现有、完整的第一组观测单独执行固定观测对照（SOURCE_ROOT 填实际目录）：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh controlled GPU1 /absolute/path/to/SOURCE_ROOT
```

只读重新制图，不调用模型：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh review /absolute/path/to/FIRST_ROOT /absolute/path/to/SECOND_ROOT
```

生成的 `review_*/index.html` 比较 2D、WildDet 投影、家具/内容数量、3D 交互框、XZ、2:1 状态方向图。每个 `world_*` 保存 `G_full.json`、`G_out.json`、`semantic_world.json`、`host_interventions.json`、`object_relations.json`、`numeric_review.json`。`causal_model.json` 和 `mechanism_graph.json` 是实际运行的生成表与父变量，不是事后画的装饰图。

## 验证与接下来的判断标准

本轮仅运行 CPU 合成回答测试和本地依赖只读检查，未调用真实 API、加载 GPU 模型、提交 GPU 任务或安装依赖。测试检查宿主干预只影响相应后代、家具变体更新内容、挂袋在锚点下方、书架内层支撑、刚性组内关系、视锥外完整框、失败样本保留、固定观测 SHA 相同和全链路可视化。

真实运行后分别看：新物体的独立实例理由、家具与内容是否对应、是否有机械重复、输入可见区域是否保持、空间组是否紧凑合理、支撑/悬挂是否符合具体变体。数量只作为覆盖统计，不能单独作为质量或因果收益。若固定观测下 Qwen 仍明显弱于 API，应保留“冻结 8B 模型在该语义/空间接口上的能力限制”这个解释，不继续无依据地增加因果节点。
