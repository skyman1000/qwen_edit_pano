# CauPano：以可干预 World State 为主线的实施方案

日期：2026-10-12。本文是后续实施设计，不表示新模块已经实现或效果已验证。本轮不修改正在使用的 Oracle 模型、训练配置或已有实验结果，不执行 GPU 作业。

最新执行范围更新：用户已重新训练带位置编码的 Adapter，目前只实施 Observer 与免训练 Completion。下文“问题 A”的 Adapter 诊断及 ERP 合流待新训练完成后再安排。上游代码与手动执行步骤已实现，见 [QWEN_UPSTREAM.md](completion/QWEN_UPSTREAM.md)；CPU 检查不代表真实 Qwen 推理效果已验证。

## 执行优先级补充：下一轮只回答两个问题

本节明确下一轮执行顺序，优先于下文中的长期展开建议。依据 `INDOOR_RESULT_REVIEW_2026-10-11.md`，已保存的 step500/1000 室内对照尚未显示正确 GT 相对 shuffled 的稳定空间优势。因此不能把下游控制当成已解决，也不应先铺开所有世界候选的 ERP 和因果编辑。

**问题 A：当前 World Branch 是否读取对象位置，并在 ERP 中执行位置变化？**

- 固定现有诊断样本和 checkpoint。实现一次固定 noisy latent/timestep/noise 的 no-world、correct、cross-building shuffled、仅 center 在对象间置换的预测/损失对照；train 与 val 分开汇总。三组固定噪声等级用于避免结论只依赖一个时间步。位置置换属于敏感性诊断，不代表合理世界，也不证明因果控制。
- 从固定样本中选三例满足编辑条件的场景；选择依据 GT 几何与原始输入，不看生成效果。每例只移动一个未观察到的大物体，保持 category/size/其余对象/Local RGB/prompt/生成噪声不变，输出原世界与编辑世界配对图。若没有合理空位则记录不适用，不强行构造穿墙位置。
- 损失差与像素差仅是使用条件的证据；最终必须检查物体是否向要求方向移动、无关区域是否保持。当前 small sample 只用于决定下一版本，不能宣称整体提升。
- 一轮后形成明确分支：位置控制有证据则用当前分支接预测世界；没有证据则将新版本工作集中到空间注入与训练条件/预算，不继续叠加 Completion 去解释图像结果。若需要新训练，另列实验设计；本计划不启动。

**问题 B：不训练 Completion 时，Qwen 能否在固定观测下提出有用的隐藏世界？**

- 使用原 CauPano 验证划分，预先固定每建筑三张、当前四建筑共十二张；按清单固定抽样，保留失败例，不按模型效果选图。可复用匹配样本的 Observer/caption 缓存；M3DLayout 原生房间子集只作为旧诊断记录。
- 同一图片分别使用 GT G_obs 与 predicted G_obs，各采样三个 hidden 候选，共 72 个布局。只预测 hidden；由程序原样拼接 G_obs。GT 仅作为独立诊断条件，实际预测路线不读取它。
- 初版使用当前 Adapter 的 category/center/Local AABB 作为隐藏对象数值目标，不强制模型增加不被 Adapter 使用的精确 rotation。已有 Observer OBB 继续用原角点转换，不改变几何定义。
- 保存原始候选、对象级布局可视化及少量明确指标。第一轮不加入自动布局修复器，以便测出 proposer 本身的能力；明确几何问题单独标记，不能为了好看静默贴地/缩放/删物体。
- 观测保持是程序保证的契约，不作为模型性能成绩。重点比较隐藏候选合理性、与可见证据的冲突、GT 参考匹配及候选差异。GT 与预测条件都差则优化 Completion 提议；GT 好但预测差才优先改 Observer。

**合流与后续：**A 有基本空间控制证据后，取 B 中固定的小批场景运行 no-world、GT full、predicted full 的同条件 ERP 对照；Observed-only 可另列缺失条件诊断，不能冒充训练匹配的消融。选择候选预先固定 seed，不用 GT 挑最佳结果。若 A 未通过，B 仍可完成布局级评估，但端到端结果只能标为接口试验。自然语言因果规划延后；本轮 A 的直接数值干预先检验其最关键的下游前提。

**下一批代码交付限定为：**固定实验清单；Qwen hidden-only 推理与既有 graph 合并；固定条件的 World 诊断和单对象移动；复用当前加载器的 predicted-world 推理入口。新增入口均待实现，不把设计名称写成可立即执行的命令。沿用 qwen3vl 环境运行上游，qwen360 运行已有全景模型，通过 JSON 串联；无需共用一个 Python 进程。

## 1. 核心决定

目标保持为 Local RGB + optional text → World State → Qwen-Pano → ERP，并在同一世界上执行显式干预。

在不训练 Completion 的约束下，下一版优先实现同一个 Qwen3-VL 的两种调用角色：Observer 提取可见实例；Hidden Proposer 在锁定 G_obs 的条件下提出隐藏对象。M3DLayout 保留为替换后端和已有对照，不是强制串联环节。两阶段方案的优势是输入约束明确、误差可分解，并非已经证明比其他方案效果好。

```text
Local RGB ──→ Qwen Observer ──→ G_obs
    │                              │
    │              scene text ─────┤
    │                              ↓
    │                 Qwen Hidden Proposer
    │                  + 简单几何约束执行
    │                              ↓
    │                         G_full[k]
    │                         /       \
    │                  普通生成       显式干预
    │                     │       Graph + Planner
    │                     │            ↓
    │                     │          G_cf[k]
    │                     └─────┬──────┘
    │                           ↓
    │                  当前字段转换 / tensorize
    │                           ↓
    │                   World Encoder + Adapter
    │                           ↓
    └──────────────────────→ Qwen-Pano ←── optional text
                                ↓
                               ERP
```

G_full 在当前阶段表示目标全景相关的完整对象状态假设，不等于恢复完整房屋、墙面材质、照明及所有真实物体。

## 2. 研究依据与现有证据

- [Qwen3-VL 技术报告](https://arxiv.org/html/2511.21631v2) 描述了单目 3D grounding 和 9DoF 框学习；它不证明当前 8B 模型能精确恢复镜头背后的米制布局。可见实例的完整 3D 框本身也含推断成分，不能把预测框叫作直接测量。
- [LayoutGPT](https://layoutgpt.github.io/) 展示了 LLM 生成 3D 布局与 partial scene completion 的可行性；其条件与数据不等于 CauPano。
- [Holodeck](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Holodeck_Language_Guided_Generation_of_3D_Embodied_AI_Environments_CVPR_2024_paper.html) 使用语言模型提出空间约束，再进行布局优化。这支持“模型提议、程序执行几何”的分工，不能据此保证 Qwen3-VL-8B 的结果。
- [M3DLayout](https://arxiv.org/abs/2509.23728) 提供文本条件布局生成数据及 AR/diffusion 基线。[官方 AR 配置](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml) 的文本条件应按其 checkpoint 保留，不能因为用户输入结构化对象就切换模型架构。
- 本地 run002 的 33 次 GT-visible 续写中，18 次首步 END；共同未见目标的空间匹配为 3/195。该指标不是多解合理性的完整评价，但已不足以支持把这份 checkpoint 作为可靠主链。详细证据见 `completion/ANALYSIS_VISIBLE_PROBE_RUN002.md`。
- 原生空间前缀与任意可见集合不同；仅重排可见对象不会补上缺失的前部对象。纯文本生成的新布局也不能未经配准直接拼接原 G_obs。
- [ATISS](https://research.nvidia.com/labs/toronto-ai/ATISS/) 使用集合建模并支持部分场景补全，说明 AR 并非天然不适用；但它的房间条件、类别和训练域同样需要适配。此时不增加新的模型迁移任务。

## 3. Observer 与 Hidden Proposer 的最小接口

### Observer

输入 Local RGB；已知图像内参/FOV 可作为相机元数据使用，不输入 GT 深度、隐藏标签或真实房间布局。

输出：可见对象 ID、类别、相机坐标下中心、完整 OBB 尺寸及旋转；2D 框用于检查投影；简短可见场景描述用于语义辅助。对象确实可见与其 3D 尺寸精确是两件事。

沿用已核对的 Qwen camera → CauPano Local 转换，不让每次 prompt 自定义坐标、欧拉角顺序或半尺寸含义。单目尺度必须作为估计评测，不能用验证 GT 拟合尺度后称作实际推理结果。

### Hidden Proposer

输入：原图、锁定的 G_obs、可见场景描述、可选场景文本。输出仅新增隐藏对象及必要的布局假设，不返回一份可任意改写的 observed 列表。

初版每图固定生成 3 个候选，分别组成 G_full[k]。所有候选的 observed 部分逐字段保持一致。描述中的视觉事实与常识推断分开；用户要求新增某物体也不变成视觉证据。

同一模型可以一次输出 observed/hidden 两个列表，作为小规模对照；主实现采用两次调用以固定观测条件。这不是额外训练。

几何处理只作用于 hidden：去除与已知实例重复的提议，处理明显体积穿插，执行明确支撑/边界约束。接触、包含和桌面物体不能用通用“框相交即错误”删除。缺乏房间边界时，不虚构一个 GT 房间盒作硬裁剪。

隐藏不等于全部位于相机背后：视锥内被遮挡的实例也可能不可见。只根据可用视觉几何证据判断冲突，不按视锥一刀切。

当前 Local 是相机轴坐标，不是默认重力对齐坐标。需要落地/支撑约束时，使用显式估计的重力与地面假设，必要时在辅助坐标系执行，再变回训练 Local；不默认相机高度或把 image-up 当作重力。

## 4. 与已训练 Adapter 对齐

唯一神经输入接口继续是 `objects_local_aabb_masked_v1`：category_id、center_local_m、size_aabb_local_m、两个 validity 及 padding mask。规划层可以携带 ID、OBB、关系、observed/hypothesized 来源，但这些都不新增为 Encoder 特征。

转换步骤：

1. 类别按当前训练词表显式映射，不因为字符串像英文名就假定类别 ID 相同；无对应类按训练契约处理并统计覆盖。
2. 使用已核对的旋转约定构造 OBB 八角点。
3. 将角点转换到与训练完全相同的 Local 坐标。
4. center 为中心；size_aabb_local_m 为各轴 max(corners)-min(corners)，即完整边长。
5. validity 表示该数值可用，不是置信概率。padding 标识真实对象，不用来给隐藏对象加权。
6. 交给现有 tensorize 和 World Branch；对象容量读取 checkpoint/bundle，不静默截断。

输入 JSON 要满足现有 graph schema；ID/source 等元数据可以保留，但 tensorize 不读取它们作为特征。字段对齐只是接口成立，不保证预测误差分布与 GT 一致，也不保证 Adapter 已学会位置控制。

当前限制：关系边不直接影响模型；纯 yaw 编辑可能不改变 AABB；对称物体的旋转尤其如此。第一版优先移动、删除及明确尺寸变化，不声称能控制独立朝向、材质或同类物体外观身份。

## 5. 场景图与有限机制的因果规划

场景图的 near/left_of/on 是关系描述，不自动构成 SCM。根据一个静态世界调用 LLM，无法识别唯一真实因果机制。Room size → furniture layout 可作为设计模型，不是从单图识别出的自然因果规律。

构图：几何关系尽量计算；LLM 辅助提出功能、支撑和附着关系假设。执行：LLM 将自然语言变成对象 ID 与结构化动作，程序修改状态、按显式机制传播、重算几何关系。LLM 不重写完整世界。

首版动作：移动独立对象、删除独立对象；随后加入明确的刚性附着/带内容物移动规则。普通 near 关系不传播；床移动不默认带走床头柜；桌面接触也不自动证明物品一定跟随桌子。

一个可执行机制：对明确约定“随父对象一起搬动”的物体，保存相对变换 Δ_pc，令 T_child = T_parent Δ_pc。干预 do(T_parent = T_new) 后，用同一个 Δ_pc 重算 child；其他对象保持原值。这是声明的模型机制，不是物理仿真。

反事实始终从同一个 G_full[k] 派生 G_cf[k]，保留同一隐藏假设与机制参数。不能在移动后重新采样一个全世界，再把差异解释为干预结果。删除支撑物的后果需单独定义，初版不扩展到自动坠落仿真。

[Deep Structural Causal Models](https://arxiv.org/abs/2006.06485) 强调反事实中的外生变量推断。固定扩散 seed 有助于控制生成随机性，但不等于已经从真实图像完成 SCM 的 abduction。因此首版应称“显式机制下的结构化干预与反事实生成”，不宣称单图因果发现或真实物理反事实恢复。

## 6. 原始 Local RGB 与反事实的条件冲突

当前 Qwen-Pano 仍接收原始 Local RGB。若把可见椅子移走或删除，原图与 G_cf 会冲突，不能假定 World Adapter 自动覆盖视觉条件。

第一阶段选择原视图不可见、且预计不直接改变原视图证据的物体进行干预，同时固定 Local RGB、文本、采样设置和 seed。这能减少冲突，不保证阴影或反射等影响不存在。

后续可见物体干预再比较：仅 world 修改、仅显式编辑文本、world + 编辑文本。必要时研究同步修改 Local 条件或新的掩码条件接口；目前代码不能直接假定支持该功能。改变 Local 的实验属于新的条件设置，需单独报告。

## 7. 接下来实施顺序

### A. 继续独立 Oracle 评测

保持当前冻结基座、Panorama LoRA 及训练方案。用正确 GT、constant-trained、shuffled 判断状态信息是否有效；固定 World 做单对象位移/删除敏感性评测。输出变化不等于正确遵循干预。Observer/Completion 可以并行开发，端到端效果结论仍依赖下游控制有效。

### B. 实现免训练 Qwen completion 闭环

1. 固定现有验证建筑中的小清单；已有 11 图可复用，但它是诊断集且含 M3DLayout 官方训练场景，不能当独立泛化证据。正式全景实验使用原 CauPano 验证划分。
2. 对同一批原始图片保存可见状态、描述和局部框投影，先评估类别、位置、大小，而非继续统计 JSON 能否解析。
3. 先用 GT G_obs 调用 Qwen proposer，生成每图 3 个候选，隔离 Observer 误差。
4. 替换为 predicted G_obs，保持同样生成协议。最终推理不读 GT；GT 只用于上述诊断及评价。
5. 将候选接入当前 Adapter，比较 observed-only、Qwen full、已有 M3DLayout 方案。M3DLayout 的对象范围与类别缺失必须单列，不能把裁剪过的房间输入当作完整 G_obs。正式任务不删跨房间可见对象以迎合后端。
6. 同图生成配对 ERP；检查是否服从选定的 G_full，而不是只看生成图片好不好看。

优先完成 B 的 2–5，暂不引入新 Completion 权重或训练任务。单次 Qwen 全量生成与几何处理的消融在闭环跑通后补充，不先铺开大量实验组合。

### C. 并行实现最小干预器

直接对 GT G_full 做单对象移动/删除和一个明确附着传播例子。检查编译前后真正变化的 Encoder 字段，再生成 factual/counterfactual ERP。随后将完全相同的干预器接到 predicted G_full。

这条线不必等待 Completion 达到高精度；它能独立回答 World 表示及 Adapter 是否支持可控干预。

## 8. 判断方案优劣的标准

| 层面 | 主要判断 |
|---|---|
| Observer | 可见实例类别、匹配后的 3D 中心/尺寸、与输入图的投影一致性 |
| Completion | 观测保持、隐藏布局合理性、几何冲突、对单一 GT 的参考匹配、多候选差异 |
| ERP | 对选择世界的对象/位置服从、可见区域保持、隐藏区域质量、接缝质量 |
| Intervention | 目标动作成功、声明依赖的传播、无关对象保持、图像是否跟随字段变化 |

同一 G_obs 的候选必须都保持 observed。单一 GT 匹配低不必然代表不合理，但几何自洽也不代表恢复真实世界；两者分别报告。不要只用生成候选的同一个 Qwen 给自己打分。

少量人工盲评结合 GT 几何和图像可见证据即可起步；无需先搭大型评价系统。按建筑报告，不根据失败结果换验证集。

固定候选数与推理预算。报告所有候选或使用仅依赖输入证据的预先指定选择规则；用 GT 选最佳候选只能标为 oracle 上界。

## 9. 建议代码归属（待实现）

复用当前 `world_pipeline/completion/observe.py` 和已核对的几何桥接。新增职责建议为：

```text
world_pipeline/
  completion/
    qwen_hidden.py          # 锁定 G_obs，生成 hidden 候选
    assemble_world.py       # 合并候选，输出既有 graph schema
  causal/
    scene_graph.py         # 派生关系与明确机制
    interventions.py       # 动作执行与字段差异
  predicted_world_inference.py  # 复用冻结模型加载与 World Branch
```

以上名称是实施位置建议，不是已存在的执行入口。避免复制整套推理器或把新的长期主链继续塞进 M3DLayout probe 脚本。是否拆文件以实际代码复用为准。

最先需要完成的交付是：一张局部图 → 固定可见状态 → 三个完整世界候选 → 当前 Adapter 对应 ERP；同时一份 GT 世界 → 明确移动/删除 → 配对 ERP。两个闭环分别验证上游世界构建与下游干预控制。
