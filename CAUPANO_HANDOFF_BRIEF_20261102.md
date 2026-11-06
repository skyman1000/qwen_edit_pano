# CauPano 简要交接：已恢复原 v13

更新：2026-11-06。工作区 `/data-nfs/gpu1-2/u13529658780/DIT360`，主项目 `qwen_edit_pano/`。

**本文、其它交接文档及部分代码由 AI 生成，不一定正确。接手者必须独立判断，核对源码、配置、原始回复、日志与可视化。不要把历史推荐或 CPU 测试通过当作实际补全质量已验证。**

## 目标

早期工作为 DiT360 / Qwen-Image-Edit-2511 全景 LoRA 和 World Encoder / Adapter。当前任务是**免训练因果场景补全**：从一张局部 RGB，保留可见物体，生成与观测相容、语义和空间粗略合理的周围 360° 世界状态，`G_full = G_obs ∪ ΔG`。隐藏世界不唯一，不要求恢复唯一真值。

功能机制必须真实改变隐藏对象生成概率。不要靠长解释冒充因果推理，不设最少数量、必出类别或后方配额；不要为了修改而修改。当前状态到 RGB 全景的下游尚未接通，ERP 可视化是状态框投影。

## 当前恢复的模型与实现

用户要求恢复本次对话交接前的 `organized_functional_program_scm_v13`。原核心仍在，12 个关键文件与 10 月 28 日实际记录哈希一致；118 个当前 v13 基线绑定全部匹配。后续 joint、整室规划、simple、scope、stable 等分支保留作历史实验，**不作为当前默认方法**。

```text
局部 RGB + HFOV
→ Qwen 二维实例 → WildDet3D 三维框 → 冻结 O+
→ 完整功能装配候选域 D
→ 功能组织 H → 各角色额外装配存在 A_r
→ 完整程序与相机方位联合变体 V_r
→ 对象 E / 方位 R / 宿主安装 S
→ 固定尺寸、朝向与位置、原数值求解
→ G_full 及平面图、投影、交互 3D、2:1 ERP
```

离散结构：`q(H|O+,D) × ∏r p(A_r|H,O+,D)p(V_r|A_r,H,O+,D)`。
H 为混合用途或某功能强调，不推断住户具体生活史。给定 H 后角色条件独立是近似；完整程序可包含多个共存组和用品。S 是声明的宿主安装依赖。

大模型提出候选、提供机制短选择、预测尺寸与初始布局。程序负责计数平滑、CPT、均匀噪声及逆 CDF 采样、实例展开、坐标变换和几何求解。默认 K=4 次短选择，存在率 `(n_present+0.5)/(K_valid+1)`，如 3/4 票为 0.7。K=4 不是必须输出四个世界。概率实际作用于库存，但未校准、未识别真实因果，也尚未证明因果收益。

核心：`world_pipeline/completion/assembly_organized.py`、`assembly_compositional.py`、`world_scm.py`、`assembly_multiscene.py`。几何：`assembly_organized_geometry.py`、`assembly_coordinate_transport.py`、`assembly_spatial.py`。

## 尝试与关键教训

- 曾迭代功能需求、装配激活、完整功能程序、方位和宿主依赖，形成 v13；也修复过重复识别导致空观测假成功、数值字段解析、API 限流和碰撞搜索预算问题。
- 物体变少或后方不足不能直接归因于因果图：候选域缩水、漏检、提案接口拒绝和几何失败需分开分析。更多物体或更高导出率也不等于更合理。
- 交接前较丰富的 Qwen 世界使用 **Qwen＋API 共享候选**。纯本地 Qwen 候选较少，不能把差异全部归因于模型结构。后续整份程序选一、强制区域规划等替代流程曾在实际运行退化，当前不继续叠加这些流程。
- 当前五图冻结观测为 `outputs/local_observation_fixed_002`，16/8/8/10/3，共 45 个实例；是预测，厨房仍可能漏掉主家具。恢复模型不应恢复旧的空厨房观测。
- 局部观测实验入口 `scripts/local_observation.sh`，只运行 Qwen 二维＋WildDet3D，不新增隐藏状态。用户已停止“直接 RGB→结构化完整状态”基线，不继续补跑。未来图像级无因果对照尚未实现。

## 参考与下一步

仅借鉴思想，并非完整复现或因果性证明：

- [Human-Centric Indoor Scene Synthesis Using Stochastic Grammar](https://arxiv.org/abs/1808.08473)：功能装配、支撑与可供性。
- [Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image](https://arxiv.org/abs/1808.02201)：功能、几何和观测联合约束。
- [SD-SCM / Language Models as Causal Effect Generators](https://aclanthology.org/2025.emnlp-main.107/)：规定结构、语言模型机制和模型内干预；不等于现实因果识别。

当前唯一推荐本地入口：`scripts/world_completion.sh`，固定原 v13 一轮提案，复用冻结观测，重新估计候选、概率与布局。完整运行命令、恢复范围、备份与验证见 [RESTORE_ORIGINAL_V13_20261106.md](RESTORE_ORIGINAL_V13_20261106.md)。真实 GPU 任务由用户执行。

11 月 6 日实际五图恢复核查见 [结果与根因](reports/original_v13_review_20261106_001/README.md)：库存与旧纯本地 v13 一致，候选范围偏窄、辅助关系拒绝、类别映射及厨房观测漏检需要分开处理。已增加显式可选 `CAUPANO_PROPOSAL_INTERFACE=closed_optional_v1`，见 [执行说明](V13_CATEGORY_OPTIONAL_RELATIONS_20261106.md)；仅调整类别/关系候选接口，原因果机制和几何保持不变。默认仍是原 v13。新接口通过 8 项 CPU 测试，真实 GPU 质量尚未验证；历史两条未知类别观测原样保留，新增类别必须合法。

双后端历史入口 `scripts/organized_multiscene.sh` 仍在，API 为硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`，需环境变量 `SILICONFLOW_API_KEY`，不要打印 Key。纯本地入口不需要 Key。后续先固定方法核查候选与实际世界，不再同时更换因果结构、提案和布局。
