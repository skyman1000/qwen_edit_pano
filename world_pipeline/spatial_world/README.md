# Learned-ID World：输入3D框驱动的局部ERP残差

10月26日后续实现：新的语义＋几何拼接、可学习实例attention、A/B绑定监督128图小试已在独立`../grounded_world/`落地；CPU检查通过，GPU待用户执行。入口是`run_object_world_grounded.sh`，说明见`../grounded_world/README.md`。本目录v1训练与旧权重保持原样；不要用本目录的calibrate命令代替新实验。

日期：2026-10-25。新版本 `polished_single_face_id_region_spatial_world_v1`。保留原A分支/权重/结果；所有GPU运行由用户手动执行。

10月26日最新：8图生成及CPU分析也已完成，不能将其视为对象控制验收通过。GT电视区域100%可达，自身平均routing权重0.9653、其它对象0，仍未恢复可辨认电视；位置写回缺失不足以解释这一失败。椅子区域有明显混合。当前重点转向类别语义与视觉融合、旧强World路径与新弱local路径的耦合，而非只加位置Loss或继续延长32图训练。新设计和128图候选在`research/audits/spatial_generation_results_2026-10-26/`，新训练架构尚未实现，GPU未由助手执行。

## 为什么是这个改动

实际固定噪声诊断表明：A10000能改善去噪，但所选对象的类别/位置关联不稳定，20/40层对共享常量global token的注意力增加。这支持验证直接的对象→位置通路，但没有证明global是唯一原因，也没有证明任何空间结构一定成功。

新前向：

```text
现有ID＋几何Encoder ──→ 原World attention（保留） ──→ target hidden
          │                                           ↑
          └→ 对象投影＋局部query调制 → 门控残差 ──────────┘
输入中心/完整AABB ──→ ERP射线支持权重 ──────↑
```

局部分支用输入几何计算权重 `S(q,i)`，聚合对象特征，经图像query调制后写回对应target位置。global槽的空间权重恒为0，但原attention路径仍可使用global。

保留：单张face2 Local、对齐的polished GT、现有3D字段/ID编码、epoch015 Panorama LoRA scale0.5、20/40/59层、`flow+.5cube+.5yaw+.25region`。不新增attention CE、不改attention logits为球面bias、不引入新的文本网络。不是对象识别损失；生成语义仍需实测。

## 几何与梯度核对

- 输入相机原点，x右/y图像上/z前，米制中心和AABB完整边长；内部除2得到半边长，不改变外部字段。
- ERP经度`[-pi,pi]`、纬度从上到下`[pi/2,-pi/2]`，射线与现有`erp_rays`一致。
- 在未padding的packed token网格，每格4×4子射线与AABB求交，得到面积比例；用3×3的121可分离核平滑。横向周期、纵向边界复制。它是ERP网格上的近似支持，不是球面等面积积分或精确可见分割。
- 重叠权重除以`max(1, sum_i coverage)`，避免对象数量直接扩大残差。没有用粗框推断硬遮挡。
- 缺失/无效几何、相机位于框内的对象不进入新局部通路，仍保留原条件路径。相机内框若直接光栅化会覆盖全图，必须特殊处理。
- GT实例mask只进入现有region Loss和诊断读数，**不进入新前向模块或生成推理**。输入仅为现有tensorize的类别/几何/有效位。
- 几何投影是确定的非可微输入预处理；Encoder、局部value/query/out投影和gate正常接收梯度。当前不训练Completion，无需对输入3D框求梯度。
- 新gate初始化0；CPU真实小型Edit测试中，初始输出与原A严格一致。新gate打开后局部参数有梯度。checkpoint捕获tokens和routing，反向不依赖可变最新样本。
- 新增4,935,171参数，World共10,244,870参数；不复制或训练20B主干。

## 先执行两步smoke

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=00:20:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh smoke
```

smoke从A10000的World权重初始化。检查的是“新增gate为0时等于warm-start A”，而不是等于关闭World。需完成两步、生成`SMOKE_COMPLETE.json`，其中`zero_added_gate_matches_warm_start_exact`、`spatial_non_gate_gradient_seen`应为true。这验证接口/梯度，不证明生成质量。

v1已由用户完成GPU验证，空间pilot记录峰值已分配显存约41.418GiB；CPU activation offload与原A相同。新的mechanism入口尚未运行GPU。没有新下载，也不需要重建训练数据/缓存。

## 200步机制实验：先证明能学会

默认32张训练图，包括原条件诊断的4张训练图，另28张由固定种子选择。没有val参与优化。有效batch4，共800次样本使用，即25次平均曝光。新旧组相同数据顺序和逐全局microbatch随机种子。

空间组：

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh pilot
```

匹配控制组（另一个终端/节点，可并行）：

```bash
srun -p debug --nodelist=GPUY --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh pilot --mode control
```

控制组同样从A10000加载，继续训练原Encoder/Adapter，局部分支关闭且冻结；不是恢复旧B。两组均使用新AdamW、LR1e-4、warmup20、region0.25，不继承旧优化器。新结构是warm start；该新实验中断后才严格加载其optimizer/config续训。

每50步保存。重复相同命令自动选该stage/mode下最大完整checkpoint，在新run目录恢复；最多重做最后一次保存之后的步骤。已达到目标则直接退出。训练后不自动评估。

如果用两张卡共同训练一个组：在同一节点申请`--gres=gpu:2 --cpus-per-task=8 --mem=128G`，命令末尾加`--gpus 2`。有效batch仍为4。它与“两张卡各训练一个组”是两种不同安排。

结果目录：

```text
qwen_edit_pano/outputs/object_world_spatial_v1/
  smoke/spatial/run-.../
  pilot/spatial/run-.../
  pilot/control/run-.../
```

pilot是刻意小样本过拟合检查，不是最终泛化训练或验证成绩。日志step是新增步骤；source A10000另存`initial_step`，不混淆成原训练的相同步数。

## 训练后手动诊断与生成

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=02:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh diagnose
```

默认自动选择空间组最新full/pilot checkpoint，与原A10000对照，同样4train/6val、相同后验噪声/时间步、300条记录。复用已经完成的10图缓存。运行控制组诊断加`--mode control`；两组先各自相对同一A10000比较，再作等预算判断。

**本轮诊断协议有明确改进**：只重排输入AABB整框位于90度Local视锥外的对象；Local相交/未知对象保持不变。类别直方图仍保持，重排中心时也保证新框不进入Local。与旧“全部对象重排”的数字不是相同协议，不能直接混表。两个checkpoint在新诊断内部按同一协议重算。

诊断输出在`outputs/object_world_spatial_condition_diagnostic_v1/<run-mode-checkpoint>/`，包含`report.html`及JSON/图。新增局部读数保存在records的`spatial`/`routing`中；`ATTENTION.png`仍是原attention路径的读数，不是新增空间分支的准确率。

可选的固定客厅生成（6图：A10000参考、World-off、新正确GT、移动TV、TV改镜子、删除TV）：

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=02:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh probe
```

该客厅是反复开发的旧val图，不是新的最终test。GT图只在生成后读出用于review。删除token不是明确图像删除指令，不能用单个删除失败否定全部条件控制。

## 何时做全量训练

先看pilot在相同预算下是否增强正确类别/位置优势，并用少量生成检查类别与位置跟随、Local保持和图像质量。如果只是训练Loss更低、对象仍不响应，不直接加长训练。

`full`入口已准备：7390张train、2000个新增步骤、warmup100，仍以A10000初始化；与32图pilot分开，是一个新的全量实验，不会把pilot的过拟合优化器混入正式训练。先验证pilot后再使用。相同stage命令支持自动恢复。

## 验证范围与论文依据

CPU11项测试通过；38个真实状态的网格/权重/控制检查通过。10张固定诊断图的60个选中实例中57个有空间支持，3个相机内AABB回退原路径；这是粗几何覆盖检查，不是57/60生成成功率。新版本诊断/生成的CPU入口检查也已通过，使用的临时合成checkpoint只测试加载和协议，不代表实际训练。

详见`research/audits/spatial_world_review_2026-10-25/`。没有GPU训练/推理由助手执行。

- [InstanceDiffusion](https://arxiv.org/html/2402.03290v1)：实例区域约束前向交互。
- [PanoCtrl，2026预印本](https://arxiv.org/html/2608.20691v1)：球面区域采样/残差写回，与区域Loss分开设计。
- [LACONIC](https://arxiv.org/html/2507.03257v2)：相机坐标3D条件＋额外attention，支持冻结主干范式；也说明没有显式写回本身并非逻辑错误。

本实现是针对本项目的轻量可消融方案，不是上述方法完整复现，不能保证仅靠空间通路解决类别生成、预测状态噪声或polished与mesh标注差异。

## 已完成：同一个Spatial200的local开关与一次回写

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh mechanism
```

默认固定`run-1792923761168994486/checkpoint-step000200`，4train/6val，sigma0.8008；4种模式共40次单步前向：

- `native_v1`：原v1，local开启。
- `measured_v1`：按相同运算测量实际局部回写，要求预测逐元素等于native_v1，否则停止解释。
- `local_off`：同一份Spatial200参数，仅关闭local；旧Encoder/旧attention/完整状态均保留，不是World-off，也不是另一套Control200权重。
- `single_cast`：同一份Spatial200参数，将旧/新残差在FP32合并后统一回写。仅在这个诊断模型中切换，不改变v1训练器或checkpoint。

输出在`outputs/object_world_spatial_mechanism_v1/<run-checkpoint>/`：`report.html`、`LOCAL_EFFECT.png`、`SUMMARY.json`、逐图records。用户已完成默认40条，测量实现10/10精确复现原预测，无需重跑。本次CPU分析没有从Slurm日志认证实际运行耗时。

CPU5项机制检查通过；真实Spatial200可严格加载，10图缓存齐全。整个过程不训练、不生成完整全景、不下载、不改已有权重。

注意：本轮Spatial/Control训练的`binding_weight`都为0，比较的是有无前向空间支路；它不能证明实例attention绑定Loss无效。区域Loss0.25与绑定Loss是不同监督。`mechanism`检查的也是通路强度/数值，不是对绑定Loss的评价。

## 已完成：只训练local的100步校准

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh calibrate
```

从原Spatial200完整权重初始化，不重置local。旧Encoder/attention/gate全部冻结，仅local的4,935,171参数可训练。新FP32 AdamW，eps1e-8、lr1e-4、warmup20、global batch4；同32张train新增100步/12.5轮，从原第200步后的采样/噪声索引继续。保持原v1前向及两次BF16回写、region0.25、binding0、ID编码、LoRA0.5和20/40/59层。

新输出`outputs/object_world_spatial_v1/calibrate/spatial/`，每25步保存，重复命令自动从完整checkpoint续训，不自动评估。每步检查冻结模块没有梯度，每个checkpoint用`FROZEN_CHECK.json`确认旧Encoder/attention及缓冲逐tensor精确不变。模型forward版本不变，config中的`optimization_policy=local_only_calibration_v1`明确区分训练策略，不能用它恢复原联合训练的optimizer。

助手已完成真实权重/32图缓存CPU预检查与局部梯度/精确恢复测试；没有GPU运行。按旧joint训练约59秒/步估算，本轮约1.5～2小时，不含排队，新的GPU速度/显存尚未实测。不是全量训练或已证明改进的方案；同时改变冻结策略和eps，用于验证该local通路能否学出更明确条件响应，不能单独归因于eps。

训练完成后，用户另行执行：

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh calibrate-diagnose
```

默认与它的初始化Spatial200比较，4train/6val、3sigma、5模式，共300次前向；验证图仍是开发验证集。优先看被修改实例的正确类别/位置优势，不能以重建loss下降、残差或gate增大代替对象控制验证。诊断结果已写入`outputs/object_world_spatial_condition_diagnostic_v1/<new-run>-spatial-<checkpoint>/`。

更新：上述100步校准与300条诊断现已全部完成，不需要重复。高sigma的被改实例区域类别gap -0.15993%（正确更优0/6）、位置gap -0.18142%（0/5）；中sigma分别+0.21342%（5/6）和+0.18149%（4/5）。指标定义/限制见最新REPORT，不将latent误差当生成准确率。

## 已完成：校准权重8图生成

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=02:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_spatial.sh calibrate-probe
```

自动选最新完整calibrate checkpoint，另用其初始化Spatial200做correct对照，共8图：旧correct、baseline、校准correct、move_tv、mirror_tv、remove_tv、hidden_chair_to_plant、add_floor_plant。LoRA0.5、28步、CFG4、同Local/种子。不是重训；不自动进行后续训练或大型评估。CPU真实权重/干预检查通过，GPU尚未运行。

输出`outputs/object_world_spatial_probe_v1/<new-run>-spatial-checkpoint-step000100/`，包含原有`comparison.html`、`direction_review.jpg`及完整ERP/固定视角对象区域可视化。重复命令补齐已有协议内的未完成图。新可学习空间attention/绑定Loss尚未实现，不能将此命令误认为训练新架构。

更新：默认8图已全部完成，无需再次执行。结果与源码/实际路由的核对见`research/audits/spatial_generation_results_2026-10-26/REPORT.md`、`ANALYSIS.json`。`TRAIN128.json`是覆盖61栋train的候选ID清单，缓存齐全；`NEXT_EXPERIMENT.json`明确为设计方案，不是已有trainer配置。下一步应实现新语义Encoder与可学习实例交互，按匹配A/B验证绑定监督增量；不能只根据当前latent误差或像素差声称语义生成成功。
