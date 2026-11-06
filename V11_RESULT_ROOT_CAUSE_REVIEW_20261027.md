# v11 实际结果：先验证条件机制，再决定是否改因果图

2026-10-27，Asia/Shanghai。来源为用户运行的 `outputs/holistic_completion_20261026_235517_1710340`。本轮保持主生成代码、DAG、先验和平滑参数不变，只增加只读审计和固定 D 的机制诊断脚本；没有调用新 GPU/API。

**结论：当前没有证据表明 v11 改善了完整世界补全。Qwen 的空集集中减少，但出现了更明确的“单装配、左侧、菜单第一项”选择失效。API 实验中断，不能据此比较两后端的生成质量。下一步应固定因果结构，先修正并验证机制输入/接口和实验完整性，而不是再加潜变量。**

[28 个真实世界的同坐标可视化与原始证据](../research/audits/v11_result_review_20261027/actual_001/index.html)；[完整审计 JSON](../research/audits/v11_result_review_20261027/actual_001/audit.json)。包含 v9.1、v10 端到端全部样本，以及本次 Qwen 的端到端、直接库存、RGB 对照全部样本；空样本和失败状态保留。图为状态框，不是 RGB 全景。

## 1. 这次到底运行成功了哪些部分

| 路径 | 后端 | 四次新增记录 | 有限域 E[N] | 最终中心 Z<0 / 全部新增 |
|---|---|---|---:|---:|
| v9.1 端到端 | Qwen | 7,6,9,9 | 8.0052 | 17/31 |
| v10 端到端 | Qwen | 0,0,0,20 | 4.1160 | 3/20 |
| v11 端到端 | Qwen | 1,1,2,2 | 1.8100 | 2/6 |
| v11 直接库存 | Qwen | 1,1,1,3 | 1.9286 | 0/6 |
| v11 RGB 库存对照 | Qwen | 1,1,2,1 | 1.3000 | 0/5 |
| v11 API / 同库存布局 | — | 未完成 | 不可用 | 不可用 |

40 个计划世界中，仅 16 个 Qwen 世界导出；其余 24 个依赖阻断，包括 20 个 API 世界和 4 个 Qwen pooled-layout 世界。**未完成不是空补全，也不是数值布局失败。**

API 提案阶段两次 HTTP 500，各约 241 秒；机制阶段 reference 四次有文本回复，但 recognition_0 再遇 HTTP 500。没有形成 API 机制表，pooled 表随之无法形成。官方把 500 归类为未知服务错误；这不是判定 DeepSeek 因果/空间能力的有效实验。[SiliconFlow 官方错误处理](https://docs.siliconflow.cn/en/faqs/error-code)

## 2. 物体变少，主要发生在几何之前

### 2.1 共同候选域再次缩水

v10 有 11 单位/21 实例；v11 只有 **6 单位/11 实例**：额外沙发、边桌、杂志篮、桌椅、边桌/杯/杯垫、书柜/两本书。

原因有两类：

- API 四个计划提案请求中有两个服务失败，候选支持不完整。
- Qwen 最终仅贡献 **1 条非空实例**。一次提案是观察功能的空重述；另一些回复把 `lighting/storage/support` 放入“物理承载表面”枚举，或遗漏追加要求的 `observed_host_name`，未通过接口。原始修复回复仍重复了若干错误。

遗漏宿主名称与填错宿主身份不是同一个问题。精确 obs ID 本来已决定名称，要求模型再重复一遍会引入冗余失败。未来接口应从固定 ID 绑定缺失名称，并拒绝显式矛盾名称；不能猜测或更换宿主。普通用途词也不能被自动改成 top/seat 等物理表面。

本轮做了**离线假设修复重放**：保留非表面词为 unresolved metadata、从原 ID 补名称，不改宿主。找回的主要是托盘和脚凳，仍高度接近已经可见的物体。因此，“修接口找回更多对象”不等于“更合理的隐藏世界”；提案的观测重述与重复实例问题也需评价。本轮没有把这些重放内容写回 D。

### 2.2 模型实际几乎把联合集合当成单选

| 选择阶段 | 有效请求 | 单装配请求 | 被选单位处于随机菜单第 1 位 | 实际所选方位 |
|---|---:|---:|---:|---|
| 功能联合选择 | 48 | 46 | 31/51 条选择记录 | left 49，right 2，rear 0 |
| 直接库存 | 48 | **48** | **47/48** | **left 48** |
| RGB 联合选择 | 48 | 47 | 9/53 条选择记录 | left 53，rear 0 |

功能路径中两次多选发生在 q(H)=0 的诊断 H；正识别质量支持上的条件选择全为单装配。数学模型允许多个装配共存，实际 LM 条件机制却没有可靠实现这种联合采样。

直接库存的候选顺序每次都随机打乱，仍 47/48 选择第一个。不能用“前面的候选恰好一直最好”解释这个现象。这里没有严格识别偏差的因果来源，因为原请求的菜单顺序和解码噪声随 draw 一起变化；但足以说明当前选择机制需要受控验证。

全部直接选择仅生成 15 tokens、以 EOS 结束，预算为 1536；不是输出被截断。增大 token 上限无法解释或修复这种行为。

原输出示例是一条 `assemblies:[{id,region}]`，可能强化单选解释；方位选项又一直按 left/rear/right 展示，可能强化首项选择。**这些是待验证假设，不作为已经证实的根因。** 选择偏好与位置敏感性已有一手研究支持，但 MCQ 结论不能直接等同于本项目的生成任务。[ICLR 2024：Large Language Models Are Not Robust Multiple Choice Selectors](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html)

### 2.3 库存/空间机制的输入遗漏了已生成的信息

`joint_prompt` 使用的简化菜单保留类别、宿主、功能标签和部分关系，却省略了程序 description、group.spatial_reason、对象 placement_hint 与 distinct_instance_reason。

例如 a3 原提案明确说“工作区可能在相机对面”，但选择机制看不到这段 spatial_reason 和桌子的 placement。主路径 O+ 又只有对象 ID/名称，不含观测几何。在这种输入下要求模型合理决定原相机后方方位，存在明显的信息缺口。

这不是让模型输出更多解释，而是把**已经存在、会影响概率决策的语义/空间条件真正传入机制**。是否有效应在相同 D 下验证，不能同时重新提案、换概率和换布局再归因。

### 2.4 因果收益目前也未得到证明

现有 H 不同行的提示菜单顺序和本地解码 seed 都依赖 H 的 stage 名。四次小预算下，即使 CPT 数值随 H 改变，也可能混入菜单位置敏感性和有限采样误差。应在固定菜单、固定初始 seed 下改变 H，单独看功能变化是否带来可解释的实例/方位概率变化。

SD-SCM 提供指定结构与 LM 条件机制的采样/干预框架，不保证任意 LM 接口实现了期望语义，也不证明当前场景图已识别为真实物理因果图。[SD-SCM 原文](https://aclanthology.org/2025.emnlp-main.107/)

## 3. 后方究竟有没有生成

端到端 world_04 有两条完整后方框：桌子约 [-5.35,-1.40,-3.50]、椅子约 [-4.45,-1.37,-3.30] 米；方位角约 -123°/-127°，属于**左后方**。中央后方（|角度|>135°）没有对象。

但原始 reference、条件、RGB、直接选择共 148 条有效请求，均没有选过 rear。两条后方框来自 left 域内的负 Z 布局——left/rear 域本来可以重叠。因此，它们不能证明模型学到了合理的 rear 方案分布。

端到端声明 rear 的概率约 0.026604，全部由平滑尾项产生：

\[
P(\exists\,R_i=rear)=0.04\left[1-(5/6)^6\right]=0.026604.
\]

每个单位均有三种合法方位，尾项单位存在质量为 1/2。后方从“未得到原始票”到获得小非零概率，属于模型平滑支持，不是功能证据支持。

E[N]=1.81 可拆为条件原始票 1.31、参考票 0.28、均匀尾项 0.22。几何前的库存已经很少；本次没有发现求解器删除采样对象。

## 4. 哪些局部改动有效，哪些不能判断

- 空集集中减少：四个端到端样本均非空，P(empty)=0.000625。但它退化为稀疏单选，完整性并未改善。
- 明确方位仍受保护；但本次没有原始 rear 票，不能根据这次生成认证后方机制有效。
- 实例身份和观测保留：已有世界保持 G_obs，新增数量与采样库存一致。
- 端到端新增物体**全为无宿主地面对象**，承载检查次数为 0。因此不能把本次导出成功归功于新支撑代理，也不能把物体少归因于该约束。
- 直接库存 world_04 有两个承载物，中心代理通过，但杯垫尺寸为 [0.1,0.1,0.1] 米，仍暴露数值/语义能力问题。中心代理通过不是物理认证。
- 杂志篮被称为“靠近灯”，实际中心距约 3.236 米、框表面间距约 2.255 米。功能文字尚未得到空间使用关系的充分支持。
- RGB 条件 48 次中 39 次选择额外灰沙发，且几次数值尺寸与观测沙发近乎相同。它可能是新增实例，也可能是观测复制倾向；需要身份依据，不能因为类别相同就一律删除或一律接受。

## 5. 下一步固定什么，验证什么

**先固定 v11 的图结构和数学采样机制**：

\[
q(H\mid O,D)\,p(B,R\mid H,O^+,D),\qquad
B\to E\to S\leftarrow S_{host},\quad (O,H,B,R,E,S)\to X.
\]

当前问题主要是 p(B,R|...) 的估计机制未通过有效性检验。再加生活史、房间容量、数量变量或后方配额，无法修复“47/48 选菜单第一项”。结构暂不扩张，不表示方法已定稿。

已提供小规模的**固定 D、固定观测、配对 seed/菜单的诊断脚本**：[selection_probe.py](../research/audits/v11_result_review_20261027/selection_probe.py)。它不重新提案、不生成布局、不改原 CPT，也不把失败记成空世界。

| 条件 | 输出接口 | 决策信息 | 状态选项顺序 | 主要检验 |
|---|---|---|---|---|
| A | 原可变长列表 | 原简化菜单 | 原顺序 | 复现旧机制 |
| B | 每个候选都赋 none/方位 | 原简化菜单 | 原顺序 | 单行示例/单选解释 |
| C | 完整赋值 | 保留已有空间理由、placement 和观测预测框 | 原顺序 | 机制信息遗漏 |
| D | 完整赋值 | 同 C | 仅重排合法状态 | 选项顺序敏感性 |

完整赋值只要求每个候选写一个状态，**none 仍合法，也可全部 none**；不要求增加任何对象。选项重排不改变合法域或先验权重。各条件使用相同候选顺序与初始本地 seed；API 不传 seed。A 的四个提示词和 seed 已逐字核对与原 h4 请求相同。

可以先只跑 h4：16 次初始选择/后端。若同时跑 h4、h5（后者只增加 work 标签），则 32 次/后端，能在固定呈现条件下看加入 work 后桌椅概率的响应。每次 JSON 最多一次修复，最大请求预算为初始两倍；遇到传输错误立即停止后续请求，不自动重试。API 结果只是同一 D/H 上的文本条件机制测试，不代表它独立完成了图片提案或端到端补全。

观察结果时：B 数量增加不自动判胜，应看是否摆脱单选/首项依赖并保持身份合理；C 是否改善功能与原相机空间组织；D 是否只因重排而大变。若冻结 Qwen 仍不能稳定实现联合选择，而 API 能实现，则有能力层面的证据。若 API 仍服务失败，则继续把它记为缺失实验。

只有上述机制表现可信，才值得修复提案单一 schema、重建完整 D/CPT，并执行同库存几何与全链路比较。避免为恢复数量而放行观测复制或伪造支撑表面。

## 6. 执行方式

本轮已完成只读审计和 prepare-only；没有执行下列真实模型请求。原主生成代码 92 项哈希不变、5,261 份来源 JSON 保持不变。诊断脚本的 CPU fixture 不代表真实效果。

Qwen：下面运行两个 H、四种条件，每种四次；共 32 次初始请求。输出必须用新目录。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  research/audits/v11_result_review_20261027/selection_probe.py \
  --source qwen_edit_pano/outputs/holistic_completion_20261026_235517_1710340 \
  --backend qwen --profiles h4 h5 --draws 4 \
  --out qwen_edit_pano/outputs/v11_selection_probe_qwen_20261027_001
```

API：同一候选、H、条件、预算，沿用 `SILICONFLOW_API_KEY` 和硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  research/audits/v11_result_review_20261027/selection_probe.py \
  --source qwen_edit_pano/outputs/holistic_completion_20261026_235517_1710340 \
  --backend api --profiles h4 h5 --draws 4 \
  --out qwen_edit_pano/outputs/v11_selection_probe_api_20261027_001
```

输出 `index.html`、`summary.json`、`results.json` 包含每条件有效数、真实选择、单位存在频率、方位、菜单位置和 H 差值；`plan.json` 包含提示词/seed/来源哈希。频率是少量机制调用的诊断，不是校准的世界概率。需要先查看计划时，在命令末尾添加 `--prepare-only` 并使用另一个新目录；不会调用模型。

只读审计可复现：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  research/audits/v11_result_review_20261027/review_result.py \
  --out research/audits/v11_result_review_20261027/my_review_001
```
