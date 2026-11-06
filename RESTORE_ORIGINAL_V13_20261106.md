# 恢复交接前的 v13 因果场景补全

2026-11-06。本文由 AI 整理，不一定正确；应以源码、原始模型回复及实际结果独立核验。

## 恢复结论与范围

用户要求恢复本次对话交接前的模型。本次以 10 月 28 日实际运行的 `organized_functional_program_scm_v13` 为依据，而非后续 AI 推荐的 joint、panorama_plan、simple、scope 或 stable 流程。

实际核对发现原核心仍在：H/A/V、计数平滑、结构采样、程序展开与几何等 12 个关键文件，与 `outputs/kitchen_observer_20261028_144049/suite.json` 保存的哈希完全一致。当前 v13 基线清单的 118 个绑定也全部一致。因此无需重新设计或覆盖这些文件。

与该 10 月 28 日快照相比，仍有三个外围文件存在差异：`assembly_multiscene.py`、`assembly_compositional_experiment.py` 的单后端适配，以及 `assembly_io_recovery.py` 的观测接口修复继续保留。本次恢复因果生成方法和入口，不宣称整个项目已逐字节回退。

本次实际变更：

- `scripts/world_completion.sh` 固定委托原 `organized_local_world.sh`，不再根据环境变量或旧运行目录自动转入二轮提案实验；发现这类配置时明确退出。
- 简版交接重新整理，长交接和旧执行说明顶部指向本页，消除相互覆盖的默认推荐。
- 原核心、当前观测、历史结果及实验分支不删除、不修改；旧实验仅作为历史对照，不作为当前默认方法。

修改前文件与审计存于 `../research/audits/restore_original_v13_20261106_001/`。当前工作区没有可用的 Git 仓库，依据运行产物的源码哈希核验，未使用 Git 回退。

这里恢复的是生成方法及明确的执行路径，不能承诺新请求复现历史所有物体。旧 Qwen 丰富结果使用了 **Qwen＋API 共同候选域**；当前纯本地入口只使用 Qwen。当前观测也有更新。这些配置差异必须与模型结构区别对待。

## 恢复后的模型

```text
局部 RGB + HFOV
→ Qwen 二维实例识别 → WildDet3D 三维框 → 冻结观测 O+
→ 完整功能装配候选域 D
→ 不确定功能组织 H
→ 各功能额外装配是否存在 A_r
→ 完整装配与相机方位的联合变体 V_r
→ 对象存在 E_i、方位 R_g、宿主安装 S_i
→ 固定尺寸 → 朝向与位置 → 原数值几何求解
→ G_full = G_obs ∪ ΔG 与可视化
```

离散分解为：

`q(H | O+, D) × ∏_r p(A_r | H, O+, D) p(V_r | A_r, H, O+, D)`。

H 为混合用途或强调某功能，不推断住户生活史。给定 H 后角色条件独立是原近似；一套完整程序内部可以包括多个共存组和附属用品。V 同时选定程序与方位，不独立拼凑。S 表示声明的宿主安装依赖。

大模型提出候选、作短选择、估计尺寸和初始布局；程序负责验证、计数平滑、建立 CPT、生成均匀噪声和逆 CDF 采样、展开实例、坐标变换及几何求解。默认每条件 K=4 个短选择，存在率为 `(n_present + 0.5)/(K_valid + 1)`。这些概率确实进入库存采样，但不是经过校准的现实后验；没有从照片识别真实因果图。

不加入数量或后方配额，不用“无直接视觉证据”排除一切隐藏假设，也不自动增加解释文字。局部观测来自预测而非真值；厨房等观测仍可能漏检。ERP 是状态框投影，不是生成的 RGB 全景。

## 执行

以下使用现有五图冻结观测，共 45 个实例。每图只生成 **一份** 世界，K=4 指机制短选择次数；不调用 API，无需 API Key。不自动沿用实验目录里的提案、概率或布局。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_LAYOUT_RGB=0
export CAUPANO_PROPOSAL_ROUNDS=1
export CAUPANO_WORLDS=1
export CAUPANO_DRAWS=4

srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/world_completion.sh run \
  qwen_edit_pano/outputs/original_v13_restored_20261106_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

输出目录必须尚不存在。首页为 `outputs/original_v13_restored_20261106_001/index.html`，包含原局部图、状态平面图、原视角投影、六方向、交互 3D 和 2:1 ERP 状态投影。空样本和失败如实保留，退出码不等于语义评价。

原来的 Qwen/API 共同候选、匹配配置的双后端流程仍为 `scripts/organized_multiscene.sh`，它需要 `SILICONFLOW_API_KEY`，API 型号仍为硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`。它会重新运行观测，不能直接把上述本地命令称为历史双后端实验的复现。

## 验证边界

核验结果见 `../research/audits/restore_original_v13_20261106_001/validation.json`。31 项 CPU 回归通过；10 份旧世界均可由保存的噪声精确重放结构状态，用当前代码重建的 CPT 及对象概率也与保存结果一致。五图入口预检通过，共 45 个冻结观测实例，每图一份世界。本次不运行 GPU、API 或训练。结构与工程核验通过，不等于新的真实模型补全质量已验证。
