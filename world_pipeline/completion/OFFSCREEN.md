# 从局部 RGB 推测视野外世界

本实验的目标是：理解图片中已经看到的内容，提出相机视野之外可能存在的对象，再为这些候选采样结构化几何。不是恢复画面内被遮挡的部分，也不是重新布置画面内的家具。

- `G_obs`：图片中可见的对象；部分可见的对象仍属于观测对象。
- 视野外候选：未被看到、可能存在于输入视锥之外的另一个实例。例如看到床之后，推测画面边缘之外可能有床头灯。候选不是事实。
- `G_full`：原有 `G_obs` 加上通过几何检查的视野外候选，是一个可能世界，不代表已经恢复了全部真实对象。

为兼容现有接口，新增对象仍保存为 `G_hidden.json`。在本模式下这个文件仅表示视野外对象，不包含视野内遮挡补全。对象带有 `hypothesis_scope=outside_input_fov` 和对应语义提案。

## 实现

继续使用已有 Qwen 观测结果，无需重跑 Observer：

```text
Local RGB + 已有预测 G_obs
  → Qwen：可见场景描述 + 最多 4 个视野外候选（不是固定配额）
  → M3DLayout add_object：给定候选类别、上下文与文本，采样对象几何
  → 保留整个框位于视野外的候选
  → 原始 G_obs + 新增对象 = 候选 G_full
```

提案包含类别、可见依据、假设关系和可选方向。不要求 Qwen 编造坐标。方向不确定时使用 `unspecified`；床后面不等于相机背后。只有明确提出方向时才增加方向检查。关系文本目前不是精确的几何约束，必须看结果判断是否落实。

M3DLayout 每个候选最多采样 3 次；所有失败的原始样本也保留。没有可接受几何时保留原始观测，记录 `no_accepted_geometry`，不把它算成成功补全。不会为了产生结果把对象强行移到视野外。

## 执行

使用已有 qwen3vl 环境、离线权重、现有 12 张验证图片和 `qwen_world_simple_run001/observer`。默认 GPU2；可用 `export OFFSCREEN_NODE=GPU4` 更换节点。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export OFFSCREEN_RUN=qwen_edit_pano/outputs/qwen_m3d_offscreen_run001

# 1. 仅生成视野外语义候选
bash qwen_edit_pano/world_pipeline/completion/run_offscreen.sh propose

# 2. 看图片、可见依据和假设，暂时不需要 M3DLayout 结果
bash qwen_edit_pano/world_pipeline/completion/run_offscreen.sh review

# 3. 为同一批候选生成几何；分别使用 GT / predicted 观测上下文
bash qwen_edit_pano/world_pipeline/completion/run_offscreen.sh run

# 4. 更新语义与几何报告
bash qwen_edit_pano/world_pipeline/completion/run_offscreen.sh review
```

报告：`$OFFSCREEN_RUN/review/offscreen.html`。
可选人工记录：`$OFFSCREEN_RUN/review/semantic_review.csv`。
数值和拒绝原因：`$OFFSCREEN_RUN/review/offscreen_metrics.json`。
实际输入路线的世界状态：`$OFFSCREEN_RUN/completion/pred/<sample_id>/seed0/G_full.json`。

## 如何判断结果

先看语义：是否根据可见线索提出合理的视野外实例？是否重复了已见实例？是否与画面矛盾？人工表格的三个判断列留空，不能由生成数量代替。

再看几何：原始采样有多少位于视野外、多少符合提出的方向、多少最终被接受；同时看布局图中的大小、位置与关系。最终对象在视野外是筛选保证的，不能用这一点单独证明模型能力。

GT 隐藏对象可能包括画面内遮挡对象，因此新报告仅使用“整个相机轴 AABB 可确认位于视野外”的 GT 子集作参考。这个子集是保守的，会漏掉部分视野外对象。与 GT 的类别/位置一致性衡量对该真实世界的恢复程度，不等于所有合理假设的质量。

`pred` 是实际目标路线；`gt` 仅替换几何上下文以帮助定位误差。同一语义提案仍来自 RGB 和预测观测，所以 `gt` 不是完全理想的上游 Oracle。旧 `review.html` 提供布局图，但其 broad-hidden 指标不能代替新报告的视野外指标。

## 本实验的边界与后续决定

当前桥接仍假设相机近似水平、高度 1.5 米；没有从 RGB 恢复真实房间坐标。官方 AR 权重也不能被视为已经适配任意可见对象集合。这次修改明确任务和评价，不承诺修复已出现的空间偏置。

- 如果语义候选不合理，先定位可见观测或候选依据，不调 M3DLayout 采样数量。
- 如果候选合理但几何持续失败，问题在几何补全/坐标桥接，不能据此否定“图片外可能有该对象”。不要反复改提示词或增加重试来掩盖。
- 如果候选与几何均合理，就保留这条简单路线，之后再用训练完成的 Adapter 验证是否能控制 ERP。

无需在这一阶段增加因果图、训练新模型或重新训练 Adapter。
