> 2026-10-20 当前不训练实验见 [ANCHORED_WORLD.md](ANCHORED_WORLD.md)：Qwen二维识别→WildDet3D固定G_obs→Qwen锚点布局→G_full。新布局仍待用户GPU验证。NO_TRAIN_TEXT.md和下方训练路线均为历史方案说明。

# CauPano：可见对象条件下的隐藏对象补全

当前正式入口是 `hidden_data → train_hidden → sample_hidden → evaluate_hidden`。
详细依据、修改范围和执行顺序见 [TASK_COMPLETION.md](TASK_COMPLETION.md)。

训练：`GT G_obs + GT hidden[:k] → hidden[k] / END`。
推理：`GT 或 Qwen G_obs → 多个可能的 G_full`。
全部观测对象始终保留；模型只追加对象。类别、相机局部米制中心、local-axis AABB 完整边长与现有 Adapter 一致。

- `observe.py`：图片 → 预测 G_obs；同一 Adapter 类别只查询一次。3D 精度仍需实测。
- `hidden_data.py`：使用现有 Oracle Bundle 的 split 和对象集合，准备训练对与训练集归一化统计。
- `hidden_model.py`：复用官方 M3DLayout AR 主干与类别/位置/尺寸 heads，适配当前任务的采样和损失。
- `train_hidden.py`：训练独立 Completion；保存 best/last、优化器和随机状态。
- `sample_hidden.py`：加载任务 checkpoint，生成 Adapter 可 tensorize 的 G_full。
- `evaluate_hidden.py`：GT-observed 分支，按建筑、seed 报告隐藏对象指标。
- `run_hidden.sh`：手动执行入口，使用 qwen3vl 环境。

旧 `complete.py`、`run_indoor_val3.sh complete`、`run_size_ablation.sh` 只用于历史官方预训练推理诊断。
不要继续把修改这三张图的输出当作正式 Completion 开发。旧操作说明归档到 [README_PRETRAINED_HISTORY.md](README_PRETRAINED_HISTORY.md)。
