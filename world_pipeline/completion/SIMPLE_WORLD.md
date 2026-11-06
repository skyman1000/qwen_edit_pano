# 简化版 Local RGB → G_obs → 候选 G_full

2026-10-13。使用已有 qwen3vl 环境和本地8B权重，不新增模型、不训练、不动Adapter。GPU由用户执行。此实现的几何与补全效果尚未通过真实模型验证。

## 官方核验与采用范围

- [Qwen官方3D cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb)：目标定位使用短查询、bbox_3d与label输出，没有提供可复制的具体三维答案。这里借用它的目标定位方式，增加现有2D实例区域作为指代和已知相机内参。该笔记本文字中的旋转单位与可视化处理存在差异；这里明确采用可视化x→y→z旋转、角值乘pi的数值约定，保持既有角点桥接。
- [Qwen官方生成参数](https://github.com/QwenLM/Qwen3-VL#generation-hyperparameters)：Instruct建议采样，temperature=.7、top_p=.8、top_k=20、repetition_penalty=1、presence_penalty=1.5。已有本地generation_config并没有包含presence_penalty；新增简短LogitsProcessor，仅对已经生成的token ID施加一次1.5惩罚，不惩罚仅出现在prompt中的token。这是Transformers路径的实现，不声称复现官方API所有内部行为。不能保证该参数提高几何精度。
- [LayoutGPT](https://layoutgpt.github.io/) 支持LLM进行布局规划和部分场景补全的研究方向，但有自己的in-context示例、房间条件和评测。这里仅借鉴结构化规划思路，没有复现该模型；语义计划→几何转换是本项目的两次调用设计，不是论文已经验证的Qwen隐藏补全方案。
- [M3DLayout官方配置](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml) 使用文本条件；已有任意可见集合实验不理想。本轮不调用它，也不声称关掉BERT或直接拼接独立布局即可解决补全。

## 本轮只改两件事

1. 复用run003的2D实例框，重新估计全部3D框。旧3D值不会读取、复制或用于初始化。3D请求完全移除具体答案数值示例，避免已确认的74个示例复制问题。旧2D漏检/重复仍然可能存在，这一轮不声称解决二维检测。
2. 隐藏补全分为两次短调用：先用普通文字提出一个明确标为假设的场景延续，再以固定G_obs为条件把提议转成category/center/Local AABB。第一步不要求同时给精确米制数值；第二步也不给示例坐标。

```text
原Local RGB + 已有2D实例框 → 新3D估计 → G_obs
Local RGB + G_obs → 隐藏场景假设文字
假设文字 + G_obs → G_hidden → 程序原样拼接 → G_full
```

G_obs仍是类别+Local米制中心+AABB完整尺寸+validity。生成框使用同一坐标系。新Fourier特征由Adapter内部计算。3D查询失败保留2D事实并记录valid=false；不会把未知几何编造成有效数值。

默认每张图一个候选，最多12个新增对象；这是小规模可行性预算，不是完整世界对象数定义或强制数量。不强迫输出非空，也不把空补全当成任务成功：status里的outcome会是no_hidden_proposal。观测对象由程序保持，模型生成的hidden标为假设。

## 直接执行

复用原来的12张图片，不需要prepare，也不需要再运行旧run_upstream命令。新脚本使用独立输出变量，不受此前export的UPSTREAM_RUN影响。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# 重新估计3D，复用2D；不读取run003三维答案。
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh observe

# 预测G_obs → 一份语义提议 → 一份结构化候选。
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh complete

# CPU生成可视化、语义提议与状态指标。
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh review
```

GPU动作内部srun申请1卡，48G是主机内存，4小时是申请时间而非完成保证；不占用已有Slurm训练任务分配的GPU。需指定节点时，在相应命令前设置 `SIMPLE_WORLD_NODE=GPU4`（选择实际空闲节点）。

输出 `qwen_edit_pano/outputs/qwen_world_simple_run001/`：

- `observer/<id>/G_obs.json`：新三维估计；`reused_instances.json`记录二维来源。
- `completion_pred/pred/<id>/seed0/hidden_plan.txt`：语义假设。
- 同目录 `proposal.txt`、`G_hidden.json`、`G_full.json`、`status.json`。
- `review/review.html`：原图、框、三维投影、隐藏假设和布局。

完成和partial样本可断点复用，不自动反复重试。改变输入或配置需新输出：例如 `SIMPLE_WORLD_RUN=qwen_edit_pano/outputs/qwen_world_simple_run002`，后续各步一致设置。

可选诊断：若预测条件仍补全不好，运行 `run_simple_world.sh complete-gt` 再 `review`，同样只做12次、每图一个seed。GT仅提供可见结构，真实隐藏标签只用于评价，不进入生成提示。

## 如何判断是否推进了任务

先看新3D输出是否摆脱集中复制且与图像方向/尺度相容；再看hidden_plan是否给出有意义的新增世界；最后看几何是否落实该提议并与G_obs相容。仅JSON合法、对象增多或改变seed都不代表改善。如果米制几何仍明显不准，应明确记录当前8B估计能力的限制，而不是继续堆格式修补。

CPU检查：`bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh check`。测试验证程序契约和模拟调用，不代替真实模型评测。
