# 14 栋 train / 3 栋 val：固定数据扩展实验

2026-10-08。已生成配对清单并完成 CPU plan/名单一致性检查、shell 语法检查；未下载、未构建 GT、未训练或推理。入口 `run_gt_14train3val.sh` 内部固定路径，不使用 GT_PAIRS_ROOT 等环境变量，不改变旧全量脚本。

候选共 1290 train / 174 val。已有 5 train / 1 val 保留；新增 train 为 Pm6F8kyY3z2、JF19kD82Mey、VVfe2KiqLaN、JmbYfDe2QKZ、PuKPg4mmafe、7y3sRwLe3Va、ur6pFq6Qu1A、Uxmj2M2itWa、V2XKFyX4ASd；新增 val 为 zsNo4HB9uLZ、X7HyMhZNoso。

新训练建筑按候选图片数 32..155 范围内的排序均匀取九个位置，避免仅选极小或极大建筑；验证新增建筑分别有 51、81 条配对。图片数只作规模代理，不能声称已核验语义多样性或实际下载体积。官方建筑划分保持不变。按此前七栋平均体积粗估新增原始包约 43 GiB，不包括派生文件；具体建筑可能偏离平均，注意实际磁盘余量。

## 1. 下载缺失原始资产

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh download
```

下载器保留原有数据条款交互。只检查固定 17 栋所需的八类 GT 资产，已有有效 ZIP 校验后跳过，因此当前缺失对应新增 11 栋；不下载 skybox 或 polished 全景。中断后原命令续跑。结束会刷新 inventory，期望 fully_present_houses=17；这个指标仅表示文件存在，不表示 GT 质量通过。

## 2. 构建与导出

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh build
```

build 成功后：

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh export
```

二者支持原路径续跑；遇到错误先检查日志，不删除旧资产。源目录 `qwen_edit_pano/data/gt_sources_14train3val_v1`，导出目录 `qwen_edit_pano/data/gt_world_14train3val_v1`。

## 3. GT 验收（导出完成后，再执行后续步骤）

```bash
cat qwen_edit_pano/data/gt_world_14train3val_v1/EXPORT_COMPLETE.json
```

检查 failures.json、review.html、gt_manifest.jsonl 中 flags 与投影。将真实检查后的决定记录到同目录 review_decisions.json；不能批量自动 approved。最终使用全部 approved 且无 flags 的样本，实际数量可能低于 1290/174。若某栋没有可用样本，应解决/明确该问题，不能悄悄宣称仍覆盖 14/3 栋。助手可在导出后读取产物协助验收。这里是数据质量检查，不是重复 7 条 smoke。

## 4. 建立扩大规模的 epoch006 bundle

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --qos=normal --time=00:30:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh bundle
```

生成 `qwen_edit_pano/data/oracle_14train3val_epoch006_v1` 并 preflight。会检查所有选定建筑均有通过验收的样本。继续复用原 full 配对条件缓存和 epoch006 checkpoint，没有修改原训练清单。此阶段不加载生成模型。

## 5. 训练 full 与 constant（两个独立作业）

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh train-full
```

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh train-constant
```

冻结 epoch006 和基础模型，从头训练 World 分支。相同 seed 0、lr1e-4、1000 optimizer steps、warmup100、accumulation4、每100步保存、CPU激活存储。full 输入真实GT类别/中心/尺寸/mask；constant 无每图状态，排除额外训练本身的影响。不接入 Observer/Completion，也不附加 GT 文本。

1000步是第一阶段预算，不是最优训练长度的结论。若保留1290条，对应4000次有放回抽取、约3.1次数据集等量访问，不能声称每张均遍历。先看保存点和验证表现，再决定延长；仅有训练loss下降不代表状态收益。96G是CPU内存；使用此前训练成功的GPU规格，旧大显存卡smoke不保证48G通过。24小时是申请上限，不是时长保证。

训练中断可指定已有保存点与新输出目录，例如（替换实际存在的步数）：

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh train-full \
  qwen_edit_pano/outputs/oracle_14train3val_epoch006_full_v1/checkpoint-step000500 \
  qwen_edit_pano/outputs/oracle_14train3val_epoch006_full_resume_v1
```

最终总步数仍为1000，不是额外1000步。恢复时数据、代码和训练配置必须一致。

## 6. 独立 val 对照推理

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh eval-full
```

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_14train3val.sh eval-constant
```

默认取训练step1000，评估全部合格val（最多174），full生成baseline/correct/shuffled，constant仅生成其固定条件结果（接口名correct不代表输入了GT）。同seed0、28steps、CFG4。若174条全通过，共696张图，耗时可能较长；当前推理入口不支持断点续跑，输出目录须新建。可通过入口原有 oracle_inference 的 --limit 先作资源估时，不据截取前几张下跨建筑结论。

若训练恢复到新目录，eval-full/constant 支持追加 checkpoint 和新输出目录两个位置参数。输出默认 `qwen_edit_pano/outputs/oracle_14train3val_epoch006_{full,constant}_val_v1`。

按三栋建筑分别检查Local保真、镜头外类别/数量/方向、布局、接缝，再汇总；不要让图片多的建筑掩盖另两栋退化。shuffled沿用已有同split错配，仅诊断条件依赖。三栋val仍是初步oracle验证，不等于完整数据集或预测状态的部署效果。
