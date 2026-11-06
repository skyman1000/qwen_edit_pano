# 原 v13 的可选类别与关系接口实验

2026-11-06。AI 生成说明，需独立核验。实际旧结果的根因、文献和模型解释见 [结果核查](reports/original_v13_review_20261106_001/README.md)，[可视化](reports/original_v13_review_20261106_001/index.html)。

本次保留原 v13 H/A/V、概率平滑、采样和几何。仅调整新候选提案：从项目合法对象类别数字表选择 ID、保留具体物名，辅助关系仅审计、支撑可选，缩短提案提示并保留整室语义。默认仍是原 v13，需要显式设置 `CAUPANO_PROPOSAL_INTERFACE=closed_optional_v1` 才使用新接口。没有数量、后方和密度配额，没有训练，没有复用旧候选或读取 GT 世界。实际是否改善等待 GPU 结果。

## 新跑同一五张图

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_LAYOUT_RGB=0
export CAUPANO_PROPOSAL_ROUNDS=1
export CAUPANO_WORLDS=1
export CAUPANO_DRAWS=4
export CAUPANO_PROPOSAL_INTERFACE=closed_optional_v1

srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/world_completion.sh run \
  qwen_edit_pano/outputs/v13_category_optional_20261106_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

纯本地 Qwen，不需要 API Key。根据可用节点调整 GPU3；输出目录必须是新目录。每图只输出一份采样世界，K=4 表示机制短选择次数。CPU 准备及历史测试不占用 GPU，不是实际补全实验。

自动页面：`qwen_edit_pano/outputs/v13_category_optional_20261106_001/index.html`。包含原图、平面图、投影、3D 和 2:1 ERP 状态投影；框不代表生成了 RGB 全景。标签包含对象编号、类别数字、名称；映射成功的新增物体使用同一颜色。冻结观测有两条历史未知类，原样保留，没有偷偷映射到 misc；新增对象必须是词表中合法 ID。

## 中断后继续

在保留上述环境变量的终端中，将运行参数改为：

```bash
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/world_completion.sh resume \
  qwen_edit_pano/outputs/v13_category_optional_20261106_001
```

resume 复用已完成的阶段记录，不删除失败记录、不反复采样直到得到想要的世界。提案 partial 可以继续导出存活候选，但整个命令保留非零退出状态，需要阅读日志。

仅重做页面可在 CPU 执行：

```bash
CAUPANO_PROPOSAL_INTERFACE=closed_optional_v1 \
  bash qwen_edit_pano/scripts/world_completion.sh report \
  qwen_edit_pano/outputs/v13_category_optional_20261106_001
```

原 v13 对照使用 `CAUPANO_PROPOSAL_INTERFACE=original` 和独立新目录。不要把新旧接口目录混用；每次绑定源码、类别表和原基线哈希。新接口改变候选域及提示，不是纯因果消融。

## 文件与检查

- 新接口：`world_pipeline/completion/v13_category_contract.py`。
- 新执行脚本：`scripts/v13_category_contract.sh`；统一入口：`scripts/world_completion.sh`。
- 项目对象词表：`data/gt_world_supplement_v1/vocabulary.json`。读取类别表，不读取场景 GT 几何。
- CPU 测试：`world_pipeline/completion/test_v13_category_contract.py`，8 项通过；使用固定测试回复代替真实模型，贯通原采样、几何与全部可视化。
- 核查记录：`research/audits/v13_category_contract_20261106_001/`（相对工作区根）。

已声明的合法安装宿主仍参与原几何；省略 attachment 表示支撑未知的独立放置，没有保证支撑正确。普通关系只保留审计，没有新增关系软优化。类名一致和几何检查通过也不能保证场景合理。
