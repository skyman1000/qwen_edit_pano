> **2026-11-06 恢复说明：当前默认方法为交接前 v13 H/A/V，执行入口固定为 `scripts/world_completion.sh`。joint、panorama_plan、simple、scope、stable 等后续推荐只作历史记录。恢复依据和当前命令见 [RESTORE_ORIGINAL_V13_20261106.md](RESTORE_ORIGINAL_V13_20261106.md)。本页历史记录可能过时，不应覆盖该恢复说明。**

# 当前执行说明：固定成功观测，运行本地 Qwen v13 完整补全

2026-11-05：默认主线已正式恢复并固定v13源码，统一入口及新的输出目录见 [V13_RESTORED_20261105.md](V13_RESTORED_20261105.md)。下面保留原v13执行记录；不要覆盖已完成的旧目录。

2026-11-04：本页v13已作为基线运行并分析，仍为已验证完整任务入口。新joint_world实验在真实五图上未生成合法世界，暂不替代v13；[差异原因与恢复顺序](reports/joint_world_002_root_cause_20261104/README.md)。下面保留v13原命令与诊断，已有`v13_local_world_001`不能再作为新输出目录；需要新执行时使用尚不存在的目录名。

更新：2026-11-03。上一次命令只运行观察器，**没有执行世界补全**。`local_observation_fixed_002/observation_only_run.json` 明确记录 `hidden_state_generation=false`；五个场景都没有 `completion/`。因此没有新增对象与该命令一致，把这条观察器命令作为完整任务的执行建议是错误的。

五图的二维和三维阶段都已完成，实例数为 **16 / 8 / 8 / 10 / 3，合计45**。保持它们的原始框、类别和度量值，直接接回交接前的 `organized_functional_program_scm_v13`。不再运行 `global_world` 的新模型，也不重跑或修改观察器。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_LAYOUT_RGB=0
export CAUPANO_WORLDS=4
export CAUPANO_DRAWS=4
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/organized_local_world.sh run \
  qwen_edit_pano/outputs/v13_local_world_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

输出目录必须不存在。脚本使用当前 srun 分配，不嵌套 srun。只加载部署的本地 Qwen3-VL，不调用 API，不重新加载 WildDet3D，不训练。`CAUPANO_WORLDS=4` 是每图四个独立采样世界；`CAUPANO_DRAWS=4` 是原版每行机制的四次短选择，两者不同。

实际执行链：

```text
原样冻结 G_obs → prepare-scenes → propose(qwen)
             → pool(qwen) → mechanisms(qwen) → worlds(qwen) → report
```

入口为 `world_pipeline/completion/organized_local_world.py`。它调用现有 v13 实现；仅将原来强制两路候选池化的接口改为可选后端。原来的默认 Qwen/API 双路行为保留。H/A/V、计票平滑、逆CDF采样、对象与宿主展开、先尺寸后坐标及几何检查沿用 v13。每图重新提案，不复用旧世界或旧概率表，不设最少对象数、必出类别和后方配额，不因空增量或失败而反复重抽。

运行后打开：

```text
qwen_edit_pano/outputs/v13_local_world_001/index.html
```

页面显示各世界的可见数、计划新增数和已放置新增数，提供完整世界、假设与库存、因果模型、依赖关系以及失败记录。蓝色为观测，橙色为已映射新增，红色为未映射新增；包含原图、原视角重投影、平面图、六方向、2:1 ERP 和交互3D。ERP 是框状态投影，未生成隐藏 RGB。

关键产物：

```text
ROOT/local_world_run.json
ROOT/phase_records/SCENE_PHASE_qwen.json
ROOT/scenes/SCENE/completion/shared/domain.json
ROOT/scenes/SCENE/completion/shared/mechanisms/qwen.json
ROOT/scenes/SCENE/completion/end_to_end/qwen/completion/causal_model.json
ROOT/scenes/SCENE/completion/end_to_end/qwen/completion/world_01/semantic_world.json
ROOT/scenes/SCENE/completion/end_to_end/qwen/completion/world_01/G_full.json
ROOT/scenes/SCENE/completion/end_to_end/qwen/completion/world_01/G_out.json
ROOT/scenes/SCENE/completion/end_to_end/qwen/completion/world_01/object_relations.json
ROOT/report/results.json
```

失败世界可能没有 `G_full.json`；`semantic_world.json` 保存进入布局前的新增库存，`G_full_candidates.json` 保存未通过数值检查的布局候选。报告区分 `not_run`、`blocked_dependency`、`phase_failed`、`empty_candidate_domain`、`observed_only_sample`、`layout_failed`、`rejected_layout_candidate`、`invalid_world`、`exported_with_new_objects`。`placed_new=null` 表示没有有效布局，不能解释为模型认为隐藏世界为空。

只要有世界失败或抽到空增量，整批返回1并保留成功世界；首页给出“含新增且通过数值检查”的数量，不能单凭 srun 的退出码判断全部失败。空增量保持原样，不伪装为完成了隐藏补全。v13 尚不具备像素级 RGB 似然和遮挡验证，数值通过不等于语义、真实空间或因果收益已验证。

中断后可用相同 srun 参数，将脚本末尾改为：

```bash
bash qwen_edit_pano/scripts/organized_local_world.sh resume \
  qwen_edit_pano/outputs/v13_local_world_001
```

已完成的逐图阶段不会重跑。失败阶段记录也保留，`resume` 不会删除失败后重抽直到成功；若源码或输入改变，哈希检查拒绝混用。

根因对照、研究依据与验证范围见 [V13_LOCAL_WORLD_REPAIR_20261103.md](V13_LOCAL_WORLD_REPAIR_20261103.md)。以下保留旧观察器说明，仅用于历史诊断。


局部观察器的历史诊断见 [LOCAL_OBSERVATION_LOSSLESS_FIX_20261103.md](LOCAL_OBSERVATION_LOSSLESS_FIX_20261103.md)；该观察器入口不执行隐藏世界补全，不再作为本任务的运行命令。
