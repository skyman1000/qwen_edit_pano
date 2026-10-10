# 14 train + 3 val 源资产修复（2026-10-09）

入口：`repair_gt_sources.py`。只用 CPU；不下载、不训练，不改原始 ZIP、旧源目录、训练字段、Panorama LoRA 或 Encoder/Adapter。

原结果：1464 候选中 1047 已构建（978 train / 69 val）；225 几何检查失败、3 朝向匹配不足；两栋建筑被类别冲突阻塞，涉及另 189 候选。已构建不等于已导出/验收。

## 修复范围

- `7y3sRwLe3Va` 的 1 个对象、`X7HyMhZNoso` 的 2 个对象存在类别文字与数字标签冲突。保留原几何、实例及数字标签，用独立记录隔离冲突，在源对象 catalog 中关闭其 category supervision。现有 exporter 会排除这些对象的训练条目；其 mesh 仍参与遮挡。没有猜测新类别，也不整栋丢弃。原失败报告保存为 `region_validation.original.json`；新 region 报告仅声明隔离后的结构可用。
- 225 几何失败中 174 个满足尝试重新融合的观测条件，51 个不满足。排除无比较点或中位相对深度误差 >0.10 的观测，保留 >=9 个观测且覆盖全部 6 个 yaw 组，再用原算法融合传感器投影。仍要求覆盖率 >=0.25、全景中位相对深度误差 <=0.10，且原几何基础检查通过。**检查范围改为保留的观测，不代表已修好被隔离观测的真实误差。** 全部原逐视角检查和前后指标写入 `repair_audit.json`。
- 朝向不足：独立副本增加 SIFT 特征和 RANSAC 次数，保留偶数 yaw 拟合/奇数 yaw 留出检查、2.5°、60/30 个内点、3 个留出视角等原阈值。已对旧 3 条失败测试，仍未通过，继续隔离；不反复更换规则直到“通过”。
- 只修复失败阶段。已有成功阶段通过软链接复用，因此**不要删除旧源目录**。新输出 `qwen_edit_pano/data/gt_sources_14train3val_repaired_v1`。
- 断点恢复核对输入/代码签名与已完成文件哈希；同一命令续跑。已完成但仍不合格的尝试保留诊断，不反复计算；执行中断的未完成阶段归档后重建。不要同时运行两个构建任务。

## 用户执行

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -u -m qwen_edit_pano.world_pipeline.repair_gt_sources
```

中断后重用上面命令。查询进度无需 GPU：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.world_pipeline.build_status \
  --root qwen_edit_pano/data/gt_sources_14train3val_repaired_v1
```

整批结束后有剩余质量失败会返回非零，并写 `BUILD_RESULT.json`；这不撤销已成功的修复。先查看完成数量与失败原因，不要循环重跑。174 是可尝试数量、189 是解除建筑级阻塞的候选数量，都不是保证恢复数量。

## 修复完成后导出

先从最终结果创建成功源样本清单（不会批准 GT），保留未恢复 ID/原因：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.world_pipeline.prepare_source_ready_pairs \
  --sources qwen_edit_pano/data/gt_sources_14train3val_repaired_v1 \
  --pairs qwen_edit_pano/data/paired_gt_14train3val_v1 \
  --output qwen_edit_pano/data/paired_gt_14train3val_repaired_ready_v1
```

该清单只需创建一次；已存在时不要覆盖。确认统计结果后导出全部成功源（不限制 512 条）：

```bash
srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -u -m qwen_edit_pano.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_gt_14train3val_repaired_ready_v1 \
  --world-root qwen_edit_pano/data/gt_sources_14train3val_repaired_v1 \
  --train-limit 0 --val-limit 0 --threads 4 --require-all-pairs \
  --output qwen_edit_pano/data/gt_world_14train3val_repaired_v1
```

导出中断后，在这条命令末尾加 `--resume`。旧 `run_gt_14train3val.sh export/bundle/train-*` 指向旧目录，不能直接用于本次修复产物。下一步检查导出报告及修复样本投影，记录 review decisions；再为新 GT 创建 epoch006 oracle bundle、训练 full/constant 对照。不要自动把恢复候选全部批准为训练数据。

## 已执行 CPU 验证

- 两栋冲突建筑：原 region validator 确认只有列出的类别冲突；隔离后原 object-link / canonical validator 通过。
- 正常控制样本：重新融合的 sensor depth/valid 数组与旧数组完全一致。
- 两个失败样本重新检查通过；另一个覆盖率不足，仍拒绝，证明没有自动清空所有失败。
- 5 个单元测试：观测数量/方向不足拒绝、额外 region 错误拒绝、原报告保留、仅冲突对象屏蔽、未完成 catalog 不提交完成标记。
- 两条实际样本跑通修复编排及再次运行的哈希恢复：`7y3sRwLe3Va_0662f8da68b94d83a47af1e38f28f4e5`、`5q7pvUzZiYa_07304b100af64280bbaf510f06c7069c`。前者实际 catalog 的冲突实例 437 已关闭类别监督。
- 旧源构建代码的哈希签名保持一致。尚未运行整批修复，也未进行本次 GPU 训练。

小规模验证目录 `qwen_edit_pano/data/gt_repair_cpu_check_v1` 仅供诊断，不作为训练 GT 或正式修复完成记录。
