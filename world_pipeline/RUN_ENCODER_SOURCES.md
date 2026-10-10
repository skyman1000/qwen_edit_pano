# Encoder 所需源字段精简构建，2026-10-09

旧14/3构建快照：900完成、200样本质量失败、189因两栋region校验失败阻塞、175尚未报告。总1464。完成是源世界状态，不是导出/验收通过。之后数量会变化。

新入口 build_encoder_sources 读取旧根目录并写入独立 gt_sources_14train3val_encoder_v1。复用已完成 canonical/alignment/geometry/world（软链接）；缺失 world 使用 lean_world。没有修改旧 expand_gt_sources、qwen_pano 构建器、质量阈值或正在运行的脚本。旧源输入与代码签名仍检查。已完成阶段不会为删除无用字段而重写。

新增 encoder_source_catalog_v1 只提供现有 prepare_gt 实际需要的对象类别、源OBB、有效性标记、相机及最小room元数据。没有无遮挡对象投影、可见比例、启发式关系、语义朝向、地板/天花板推导；省略的高度明确为null，不用0伪装。它不是旧完整world schema的替代品，不应用于需要这些字段的其他消费者。

相机朝向、深度/mesh校验及后续Local/ERP全场景遮挡投影保留。prepare_gt仍重新计算可见实例，生成local_evidence、bbox2d等；这些字段支持筛选、observed对照与验收，不能为了提速用包围盒猜测。

CPU检查：三个建筑 5q7pvUzZiYa / V2XKFyX4ASd / ur6pFq6Qu1A 各一个已完成样本，分别411/474/569对象。精简源目录中每个保留对象字段与旧结果精确相同，保留camera/room字段精确相同。语法检查通过。未执行整批构建、端到端导出或GPU实验；没有声称实测加速倍数。旧日志world阶段中位间隔约46秒只是优化动机，不是新的耗时。

## 执行

先停止旧构建作业（上次确认job65087）；不要停止Panorama训练。若旧作业已结束，不需取消。新入口会锁住旧源根目录，旧writer还在运行时拒绝启动。

```bash
squeue -j 65087
# 确认仍是这次build后执行：
scancel 65087
```

然后用户运行（不用设置任何路径环境变量）：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -u -m qwen_edit_pano.world_pipeline.build_encoder_sources
```

中断后同一命令续跑。新产物软链接依赖旧源目录，不能删除旧根目录来腾空间。

```bash
python -m qwen_edit_pano.world_pipeline.build_status \
  --root qwen_edit_pano/data/gt_sources_14train3val_encoder_v1
```

原先的 run_gt_14train3val.sh build/export 仍指向旧全量字段目录；切换后不要混用。精简构建仍会保留两栋region失败及已有几何flags，结束时可能返回非零；BUILD_RESULT.json会保留成功结果及失败记录。不要因此重新下载所有数据或循环盲目重跑。

下一步先处理这些失败：区分类别标注冲突、解析问题和几何质量不足；修复需有证据，不能取消校验。之后导出应显式 --world-root qwen_edit_pano/data/gt_sources_14train3val_encoder_v1 并使用新的输出目录。未解决时不能沿用全量覆盖/14+3验收完成的表述。当前prepare_gt的诊断可视化仍保留，本次优化针对实测较慢的world派生计算，尚未优化导出存储。
