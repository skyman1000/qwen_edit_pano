# 精简构建结果与继续导出

BUILD_RESULT.json 已产生，说明构建遍历结束；退出1是整体存在未解决质量失败，不是最后一条 zsNo4HB9uLZ 的崩溃。

总1464：成功1047（train978/val69，13栋train/2栋val）；样本失败228；两栋canonical失败覆盖189条（train7y3sRwLe3Va 108条、val X7HyMhZNoso 81条）。成功仍只是源构建，未经过GT导出/验收。

228条中：225条包含source_camera_depth_mesh_disagreement，其中部分同时有coverage/整体深度偏差flags；另3条为WEAK_MATCH_REQUIRES_REVIEW。标签失败核对到具体源记录：7y3sRwLe3Va region20/object5 semseg='workbench'，mesh mapping1659='washbasin top'；X7HyMhZNoso region24/object3、4 semseg='doorfra,e'，mesh也为mapping1659。尚未判定应信任哪项，不猜标签、不关闭校验。

新增 prepare_source_ready_pairs.py 已运行，成功world检查点文件哈希及身份核对通过；创建 paired_gt_14train3val_source_ready_v1。selection.json保存全部417条排除原因、原清单/构建结果哈希、实际建筑覆盖。这是显式缩小的候选清单，不覆盖原14/3目标清单，也不是自动approved。目标14train/3val仍有一栋train一栋val待修复，不能声称达到原目标。

prepare_gt --preflight 已通过：978/978 train和69/69 val全部找到对应world。未执行渲染或GPU。

## 用户下一步

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
  -m qwen_edit_pano.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_gt_14train3val_source_ready_v1 \
  --world-root qwen_edit_pano/data/gt_sources_14train3val_encoder_v1 \
  --train-limit 0 --val-limit 0 --threads 4 --require-all-pairs \
  --output qwen_edit_pano/data/gt_world_14train3val_source_ready_v1
```

仅在已有该导出目录且需续跑时，原命令追加 --resume。不要用旧 run_gt_14train3val.sh export/bundle，它们指向旧目录并要求原覆盖。不要重新运行整个build来试图消除固定质量失败。

完成后读取新目录EXPORT_COMPLETE.json、failures.json、gt_manifest.jsonl和review.html。导出时仍可能发现局部坐标/可见阈值问题；不允许自动approve。验收后建立指向此新GT目录和其review_decisions.json的新oracle bundle，沿用epoch006及原full配对条件缓存。尚未开始训练。

对于剩余417条，单独诊断源标注与几何。若解决两栋问题，使用版本化修复、重新验证受影响源后再扩展候选；不把这次成功子集伪装成完整14/3。
