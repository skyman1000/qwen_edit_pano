# Oracle smoke重载检查与下一步

后续用户要求补齐全量GT：请优先使用 `RUN_GT_FULL.md` 的下载→源状态构建→可恢复导出流程。本页251/39命令仅转换已有旧状态，不能生成缺少的源状态。当前磁盘余量也不足以直接全量执行，详见新说明。

已检查`qwen_edit_pano/outputs/oracle_full_smoke_reload_v2`的完成记录和两组baseline/correct/gt_erp图像，以及训练SMOKE_COMPLETE记录。

- 训练记录：零门控基线一致、base/LoRA冻结、Encoder梯度、参数更新、保存重读均通过。实际训练GPU总显存约139.8GiB、峰值约41.43GiB，不等于48G卡已验证通过。
- 重载：2张训练样本、baseline/correct均完成；correct使用各自GT状态。结构分支产生可见变化，没有整体图像崩溃。
- 客厅：画框、门窗和沙发有变化，但玻璃门/壁炉等整体布局仍偏离GT。
- 卫生间：门板、地面和细节变化，但没有恢复GT的卧室、马桶等布局；镜子/洗手池比例也仍不符。
- 结论：工程接入smoke通过；两步、两张训练样本不足以证明结构收益或否定方案。绝对残差RMS没有对应hidden RMS分母，不能单凭末层数值大断言失稳。

2026-10-06 CPU preflight通过：paired_full_v1的7521 train/925 val中，已有旧世界状态身份匹配251 train/39 val。这是候选数量，不是验收数量；下载RGB全量不等于GT状态全量派生完成。

## 当前执行：导出已有世界状态的全部匹配候选

不改旧目录，不改变可见性标准，不重新标注，不运行Observer。不申请GPU。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export PYTHON_BIN=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=04:00:00 \
  "$PYTHON_BIN" -u -m research.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_full_v1 \
  --train-limit 0 --val-limit 0 --threads 4 \
  --output qwen_edit_pano/data/gt_world_expanded_v1
```

时间是申请预算，不是完成时长保证。输出已存在时入口拒绝覆盖；不要删除旧结果后盲目重跑。

完成后检查EXPORT_COMPLETE.json、failures.json、gt_manifest.jsonl、review.html和各样本bridge.json。出现flags不等于导出失败，也不自动允许进入训练。

## 导出后的顺序

1. 汇总失败、建筑覆盖、对象数量与告警；对齐错误与Local/ERP采样密度问题分开处理。当前Local与ERP均使用16像素阈值，二者采样密度不同，不能将Local达标但ERP未达标直接等同几何错误，也不能未经检查删除告警。必要修改采用独立策略版本/输出，保留旧数据可追溯性。
2. 检查投影和来源后创建新review及bundle，确保有合格独立val；39个val候选仅来自有限建筑，不当作充分泛化证据。字段类别/坐标/尺寸/有效掩码沿用已定义合同。
3. 固定同一个epoch001 Panorama LoRA，冻结base和LoRA，分别训练constant、observed、full，保持样本/初始化seed/预算相同。训练预算根据实际合格数量确定；不把7条pilot的200步直接作为正式配方。
4. 在同一独立val以相同Local、普通文本与seed比较baseline及各分支。检查Local保真、镜头外类别与布局、接缝；full分支增加shuffled状态作为条件依赖诊断。constant控制额外参数收益，observed与full比较镜头外GT信息收益。observed仍是可见对象的GT三维信息，不是Qwen3-VL预测。
5. 有可信GT收益后再推进Observer未知几何估计和Completion。若full无收益，先诊断表示/训练/注入，不把问题传递给预测模块。
