# GT-as-Text 结果与下一步

查看了原 baseline、规则 correct、新 llm_correct/llm_shuffled 和两条真实 ERP。新实验为助手校订后的 Qwen3-VL 文本，不是原始 LLM 输出。2 个 train 样本、seed 0；epoch006、GT 来源、种子、采样参数与原实验一致。新文本 297–326 tokens，实际编码长度低于限制。以下为人工视觉判断，不是统计显著性或量化几何评估。

- 浴室：baseline 没有补出床和马桶；规则 correct 有床和马桶，已有畸变。校订文本 correct 仍有床，但卫浴处出现异常织物堆、镜面/卫浴结构重复或不合理反射，比规则文本没有清晰的整体优势。镜像不能直接按独立对象重复计数。shuffled 把沙发、植物和厨房式设施带入浴室周围。
- 客厅：校订 correct 引入门、更多设备和植物，但 GT 中左右连续玻璃开口的布局被大幅改变，没有可靠恢复真实空间。原本的前景家具变形仍在。shuffled 同样改变墙、门和大型家具配置。
- 两例支持文字会改变生成内容，未证明逐对象数量、方向、尺度得到正确控制。仅凭两例不能宣判文本路线无效，也不能证明 Adapter 必然有效。暂不继续围绕这两张调文案。

当前 GT 资产核对：pilot 导出 11 条（9 train/2 val），4 条 flags；现有 reviewed oracle bundle 7 train/0 val，绑定 epoch001。expanded 导出仅 31 条 train（11 条 flags），无 EXPORT_COMPLETE；gt_world_available_v1 尚未生成。原始图像下载完成不等于 GT 构建/验收完成。

下一阶段优先完成已有原始资产对应的 GT 构建、导出和独立 val 验收；不等待覆盖全部建筑才开始实验。期间可用 7 条已验收 GT 在 epoch006 上做 200 步学习能力诊断。它不能代替独立验证。

## 可选并行：epoch006 小样本学习诊断

沿用现有 Encoder/Adapter，不改网络、loss、GT schema。不使用 epoch001 的旧 world checkpoint，不从两步 smoke 续训。已有 smoke 已验证接口；此处测试更多训练后是否出现可学习的结构收益。200 步是诊断预算，不保证收敛，未改善时先检查训练状态，不能直接判定方案无效。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export PYTHON_BIN=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --qos=normal --time=00:20:00 \
  "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.oracle_data \
  --gt qwen_edit_pano/data/gt_world_pilot_v1 \
  --review qwen_edit_pano/data/gt_world_pilot_v1/assistant_review_decisions_2026-10-02.json \
  --base-checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006 \
  --output qwen_edit_pano/data/oracle_epoch006_pilot_v1

"$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.oracle_train \
  --bundle qwen_edit_pano/data/oracle_epoch006_pilot_v1 --preflight

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  "$PYTHON_BIN" -u -m qwen_edit_pano.world_pipeline.oracle_train \
  --bundle qwen_edit_pano/data/oracle_epoch006_pilot_v1 \
  --condition full --purpose pilot --steps 200 --warmup-steps 20 \
  --activation-storage cpu \
  --output qwen_edit_pano/outputs/oracle_epoch006_full_fit_v1

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=04:00:00 \
  "$PYTHON_BIN" -u -m qwen_edit_pano.world_pipeline.oracle_inference \
  --checkpoint qwen_edit_pano/outputs/oracle_epoch006_full_fit_v1/checkpoint-step000200 \
  --split train --limit 7 --seed 0 --steps 28 --cfg 4 \
  --modes baseline correct shuffled \
  --output qwen_edit_pano/outputs/oracle_epoch006_full_fit_eval_v1
```

逐条执行，上一步成功后再执行下一步。未由助手启动训练/推理。96G 是 CPU 内存；旧 smoke 在大显存卡通过，不能据此保证 48G 训练成功。新目录保留旧输出。当前只记录训练图诊断，不声称 val 收益。

## 主线：准备独立验证后做等预算对照

已有 CPU 构建脚本支持续跑：设置 GT_PAIRS_ROOT=paired_gt_available_v1、GT_SOURCE_ROOT=gt_sources_available_v1、GT_EXPORT_ROOT=gt_world_available_v1（均在 qwen_edit_pano/data 下），执行 run_gt_full.sh build，成功后 export。导出成功仍需检查 flags、相机投影及建筑划分，不能自动把所有候选标为 approved。

固定 epoch006，新建已验收 train/val bundle。分别训练 constant 与 full，之后增加 observed；保持训练样本、种子、步数和优化设置一致。constant 是训练一个不接收样本世界状态的分支，用于排除额外训练本身带来的收益；shuffled 是同一个 full 模型推理时错配 GT，用于检查条件依赖，不能替代 constant。

验证同时检查 Local 保真、镜头外类别/数量/方向、布局、接缝；使用与 GT 对齐的透视投影辅助观察，区分 ERP 固有畸变与真实生成缺陷。固定评估样本与 seed，不用这两张训练图挑最佳设置。

现有 Adapter 以类别、精确中心、AABB 尺寸和有效性 mask 编码世界对象，但没有显式对象投影到 ERP token 的位置约束，空间对应仍需学习。因此它是待检验的方案，不是几何正确性的保证。若训练后仅改变类别而空间控制仍弱，再考虑单独增加投影/方向约束；不在首轮同时改网络和损失。Observer、Completion 联训继续后置。
