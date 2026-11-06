# 原生Edit / 全景LoRA / 全景LoRA+GT状态：923张共同验证集

本轮使用已完成A组10000步Adapter：

`qwen_edit_pano/outputs/object_world_id_A_dual_full_v1/run-1792820440973994739/checkpoint-step010000`

其绑定epoch015/LoRA0.5、20/40/59层、width256、类别ID编码、groupA/binding_weight0。它是全量训练产物；不自动换成旧Fourier750或新Spatial200小规模校准权重。

Adapter数据包为7390 train / 923 val，11栋验证建筑。原生/全景比较是925 val。缺少的两个ID写入新plan.json；三组统一923张，不能把旧925分数直接拼到新GT行。923条Local来源、ERP Arrow身份、mirror/roll均已与原925清单核对一致。

生成只补GT组。原生和epoch015/0.5图片直接引用已有完成结果。GT组输入是相同Local RGB、相同普通任务prompt和该样本真实GT对象类别/三维几何状态；GT图像或实例mask不作为模型条件。此为Oracle实验，不代表预测状态部署性能。World在正负CFG两支相同。

## 1. 双卡生成GT组

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch qwen_edit_pano/scripts/benchmark_gt_generate.sh
```

数组0/1各申请1GPU、4CPU、64G、72小时，最多并行两卡，分别462/461张。923张各生成一次，不做多seed。1024×2048、28steps、CFG4、LoRA0.5，seed0/per-ID，与已有两组完全一致。

两卡独立生成，不需要模型切分。可同节点也可不同节点，实际并发取决于资源。输出：

`qwen_edit_pano/outputs/native_pano_gt_A10000_val923_v1/`

其中 `generation/shard0_attempt001`、`shard1_attempt001` 保存GT组；每样本为local.png、correct.png、correct.json、gt_erp.png。只生成correct，不额外生成baseline或shuffled。

断点恢复：重新提交相同脚本，已完成样本保留，只补缺失到新attempt。也可只恢复一个分片：

```bash
sbatch --array=0 qwen_edit_pano/scripts/benchmark_gt_generate.sh
sbatch --array=1 qwen_edit_pano/scripts/benchmark_gt_generate.sh
```

SHARD0_COMPLETE.json和SHARD1_COMPLETE.json均出现后进入评估阶段。

## 2. 创建三组共同清单，评估

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.benchmark_gt manifests

sbatch qwen_edit_pano/scripts/benchmark_gt_evaluate.sh
```

manifests阶段引用923张原生和923张全景LoRA的旧图片，合并GT两分片。它还将已完成925组的Local/Cube逐图记录过滤为923条/1846个区域记录，保存到新目录，不覆盖旧报告。两组的这些逐图数值无需重新推理计算，重新聚合即可。

评估是数组0=native、1=finetuned、2=gt_state，最多并发2个任务，每任务1GPU。三组均按相同923张重算集合级/旧11项协议，以确保FID等集合指标可比。GT组补算Local/Cube三项。

保留此前协议：CLIP ViT-B/16、Places365 ResNet18/splits1、FAED高512/PIL bicubic、每极裁0.125、cubemap512、QAlign本地权重、batch1；LPIPS Alex v0.1、SSIMc RGB/no-downsample、DISTS，FP32 RGB[0,1]，512像素90度固定投影。参考GT像素不进入生成。

## 3. 三方法汇总

等三个评估任务成功后：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.benchmark_gt summarize
```

输出 `comparison_three_methods.md/csv/json`，明确列出三方法11项，以及Local和Cube各自的LPIPS/SSIM/DISTS；另给Nonfront/Back三项。Native vs panorama用于整体方案比较；panorama vs GT用于衡量固定基线增加GT状态通路的效果。三方法GT参考一致，不能把GT有条件的优势称为已实现可部署的预测状态效果。

指标失败时保留报告，使用新tag重试；三个arm均应使用相同tag后汇总：

```bash
sbatch qwen_edit_pano/scripts/benchmark_gt_evaluate.sh --eval-tag evaluation_retry1
# 完成后：
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.benchmark_gt summarize --eval-tag evaluation_retry1
```

## 已有Local/Cube分数与时间依据

旧925张的Local/Cube已经算过，位于 `outputs/native_vs_pano_epoch015_scale05_val925_v1/paired_metrics_512_cube/{native,finetuned}/summary.csv/json`。此前汇总未完整展开Local表，并非指标遗漏。本轮会在923共同子集上重新汇总。

实际A10000 living_probe的一张correct图generation_seconds=242.37秒。两卡462/461张按此速度约31小时，建议预留30～40小时、不含排队；不是全923速度保证，运行日志会持续输出耗时和剩余估计。先前925两组旧11项实测各约43分钟，Local/Cube各约22～23分钟；本轮两基线配对指标复用缓存，三组两卡评估可按约1.5～2.5小时规划，不含排队/加载差异。

CPU检查：真实权重严格加载、5,309,699参数；923条数值对齐与数据身份检查；精确IDs与既有GT推理入口preflight；两分片/按ID种子/已完成样本恢复的模拟检查；三方法923条清单和1846条旧区域记录复用模拟检查。GT图像仍需由用户执行生成，未声称本轮GPU成功；未下载或提交任务。
