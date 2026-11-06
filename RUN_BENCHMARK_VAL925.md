# Native Edit vs panorama epoch015 / LoRA0.5（无GT世界状态）

用户已选定epoch015、强度0.5。本实验使用完整paired_full_v1/val.jsonl：925图、11栋建筑；训练集61栋建筑不重叠。它是开发验证集，已用于选checkpoint，不能再称为独立最终测试集。

两组均为Local RGB + 相同普通任务prompt，1024×2048，28steps、CFG4、seed0/per-ID，每图每组一次，共1850个ERP。不加载World Encoder/Adapter，不读取GT结构状态，也不按GT字段是否可用筛选成val168。GT ERP只在生成后读入作参考。

- native：原始Qwen-Image-Edit-2511，不加载LoRA，不安装panorama padding。
- finetuned：同一基础snapshot + epoch015 LoRA0.5 + 已训练的panorama padding。

这是完整panorama方案对原生模型的比较；没有单独隔离LoRA与padding的贡献。任务是局部图像补全，与论文text-to-panorama不能直接同表宣称优劣。

## 1. 两张GPU分别生成

从项目根目录提交；不指定节点，由Slurm分配空闲卡。数组task0=native，task1=finetuned，最多并发两张GPU；每组1GPU/4CPU/64G/72小时。不会由助手提交。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch qwen_edit_pano/scripts/benchmark_val925_generate.sh
```

输出固定在 `qwen_edit_pano/outputs/native_vs_pano_epoch015_scale05_val925_v1/`。

生成中断后原命令重新提交：仅复用本实验内有完整sample.json和输出图片、同参数/同manifest记录的样本，补生成缺失项到新attempt目录，已完成图片不覆盖。每个arm有锁，防止同时重复生成。若只需恢复一个arm：

```bash
sbatch --array=0 qwen_edit_pano/scripts/benchmark_val925_generate.sh  # native
sbatch --array=1 qwen_edit_pano/scripts/benchmark_val925_generate.sh  # finetuned
```

`NATIVE_COMPLETE.json` 和 `FINETUNED_COMPLETE.json` 均出现后进入下一步。

## 2. 固定评估清单并运行指标

只需CPU创建精确manifest，避免将Local/GT/拼图误读为生成图：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.benchmark manifests

sbatch qwen_edit_pano/scripts/benchmark_val925_evaluate.sh
```

评估仍是两个并发单GPU数组任务，分别计算两组，复用旧 `DiT360/.venv/bin/python` 和本地权重，不重装环境。

旧11项：FID、FIDclip、FIDpole、FIDequ、FAED、Places365 IS、CS、Q-Align quality/aesthetic、BRISQUE、NIQE。

沿用之前参数：裁掉每极0.125；cube面512；FAED高512、pil-bicubic；Places365 ResNet18、IS splits1；CLIP ViT-B/16；本地Q-Align；batch1。IS splits1不是多个生成seed。

额外LPIPS/SSIM/DISTS：pyiqa 0.1.15，LPIPS Alex v0.1，SSIMc RGB/no-downsample，DISTS；RGB FP32 [0,1]、无autocast；复用旧512像素、90度FOV的py360convert双线性投影。报告六面均值、各面、排除front的五面均值。五面不等于精确不可见区域；Local固定前向投影指标仅作辅助诊断，不作为整体全景质量排名。

指标报错时保留旧目录，可只重跑失败arm并指定新报告名：

```bash
sbatch --array=1 qwen_edit_pano/scripts/benchmark_val925_evaluate.sh \
  --eval-tag evaluation_retry1
```

汇总前要确保两个arm在同一个eval-tag下都有成功报告；重跑同一tag遇到已有成功报告会复用，未完成11项报告要求新tag。三项配对指标支持按样本恢复。

## 3. 汇总

两个评估任务都成功后：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.benchmark summarize
```

如使用其他报告名，附加 `--eval-tag evaluation_retry1`。

主要产物：

- `plan.json`：925个ID与固定配置。
- `generation/`：两组生成输出及attempt清单。
- `manifests/native.jsonl`、`finetuned.jsonl`、`reference.jsonl`。
- `evaluation/native/`、`evaluation/finetuned/`：旧11项及逐图结果。
- `paired_metrics_512_cube/native/`、`finetuned/`：LPIPS/SSIM/DISTS，含逐图六面分解。
- `comparison.csv/json`：两组指标与差值。

## 协议解释与验证状态

论文来源：https://arxiv.org/html/2510.11712v1#A3 。本地旧协议是独立复现，裁剪量、部分权重/后端等未验证与论文完全相同；本轮保持此前配置，不为追逐论文数字改参数。参考集为本轮925张polished GT，不混用旧1092 stitched参考集。

SSIM/LPIPS/DISTS的GT像素/特征对应指标会惩罚合理的隐藏世界多解和全局相机差异，不能取代整体观感、接缝和球面几何检查。CS采用实际固定任务prompt，不能当作详细场景语义遵循评分。925张的小样本分布距离仍需谨慎解释，后续独立test应在配置固定后另做。

CPU已验证：925个ID/11栋划分；模拟生成的native/tuned命令、单样本完成后恢复、缺失项判定、清单生成；已有8对真实图片在旧11项入口dry-run均ready。额外三项沿用先前审计过的投影和相同pyiqa模型配置。没有执行本轮正式GPU生成或指标计算；ready不是已计算成功。

清理列表见 `qwen_edit_pano/audit/cleanup_2026-10-20/REVIEW.md`；本轮没有删除文件。
