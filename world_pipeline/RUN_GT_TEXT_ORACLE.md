# GT-as-Text Oracle v1：零训练对照

本次仅新增gt_text.py、gt_text_oracle.py、test_gt_text.py及本文。没有改Observer、Completion、GT构建、Adapter、Panorama LoRA或正式配对训练。按用户要求，助手没有执行测试、prepare或推理；以下命令由用户手动运行。

固定checkpoint：qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006。GT来自已有gt_world_pilot_v1与assistant_review_decisions_2026-10-02.json，仅选approved且无数值flags；目前7 train/0 val。属于训练集上的信息利用诊断，不是独立验证集收益。旧epoch001 Adapter smoke生成图不能作这次epoch006基线，baseline必须重新生成。

## 实验定义

- baseline：Local RGB + checkpoint中的原普通任务prompt。
- correct：同一Local RGB + 同一任务prompt + 对应GT对象布局文本。
- shuffled：同一Local RGB + 同一任务prompt + 同split另一条GT的布局文本。固定shuffle_seed产生无自匹配的循环置换；供体来自整个已审核同split池，不只来自limit选择的目标。prompts.json记录每个source_id。
- 冻结基础模型、内部图文编码器、VAE、Panorama LoRA；不加载任何World Encoder/Adapter。每种prompt重新走原生Edit图文编码，禁止复用普通prompt的条件缓存。
- 1024×2048（取checkpoint配置）、28steps、CFG4、negative=' '、target-only circular padding。VAE FP32、模型BF16，与现有配对推理约定一致。每个样本/seed在三种模式下使用同一个CPU噪声generator seed，并重置torch RNG。

## 不先使用LLM润色

先用可追溯的规则转换，保留GT原类别，不让LLM纠错、增补颜色/材质或虚构关系。GT本身可能有原始类别噪声和不完整标注，规则文本不把这些噪声当成已经修复。

默认spatial格式：全部对象按类别、相机水平八方向、距离档聚合并记录数量；前方对应输入观看方向，左右对应输入图像，近<2m、中2–4m、远>=4m。它省略精确坐标、垂直位置、尺寸、房间和关系，因此是G_full的有损文本投影，不能声称传入完整原始JSON全部信息。聚合不丢对象，未知位置明确unknown。原始实例ID、完整图像caption、房间字母码不进入prompt。

可选metric格式：逐对象类别、局部center_xyz和camera-axis AABB全尺寸，保留至0.01m，缺失值unknown；x向右/y图像向上/z向前，y不一定是重力向上。没有把AABB当成语义朝向。此格式可能超出长度预算，默认报错而不截断；不是保证每条GT都能容纳。

plain prompt默认最多1024个processor tokenizer tokens；另在内部图文编码器forward之前限制图像+模板+文本总输入4096 tokens。当前本地diffusers 0.37.0实现的图文编码路径没有显式截断，max_sequence_length参数并不能替代这个检查。运行记录text_token_counts、实际encoded_lengths与峰值allocated显存。若版本改变，先核对原生编码路径，不能据参数名推断不会截断。

## 1. CPU测试及提示词检查（用户执行）

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export TXT_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python

"$TXT_PY" -m unittest qwen_edit_pano.world_pipeline.test_gt_text -v

srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=8G --qos=normal --time=00:20:00 \
  "$TXT_PY" -u -m qwen_edit_pano.world_pipeline.gt_text_oracle \
  --split train --limit 2 --seeds 0 --text-format spatial --prepare-only \
  --output research/audits/gt_text_epoch006_plan_v1
```

检查prompts.json和gt_projection.json，确认类别/方向/计数来自GT、没有完整图像caption或额外场景设定。prepare-only不会加载tokenizer/model，也不会生成图；token预算在GPU入口加载模型后检查。默认checkpoint、GT、review路径在入口中显式固定，可以通过同名参数覆盖。输出目录必须新建，不覆盖旧结果。

## 2. 两个样本、三个模式：6张图

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=04:00:00 \
  "$TXT_PY" -u -m qwen_edit_pano.world_pipeline.gt_text_oracle \
  --checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006 \
  --split train --limit 2 --seeds 0 --text-format spatial \
  --steps 28 --cfg 4 --offload model \
  --output qwen_edit_pano/outputs/gt_text_epoch006_2samples_v1
```

不要占用正在正式训练的GPU；按集群可用情况加--nodelist。96G是CPU内存请求，不是GPU显存保证。尚未实测该文本长度下显存峰值。如OOM，先提供失败阶段和日志，不直接改图像分辨率或偷偷删对象。可用--offload sequential显式换更慢的offload策略，需三组共同使用并写入新目录。

输出：run_config.json、prompts.json、gt_projection.json、text_token_counts.json、runtime.json；每个ID/local.png、gt_erp.png、seed-0/{baseline,correct,shuffled}.png及同名json。GT ERP RGB仅在该样本全部生成后保存作对照。PREPARED与COMPLETE分别表示准备和生成完成，不代表效果改善。不支持本入口中途resume；失败输出保留，重试换新目录。

## 3. 首轮无接口问题后，7条GT × 3 seeds × 3模式

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  "$TXT_PY" -u -m qwen_edit_pano.world_pipeline.gt_text_oracle \
  --checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006 \
  --split train --limit 7 --seeds 0 1 2 --text-format spatial \
  --steps 28 --cfg 4 --offload model \
  --output qwen_edit_pano/outputs/gt_text_epoch006_7samples_3seeds_v1
```

得到63张图。预算不是已测时长。metric格式是可选的第二文本表示实验：把text-format改metric、输出换新目录，其他参数保持一致。若超长，先查看计数决定是否设计另一种明确版本的压缩表示；不要只对正确文本截断而让shuffled保留更多对象。

## 判读与后续

检查Local保真、镜头外对象类别/数量/方向、整体布局、接缝和畸变；按样本列出correct相对baseline的改善/不变/退化，不只挑好图。shuffled检验对文字的依赖性，但场景类别/对象数量/长度也不同，不能单靠“错配变差”证明精确三维控制。不要只用全图PSNR或同一Observer自评下结论。

correct稳定改善且shuffled破坏布局，支持先走文本条件路线；还需在独立建筑验证，并测试预测G_full的不完美文本。少量train GT上的改善不足以宣布Adapter无必要。

如果没有改善，先判断GT映射、文本信息损失、长度和模型遵循程度；不能据一种prompt否定结构条件。之后可单独比较metric、类别-only或LLM忠实改写；LLM必须受规则事实约束，保存原文本/输出并核对数字和对象，没有新增/改写事实才是表达方式对照。

未来与Adapter比较时也要固定同一epoch006、同一GT子集、同一生成设置；现有epoch001的两步Adapter smoke不构成公平对照。本实验不替代、也不修改已有Adapter方案。

新增的独立 LLM 忠实改写对照及执行命令见 [RUN_GT_TEXT_REWRITE.md](RUN_GT_TEXT_REWRITE.md)。它复用本实验已有结果，只新增 llm_correct / llm_shuffled 两种模式；原入口不传新参数时仍运行原三组。
