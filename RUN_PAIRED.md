# Local RGB → ERP：已有数据先跑通

本入口只做 panorama-only 条件生成；不修改 qwen_pano、World Adapter、Qwen3-VL 或因果模块。用户手动执行 GPU 命令。真实模型训练/推理是否成功，以 smoke 与 inference 输出为准，不能用 CPU 测试替代。

## 0. 本次实现与原方案的差异

- 原生 Edit 图像条件：局部图分别进入 Qwen2.5-VL 图文编码和 reference VAE；只对目标 ERP 加噪与监督。
- target VAE fp32 posterior.sample；reference VAE fp32 posterior.mode，参考像素先按原生 prepare_latents 转 bf16，再转 fp32 编码；编码后参考 latent 转 bf16。缓存与推理保持一致。
- circular padding：训练临时补目标 token，推理持久补目标 scheduler state；不补参考图，不改变其原生 RoPE；正确处理 zero_cond_t 的目标/参考边界。
- 固定 skybox2、同步水平翻转；不采用旧的连续随机 ERP yaw，不扩大每 epoch 样本数。翻转按 seed/epoch/id 确定，便于 epoch 边界恢复。
- 实际抽查发现 polished ERP 比 face2 有固定水平朝向偏移。prepare_pairs 会搜索并校准 GT 的水平旋转/镜像，记录每行结果。已抽查 8 train + 2 val 全部为无镜像、roll_fraction=0.75（等价向左滚动1/4周），匹配分数0.975–0.993。不能把这个小样本结论冒充全量验证。配对注册只用于监督坐标，不给模型输入 GT。
- score<0.65 或方向区分 margin<0.03 的候选写入 rejected，不进入训练清单；这些是数据质量筛选阈值而非新训练损失。需人工抽查预览，不确定时不要下调阈值强行通过。
- 简单固定任务指令，旧全景 caption 不进入条件分支；没有新增损失，也没有已验证的性能改进声明。
- 1024×2048、25 epochs、B1、accum4、LR5e-5、rank/alpha64、dropout0.05、AdamW与原配置一致；padding1、cube/yaw0.5、seed0、原 timestep 表和采样方式保持。
- warmup 已按用户选择改为默认2350步（--warmup-steps 2350），然后恒定；对应7521条、25epochs、accum4的原5%预算公式。小子集不会达到原来的64750次更新；32条×25epochs只有约200次更新，一直处于 warmup，所以小样本试跑不能用于判断收敛质量。
- 正常训练前两个 epoch 不加几何项，从第三个加入；smoke 有意立即启用几何分支，以8个 microbatch 检查完整损失。
- 原生图文模板与视觉 token 长度由 EditPlus 管理，不套旧纯文本512长度缓存。缓存与输出独立；模型只读已配置的 Hugging Face 缓存。

## 1. 工作目录与资源

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export PYTHON_BIN=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python

# 沿用已有脚本的 debug / GPU3，按实际可用节点修改这一行即可。
# 这些参数不代表已测得真实训练显存需求；CPU offload 仅用于推理。
GPU=(srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=96G)
```

所有输出路径需使用新目录。失败的缓存可以原配置重跑；训练新跑若目录已非空，请使用新输出名。训练只支持单GPU、batch1；不能直接增加GPU数改变有效batch。

## 2. 更新可用清单，冻结一个小子集（CPU）

```bash
"$PYTHON_BIN" qwen_edit_pano/prepare_official_splits.py

bash qwen_edit_pano/scripts/paired.sh prepare \
  --output qwen_edit_pano/data/paired_pilot_v1 \
  --train-limit 32 --val-limit 4 --preview-limit 32
```

train-limit 是待筛选候选数量，可靠配对数可能较少；以 pair_report.json 为准，smoke 至少需要8条训练样本。清单固定后不会随后台下载变化。

查看目录中的 train_preview_*.jpg、val_preview_*.jpg：上排是局部图、校准后 GT 正前方和校准后 ERP；下排是校准前 GT。输入与上排 GT 正前方应方向一致、物体对应；细小 JPEG/抛光差异可能存在。

`heldout.jsonl` 始终包含完整官方 val+test 的建筑/样本身份，不仅是选出的几个验证样本。训练会检查建筑和 panorama UUID 不重叠。

## 3. 可选 CPU 接口自检

```bash
bash qwen_edit_pano/scripts/paired.sh test
```

使用小型随机 Diffusers 模型测试，不加载2511大模型、不占GPU、不下载模型。已验证：reference/text RoPE、temporary/persistent 一致性、参考图对目标的影响、checkpointed backward、LoRA 更新与保存重载、cube/yaw 分支。

## 4. 缓存局部图条件（GPU）

```bash
"${GPU[@]}" bash qwen_edit_pano/scripts/paired.sh cache \
  --manifest qwen_edit_pano/data/paired_pilot_v1/train.jsonl \
  --output qwen_edit_pano/cache/paired_pilot_v1
```

每条缓存原图与水平翻转两种条件，内容是原生图文 embeddings、mask、干净参考 latent；不缓存 GT 作为条件。输出 complete.json 后再训练。不要使用 qwen_pano/cache 或旧 text-only cache。

## 5. 最小真实训练验证（GPU，自动停止）

确认配对预览后执行；--alignment-reviewed 是你对本次数据预览检查的记录，不会自动替你断言所有数据正确。

```bash
"${GPU[@]}" bash qwen_edit_pano/scripts/paired.sh smoke \
  --manifest qwen_edit_pano/data/paired_pilot_v1/train.jsonl \
  --heldout-manifest qwen_edit_pano/data/paired_pilot_v1/heldout.jsonl \
  --condition-cache qwen_edit_pano/cache/paired_pilot_v1 \
  --output qwen_edit_pano/outputs/paired_smoke_v1 \
  --workers 0 --alignment-reviewed
```

此命令不训练25个epoch；两次 optimizer step 后自动返回。accum4通常对应8个microbatch。第一步LR=0是warmup行为，第二步检查LoRA参数实际改变。workers0只用于调试。

检查：
- outputs/paired_smoke_v1/parameter_report.json：仅 LoRA 可训练，基础模型和 VAE 冻结。
- outputs/paired_smoke_v1/smoke-checkpoint/SMOKE_COMPLETE.json：finite loss/几何项、梯度、参数更新和检查点重载结果。
- 检查点包含 adapter_resume.safetensors、Diffusers LoRA、optimizer/scheduler/RNG；不保存冻结大模型。
- 没有 SMOKE_COMPLETE.json 就不能称 smoke 通过；成功也不等于图像质量好。

## 6. 重新加载 checkpoint，验证推理（GPU）

```bash
"${GPU[@]}" bash qwen_edit_pano/scripts/paired.sh infer \
  --checkpoint qwen_edit_pano/outputs/paired_smoke_v1/smoke-checkpoint \
  --manifest qwen_edit_pano/data/paired_pilot_v1/val.jsonl \
  --output qwen_edit_pano/outputs/paired_smoke_val_v1 \
  --limit 2
```

默认28步、CFG4、negative空格、seed0按ID派生、1024×2048。每条输出 input.png / generated_erp.png / gt_erp.png。GT 只在生成结束后读取保存供对照，不传给模型。COMPLETE.json只表示生成和尺寸检查完成；接缝、极区和局部保真需要视觉评估。两步训练预计不会明显改善原生质量。

同条件原生基线（不用 checkpoint，也不安装全景padding）：

```bash
"${GPU[@]}" bash qwen_edit_pano/scripts/paired.sh infer \
  --manifest qwen_edit_pano/data/paired_pilot_v1/val.jsonl \
  --output qwen_edit_pano/outputs/paired_native_val_v1 --limit 2
```

自己的图片：用 --image /absolute/path/local.jpg 替换 --manifest。训练目前局限于水平90度 skybox图；任意FOV/俯仰/真实相机视差的泛化尚未验证。

## 7. smoke + inference 通过后，使用当前全部可配对训练样本

仍然不使用测试集训练或调参。先更新可用清单，再创建新数据/缓存/输出目录；不要修改 pilot 或正在训练的 manifest。

```bash
"$PYTHON_BIN" qwen_edit_pano/prepare_official_splits.py
bash qwen_edit_pano/scripts/paired.sh prepare \
  --output qwen_edit_pano/data/paired_available_v1 \
  --train-limit 0 --val-limit 0 --preview-limit 32

"${GPU[@]}" bash qwen_edit_pano/scripts/paired.sh cache \
  --manifest qwen_edit_pano/data/paired_available_v1/train.jsonl \
  --output qwen_edit_pano/cache/paired_available_v1
```

检查新预览与拒绝列表后，手动开始25epoch实验（此命令不会由助手执行）：

```bash
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=26 --mem=96G \
  bash qwen_edit_pano/scripts/paired.sh train \
  --manifest qwen_edit_pano/data/paired_available_v1/train.jsonl \
  --heldout-manifest qwen_edit_pano/data/paired_available_v1/heldout.jsonl \
  --condition-cache qwen_edit_pano/cache/paired_available_v1 \
  --output qwen_edit_pano/outputs/paired_available_v1 \
  --workers 25 --alignment-reviewed
```

需要更长作业时，按集群规定更换分区/时限；不要假设 debug 分区能容纳完整训练。代码本身不请求资源或提交任务。

训练日志 train_log.jsonl；每个epoch保存 checkpoint-epochNNN，COMPLETE.json 最后写入。查看第3个epoch以后 cube/yaw 已进入总损失。验证集不会自动运行昂贵推理，可手动对指定 checkpoint 用上述 infer 命令评估，选择模型时只看验证集。

恢复示例：重用原训练命令和原参数，附加
```text
--resume qwen_edit_pano/outputs/paired_available_v1/checkpoint-epoch003
```
仅允许完整epoch边界恢复；smoke checkpoint 不作为正式训练起点。代码、模型、数据、缓存、workers、总epochs或关键参数变化会拒绝resume。不同实验从基础模型和新输出开始。

全部下载完成后再次建立新 available 清单，实际候选上限为官方train 7523 / val925；质量筛选后的数量看报告。完整10359包含test1911，不能全部拿来训练又声称独立测试。历史全量训练checkpoint也不能充当未见这些测试建筑的公平基线。

## 2026-10-02：改用 sbatch 后台提交

全量配对训练使用新增的 `scripts/train_paired_pano.sh`，不是旧的纯文本 `scripts/train_pano.sh`。提交后不依赖SSH连接；没有由助手提交任务。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch qwen_edit_pano/scripts/train_paired_pano.sh
```

默认GPU1、normal、72小时、1GPU、26CPU、256G内存（内存申请沿用train_pano.sh），workers25、warmup2350，其余训练参数不变。日志为 `qwen_edit_pano/outputs/qwen-edit-paired_JOBID.log`，新训练目录为 `qwen_edit_pano/outputs/paired_full_JOBID`。如需其他节点，在脚本路径前加 `--nodelist=GPU3`。目录名随作业ID变化，不覆盖旧实验。

恢复时复用原训练目录，仅选择含COMPLETE.json的最新完整epoch：
```bash
sbatch qwen_edit_pano/scripts/train_paired_pano.sh \
  --resume qwen_edit_pano/outputs/paired_full_JOBID/checkpoint-epoch003
```

JOBID和003替换为实际值；如恢复此前srun任务，可传原paired_full_v1目录下的完整检查点。仍只支持epoch边界恢复，未完成epoch需要重跑。

2026-10-02实际只读查询：`scontrol show partition debug` 返回 MaxTime=3-00:01:00；`sacctmgr -n -P show qos long format=Name,MaxWall,MaxTRESPerUser%100,Flags%100` 返回 `long|7-00:01:00|cpu=25,gres/gpu=1,node=1|`。long没有PartitionTimeLimit标志，不能覆盖当前分区时限。若管理员后续更改配置，可重新核实；不要直接把本脚本改成long/7天，26CPU也超过long的25CPU上限。

## 全量缓存完成后的更新（此前 srun 用法）

用户确认采用 warmup=2350。缓存不依赖warmup，无需重建；旧smoke仍是历史接口验证，不续训它。

CPU全量检查：
```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m qwen_edit_pano.audit_paired_ready \
  --data qwen_edit_pano/data/paired_full_v1 \
  --cache qwen_edit_pano/cache/paired_full_v1 \
  --output qwen_edit_pano/audit/paired_full_v1_ready.json
```

正式训练建议normal QOS、显式71小时时限（--time=2-23:00:00）、1GPU、26CPU配workers25。long QOS的7天不会覆盖debug分区约3天的上限；long还限制25CPU，不适合照抄26CPU命令。

训练日志新增epoch_elapsed_seconds/estimated_epoch_seconds，便于开始后检查单epoch耗时；只是粗略估计，不是完成保证。仅epoch结束后保存，作业时限或中断会丢弃当前未完成epoch，恢复从最近COMPLETE检查点继续。当前未实现任意step恢复。

全量正式训练（人工检查预览后使用 --alignment-reviewed）：
```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodelist=GPU1 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=26 --mem=96G \
  --qos=normal --time=2-23:00:00 \
  bash qwen_edit_pano/scripts/paired.sh train \
  --manifest qwen_edit_pano/data/paired_full_v1/train.jsonl \
  --heldout-manifest qwen_edit_pano/data/paired_full_v1/heldout.jsonl \
  --condition-cache qwen_edit_pano/cache/paired_full_v1 \
  --output qwen_edit_pano/outputs/paired_full_v1 \
  --epochs 25 --workers 25 --warmup-steps 2350 --alignment-reviewed
```

checkpoint-epoch003/COMPLETE.json存在后，验证示例（请安排在GPU空闲时）：
```bash
srun -p debug --nodelist=GPU1 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/paired.sh infer \
  --checkpoint qwen_edit_pano/outputs/paired_full_v1/checkpoint-epoch003 \
  --manifest qwen_edit_pano/data/paired_full_v1/val.jsonl \
  --output qwen_edit_pano/outputs/paired_full_val_epoch003 \
  --limit 8 --steps 28 --cfg 4 --seed 0
```

时限中断后，保留原训练命令全部参数，附加 --resume qwen_edit_pano/outputs/paired_full_v1/checkpoint-epochNNN；NNN须替换为最后一个有COMPLETE.json的完整epoch。25是总epoch数，不是额外训练25轮。不要使用smoke检查点续训正式实验。
