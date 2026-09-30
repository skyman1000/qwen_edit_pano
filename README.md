# Qwen-Image-Edit-2511 panorama 迁移

本目录实现 **Edit-2511 权重的纯文本 ERP 生成微调**，保持原 caption → panorama 目标；不执行参考图编辑、不使用 target conditioning、不混合 perspective。旧目录和模型输出不作任何修改。完整模型验证尚未执行，须由用户运行下方 smoke 作业。

## 1. 原 Qwen-Pano 流程与实际正式参数

证据：`audit/baseline_training_config.json`、`baseline_pano_config.json`、`baseline_COMPLETE.json`，来自 `qwen_pano/outputs/pano_official_full_59263/checkpoint-epoch025/`。

实际基础模型为 Qwen-Image-2512，snapshot `25468b98e3276ca6700de15c6628e51b7de54a26`。正式训练已经完成 25 epochs / 64750 optimizer updates。数据为 10359 张 Matterport3D polished 全量官方训练集，没有 perspective，没有重新切分。沿用原 official-full protocol，包括其已有的 benchmark UUID overlap=1077，不将该 benchmark 宣称为独立测试集。

冻结文本编码器并预缓存 caption；冻结 VAE，posterior.sample 编码 target，按模型 mean/std 标准化；uniform timestep sampling，`x_t=(1-sigma)*z+sigma*noise`，目标 velocity 为 `noise-z`。只优化 attention Q/K/V/out 的 LoRA。

## 2. DiT360 panorama-only 对应关系

已阅读官方 `train.py`、`train.sh`、`src/dit360.py`、`src/data.py`，并通过下载官方 commit `3779fe7965473f6824994c663a0ae7a76bc7aafa` 的四个文件，确认与本机 DiT360 文件逐字节一致。参考的是 panorama 入口，不是 mixed 入口。

对应保留：bicubic ERP resize、水平翻转 p=0.5、水平随机 roll、全白有效 mask；边界 packed token 与其原始 RoPE 一并 circular pad；训练 forward 后裁剪；推理保留 padded scheduler state，最后裁剪再 decode。loss 为 flow MSE，加 cubemap/yaw velocity-space supervision；yaw 从 60/180/300 度抽取。前两轮只把 flow 加入总 loss，第 3 轮开始 `flow+0.5*cube+0.5*yaw`；前两轮照常计算辅助项并消费 RNG。几何函数直接复用现有 vendored DiT360 实现。

DiT360 原始后端为 FLUX/Lightning；当前 Qwen-Pano 已将后端替换为 Qwen native flow matching/Accelerate。本次保留当前 Qwen 版本，不重新引入 FLUX 的编码或训练框架。

## 3. Edit 接口差异与任务选择

官方模型索引是 `QwenImageEditPlusPipeline`。参考图像经 Qwen2.5-VL processor 进入文本/视觉编码，同时经 VAE mode 编码、pack 后拼接到 noisy target tokens。`img_shapes` 区分 target 与 reference。transformer config 的 `zero_cond_t=True` 使 target 使用 t modulation，reference 使用 0 modulation；不是简单改模型字符串即可。

Diffusers 0.37.0 的官方 Edit wrapper 开头访问 `image.size`，后续又引用仅在图像分支赋值的变量；其 prompt encoder 直接访问视觉输入字段。因此这里不把 `image=None` 传入该 wrapper，也不拿空白图/随机图/target 冒充参考图。

为了保持原训练任务，本实现显式采用 **无 reference token 的 Edit backbone T2I adaptation**：用 Edit snapshot 的 tokenizer、text encoder、VAE、scheduler、transformer，复用 QwenImagePipeline 的原 T2I 文本模板和 denoising loop。没有关闭 `zero_cond_t`，没有修改基础权重或新加可训练结构。只有一个 target grid 时，原生 forward 的 modulation index 全为 0，选择 t 分支；零时间条件分支没有 reference token 使用。

这保持了数据和条件生成目标，但改变了 Edit 模型原先的参考图条件分布；它不是官方 image-edit finetuning，也不保证保留编辑能力或获得相同生成质量。若要求真正的编辑训练，则需要合法的独立 source/instruction/target 数据，会改变本轮任务，未实施。

## 4. 最小必要修改

- 模型索引校验改为 EditPlus，并校验 `zero_cond_t=True`、`guidance_embeds=False`、`in_channels=64`。
- circular padding 后，native Edit `modulate_index` 必须覆盖 padded tokens：forward 接收 padded `img_shapes`；RoPE pre-hook 恢复原网格，post-hook 复制原边界频率。因此保留原 panorama 位置语义、文字 RoPE 和 persistent inference。
- 文本缓存使用 Edit snapshot 重新编码，继续原 T2I 模板、max sequence length=512；缓存和 checkpoint 均记录 `conditioning` 标识，防止误读旧缓存/adapter。
- 仅允许 repo-panorama，不允许 perspective manifest；所有可写训练/推理输出限制在本目录 outputs，缓存限制在本目录 cache。
- 新增显式 smoke 测试和冻结检查，不改变正式训练流程。

## 5. 文件与依赖

独立入口：`train.py`、`inference.py`、`cache_text.py`、`pipeline.py`、`circular.py`、`common.py`、`resume_compat.py`。

复用桥接：`data.py`、`losses.py`、`profiles.py`、`inference_state.py`。这些从原 `qwen_pano` 导入必要函数，间接使用原 `cached_data`、geometry 和 vendor；因此请保留同级旧目录。未复制数据集、图片、旧缓存、旧输出或无关研究模块。

验证：`validation.py`、`audit_static.py`、`test_interface.py`。脚本：`scripts/{cache_text,smoke_test,train_pano,inference_pano}.sh`。`audit/` 保存正式训练配置副本、源码哈希和官方接口元数据。源代码哈希包含复用依赖，用于严格 resume 校验。

## 6. 保持不变的参数

| 项目 | 实际值 |
|---|---|
| ERP / epochs | 1024×2048 / 25 |
| batch / accumulation / world size | 1 / 4 / 1；effective batch=4 |
| LR / optimizer | 5e-5 / AdamW |
| betas / weight decay / eps | (0.9,0.999) / 1e-5 / 1e-6 |
| LoRA | rank=64, alpha=64, dropout=0.05, gaussian init |
| targets | attn.to_q, attn.to_k, attn.to_v, attn.to_out.0 |
| LR schedule | repo-step，3237 updates linear warmup，随后 constant |
| timestep / loss weighting | native scheduler table / none（uniform） |
| padding / cube / yaw / seam | 1 / 0.5 / 0.5 / 0 |
| geometry start | epoch index=2，即第 3 轮 |
| seed / workers | 0 / 25 |
| data | 原 manifest、Arrow shards、HF revision、顺序及增强 |
| precision | frozen transformer bf16，LoRA fp32，VAE fp32 |
| gradient clipping / VAE tiling | 不启用 / 不启用 |
| dataloader | drop_last=True，persistent_workers=True，shuffle seed 同原实现 |
| checkpoint / log | 每 epoch 保存 adapter、optimizer、scheduler、RNG；完成后 COMPLETE；每 10 updates JSONL |
| inference | 当前 shell 脚本：28 steps，CFG=4，negative prompt=" "，seed=0/per-id，verbatim，LoRA scale=1，model CPU offload |

原 inference.py 默认 50 steps，但当前正式 inference_pano.sh 显式传 28；新脚本保留 28，Python 默认仍为 50。可用 `--steps` 显式覆盖。模型 scheduler config 与原 snapshot 已核对，见 `audit/interface_audit.json`。

## 7. 无法完全相同之处

基础权重/snapshot 必然不同；text embeddings 必须重算，不能未经核对复用 2512 缓存。Edit 保留原生 zero_cond_t 分支并增加 padding-aware shape 处理，但不注入参考图。随机 seed 和采样代码相同不代表不同模型的初始化/RNG消耗、输出可逐位相同。

不关闭 Edit 结构、不替换成 T2I 模型权重。任务仍为 text-to-panorama，但这个适配不能被表述为官方原生 Edit 任务或已证实质量等价。

## 8–9. 验证状态与参数检查

已经执行：Python AST、各 shell `bash -n`、CLI `--help`、profile 与正式配置逐项对比、manifest SHA256、原主要源文件未变检查。未运行任何 GPU 作业或完整模型加载，未宣称 finite loss、真实 LoRA 梯度、完整模型保存/加载或生成成功。

可选 CPU 小模型接口测试 `python -m qwen_edit_pano.test_interface` 也交由用户运行：检查真实 Diffusers zero_cond_t forward、batch=2、masked text、temporary/persistent padding 一致性、与无参考条件数学等价的 modulation 分支、gradient checkpointing backward。它不替代完整权重验证。

GPU smoke 使用原数据、1024×2048、batch=1、accumulation=4，共 8 microbatches/2 optimizer steps；首步 warmup LR=0，第二步检查 LoRA 参数确实变化。仅 smoke 将几何 loss 从第一批启用以覆盖其梯度，不修改正式训练 epoch schedule。

运行时断言：所有 trainable 名称包含 lora_，VAE/base 冻结；所有 LoRA 梯度存在且有限，至少一个非零；冻结参数无梯度；checkpoint 保存后故意扰动一个 adapter 参数，调用 Accelerate reload 并逐 tensor 比对。新进程从 Diffusers 导出 LoRA 重载，28 steps 生成一张 2048×1024 图片。`parameter_report.json` 给出实际 trainable/frozen 数量，未加载模型前不捏造计数。

只有训练检查通过才写 `smoke-checkpoint/SMOKE_COMPLETE.json`；推理也通过才写 `VALIDATION_COMPLETE.json`。smoke 不写伪造的 completed epoch，也不能用于正式 resume。正式训练使用全新 output。

## 10. 下载、缓存、验证、正式训练命令

所有命令在项目根目录执行。脚本沿用现有 debug / GPU4 / normal 资源设置，不自动提交任何任务。模型权重需要预先下载；默认脚本 offline=1。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
eval "$(conda shell.bash hook)"
conda activate qwen360
mkdir -p qwen_edit_pano/outputs qwen_edit_pano/cache

# 在可联网的节点执行；仅下载模型，不运行训练。
# 默认保存到 qwen_edit_pano/cache/hub，与旧缓存隔离。
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c 'from qwen_edit_pano.common import model_snapshot, DEFAULT_MODEL; print(model_snapshot(DEFAULT_MODEL))'

python -m qwen_edit_pano.audit_static
# 可选，无需权重下载的小型 CPU 接口测试：
python -m qwen_edit_pano.test_interface

# 先完成新的完整文本缓存。
sbatch qwen_edit_pano/scripts/cache_text.sh
# 等上一步成功结束再提交；包含训练和推理完整 smoke。
sbatch qwen_edit_pano/scripts/smoke_test.sh
```

确认 `qwen_edit_pano/outputs/smoke_<JOB_ID>/VALIDATION_COMPLETE.json` 成功后，正式命令：

```bash
sbatch qwen_edit_pano/scripts/train_pano.sh
```

正式输出为 `qwen_edit_pano/outputs/pano_official_full_<TRAIN_JOB_ID>/`。运行必须使用 diffusers==0.37.0，其他包沿用现有 qwen360 环境。使用本地模型目录时，给上述 cache/smoke/train/inference 作业统一设置 `QWEN_MODEL=/absolute/path/to/edit2511_snapshot`。检查点绑定 snapshot 的绝对路径，不可在中途切换。

## 11. 推理命令

替换 TRAIN_JOB_ID 为自己的正式训练作业号：

```bash
sbatch qwen_edit_pano/scripts/inference_pano.sh \
  qwen_edit_pano/outputs/pano_official_full_<TRAIN_JOB_ID>/checkpoint-epoch025
```

仅推理一张的试跑可追加 `--limit 1`。正式默认读取原 1092 prompts，输出为独立 `qwen_edit_pano/outputs/pano_inference_<INFER_JOB_ID>/`，逐图保存 PNG、sidecar、generation_config 和 generated.jsonl。自动核对尺寸，绝不以拉伸补成 2:1。

参考：[DiT360 panorama code](https://github.com/Insta360-Research-Team/DiT360/blob/3779fe7965473f6824994c663a0ae7a76bc7aafa/train.py)、[Edit 官方模型](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)、[Diffusers 0.37.0 transformer](https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/models/transformers/transformer_qwenimage.py)。
# qwen_edit_pano
