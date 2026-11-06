# 从零训练：逐图场景描述＋World，去掉binding

独立实验 `caption_fresh_world_nobinding_v1`。所有GPU和提交由用户执行；助手只完成CPU准备/检查。本目录复用原有grounded模型结构，不更改旧A/C训练器、权重或结果。

## 训练定义

| 项目 | 本实验 |
|---|---|
| 冻结基座 | 原Edit2511＋Panorama epoch015 LoRA，scale0.5 |
| 训练集 | 现有7390条，单张原skybox face2＋相机局部米制对象状态；polished ERP作为目标 |
| 文字条件 | 固定扩展指令＋该图polished Arrow caption，约20–63词；每条训练样本都提供描述 |
| 类别 | 保留冻结Qwen名称特征；语义投影、几何网络、融合、Q/K/V/out等可训练参数重新随机初始化 |
| 空间接口 | 保留当前输入3D框投影得到的候选attention；零null K/V；20/40/59层 |
| 初始化 | 新Encoder/Adapter＋新Adam；只有残差gate零初始化；不继承旧C/A或smoke训练后的参数/优化器 |
| Loss | `flow + 0.5 cube + 0.5 yaw + 0.25 region`，从第一步启用；无binding标签/损失 |
| 精度/内存 | 冻结基座BF16，World及Adam FP32，gradient checkpointing＋activation CPU storage；微批1 |
| 双卡 | 一个模型的两个副本同步World梯度，每卡累积2图，全局batch4；不是两组独立实验 |
| LR | 1e-4，正式训练前100步线性warmup，其后恒定；目标步数不影响恢复后的调度 |
| 保存 | 每100步及目标终点；完整保存World、Adam、配置和完成标记 |
| 默认目标 | 2000步、8000次样本曝光，约1.083轮；不是收敛承诺。10000步约5.413轮 |

caption从实际目标Arrow的文件/batch/row读取，不按相似文字匹配；原caption完整保留在输出根目录 `CAPTIONS.jsonl`。没有重标注、下载、ZIP CRC扫描或文件哈希。

文字中的位置短语需要处理：原polished ERP会绕yaw对齐到Local，因此其描述里的“on the left/right”“in the center”等未标定图像方向不能直接当作当前相机方向。`data.scene_caption`保守去掉这些短语，保留对象、风格和物理关系（如“in the center of the room”）；不使用LLM改写，不新增属性。最终实际prompt保存在每个新缓存的metadata中。原始CAPTIONS索引不改。

这仍不能保证方向控制：候选mask只限制新增World分支的直接写入，region只加权去噪误差。去掉binding是在简化目标，不是已证实可以根治涂抹。

## 推荐提交

在项目根目录，用户自行替换节点和qos：

```bash
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_caption_world.sh 2000
```

脚本分配同一节点两张GPU、8 CPU、128G主存，默认24小时时限。默认没有固定nodelist。可用Slurm选项覆盖partition、qos、时限等。双卡各自仍需容纳完整冻结基座，不将48G显存合并；保留既有48G训练的checkpoint/CPU activation策略。实际GPU显存与速度须由本次smoke/训练日志确认。

用户执行后依次发生：

1. CPU核对配对/原始caption；既有清单相同时不重写。
2. 两卡分片为3条代表训练样本计算新的“Local＋描述”条件embedding。这3条覆盖最大215对象样本、最长prompt样本和首条样本。
3. 双卡两步smoke，检查零gate等于冻结基座、语义和V/out有梯度、基座无梯度、无binding、rank参数一致及显存峰值。
4. smoke通过后，双卡分片补齐7390条训练caption embedding。VAE、DiT不参与此缓存阶段；原reference、target posterior和region缓存复用。
5. 重新创建正式Encoder/Adapter与空Adam；逐tensor确认与smoke训练前的初始化相同，从第0步开始正式训练到指定目标。不会加载smoke训练后的参数。原样本/噪声流也从0开始，没有旧C600的偏移。

smoke或缓存失败则退出，正式训练不会启动。没有自动生成评估、延长目标或再次提交。

旧通用prompt embedding不用于训练；`data.load_condition`只从旧cache读取图像reference/ref_shape，从新cache读取embedding/mask。因此更换文字真正改变了模型条件，不是仅修改说明文件。

## 中断恢复与阶段运行

重复同一个sbatch命令，会跳过已有有效缓存及完成的smoke，并恢复本实验最新完整checkpoint。允许换为不同的两张GPU继续训练，不承诺不同硬件逐位一致。不会恢复旧A/C检查点；不完整checkpoint被跳过。

例如完成2000后，明确决定继续到10000时：

```bash
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_caption_world.sh 10000
```

实际是恢复到总计10000步，不是额外10000步。超过作业时限时最多回退到最近保存点，需用户重新提交；没有自动重排队。

单独进行短smoke：

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:2 --cpus-per-task=8 --mem=128G --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_caption_world.sh smoke --gpus 2
```

`prepare`为CPU阶段；`cache`为全训练集GPU缓存；`check`为CPU缓存检查；`train`要求已完成对应GPU数量的smoke与缓存；`all`自动串起上述阶段。需要独立重复实验时用 `--root qwen_edit_pano/outputs/<新的目录>`。

可手动在实验根目录创建 `STOP`，在下一个optimizer step保存后结束；继续前需用户移走该STOP文件。

## 输出与评估边界

默认根目录：`qwen_edit_pano/outputs/object_world_caption_fresh_nobinding_v1/`。

- `DATA_PLAN.json`、`CAPTIONS.jsonl`、`PREPARED.json`：源数据与原始文字；无文件哈希。
- `conditions/<id>.safetensors`：新图文条件embedding/mask及实际prompt来源，旧图像缓存保留。
- `smoke/2/run-*/.../SMOKE_COMPLETE.json`：用户运行后的GPU检查结果。
- `train/run-*/train_log.jsonl`、`checkpoint-stepXXXXXX/`：正式训练日志和断点。
- Slurm日志：项目根目录 `world-caption-fresh_<jobid>.log`。

本版输入为Local＋状态＋场景描述；每条训练样本都给描述，没有擅自引入caption dropout。正式推理应给同类来源的描述。验证GT全景caption含额外目标信息，需明确写作描述条件/Oracle实验；不能把其结果直接称为“仅Local＋状态”成绩。将来只提供Local和预测状态时，需构造可得到的摘要或另外验证缺描述情形。

新实验与旧C同时不同在初始化、文字条件和binding，不能单独归因于binding。后续自然状态评价应使用同文字条件的World关闭/开启对照，判断是否仍主要依赖文字而忽略状态。旧Full-C专用评估入口会拒绝本版本，不要直接用旧通用prompt评估冒充同一条件。

## 已完成验证

实际8313条caption映射、7390条旧reference/posterior/region缓存头检查通过；新caption缓存尚需用户GPU作业生成。6项新CPU测试通过，包括真实tiny冻结Edit两步无binding反传、语义/V/out梯度、FP32 Adam精确恢复、文字缓存来源、原图reference复用和方向短语处理。共享梯度同步工具的双进程CPU Gloo测试通过，梯度最大绝对差约7.45e-9，参数差约1.46e-11；不是GPU NCCL或48G峰值认证。GPU检查由用户提交作业的smoke完成。

方法背景：[DiT360官方训练](https://github.com/Insta360-Research-Team/DiT360)提供image/caption条件训练；[LACONIC §3.4](https://arxiv.org/html/2507.03257v2)训练时不用全局caption，说明全景描述不是状态生成的普遍必要条件。这里是用户指定的合理新候选，不是已证明优于旧C的修复。
