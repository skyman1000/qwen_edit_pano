# GT World Encoder＋Adapter pilot：用户手动执行

本入口只训练新World Encoder/Adapter。GT来自Matterport原始house对象/mesh，经gt_world_pilot_v1相机转换和可见性派生；完全不读取Qwen3-VL预测。Local仍通过原Edit图文embeddings与reference latent输入；完整ERP RGB仅为监督目标。

已核实paired_full_63637/checkpoint-epoch001完成1 epoch/1881 optimizer steps。可以固定作初期基线，不是25epoch最终模型，不覆盖它、不恢复其optimizer、不改变正在运行的Edit训练。代码全在research/world_pipeline/oracle_*.py。

当前GT筛选后7 train/0 val，仅用于接入smoke、重载和小样本拟合。4个flags样本仍排除，不能宣称验证集收益。purpose=oracle在没有合格val时明确拒绝执行。后续须解决GT可见性规则并扩展独立val，再用相同预算的constant/observed/full分支做正式oracle；全局打乱只是补充诊断。

## 模型及训练定义

- 共用特征版本objects_local_aabb_masked_v1：类别、局部中心米制3维、当前相机轴AABB全尺寸3维、中心/尺寸有效性2维。未知类别编码0；未知几何零占位但mask=0，真零中心mask=1。ID/来源/房间/二维框不进入首版Encoder。
- GT G_full范围沿用已验收pilot；不是全建筑所有物体。G_obs对照由其中local_evidence=visible筛出，其三维信息也是oracle，不是Observer预测。
- WorldEncoder：类别embedding＋几何MLP＋LayerNorm，固定128对象上限及padding mask，超限报错不截断；额外一个始终有效的空/全局token。无对象位置序号embedding，支持集合置换。
- Adapter：默认20/40/59层，width256，8头cross-attention，目标ERP hidden作query，world tokens作key/value。tanh零初始化gate，按冻结hidden RMS缩放残差。直接残差只加到目标ERP（含circular padding），保留reference切片；后续原生attention可能间接影响reference，不能宣称全程不变。
- 冻结Edit基础模型、Panorama LoRA、VAE与内部图文编码器；LoRA保持eval关闭dropout。内部编码器通过既有缓存复用，不在训练时加载。
- batch1，累积4，AdamW lr1e-4/weight_decay1e-5，梯度裁剪1。200步pilot warmup20，smoke2步warmup0。与原训练独立，不套2350步warmup。固定1024×2048，不做flip/yaw增强。
- 每步从7条中有放回抽样，目标VAE仍posterior.sample、reference沿用cache的mode。先用GPU编码目标posterior mean/std到新目录后释放VAE，再加载Transformer，避免二者同时驻留。每个microbatch重新采样后验，不缓存固定目标latent。
- 复用原flow+0.5cube+0.5yaw实现。本独立小实验从第一步启用几何项，各对照保持一致；不修改原25epoch的损失启用时机。
- 梯度检查点显式传world tokens/mask并捕获目标边界，防止backward时串样本。仅保存world权重和optimizer，不复制20B基础权重。
- 推理保持28steps、CFG4、每ID固定seed，正负CFG分支同一world；默认同一进程生成baseline bypass与correct GT，保留Local条件。

## 1. CPU准备冻结输入清单

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export PYTHON_BIN=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=8G --qos=normal --time=00:10:00 \
  "$PYTHON_BIN" -m research.world_pipeline.oracle_data \
  --gt qwen_edit_pano/data/gt_world_pilot_v1 \
  --review qwen_edit_pano/data/gt_world_pilot_v1/assistant_review_decisions_2026-10-02.json \
  --base-checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch001 \
  --output qwen_edit_pano/data/oracle_pilot_v1
```

复用paired_full_v1正式清单和条件cache，核验hash、建筑split、GT与RGB对应、朝向及LoRA配置。只选已approved且无flags样本。预期train_count7、val_count0。保留原图和cache，仅新建BUNDLE.json及samples.jsonl。

可选CPU测试（助手已执行，不必重复）：

```bash
"$PYTHON_BIN" -m research.world_pipeline.oracle_test -v
"$PYTHON_BIN" -m research.world_pipeline.oracle_train \
  --bundle qwen_edit_pano/data/oracle_pilot_v1 --preflight
```

## 2. GPU smoke（先执行，不直接跑长训练）

实际首轮GPU失败后已修复两项：训练autocast使用cache_enabled=False，且no-grad smoke对照与训练分开上下文；默认`--activation-storage cpu`将反传需保存的非叶激活移到CPU（不搬冻结权重、不改BF16）。原v1失败目录保留，用下述v2新目录。memory.jsonl记录模型加载、前向/反向及激活复制量。CPU 7项测试通过；真实GPU重跑结果仍待用户确认。原GPU2日志为本进程OOM，GPU4日志为checkpoint元数据错误，不应混为同一原因。

节点GPU2仅示例，换成不占用正式训练的节点。主内存96G不是显存申请；20B生成模型显存需求高于8B Observer，真实峰值待本机smoke确认。不要为避免OOM静默换分辨率/模型/精度。

```bash
srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=02:00:00 \
  "$PYTHON_BIN" -u -m research.world_pipeline.oracle_train \
  --bundle qwen_edit_pano/data/oracle_pilot_v1 \
  --condition full --purpose pilot --smoke --activation-storage cpu \
  --output qwen_edit_pano/outputs/oracle_full_smoke_v2
```

成功需checkpoint-step000002/SMOKE_COMPLETE.json与COMPLETE.json。检查零gate与基线输出完全相同、base/LoRA无梯度、第二步Encoder出现非零梯度、world参数更新及保存张量重读一致。train_log包含有效残差（可因BF16小门控而很小）、梯度和显存。smoke不承诺可见生成改善。

## 3. GPU重载推理验证

```bash
srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=02:00:00 \
  "$PYTHON_BIN" -u -m research.world_pipeline.oracle_inference \
  --checkpoint qwen_edit_pano/outputs/oracle_full_smoke_v2/checkpoint-step000002 \
  --split train --limit 2 --modes baseline correct \
  --output qwen_edit_pano/outputs/oracle_full_smoke_reload_v2
```

每样本local.png、baseline.png、correct.png、gt_erp.png及模式元数据。GT结构确实进入correct分支，GT ERP在生成后才用于存对照。训练样本上的结果仅用于接入检查；两步world仍可能几乎无可见影响。

## 4. 可选：200步拟合pilot（不是当前优先步骤）

2026-10-06：`oracle_full_smoke_v2`与`oracle_full_smoke_reload_v2`已完成。两张训练样本的图像发生变化，但没有明确的GT结构改善证据。当前优先扩展GT候选并建立独立val，见`ORACLE_RELOAD_REVIEW_2026-10-06.md`。下述7条训练集拟合仅为可选学习能力诊断，不应替代验证集实验。

```bash
srun -p debug --nodelist=GPU2 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=24:00:00 \
  "$PYTHON_BIN" -u -m research.world_pipeline.oracle_train \
  --bundle qwen_edit_pano/data/oracle_pilot_v1 \
  --condition full --purpose pilot --steps 200 --warmup-steps 20 \
  --output qwen_edit_pano/outputs/oracle_full_fit_v1
```

200步为小样本拟合预算而非已验证最佳超参；先用smoke耗时估算作业时长，24小时是申请上限，不是耗时预测。每50步保存，也在最后一步保存。若需恢复，使用同一配置加`--resume .../checkpoint-step000050`并指定新output，steps为最终总步数；禁止从smoke改配置继续冒充相同实验。

推理命令同第3步，checkpoint改为oracle_full_fit_v1/checkpoint-step000200，output换新目录，可加`--modes baseline correct shuffled`。shuffled只在同split样本之间确定性错配；它是条件依赖性诊断，不是等预算对照。

## 5. 正式收益实验前的门槛

不能用当前7条train作成功结论。先处理GT四条flags、验收/扩展GT train/val与建筑覆盖；再重建新bundle。相同正式checkpoint、相同样本、训练步数、初始化seed分别训练`--condition constant`、`observed`、`full`，选`--purpose oracle`；三个分支除world内容之外预算一致。constant只有固定空token，不泄漏每图对象数量。模型checkpoint变更意味着新实验，不与epoch001结果混报。

评估应包含Local保真、镜头外类别/布局、接缝/极区；按建筑汇总，结合GT几何和独立人工核查，不能只让同一个Observer给生成图打分。没有GPU smoke/生成结果前，不声称接入已经在20B模型实测通过。
