# A10000起点：无硬路由的训练绑定对照

用户手动执行全部GPU任务及Slurm提交。独立版本`id_A10000_train_only_binding_refine_v1`，保留旧A/B/C和caption实验，不修改旧数据、缓存、训练器或检查点，不计算文件哈希。

两组均继承同一A10000 **World和Adam**，继续原7390训练数据/923验证图、单Local face2、polished ERP、固定扩展指令、冻结epoch015 LoRA(scale0.5)。原ID、Fourier、相加后LayerNorm、可学习global、20/40/59层全部保留，无名称语义、融合改造、球面bias或前向区域限制。只用已有缓存，不生成新图文条件缓存。

| 组别 | 目标 |
|---|---|
| control | flow + .5 cube + .5 yaw + .25 region |
| binding | 相同目标 + .01 coverage + .001 concentration |

两项辅助权重正式训练前200个**新增步**从0渐进至表中值；这是保守起始候选，不是已验证最优。覆盖项为已有head平均目标概率的实例平衡CE。集中项是实例平衡的`-log(1-A(q,i))`：负例只从有效支持的其它合格实例采样，不用整图`1-mask`，目标边界留1个token守卫，ERP横向周期。集中项是针对本项目的采样BCE适配，不是逐公式复现TokenCompose/RegionRoute的比例或KL。

每图至多32个可靠实例、每实例8正例+至多8负例；排除39/40、支持像素至少64；混合单元正例保留面积质量。无可靠负例时该实例无负项，空监督为有限可微零；未监督对象仍保留输入token。GT mask只用于Loss，不进入attention前向或推理。监督只读40层真实Q/K，padding重复列及reference不采样。mesh支持不是polished像素认证。

新增集中项使用稳定的other-key logsumexp计算`log(1-mean_head_probability)`，避免概率接近1时相减或将概率错当logit。原CE正例权重均值归一化，负项在有负例的实例间独立归一化。原全局/区域目标保持不变。

## 用户运行

项目根目录，**两组各一张卡并行**（两个独立作业，不是一个模型双卡DDP）。可用同一GPU节点的两张空闲卡，也可指定不同节点。节点名GPUX不是CUDA卡编号，具体卡由Slurm分配。

```bash
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_binding_refine.sh control
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_binding_refine.sh binding
```

默认每组新增1000步：A10000→11000，约增加0.541轮；不是从零训练。时间紧时两个命令都加末尾参数`500`，则到10500。两组应使用相同预算。

每组先CPU检查共享缓存路径及代表样本头/支持，再单卡两步smoke；通过后从A10000重新加载World和Adam正式训练。**不继承smoke更新**。每100步保存World/Adam/config，结束不自动评估、不自动延长。重复相同命令会跳过已通过的smoke并从该组最新完整正式检查点继续，忽略未提交完成标记的半成品。

单独短检查：

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_binding_refine.sh binding --stage smoke
```

CPU检查（无GPU，无下载）：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m \
  qwen_edit_pano.world_pipeline.binding_refine_world.run binding --stage check
```

默认输出`qwen_edit_pano/outputs/object_world_binding_refine_v1/{control,binding}/`：`CPU_CHECK.json`、`smoke/run-*`、`train/run-*/train_log.jsonl`与`checkpoint-step010100...011000/`。Slurm日志`world-bind-refine_JOBID.log`。可创建组目录下`STOP`使其下一完整step保存退出，继续前需移走STOP。独立新重复可通过run入口`--root qwen_edit_pano/outputs/新目录`。

正式每组单卡微批1、累积4、FP32 World/Adam、BF16冻结基座、gradient checkpointing及activation CPU storage；沿用A的LR1e-4与Adam(.9,.999)/eps1e-6/decay1e-5，clip1。两组均从绝对step10000的micro-index40000继续，逐microbatch种子相同，辅助采样用独立NumPy RNG。新的两组之间流匹配；不承诺跨硬件逐位一致或复现历史单/双卡混合训练的所有数值。

此版本是新目标的明确warm start，不绕过原训练器严格resume。仅接受自己同组、同源stat/配置的检查点恢复。旧C/旧B检查点不能混用。GPU烟测需用户实际执行，CPU通过不代表48G显存或最终生成控制已验收。

CPU验证已完成：8项测试通过，覆盖真实tiny Edit的原A前向/梯度完全一致、辅助监督不改变前向、checkpoint与CPU activation storage梯度一致、重算前清除/改变状态、极端logits/空支持/无效padding、同类实例与ERP环绕守卫、独立辅助采样不消耗训练RNG、Adam精确恢复与拒绝跨组恢复，以及smoke隔离/正式断点/有限预算的runner测试。实际A10000全部5309699参数严格加载，参数顺序与原A一致，Adam恢复到10000步。7390条已有缓存路径均齐全；代表样本头/标签已检查，最大215对象样本选32实例、256正例+256负例。没有宣称全量polished像素标签认证。检查记录位于`research/audits/world_binding_no_routing_2026-11-01/IMPLEMENTATION_CHECK.json`。

先用相同六图、seed及采样检查真正的对象类别、相机方向、可见数量、投影尺度与Local/透视画质；不把辅助CE下降、对象区域RGB降低或正确/错误图像不同当作成功。原A10000为零新增预算参照，control与binding为等新增预算对照。只有生成收益和画质同时成立才考虑长训。训练后评估需识别本新版本，不要直接用旧C专用评估器。

## 10500步训练后的独立评估

两组已完成500个新增步。用户手动运行（不训练）：

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_binding_refine_eval.sh
```

固定历史六图和seed，对比Panorama-only、原A10000、control10500、binding10500，共24张；同pipeline/同GPU/同采样，原ID World前向统一严格加载。GT只供生成后对照，实例mask只供只读attention评分。观察真实40层Q/K调用0/1/28/29/54/55，不改变前向、不提供额外生成条件。CPU预检`bash .../run_binding_refine_eval.sh --preflight`无需GPU。

输出`object_world_binding_refine_v1/evaluation/step10500/quality_views/quality.html`、`comparison.html`、`METRICS.json`、`SAMPLING_ATTENTION.json`。已有图片支持同GPU UUID/runtime的部分恢复；更换设备用`--output qwen_edit_pano/outputs/新的评估目录`。训练分析及曲线见`research/audits/binding_refine_results_2026-11-01/`。先完成此次验收，不自动继续训练或修改Loss。

## 已完成验收的结论（2026-11-01）

24张同H200自然生成已完成。binding相对control的Local RGB MAE略低1.43%，GT实例RGB MAE却高2.95%（4/6图更高）。实际采样36/36读数正例CE更低、36/36负例BCE更高；90度视图未显示目标类别/布局得到相应恢复。此配方不满足预定扩训门槛，**不建议继续到11000或加大辅助系数**。这是本配方在开发面板上的停止决定，不是所有binding方法无效的结论。

CPU独立指标复算误差0，旧Panorama/A10000的12张新旧图解码像素完全相同。完整观察、证据边界及后续条件接口设计候选见`research/audits/binding_refine_evaluation_2026-11-01/REPORT.md`。没有修改训练器、强制停止用户任务或提交新训练。
