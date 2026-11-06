# 从零双卡实例绑定组

本版本是原A结构的独立训练实验：7390 train / 923 val，单Local face2，已对齐polished ERP目标，固定扩展指令。不是A10000续训，不是旧B1000，也不是grounded/C结构。

## 提交

在项目根目录执行，GPU任务由用户手动启动：

```bash
sbatch --nodelist=GPUX --qos=xx qwen_edit_pano/world_pipeline/submit_binding_fresh_dual.sh
```

默认总目标10000步。两张GPU共同训练一个World；每卡微批1、累积2，再平均两卡梯度，全局batch=4。CPU activation storage和梯度检查点适配已有48GB训练设置。默认申请8 CPU、128GB主机内存；指定节点必须能同时分配两张卡。

脚本先做CPU契约检查，再用两张GPU跑独立两步smoke。smoke验证零门控等于关闭World、监督标签不改变前向、两种绑定项有效、Q/K/类别梯度存在、基座冻结、两卡参数相等。通过才开始正式训练；正式训练重新加载原A保存的**未训练初始张量**，新建Adam，不继承smoke。2026-11-03已确认用户执行的双卡smoke及10000步正式训练完成，见下文。

CPU检查可独立执行，不分配GPU：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m qwen_edit_pano.world_pipeline.binding_fresh_world.run --stage check
```

## 固定配置

- 冻结Panorama epoch015，LoRA scale=0.5；原始模型与Panorama LoRA均不更新。
- 原A ID Embedding 42x256、几何Fourier 392维MLP，相加后LayerNorm；保留learned global；对象分支20/40/59层、宽256、8头。World参数5,309,699个。
- 损失：`flow + .5*cube + .5*yaw + .25*region + ramp*(.01*positive_CE + .001*reliable_other_BCE)`。
- 绑定项只监督block40实际Q/K；前200步线性增加到目标权重。不改变attention前向候选，不加入名称特征或新的空间写回。
- 正例为可靠实例区域对应token；负例只来自其它合格实例，忽略未知/背景补集，并在目标实例周围留1 token保护带，ERP横向周期处理。同类不同实例仍使用不同slot。
- 至少64有效监督像素，排除39/40泛类，每图最多随机监督32实例、每实例8正例和最多8负例；输入对象不因辅助监督筛选而丢弃。
- GT实例掩码只用于计算训练辅助项，不作为图像/状态生成条件；没有逐图GT全景描述。
- AdamW lr=1e-4，warmup100后恒定，betas=.9/.999、eps=1e-6、weight_decay=1e-5、梯度先跨卡平均再clip=1。FP32 World/Adam、BF16 autocast且关闭cast缓存。

## 断点与日志

输出：`qwen_edit_pano/outputs/object_world_binding_fresh_dual_v1/`。

- `smoke/run-*`与`train/run-*`分开。
- 每500步保存World、完整Adam及config，最后才写`COMPLETE.json`；不读取未写完的检查点。
- 重复相同sbatch命令，只恢复本版本正式训练中最高的完整步数，继续到总目标10000。拒绝原A、旧B、smoke或更改契约的检查点。不会重复增加10000步。
- 默认Slurm时限3天，提前600秒请求在完整optimizer step后保存；正常保存退出后尝试`scontrol requeue`。集群是否允许自动重排由其策略决定；服务器重启/硬杀后也可重新提交相同命令，从最近完整检查点恢复，最多损失499个未保存步。
- `world-bind-fresh_<jobid>.log`是Slurm日志，正式目录中的`train_log.jsonl`记录4条样本、时间步、损失、梯度、门控和显存。
- 不自动评估，不自动扩展到10000步之外。需要主动停止并保存时，可在根输出目录创建`STOP`；恢复前将该文件移走。

两张L40参考约78–83小时，建议预留3.5–4天；两张H200参考约28–33小时，建议预留30–36小时，均不含排队。10000步约40000次样本曝光，即5.41轮。H200参考来自其它结构的既有双卡日志，不是本版本实测。

## 检查范围与研究边界

2026-11-01已完成CPU检查：全部7390条训练实例数组能解码，尺寸512x1024、有效位、对象槽位实例ID均无错误；1条样本没有合格绑定对象，辅助项为零，仍正常使用原生成损失。现有条件/后验/区域缓存路径齐全；不新建缓存、不下载、不计算哈希。

13项独立CPU回归覆盖原A前向等价、实际微型Qwen零门控训练、checkpoint重算与CPU activation storage、辅助损失数值/稳定性/实例顺序/周期边界、两rank梯度平均数学等价、完整Adam恢复和driver smoke隔离。没有实际运行CUDA/NCCL测试，也没有证明生成效果改善。

本实验测试“绑定监督从初始化开始共同学习，经过与A相当的训练预算后是否改善状态控制”。此前A10000后追加500步的结果不回答这个问题；反过来，从零训练也不保证解决attention概率改善但生成语义不改善的问题。这里不把辅助项下降当作对象识别/位置准确率。

研究核对与详细边界见：`research/audits/binding_fresh_dual_2026-11-01/REPORT.md`。

## 2026-11-03实际结果与生成评价

用户已用双H200完成10000步，耗时27.81小时。全部7390 train ID被覆盖，40000次曝光，loss/参数/最终Adam均有限，无梯度裁剪触发。正、负绑定训练项均明显下降；与A匹配记录的最后1000步相比，共享生成误差约+.22%、region约+.30%。这尚不能证明生成优于A；分析时没有新生成图。报告与曲线：`research/audits/binding_fresh_results_2026-11-03/`。

直接评估现有权重，不需要继续训练：

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_binding_fresh_eval.sh
```

固定6张历史验证图片，Panorama-only / A10000 / fresh_binding10000三组共18图，相同Local/GT状态/seed、LoRA0.5/CFG4/28步，同一GPU/runtime。GT RGB和实例区域不作为生成条件；没有逐图GT描述或对象编辑。另有CPU `--preflight`。

生成后查看：`evaluation/step10000/quality_views/quality.html`。observer记录的类别/实例token-top1、global质量和正/负区域attention是代理指标，不等于生成图识别准确率。H200纯生成约15分钟，建议连加载和CPU可视化预留30–60分钟；48GB CPU offload更慢。新增2项CPU测试验证读数和观察器前向不变；GPU生成由用户执行。

## 自然生成完成后的状态干预诊断

实际18图在L40/GPU3完成，World单图约228秒。新组相对A的Local平均RGB误差−8.88%，选中对象RGB误差+1.59%，未确认稳定物体/布局收益；实际采样正确实例token-top1约43.3%，但它不是生成图识别准确率。详见`research/audits/binding_fresh_evaluation_2026-11-03/REPORT.md`。

固定客厅、同Local/seed，原A与新组各做7种状态：correct、仅对象列表反序、TV换挂画、TV换镜子、TV中心x+1m、移除TV、隐藏椅子换植物，共14图。修改前后目标框均完全在Local视野之外。删除token不是显式absence指令，本轮是字段控制诊断，不改训练目标为编辑任务。

```bash
srun -p debug --nodelist=GPUX --qos=xx --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_binding_fresh_state_probe.sh
```

L40纯生成参考约53分钟，连加载和CPU可视化预留1–2小时。完成后打开`state_interventions/step10000/review.html`，可滑动比较相同视角的correct与变体、两模型，并显示输入AABB参考框。合成状态没有编辑后GT，变化幅度不等于成功率。3项新增CPU测试与真实数据preflight已通过；GPU运行由用户执行。

实际14图随后已在GPU2/L40完成。分析见`research/audits/binding_state_interventions_2026-11-03/REPORT.md`：新组移除TV token后主要屏幕样面消失，原A基本保留；新组修改TV位置也有更强响应，但未在指定新位置生成明确电视，类别替换亦未确认成功。类别/位置变化确实传到实际Encoder/V向量，不能简单归因为字段没读入。该结果支持部分对象条件响应改善，尚不足以认定可靠状态控制或直接继续原样长训。
