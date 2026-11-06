# 单卡各组的ID / 绑定监督消融

2026-10-20：实现与CPU检查已完成，尚未由助手运行GPU、下载或提交Slurm。

两组各自一个独立sbatch作业/一张GPU，无DDP和互相依赖。冻结同一个Edit2511 + epoch015 Panorama LoRA(scale0.5)，使用7390train、单face2、polished目标、ID Embedding(42,256)、原Fourier几何和20/40/59层。两组可训练参数都为5309699。

- A：flow + .5cube + .5yaw + .25对象区域速度Loss。
- B：A + 实例绑定覆盖CE，权重前200步线性升至.01，只监督40层head平均注意力。

绑定监督取已有mesh实例与object_loss_valid，不做GT mask前向路由，不加入空间bias。按可靠支持像素>=64、类别非39/40筛选辅助监督候选，每图最多轮换抽32实例、每实例8个按软覆盖质量采样的query。全部对象token仍保留，泛化类别/小实例也仍参与原训练。采样使用局部NumPy RNG，不消耗训练的Python/Torch随机流。

两组seed0，新初始化与数据/噪声流相同；各2000 optimizer步（约1.083轮）、accum4、lr1e-4、warmup100，保存500/1000/1500/2000。`initial_world.safetensors`可在跑完后逐张量比对初始值，无内容哈希。旧名称编码权重、源数据、共享缓存和旧训练器保留。

## 用户提交

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
sbatch --job-name=id-A --nodelist=GPU1 --qos=normal qwen_edit_pano/world_pipeline/submit_object_world_id_ab.sh A
sbatch --job-name=id-B --nodelist=GPU2 --qos=normal qwen_edit_pano/world_pipeline/submit_object_world_id_ab.sh B
```

GPU1/GPU2与qos由用户选择；资源空闲时并行，否则各自排队。脚本默认debug、48小时、一GPU、4CPU、64GB主机内存；Slurm命令行可覆盖partition/time/qos。64GB是主机内存，不是GPU显存。

每个作业先运行各自两步GPU smoke，检查零gate的基座一致性、基座冻结、类表/Encoder更新，B额外检查零gate时绑定Q/K/类表梯度；通过后退出smoke进程释放显存，重新相同seed独立初始化正式训练。正式训练不会从smoke继续。任一步失败立即结束该作业，不影响另一组。

训练后每组固定六张val各生成correct/shuffled，共12张新全景；复用既有Legacy固定六图中的6张baseline，并核对相同基座/scale/数据面板/seed/采样协议，将来源写入BASELINE_REUSE.json。没有遍历六栋建筑。复用只读取旧结果，不等待另一新作业。

默认输出：

```text
qwen_edit_pano/outputs/object_world_id_ab_A_v1[_smoke|_val6]
qwen_edit_pano/outputs/object_world_id_ab_B_v1[_smoke|_val6]
```

日志`world-id-ab_JOBID.log`。返回两个JobID后可关闭本地电脑，训练由Slurm管理。

若同名输出已经存在，脚本拒绝覆盖；确需重新从头运行，使用新tag，例如脚本最后参数`A v2`与`B v2`。长训中断优先按完整checkpoint恢复，不删除旧输出。恢复入口：

```bash
# 用户手动GPU任务内执行；group与原checkpoint一致，新输出目录。
python -m qwen_edit_pano.world_pipeline.id_binding_world.train \
  --group B --resume CHECKPOINT_DIRECTORY --output NEW_DIRECTORY \
  --steps 2000 --warmup-steps 100 --binding-ramp-steps 200
```

必须使用qwen360解释器；其余设置与原config一致。新目录续训结束后的评估可单独调用本分支inference，不能改用旧semantic版本inference。

## 验证与限制

11项CPU测试通过：真实微型Edit完整flow/cube/yaw/region/binding反传；有无checkpoint及激活转存输出/梯度一致；清除/改变运行状态后重算正确且不重复收集Loss；关闭辅助监督与旧Fourier路径一致；同类多实例、软支持、空支持、排列、Q/K梯度、相同初始化/RNG；无Slurm的假Python runner测试。

两组真实CPU预检：7390train、923val，缓存缺失0。实际215对象样本，绑定候选125、选择32、256个样本对，每一对都指向正确源实例。旧baseline复用协议通过。详见research/audits/id_binding_ab_2026-10-20/。

测试中修正了对全-inf padding列做logsumexp引起的NaN梯度：先选择合法目标槽位，再跨head聚合。没有运行生产GPU反传；真实48G适配由作业开头smoke确认。按旧54秒/步粗估2000步约30小时，B额外开销和生成需另计，48小时是申请上限，不是完成保证。

实例mask是mesh几何支持，不是polished逐像素认证；辅助Loss改善不等于最终生成控制改善。最终仍看固定六图正确/错误条件的对象类别、位置、数量和Local保持。
