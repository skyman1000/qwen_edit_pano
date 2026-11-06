# Semantic Grounded World v2：一次有边界的128图A/B小试

2026-10-28提交前复查：用户实际双卡Full-C smoke已通过，覆盖215对象，峰值41.3945GiB，World/Adam从C600正确加载，语义/几何/Value实际更新，基座冻结。21项CPU回归通过。新sbatch入口`submit_object_world_full_c.sh`默认双卡新增2000步、每100步保存、断点恢复、不自动评估/延长；由用户提交。详见[提交前复查及短命令](../../../research/audits/full_c_presubmit_2026-10-28/REPORT.md)。下面的GPU待执行描述属于此前状态；当前仅正式全量训练仍待用户启动。

2026-10-28更新：用户已完成后续C/D各600步及300条诊断/20张生成图。类别对比项未显示可靠的对象控制增益；不再叠加新Loss。固定C600结构/损失，完整7390条训练标签已CPU准备完成（219148对象—视点绑定对），新7项CPU检查通过。首段2000新步（全局batch4，约1.08轮）的独立入口`run_object_world_full.sh`每100步保存，不自动延长/评估。真实结果、功能性瓶颈与不确定性见[10-28结果分析](../../../research/audits/category_results_2026-10-28/REPORT.md)，执行/CPU检查状态见[全量固定方案](../../../research/audits/category_results_2026-10-28/FULL_TRAINING.md)。新GPU smoke/训练仍由用户执行；全量使用不等于对象控制已验收。

当前版本`polished_single_face_semantic_grounded_world_v2`。2026-10-27用户已完成A/B各200步、300条诊断和16张生成图；对象控制尚未通过全量训练验收，不能只凭绑定Loss下降继续长训。实际结果、类别Value压缩证据与64次前向诊断指令见[10-27分析](../../../research/audits/grounded_results_2026-10-27/REPORT.md)。助手没有执行GPU训练、推理或下载。旧World、训练器、checkpoint与数据均保留。

后续64条Value消融已完成，见[Value结果与训练目标改进提案](../../../research/audits/grounded_value_probe_2026-10-27/REPORT.md)。它显示V类别内容参与前向，但训练电视区域没有稳定收益。对象区域类别条件对比已实现为独立C/D续训，见[实现、检查与执行指令](../../../research/audits/grounded_category_continue_2026-10-27/README.md)。新增8项CPU检查及原14项回归通过；GPU smoke/新生成效果尚待用户执行。新入口是run_object_world_category.sh，原pilot命令仍只对应旧v2小试。

## 为什么重做这个小试

上一轮电视GT区域100%可达、自身路由平均0.9653、没有其它对象混入，仍未生成可辨认电视。因此不能把失败只归结为缺少位置绑定。椅子区域自身路由只有0.3578，其它对象0.6422，确有混合问题。推理20/40层的总残差约18.8%/15.8%，新增local名义残差仅0.033%/0.064%；这是耦合风险，尚非旧路径是唯一根因的证明。

v2是一个组合候选：语义＋几何分开编码并拼接、可学习实例交互、从冻结Panorama基座起零门控新World，匹配A/B只隔离绑定Loss的增量。不能把可能的整体收益单独归因于“换语义”或“去旧World”。若两组均不能在训练图形成对象恢复/响应，不自动扩大训练或继续换模块。

## 实际实现

- 单张原face2 Local＋原相机局部米制center/full AABB/valid位→对齐polished ERP。没有四方向训练，没有更换坐标契约，没有新增球面距离bias。
- 使用已有42×3584冻结Qwen类别名特征；ID只作查表索引，不改变GT字段。语义与Fourier几何分别投影/归一化后拼接。大语言模型不加载、不训练，新增Encoder＋Adapter共6,348,547参数。
- 注入层20/40/59、宽256、8头。图像Q对输入框投影允许的对象K/V计算注意力。空key/value严格为0，没有旧的learned global value常量通路。无候选位置精确不更新。
- 旧ID World/Spatial权重不加载。冻结epoch015 Panorama LoRA0.5及原Qwen，两个新组都从相同seed的新World开始；零门控必须逐元素复现Panorama-only输出。
- GT实例mask只作辅助Loss标签，绝不生成attention候选，也不作推理输入。候选由输入3D框独立投影。投影统一用CPU，避免CPU标签准备与CUDA边界浮点差异。
- 未知几何/包围相机的框不强行投成全天区域，保留字段但不进入局部写回；由基座补全。此小试不认证这类对象的可控性。

两组原生成目标相同：`flow + 0.5 cube + 0.5 yaw + 0.25 object-region velocity MSE`。

| 组 | 区别 |
|---|---|
| v2-A | 无额外实例绑定Loss |
| v2-B | 增加0.01×实例coverage Loss，仅40层，前20步线性启用 |

coverage Loss读取实际SDPA相同的Q/K和允许对象集合，监督真实实例slot；它不是对象类别识别Loss，也不保证生成对象。0.01是待验证起点，不是最优值。只比较这项增量时，两组Encoder/结构/初始化/数据/噪声/优化器相同。posterior采样、噪声和时间步使用按全局microbatch编号生成的独立CPU随机流，保证A/B及断点前后实际输入一致；不同GPU计算仍可能存在浮点数差异。

## 数据与监督

复用`research/audits/spatial_generation_results_2026-10-26/TRAIN128.json`：128图、61栋train，39图含电视，无val。全部条件、目标posterior、区域权重、实例源文件齐全，不下载、不重建图像、不处理其它字段、不做哈希/CRC扫描。

新的稀疏训练标签在`qwen_edit_pano/data/grounded_world_v2_support/`。4128个实例可用；过滤generic870、过小/无有效支持1416、可达比例不足0.5的51个实例。过滤只影响辅助监督，不重写原样本。按实例等权、每实例8个fractional-support抽样query，最多32实例。位置用canonical ERP列，避免padding seam重复计数。类别39/40不用于绑定标签。

## GPU执行

从项目根目录运行。Node/QoS均可自行修改。无需再次执行CPU准备。

先跑A/B各2步smoke，约5～10分钟、不含排队；覆盖本128图中对象最多的188对象样本：

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=00:30:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_grounded.sh smoke
```

smoke通过后，两组分别提交到两张卡/两个终端，可并行。每组固定200步、global batch4、800暴露/6.25轮，lr1e-4、warmup20、FP32 World/Adam eps1e-8、BF16冻结基座、CPU activation storage。旧任务约55～60秒/步，暂按每组3～4小时规划；新模型CUDA速度和显存未实测，以smoke和前10步日志为准。

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_grounded.sh pilot --group A
```

```bash
srun -p debug --nodelist=GPUY --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_grounded.sh pilot --group B
```

只有一张卡就依次运行。每50步保存。中断后重复同一命令，自动从该组最新完整checkpoint恢复；不从smoke或旧World恢复。配置/数据改变会拒绝静默恢复。每步日志给出`recent_seconds_per_step`和`eta_seconds`。每组结束不自动评估、不延长训练。

新输出：`qwen_edit_pano/outputs/object_world_grounded_v2/{smoke,pilot}/{A,B}/run-*/`。

两组结束后明确运行一次有限评估：

```bash
srun -p debug --nodelist=GPUX --qos=normal --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=03:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world_grounded.sh evaluate
```

先300次固定噪声诊断（4train/6val、3sigma、两组各correct/repeat/错类别/错位置/World-off）；再生成16张ERP（2train＋2val，每图baseline/正确A/正确B/一次B干预）。优先按输入字段选择镜头外电视，不根据生成分数挑图；有电视时反事实只改一个镜头外电视为镜子或移动1m，否则使用镜头外对象重排。GT仅在生成后读取用于对照。render的val来自同6张开发图，不额外做六栋全建筑评测。

预估评估约1～1.5小时，依赖GPU/CPU offload；首次运行以实测为准。两组并行合计约4～6小时，单卡顺序约7～10小时，不含排队；不是保证时长。

输出`outputs/object_world_grounded_v2/evaluation/<matched-checkpoints>/diagnose/{SUMMARY.json,CONDITION_GAPS.png,report.html}`及`render/{comparison.html,LOCAL_RGB_MAE.json}`。可单独运行`diagnose`或`render`，重复相同命令补齐缺失记录。GT原框裁剪不是生成对象检测；反事实没有编辑后GT，不能对反事实继续配原图作训练目标。

## 预先约定的判定与停止条件

200步只是可行性小试，不能保证收敛，也不能仅因200步失败宣布架构无法学习。

1. 必须先看训练图：正确状态能否恢复电视等主要对象，改变类别/位置是否引起相应对象变化；GT裁剪与生成同时检查。只有attention更集中、gate更大或MSE更低不算通过。
2. 检查开发val上的同类趋势、输入Local保持及图像质量。A/B都没有对象恢复/响应，则停止这条候选的全量扩展，先审查训练图拟合、监督与优化证据，不自动加步数或再叠模块。
3. A有效而B无额外收益，保留A；B在对象响应且图像质量上更好才保留B。若训练有效但val不足，优先扩大独立布局多样性，不能只在32/128图上无限重复。
4. 本阶段不提供自动全量长训入口。通过后再决定7390图上的明确预算，以及类别语义/旧路径保留等单独消融。

## 检查范围

14项CPU测试通过：真实tiny Edit的零门控基线一致性、非零门控下标签不改变预测、实际Q/K/语义梯度、null/unrouted不更新、seam/persistent-grid对应、checkpoint/CPU-offload梯度精确一致、优化器精确恢复、原panorama＋region＋binding目标的双组两步反传、真实128图4128实例的可达监督、A/B同初始化、实际latent/噪声/时间步跨组及恢复一致、launcher恢复与CPU评估入口。最终完整结果见审计目录的`FINAL_CPU_TESTS.log`。

开发期CPU评估入口用`/tmp`的明确未训练fixture测试软件流程，不是200步模型成果。用户后续真实GPU运行见10-27分析；两组已完成，训练峰值约41.4GiB。新增Value诊断另外通过3项CPU测试，尚未由助手执行GPU。

检查与设计依据：`research/audits/grounded_world_v2_2026-10-26/`。原始问题的证据：`research/audits/spatial_generation_results_2026-10-26/REPORT.md`。
