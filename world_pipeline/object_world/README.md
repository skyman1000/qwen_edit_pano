# polished 单 Local Encoder / Adapter 新训练分支

2026-10-20新实验入口：用户选定简单ID Embedding，比较原Loss与新增实例绑定Loss。独立实现`id_binding_world/`与`submit_object_world_id_ab.sh A|B`已经准备，11项CPU检查通过，尚待用户GPU执行。两组各一张卡、2000步、GPU smoke通过后从头训练，旧语义分支保持原状。当前提交与验证说明见[ID A/B README](../id_binding_world/README.md)。不要用本目录的semantic训练/推理入口代替该新实验。

2026-10-19数值校准已完成：14/14前向，峰值39.80GiB，重复输入/探针开关输出差异均0；仅颠倒对象顺序仍约0.505%，yaw/类别变化与之同量级，错误yaw的区域Loss两图都略低。无需重复该校准。当前不直接提交多天长训，先独立改进World表示/数值路径；实际结果与限制见[校准报告](../../../research/audits/legacy_diagnostics_2026-10-19/NUMERIC_CALIBRATION_REPORT.md)。下方待执行校准、训练等内容为较早状态。

2026-10-19最新实际进度：Legacy诊断已补齐96条记录，checkpointed反传通过，峰值41.33GiB，没有更新参数。250→500步去噪指标改善增长，但对象均值干预保留98.5%收益，尚未显示可靠逐对象控制。类别/yaw扰动与纯顺序对照差异相近，已准备14次前向的重复/探针/顺序校准，尚待用户手动GPU执行。当前应执行[完成结果报告中的srun命令](../../../research/audits/legacy_diagnostics_2026-10-19/RESULTS_REPORT.md)，不直接启动下方历史长训或重新测试Spread。

2026-10-19诊断显存修正：48GB卡在新增的“不启用checkpoint反传对照”阶段OOM，已保存56/96条条件结果。诊断默认改为checkpointed反传smoke，保持CPU激活转存，不再默认运行完整CUDA parity；CPU parity测试仍保留。新增--resume可在原诊断目录补完剩余40条，旧PLAN/results保留，阶段显存写入memory_log.jsonl。详见[显存修复说明](../../../research/audits/legacy_diagnostics_2026-10-19/MEMORY_FIX.md)。这不是对原训练器或checkpoint的修改。

2026-10-19当前决定：用户选定Legacy（20/40/59），停止扩展Spread实验，已有Spread结果保留。本轮Baseline是冻结epoch015 Panorama LoRA、scale0.5，关闭World，并非原始Qwen。两组500步及各18张生成已完成；当前优先进行Legacy条件依赖诊断，见[原因排查与初版对照](../../../research/audits/legacy_diagnostics_2026-10-19/REPORT.md)。下面两组选层和万步提交说明属于历史方案，不表示应再次提交。

```bash
# 用户执行；复用既有缓存，0张新生成图，0次optimizer更新。
sbatch qwen_edit_pano/world_pipeline/submit_object_world_legacy_diagnosis.sh
```

该任务全新运行默认对Legacy250/500步、已曝光/未曝光各一条train视点、sigma0.2/0.5/0.8做90次前向，另做一次开启checkpoint的反传smoke；不是新增训练或独立验证质量评测。输出outputs/object_world_legacy_diagnosis_v1，日志world-legacy-diag_JOBID.log。没有下载或全量缓存构建。CPU70状态表示检查和相关测试通过，生产GPU诊断发生的OOM及恢复见上方说明。

长训前深度复查更新：两步smoke验证了GPU训练接口，尚未证明对象控制有效。全量区域权重与源记录复核通过，53项CPU测试通过；另发现语义分支尺度偏弱、微小区域等权放大的方法风险。当前建议先生成重载检查，再运行两组500步选层pilot，不直接提交下面的10000步接续任务。完整证据、修改与论文对照见[深度复查报告](../../../research/audits/object_world_deep_review_2026-10-18/REPORT.md)。

同日评测协议更新：pilot严格使用固定6张val图片，每组baseline/correct/shuffled共18张生成ERP、504次采样迭代，不会遍历6栋建筑的全部视点。ID、seed、错误World来源和GT对象筛选规则固定在research/audits/object_world_deep_review_2026-10-18/diagnostic_panel6.json。加载GPU前打印预算，生成时逐张打印耗时与剩余时间估计；额外的对象对照图只做CPU裁剪，不增加生成。LACONIC过滤规则的适用范围与参数分析见[后续分析](../../../research/audits/object_world_deep_review_2026-10-18/OBJECT_EVAL_AND_HPARAMS.md)。

源数据已完成：world_polished_repair_v2，7390 train / 923 val，最多215对象，失败为空。133条原候选仍明确排除。无需重建源对象字段。旧fourier_world与旧checkpoint保留；旧训练器没有类别文本语义或对象区域Loss，不能直接使用新精简清单。

2026-10-18实际结果：用户已完成生产GPU两步smoke与全量缓存。SMOKE_COMPLETE全部检查为true，第二步Encoder和类别语义投影都有非零梯度，峰值约41.40GiB；CACHE_COMPLETE覆盖7390条，ID顺序与train一致，条件/posterior/region缓存缺失均0。正式训练尚未启动。GPU与下载由用户手动执行，没有下载入口，模型全部local_files_only。

## 关闭本地电脑后的sbatch训练

已只读查询debug分区，最长运行时间3-00:01:00。按smoke稳态约56秒/optimizer step粗估10000步约155小时，不含排队，实际全数据速度可能不同。准备三段3500→7000→10000总步数，分别新增3500、3500、3000步；每个任务时限72小时，前一段成功后后一段才启动。

```bash
# 在项目根目录，由用户执行。内部调用sbatch提交三个任务并保存job ID。
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u -m \
  qwen_edit_pano.world_pipeline.submit_object_world_chain
```

看到ALL THREE JOBS SUBMITTED及三个job ID后即可断开SSH/关闭本地电脑。助手只运行了--dry-run预检，没有提交实际任务。重复提交会因_submission.json或输出已存在而拒绝，避免重复占卡。任务脚本为world_pipeline/submit_object_world_train.sh；内部是sbatch，非依赖本地终端的后台进程。

三段共享模型/训练设置并恢复optimizer与采样步数；没有从头训练三次，也不从smoke权重开始。输出依次为outputs/object_world_polished_semantic_region_v1、同名_part2、同名_part3。最后checkpoint为_part3/checkpoint-step010000，默认根目录的COMPLETE只代表第一段3500步。提交记录位于outputs/object_world_polished_semantic_region_v1_submission.json，Slurm日志为项目根目录object-world_JOBID.log。

```bash
squeue -u u13529658780
tail -f object-world_JOBID.log
```

JOBID换成提交返回的数字。后一段因Dependency排队属正常；前一段失败/超时则后续不会继续，已提交ID仍保存，按下面的extend说明从最近完整checkpoint恢复到新目录。时间估算不是72小时内每段一定完成的保证。

smoke第一步零门控使Encoder梯度为零，gate先更新；第二步已有非零Encoder/语义投影梯度是预期。两步loss下降受随机噪声与时间步影响，不证明收敛/质量提升。第59层绝对残差1810不能单独判断异常，原smoke未记录该层输入RMS，不能补造相对比例。正式训练新增只读world_relative日志（target_rms、relative_delta_rms），CPU检查确认不改变前向与梯度。7项CPU测试含两项假sbatch测试及真实数据dry-run通过，没有提交GPU任务。

## 当前训练方式

- 输入仍是单张原始face2 Local RGB，以及GT Oracle对象状态；polished ERP仅作目标，实例mask仅作训练监督。没有四方向训练或球面空间Bias。
- 冻结Edit2511、epoch015 Panorama LoRA（scale0.5）、Qwen文本编码器、VAE。只训练新的World Encoder与Adapter。原生Local图像条件保留。
- 类别ID保持官方37类定义，作为查表键，不作为连续数值。用本地Qwen2.5-VL文本编码器的最后隐藏层提取真实类别名称token，平均名称部分的token，排除模板前缀/句点/padding；离线保存固定特征，L2归一化后经可训练投影。不会直接把JSON数字改成字符串交给旧MLP。该语义初始化是否改善控制需要实测，不代表开放词表或对象识别器。
- 几何仍为米制相机局部中心、AABB完整边长、两项valid和Fourier32编码。动态对象token长度，上限256，当前215全部保留，无截断。对象ID不进入Encoder。
- 保留target-only门控残差attention与20/40/59注入层，width256；从零门控新初始化，不恢复旧Fourier训练权重。
- Loss = 原flow + 0.5 cube + 0.5 yaw + 0.25 object_region。后者对每个有有效区域的实例分别平均像素/通道速度预测误差，再平均实例。先按高分辨率实例面积分配权重，再area pooling到128×256 latent网格，保留小于一个latent格的对象贡献。没有有效监督像素的对象不参与区域项；极区支持沿用构建时的object_loss_valid，不改全局全景损失。
- 0.25是保守初始实验系数，不是已证最优。区域Loss是对象区域的去噪监督，没有从生成图检测物体并回归三维位置，也不保证attention自动获得精确位置绑定。mesh/polished差异和微小实例噪声仍会影响此项。
- AdamW lr1e-4、accumulation4、梯度裁剪1、warmup500后恒定；正式初始预算10000 optimizer steps，约5.41次train曝光。不是因为扩大数据就沿用旧1000步：7390条下1000步仅0.54次曝光。最终预算与区域系数需要验证集实验选择。

## 顺序执行

在项目根目录执行。各GPU步骤正常结束后再运行下一步。CPU准备和GPU缓存可重复执行以补缺；训练/推理使用全新输出目录，断点训练用extend。

```bash
# 1. CPU准备：复用旧缓存指针、生成区域权重，不修改源数据。
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh prepare

# 2. GPU：37类名称特征 + 两个smoke样本的缺失VAE缓存。
# 两样本包括训练集最多对象的样本，用于覆盖容量。
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh cache-smoke

# 3. GPU：两步optimizer检查；不会被当成正式训练结果。
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh smoke

# 可选：重新加载smoke checkpoint，在两张val图片生成baseline/correct/shuffled。
# 两步权重只能核查接口，不能判断最终质量。
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh eval-smoke

# 4. GPU：补齐全训练集的目标VAE缓存，已完成项不重算。
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh cache

# 5. GPU：正式Encoder+Adapter训练。
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh train
```

smoke通过应生成：outputs/object_world_polished_semantic_region_v1_smoke/checkpoint-step000002/SMOKE_COMPLETE.json。检查零门控与baseline完全一致、冻结基座无梯度、Encoder有梯度与参数更新、区域Loss开启和checkpoint往返一致。实际CUDA/BF16、显存与耗时由此用户执行步骤验证。正式训练不从两步smoke权重继续，而是相同新结构独立初始化。

CPU预检实测：7390条Local原生条件缓存全部复用；目标posterior复用1262，需新编码6128，另提取37类特征。CPU准备不复制全量RGB，不导出额外三维字段。没有文件内容哈希验证；为定位旧缓存，只复用旧命名函数对短字符串生成既有文件名，不扫描文件内容计算摘要。

新训练pack：data/object_world_polished_v1，含PLAN、samples、regions、缺失目标posteriors和class_features；必须保留旧full_v2、repair_v2和所引用旧条件/目标posterior目录。

正式输出：outputs/object_world_polished_semantic_region_v1，日志train_log.jsonl，checkpoint每500步保存。日志单独记录object_region及object_region_weighted；有这些数值和梯度仍不等于控制有效。

## 中断与评估

24小时不保证完成10000步；根据日志速度估算。用实际存在且含COMPLETE.json的最新checkpoint续跑到新的输出目录。示例（仅当step500确实完成）：

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_object_world.sh extend \
  qwen_edit_pano/outputs/object_world_polished_semantic_region_v1/checkpoint-step000500 \
  qwen_edit_pano/outputs/object_world_polished_semantic_region_v1_continued001 10000
```

保持pack、类别缓存、warmup、lr、accumulation等设置相同，总步数必须大于恢复步数。旧Fourier checkpoint结构不兼容此分支。

训练后使用`run_object_world.sh eval CHECKPOINT NEW_OUTPUT`在固定六张val图片生成baseline/correct/shuffled，固定同一微调基座、Local、prompt、seed与采样设置。六张图片仅作早期诊断，不能代表全部验证集。新分支没有自动训练时val loss选优或正式检测器空间指标；不能仅用train loss选择最优控制checkpoint。923条val不能进入optimizer。

## CPU验证

5项测试通过：大小实例贡献平衡、子格区域保留/梯度、无效与未表示区域排除、空支持为零、215对象容量与未知几何、冻结语义及对象排列、真实微型Edit在checkpoint开启/关闭时区域梯度一致且基座冻结。另验证本地真实tokenizer的名称token提取接口、真实旧条件与VAE缓存header、真实样本的512×1024实例到128×256权重（mean=1）、完整源集合和旧缓存可复用数量。

额外真实CPU集成：使用原本地Qwen配置和tokenizer缩小后随机初始化微型Qwen2.5-VL，文本无图像参数的名称特征提取正常；未加载生产权重。两个真实train样本（含215对象）加一个val完成训练pack与区域权重准备，保存于object_world_cpu_probe_20261018诊断目录；正式默认pack尚未生成，运行上述prepare创建。GPU smoke另外检查类别语义投影收到非零梯度。

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m unittest \
  qwen_edit_pano.world_pipeline.object_world.test_training
```

CPU检查不替代生产GPU smoke，尚未宣称正式训练已跑通或生成效果改善。
