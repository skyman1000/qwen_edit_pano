# polished 单 Local 对象数据构建

最新执行结果：world_polished_repair_v2已完成8313条（7390train/923val），failures为空、all_candidates_built=true，最大215对象。源对象数据构建无需再跑。下一步是新的[object_world训练分支](object_world/README.md)：CPU训练pack/区域权重准备、GPU类别语义与缺失目标posterior缓存、两步smoke，再正式训练。新代码已实现并做CPU检查，尚未执行GPU smoke或正式训练；旧Fourier训练器不包含这些改动。

以下故障与目录状态为此前记录，以以上最新结果及实际BUILD_RESULT为准。

当前故障恢复入口：repair_v1准备中断，已有7007条record，run_build.sh尚未生成；且复用所需的outputs/polished_alignment_review_run001目录目前已不在原路径。下文较早的--recovered-review命令当前不可执行。原full_v2的8282条成功数据仍可复用。

准备器现支持--resume：核对已存在配置、候选和对象记录，补齐缺失项；配置或记录不同会拒绝，已有记录不重写，持有BUILD.lock避免并发。由于v1引用的恢复目录不可用，使用新repair_v2，不传--recovered-review，将31条恢复候选重新交给CPU构建；保留半成品v1不删除。

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u -m \
  qwen_edit_pano.world_pipeline.prepare_polished_repair \
  --exclude-missing-poses \
  --registration-audit research/audits/polished_build_failures_2026-10-18/registration_checks.jsonl \
  --exclude-unregistered \
  --output qwen_edit_pano/data/world_polished_repair_v2 \
  --candidate-output qwen_edit_pano/data/world_polished_repair_candidates_v2 \
  --resume

srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
  --qos=normal --time=02:00:00 \
  bash qwen_edit_pano/data/world_polished_repair_v2/run_build.sh
```

首条应报告seeded_successes=8282、registration_retries=31；等待其打印Prepared only后执行第二条。首条中断可用相同--resume命令继续，不计算哈希或重建8282条mask。第二条最终目标仍为8313（7390train/923val），排除133条，原始全集覆盖仍为false。9项CPU测试通过，包括中断补齐、已有记录不重写、不同配置/记录拒绝及并发写入拒绝。

以下为检查目录曾存在时的记录，不能据此假设该目录当前仍可用。

最新磁盘结果：对齐检查已完成，无需重新运行下面的检查构建命令。`outputs/polished_alignment_review_run001/summary.json`报告12条原成功样本与31条恢复样本共43条渲染成功，失败为空；31条对象/mask已全部生成。检查页位于同目录review.html，可离线打开。数值通过与页面生成均不代表人工逐图认证。

下载齐全不等于逐视点标定齐全；现有107条缺口没有明确的补ZIP目标。配准数值门槛、16像素和54度纬度均是实现选择，需以实际状态/区域对应关系评估，不能当作任务天然要求。以下保留检查页的复现命令；已有目录需要追加--resume。

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_polished_world_review.sh \
  --output qwen_edit_pano/outputs/polished_alignment_review_run001
```

默认覆盖12条已有成功样本（复用/新建、多对象、低正前方NCC、低区域Loss覆盖及不同视点）和31条恢复候选。候选只在检查目录的recovered/下新建，原8446候选与8282成功产物不改写，不决定排除任何样本。已成功的候选在同一命令追加--resume后复用。全程CPU，没有下载/哈希/训练/模型。

检查目录的review.html嵌入图片，可单独复制到本地离线打开。展开样本，比较Local和polished正前方；点击对象或选对象ID查看三维中心/AABB尺寸、实例轮廓及纬度裁剪影响。绿/红线表示参与/未参与当前对象区域Loss；单对象橙线为AABB投影。正前方mask从ERP采样，不能冒充独立Local标注。AABB也不是可见像素的紧框。页面支持导出人工意见，所有样本初始待审，不自动标为训练可用。

检查页保留所有样本的待审状态，不自动改变原数据集。错误与未完成情况写入summary.json和review.html，不会静默过滤。

2026-10-18全量实际结果：8446条候选中已完成8282（7362train/920val），164条失败。其中107条在现有house位置、conf相机、raw pose/intrinsics及undistorted color中均没有对应UUID；另57条原配准失败。`REUSED_EXISTING_STATE`表示正常复用。最后非零退出是完整性保护，不是GPU错误。原目录保持部分构建状态。

CPU对57条失败重新做标定检查：保持角误差2.5度、拟合60内点、验证30内点/3个观测不变，允许先偶数yaw拟合、失败后用奇数yaw拟合且由偶数yaw独立验证。31条通过，26条仍失败。31条在检查目录的recovered/中已全部完成对象/mask构建，但尚未合并到正式数据清单。

建议另立明确筛选后的8313条集合（7390train/923val），排除133条并保存原因，原8446条候选与8282条已构建结果不改动。以下命令明确选择排除缺位姿和仍未通过标定的样本；如要求8446条全覆盖，就应继续恢复位姿/配准，不能把这个子集称为原始全集：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m \
  qwen_edit_pano.world_pipeline.prepare_polished_repair \
  --exclude-missing-poses \
  --registration-audit research/audits/polished_build_failures_2026-10-18/registration_checks.jsonl \
  --exclude-unregistered \
  --recovered-review qwen_edit_pano/outputs/polished_alignment_review_run001

srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/data/world_polished_repair_v1/run_build.sh
```

第一条准备新的candidates、CONFIG、SELECTION和成功记录引用，不执行图像配准。新增--recovered-review会核对检查目录的源配置、31条记录的对象、配准指标与mask支持，直接复用已完成产物。正常应报告seeded_successes=8313、recovered_successes=31、registration_retries=0。第二条随后生成正式train/val和BUILD_RESULT；此时不再重跑31条配准。新目录world_polished_repair_v1和world_polished_repair_candidates_v1必须不存在；准备完成后不要重复运行准备命令，续跑直接执行第二条。旧数据目录和检查目录的recovered/不可删除，因为新集合引用其成功资产。

修复输出的BUILD_RESULT会同时报告requested=8313、original_requested=8446、excluded_from_original=133。即使所选8313条全部完成，all_original_candidates_built仍为false。新失败不会被静默剔除或自动改成完成。

修复准备器与配准共7项CPU测试通过。检查产物复用仍明确记录排除项与来源，不将页面渲染当作视觉认证。本次未执行正式筛选准备命令、GPU或下载。历史调查见research/audits/polished_build_failures_2026-10-18/REPORT.md，以最新产物和本节说明为准。

本入口落实用户最新范围：polished ERP目标、原始face2单图输入、当前37类及局部米制中心/AABB完整边长；对象区域监督与类别名称。没有球面bias、四方向训练、模型加载、GPU、下载或内容哈希验证。保留旧数据和旧训练实现。

2026-10-18本地检查：61栋train、11栋val的七类精简构建依赖齐全，missing_zip_files=0；42个polished Arrow存在，8446个Local ZIP成员均找到。这个结论来自文件与ZIP目录，未做全量CRC/载荷认证。

## 执行

在项目根目录由用户手动运行，默认两个建筑worker、每个四个CPU线程。建议分配8 CPU、48GB内存，不请求GPU：

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
  --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_polished_world_build.sh build
```

若已经在合适的计算节点中，直接执行同一bash脚本即可。重复同一命令会续跑自身已完成的样本；不覆盖旧World实验。默认输出data/world_polished_full_v2，日志data/world_polished_full_v2.log。

查看进度（纯CPU本地文件，不计算哈希）：

```bash
bash qwen_edit_pano/world_pipeline/run_polished_world_build.sh status
```

代码默认复用当前已选入实验的1430条状态，仅从旧mask拷贝instance/valid；为其余7016条候选新建对象与投影。新生成时以原始图和已知透视相机配准，拟合even source yaw、检验odd source yaw。多张原始图只用于离线相机标定，训练仍是单face2。保留源相机手性，不强制det=+1，也不强制level到世界竖直。

使用全部区域三角形作遮挡，包括未标注表面；区域实例到house对象使用精确segment集合关联。只导出任务对象类别、中心、尺寸、两项valid和实例监督关联。对象选择仍为Local512或ERP512x1024可见像素>=16，与当前范围一致。不会静默截断超过160的场景；BUILD_RESULT会报告实际最大对象数，后续训练据此设置容量。

不生成房间、关系、材质、传感器深度/法线、全量RGB副本、文本语义向量或VAE缓存。polished继续由原Arrow指针加载。语义名称保存在category_names.json；之后再由用户手动运行模型提取类别向量。

## 输出

```text
CONFIG.json
category_names.json
train.jsonl / val.jsonl
progress.json
BUILD_RESULT.json
samples/<sample_id>/
  record.json
  object_regions.npz       # instance / instance_valid / object_loss_valid
  calibration.json
  instance_preview.jpg     # 每个建筑首个成功样本
  failure.json             # 仅失败时存在
```

监督mask为512x1024，训练RGB目标仍为原polished ERP，可按原训练分辨率加载。object_loss_valid只包含当前对象集合和保守的|latitude|<=54度支持，以避开polished生成修补极区；这是可配置的训练监督支持，不是官方像素有效性标注，也不裁切输出或改变原全景画质损失。最终区域Loss仍需在loader中按面积下采样到实际latent网格。

train/val清单为新精简格式，**不能直接传给旧oracle_train/fourier_world训练器**。完成CPU数据构建后，还需接入新loader、类别文本语义缓存、原生Local条件/目标VAE缓存，以及对象区域Loss；这些GPU步骤本入口不执行。

## 完成与失败

目标为7521 train /925 val。只有BUILD_RESULT.json的all_candidates_built=true、counts达到目标且failures为空，才表示候选字段全部生成。弱纹理/反射等视点可能配准不足，明确记录失败，不伪造对象或静默当作完整全集。失败后进程非零退出； inspect failure.json后决定修复，重跑同一命令会保留成功项并重试失败项。候选全部生成也不是逐图人工空间认证。

相机仍是house P位置加图像旋转配准，未新增传感器深度/mesh SE3校准。因此需看首样本预览；不能以有mask或匹配数达标认定像素级精确无误。

## 已做CPU验证

在三个建筑的新样本上验证完整新建，分别得到21、64、23个对象；验证一条32对象状态复用；并行两个worker正常完成train/val样本。固定既有camera basis后，新mesh投影与原instance/valid数组逐像素一致率均1.0。另将已知样本全量重建，新旧旋转差约0.151度、共同对象中心最大差约1.3cm；一个微小对象因可见阈值发生变化，因此默认仍精确保留已选入的旧状态。

这些是小规模真实CPU集成检查，不是全量构建结果。完整新数据由用户执行，助手未启动全量或任何GPU任务。
