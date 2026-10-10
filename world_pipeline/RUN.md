**GT小样本准备与Qwen3-VL Observer Pass A：用户手动执行**

本实现与正在运行的Edit训练隔离，代码仅在 `qwen_edit_pano/world_pipeline/`。
不修改 `qwen_pano`、`qwen_edit_pano` 顶层训练Python文件、旧manifest/cache/outputs，也不启动训练。
GT和Observer输出都是实验数据，不被当前训练入口读取。

第一步可直接在现有qwen360环境运行。第二步需要独立环境及已有/自行下载的Qwen3-VL模型。
本文所有Slurm任务均由用户手动 `srun`，没有sbatch。

**1．GT依赖/样本预检（CPU，可选）**

在项目根目录执行；节点可改成有空闲CPU资源的节点，不申请GPU。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export PYTHON_BIN=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --qos=normal --time=00:10:00 \
  "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.prepare_gt --preflight
```

本轮已检查到默认 `paired_pilot_v1` 中有9条train、2条val已有世界状态。预检不会构建mesh、写数据或加载模型。

**2．生成第一批GT与投影预览（CPU，先执行这一项）**

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=32G --qos=normal --time=02:00:00 \
  "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_pilot_v1 \
  --train-limit 9 --val-limit 2 --threads 2 \
  --output qwen_edit_pano/data/gt_world_pilot_v1
```

不需要重新下载、准备RGB配对或构建Edit条件缓存。此处用pilot只为了首先检查11个样本，不影响正在使用全量7521条的训练。

程序具体做：

1. 按官方建筑split关联现有world_state，复核类别TSV哈希。
2. 从ZIP读取原始face2，从Arrow读取当前已校准的polished ERP。
3. 独立计算旧几何ERP→face2的yaw/mirror注册，并组合其相机基；不将polished的roll重复套给旧世界坐标，不水平化Local图。
4. 对比旧ERP与polished ERP的前后左右上下六个投影；检测当前配对目标相对于face2是否仍有朝向残差。
5. 复用全场景region mesh与精确实例链接，用Open3D **CPU** 直接投射Local/ERP射线计算遮挡；未知语义面仍遮挡后方物体。
6. 从源OBB计算当前相机轴AABB全尺寸，检查角点计算与坐标往返。
7. 导出G_full、G_obs、G_hidden及可见实例框；G_obs不携带完整房间几何或GT房间类型。

输出目录结构：

```text
gt_world_pilot_v1/
  contract.json                  # 版本、坐标、阈值、源码/来源哈希
  vocabulary.json                # 固定类别词表，无样本标签
  gt_manifest.jsonl              # GT及检查结果索引
  observer_inputs.jsonl          # 仅Local RGB白名单，供第二步
  EXPORT_COMPLETE.json           # 导出完成，不是训练/视觉验收通过
  failures.json                  # 读取/构建失败及原因
  review.html                    # 全部样本的预览索引
  review_decisions.json           # 初始全为pending，由用户记录验收
  scene_sources/                 # 复用mesh/instance文件哈希
  samples/<sample_id>/
    local_rgb.png
    gt_erp.png
    gt_front.png
    legacy_front.png
    review.jpg                   # RGB对应＋mesh轮廓/ID/框
    local_overlay.png
    erp_overlay.png
    local_visibility.npz
    erp_visibility.npz
    bridge.json                  # 相机基、配准分数、多视图一致性、检查标志
    provenance.json
    G_obs.json
    G_hidden.json
    G_full.json
```

输出目录必须全新；程序拒绝覆盖已有目录。任务中断后保留部分文件用于诊断，重跑换新目录；本阶段不提供断点复用，避免混入不完整样本。
`EXPORT_COMPLETE.json` 的 `failures` 应为0。若非0，程序退出码2，先查看 `failures.json`，不要把部分导出当作完整通过。
有 `numerical_flags` 的样本也会保留预览，便于定位问题，但评测不会接受这些样本。

**3．查看投影并逐样本记录验收**

打开输出中的 `review.html`，或直接看各样本的 `review.jpg`。

| 图像位置 | 检查内容 |
|---|---|
| 上排左：Local；中：GT正前方；右：旧几何ERP正前方 | 对象、方向对应；是否有系统性偏移、倾斜或镜像 |
| 下排左：Local＋mesh轮廓/可见框 | 绿色边界是否大致落在对象边缘；黄框是否包含实际可见区域 |
| 下排右：ERP＋mesh轮廓 | 前后侧方、门口跨房间对象、接缝及顶部底部是否对应 |

配准NCC不是物理精度证书。旧相机原点仍是候选，近处拼接视差/mesh空洞可能造成真实偏差。不能因为程序运行完就批量批准。

通过某个样本后记录决定（下面ID是实际候选，但仅在你看过并确认后执行）：

```bash
"$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.review_gt \
  --root qwen_edit_pano/data/gt_world_pilot_v1 \
  --id 5q7pvUzZiYa_7dc12a67ddfc4a4a849ce620db5b777b \
  --decision approved --notes '已检查三张RGB对应和Local/ERP实例轮廓'
```

对其他已检查ID重复此命令；也可重复 `--id` 指定多个**确实逐一检查过**的样本。明显错位时用 `--decision rejected --notes '具体问题'`。这只更新review_decisions，不修改GT图或宣称可用于正式训练。

以下限制已写入公共pilot契约：

- schema=`caupano.local_world.v0.1-pilot1`，实验接口，不等于已冻结的正式schema。
- x=image-right、y=image-up、z=view-forward，米制；image-up不默认等于重力up。
- size是当前相机轴AABB的全边长；从源OBB角点得到。
- 字段有value/valid/source，未知用null；Observer不会填造3D坐标。
- 可见性格点固定Local512×512、ERP512×1024，实例至少16像素；与原图/生成分辨率解耦。
- 如果Local达门槛而ERP未达，保留对象并标记待解决，禁止自动认证。
- room归属冲突仅使相应字段无效；未把有效物体几何一起删除。
- G_obs的可见对象3D仍是真实完整OBB转换的oracle信息，强于单图Observer输出。
- `G_hidden` 的房间几何是补全监督目标，不可作为Completion的观测输入。
- 不生成关系、语义朝向或可见性置信概率；GT对象ID只用于匹配/评估。

**4．Observer独立环境（第二步，可在空闲GPU上开展）**

第一步GT通过前也可做Observer推理，看类别/2D框；但定量评测只接受第3步已验收且无numerical_flags的GT。
本版实现Pass A：可见对象类别＋2D框＋自由文本room候选。**没有实现米制3D Pass B、Observer微调或Completion。**

先在终端创建新环境，复用现有torch安装。不要在qwen360内直接安装新transformers。

```bash
conda create -y \
  --prefix /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl_observer \
  --clone /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360

export OBS_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl_observer/bin/python

srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --qos=normal --time=00:30:00 \
  "$OBS_PY" -m pip install 'transformers==4.57.1'
```

已有相同独立环境则跳过创建，运行下面预检。4.57.1提供Qwen3VLForConditionalGeneration；使用SDPA，不要求编译FlashAttention。[对应官方接口文档](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl)

如果Qwen3-VL-8B-Instruct尚未缓存，才由你执行这条CPU下载命令，沿用用户HF缓存，不下载到项目：

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --qos=normal --time=04:00:00 \
  env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$OBS_PY" -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-VL-8B-Instruct"))'
```

若已经下载到别处，可以在后续 `--model` 传已有snapshot绝对路径，不必重下载。本程序加载模型始终 `local_files_only=True`，不隐式联网。run_config记录实际snapshot路径，重复实验可直接指定该路径固定revision。

CPU预检：

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --qos=normal --time=00:10:00 \
  "$OBS_PY" -m qwen_edit_pano.world_pipeline.observer \
  --inputs qwen_edit_pano/data/gt_world_pilot_v1/observer_inputs.jsonl \
  --vocabulary qwen_edit_pano/data/gt_world_pilot_v1/vocabulary.json \
  --model Qwen/Qwen3-VL-8B-Instruct --preflight
```

**5．Observer推理（GPU，由你执行）**

更新：下述compact提示词+1.05方案已在第二轮失败，停止推荐；请按 `OBSERVER_DIAGNOSIS_2026-10-02.md` 做有限A/B对照。使用 `--prompt-version baseline_v1 --bbox-scale 1000 --repetition-penalty 1.0`，分别 `--decoding greedy` 与 `--decoding checkpoint`，均仅2图、新目录。暂无已验证成功的Observer配置。

2026-10-02实际两图诊断：BF16加载正常；首图重复不完整对象至4096 token，第二图输出0–1000框而旧契约要求0–1，两图均被严格校验拒绝。新增显式 `--bbox-scale 1000`（按约定除以1000保存公共图，绝不猜测尺度）、紧凑完整JSON提示，以及可选 `--repetition-penalty 1.05`。建议先用这两个参数和 `--limit 2` 在全新输出目录复测。重复惩罚属于实验设置，可能影响类别和数字输出，尚未经GPU验证；通过JSON校验不代表检测准确。旧失败输出保持不变。新运行记录generation.json中的token数量及截断状态。

选择没有被全景训练占用的GPU节点；以下GPU0只是示例。资源与时限是申请值，尚未在本机实测8B推理峰值。

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
  "$OBS_PY" -m qwen_edit_pano.world_pipeline.observer \
  --inputs qwen_edit_pano/data/gt_world_pilot_v1/observer_inputs.jsonl \
  --vocabulary qwen_edit_pano/data/gt_world_pilot_v1/vocabulary.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --image-size 768 --max-new-tokens 4096 \
  --output qwen_edit_pano/outputs/observer_pass_a_pilot_v1
```

默认运行全部导出样本。首次想只测试两个，可加 `--limit 2`，并使用单独输出目录；这两个按清单顺序取，不保证覆盖train和val。后续全量pilot须用新输出目录。

Observer只读取白名单manifest中Local PNG及哈希、全局类别词表，不读取GT结构/GT ERP。图像缩到最长边768，框统一归一化到[0,1]，不受此等比resize影响。输出使用greedy解码，固定参数便于定位错误，不声称是最优质量配置。

每个样本包含 `raw.txt`、`status.json`；成功解析时还有 `G_obs.json` 和 `boxes.jpg`。3D字段均为null，room候选不会直接冒充Matterport房间代码。不合法JSON、错误坐标尺度或达到输出token上限会被明确记录；不静默修框或猜单位。

**6．Observer与人工验收GT对比（CPU）**

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --qos=normal --time=00:10:00 \
  "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.evaluate_observer \
  --gt qwen_edit_pano/data/gt_world_pilot_v1 \
  --predictions qwen_edit_pano/outputs/observer_pass_a_pilot_v1 \
  --review qwen_edit_pano/data/gt_world_pilot_v1/review_decisions.json \
  --iou 0.5 \
  --output qwen_edit_pano/outputs/observer_pass_a_pilot_v1_metrics
```

没有人工approved样本时会拒绝评分。train/val分开统计，不使用test。
`metrics.json` 包含类别一致且IoU≥0.5的最大匹配precision/recall/F1、每类统计、每图匹配、unknown与格式失败数。
这是小样本检测诊断，**不是COCO AP或正式泛化成绩**；暂不评价3D和room type准确率。格式失败按该图GT全部漏检计，同时单列；不可解析文本的FP无法可靠统计。GT过滤小实例/结构类且mesh可能不完整，因此还要对照boxes.jpg人工检查错误来源。

先用这些结果判断可见对象识别和定位是否可用，再实现/评测3D Pass B。不要把G_obs.json的存在当成字段可靠性证明。

**可选扩展与本轮验证范围**

11条通过后，可基于全量配对清单另建较大的GT pilot，例如32 train＋8 val：

```bash
srun -p debug --nodelist=GPU0 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=32G --qos=normal --time=04:00:00 \
  "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_full_v1 \
  --train-limit 32 --val-limit 8 --threads 2 \
  --output qwen_edit_pano/data/gt_world_expanded_v1
```

选择在已有world身份交集中按建筑轮流取样；本轮不会要求重新标注或下载整套数据。扩大数据前应先解释小样本的flags，不能通过降低阈值掩盖错位。

开发者CPU检查命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PYTHON_BIN" -m qwen_edit_pano.world_pipeline.test_pipeline -v
```

测试包括yaw/mirror与像素roll一致、OBB/AABB全尺寸与反射、真实Open3D CPU遮挡、未知面仍遮挡、合成场景完整单样本导出、G_obs房间信息隔离、输入白名单、Observer严格解析、重复/错类匹配及人工验收评分门槛。
另已进行一条实际样本的CPU图像配准检查，旧ERP→face2为无镜像/零roll，六视图NCC约0.952–0.999；这是单样本RGB检查，不是全量3D验收。
没有执行完整11条GT渲染、没有安装Observer环境/下载模型，也没有运行真实Qwen3-VL GPU推理。完整数据导出和GPU步骤由用户执行。
