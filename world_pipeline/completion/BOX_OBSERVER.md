# 单图物体框入口：只输出 Adapter 需要的类别、位置、大小

2026-10-20 更新。用户已手动跑通 `box_observer_run001`：17 个数值框来自 WildDet3D。`box_world_run004` 新增的 4 个框全投影在局部图内；`box_world_run005` 文字修订未改变 JSON；`box_world_run006_icl` 加入三个训练布局示例后改为 5 个新候选，但 5/5 仍投影在照片内，并有明显的 AABB 重叠风险。实际 `box_world_run007_program` 的 3 个新增框均在视野外、无AABB重叠，但椅子/边桌高度明显缺少观测参照，“靠窗”描述也未落实。当前只改高度表达，待测 program v2；没有据零告警宣称完整场景可用。

WildDet3D 负责可见物体的数值几何，Qwen 负责未观测内容假设；复用检测结果不会再次加载 WildDet3D。当前图无需重新安装、下载或重新检测。以下首次安装段落仅供新环境参考。

## 路线与依据

待测简化路线：`RGB → Qwen 类别/2D框 → WildDet3D 三维框 → 固定 G_obs → Qwen 语义分组/相邻关系 → 几何编译 → G_full`。

实际送进 Adapter 的信息仍然只有 category_id、center_local_m[3]、size_aabb_local_m[3]，另有原有有效性掩码。模型内部的四元数、2D框、置信分数仅用于获得与检查这六个数字，不增加 Adapter 输入。

联网核验的官方方法：

| 方法 | 官方输入与输出 | 本次选择 |
|---|---|---|
| [WildDet3D](https://github.com/allenai/WildDet3D) | 单 RGB，可给类别或2D框，预训练模型直接输出米制3D框 | 当前观测入口；开放词汇、可以复用现有对象清单 |
| [官方 VLM demo](https://github.com/allenai/WildDet3D/tree/main/demo/vlm) | 明确使用 Qwen3-VL-8B 的2D框引导 WildDet3D | 与当前工程直接对应；复用已有2D结果，无需vLLM服务 |
| [Cube R-CNN / Omni3D](https://github.com/facebookresearch/omni3d) | 单 RGB + K → 类别、3D中心、旋转框角点；有预训练权重 | 可作固定类别基线，但全量模型50类不覆盖项目37类全部对象，例如plant；依赖较旧 |
| [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) | 单 RGB 的开放词汇3D检测；其公开 demo 还依赖预生成深度/内参 | 可参考，当前不再叠加独立深度环境 |
| [Qwen3-VL 3D cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb) | 官方支持原生3D框 | 功能存在；项目run014数值结果未通过，不等于模型不支持 |
| [Total3DUnderstanding](https://github.com/GAP-LAB-CUHK-SZ/Total3DUnderstanding) | 单图联合布局/物体框/网格 | 可以借鉴框几何，当前无需整套移植 |
| [SpatialLM](https://github.com/manycore-research/SpatialLM) | 点云→结构与物体框 | 不能直接替代纯RGB入口 |
| [LayoutGPT §5.3](https://layoutgpt.github.io/static/layoutgpt_arxiv.pdf) | 已存在对象/布局条件下的场景补全 | 继续借鉴其“固定已有对象、增加合理家具”思想；它不解决RGB观测几何 |

这些方法的适用性判断基于官方输入输出和实现，不是本项目上各模型的性能排名。WildDet3D也不能保证仅凭一图恢复真实尺度；此次应实测粗框是否已够用。

## 几何桥接（核对了实现，不直接照搬维度名）

[官方推理说明](https://github.com/allenai/WildDet3D/blob/main/docs/INFERENCE.md)、[coder_3d.py](https://github.com/allenai/WildDet3D/blob/main/wilddet3d/head/coder_3d.py)、[官方画框实现](https://github.com/allenai/WildDet3D/blob/main/wilddet3d/vis/visualize.py)、[vis4d v1.0.0](https://github.com/SysCV/vis4d/blob/v1.0.0/vis4d/op/box/box3d.py)交叉核验：

- 原生中心在OpenCV相机系：x右、y下、z前，单位米。项目只翻转y符号，保留相机原点，不假装相机已水平。
- 官方说明表格将三个维度写成w/h/l，但实际coder与vis4d使用W/L/H，原生未旋转框的相机XYZ边长为L/H/W。不能把原生三个数字直接当作Adapter的size_xyz。
- 本入口调用同一 `boxes3d_to_corners(..., AxisMode.OPENCV)`，然后对转换到项目坐标的8个角点计算 `center=(max+min)/2`、`size=max-min`。保留旋转框的外接AABB，不输入旋转。
- 原始图像K由给定HFOV计算；原始像素2D框交给官方API。resize/padding完全由官方preprocess/wrapper负责，避免重复缩放K或框，也不缩放米制坐标。
- 每个ID一次 `geometric: category` 请求，沿用Qwen分类，不重新匹配不同物体。几何模式仍可能选择到错误实例；记录返回2D框、原框IoU和分数供检查，不把“返回一个框”当作已验证。
- 不做物体落地、重力旋转、形状生成或布局优化。为兼容已有completion，G_obs附带一个包住相机和框的显示范围，明确标记它不是房间重建；Adapter不读取该范围。

## 手动安装与下载（仅第一次）

以下由用户执行。独立环境，保留现有qwen3vl/qwen360。安装按[官方README](https://github.com/allenai/WildDet3D#installation)的Python/PyTorch/vis4d版本。此服务器当前默认nvcc为12.0，所以下面给独立环境安装12.1工具链，与官方cu121构建相配；无需改系统CUDA。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
conda create -n wilddet3d python=3.11 -y
conda activate wilddet3d
conda install -c nvidia cuda-toolkit=12.1 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4

git clone --recurse-submodules https://github.com/allenai/WildDet3D.git qwen_edit_pano/third_party/WildDet3D
git clone https://github.com/SysCV/vis4d_cuda_ops.git qwen_edit_pano/third_party/vis4d_cuda_ops
python -m pip install 'setuptools<80' wheel ninja
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install vis4d==1.0.0

# 编译器构建脚本检查torch.cuda.is_available()，需要在有GPU的节点编译。
# GPU1可换成你指定的节点。源码已下载，此步用本地源码构建。
srun -p debug --nodelist=GPU1 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=24G --qos=normal --time=01:00:00 \
  "$CONDA_PREFIX/bin/python" -m pip install --no-build-isolation --no-deps \
  ./qwen_edit_pano/third_party/vis4d_cuda_ops

python -m pip install -r qwen_edit_pano/third_party/WildDet3D/requirements.txt

# 下载两个文件；第二个虽只用于读取模型配置，当前官方实现仍需要缓存它。
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('allenai/WildDet3D', 'wilddet3d_alldata_all_prompt_v1.0.pt',
                local_dir='qwen_edit_pano/third_party/WildDet3D/ckpt')
hf_hub_download('robbyant/lingbot-depth-postrain-dc-vitl14', 'model.pt')
PY
```

若终端以前设置过 `HF_HUB_OFFLINE=1`，先在这个下载终端 `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`。推理入口自己设置离线模式。完整checkpoint约4.7GB，LingBot配置所在checkpoint约1.28GB。无需单独下载SAM3权重、训练数据、物体资产或MoGe。

用户已在独立 wilddet3d 环境完成实际推理，现有环境无需重复安装。安装配方不等于所有机器均已验证；换机器时以手动安装与预检为准。

## 当前图片直接执行

当前17个对象的类别/2D框可以直接复用；旧三维数字完全不使用。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/completion/run_box_observer.sh check

bash qwen_edit_pano/world_pipeline/completion/run_box_observer.sh preflight \
  --source qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json \
  --hfov 90 --output qwen_edit_pano/outputs/box_observer_run001

bash qwen_edit_pano/world_pipeline/completion/run_box_observer.sh run \
  --nodelist=GPU1 \
  --source qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json \
  --hfov 90 --output qwen_edit_pano/outputs/box_observer_run001
```

`BOX_PY=/绝对路径/python`可覆盖独立环境Python。预检只检查本地文件、依赖、缓存和CPU角点转换，不加载模型、不用GPU；不代表GPU推理或精度通过。

输出：`objects.json`是最小的`category / position / size`列表；`G_obs.json`是现有Adapter可读取的包装。`native_boxes.json`记录原生框、角点、分数和框匹配IoU；`instance_binding.png`中红色是Qwen输入框，绿色是检测器选择的框；`projection_boxes.png`显示最终Adapter AABB投影。任何单个对象失败会保留之前的原始结果，禁止回退到旧Qwen数值冒充新预测。

完成后先做已有CPU对照，再决定是否补全，无需先追求精确重建：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python -m \
  qwen_edit_pano.world_pipeline.completion.audit_scene_consistency \
  --prediction qwen_edit_pano/outputs/box_observer_run001/G_obs.json \
  --output research/audits/box_observer_run001_gt
```

重点查看床/窗/画框是否恢复合理的正面宽度、框是否对应同一物体、同类别同实例下的米制误差是否比run014降低。GT对照保持原有粒度限制，床可能含多个部件，不能机械要求对象数量一一对应。

历史 boxes 基线让 Qwen 一次联合提出新增对象、三维中心和完整尺寸，观测状态保持不变。run004～006 表明这一路径本图尚不可用，以下仅供保留对照，不建议继续反复运行同类自由 xyz 提示。

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --hfov 90 --completion-mode boxes \
  --reuse-state qwen_edit_pano/outputs/box_observer_run001/G_obs.json \
  --output qwen_edit_pano/outputs/box_world_run001
```

这一步仍是合理场景假设，不是探知真实视野外场景。借鉴LayoutGPT由语言模型输出数值布局的设计，但不是完整复现其示例检索/房间坐标实验。boxes模式保留模型坐标并做原有只读投影诊断，遮挡并未经过真实深度遮挡验证，当前Adapter对预测G_full的全景遵循能力仍需要另外实测。

### 当前下一步：保留语义分组与几何编译，补上观测高度参照

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --hfov 90 --completion-mode program \
  --reuse-state qwen_edit_pano/outputs/box_observer_run001/G_obs.json \
  --output qwen_edit_pano/outputs/box_world_run008_height_anchor
```

只请求一次 `placement_program`。Qwen 自己提出新增对象、完整尺寸、功能分组、组位于相机 left/right/back 哪个区域，以及组内边桌在椅子右边、画在新柜子上方等简单关系。它不再计算最终 xyz。一个小型 JSON 编译器按完整框的边界间距建立组内布局，再将整组放入选定的相机视野外半空间。切向位置与余量连续可选，不是三个固定射线或半径网格。

这是借鉴 LayoutVLM 的“语义关系交给模型、数值几何交给程序”的本项目简化版本，不是完整复现其渲染/可微求解器。编译器不执行模型生成的代码，不需要资产或新权重。17 个观测状态不变，组内相对位置不因外部摆放改变，不做重力对齐或地板修正。

run007 的 v1 根对象直接预测 `center_y_m`，椅子中心 +0.35m，边桌中心也为 +0.35m；观测床/床头柜中心分别约 -1.10/-1.50m。v2 `camera_region_group_observed_y_anchor_v2` 不再独立猜根高度：Qwen 输出 `height_anchor_id`（现有观测ID）、`height_alignment`（bottom/center/top）、`height_offset_m`（有符号相机y偏移）。程序对齐相同AABB边界/中心后加偏移。例如底边对齐时 `new_y = ref_y - ref_size_y/2 + new_size_y/2 + offset`。横向子物体显式选择 `vertical_alignment`，可按底边对齐不同高度的桌椅。above/below 的边界间隙语义保留，不偷偷转换成支撑。

这些是内部布局字段；最终Adapter仍只接收类别、中心xyz、完整尺寸xyz和原有有效掩码。观测camera-y边界只是粗相对高度先验，绝不是测得的地板：相机倾斜、AABB膨胀和观测误差仍影响结果。新输出增加 `camera_y_refined.png` 高度区间预览，避免只看x/z俯视图；`completion_conversion.json.height_references` 记录实际参照与高度。

此模式先处理真正的视野外补全，不覆盖视野内遮挡推断。相邻关系只引用同组已有的新对象 ID；不构造全部对象两两关系图，不把 above 当支撑。组之间、与观测框之间的剩余 AABB 重叠会明确报告，不自动平移、缩小或删除对象。

`outside_fov` 通过是几何构造的结果，不能拿它证明 Qwen 已学会三维规划或完整场景合理。还必须检查功能选择、对象数量、尺寸、各组位置及 `suspected_aabb_overlap_pairs`；重叠仅为保守风险，不能等同于网格碰撞。run007复核见 `research/audits/box_world_run007_program_review_2026-10-20/REPORT.md`。不要继续改相机高度常数或反复追求零告警。

用户已实际运行v2 `box_world_run008_height_anchor`：Qwen选择observed_0012床头柜作为高度参照，椅子/边桌camera-y下边界均为-1.87065m；新增3个框位于视野外，无AABB重叠，17个WildDet3D观测完整保留。本图可进入粗原型的全景控制诊断；这不是地板/物理支撑或跨图泛化认证。当前不再修改补全算法，下一步用已有Adapter作同图同种子的baseline/G_obs/G_full对照，执行方法见[预测状态到全景](../fourier_world/PREDICTED_INPUTS.md)。

最终 Adapter 字段仍然只有类别、相机 xyz 中心、完整 AABB xyz 尺寸及既有有效性掩码；分组与关系属于推理中间表示。

### 历史：训练布局示例引导，run006_icl 实测未通过

LayoutGPT 官方实现会以 user/assistant 对话提供训练布局示例，再提出当前布局问题。当前代码增加可选 `--layout-examples`，复用这一具体设计；不是声称完整复现 LayoutGPT 的房间坐标、给定房间尺寸或检索实验。现有 WildDet3D 观测、Adapter 字段和只读诊断均保持原流程。

已用 CPU 准备 `qwen_edit_pano/data/bedroom_layout_examples_v1/bedroom.json`，来自 ULsKaCPVFJR、5q7pvUzZiYa、759xd9YjKW5 三栋训练建筑。只取已验证属于同一主房间的家具级对象，不把邻近房间的对象混入；输出示例使用真实视野外且全景中有像素的物体。坐标按本项目 camera-local/full AABB 展示到厘米，文件记录 source graph、对象 ID 和哈希，加载时从源数据复算并拒绝被篡改的示例。不会读取当前验证场景 GT。这些是家具子集示例，不是完整房间逐对象标注；不限制当前预测类别或数量。

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --hfov 90 --completion-mode boxes \
  --reuse-state qwen_edit_pano/outputs/box_observer_run001/G_obs.json \
  --layout-examples qwen_edit_pano/data/bedroom_layout_examples_v1/bedroom.json \
  --output qwen_edit_pano/outputs/box_world_run006_icl
```

这次只调用一次 `scene_boxes`，不传入 run004/005 的旧候选；Qwen 自己重新决定新增物体及布局。三个示例以独立的 user/assistant 对传入，当前照片和问题在最后一条 user 消息中，不混成“全部保持不变”的候选列表。`G_obs` 原值不会按示例取整或修改。

这是无训练的 few-shot 实验，**实际使用其他训练场景的 GT 作为提示示例**，配置明确记录 `training_gt_examples_used=true` 和 `target_scene_gt_used=false`。本示例库仅供当前卧室案例；不同房间用途不能直接套用卧室示例。重建同一示例库的 CPU 命令为 `python -m qwen_edit_pano.world_pipeline.completion.layout_examples --output 新目录`，当前库已存在，无需重建。

CPU 已核对官方处理器实际编码：5944 input tokens、1 张当前图片、7 条 user/assistant 消息；无模型权重加载。尚未运行 Qwen 推理，不能承诺示例一定改善。若结果可用，再与同版去掉 `--layout-examples` 的新生成比较，才能归因示例收益；当前单图成功也不代表跨图有效。

### 历史：run004 的受控纠错实验，run005 实测未改善

保留 17 个 WildDet3D 观测框与 4 个新增对象的 ID/类别，只给 Qwen 一次真实投影反馈，重新提出新增框的位置、大小和摆放理由。不会删除对象或将 outside_fov 改成 occluded 来减少警告，也不会直接推移坐标。若修订仍错，保留错误并报告，不能承诺此次模型一定成功。

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --hfov 90 --completion-mode boxes \
  --reuse-state qwen_edit_pano/outputs/box_observer_run001/G_obs.json \
  --reuse-completion qwen_edit_pano/outputs/box_world_run004/scene_boxes.json \
  --output qwen_edit_pano/outputs/box_world_run005
```

只请求 `scene_boxes_revision`；不会重新请求 observation、observed_layout、room_contents 或 scene_boxes。输出 `completion_initial.json`、`scene_boxes_revision.*`、`completion_feedback.json`、`G_full.json` 供对比。`completion_conflicts_after=0` 只表示该项视野外检查通过，仍须检查布局是否合理，不能证明物体落地、无碰撞或全景生成有效。

历史planned模式若已生成`room_contents.json`但在后续验证/布局阶段失败，可以在新输出目录添加
`--reuse-contents 旧输出目录/room_contents.json`继续几何与摆放。必须同时使用
`--completion-mode planned --reuse-state .../G_obs.json`，并且原计划绑定的数值G_obs完全一致。
这会保留原始内容计划，避免重新请求Qwen改变对象；不是跳过关系验证。

## 换一张图片

新增 `--inventory-only` 让Qwen只做类别/2D框，跳过所有旧三维猜测和补全：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --image /你的图片绝对路径.png --hfov 90 \
  --inventory-only --output qwen_edit_pano/outputs/new_image_inventory001

bash qwen_edit_pano/world_pipeline/completion/run_box_observer.sh run \
  --nodelist=GPU1 --hfov 90 \
  --source qwen_edit_pano/outputs/new_image_inventory001/G_obs.json \
  --output qwen_edit_pano/outputs/new_image_boxes001
```

HFOV应使用实际值或明确的估计值。后续completion的`--image`也需指定这张新图片。不要对新图片运行默认绑定旧图的GT审计。
