# 六张验证图：局部观测与视野外状态补全

2026-10-20 实际更新：run001有2例完成、4例接口报错。已核验并修复官方类别桥接及above/below非适用参数处理，CPU重放写入run002；原批次保留不改。run002已有4例完整状态，另2例已有二维观测、等待WildDet3D和completion。详见`research/audits/box_world_val6_failure_fix_2026-10-20/REPORT.md`。当前请使用run002，勿在改代码后强行续写原run001。

## 各模块的作用

1. Qwen3-VL：读取局部RGB，输出可见对象类别、实例ID、二维框。二维框表示图像位置，不包含米制深度和大小。
2. WildDet3D：读取同一RGB、由HFOV计算的相机内参、二维框及类别提示，使用预训练模型估计可见对象的三维框。它不是仅用二维坐标做确定性转换。当前用官方`geometric: category`模式获得每个提示的一对一候选，再将原生框角点换成项目相机系的完整AABB。候选可能对应错对象，必须看二维提示与模型选框是否一致。
3. Qwen3-VL：同时读取原RGB和固定G_obs，提出视野外对象、完整尺寸、功能分组及相对摆放；当前输出声明式JSON布局。
4. 本项目`placement_program.py`：普通Python/NumPy计算模块，无模型权重，不训练。按相机区域、包围盒边界间距、观测camera-y参照计算新增xyz。视野外位置由构造保证，框重叠只读报告。它不是WildDet3D，也不是完整LayoutVLM优化器，不能证明地板、墙体或物理支撑。
5. 保存`G_full = G_obs + 新增对象`，17个或其他数量的观测对象均不被补全过程移动。Adapter最终读取的仍是类别、中心xyz、完整尺寸xyz和原有效性掩码。

官方依据：[WildDet3D VLM demo](https://github.com/allenai/WildDet3D/tree/main/demo/vlm)、[官方推理说明](https://github.com/allenai/WildDet3D/blob/main/docs/INFERENCE.md)。当前程序是本项目的工程适配，不宣称完整复现LayoutGPT/LayoutVLM。

## 冻结样例

已准备`qwen_edit_pano/data/box_world_val6_v1/inputs.jsonl`和输入预览`inputs.jpg`。6张图来自现有验证集4栋建筑，包含另一间卧室、客厅、厨房、浴室、桑拿房和走廊；没有复用当前run008的卧室观测。选图根据输入RGB的场景多样性，选择发生在新预测之前，不是随机抽样或独立测试集统计。

图片来源、split和哈希保存在`selection.json`。只从现有PNG或原skybox压缩包读取Local RGB，不读取场景GT框或GT ERP。界面显示用的房间标签不传给任何模型。走廊/桑拿房用于暴露稀疏或特殊场景下的误补全，不应为了“全景完整”强制增加若干家具；新增对象列表允许为空。

## 用户手动执行

无需新环境、安装或下载。三个阶段分别调用已经跑通的原CLI：

```text
semantic_world --inventory-only [qwen3vl环境]
box_observer --source 本图inventory/G_obs.json [wilddet3d环境]
semantic_world --completion-mode program --reuse-state 本图observer/G_obs.json [qwen3vl环境]
```

每个阶段是单独进程，Qwen和WildDet3D不会同时占用显存。整个批次只申请一个GPU allocation，内部没有嵌套srun。为复用原有接口，每图按顺序加载/释放模型；因此有重复加载开销，当前不为节省加载时间重构模型代码。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# CPU预检；已由助手执行通过，可自行复核。
bash qwen_edit_pano/world_pipeline/completion/run_box_world_val.sh preflight \
  --output qwen_edit_pano/outputs/box_world_val6_run002

# 唯一需要用户提交的GPU步骤；GPU1可自行替换。
bash qwen_edit_pano/world_pipeline/completion/run_box_world_val.sh run \
  --nodelist=GPU1 \
  --output qwen_edit_pano/outputs/box_world_val6_run002

# 可单独执行CPU汇总，包括超时中断后已有的部分结果；不重新跑模型。
bash qwen_edit_pano/world_pipeline/completion/run_box_world_val.sh review \
  --output qwen_edit_pano/outputs/box_world_val6_run002
```

run会先CPU预检再srun。CPU节点上依赖可能提醒CUDA不可用；以最后`preflight_passed`判断预检，不表示GPU推理已经执行。若部分样例失败，其余样例继续，最终进程返回非零，失败详情在summary和日志中。输入/代码未改变时，重跑相同run命令会核对哈希并跳过已完成阶段；失败阶段写到新的编号目录，保留旧日志和原始回复。代码或输入改变需换新批次输出目录。

## 可视化和状态文件

运行后打开`qwen_edit_pano/outputs/box_world_val6_run002/index.html`，每例包含：

- `review.jpg`：原图、Qwen二维框、WildDet3D三维AABB投影、补全俯视布局、camera-y高度区间、提示框与检测器选框对照。
- `scene_3d.html`：可拖动旋转/缩放的完整三维框，蓝色观测、橙色假设、红点相机、灰线输入视野。图例可以切换观测/新增对象组，hover显示ID、类别、原相机xyz及完整尺寸。
- 三维图为显示方便把相机y放在屏幕竖向；坐标轴明确标注，未修改状态xyz，不代表测得重力方向。没有画假定地板或房间墙面。
- 每个完成阶段的`G_obs.json/G_full.json`、原始Qwen回答和独立日志。
- `summary.csv/summary.json`：数量、失败、视野外检查和重叠风险，不给未经验证的语义正确率。

Plotly已存在于qwen3vl环境，交互HTML包含本地脚本，不依赖CDN或网络。当前run008的实际结果已额外生成交互预览：`research/audits/box_world_val6_setup_2026-10-20/run008_scene_3d.html`；它不是六张新图的预测。

优先检查：二维框是否识别对对象；三维投影是否与对应图像区域相容；补全是否符合房间用途且未与可见证据冲突；高度/间距是否粗略合理。状态文件写出、模型返回一个三维框、视野外检查通过，都不等于语义或完整场景已验收。

本批仅验证状态链路，不生成全景、不训练。CPU回归、环境预检及保存回答重放已完成；run002中两例尚未执行的GPU步骤仍由用户手动运行。`recovery.json`记录原批次哈希、复制和重放来源；重放保留原prompt版本，不声称它们是新prompt下的新预测。
