# GT 可见集合与空间前缀对照

新增代码已编写；按照用户要求，本轮没有执行准备、测试或推理。所有运行步骤由用户执行。

## 执行顺序

在 DIT360 根目录，继续使用已有 qwen3vl 环境：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/completion/run_visible_probe.sh check
bash qwen_edit_pano/world_pipeline/completion/run_visible_probe.sh prepare
```

check 仅运行合成几何测试，不加载模型。prepare 仅使用 CPU：读取已批准 Oracle Bundle 的 val 行、源 OBB、已有 G_obs/bridge 和官方 ZIP，不修改原训练数据。

查看 prepare 打印的实际样本数，并打开：

```text
qwen_edit_pano/data/m3dlayout_visible_probe_v1/audit.json
qwen_edit_pano/data/m3dlayout_visible_probe_v1/review.html
```

只有 prepare 成功且 review 中对象对应没有明显问题，再按顺序运行：

```bash
VISIBLE_NODE=GPU2 bash qwen_edit_pano/world_pipeline/completion/run_visible_probe.sh describe
VISIBLE_NODE=GPU2 bash qwen_edit_pano/world_pipeline/completion/run_visible_probe.sh run
```

describe 仅根据 Local RGB 生成一次简短描述；两种补全条件共用它。run 加载现有 M3DLayout 权重一次，每条样本两条件、三个 seeds，总次数为实际样本数×6。不训练 Completion，不改官方采样器。

prepare 允许直接重跑中断后留下的目录，重新生成清单；以 bundle.json 作为准备完成标记，已经完成的准备不会被覆盖。只读取 bundle.json/inputs.jsonl 中列出的样本，未被新清单引用的残留样本目录不参加实验。describe/run 仍使用新输出目录。如果要重做已经完成的实验，四条命令之前统一设置新的目录：

```bash
export VISIBLE_BUNDLE=qwen_edit_pano/data/m3dlayout_visible_probe_v2
export VISIBLE_RUN=qwen_edit_pano/outputs/m3dlayout_visible_probe_run002
```

## 选样与真实可见性

目标是原验证集每栋 3 个不同的室内房间，seed=0 固定打乱后选择首批满足数据对应条件的样本，排除先前调试三图。不足时报告实际数量，不以生成质量换图。所有未选样本原因保留在 audit.json。

官方布局无 house_object_id。脚本先通过同一房间内的类别组、半尺寸与中心对应恢复 ID，再从已有 G_obs 的 gt_house_object_id 集合传递可见性。不使用简单视锥判断或 Qwen 猜测。

桥接假设为 world z-up → native y-up 的水平轴交换/符号翻转 + 一个全房间平移。它不是任意房间配准求解器；不改变尺度，不逐对象移动，不自动拟合任意 yaw。要求所有 native 对象能唯一匹配，中心、半尺寸及变换后八角点误差均不超过 2 cm，且已观察对象中心与现有 G_obs 一致。源码若存在不同的重定向/OBB 提取约定，prepare 会明确排除，而非强行套桥接。

若没有可用样本，prepare 保存审计后退出；不要执行 describe/run，也不要为了凑够 12 个直接放宽阈值。应根据 audit 的具体失败原因核实发布数据的变换/框约定。这是输入能否对应的检查，不是模型性能筛选。

每条样本保留 source_matches.json，可查看原始 OBB 与 native 对象对应。布局图蓝色表示输入对象，灰色表示本组剩余对象。

## 两组输入

- prefix：该房间官方排序的前 N 个对象。
- visible：该房间 official inventory 中、在原 G_obs 出现的 N 个对象。

两组都直接用 native 对象的几何数值，消除几何来源差异；排序规则与前一轮 native_probe 一致。N 为实际可见 native 对象数。文本、全 1 的 64×64 room mask、类别表、统计、采样器、seed 与总解码预算一致。解码预算为 120−N（包括 END 的采样次数）。

这是“GT 可见性在官方单房间对象集合上的转移”，并不等于完整 CauPano G_obs/G_full：官方删去的对象和跨房间可见对象不进入本轮模型输入，其 ID 分别记录在 observed_outside_native_ids、observed_outside_primary_room_ids。原 G_obs 完整保留在原文件中。

两组各自的待补对象不同；common_heldout_indices 记录两组都未输入的对象。比较时分别报告各自剩余目标，也可对共同未见集合做参考，不能把不同目标的 IoU 直接当公平排名。

官方 mp3d_train/val/test 全部只用于定位相同房间，official_split 在输出中明确记录。CauPano val 不保证是 M3DLayout 预训练未见建筑；本轮是接口/条件诊断，不宣称跨建筑泛化。

## 输出

```text
qwen_edit_pano/data/m3dlayout_visible_probe_v1/
  audit.json
  bundle.json
  inputs.jsonl
  review.html
  samples/<id>/
    sample.json
    source_matches.json
    prefix_reference.png
    visible_reference.png
    native_boxes_local.json

qwen_edit_pano/outputs/m3dlayout_visible_probe_run001/
  captions/<id>/description.txt
  comparison/summary.csv
  comparison/<id>/<prefix或visible>/seed000/
    layout_scene.json
    layout_topdown.png
    generated_local_boxes.json
    raw_normalized.json
    summary.json
```

generated_local_boxes.json 是使用 GT 配准变回 Local 的新增候选框，包含完整 Local-axis AABB 长度，仅供诊断，不是完整 G_full，也不代表已经实现预测相机配准。prefix 组尤其不是图像观测条件，不能用于最终生成效果声明。

结果判断：prefix 正常而 visible 异常，优先关注可见集合条件不适配；两者都差，先核查桥接与原生行为；visible 基本合理，再在同一批图上换成 Qwen 预测实例框。保持 Adapter 字段不变。
