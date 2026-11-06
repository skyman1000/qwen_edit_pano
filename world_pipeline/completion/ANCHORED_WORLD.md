# 当前实验：局部观测 → 相机周围的完整场景假设

2026-10-20。Qwen3-VL-8B + 已有 WildDet3D + 本项目锚点偏移编译器。当前新入口是 `--completion-mode anchored`；历史 program/boxes/planned/direct 留作对照，不把它们的旧结果改写成新方法的结果。

这是一条项目组合方法，不是安装/完整复现 LayoutGPT、LayoutVLM 或 Holodeck。

- 参考 LayoutGPT：固定部分对象，在匹配训练示例下续写场景。用当前 JSON/相机坐标适配其结构化布局思想；没有已知真实 floor plan，也没有照搬其房间归一化坐标。
- 参考 LayoutVLM/Holodeck：语义选择与数值执行分工、使用明确对象引用。当前只编译位置，不采用它们完整的资产、物理/可微优化或仿真系统。
- 实际推理模型仍是 Qwen3-VL-8B。WildDet3D只估计观测几何；没有调用其他专用布局模型。ReSpace/ATISS/扩散布局器不是当前必需依赖。

## 已修改的核心

旧 program 强制整个家具组在 left/right/back 某个视野外半空间，且已有对象只参与 camera-y 高度参考。这个实现能构造视野外框，但不足以自由规划完整场景。

新入口不再强制三种区域、不整体平移、不设置方位数量配额。Qwen 同时输出新对象、完整尺寸和定位参考：

```json
{
  "room_type": "bedroom",
  "scene_hypothesis": "One plausible continuation of the visible room.",
  "additions": [
    {
      "id": "new_01",
      "category": "cabinet",
      "size": [1.2, 1.8, 0.6],
      "anchor_id": "observed_0000",
      "offset_m": [-3.0, 0.2, -4.0],
      "description": "storage cabinet",
      "reason": "Storage positioned relative to the observed furniture.",
      "unseen_reason": "outside_fov"
    }
  ]
}
```

数值仅为语法示例。最终中心 = 参考对象三维中心 + offset_m。偏移始终沿相机 x右/y图像上/z前，单位米，不随锚点家具旋转。`__camera__` 表示相机原点。也可引用另一显式新增对象，列表顺序不限，位置引用链须能连接到固定观测或相机；这是位置计算依赖，不是物理支撑图。

编译器只做明确的向量加法，不会把不合理位置自动挪走。图片投影不一致与AABB相交仍记录为诊断。AABB相交不自动等于真实碰撞，AABB顶面不自动等于桌面/座面；不以此强制删对象、挪动物体或宣称物理正确。`occluded` 仍属未验证假设，不能通过换标签把视野检查失败当成功。

G_obs 的所有对象字段逐值保留。新模式不经历史相机高度/地板布局初始化，G_full也不新增猜测地板。绘图外框是显示包围范围。最终输出仍为相机中心/完整camera-AABB，不是ERP平面二维框；生成器继续直接输出2:1 ERP RGB。

## 类别边界

当前Adapter以固定类别ID查embedding，只读取类别、中心xyz、完整尺寸xyz和有效位/mask。它不是开放词汇编码器，不读取description/anchor/关系文本。

新入口提示只用37个规范类别名；意外出现细分类名时，只接受本项目绑定的官方 Matterport TSV 中无歧义的映射，并保留原始标签与来源。未知类别不分配新ID、不静默扔到misc、不丢对象。

实际映射：nightstand→chest_of_drawers(13)，desk lamp→lighting(28)，clock/candle→objects(39)，raw wardrobe→furniture(36)。不要将另一预训练模型的旧别名表（wardrobe→cabinet）误当官方原始标签映射。直接规范类别cabinet仍然是ID7。

粗类别映射只让接口可用，不提供细分类别控制：clock映射objects后，Adapter不会从description得知它一定是钟表。单改词汇表不能给冻结Adapter增加已学到的新语义。

## 匹配示例与数据使用

已在CPU准备 `qwen_edit_pano/data/anchored_scene_examples_v1/pool.json`：629个候选，来自现有16栋train建筑，不更新权重。候选为同主房间的家具/固定设施子集；真实输入来自GT可见对象，续写来自ERP可见且当前视野外的对象。它们只是上下文示例，不是完整GT场景，也不是针对当前预测的标准答案。

运行时按可见主要家具类别的交并比检索，最多3条、来自不同训练建筑；没有匹配则不用示例。不是按未知的真实房间尺寸检索，也不强行套卧室例子。只访问train来源，排除目标照片及manifest中的heldout建筑，校验源图状态、hash、FOV和词表。目标验证场景GT不作为输入。

示例使用同一anchor/offset协议，从原始训练中心计算偏移；显示到毫米以缩短上下文。这并不是要求模型达到毫米精度。

## 手动执行

下面复用六例中已有客厅的9个WildDet3D观测对象。只运行一次新的Qwen场景请求，不重跑检测、不训练、不下载。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 --hfov 90 --completion-mode anchored \
  --image qwen_edit_pano/outputs/box_world_val6_run002/X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e/observer_001/local.png \
  --reuse-state qwen_edit_pano/outputs/box_world_val6_run002/X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e/observer_001/G_obs.json \
  --layout-examples qwen_edit_pano/data/anchored_scene_examples_v1/pool.json \
  --output qwen_edit_pano/outputs/box_world_living_anchored_run001
```

GPU1可替换，输出必须新目录。`preflight`代替`run`可以只做CPU预检。若要比较ICL作用，在另一个新目录运行同命令但去掉`--layout-examples`；相同图、G_obs、seed和布局协议。旧program是另一种布局表达，不能将新旧差异全部归因于示例。

每次输出 `anchored_scene.txt/.json/.request.json`、`completion_conversion.json`、`layout_examples.json`（如果提供示例）、`G_obs/G_hidden/G_full.json`、投影图、camera-y图、相机xz布局图及 `scene_3d.html`。后者自包含，可在本地浏览器旋转查看，蓝色观测、橙色新增。

completed表示产物写出，不表示预测正确。需要看：新增内容是否适合当前图片；布局是否与固定观测相容；定位锚点与实际偏移是否符合理由；尺寸/位置有无明显问题；然后才用已有predicted_inference.py比较无World/G_obs/G_full的同种子ERP。不要用方向均匀或零警告替代这些判断。

CPU回归和输入处理检查不会证明Qwen能规划正确的未知场景；本轮尚未执行真实GPU生成。
