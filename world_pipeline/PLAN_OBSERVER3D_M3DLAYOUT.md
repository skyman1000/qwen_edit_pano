> 2026-10-11：正式实现已改为可见条件下的隐藏对象训练，当前入口和执行顺序以 [completion/TASK_COMPLETION.md](completion/TASK_COMPLETION.md) 为准。下文保留作历史方案，不代表当前全部实现状态。

# Qwen3-VL 可见 3D 对象观测 → M3DLayout 补全 → Qwen-Image-Edit：分步实施方案

日期：2026-10-10。状态：**设计文档，尚未实施。**

本文只规定下一轮开发、用户手动运行和验收流程；编写本文不代表安装依赖、下载权重、执行推理或训练已经完成。所有标注“新增”的模块和命令接口都需要以后实现。当前 Encoder/Adapter 训练继续使用原配置。

## 1. 本轮目标和结论

验证下面的分支是否能产生与局部 RGB 相容、对 ERP 生成有用的对象级世界状态：

```text
Local RGB ──────────────────────────────────────┐
    ↓                                           │
Qwen3-VL：可见对象清单 + 3D grounding              │
    ↓                                           │
Predicted G_obs                                 │
    ↓                                           │
M3DLayout AR：已有对象 → 对象续写                  │
    ↓                                           │
Predicted G_full                                 │
    ↓                                           │
字段/坐标桥接 → World Encoder + Adapter           │
    ↓                                           ↓
             Frozen Qwen-Image-Edit + Panorama LoRA
                          ↑
                    可选用户文本
                          ↓
                       2:1 ERP
```

**最终字段可以与当前 Adapter 匹配，不需要新增 Encoder 字段。** 但必须同时通过：

1. schema 和张量兼容；
2. 坐标、米制尺度和 AABB 定义一致；
3. 预测与照片一致，并在下游实验中有用。

第 1 项通过不代表第 2、3 项成立。Oracle 训练过的分支也不保证能直接适应预测噪声。

本轮不要求 Qwen 生成长描述，不要求预测完整 floor plan。主要输入为可见对象 3D 框；简短文本是官方预训练模型探测时的可选对照。用户文本与内部生成的提示分别保存，首轮下游比较不改变生成 prompt。

## 2. 已有实现与缺口

| 组件 | 当前可复用内容 | 本轮缺口 |
|---|---|---|
| `observer.py` | Local-only 清单、独立 Qwen 环境、Pass A 类别和可见 2D 框、原始输出留档 | 无 3D 推理，中心和尺寸为 unknown |
| `evaluate_observer.py` | 基于审核 GT 的类别感知 2D 匹配 | 无 3D 评测；清单哈希校验不能直接用于任意新子集 |
| `geometry.py` | 源 OBB 角点到 local AABB 的定义和数值检查 | 无 Qwen/M3DLayout 几何桥接 |
| `common.py` | `validate_graph`、schema、frame、字段包装、类别范围 | 上游预测必须填写现有必需 metadata |
| `oracle_model.py` | `tensorize`、WorldEncoder、WorldBranch | 无需改变特征契约 |
| `oracle_data.py` | 审核 GT Bundle 和冻结资源校验 | `load_bundle` 明确要求 GT，不允许伪装预测 |
| `oracle_inference.py` | 重载冻结 Qwen/Pano/World、保留 Local RGB 条件 | 只支持 GT 对照，需要独立预测入口 |
| M3DLayout | 官方 AR、补全方法、训练代码、权重入口 | 版本固定、环境验证、类别/几何桥、任务适配均待做 |

不要把 `qwen_pano/caupano/world_condition.py` 的旧特征实现当作当前契约。本轮以 `qwen_edit_pano/world_pipeline/oracle_model.py` 为准。

## 3. 文件布局

以后按需创建；本文不创建这些模块、环境、数据或输出目录。

```text
qwen_edit_pano/
  world_pipeline/
    PLAN_OBSERVER3D_M3DLAYOUT.md          # 本文
    completion/                         # 新增 Python 子包
      __init__.py
      freeze_inputs.py                  # 实验子集与独立输入/评测清单
      observer3d.py                     # Pass B；复用 Pass A 结果
      observer_geometry.py              # 坐标/角度/尺寸解码和角点
      evaluate_observer3d.py            # GT 只在评测阶段读取
      m3dlayout_bridge.py               # 对象框 ↔ M3DLayout 张量
      m3dlayout_probe.py                # 官方权重能力探测
      prepare_completion.py            # 本任务训练记录
      train_completion.py              # AR 适配训练入口
      sample_completion.py             # 冻结模型生成多个候选
      export_world.py                  # 当前 graph schema 导出
      evaluate_completion.py
      sample_predicted_world.py         # 独立预测状态 ERP 推理
      tests/                           # 几何、采样、输入隔离等必要测试
    configs/observer3d_completion_v1/    # 新增版本化配置
      observer.json
      category_bridge.json
      geometry_contract.json
      completion.json
      evaluation.json
  third_party/M3DLayout-code/            # 以后固定官方 commit
  data/observer3d_completion_v1/          # 以后只建清单/轻量记录，引用原资产
    EXPERIMENT.json
    inputs_dev.jsonl
    inputs_val.jsonl
    evaluation_dev.jsonl
    evaluation_val.jsonl
    completion_train.jsonl
    completion_dev.jsonl
  outputs/observer3d_completion_v1/       # 每个 run 建新子目录
    observer_dev_run001/
    official_probe_run001/
    completion_train_run001/
    completion_val_run001/
    world_export_run001/
    erp_eval_run001/
```

源码放 `world_pipeline/completion`；实验输出继续位于 `data` 和 `outputs`，兼容当前 `new_output` 允许的根目录。模型权重优先复用独立缓存，通过配置引用，不复制进代码目录。

## 4. 冻结范围与实验输入隔离

### 4.1 保持当前训练

- 不改正在运行的 Full/Constant 训练代码、清单、cache、优化器或 checkpoint。
- 不给当前 Oracle training 混入 Observer/Completion 预测。
- 下游评测只重载完整写入、有完成标记和哈希的 World checkpoint。
- 使用该 World checkpoint 绑定的 Qwen base、Panorama LoRA、分辨率、padding、类别映射和 object 上限，不按历史说明手填另一版本。
- World checkpoint 必须是 `condition=full` 才进入主要预测 G_full 对照。Constant/Observed 分支单独标记，不误当 Full。

### 4.2 首轮样本

建议从现有 14 栋 train 中固定 30–50 个已通过对应 GT 审核的 local views 做开发诊断，覆盖不同房型、对象数量、遮挡和距离。按 ID 和固定 seed 选样，不看预测结果挑样。

开发样本可以诊断训练建筑上的表现，不作为建筑泛化证据。Completion 超参数需要调节时，从 14 栋内部固定开发子集并在适配训练中排除；相同房间/全景的关联视图不得跨内部 train/dev。样本足够时优先按建筑留出内部 dev。

正式评测沿用原 3 栋 val 的合格样本，不变更验证建筑。174 是历史候选数，记录实际合格数、失败数和排除原因。不能为满足数量而放宽既有 GT QC。

### 4.3 两套清单

`inputs_*.jsonl` 仅提供推理所需：ID、split、local RGB 路径/哈希、宽高、已知拍摄内参或明确的 camera protocol。复用现有 Pass A 时保持它要求的精确键集合；额外相机配置走独立受限 sidecar，不直接增加旧入口不接受的字段。

`evaluation_*.jsonl` 才记录对应 GT G_obs/G_hidden/G_full、review、建筑/房间分组和 GT 几何来源。预测程序不读取它。ID 仅用于关联文件，不能拼入模型 prompt。

禁止预测阶段读取 target ERP、GT hidden list、full-scene caption、GT mesh/depth 或 GT 房间轮廓。GT 可以用于训练集监督和离线评测，不能逐样本校正 val 预测后再当纯预测成绩。

## 5. 最终 Adapter 字段契约

当前固定：

```text
FEATURE_VERSION = objects_local_aabb_masked_v1
schema_version = caupano.local_world.v0.1-pilot1
frame = local_view_x_right_y_image_up_z_forward_metres
geometry = [cx, cy, cz, sx, sy, sz, center_valid, size_valid]
```

| 字段 | 预测路径 | 严格含义 |
|---|---|---|
| `category_id.value` | Qwen/Completion 类别 → 固定映射 | 现有 Matterport canonical ID，不能用 M3DLayout 编号冒充 |
| `center_local_m.value` | 对象中心 → local 相机坐标 | 单位米，x 向右、y 向图像上、z 向前 |
| `size_aabb_local_m.value` | 3D 框全部角点 → local → 各轴 max-min | 完整边长，不是半径、OBB 尺寸或可见部分尺寸 |
| `center_local_m.valid` | 由解析/几何可用性确定 | 不代表与 GT 相等或高置信度 |
| `size_aabb_local_m.valid` | 同上 | 未知置 null/false，不把零当未知的有效值 |
| padding mask | `tensorize` 根据真实对象数生成 | 不由 Qwen/M3DLayout 预测 |

现有可用类别集合由 `common.ELIGIBLE` 与冻结 vocabulary 联合检查；当前为 3–40 排除 17。Encoder embedding 的 0–41 索引范围不等于全部索引都是可输出类别。

当前 tensorize 不使用 room、yaw、2D bbox、track ID、source、confidence 或 relations 作为特征。它们可保留用于上游处理和追踪，**不扩大当前 Encoder 输入**。

不过 `validate_graph` 会要求对象包含 `bbox2d_visible_xyxy_norm` 和 `in_primary_room` 等包装字段。没有信息时按现有规范 null/false；不能因为不参与模型就省去必需键。

### 5.1 图与 tensor 的接口示例

以下只是未来实现的接口示意，不是已存在的 CLI：

```python
from qwen_edit_pano.world_pipeline.common import field, validate_graph
from qwen_edit_pano.world_pipeline.oracle_model import tensorize

obj = dict(
    track_id="generated_0001",
    category_id=field(mapped_id, source="completion", method="category_bridge_v1"),
    center_local_m=field(center_xyz, source="completion", method="box_to_local_v1"),
    size_aabb_local_m=field(aabb_full_xyz, source="completion", method="corners_ptp_v1"),
    bbox2d_visible_xyxy_norm=field(None, reason="not_observed_in_local_rgb"),
    in_primary_room=field(None, reason="room_membership_not_estimated"),
    local_evidence="unknown",
)
# graph 还需 schema/frame/id/split/scope/objects 等元数据。
validate_graph(graph)
category, geometry, padding_mask = tensorize(
    graph, max_objects=checkpoint_max_objects, condition="full"
)
```

预测对象不要使用 `field` 默认的 `source='gt_derived'`。新生成的对象默认 `local_evidence='unknown'`，因为“新增”不等于已经证明它在 local 不可见；只有有独立依据时才标 `unobserved`。

tensor 预期形状为 `[1,M+1]`、`[1,M+1,8]`、`[1,M+1]`；slot 0 是现有 learned empty/global token。M 从绑定 Bundle 读取，不硬编码 128。超限报错并记录，不偷偷截断。

## 6. 步骤一：冻结配置，先完成 CPU 接口设计

**输入：** 当前训练配置/Bundle、已审核 GT、Local-only 清单、Qwen 模型位置。

**需要新增：** `freeze_inputs.py`、配置模板和协议检查测试。

`EXPERIMENT.json` 至少记录：

- train/dev/val ID 清单与建筑分组；所有清单哈希；
- 原始图与输入预处理哈希；
- World checkpoint 后续待绑定的路径/哈希，不提前猜测 final checkpoint；
- Qwen model snapshot、解码配置、prompt 版本；
- M3DLayout commit、config、权重/归一化统计/类别表哈希；
- 观测随机种子、补全随机种子、ERP 随机种子；
- 类别映射、坐标转换、尺寸/角度规范版本；
- 全部阈值及其来源：工程有效性规则或 train/dev 决策。

**验收：** 清单不重复、不跨建筑泄漏；预测输入不含目标信息；数据和输出新建目录；配置中不存在待补全值时才能正式运行。

## 7. 步骤二：Qwen3-VL Pass A + Pass B

### 7.1 Pass A

复用已存在的 Observer：可见类别、可见 2D 框、room candidate。它不负责猜镜头外对象。先固定已有 baseline 配置，不同时改变提示词、解码和分辨率。

旧文档中的 `baseline_v1 / bbox-scale=1000 / checkpoint decoding / image-size=768 / max-new-tokens=4096` 可作为待核对的起点，不能把它称为已经在新样本集验证的最佳配置。

### 7.2 Pass B

新增 `observer3d.py`。按 Pass A 中出现的类别查询，优先保持完整图像；逐类别输出全部该类对象的 3D 框。先不裁图，避免改变相机坐标原点和内参而没有恢复。

提示词设计原则：要求输出可见实例的类别和完整对象 3D 包围框；不列镜头外对象；不能判断时保留失败/未知，不编造满足格式的数值。按官方格式先测试，单位/角度在配置中定义并验证，不用提示词强制转换替代几何契约验证。

每个类别查询保存：完整 prompt、原始响应、JSON、解析状态、耗时、token 数、终止原因、输入图尺寸、模型版本和随机种子。

匹配 Pass A 与 B：用类别和显式实例线索/投影关系做一对一匹配。多个相似实例无法确定对应时记录 ambiguous，不强行匹配。完整 3D 框投影与可见 2D mask 框可能不同，不能把二者 IoU 不足直接判为不存在。

不新增第三个描述生成 Pass。官方权重探测需要文本时，从 room candidate 或既有可见清单按固定模板生成简短英文提示，标记为局部证据，不添加 hidden 内容。

### 7.3 输出

```text
observer_dev_run001/<sample_id>/
  pass_a_raw.txt
  pass_a.json
  pass_b/<query_id>/raw.txt
  pass_b/<query_id>/parsed.json
  observer_boxes_raw.json
  observer_boxes_local.json       # 可追踪的完整角点/OBB sidecar
  G_obs.json                     # 当前 schema；几何未通过则对应字段 unknown
  overlay_2d.png
  overlay_3d.png
  status.json
```

实际复用 Pass A 结果时引用原始目录即可，不必复制上述文件。`COMPLETE.json` 仅代表处理结束，并不代表所有样本成功。

## 8. 步骤三：锁定几何规范与 Observer 评测

### 8.1 必须校验的约定

- Qwen 原始相机的 y 方向、旋转次序、角度尺度；官方说明存在角度文档差异，不能按数值范围猜每个样本的单位。
- 当前 local 协议与原图、resize 后内参之间的关系。resize 后 fx/fy/cx/cy 同比例变换，保留实际 resize 尺寸。
- Qwen 所用 camera 表示到 local 的转换；不能读取 GT hidden 来拟合 val 的尺度/朝向。
- 完整 OBB 尺寸与 half-extents；yaw-only 框与 9-DoF 框的表达差异。
- 坐标基可能涉及轴反射；使用已验证的角点基变换，不把任何轴变换都当作普通 Euler rotation。

官方仓库回复称不强制在 prompt 提供 K；本项目仍用已知拍摄协议的 K 做投影核查。不把单目米制预测当作无误差测量。

### 8.2 几何转换

以列向量约定举例，必须在实现配置写清楚，不能与现有行向量代码混用：

```text
corner_q = center_q + R_box_q × (signs ⊙ half_extents_q)
corner_local = A_q_to_local × corner_q + t_q_to_local
center_local = A_q_to_local × center_q + t_q_to_local
size_aabb_local = max(corner_local, axis=corners) - min(corner_local, axis=corners)
```

先使用已知合成框验证几何函数，再用 train/dev GT 诊断整体约定。只可确定一次固定转换，不能对每张 val 图尝试多个轴方向后选 GT 最好的版本。

全局对象允许 `z<0`，否则会错误排除相机背后的对象。可见框做相机近裁剪后再投影，不能直接除以非正 z。距离有效性和 z 正负是不同检查。

### 8.3 两类评测分开

工程检查要求：所有发往 Completion 的框都有有限中心、正尺寸、合法类别、已解码旋转；否则记录 unusable，不静默修复。

准确性报告：

- Pass A detection P/R/F1、解析失败与漏检；
- 在独立 2D/实例匹配确定的对象上，报告 3D 中心误差、径向距离误差、方向夹角误差、AABB 尺寸绝对/相对误差；
- 同时报匹配覆盖率、可用 3D 比例和全部失败数，不能只报成功框误差；
- median/P90 以及按类别、建筑、遮挡程度分组；
- 3D 框投影仅作一致性检查，投影正确不证明深度/尺度正确；
- 若算 3D IoU，清楚区分 local AABB IoU 与 OBB IoU，不能混称。

现有 GT 2D 框来自 mesh 可见区域，存在实例粒度/覆盖限制；人工诊断与原始严格评分并列保存，不临时改 GT 凑分。

**进入下一步条件：** 几何契约已确定、可用率和误差报告完整。没有预设“误差低于某值必然改善 ERP”的结论。几何明显不稳定时，先停在 Observer/GT Completion 诊断，不直接跑全套 ERP。

## 9. 步骤四：M3DLayout 官方模型能力探测

### 9.1 环境与资源

以后由用户手动固定官方 commit、下载权重并建立独立 M3DLayout 环境。Qwen Observer 使用独立 qwen3vl 环境，Qwen-Image-Edit 继续 qwen360，不能为了第三方依赖升级原训练环境。

只做对象布局实验，不先下载完整场景资产库/渲染数据。记录实际运行峰值显存和耗时，再安排后续资源；不预先保证某张卡必然足够。

### 9.2 类别桥

分别建立 `caupano_to_m3d` 与 `m3d_to_caupano`。官方细类别回到 canonical 可多对一；canonical 回到多个细类别存在歧义，不假装可逆。

首轮只使用有明确映射的 context；未映射对象保留在 Observer 原记录与最终 observed graph，单独统计它们没有约束 Completion 的影响。不能为了复用 checkpoint 随意把未知类别全变成 chair 或 generic furniture。

### 9.3 官方 checkpoint 坐标约束

官方权重使用自己的布局坐标、归一化与训练范围；本项目输入是局部相机坐标。**字段形状一致并不代表可以把 local 米制框直接归一化后交给原权重。**

先核对官方数据转换与真实样本，确定布局原点、竖直轴、尺度、朝向和尺寸。只有能用实际推理时可得信息建立确定的映射时，才作为正式零样本桥接。

若缺少房间中心/地面高度/相机相对布局位姿，不能用 GT full room 或 GT hidden 补出它们。可做“仅由 observed 推导的坐标假设”探测，但所有这类结果标记为 exploratory，不能宣称与官方训练分布一致。

若该问题无法由可用信息解决，**结束原权重直用探测，进入第 10 节任务适配**，不反复调坐标猜法追逐几张好图。

### 9.4 探测模式

| 模式 | 初始框 | 文本 | 解释 |
|---|---|---|---|
| GT-context diagnostic | GT observed 的兼容表示 | 简短局部提示 / none | 单独诊断 Completion；明确使用 GT |
| Pred-context + short text | Qwen 可用 observed | 固定模板 | 官方文本条件权重的主探测 |
| Pred-context + no text | 同一批 Qwen observed | `None` | 去文本消融，不假设训练时见过 |

room mask 均为全 1 的 64×64，dtype/device/batch shape 按固定源码验证。不得解释为预测的房间几何。

官方 AR 训练采用排序场景的随机前缀，与任意 observed 集合不同；`complete_scene` 能追加对象不等于已经通过本任务验证。先固定 4 个 completion seeds，全部保存。

在当前检查过的实现中，`max_boxes` 是循环追加次数，**不是包含 context 的总数**。未来 wrapper 必须根据已观测数和 Adapter M 计算新增上限，并正确处理 START/END、输出裁剪边界和未生成 END 的状态。

## 10. 步骤五：面向本任务的最小 Completion 适配

此步只在前述诊断后由用户决定运行，不属于现在编写文档时执行的工作。

### 10.1 第一版选定的简化表示

为尽量匹配当前 Adapter，任务适配第一版优先采用：

```text
category = 当前 canonical vocabulary
translation = center_local_m
size = size_aabb_local_m 的完整边长
angle = 0（明确采用 local-axis AABB 表示）
room mask = 固定全 1
text = 无，主实验
```

这是**对 M3DLayout 数据表示的显式改造**，需要相应训练；不能把这套张量喂给官方 OBB checkpoint 并称为等价输入。它使生成结果直接符合当前 Encoder，避免第一版还要学习可靠语义朝向或房间 floor plan。

Qwen 的旋转仍用于把原始 3D 框转换成 local AABB；转换后 Completion 才使用 angle=0。原始 OBB/角点保存在 sidecar，不把 AABB 逆推成原 OBB。

复用 AR 主干和类别、中心、尺寸输出头的结构。angle head 可以保留参数结构，但训练目标/推理输出固定为零并屏蔽无意义 angle loss，明确记录这项修改；不能声称原 loss 完全不变。

若以后要复用完整 yaw/OBB prior，另做版本，处理重力轴与 9-DoF 降维；不与该 AABB 第一版混报。

### 10.2 数据和采样

从审核后的同一世界协议导出 GT observed/hidden。G_full 范围是目标 ERP 协议下的对象集合，不是全建筑家具，也不一定严格只包含 primary room。

```text
context = all GT observed + GT hidden[:k]
target  = GT hidden[k]，若 k < hidden_count
target  = END，若 k == hidden_count
```

hidden 在固定 local frame 内按 x/z/y/category 等冻结规则排序；context 不得截断 observed。只根据 train 估计类别分布、归一化范围、数量预算和阈值。val 不更新统计。

Sampler 必须覆盖 empty observed、empty hidden、正常 END 和容量边界。损失使用类别项与连续中心/尺寸的原输出分布目标；END 样本仅计算分类项。检查官方 loss 是否实际屏蔽，必要时实现，不能只写在说明里。

训练数据不对 hidden 的 z 符号做限制；local 视锥内也可能有被遮挡对象，不能简单把“视锥内”全部排除。

### 10.3 初始化与训练顺序

先使用 GT G_obs 训练/评测 Completion，隔离 Observer 误差。当前数据规模是否足够由 dev 曲线和泛化结果判断，不承诺从头训练必然成功。

初始化选项：从头训练作为清晰参照；或只迁移结构/形状兼容的主干权重，明确重置不兼容输入输出层。迁移不能只看 shape，类别顺序与几何语义必须核查，并导出 loaded/missing/unexpected 参数清单。

先复用官方优化器设置作为起点，固定一个有上限的短预算，按 dev 的 NLL、续写/END、重复和数量分布检查后再扩展。具体步数、batch、耗时在读取实际 approved 数量和测得吞吐后写入配置，不在本方案捏造可用预算。

GT-context 有效后再评测 Pred-context 的分布差距。需要时，使用 train RGB 的真实 Observer 预测做适配；保留 false positives，记录 missed observed，不能简单把 GT 隐藏集拼到所有预测后面造成重复标签。另定义版本化对象匹配与 target 策略再训练。

### 10.4 与无学习基线比较

至少比较 observed-only、只依赖房型/类别频率的先验或 train-only 检索补全、AR 补全。检索若使用，明确仅检索训练资产并按相同坐标规则处理。这样才能检查是否真的学习到了观测条件，而不是只生成常见家具组合。

## 11. 步骤六：构建预测 G_full 并做独立验收

### 11.1 合并规则

1. 保留原 Observer graph 的所有 observed 记录。只有几何合格对象送进 Completion context，但未合格记录不得悄悄消失。
2. 官方模型探测的输出去掉 START/END、反归一化和反变换；AABB 适配模型按自己的冻结 stats 反归一化。
3. 对新增框做类别映射、有限值/正尺寸检查，生成独立 track ID。
4. 最终 observed 部分使用原 graph 数值，不使用第三方 encode/decode 重写，避免悄悄漂移。
5. `G_full_pred.json` 设置 `scope='full'`、明确 prediction stage 和 source；不改成 GT、不写入 Oracle Bundle。
6. 输出完整失败/剔除/超限记录。主比较中的无输出样本记作失败，不能只比较成功子集。

默认不做偷偷去重、推开碰撞、缩放到单位立方体或挑最好 seed。若以后加入修复，raw 与 repaired 两套结果都保存并分别报告。

### 11.2 对象范围与可见性限制

新增对象只是 hidden 候选。仅凭 bbox 不能精确复现 GT 全场景 mesh 可见性，也不能因落在 local 视锥内就判为错误。先报告投影冲突、遮挡不确定性和重复候选，人工检查典型案例。

部分 GT 完全不可见对象不属于当前 Adapter 的目标集合；模型生成任意完整房间布局与此存在范围差距。该差距在 probe 中明确报告，在任务适配训练中沿用现有 G_full 筛选协议减少差异。

### 11.3 导出验收

- `validate_graph` 通过；vocabulary 和 geometry version 匹配；
- `tensorize(..., condition='full')` 通过；形状、dtype、有效位、padding 正确；
- observed 对象 ID/几何和值有效性与输入一致；
- 对象总数不超过绑定 Bundle 的 M，全部 source 正确；
- round-trip 数值检查与单次几何定义检查分开；
- 不能要求量化、截断、yaw 拟合或不可逆类别映射也“无损回环”；这些误差单独测量。

### 11.4 补全质量报告

分别报告 GT-context 与 Pred-context。主要指标：对象数误差、类别集合/数量 P/R/F1、类别匹配后的中心/尺寸误差、END 比例、容量截断率、重复率、角度分布、可见区域冲突，以及多 seed 多样性。

GT observed/hidden 区分用于评测，不可让推理端读取。GT 世界只是多种合理隐藏世界中的一个；几何对 GT 的误差与人工合理性分别报告，不把不同于 GT 一律视为不合理。

箱体重叠只是粗略 collision proxy，支持、嵌套等情况不能一律判物理碰撞。没有预测房间包络时不声称测出了完整 room violation；若用 GT room 做评测，明确标记 evaluation-only。

## 12. 步骤七：独立预测状态 ERP 推理

新增 `sample_predicted_world.py`，复用当前模型加载和 WorldBranch，但不放松 `oracle_data.load_bundle` 的 GT 限制。

入口分开接收：冻结 World checkpoint；已绑定的生成资源配置；Local-only 输入清单；预测 graph 清单；输出路径。GT 参考图与 GT graph 只在明确标记的对照/评测路径加载。

加载时核对 checkpoint feature version、权重哈希、原 Bundle 和冻结资源、graph id/split/vocabulary。实际 forward 仍为：

```text
image = 原始 Local RGB
prompt = 固定 panorama 指令（首轮不引入不同用户文本）
world = tensorize(当前对照的 graph)
```

所有预测路径记录 `oracle=false`；GT-full 单独标记 `oracle=true`。正负 CFG 的 World 使用方式、steps、CFG、分辨率、padding 与当前 Oracle 协议一致。

第一轮对照：

| 模式 | World 输入 | 目的 |
|---|---|---|
| Baseline | World bypass | 冻结 Qwen-Pano 基线 |
| GT Full | 正确 GT G_full | 同一 checkpoint 的 Oracle 参照 |
| Pred Observed | Qwen G_obs | 预测观测自身的贡献 |
| GT Observed + Completion | GT context 的补全 | 隔离 Observer 误差；明确是混合 Oracle 条件 |
| Pred Observed + Completion | 预测 G_full | 本轮主方法 |
| Shuffled Pred Full | 固定错配的预测 G_full | 检查是否使用 sample-specific 结构 |

上述主要比较使用同一个 Full-trained checkpoint，`Pred Observed` 仅为推理消融，不冒充单独训练的 Observed 模型。Constant-trained 若已有正式完成的 checkpoint，可作为额外参照。

生成耗时较高，先固定完成种子做 dev 成对比较，再扩展多种子。Completion seed 与 ERP seed 独立记录；比较多个 Completion 候选时固定 ERP seed。四个候选全存，不以 GT ERP 挑选最好候选当主成绩。

每样本保存 Local、各模式 ERP、graph/权重哈希、随机种子、World gate/残差诊断、失败状态。评估 Local 内容一致性、隐藏类别和布局、整体画质、接缝/极区；按原 3 栋 val 分别报告。不要仅用同一个 Qwen Observer 自评。

语义改善而空间无改善时，先检查几何与条件依赖；有变化但与 shuffled 相当不能宣称有效结构控制。

## 13. 必要测试与每步停止条件

| 测试/验收 | 必须覆盖 | 未通过时 |
|---|---|---|
| 坐标单测 | 前/后/左/右/上/下，旋转长方体，已知完整边长，轴反射 | 停止向 M3DLayout 输送几何 |
| Observer 解析 | 非有限值、负尺寸、缺字段、重复实例、截断、未知角度规范 | 留存失败，不自动猜值 |
| 桥接 | frozen stats、类别映射、归一化/反归一化、范围外值 | 修桥接；不通过 val GT 调范围 |
| Completion sampler | 3 observed + 2 hidden 的全部 k、空集合、END | 不训练 |
| 补全保留性 | observed 无丢失/无改写，新增数预算，END 去除 | 不导出给 ERP |
| 输入隔离 | 预测清单不含 target/GT；全局统计 train-only | 修数据流 |
| Adapter contract | graph validate、tensor shapes、slot 0、mask、超限拒绝 | 不跑生成 |
| 推理隔离 | 同一 RGB/prompt/seed，仅 World 变化，bypass 一致性 | 不报告方法增益 |

这些测试针对新几何和新接入，不重跑已经完成的小数据 Adapter 训练 smoke。

## 14. 用户逐步执行清单与未来 CLI 约定

**当前可以执行的只有已有 Observer Pass A；Pass B、桥接、补全、预测 ERP 入口均待实现。** 本文不提供假装现在可运行的新脚本命令。

每完成一行再继续下一行；GPU 作业均由用户在独立分配资源上运行。

| 顺序 | 开发/执行动作 | 用户检查的交付物 |
|---|---|---|
| 1 | 实现/运行冻结输入与配置 | `EXPERIMENT.json`、dev/val 清单、全部真实路径 |
| 2 | 实现 Pass B 与几何单测 | prompt、解码规范、CPU 测试结果 |
| 3 | 用户运行 dev Qwen Pass A/B | raw、G_obs、投影图、失败分布 |
| 4 | CPU 3D 评测并锁定桥接 | 中心/尺寸/方向误差、可用覆盖率、geometry contract |
| 5 | 固定 M3DLayout 源码/环境/权重 | commit、依赖锁定、归一化/类别/坐标核对 |
| 6 | 用户运行官方 checkpoint 探测 | GT/Pred context、短文本/无文本、多 seed 布局 |
| 7 | 按诊断决定是否做 AABB AR 适配 | train/dev records、loss/采样测试、训练配置 |
| 8 | 用户运行 Completion 训练或冻结 probe 模型 | 完整 checkpoint、dev 评测、已声明适用范围 |
| 9 | 导出并验收预测 G_full | schema/tensor 报告、保留性和失败报告 |
| 10 | 实现独立预测 ERP 入口；用户运行 | 同一冻结 checkpoint 的成对结果 |
| 11 | 配置固定后在原 3 栋 val 评测 | 每建筑结果、全部失败、多 seed 报告 |

未来 CLI 实现约定（**接口规格，不是可执行命令**）：

| 拟新增模块 | 主要输入参数 | 输出 |
|---|---|---|
| `freeze_inputs` | `--bundle --gt --review --config --output` | 隔离的预测/评测清单 |
| `observer3d` | `--inputs --pass-a --model --vocabulary --config --output` | raw + G_obs + 几何 sidecar |
| `evaluate_observer3d` | `--evaluation-manifest --predictions --config --output` | 独立 3D 报告 |
| `m3dlayout_probe` | `--graphs --geometry-sidecars --model-config --weights --bridge-config --output` | 原始续写与逆转换结果 |
| `prepare_completion` | `--train-manifest --dev-manifest --config --output` | AABB AR 数据与 train stats |
| `train_completion` | `--data --config --output` | 专用 Completion checkpoint |
| `sample_completion` | `--checkpoint --graphs --config --output` | 所有 seed 候选与状态 |
| `export_world` | `--observed --completions --world-contract --output` | 预测 G_full 清单 |
| `sample_predicted_world` | `--world-checkpoint --inputs --world-manifest --config --output` | 预测结构条件 ERP |

每个新增 GPU 入口应实现 CPU `--preflight`：检查路径、哈希、环境、维度/协议，不加载大模型、不隐式联网下载。先给用户明确的 preflight 结果，再提供已落地的实际命令；Slurm 分区、节点、显存和时间以当时资源为准。

## 15. 如何根据结果决定下一步

- Qwen 几何可靠、GT/Pred context 都能补全：继续同 checkpoint ERP 对照。
- GT context 好、Pred context 差：优先处理 Observer 几何/类别/漏检，或训练噪声适配，不归咎于 Adapter。
- 原权重效果差且桥接/条件分布不匹配：使用本任务 AABB AR 适配，不能由此否定补全架构。
- GT-context 的适配模型也无增益：检查有效训练规模、END、数量先验、target 范围、重复和空间条件；暂不扩大系统。
- 布局有效但 ERP 不用结构：回看同期 Oracle Correct/Constant/Shuffled 证据，再检查预测分布差距。
- 全部字段能 tensorize 但几何不可靠：只报告“接口接通”，不报告“空间条件有效”。

## 16. 核验依据与未解决事项

仓库本地依据：`observer.py`、`evaluate_observer.py`、`common.py`、`geometry.py`、`oracle_model.py`、`oracle_data.py`、`oracle_inference.py`。旧 RUN 文档中的早期 7 条样本/epoch001 信息不是本轮配置来源。

联网核验依据：

- [Qwen3-VL 技术报告](https://arxiv.org/html/2511.21631v1)：单视图 3D grounding 训练与评测；不等于本项目精度已经验证。
- [官方 3D grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb)：输出格式、投影示例。
- [Qwen 3D 接口讨论 #1501](https://github.com/QwenLM/Qwen3-VL/issues/1501)：米制、角度归一化与顺序说明；需与固定版本实测核对。
- [M3DLayout 官方仓库](https://github.com/Graphic-Kiliani/M3DLayout-code)：代码、模型入口。
- [AR 源码](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/networks/autoregressive/autoregressive_transformer.py)：已有框保留、text 可选、追加次数与 END。
- [AR 配置](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml)：默认文本条件。
- [数据集实现](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/datasets/m3dlayout_dataset.py)：全 1 room mask、排序场景随机前缀、归一化路径。
- [M3DLayout 论文](https://arxiv.org/html/2509.23728v2)：原任务是文本条件布局生成，不能把本方案视作论文已验证的相机视野补全。

尚待实施确认：实际权重能否加载；固定 commit 的尺寸/相机约定；官方训练建筑是否与本项目 val 重叠；实际 approved 数据量和各类覆盖；Qwen 的本域几何精度；Completion 与 ERP 的实际运行成本。使用官方预训练模型时应核查训练集重叠，未查清前不作严格未见建筑泛化声明。

本轮第一项开发任务是 **第 6–8 节：冻结样本、实现 Pass B 和几何评测**。先取得可信的可见 3D 观测诊断，再决定官方权重探测与本任务补全适配的投入。
