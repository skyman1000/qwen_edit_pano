# Qwen 局部观察与隐藏世界补全：手动运行

此版本只做上游状态预测和 CPU 评价；不加载/修改正在训练的 Fourier Encoder/Adapter，不生成 ERP，不训练 Completion，不依赖 M3DLayout 权重。

## 当前 Observer v3：简化输出，保留部分有效观测

run002仍有嵌套数组格式错误、重复续写，以及只有两个数的3D框。当前二维请求改为标准 `[{"bbox_2d":[100,200,300,400]}]` 格式，使用完整数字示例，每类别至多8个清晰实例；不再用符号占位/省略号示例。此上限是初版观察预算，会漏掉密集场景中的部分实例，达到上限在status中明确记录。

每个2D/3D请求失败只重试一次，使用seed=1，保存独立retry原文，不无限修补JSON。失败2D类别记录在issues中，其余类别继续；失败3D保留已检测到的类别与2D框，center/size设为null且valid=false。绝不从两个数猜出剩余七个参数。全部类别都无法定位仍报failed。

存在未解决查询时输出status=partial，显示observed数量、valid_3d数量与issues数量。partial不代表完整成功，报告会标出，且统计3D覆盖率；补全可以读取这份部分有效状态，原样保留未知几何，不把它当作已恢复的三维结构。二维匹配可继续衡量实例识别，未知3D不参与匹配后几何误差均值。

默认输出为 `qwen_upstream_val12_run003`。复用同一份12张图片清单，不重新prepare。run001/run002保留不改；v3目录内重跑会跳过completed/partial样本，未解决项看status.json，不自动无限重试。当前不宣称模型效果已验证。

## Observer v2 修正（run001 真实失败之后）

旧请求同时生成描述、所有类别、所有实例与实例描述，四张图发生重复枚举直到8192 token，三张图的后续实例漏写 category。提高长度不能解决重复退化。v2 改为：

1. 一条短请求识别可见类别和一句可见描述，最多768新token。
2. 每类别独立生成二维框，最多32个实例、1536新token；类别由该查询明确赋予，每个框不用重复生成类别或描述。
3. 每个2D实例独立查询3D框，最多512新token，保留原相机/坐标转换。

观察请求采用本地模型 generation_config 的采样参数：temperature=0.7、top_p=0.8、top_k=20、repetition_penalty=1.0，固定 seed=0。采样策略与短请求旨在降低重复退化，效果仍需实际推理验证。完全相同的同类别2D框合并一次并记录索引；不做近似框合并，不从未闭合响应截取一段冒充完整结果。越界坐标直接报告，不裁剪到图内。每类别32是预算上限，达到上限会写入status，并不代表没有更多实例。

默认输出改为 `qwen_upstream_val12_run002`，保持原12张图片，不重新prepare；run001原始结果不修改。新协议要重跑全部12张，避免混用不同Observer方法。新目录中可继续断点重跑。

## 输入与方法

- 现有 Oracle manifest 的 val 建筑中，每栋固定随机取三张已标为室内的样本。使用 seed=0；原始验证划分不变。小清单是开发集，不能代替完整 held-out 评测。
- 直接从已下载的 Matterport skybox 压缩包读取 Local face2，不需要自行填写图片路径，不读取目标 ERP。
- Observer 先识别可见类别，再按类别定位2D实例，最后针对每个框在完整原图中查询3D。没有裁剪改变相机。场景描述在类别查询时获得，不额外请求 caption。
- 2D 框为 0..1000 的 xyxy；Qwen 3D 框使用米制 camera x右/y下/z前，完整 OBB 边长，绕 x/y/z 顺序旋转且角值除以 pi。复用 qwen_corners 转换到 CauPano y上 Local，并用角点计算完整 AABB。
- [官方 3D cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb) 的说明文字与可视化对旋转命名/单位并不完全一致。本实现明确提示并固定数值约定，沿用其可视化 x→y→z、角值乘 pi 的计算；不能仅靠 prompt 保证模型遵循约定，需检查生成框投影。
- face2 使用 90° 视场；prompt 中内参随展示图尺寸同步缩放。未输入 GT 相机外参、重力、深度、隐藏标签或真实房间边界。投影图只检验估计几何与图像的一致性，不证明精确米制尺度。
- Completion 接收 Local RGB、同一可见描述、固定 G_obs 和可选 scene_text，只输出 hidden 对象的类别、中心及 Local AABB 完整尺寸。允许空结果，不强迫增加家具。
- gt / pred 分别使用 GT 可见状态和预测可见状态；GT prompt 只保留白名单中的观测字段，去掉真实房间信息、source/instance IDs、可见性计数等。GT 隐藏对象只用于报告。
- 每条件三个 seed。最多新增64件，同时不能超过当前 Bundle 的160总容量。该上限是推理预算，不表示每个真实场景少于64个隐藏对象；报告 at_capacity。超出模型声明容量则记录失败，不截断。
- G_obs 由程序逐字段原样保留。hidden 的 local_evidence=unknown，source=qwen_hidden_hypothesis；这是未测量的隐藏假设，不能伪称验证过的不可见性。Full 模式仍将其送入 Encoder。
- 新 Fourier Encoder 在内部把原有几何转换成392维特征。上游仍提供原始8维几何及类别/mask；不把 Fourier 值写回 JSON。

## 执行

从项目根目录执行。所有命令使用已经安装的 qwen3vl 环境和本地模型，不下载权重。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# CPU：接口/几何/指标/假模型流程检查。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh check

# CPU：固定12张左右的图片清单；只执行一次。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh prepare

# GPU：实例定位 → 3D估计 → G_obs；不分配给正在训练 Adapter 的 GPU。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh observe

# CPU：此时已能查看 Observer 的框和误差；Completion 尚无结果属于正常状态。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh review

# GPU：先正确观测条件，再实际预测条件。各12×3个布局，不生成ERP。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh complete-gt
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh complete-pred

# CPU：更新两组报告。
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh review
```

GPU 动作由脚本内部 srun 申请1卡、48G主机内存、4小时；时间是申请预算，不是完成保证。默认由 Slurm 分配节点。如需指定空闲节点，例如：

```bash
UPSTREAM_NODE=GPU2 bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh observe
```

中断后同样的 observe/complete 命令可重跑：成功样本/候选跳过，Observer 已保存的请求结果可以继续复用。不会把解析失败当作零对象成功。失败的 raw/status 会保留，报告列出失败，GPU命令最后返回非零。不要因失败而重新 prepare 或更换图片。

新参数/新提示词/修改输入需使用新输出目录；例如设置 `UPSTREAM_RUN=qwen_edit_pano/outputs/qwen_upstream_val12_run004` 并在后续各步一致使用。同一个目录只用于同配置继续执行。

可选：在相同图片上运行旧 Observer 作对照：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh baseline
bash qwen_edit_pano/world_pipeline/completion/run_qwen_upstream.sh review
```

旧 Observer 的每类别上限8、类别式查询等设置保持原实现，因此比较的是两套完整 Observer 配方，不是只改变实例提示的严格单因素消融。旧代码的 partial 状态在报告中保留，能评估已输出实例，但必须同时报告漏检/失败情况。

## 输出

```text
data/qwen_upstream_val12_v1/
  inputs.jsonl              # 图片、建筑、FOV、可选文本；不含隐藏GT
  references.jsonl          # 只用于GT诊断和评价
  <id>/local.png
  <id>/GT_obs.json
  <id>/GT_hidden.json

outputs/qwen_upstream_val12_run003/
  observer/<id>/
    categories.txt / categories.normalized.json
    ground_chair.txt ... / grounding_audit.json
    instances.normalized.json / instances.png
    description.txt
    obs_0000.txt ...         # 每实例原始3D响应与请求
    boxes.json / G_obs.json / status.json
  completion_gt/gt/<id>/seed0/
    proposal.txt / proposal.request.json / proposal.json
    G_obs.json / G_hidden.json / G_full.json / status.json
  completion_pred/pred/<id>/seed0/...
  review_gt/review.html / metrics.json
  review_pred/review.html / metrics.json
```

## 如何阅读结果

1. 先看原图与红色2D实例框：同一物体是否重复、是否漏检、指向的实例是否正确。
2. 橙色3D投影只画相机前方可投影的完整 OBB 边，没有做遮挡推理；不要求完整框严格贴合可见像素。图外/近相机框需结合 raw 数值看，不能单靠一张投影定论。
3. 布局蓝色为固定观测、橙色为隐藏提议、绿色虚线为一份 GT 参考；每个报告中同一样本各seed使用相同坐标范围。二维图是相机轴切面，不是已重力对齐的房屋平面图。
4. Observer 的同类2D IoU≥0.25匹配用于对应可见实例，再报告中心/尺寸误差；另有同类3D AABB IoU≥0.25匹配。后者会筛掉几何错误大的对象，不能只报告其匹配后均值。
5. hidden 记录同类3D AABB匹配、precision/recall、数量，按建筑汇总。GT是一个真实参考，不是所有合理补全的唯一答案。人为检查可见空白冲突与通道合理性；没有从RGB精确恢复自由空间，代码不假装能自动判定所有冲突。
6. 与 observed 的高 AABB 重叠只标记复查，不直接判成碰撞或删除。跨seed匹配相似度仅衡量重复程度，不是质量得分。
7. 汇总包含completed和partial记录，报告partial_records与valid_3d_count；失败/缺失另列。匹配后3D误差只统计具有有效几何的对象，必须同时看三维覆盖数量，不能忽略缺失后宣称提升。JSON成功不等于状态合理。

本轮的决策：GT条件都差时先看 Completion；GT条件合理、预测条件变差时优先看 Observer；两者合理后等待新 Adapter 完成再接ERP。M3DLayout已有实验保留，本代码不会调用其训练或续写入口。
