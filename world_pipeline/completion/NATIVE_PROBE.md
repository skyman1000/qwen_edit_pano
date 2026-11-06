# 官方原生条件诊断（不训练）

目的：先判断当前 M3DLayout 权重、官方推理配置与发布数据能否在原生条件下合理工作，再决定是否继续投入 Qwen 预测结构桥接。此处没有相机输入，因此结果不是 CauPano 的图像条件补全评测，也不输出可直接给 Adapter 的 G_full。

在 DIT360 根目录执行，沿用 qwen3vl 环境：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_native_probe.sh prepare
M3D_NODE=GPU2 bash qwen_edit_pano/world_pipeline/completion/run_native_probe.sh run
```

prepare 使用 CPU，直接读现有 ZIP，无需解压或重新下载。默认选每个来源验证集各 2 个场景，共 6 个。候选条件固定为 4–120 个对象、所有类别存在于官方 demo 词表、描述非空；按 scene_id 排序后 seed=0 打乱抽取，不根据生成结果选图。不会更改 CauPano train/val 划分。

run 加载一次现有 autoregressive_59000.pth，6 场景 × 2 模式 × 3 seeds = 36 次推理。两模式共用数据集原描述的 room_type、summary 及前 3 条其他句子；不进行文本改写。该确定性文本选择仅借鉴官方文本采样字段，不声称完整复现训练文本分布。

- text：仅文本，从空布局生成。
- prefix：相同文本 + 官方排序的前一半对象，续写剩余对象。排序为中心坐标先除以 0.5 再 np.round，按 x、z、y 排序。输入不加 START，由官方 complete_scene 添加。官方 max_boxes 是解码次数上限，包含 END；总预算为 120 次减已有前缀数量。

固定官方 demo 的类别顺序、归一化统计、全 1 的 64×64 room mask 和原始采样器；text_condition=true。全 1 mask 是 demo 条件，不是恢复出的真实房间轮廓。原生 size 原样参与归一化，绘图依照官方 _get_2d_polygon 使用半边长。原生完整描述可能提及隐藏对象，仅用于诊断，不能拿来评估仅局部图像输入的最终能力。

## 输出

数据目录：`qwen_edit_pano/data/m3dlayout_native_probe_v1/`

- `audit.json`：ZIP 指纹、类别差异、实际训练数据范围与 demo 统计对照。
- `bundle.json`：固定 6 场景、文本、排序后对象及 prefix_count。
- `<source>/<scene_id>/reference.json` 和 `reference.png`：真实布局，蓝色为给模型的前缀，灰色为其余真实对象。

推理目录：`qwen_edit_pano/outputs/m3dlayout_native_probe_run001/`

- `summary.csv`：36 次结果，新增对象数量、END、前缀数值是否保持、异常 START 次数。
- `<source>/<scene_id>/<text或prefix>/seed000/` 等：`layout_scene.json`、`layout_topdown.png`、原始 normalized 输出和 summary。
- 蓝色为前缀、橙色为生成对象；绘图为原生旋转框，不是 Adapter 所需的相机轴 AABB。不同图片自动缩放，比较时须看坐标刻度。

## 已发现的发布版本差异

当前下载的训练 JSON 含 bar_chair、kitchen_island，而本地官方 demo 的词表没有这两类。保留 demo 词表，不能把新数据类别插入后套用旧权重；也不能因为 strict load 成功就断言权重类别语义匹配。实际数值统计的差异见 audit.json，不用新数据统计覆盖预训练推理统计。

本次实际审计：训练 JSON 共 103 个对象类别，demo 为 101 个对象类别（另加 START/END 共 103 个 token）。位置和角度范围一致；size_y 上界分别为 4.569929599761963 和 4.469733238220215，其余尺寸上下界一致。不要混淆对象类别数与包含特殊符号的 token 数。

因此本实验是“发布数据在当前官方推理配置下的兼容性诊断”，不是已确认完全版本一致的官方复现。若两组都差，权重对应的训练词表和统计仍需澄清，不能直接归咎于模型能力。仅验证集少量样本也不足以证实总体效果。

## 结果如何决定下一步

1. 两组都明显异常：优先检查权重配套的词表、归一化及官方采样实现，不改 Observer、不继续拼接纯文本布局。
2. text 合理、prefix 异常：聚焦结构条件续写与输入表示，暂不接 Qwen 预测框。
3. 两组都基本合理：进入 Matterport GT 可见对象集合诊断，使用可核实的原生 OBB 和相机变换，以隔离“空间前缀”和“真实可见集合”的差异；之后才比较 Qwen 预测框。排序可见对象并不能把可见集合变成官方训练前缀。

END 和前缀保持只说明接口执行，不能证明补全正确。主要检查描述中的关键类别、尺度/高度、重复和穿插、与前缀相容性、全部 seeds 的稳定性。真实隐藏布局不是唯一合理答案，不要求随机生成逐对象复原原场景。

最终给 Adapter 时，需要将同一世界内的对象 OBB 变换到 Local 相机坐标，由变换后 8 个角点求 center 和完整 local-axis AABB size，再映射 category_id/validity/padding。原生布局缺少相机注册，不能仅改字段名就注入 Adapter。
