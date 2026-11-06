# 不训练：局部描述与结构化观测的预训练推理对照

本轮遵循用户“不训练”的要求，只加载现有 Qwen3-VL 和官方 M3DLayout AR 权重。
不执行 `run_hidden.sh train`。此实验检验预训练迁移能力，不等于任务适配已完成。

## 三组对照

1. `text_only`：局部 RGB → Qwen 描述 → 官方 `generate_boxes(text=...)`。输出原生场景坐标下的 `layout_scene.json`，不是相机对齐的 G_full。
2. `boxes_room`：已有 Qwen 观测框 + 一句房间类型 → 官方 `complete_scene`。作为原有简单文本条件基线。
3. `boxes_caption`：相同观测框 + 同一张局部图片的详细描述 → 官方 `complete_scene`。保留原始 G_obs，仅追加新生成对象，导出 G_full。

不是先生成独立完整布局、再直接拼接 G_obs。独立生成缺少与输入相机的配准，也会重新生成已观察对象。共同条件补全在生成时就接收观测框；最终的拼接只包含原始观测与新生成对象。

官方依据：[文本生成示例](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/gradio_demo.py)、[AR generate_boxes / complete_scene](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/networks/autoregressive/autoregressive_transformer.py)。

## 当前实验边界

- 官方 `text_condition=true`、BERT、类别词表、统计量、权重和采样实现保持不变，不使用上一轮新训练模型的采样器。
- 结构分支仍沿用旧桥接：相机高度假设 1.5 米、yaw=0、水平偏移=0。这是实验性放置，不是已恢复的真实房间坐标。详细描述不能自动标定它。
- 默认 `size-convention=full` 与官方 demo 将 post-process size 作为 dimensions 的用法一致。历史半长试验改善部分物体落地，但不能仅据三张图断言 checkpoint 的所有尺寸语义。`M3D_SIZE_CONVENTION=half` 可固定为另一项实验约定；不要按样本选择最好的约定。纯文本 JSON 总是额外保留未经全长/半长解释的 `size_native`。
- 使用同一份现有 Observer 输出，避免三组之间改变观测误差。官方词表无法接收的观测类别会列在 `omitted_context_categories`；它们保留在最终 G_full 中，但没有成为 M3D 条件，不能宣称所有观测都被模型利用。
- 每张图片固定 seeds 0/1/2。总对象上限 120，接近官方训练数据容量并低于 Adapter 160；到容量上限不等于预测了 END。
- 纯文本生成数量是全部生成对象数，联合条件的 generated 是追加对象数，不能直接将两个数当作同一种质量指标。
- 三张图仅用于功能和失败模式检查，不支持整体泛化结论。不会根据这三张图加贴地、删镜子或逐图缩放规则。

## 执行

从项目根目录运行，脚本使用已有 qwen3vl 环境和本地模型 snapshot。
输入就是 `data/observer3d_indoor_val3_inputs.jsonl` 中的三张验证图，不需要补图片路径。
默认复用 `outputs/observer3d_indoor_val3_run001/<id>/observed.json` 与 `G_obs.json`。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# 1. 对三张原始局部图片生成描述，不读取 GT ERP 或完整 GT graph。
bash qwen_edit_pano/world_pipeline/completion/run_pretrained_text.sh describe

# 2. 查看描述：应只反映局部可见信息，不能把未看到写成不存在。
cat qwen_edit_pano/outputs/pretrained_text_val3_run001/captions/*/description.txt

# 3. 一次加载 M3DLayout，完成三张图 × 三组条件 × 三个 seed。
bash qwen_edit_pano/world_pipeline/completion/run_pretrained_text.sh compare

# 4. 查看数量、END、遗漏类别和跳过项；这不是效果评分。
cat qwen_edit_pano/outputs/pretrained_text_val3_run001/comparison/summary.csv
```

脚本默认 GPU2、两小时；可在命令前加 `TEXT_NODE=GPU3` 换空闲节点。
`TEXT_RUN=新目录` 可指定新的整组实验目录，describe 与 compare 必须使用相同值。
如果使用另一轮 Observer 输出，compare 前设置 `OBS_ROOT=对应目录`；必须是同一份三张图片。
该脚本不会触发 Observer 框重跑、Completion 训练或 Encoder/Adapter 训练。

## 看哪些结果

```text
outputs/pretrained_text_val3_run001/
  captions/<id>/description.txt
  comparison/summary.csv
  comparison/<id>/
    text_only/seed000/layout_scene.json
    text_only/seed000/layout_topdown.png
    boxes_room/seed000/G_full.json
    boxes_room/seed000/layout_topdown.png
    boxes_caption/seed000/G_full.json
    boxes_caption/seed000/layout_topdown.png
```

每组也保存原始 normalized 输出和 config。纯文本图的原点是场景原点，不是相机；联合图蓝色是原观测、橙色是追加对象。

先比较 boxes_room 与 boxes_caption：观测不变时，额外文本是否改变合理的隐藏对象预测，是否增加重复、重叠、浮空或可见区域冲突。再看 text_only 能否表达同类场景语义。不要从文本语义相似推导出相机几何对齐。

两组联合结果结构上可进入现有 Adapter 的 tensorize，但空间内容尚需检查，且预测 G_full 到 ERP 的推理接入不是本轮脚本的功能。CPU 测试覆盖接口分流、原观测不变、只追加新对象、纯文本不伪装成相机坐标图，以及 tensorize 兼容性；真实 GPU 生成由用户运行。
