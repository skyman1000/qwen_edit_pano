> 2026-10-20 新实验入口见 [ANCHORED_WORLD.md](ANCHORED_WORLD.md)：固定WildDet3D观测、连续相机空间锚点布局、匹配训练示例。本文及 PLANNED_WORLD.md 保留历史分支说明，不是当前默认实验建议。

# 局部状态 → 视野外粗场景（v7，2026-10-19）

目标：局部 RGB → 可见状态 G_obs → 合理的视野外对象 → 完整候选状态 G_full。继续使用已有 Qwen3-VL-8B，不训练、不下载。

## 最简流程

1. Qwen 识别可见类别、描述和 2D 框。
2. Qwen 给可见对象估计深度、大小；代码沿 2D 中心射线计算粗位置并保存 G_obs。
3. Qwen 根据原图和固定 G_obs，选择新增物体及其粗方位、距离、大小、底部离地高度。
4. 程序换算新增 xyz 并检查视野外条件。有问题时仅反馈一次，由 Qwen 修订或有理由撤回问题候选。
5. 原样复制局部状态和通过候选，再追加最终新增对象。不运行物理优化器。

借鉴 LayoutGPT 的结构化粗布局规划；方位到坐标的换算是本项目的工程简化，不是论文原方法的复现。LayoutVLM 参考语义规划与数值处理分开的思路，当前不接其资产、关系图或优化器。

## 本轮执行指令

固定 run010 的局部状态，复用其初始补全，仅对有问题的候选请求一次修订：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU2 \
  --hfov 90 \
  --reuse-state qwen_edit_pano/outputs/semantic_world_run010/G_obs.json \
  --reuse-completion qwen_edit_pano/outputs/semantic_world_run010/scene_completion.json \
  --output qwen_edit_pano/outputs/semantic_world_run011
```

GPU2 自行替换，GPU 由用户手动执行。输出目录须为新的。--reuse-state 核对原图、词表、FOV 和数值字段，原样保留 G_obs；--reuse-observation 仅复用 2D 清单，会重新预测局部几何。

从新图片开始：去掉 --reuse-state 和 --reuse-completion，增加 --image /实际路径/图片.png，并填写实际或估计的 --hfov，即执行完整三次请求，必要时再增加一次补全修订。

## Qwen 新增对象字段

```json
{
  "id": "new_01",
  "category": "cabinet",
  "description": "storage cabinet",
  "azimuth_deg": -120,
  "distance_m": 2.8,
  "bottom_height_m": 0,
  "size": [1.2, 1.8, 0.5],
  "reason": "storage appropriate for this room"
}
```

顶层回复包含 room_type 和 additions。类别、对象数量、方位、距离和大小由 Qwen 选择，程序不预置必须出现的家具。

azimuth_deg 为相机方位角：0° 前方、-90° 左方、90° 右方、±180° 后方；可取连续角度。不是物体自身旋转角。旧 camera_region 格式仍可解析，completion_conversion.json 会记录 legacy_region_ids。

水平距离 r、方位角 θ 对应 x=r sinθ、z=r cosθ；中心 y=bottom_height_m+size_y/2-camera_height。默认相机高度 1.5m 是假设。不会自动把进入照片的盒子推走。

这是新的方向参数化场景假设，不是把旧 xyz 搬到外面后宣称模型已预测正确。当前版本先处理照片之外的补全，不另外猜测照片内部被家具挡住的小物件。很大的盒子或过近的距离仍可能进入视锥，检查会照实报告，不自动推走。

## 查看结果

- scene_completion.txt / .json：Qwen 原始方向规划；.request.json 和 .generation.json 保存提示与生成记录。
- completion_conversion.json：方向定义、旧格式回退记录、坐标换算和尺度假设。
- scene_plan.json：包含区域、距离、高度等参数的完整对象计划。
- G_obs.json：原有局部状态，补全前后原样保留。
- G_hidden.json：新增对象，已换算为当前 Encoder 需要的类别、中心 xyz 和完整尺寸。
- G_full.json：局部状态加新增状态；可见对象记录与 G_obs 完全相同。
- layout_refined.png：俯视图，蓝色已见、橙色新增。文件沿用旧命名，但不再执行优化。
- projection_refined.png：红框为原始 2D 证据，绿框为已见盒投影，橙框为进入原照片视锥的新增盒。
- geometry_report.json / summary.json：只读检查。completed 只表示产物写出，不是场景质量验收。

判断重点：新增物体和方位是否符合房间常识、同一方向多个物体是否明显挤在一起、是否破坏原照片含义。区域中心设置得正确不代表物理布局或全景控制已经验证。局部深度、大小也仍是粗假设，未因保留状态而变成真实重建。

## CPU 验证与历史结果

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh check
```

41 项测试通过，包含区域换算、落地高度、物体过大仍报告进入视锥、局部状态原样保留和真实 Encoder tensorize。research/audits/directional_completion_2026-10-19/cpu_example 是人工构造的 CPU 示例，不是 Qwen 实际结果。v5 的真实 run009 已完成；v6 连续方位格式尚待用户 GPU 推理验证。

旧 run006～run008 及审计保留。finish 仍可只读导出历史数值布局，但不会重新规划场景。旧关系校验和全场景优化函数仅保留作历史审计，默认不调用。

## run009 实际审查

新增椅子、植物、小柜子是可能的卧室内容；3/3 视野外主要由旧三射线参数化促成，不是完整布局质量证明。椅子描述“床边”，但与床中心水平距离约 4.74m。旧局部窗口/两幅画的投影覆盖二维证据仅约 27%/24%/26%。覆盖率是二维自洽检查，不能判定真实深度；不设置新的强制阈值。

v6 在 geometry_report.json 的 observed 对象记录中增加 visible_bbox_coverage，summary 明确 observation_geometry_verified=false、scene_semantics_verified=false。俯视图外框只是包围范围，不是恢复的墙壁。

详见 research/audits/semantic_world_run009_2026-10-19/REPORT.md。先跑 run010 做同输入对照，检查实际新增内容、方位和可见状态，再换不同房间验证；不要把同一张图零告警当作验收。

## v7 有限反馈

run010 椅子中心在-45°边缘且盒子进入照片；柜子为0°正前方吊柜。转换没有出错，规划未满足视野外要求。现在增加一次计算反馈，不更换模型/坐标格式。

- completion_initial.json：原候选的编译坐标。
- completion_revision.*：一次修订请求与原始回复（仅在有问题时出现）。
- completion_feedback.json：before/after、withdrawn撤回理由及revision_error。
- summary 新增 completion_revision_used、completion_conflicts_before/after、completion_withdrawn。
- 未解决或修订格式错误时 quality_status=needs_completion_revision，保留候选供检查，不静默删物体或搬到背后。

修订是本项目的有限反馈机制，非 LayoutVLM self-consistent decoding 的复现。原图中可能有遮挡，进入视锥不证明一定可见；当前版本仍限定为视野外补全，不认证遮挡补全。详见 research/audits/semantic_world_run010_2026-10-19/REPORT.md。
