# 局部观测与完整场景：当前执行说明

目标：Local RGB -> camera-local G_obs -> 合理完整场景假设 G_full。使用现有Qwen3-VL-8B，不训练、不下载。当前先检查局部几何，再运行完整补全。

## 第一步：只更新局部三维状态

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 \
  --observe-only \
  --hfov 90 \
  --reuse-observation qwen_edit_pano/outputs/semantic_world_run013/G_obs.json \
  --output qwen_edit_pano/outputs/semantic_world_run014_obs
```

--reuse-observation仅复用相同图片的类别/二维框，不复用旧中心、尺寸。正常只运行一次observed_layout请求，然后停止。--reuse-state则复用完整数值状态，两者用途不同。

新Observer返回depth_m（相机前向z，不是射线欧氏距离）、center_uv（完整三维中心投影的normalized uv）、size（相机轴完整AABB尺寸）。每个对象记录实际使用显式center_uv还是旧可见bbox中心代理。

查看G_obs.json、observed_geometry_initialization.json、observation_geometry_report.json、projection_refined.png。visible_bbox_center_proxy_count大于0说明模型未给全投影中心字段，不能宣称新参数化已全部采用。

## 第二步：CPU对照已有GT

旧run013的三图GT/坐标审计已完成。run014_obs生成后执行：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python -m \
  qwen_edit_pano.world_pipeline.completion.audit_scene_consistency \
  --prediction qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json \
  --output research/audits/semantic_world_run014_obs_gt
```

只读现有GT、源OBB、相机basis及原图，输出角点重算、投影覆盖率、可见bbox中心代理误差及同图类别+二维IoU匹配。匹配是诊断而非实例真值；GT中一个床可能由多个组件构成。

相机轴与重力分开：levelled=false；默认up=[0,1,0]只是工作先验，floor来自camera-height先验，gravity_verified/floor_verified均为false。负的地面先验距离不能直接当作真实穿地。GT仅进入CPU审计，不输入Qwen部署路径。

## 第三步：局部状态可接受后运行补全

```bash
bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh run \
  --nodelist=GPU1 \
  --completion-mode planned \
  --hfov 90 \
  --reuse-state qwen_edit_pano/outputs/semantic_world_run014_obs/G_obs.json \
  --output qwen_edit_pano/outputs/semantic_world_run015
```

正常三次请求：room_contents -> room_geometry -> room_placement。从全新图片开始，去掉复用参数，添加--image /实际图片路径并给FOV，先用--observe-only检查局部状态。

## 内容、ID关系与摆放

- 结构化对象清单是存在性承诺。额外边桌不能只写在椅子的描述里；object_registry.json及G_full.object_commitment记录该约定。
- relations使用subject/type/target对象ID；target必须是已见或已规划ID。只列少量关键关系。
- 支持near、left_of/right_of/in_front_of/behind、on。方向相对于相机轴；near为AABB表面间距，默认max_gap_m=1m，可显式指定。
- face_to可以记录，但当前AABB没有朝向，结果明确为unverified；现有Encoder不能由该字段控制朝向。
- on支撑面由room_geometry估计support_surfaces中的offset_m（沿工作up，相对父盒中心）。不会把椅背/床头板的AABB顶部自动当作座面。子候选跟随父候选，父子AABB不被一概判作碰撞。
- near/方向是软选择，未满足时报告needs_relation_review，不删除对象凑通过。on依赖必须一致。

产物：object_registry.json、room_contents_validated.json、room_contents_with_geometry.json、placement_candidates.json、room_placement.json、placement_selection.json、relation_diagnostic.json、G_full.json。

G_full继续使用现有37类ID、camera-local xyz、完整camera-axis AABB尺寸、valid和mask。文字/关系不进入Encoder，关系必须先落实为布局；没有修改权重。

## 限制与验证

局部数值仍是Qwen粗估计；候选来自有限搜索先验，不是恢复的真实墙体。支持面、重力和地面仍非测量真值。新字段的真实Qwen质量尚待运行验证。当前未实现视锥内遮挡物体补全或朝向控制。

CPU检查：`bash qwen_edit_pano/world_pipeline/completion/run_semantic_world.sh check`。

研究来源、实际GT数值、旧结论纠正与边界见research/audits/camera_scene_consistency_2026-10-19/REPORT.md。所有GPU及下载由用户手动执行，输出目录必须新建。
