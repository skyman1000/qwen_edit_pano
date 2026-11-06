# run002：停止继续修补Qwen隐藏几何提示

依据实际run002的review/metrics.json、review_gt/metrics.json、24份proposal.request.json、原始hidden_plan/proposal和成功导出的World State。未运行GPU。2026-10-13。

|条件|成功导出|新增对象|中心在输入视锥内|中心在相机背后|与GT隐藏框同类IoU≥.25匹配|
|---|---:|---:|---:|---:|---:|
|GT G_obs|11/12|66|57/66 (86.4%)|5|0|
|Pred G_obs|10/12|56|56/56 (100%)|0|0|

全部24份几何请求包含新版OFF-CAMERA任务与相机信息，排除了旧提示缓存误用。全部21份成功结果原样保留各自G_obs。失败类别为GT towel_rack、Pred refrigerator和shelves。补上别名只能改善导出，不会改变主体几何失败，故本轮不改词表。

GT组依然失败，说明不能将当前问题全部归因于Observer的错误；GT组共用图像和现有图像描述，并不是完美语义Oracle。两组成功样本集合不同，不能直接当作严格同样本优劣排名。GT组的9个视锥外中心也不自动等于合理补全。

Pred的x8F5xyUWy9e_ca2525d9ecd14dba84c49be763f20adb：文字将mirror放在negative z，实际给z=7.5。Pred的zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a：文字说door在negative z，实际z=5.0。这直接证明语义计划没有正确转换成几何，不是JSON解析修好就能解决。

GT的8194nk5LbLH_9bdde31adaa1443bb206b09bfa3c474c有多个新增框与已有框高度重合，且stairs和lighting共享相同输出几何。报告高AABB重合只是诊断，不能把所有包含/支撑关系误判为碰撞。

当前结论：有接口、解析和状态拼接闭环，但没有可靠的hidden completion。G_obs保留成功不保证可见空地被保留。不能把这些G_full作为可信条件来评判Adapter好坏。也不能靠移动到z<0或删光视锥内框来宣称补全成功。

## 下一步仅检查可见几何能力

停止重复complete/complete-gt，保留当前结果。既有Observer只有2/150个GT对象达到同类3D IoU≥.25，而2D检测也有误差。因此用同12张图的GT类别+GT可见2D框定位目标，让同一Qwen3-VL估计3D。固定模型、相机、3D提示和坐标转换；不输入GT中心、尺寸、角度、深度、房间和隐藏对象。不换验证样本、不新增环境或模型。

已准备data/qwen_gt2d_probe_val12，150个可见目标，仅含2D/类别。准备模块prepare_gt2d_probe.py可以重建。该对照是诊断，不是最终推理依赖GT，也不是完成局部到全局任务的替代品。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export SIMPLE_WORLD_RUN=qwen_edit_pano/outputs/qwen_geometry_gt2d_run001
export SIMPLE_WORLD_SOURCE=qwen_edit_pano/data/qwen_gt2d_probe_val12

bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh observe

/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
  --bundle qwen_edit_pano/data/qwen_upstream_val12_v1 \
  --observer qwen_edit_pano/outputs/qwen_geometry_gt2d_run001/observer \
  --output qwen_edit_pano/outputs/qwen_geometry_gt2d_run001/review
```

评价重点是每栋建筑3D IoU、中心/尺寸误差、原图与3D投影，不是GT2D带来的接近满分的2D匹配。与旧Observer对比要注意对象集合不同，不单凭误差均值宣称提高。

如果GT2D条件仍有明显投影/深度/尺寸错误，降低Qwen角色为语义与2D定位，后续选择带相机/深度依据的几何方法；深度也不能自动给出完整物体尺寸。如果可见3D在GT2D下可用，再改定位与实例区分。无论哪种结果，都不代表隐藏补全已解决；隐藏几何需另选有布局先验和约束的方法，M3DLayout仅保留候选，不宣称即插即用。
