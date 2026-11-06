# Qwen语义候选 → M3DLayout指定类别补全

这次回到目标主链，不重跑Observer，不训练模型，不让Qwen生成隐藏对象坐标。

```text
已有RGB预测G_obs + 同一局部图
   → Qwen提出至多4个隐藏对象实例（类别+文字关系）
   → M3DLayout.add_object(已有上下文, 指定类别, 文本)
   → 新对象几何转回Local坐标，基本筛选
   → 原G_obs + 接受的新增对象 = 候选G_full
```

## 直接执行

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/completion/run_hybrid.sh propose
bash qwen_edit_pano/world_pipeline/completion/run_hybrid.sh run
bash qwen_edit_pano/world_pipeline/completion/run_hybrid.sh review
```

propose：12张原图，每图一次Qwen调用；只生成英文描述、候选类别、粗略关系。GT隐藏对象不进入提示。候选词表取官方M3D类别与Adapter可转换类别的交集。

run：加载现有autoregressive_59000.pth一次，逐图运行GT可见和Pred可见两个条件，共24个候选世界。两组共用同一份语义提议，因此隔离的是结构化观测输入，不是纯GT语义Oracle。每个候选最多采样3次；只把被接受的新增对象加入后续上下文。不会继续调用complete_scene无限扩张。

review：CPU生成同一报告，分别展示hidden_gt和hidden_pred。先看提议/采样/拒绝/接受数量，再看布局与原图。筛选后视锥内比例为零是筛选结果，不是模型天然学会了隐藏补全；sampling_audit.json保留所有原始提议。

现有qwen3vl环境，默认GPU2，每GPU命令内部srun申请一卡。修改节点：`export HYBRID_NODE=GPU4`。新实验目录：`export HYBRID_RUN=qwen_edit_pano/outputs/qwen_m3d_hybrid_run002`，三步一致设置。run会重新计算整组，不做复杂的断点状态管理。

## 输入与输出

默认使用data/qwen_upstream_val12_v1的12图。预测观测来自outputs/qwen_world_simple_run001/observer（实际RGB预测），不是GT2D/红框诊断输出。可用HYBRID_OBSERVER显式指定另一套同样格式的真实预测。

```text
outputs/qwen_m3d_hybrid_run001/
  proposals/<id>/proposal.json           # Qwen描述及类别候选
  completion/{gt,pred}/<id>/seed0/
    G_obs.json
    G_hidden.json
    G_full.json
    semantic_proposal.json
    sampling_audit.json                  # 所有接受/拒绝的原生框与原因
    omitted_context.json                # 未进入M3D的观测对象
    status.json                         # sampled/rejected/generated/outcome
  review/review.html
  review/metrics.json
```

全部拒绝时仍保存G_full=G_obs，但outcome=no_accepted_geometry，不能记作补全成功。格式错误另记failed，不代填空提议。

## 固定且公开的实验约定

- 冻结两模型，保留官方文本分支、词表与demo归一化统计，room mask为全1。
- 调用官方add_object；输入上下文不带START/END，每次只提取倒数第二个新对象。显式传同设备one-hot类别，避免官方整数类别路径的CPU/CUDA不一致。
- 原生size解释为half extents，与已有native/visible probe约定一致；导出旋转角点的Local AABB完整边长。没有增加Adapter字段。
- 第一版明确假定相机近似水平、距假设地面1.5m，native原点水平位置与相机一致。Local AABB作为native yaw=0框；这不是恢复了真实房间坐标或OBB。GT与Pred用相同桥接，不偷用GT相机/房间配准改善GT分支。
- 原始G_obs全部保留在最终输出。仅可映射且几何有效的观测进入M3D上下文；door采用panel_door代表，其他无法映射者记录omitted_context，不能宣称模型看到了全部原观测。原观测不会被改成该细分类。
- 本轮仅做视野外候选：整个OBB位于同一相机视锥平面之外才接受，检查保守，不处理视野内遮挡物。不移动坐标、不夹紧值、不强行z<0。另拒绝同类AABB IoU≥.5重复框；不把AABB重叠当完整物理碰撞判断。
- 不保证碰撞、落地、房间边界、关系文本均满足；这些需要看图及原始采样，不能由过滤结果冒充已解决。
- 不宣称候选G_full是完整真值。此实验回答预训练M3D能否比Qwen自由写坐标提供更有用的隐藏布局，不承担整条研究的最终有效性证明。

验证：`bash qwen_edit_pano/world_pipeline/completion/run_hybrid.sh check`。CPU覆盖两阶段文件流程、坐标及半长转换、START/END处理、拒绝不污染上下文、观测保留和Adapter tensorize。GPU实测由用户执行。
