# 数据、路径迁移与清理审计（2026-10-06）

本次只修改入口/路径兼容，生成审计清单；没有删除、移动数据或checkpoint，没有下载或运行GPU任务。

## 路径迁移

实际目录已从research/world_pipeline移动到qwen_edit_pano/world_pipeline。两处都是项目下两层，common.ROOT的parents[2]仍然正确。数值几何、表示和模型代码不因搬迁改动。

已修改run_gt_full.sh的模块入口与5份启动/说明文件引用，新命令为python -m qwen_edit_pano.world_pipeline.<module>。保留research/world_pipeline -> ../qwen_edit_pano/world_pipeline兼容符号链接；它不复制数据，用于历史绝对路径、旧命令和已有来源记录。不要删除这个链接来“清理重复文件”。

修正prepare_gt恢复校验对符号链接路径的归一化；仅接受已知旧版编排脚本哈希，未知代码改动仍拒绝。旧resume_contract若需要迁移会保留before_relocation副本，历史GT数值与审核结果不改写。

验证：21项CPU几何/解析/恢复测试通过；新增搬迁签名与未知哈希拒绝断言的测试通过。旧research入口的oracle_train --preflight也通过（7 train/0 val，无模型加载）。新run_gt_full.sh plan通过。没有对现有GT目录执行真正resume或重算。

## 原始数据在哪里

检索了整个项目，以及用户目录下非环境/非隐藏缓存目录的house/region/camera包与.house文件；另外检查了Hugging Face数据缓存和当前清单的Arrow引用。未发现项目外第二套同名3D GT包。没有扫描整个集群所有磁盘，改名的离线总压缩包不在本次同名搜索保证范围。

1. 原始Matterport包：benchmark_assets/Matterport3D_raw/v1/scans/<house>/。
2. 完整polished ERP：/data-nfs/gpu1-2/u13529658780/.cache/huggingface/datasets/Insta360-Research___matterport3_d_polished/...，当前清单直接引用42个Arrow文件，约26.273 GiB。它们虽然在.cache下，却是正在使用的训练图像来源，不能按临时缓存删除。
3. metadata、stitched captions、benchmark参考图位于benchmark_assets的对应目录。它们不是缺失3D GT的替代品。

原始ZIP统计（存在性及中央目录可读性，本轮没有全量CRC）：

| 包类型 | 建筑数 | 文件体积 |
|---|---:|---:|
| house_segmentations | 7 | 0.966 GiB |
| matterport_camera_intrinsics | 7 | 0.001 GiB |
| matterport_camera_poses | 7 | 0.004 GiB |
| matterport_skybox_images | 90 | 17.276 GiB |
| region_segmentations | 7 | 0.970 GiB |
| undistorted_camera_parameters | 7 | 0.001 GiB |
| undistorted_color_images | 7 | 1.101 GiB |
| undistorted_depth_images | 7 | 7.491 GiB |
| undistorted_normal_images | 7 | 17.052 GiB |

全部146个已发现ZIP的中央目录可读；这不等于所有内容CRC重新验证。5个旧tmp对应的完整ZIP另外实际执行了CRC检查。

skybox齐全的90栋建筑与3D GT包齐全的7栋不是同一个覆盖层次。7栋GT中6栋属于本轮train/val，另1栋Vt2qJdWjCF2不属于本轮。需要补齐的是其余66栋目标建筑的GT配套包，而不是重下已完成的全景图。

| 数据层 | train | val |
|---|---:|---:|
| 当前固定配对清单 | 7521 | 925 |
| 已有原始GT包覆盖的配对身份 | 470 | 42 |
| 已有旧世界状态匹配 | 251 | 39 |

旧world_state_pilot总计315条；其中290条匹配当前train/val。当前gt_world_expanded_v1清单31条、无EXPORT_COMPLETE，属于中断/未完成结果，不能当作290条已生成，更不宜直接删除。gt_sources_full_v1目前只有清单/配置，还没有开始全量源状态构建。

## 空间与正在使用的产物

项目du占用约144 GiB；df显示的3.6TiB文件系统及3.4TiB已用量不等于这个项目占了3.4TiB。审计时可用约7.6GiB。

| 目录 | 文件逻辑体积 | 建议 |
|---|---:|---|
| benchmark_assets | 47.941 GiB | 保留原始资产 |
| qwen_edit_pano/outputs | 10.023 GiB | 正式训练仍在写入；按文件分类 |
| qwen_edit_pano/cache | 30.679 GiB | 保留，正式图像条件缓存 |
| qwen_edit_pano/data | 0.382 GiB | 保留固定清单、GT与审核记录 |
| qwen_pano/outputs | 48.170 GiB | 包含不可删的GT源资产；旧optimizer可条件清理 |
| qwen_pano/cache | 3.005 GiB | 旧基线复现缓存，非第一清理目标 |
| DiT360/outputs | 2.784 GiB | 旧比较实验，需确认是否退役 |
| research/audits | 0.001 GiB | 很小，保留证据 |

已只读查询Slurm：作业64452在GPU4运行，输出为qwen_edit_pano/outputs/paired_full_63637。日志处于第7/25 epoch，最后完成checkpoint-epoch006（11286 optimizer steps）。该目录、epoch006恢复点及日志保留。epoch001还是当前Oracle绑定的固定基线，必须保留。不要用新epoch覆盖原oracle实验基线。

以下同样必须保留：

- qwen_edit_pano/cache/paired_full_v1及其15042个原图/翻转条件缓存；paired_full_v1/official_building_split清单。
- qwen_edit_pano/data/gt_world_pilot_v1、oracle_pilot_v1；oracle_full_smoke_v2及其重载结果。
- qwen_pano/outputs/caupano/{camera,house,region,object_link,canonical,erp_alignment,geometry,world_state}_pilot。这些outputs是GT生成输入，不能因目录名叫outputs就删。
- qwen_pano/outputs/pano_official_full_59263/checkpoint-epoch025完整状态；epoch005权重被多项旧gated/PanFusion实验引用。
- Hugging Face里的Edit2511、Qwen3-VL、旧Qwen基线模型以及实际Arrow数据源。
- research/world_pipeline兼容链接。caupano实验间也有真实依赖符号链接，不能整组随意移动/删除。

## 可删除候选（本次未删除）

精确到每个文件的清单见deletion_candidates.tsv/json。所有体积都是当前测量；目录内部可能持续变化，实际删除前应重新确认作业状态。

### A：低风险，合计约0.656 GiB

5个旧无元数据tmp碎片合计642.656MiB：逐字节与完整ZIP相同长度的前缀一致，且对应完整ZIP CRC通过：

- `benchmark_assets/Matterport3D_raw/v1/scans/ULsKaCPVFJR/tmpksf_fj71` — 313.750 MiB。全部临时字节匹配完整ZIP前缀；对应ZIP CRC本轮通过：undistorted_normal_images.zip；旧无续传元数据临时片段；完整ZIP保留。
- `benchmark_assets/Matterport3D_raw/v1/scans/ULsKaCPVFJR/tmprxfe2_hk` — 0.500 MiB。全部临时字节匹配完整ZIP前缀；对应ZIP CRC本轮通过：house_segmentations.zip；旧无续传元数据临时片段；完整ZIP保留。
- `benchmark_assets/Matterport3D_raw/v1/scans/29hnd4uzFmX/tmp7dhq5c6b` — 40.000 MiB。全部临时字节匹配完整ZIP前缀；对应ZIP CRC本轮通过：matterport_skybox_images.zip；旧无续传元数据临时片段；完整ZIP保留。
- `benchmark_assets/Matterport3D_raw/v1/scans/B6ByNegPMKs/tmpg5a26q9q` — 288.328 MiB。全部临时字节匹配完整ZIP前缀；对应ZIP CRC本轮通过：matterport_skybox_images.zip；旧无续传元数据临时片段；完整ZIP保留。
- `benchmark_assets/Matterport3D_raw/v1/scans/x8F5xyUWy9e/tmpa9uj027j` — 0.078 MiB。全部临时字节匹配完整ZIP前缀；对应ZIP CRC本轮通过：undistorted_depth_images.zip；旧无续传元数据临时片段；完整ZIP保留。
- `qwen_edit_pano/outputs/oracle_full_smoke_v1/target_posteriors` — 28.001 MiB。失败smoke的可重建VAE后验缓存；v2训练与重载均完成；v1缓存需重算；保留父目录配置和失败记录。
- `qwen_edit_pano/outputs/paired_smoke_val_v1` — 1.215 MiB。失败推理目录；已有成功paired_smoke_val_v2；删除失败运行留下的图片和配置；建议先保留run_config.json。

### B：释放空间较明显，但会失去旧epoch续训能力，约16.884 GiB

qwen_pano/outputs/pano_official_full_59263/checkpoint-epoch001 至 checkpoint-epoch024 各自的optimizer.bin，共24个文件。旧Qwen-Pano已经完成epoch025；保留所有推理权重、配置、日志以及epoch025完整checkpoint，可以继续旧模型推理，但不能从删去optimizer的旧epoch精确恢复训练。

这是有明确代价的候选，不叫“无用文件”。如果仍需要回到某个旧epoch续训，排除对应文件。不要将这项建议套用到正在运行的qwen_edit_pano/outputs/paired_full_63637。

### C：仅在决定放弃对应历史实验时考虑，约6.891 GiB

- `qwen_edit_pano/outputs/paired_smoke_v1/smoke-checkpoint` — 1.407 GiB。删除后不能重放该smoke推理；先保留JSON报告；并非重复权重。
- `qwen_pano/outputs/zero_shot_2512` — 2.756 GiB。仅在放弃对应比较实验且保存指标/配置后考虑；重新生成需要GPU。
- `DiT360/outputs/mp3d_stitched1092_perid_g3_s28` — 2.729 GiB。仅在放弃对应比较实验且保存指标/配置后考虑；重新生成需要GPU。

不建议仅为整洁删除Observer小JSON、research审计或当前GT中间结果：省不了多少空间，反而丢失定位依据。未生成rm脚本，不做默认批量清理。

## 下一步

1. 先按A/B/C清单决定保留策略、处理存储空间。A+B约17.54GiB，但全量GT最终导出按现有样本体积粗估约60GiB，尚未包括新增原始包及几何中间文件；不能认为清理后就一定装得下全量。
2. 当前代码默认数据位置不变。若换磁盘，应先确定实际目标路径，再调整原始数据及派生输出位置，不能直接搬正在训练引用的Arrow/skybox/cache。
3. 使用新入口`bash qwen_edit_pano/world_pipeline/run_gt_full.sh plan`核对72栋目标清单。空间充足后依RUN_GT_FULL.md顺序运行download、build、export（用户手动srun，无GPU）。
4. 下载已有包会跳过/校验，build复用旧阶段；export可断点恢复，仍保留可见性告警。不以缺失状态筛选来冒充全量。
5. 导出完成后核验GT与独立val，再训练Encoder/Adapter的constant/observed/full对照。当前无需重跑Observer或改正在训练的Panorama LoRA。
