# 当前执行：已下载建筑的完整GT派生，再做Oracle训练

2026-10-06：磁盘可用约77GiB。72栋建筑的全量原始包/派生结果空间尚未确认足够；先将已下载齐GT依赖的6栋建筑全部派生，目标470 train/42 val。这不是全数据集GT完成。5栋train与1栋val建筑独立，初步验证的建筑多样性仍有限。

助手已完成纯CPU清单准备和官方身份/split检查：`qwen_edit_pano/data/paired_gt_available_v1`。从paired_full_v1逐行选择，保持Local、ERP、相机对齐字段不变，不改变正在训练的正式清单，不复制图像、不新建条件缓存。`selection.json`记录源清单哈希、建筑、原始依赖和数量。新调度入口支持GT_PAIRS_ROOT；不设置时仍沿用原72栋全量方案。

## 用户执行

同一终端设置；换终端需要重新设置，避免误用默认全量目录：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export GT_PAIRS_ROOT=qwen_edit_pano/data/paired_gt_available_v1
export GT_SOURCE_ROOT=qwen_edit_pano/data/gt_sources_available_v1
export GT_EXPORT_ROOT=qwen_edit_pano/data/gt_world_available_v1
bash qwen_edit_pano/world_pipeline/run_gt_full.sh plan
```

预期470/42、houses=6、fully_present_houses=6。现在不需要执行download。

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_full.sh build
```

复用已有阶段，补齐缺少的ERP对齐、mesh投影和真实世界状态。没有GPU任务。24小时是单次时限；超时中断后原命令续跑。若是几何审核错误而非中断，检查build_progress.json和logs，不要无限重跑或删除告警。470/42是构建目标，不承诺每一条都会通过几何核验。

源构建成功后：

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_full.sh export
```

输出gt_world_available_v1下的G_full/G_obs/G_hidden、可见性、预览和来源信息。同一命令自动恢复；旧gt_world_expanded_v1保留。源状态不完整时export明确拒绝，不把失败样本悄悄当作完成。

## 导出后不要直接训练

检查EXPORT_COMPLETE、failures、manifest/bridge、投影和对象上限。修正确定的坐标/表示问题，区分Local/ERP采样密度告警与真实错误；审核记录不自动全部approved。确认有效train/val及建筑数后，生成新oracle bundle，保留默认正式paired_full_v1/cache来源。GT子集清单不是新的Panorama LoRA训练清单。

然后固定原选择的paired_full_63637/checkpoint-epoch001及Edit基础模型；只训练World Encoder＋Adapter。输入Local RGB走Edit原生条件，真实G_full走结构分支，GT ERP RGB只作损失监督。先比较相同预算的constant、observed、full；observed使用可见对象的GT三维信息，并非Observer预测。验证集同输入/seed比较baseline与各分支，full加shuffled诊断。训练步数根据实际通过审核的样本数确定，不机械沿用7条pilot的200步。

这一步回答GT结构是否改善生成；有可信收益后再推进Observer几何估计和Completion。全量其他66栋的补下载、派生及多建筑验证仍是后续扩展任务，没有被此次512条候选替代。
