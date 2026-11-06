# 近似合理的完整世界假设：repaired_003

**后续状态：joint003已完成20份导出，但用户认为补全质量仍不如旧版。当前默认主线已恢复原v13，详见 [V13_RESTORED_20261105.md](V13_RESTORED_20261105.md)。本页保留为近似验收与缓存修复的历史记录，不再作为默认运行建议。**

用户最新要求：布局不必准确精准，差不多合理即可。目标仍是保留单张局部RGB中的可见实例，推断若干语义和空间合理的周围完整世界状态 `G_full = G_obs ∪ ΔG`。本次据此区分实际矛盾、近似偏差和未验证假设，不再把精确几何认证作为所有输出的前提。

## repaired_002 的实际问题

20份样本中8份含新增导出、8份布局拒绝、4份提案阻断。检查的是实际回复、固定尺寸/姿态、执行记录及可视化数据，原1270个运行文件保留。

|问题|实际证据|修改|
|---|---|---|
|未知墙面直接当作非法|8份拒绝布局共有14条未验证建筑支撑错误；实际原视锥违例和对象宿主接触失败均为0|wall/ceiling保留为明确的未测量支撑假设及警告，不再自动否决完整世界|
|把定性关系当成精确硬约束|6条关系拒绝；例如shelf的中心已高于closet，仍因两者整框没有上下完全分离被拒；“behind”还有语义叙述/相机方向歧义|定性方向、全框上下次序及假设边界作为可审阅偏好；位置求解以保持原假设附近的合理布局为主|
|坐标传递存在实际错误|已有侧面Cartesian初值被旧编译器按原点Z的符号再次解释成后方射线；出现114.76米、31米等位置|接入直接Cartesian实例化，使用原先选定的侧/后半空间及原点；不再按射线重复投影。这是代码修复，与放宽精度要求无关|
|厨房重复抽中失败提案|H0提案把吧凳/柜台支撑关系弄错，修复后又把观测吧凳当作宿主；同一场景H1已有合法程序，却输出4份blocked_proposal|保留失败提案与原始概率质量，抽样条件改为“当前已有合法语义提案”；厨房采用已存在的H1，不修改观测、不自动改宿主、不伪造失败库存|

详细原始证据见 [diagnosis.json](reports/joint_world_repaired_002_diagnosis_20261105/diagnosis.json)。先前更严格的检查及失败数据仍保留用于对照。

## 近似验收规则

|继续阻断的实际问题|允许导出并附警告的情况|
|---|---|
|改变或丢失可见实例；身份或宿主非法；无效尺寸/坐标|未重建墙/顶面的安装平面，明确记录为假设|
|新增框与原始观测视锥冲突、包住相机|定性left/right/above/behind等偏差；不冒充精确关系已满足|
|明显物体穿插；支撑物完全没有接触；floor对象没有可用地面假设|轻微预测AABB穿插、可接触的局部悬挑、小幅地面接触残差|

轻微穿插的默认工程容差为最浅方向不超过3厘米、且交集体积不超过较小框体积的5%；否则仍拒绝。地面接触容差为3厘米。它们是针对不确定预测框的工程选择，不是声明WildDet3D具有此精度，也不是按本例类别定制。未知建筑支撑不代表完成了墙面重建；`G_full`明确保存 `environment_support_assumptions`、`layout_warnings`，且 `physical_geometry_verified`仍为false。

完整库存、宿主、尺寸保持固定，不通过删对象、缩尺寸、改观测或铺满方向提高导出率。`strict_geometry_contract_passed`和原严格检查结果继续保存；当前 `world_state_ready_scope`明确为近似世界假设，仍需语义审阅。

## 提案失败与概率模型

保留原始H权重和F参考法则为 `causal_model_unconditioned.json`，另生成供输出使用的可用提案模型 `causal_model.json`，条件为“语义提案可执行”，不是“非空”或“布局成功”。合法空假设仍可抽中；布局失败仍如实记录，不重抽直到成功。

`proposal_coverage.json`记录每份H原权重、合法F质量、失败F状态、排除质量及条件化后权重。例如厨房保留的原参考质量为0.1、排除的未解决提案质量为0.9；输出采用H1仅表示它当前可执行，不能说H1是真实房间的100%后验。失败记录不再被冒充可输出的空世界或反复抽成world失败。

运行完整且只有这些假设/偏差时，状态为 `complete_with_warnings`、退出0。实际调用异常、全部提案失效或仍有重大几何矛盾时继续返回非零。没有仅改退出码掩盖错误。

## 已完成的真实缓存复核

[offline_geometry_replay.json](reports/joint_world_repaired_002_diagnosis_20261105/offline_geometry_replay.json)使用本次16份已有完整尺寸、地面、原点与姿态的真实预测重新编译，16份均通过近似检查；未新增模型回复、未丢弃库存或改尺寸。异常114米案例修正为约4.7米。这里不包含厨房4份尚无几何的库存。

CPU可视化预览为 `outputs/joint_world_repaired_002_approx_preview_001/index.html`：16份含新增导出，厨房4份为 `pending_local_inference`。预览入口禁止初始化模型，未调用API或GPU；待推理不代表又生成了4个失败世界。每份世界都有原图、原视角、俯视图、六方向、ERP状态框和交互3D，警告可在首页展开。

45项CPU测试通过，测试与来源验证见 [本次报告](reports/joint_world_repaired_002_diagnosis_20261105/README.md)。原v13观察器和共享核心未改，冻结的45个观测实例逐项保留。

## 用户执行：复用缓存，只推断缺少几何的世界

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/joint_world.sh repair \
  qwen_edit_pano/outputs/joint_world_repaired_003 \
  qwen_edit_pano/outputs/joint_world_repaired_002
```

注意这是 `repair 新目录 原补全目录`，不是 `run 新目录 观测目录`。读取本次运行已有的H、完整候选和权重并记录来源；不重跑观察器，也不让Qwen重新提出所有场景。16份语义签名匹配的世界复用经哈希校验的尺寸和姿态，只重新进行近似布局编译和可视化。厨房改用已有合法H1，缺少对应几何的4份才调用部署的本地Qwen3-VL。没有API。

结果为 `outputs/joint_world_repaired_003/index.html`。新目录交付时不存在；旧002不改写，不要直接resume旧002。中断后用同一srun资源参数，把末尾改为：

```bash
bash qwen_edit_pano/scripts/joint_world.sh resume qwen_edit_pano/outputs/joint_world_repaired_003
```

如需在全新观测上重新提出世界，原 `run NEW_ROOT OBSERVATION_SUITE`仍可使用，默认同样采用近似规则。本次建议先用repair，避免重复消耗已经拿到的模型预测。

## 实现位置

- [近似验收](world_pipeline/completion/joint_world_acceptance.py)：明确硬错误、容差、警告和环境支撑假设。
- [Cartesian实例化](world_pipeline/completion/joint_world_initialization.py)：保持已选择的坐标与半空间，消除二次射线解释。
- [可用提案条件化](world_pipeline/completion/joint_world_model.py)：失败质量保留在覆盖率记录，完整宿主依赖SCM继续使用。
- [缓存修复](world_pipeline/completion/joint_world_repair.py)：独立新目录、来源绑定、语义签名/尺寸/姿态校验，仅缺失几何走本地模型。
- [导出与报告](world_pipeline/completion/joint_world.py)：共享新预测/缓存导出逻辑，保留完整G_obs与G_full，展示假设和警告。

新厨房布局仍需真实推理后检查，不能把已复核的16份结果当作20份全部完成，也不承诺语言模型每次均提出合理对象图。当前新增数量和单一H覆盖有限，不通过设置类别或数量配额来掩盖该限制。
