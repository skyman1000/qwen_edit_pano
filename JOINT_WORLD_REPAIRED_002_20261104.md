# repaired_001 根因与 repaired_002 执行（2026-11-05更新）

**本页为上一轮记录。** repaired_002已实际产生8份含新增、8份拒绝和4份提案阻断。用户现允许近似合理布局，当前修复、已生成预览和只补缺失几何的srun命令见 [JOINT_WORLD_APPROX_REPAIR_20261105.md](JOINT_WORLD_APPROX_REPAIR_20261105.md)。不要直接续跑本页旧002。

任务仍是从一张局部室内 RGB，保留可见实例 G_obs，推断若干与观测相容、语义及空间合理的完整周围世界假设 G_full = G_obs ∪ ΔG。未知区域允许多解；不要求照片证明隐藏物体存在，也不设置最少对象数、必出类别、方向配额或重抽到成功。

## 本次运行到底发生了什么

`outputs/joint_world_repaired_001` 的20份样本为：16份合法格式的空增量，2份布局接口失败，2份选中未解决的提案。没有含新增的 G_full 导出。83次本地生成全部以 EOS 结束，实际异常为校验 ValueError；退出1由 Python 的部分失败汇总产生，srun 传播这个退出码。

上一轮恢复了完整增量接口，但还保留了会把任务改偏的选择机制。此前 CPU 测试用合成模型选择了非空程序，未暴露该机制在真实 Qwen 下的吸收偏差；理想回复也未覆盖缺组、复制模板和空地面。这些是上一版实现和验证的不足。

|层次|真实证据|根因与修复|
|---|---|---|
|候选选择|9份H被代码自动加入空增量；36次程序选择中32次投给空候选，理由反复是“没有照片证据”“不需要更多物体”|把不确定世界补全变成了最小可见解释。删除自动空候选和这个必要性选择器；空世界只接受模型自己明确提出的假设|
|概率模型|平滑后多数 F 条件行给空候选0.9概率|F改为每份H内完整提案上的显式均匀参考分布。它是公开的建模选择，不声称识别或校准了真实隐藏世界后验；不按数量、成功率或非空条件重加权|
|全局/库存语义|出现 `coexisting activities`、`fill with the concrete activity or storage area`、`evidence and uncertainty`，厨房甚至原样继承示例bookcase|提示中的内容示例被当成答案。改用无具体物体的空字段结构，明确填写具体内容；运行时拒绝原提示的精确占位词。继续复用v13的身份、宿主和关系校验|
|多组布局|休息区实际有g0和g1，原点回复及修复回复都只给g0|请求未提供完整的逐ID模板，还携带重复对象、SCM状态和噪声。现在按确切组ID请求，每次有具体模板和全局精简上下文；不猜缺失的组坐标|
|坐标约束|卧室回复 `[0.515,3.348]` 被视为原点进入视野，修复原样重复|原点是几何初值，不能把模型给出可行证明作为先决条件。允许有限原点提案，再对固定装配做最小平移的左/右/后视锥外初始化，最后仍检查完整框、碰撞、支撑和关系|
|地面|两份非空世界都接受了 `anchor_ids=[]、floor_y_m=null、basis="evidence and uncertainty"`|未知地面被当作已完成阶段。优先由程序计算观测框底部的有限地面假设，让Qwen根据原图和语义选择确切ID；不让它抄空模板或计算坐标。选择未知则明确阻断；没有可用底部候选时才请求显式未测量高度假设|
|浴室提案|wall对象把观测柜子或水槽当作墙宿主，修复未改变|新提示明确环境表面和对象宿主的区别；一次返回占位词和结构错误，避免纠正占位词后才发现宿主错误。程序不会擅自清空宿主、改成floor或删物体|

完整原始证据和1178个原运行文件哈希见 [diagnosis.json](reports/joint_world_repaired_001_root_cause_20261104/diagnosis.json)。原运行文件没有覆盖。

## 当前模型与几何边界

```text
固定RGB / G_obs
→ 提出具体全局假设 H
→ H 条件下生成完整共存增量 F，验证全部身份、宿主与关系
→ q(H|RGB,G_obs,D) × Reference(F|H,D)
→ SCM：H → F → E；S_i = E_i AND S_host
→ 固定次数抽样（保留重复、模型提出的空世界与未解决失败）
→ 冻结完整库存 → 估计并冻结尺寸
→ 选择有来源、未测量的地面假设 → 精简上下文的逐组/分批姿态请求
→ 固定尺寸/朝向/相对装配的视锥外初始平移 → 原求解器联合修正
→ 原视角完整框、碰撞、支撑、地面和关系检查 → 导出与可视化
```

H保留语言模型的有限选择计票；`CAUPANO_DRAWS`现在用于H选择，不再用于F的照片必要性比较。F的参考法则明确包含同一H下的全部完整程序，含模型确实提出的空增量和未解决提案状态；失败既不改成空世界，也不通过删除失败质量后重新归一化来冒充成功。

最小平移初始化只调整装配在XZ上的整体平移，保持对象ID、宿主、尺寸、朝向和组内偏移；枚举左、右、后视锥外半空间，取满足完整装配AABB条件且位移最小的连续解。不设置均匀方向覆盖，也不把物体强行铺满房间。请求原点、可行初值、平移量及最终检查分别记录。它只提供可行初值，不证明最终布局满足碰撞和关系；最终仍由原求解器验证。观测宿主连接的树不随自由原点搬动。

地面候选来自预测框底部，而非标准相机高度或类别白名单。选择某个底部仍是假设该物体真正接地、且相机Y近似竖直，不是实测地面；Qwen可声明未知并使该世界阻断。浮在桌上、挂在墙上的物体不应被选为锚点，实际选择仍需人工审阅。

## 修改与验证

变更集中在 `world_pipeline/completion/joint_world*.py` 及相应测试。新增 [joint_world_initialization.py](world_pipeline/completion/joint_world_initialization.py) 实现几何初始化。原v13核心、观察器和共享几何求解器保持原文件；与v13绑定的116个源码哈希继续匹配。

- [tests.log](reports/joint_world_repaired_001_root_cause_20261104/tests.log)：36项CPU测试通过，包含真实客户端到导出/全套可视化的完整集成、宿主干预、观测保留、本次真实坏回复回归，以及30组变化尺寸/朝向/偏移的完整框视锥检查。
- [reference_ablation.json](reports/joint_world_repaired_001_root_cause_20261104/reference_ablation.json)：保持这次保存的原库存、H权重与种子，移除代码注入的空状态、替换F法则后，20次抽样得到18份非空结构样本、2份未解决提案。这个受控对照定位选择机制的偏置；它没有生成新模型回复或通过几何，不是18份有效补全。原候选的占位词、未验证墙面等问题仍如实保留。
- [preflight.log](reports/joint_world_repaired_001_root_cause_20261104/preflight.log)：当前代码实际prepare五图45个冻结观测，每图4份世界，通过本地模型文件/依赖检查，未加载模型。
- [validation.json](reports/joint_world_repaired_001_root_cause_20261104/validation.json)：源码绑定、旧文件哈希和新输出目录检查。

本轮没有执行GPU推理、提交作业或调用API。新提示下Qwen的实际语义质量仍待运行；不会把CPU替代引擎的回复当作真实补全效果。

## 手动执行

执行完整补全与可视化：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_WORLDS=4
export CAUPANO_DRAWS=4
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/joint_world.sh run \
  qwen_edit_pano/outputs/joint_world_repaired_002 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

`joint_world_repaired_002`交付时不存在。仅使用部署的本地Qwen3-VL，复用Qwen二维→官方WildDet3D三维的冻结观测及词汇映射，不重跑观察器。不要resume旧001；源码和机制已变，它仍保留为失败证据。

自动生成 `qwen_edit_pano/outputs/joint_world_repaired_002/index.html`。原图、冻结观测、原视角、俯视图、六方向、ERP状态框和交互3D沿用原项目可视化；新增主要看俯视、六方向和3D。新增的 `domain_review.html` 可直接查看所有H、完整增量、每个对象的类别/类型/宿主/位置意图，即使某个候选未被采中或未进入几何，也能检查提案情况，不能把这些候选数量当成已导出的实例。

`results.json`额外提供 `completion_summary`，区分含新增导出、仅观测和未导出数量。调用失败保留具体阶段名。几何拒绝保存 `G_full_candidates.json` 及候选可视化，不冒充合法G_full。

墙/顶面支撑仍需建筑面假设，缺少验证时可能拒绝；本轮没有通过取消这些检查、缩小尺寸或删物体提高成功率。厨房原观测漏掉中岛等仍存在，本次固定观测以控制补全变量，不把可见漏检物挪到视野外冒充新增。新的提案也可能失败或产生真正的空假设，报告会如实区分；不承诺20份样本全部通过。

本次新目录因中断续跑时，使用相同srun参数，末尾换成：

```bash
bash qwen_edit_pano/scripts/joint_world.sh resume qwen_edit_pano/outputs/joint_world_repaired_002
```

## 相关工作与本次取舍

重新核对 [SD-SCM / Language Models as Causal Effect Generators](https://aclanthology.org/2025.emnlp-main.107/)：其因果结构由用户指定、机制由语言模型定义，并不能把一次“可见证据是否需要更多东西”的选择当作隐藏世界后验。因此本次明确记录H机制、F的参考法则及模型内干预范围，没有声称修复概率校准或识别了真实房间因果。

[Holodeck](https://arxiv.org/abs/2312.09067)让语言模型提出空间关系，再优化布局。本次据此区分语言模型的坐标提案与几何约束求解：坐标可作为初值调整，最终完整框、观测相容与依赖仍要检验。这里是组织方式借鉴，未复现其资产、建筑生成或效果。
