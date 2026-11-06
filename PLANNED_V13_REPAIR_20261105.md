# 全景规划接回原 v13：2026-11-05 修复说明

> 实际运行更正：`planned_v13_001` 为11份非空、3份空增量、4份规划阻断、2份无候选域；新规划出现循环截断、占位符原样放行，提案还有关系语法错误。进入原执行器的14份全部完成数值导出。下文CPU测试只证明工程性质，不能继续作为新入口默认推荐的依据。当前退回原 `world_completion.sh` 一轮主线；详见 [本轮实际根因回放](reports/planned_v13_001_diagnosis_20261105/README.md)。下文保留为实现记录，不代表本轮质量验收。

本次修复针对 `panorama_plan_001` 的真实根因。入口为 `scripts/planned_v13.sh`；旧失败入口和历史输出原样保留。实现位于 `world_pipeline/completion/plan_repair/`，原 v13 的 118 个固定源文件不修改。

目标仍为：单张局部 RGB 的冻结观测 `G_obs`，加上合理但不唯一的隐藏增量 `ΔG`。输出实例类别、尺寸、位置、依赖及可复查的假设和失败记录。规划和框级检查通过，不表示恢复了隐藏真值，也不表示已经证明因果识别或精确 RGB 一致性。

## 根因与对应修复

实际回放证据见 [失败审计](reports/panorama_plan_001_root_cause_20261105/README.md)。

| 原因 | 修复 |
|---|---|
| 观测必须恰好分区一次，遗漏温控器就阻断整个隐藏世界 | `observed_refs` 只作可选、可重叠的注释；保留观测由程序复制和逐对象比对保证 |
| 规划复述照片，再把可见内容和墙地面列作“新增” | 分开 `visible_unmodeled`、`architecture_assumptions` 和 `areas.hidden_intent`；只有隐藏增量进入旧 `new_objects` 接口 |
| h0 等假设 ID 成为物体宿主 | 使用旧实例/宿主验证器，宿主必须引用合法观测对象或同组新对象；绝不自动改宿主 |
| 原图再次驱动绝对 XYZ，把照片远处当作相机背后 | 全景规划只输出定性方向；恢复原尺寸冻结、相对宿主坐标及原相机区域编译 |
| 同模型整体“合理”评价放过可见实例换 ID | 逐实例身份核对须给出观测 ID 或原图位置证据；不确定项明确保留；有明确冲突则保留并拒绝整份候选 |
| 运行成功与补全质量混在一个 failed 计数里 | 分开请求/解析/编译异常、布局候选拒绝、无合格候选、采样空增量、非空导出；均可从页面查看 |
| 报告重建导致已存在的 views 目录报错 | 按完整状态、图像和 HFOV 绑定已生成视图；重复 report 不重新调用模型 |

身份核对仍由同一本地 Qwen 完成，可能误判或漏判。它只提供具体证据和待审项，不能用作独立的语义正确性证明。实现没有靠删除单个冲突对象、改位置或反复采样到非空来制造成功。

## 实际执行链

1. 复用 `local_observation_fixed_002` 的五张 RGB、45 个可见实例和官方 WildDet3D 产物，逐字节绑定；不重跑观察器。
2. 每张图用本地 Qwen 提出两个定性全景假设。规划不猜独立的地面高度或精确房间边界。
3. 两个假设各自建立独立 v13 候选域；每个域仍使用原 `surrounding` / `belongings` 通道和完整功能组语法。同一功能可以包含多个共存组。
4. 候选进入原 v13 池化、H/A/V 机制估计、固定预算采样、宿主依赖诊断、尺寸及坐标编译。
5. 每个全景假设默认采样两个世界，即每图四份、五图共二十份候选。各世界保留原始 `G_obs` 前缀；新增库存与 `semantic_world.json` 对齐。
6. 自动生成原图投影、俯视图、六方向、ERP 状态框图和交互三维图；未通过数值检查的候选可视化保留并明确标识。

全景假设只对候选供应进行条件化。没有把不同分支当成已校准的后验样本；v13 内部各角色仍采用原有近似机制。因此规划对跨角色依赖的表达能力尚有限，不能宣称已经解决联合世界分布学习。保留原执行器是为了隔离这次修复的影响。

没有强制新增类别、最低物体数、空间填满率或方向配额。原接口每个 alternative 最多四组、24 个实例，是已有调用容量，不是生成目标。空增量是允许的采样结果；全空提案或所有提案被拒则另记“无合格候选”，不伪装为有效空世界。

当前冻结观察器在厨房只检测到三个吧凳。规划中的可见厨岛、橱柜和设备如被识别为漏检，会记录在 `visible_unmodeled`；本次不把它们移动到相机背后冒充新增，也不声称已经修复观察器漏检。

## 手动执行

使用一个新目录。下面是一条完整运行指令；不用先 plan 再 layout。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_LAYOUT_RGB=0
export CAUPANO_PLANS=2
export CAUPANO_WORLDS_PER_PLAN=2
export CAUPANO_DRAWS=4

srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/planned_v13.sh run \
  qwen_edit_pano/outputs/planned_v13_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

入口：`qwen_edit_pano/outputs/planned_v13_001/index.html`。

- 顶层页面：每张图的全景假设、隐藏区域规划、可见漏检、建筑假设、异常原始原因。
- `branches/plan_01/report/index.html`、`branches/plan_02/report/index.html`：真实三维结果与候选，显示新增数量。
- `results.json`：完整状态分类和执行错误。
- `branches/plan_XX/scenes/SCENE/completion/proposals/qwen/`：完整候选、拒绝证据、空提案和原始调用。
- `branches/plan_XX/scenes/SCENE/completion/end_to_end/qwen/completion/world_XX/`：`G_full.json` 或明确未通过的 `G_full_candidates.json`，及语义世界、尺寸/布局、宿主干预和依赖记录。

退出码含义：`2` 为真实执行异常；`1` 为至少一张图没有非空合法导出；`0` 为每张图至少有一份非空数值导出、仍需语义检查。部分备选世界被拒或采样为空不单独使整批被叫作“执行异常”，但全部计数与失败原因保留。

重新生成报告只需 CPU，不调用模型：

```bash
bash qwen_edit_pano/scripts/planned_v13.sh report \
  qwen_edit_pano/outputs/planned_v13_001
```

若作业中断，使用同样的 srun 资源参数，将尾部命令改为：

```bash
bash qwen_edit_pano/scripts/planned_v13.sh resume \
  qwen_edit_pano/outputs/planned_v13_001
```

已完成阶段复用，不把失败标记删除后重新抽样。无最终回执的被杀调用会保留原始记录并明确说明中断位置；不能保证中断在任意写文件时刻都可以无损续跑。

## 验证范围

CPU 回归覆盖真实五个分区失败的遗漏/重复引用、厨房真实 h0 无效宿主、五场景 17 份原有效语义程序、旧数值拒绝分类，以及完整原 v13 执行与真实渲染。集成测试中的规划/提案/机制回复是显式合成夹具，不代表真实 Qwen 推理效果。

```bash
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/caupano_world_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -m unittest qwen_edit_pano.world_pipeline.completion.plan_repair.test_repair -v
```

本次未启动 GPU 推理或 API。运行后首先应判断全景规划是否确实描述视野外的合理延伸，以及可见漏检是否被冒充隐藏物体，再检查对象图和原相机投影；不以新增数量或单一退出码替代这些判断。
