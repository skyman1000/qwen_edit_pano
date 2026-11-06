# 原 v13 的候选接口与供给修复

依据：[本轮五图实证分析](reports/v13_original_recovery_001_review_20261105/README.md)。原主线已复现，40份关键产物与原纯本地版一致。当前差距集中在候选覆盖及关系表示，不能再用整体重写SCM、提高存在概率或填满空间来解决。

本次新增 `scripts/candidate_world.sh`，只替换候选提案接口并提供固定两轮本地预算；118项原v13固定源码不修改。原 `world_completion.sh` 与所有旧输出继续保留。新实现位于 `world_pipeline/completion/candidate_repair/`。

## 修复如何对应实际原因

| 已确认原因 | 实现 |
|---|---|
| 提示同时定义旧深度词和禁止旧深度词，验证器再拒绝 | 新提案提示只有一套 `farther_than/nearer_than` 原相机关系契约；不拼接相互矛盾的旧提示 |
| 落地灯“照亮桌子”被当作整个灯框必须高于桌框 | `functional_links` 显式记录 illuminates/faces/serves 等目的；above/below 需明确 whole_aabb 范围后才进入旧硬几何规则 |
| inside/in 与支撑混用，直接改near会丢掉含义 | 仅当原声明已是同宿主shelf时，将inside/in保留为contained_in语义假设并保存绑定；top、错宿主等矛盾仍给出具体纠正反馈 |
| 新增少主要由于候选域窄，休息区当前最多可达8 | 两轮固定预算，第二轮看到第一轮的功能、根对象类别和重复集合摘要，提出完整的替代或补充功能程序；不是无限重抽 |
| 长回复循环后把四万字符原文再次放进修复提示 | 请求最多两次，独立种子、短字段、有限纠错反馈；原始长回复保留，不整段回灌 |
| 数量可能被22个同宿主杯子撑大 | 页面单列同宿主/同类型/同理由的重复集合供检查；不删除这些实例、不以总数作为改善证明 |

未增加独立全景长文规划或身份自审前置。原H假设步骤仍是原流程的一部分；其描述只作简短候选组织信息，不要求完整观测分区。

## 不应误解的边界

- contained_in 是与已有 shelf 宿主一致的**内部储物假设**，不是已验证的网格体积包含；不能把 top 安装无条件改成 shelf 或 near。
- functional_links 是显式功能意图，并非新增的现实因果边。没有把照明、朝向意图声称为已经完成光学或人体可用性验证。
- 原四份失败回复含有实际矛盾，不能无损全部接受。回归验证的是：拒绝有具体原因、对象不被静默删除、合法的已有shelf包含语义可以保留；未来模型仍可能给出不合格回复。
- 原数值编译器继续执行原相机、宿主、尺寸、排序和碰撞检查。新代码不通过删对象、移动观测或切换方向把失败改成成功。
- 新候选域对应的H/A/V经验表重新估计，估计法则、平滑和采样算法不变。候选变多不自动提高存在概率；新候选也不等于校准的世界后验。
- 两轮本地Qwen不保证等价于交接前的Qwen+API候选能力。实际是否更合理，须查看功能覆盖、独立宿主、用品关系和原图相容性，而非奖励物体数。
- 当前45个冻结观测原样保留。厨房可见漏检没有在本次修复，也不允许把漏检厨岛搬到视野外当新增。

## 完整运行命令

使用一个尚不存在的新目录。只调用部署的本地Qwen，不调用API，不重跑观察器。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_LAYOUT_RGB=0
export CAUPANO_CANDIDATE_ROUNDS=2
export CAUPANO_WORLDS=4
export CAUPANO_DRAWS=4

srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/candidate_world.sh run \
  qwen_edit_pano/outputs/v13_candidates_fixed_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

流程：冻结观测复用 → 两轮候选 → 原池化 → 候选对照页 → 原H/A/V → 原尺寸/坐标 → 完整可视化。默认五张图、每图四份世界。

输出：

- `outputs/v13_candidates_fixed_001/index.html`：总览、错误和状态。
- `outputs/v13_candidates_fixed_001/candidates.html`：池化后就生成，可在三维阶段运行期间查看候选、宿主依赖与功能意图。
- `outputs/v13_candidates_fixed_001/report/index.html`：原图投影、俯视图、六方向、ERP状态框图、交互三维，包含明确标识的未通过候选。
- `outputs/v13_candidates_fixed_001/candidate_report.json`：功能覆盖、根对象类别、重复集合、下一阶段调用预算。
- `outputs/v13_candidates_fixed_001/results.json`：执行异常、空增量、几何拒绝、非空数值导出。
- 各场景 `completion/proposal_contract_rounds/round_00`、`round_01`：逐轮原始回复、错误回执和完整程序；最终程序在 `completion/proposals/qwen`。

候选数跨不同备选程序统计，不能当作一份世界数量。对照统计只读原结果，不把旧Qwen/API候选送入模型。新场景没有同观测基线时照常生成，不设旧结果前置依赖。

退出码：2表示真实请求/接口/执行异常；1表示某场景没有非空合法导出；0表示每场景至少一份非空数值导出，仍需语义核查。空增量和数值拒绝保留，不自动叫作模型执行错误。

## 只查看候选，或继续已有候选

若想先检查候选，把上述srun中的 `run` 改成 `candidates`，并用另一个新输出目录。它只执行到候选池和依赖页面，不调用机制或布局，也不声称已有三维结果。

之后在srun分配中运行：

```bash
bash qwen_edit_pano/scripts/candidate_world.sh complete 输出目录
```

它复用已绑定候选，继续原机制和三维；不重新抽候选。完整作业中断后的入口为 `resume 输出目录`。报告可在CPU重建：`bash qwen_edit_pano/scripts/candidate_world.sh report 输出目录`。

已完成请求有输入/源码/回复绑定；失败回执保留首次具体原因，不删标记重抽。进程在无最终回执时被杀会记录中断，不能保证任意写文件时刻都可无损续跑。

## 验证

12项CPU回归通过，覆盖：统一关系词表、真实四份失败提案反馈、真实shelf包含语义、top/宿主矛盾拒绝、真实落地灯候选用显式功能声明的数值回放、循环纠正不回灌长原文、缓存不重抽、重复杯子不被删除、原H/A/V与真实渲染、全空域区分、旧观测保留及可选基线。

落地灯回放保持5个对象、尺寸及布局输入不变：旧above声明仍被原求解器拒绝，显式illuminates声明可通过。它验证语义接口的重要性，不是自动把原世界重新解释为通过。

完整流程测试使用显式CPU模型回复替身，因此不能证明真实Qwen候选质量已改善。本次不运行GPU推理或API；实际五图依赖与初始化单独检查，记录见 [验证结果](reports/candidate_contract_repair_20261105/validation.json)。

```bash
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/caupano_world_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -m unittest qwen_edit_pano.world_pipeline.completion.candidate_repair.test_candidates -v
```
