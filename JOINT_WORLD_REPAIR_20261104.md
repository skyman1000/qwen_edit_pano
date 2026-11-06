# 全局世界补全修复与执行（2026-11-04）

**本页为上一轮修复记录，已被实际结果修正。** `joint_world_repaired_001`实测16份空增量、2份布局失败、2份未解决提案，零份含新增导出。当前根因、修复代码和新srun命令见 [JOINT_WORLD_REPAIRED_002_20261104.md](JOINT_WORLD_REPAIRED_002_20261104.md)，不要直接续跑本页旧目录。

本次目标仍是单张局部室内 RGB → 保留 G_obs → 提出多个合理的周围世界假设 → 输出 G_full = G_obs ∪ ΔG。未知世界允许多解；数量、方向覆盖、成功率均不能替代语义和空间合理性。只使用部署的本地 Qwen3-VL，不调用 API，不重新估计观察器。本次完成源码修复、真实回复离线回放与 CPU 集成验证；尚未执行新版本的真实 Qwen 推理。

## 为什么 v13 能补全，joint_world_002 却全部失败

同一批五图的 RGB、45 个 G_obs 实例、本地模型路径和采样配置没有变化；原 v13 的 116 个核心源码绑定仍全部匹配。v13 的真实结果是 20 个世界：17 个含新增导出、2 个合法空增量、1 个布局拒绝，不能把不同世界的对象相加当成一个世界。

|证据|根因|修复|
|---|---|---|
|joint_world_001 五图首次请求之前均失败|调用方未创建 calls 目录；旧测试替身代建目录掩盖了生产问题|调用方创建目录；集成测试走真实 MatchedLocalJSON 请求/缓存/日志代码，只替换数值推理引擎|
|002 的 12 份最终提案全被拒绝；9 份首先撞到 root_host_scope|新流程只允许根对象引用观测宿主，但模型提出新沙发→靠垫、新柜→用品这样的完整装配，违反人为的阶段边界|恢复 v13 完整增量的 new_objects 接口；同一替代程序允许同时定义新宿主及其依赖对象|
|其余首先报错为 2 份重复观测 ID、1 份观测支撑域不相容；更多错误被第一层门禁遮住|新提示把可见清单误导成待生成清单；修复提示还冻结了非法候选身份|固定观测只能出现在 observed_refs、宿主或关系引用中；语义修复允许重新提出候选，禁止把可见对象换 ID 冒充新增|
|卧室在 H 阶段因为 ["3.35","3.35"] 未进入提案|将位置、地面和区域数值当作全局语义假设的先决条件|H 只包含描述、功能、假设；尺寸、地面和位置都在完整库存确定后推断|
|另外四图的合法程序为零，仍执行 16 次 recognition 并进入 worlds|“生成失败”被编译成空域，阶段却标完成，掩盖最早阻断点|全部提案失败时阻断机制估计和世界采样，且在加载模型前检查；失败不等于合法空增量|

详细旧证据见 [002 根因报告](reports/joint_world_002_root_cause_20261104/README.md) 与 [001 日志目录故障](reports/joint_world_001_calls_failure_20261104/README.md)。仅移除 root_host_scope 并不够：对 002 的 12 份最终回复仅做容器适配、保持身份和宿主不变后，只有 3 份通过完整增量校验，9 份仍有真实错误。本次没有把这些错误回复自动改成“成功”。

## 修复后的执行逻辑

```text
导入并逐文件校验冻结观测（Qwen 2D → 官方 WildDet3D 3D → 词汇映射）
→ Qwen 提出纯语义 H（description/functions/assumptions）
→ 每份 H 提出完整共存程序 F（observed_refs + new_objects + relations）
→ v13 验证身份、新/旧宿主、循环依赖、观测支撑域和关系
→ 估计有限选择 q(H|RGB,G_obs)、p(F|H,RGB,G_obs)，编译宿主依赖 SCM
→ 固定次数 IID 抽样，保留重复、合法空增量和未解决失败的概率质量
→ 对整份已冻结库存推断尺寸，再提出明确的地面假设和连续区域原点
→ 分批推断根/依赖对象姿态，每批均携带全世界和已接受的姿态
→ 原有求解器检查视锥、碰撞、支撑、地面与声明的空间关系
→ 导出 G_full/G_out、依赖、干预检查、失败记录及完整可视化
```

这里保留同一 H 下完整程序的共存关系，不再按功能角色拆池。两个 reading 区域可以同时存在；不设置最少区域数、最少对象数或必出类别。完整程序沿用 v13 的最多 4 个语义组、24 个新增实例的传输容量，均不是生成目标。依赖重组可能产生额外的观测宿主组。

阶段修复最多一次，且边界明确：语义阶段允许重新提出合法候选；尺寸阶段允许纠正尺寸；位置阶段冻结身份、宿主和已接受尺寸。纯数字字符串仅在明确的数值字段中无损转为数值，记录原值与路径；不猜单位、不改 ID、不丢弃对象。合法替代程序不会因另一个替代程序非法而整份丢失，非法替代仍保留为失败状态。

地面不再是 H 的门槛。需要 floor 物体时，Qwen 选择确实接地的观测实例，程序计算其预测框底部中位数；没有可靠锚点时可以明确提出未测量的负 camera-Y 地面假设，或声明未知。未知地面不能认证 floor 物体。相机 Y 近似竖直、地面高度等均是假设，不能当作实测。

## 修改位置与验证

仅修改/新增 `world_pipeline/completion/joint_world*.py`、对应测试和本说明。原 `assembly_*.py`、共享几何/SCM/观察器核心、`organized_local_world.py` 与原 v13 脚本均未改变；旧运行目录未重写。

- [执行入口](world_pipeline/completion/joint_world.py)：完整增量提案、阶段依赖检查、延迟加载本地模型、尺寸后置和可视化报告。
- [语义与 SCM](world_pipeline/completion/joint_world_model.py)：复用 v13 完整增量验证器，保留完整程序、合法空状态及失败状态。
- [分阶段请求](world_pipeline/completion/joint_world_requests.py)：请求目录、语义/尺寸/姿态分别修复、数值表示审计。
- [几何](world_pipeline/completion/joint_world_geometry.py)：显式地面假设、连续原点、复用原坐标容器解析、固定库存与尺寸的求解。
- [真实回复回放](reports/joint_world_repair_20261104/replay.json)：全部 26 份 v13 原始/修复提案中，16 份含合法替代、10 份仍拒绝；五图均有通过回复，合法对象/宿主/关系与原验证器逐项一致。另回放了全部五图的 H。
- [测试记录](reports/joint_world_repair_20261104/tests.log)：25 项通过。真实客户端搭配 CPU 替代引擎走通两份世界的完整流程，16 个冻结观测逐项保留，每份新增 4 个，两个同功能区域及宿主依赖保留，完整生成交互 3D 等现有视图。数字和合成回复只用于验证代码，不能当作 Qwen 质量证据。
- [实际 prepare](reports/joint_world_repair_20261104/preflight.log)：五图、45 个冻结观测、每图 4 个世界，通过实际本地文件/环境检查；没有模型加载或推理。2712 个输入 JSON 的回放前后哈希一致。

复核 CPU 测试：

```bash
env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/caupano_world_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python -m unittest \
  qwen_edit_pano.world_pipeline.completion.test_joint_world \
  qwen_edit_pano.world_pipeline.completion.test_joint_world_contracts \
  qwen_edit_pano.world_pipeline.completion.test_organized_local_world -v
```

## 手动执行完整补全与可视化

直接执行下面的完整流水线。新目录 `joint_world_repaired_001` 在交付时尚不存在；不要 resume 旧 001/002，也不要删除它们的失败记录或改写源码哈希。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
export CAUPANO_WORLDS=4
export CAUPANO_DRAWS=4
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=04:00:00 \
  bash qwen_edit_pano/scripts/joint_world.sh run \
  qwen_edit_pano/outputs/joint_world_repaired_001 \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

结果为 `qwen_edit_pano/outputs/joint_world_repaired_001/index.html`，包含原图、固定观测、原视角投影、俯视图、六方向、ERP 框图和交互 3D；每份世界独立查看。固定观测面板明确标为“不是隐藏补全”。原视角的新增框应不进入观测视锥；检查新增主要看俯视、六方向和交互 3D。灰色 ERP 区没有生成 RGB。

脚本自动完成 propose → mechanisms → worlds → report，不是只跑诊断或观测。总状态部分失败时退出 1，仍会生成报告；查看 `results.json` 区分：`exported_with_new_objects`、`observed_only_sample`、`rejected_layout_candidate`、`layout_failed`、`blocked_proposal`、`blocked_proposal_domain`、`blocked_dependency`。未运行不是合法空世界；拒绝候选为 `G_full_candidates.json`，不冒充 `G_full.json`。

每个场景主要文件在 `scenes/SCENE/joint_completion/`：`calls/` 原始请求/回复、`domain.json`、`empirical_tables.json`、`causal_model.json`、`structural_samples.json`；每份 `world_01/` 保存库存、尺寸、地面、连续布局、数值检查、干预记录和导出状态。

中断后，使用相同 srun 参数，把末尾改为下面这一行；仅适用于本次新目录且源码未变，不会重抽失败世界：

```bash
bash qwen_edit_pano/scripts/joint_world.sh resume qwen_edit_pano/outputs/joint_world_repaired_001
```

仅重建报告可在 CPU 执行：

```bash
bash qwen_edit_pano/scripts/joint_world.sh report qwen_edit_pano/outputs/joint_world_repaired_001
```

## 查证相关工作后的取舍

核对论文原文/官方摘要时间为 2026-11-04；这里只借鉴组织方式，没有声称复现这些系统或达到论文效果。

|工作|论文支持的组织方式|本项目落实|
|---|---|---|
|[HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation](https://arxiv.org/abs/2503.16848)|通过跨尺度支撑层级组织家具、表面与小物体|保留宿主—用品结构；这不推出“语义提案时禁止新宿主”的规则，完整增量可以先定义整个层级|
|[Holodeck](https://arxiv.org/abs/2312.09067)|语言模型产生物体间空间关系约束，再优化布局|语义关系在程序中明确保存，固定库存/尺寸后由求解器落实和检查|
|[Infinigen Indoors](https://arxiv.org/abs/2406.11824)|组合约束语言与约束求解|将身份、支撑、空间关系和几何可行性分别校验，不靠删物体提高导出率|
|[Language Models as Causal Effect Generators / SD-SCM](https://aclanthology.org/2025.emnlp-main.107/)|用户定义因果结构，语言模型定义机制，执行模型内采样、干预和反事实|明确 H→F→存在 E、宿主控制安装 S 的设定；报告干预范围，不把模型内依赖冒充从单图识别出的现实因果|

## 尚不能宣称解决的部分

新版本真实 Qwen 的合理性和导出率需此次运行后判断。当前是有限候选的假设 SCM，选择频率未校准，不是已识别的真实房间因果结构或完整后验；固定次数抽样可能重复或取空世界。没有 RGB 渲染似然、完整遮挡推理、网格物理和通行验证；所有新增目前需完全处于原视锥外。wall/ceiling 缺少显式建筑面时会拒绝认证，而不是悄悄当成悬空装饰。

厨房 G_obs 仅三个吧凳，原图的中岛等仍漏检。本次保留同一观测以定位补全回归；禁止将明显可见的漏检实例挪到视野外冒充新增。后续若改观察器，应独立建立新观测版本，不能把观测变化与此次补全修复混在一起。

下一次先看 H 是否与原图相容、ΔG 是否真实新增、宿主/数量/功能组合是否合理，再看连续布局和拒绝原因。只有候选支持和语义合理性成立后，才适合做固定库存的布局对照、H 干预响应和新输入评估；不依据这五图继续定制必出类别或凑方向。
