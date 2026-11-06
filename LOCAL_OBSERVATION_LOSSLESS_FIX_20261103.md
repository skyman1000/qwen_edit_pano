# 局部观察器修复：合法记录保留、按图隔离失败

2026-11-03。本次针对 `local_observation_restored_001` 的已确认根因修改代码，不更换模型或检测框架。执行仍是本地Qwen3-VL二维实例 → 官方WildDet3D对应三维框 → 原词汇表类别；不调用推理API、不训练、不启动世界补全。

## 1. 修复边界

上次实际运行四张图完成二维，卧室首轮10条有效字段记录被“每块最多8条”整份拒绝；修复提示随后引导模型返回流程状态，最终卧室失败。外层shell将整批二维非零视为不允许运行任何三维，导致另外四图也没有WildDet3D结果。完整根因见[上次取证](reports/local_observation_restored_001_root_cause_20261103/analysis.md)。

本次保留原首轮图像、提示、解码配置、stage seed、二维坐标系、WildDet3D检查点和词汇表映射。

### 合法记录不因逻辑页大小作废

[assembly_io_recovery.py](world_pipeline/completion/assembly_io_recovery.py)将8条区分为请求的分页大小，64条仍为总计算上限。所有记录依然必须通过原字段、框范围、完成状态和重复检查。

- 一次返回10条合法且唯一记录时，完整保留，`transport_audit.json`记录8＋2逻辑页及原始对象索引；不重新识别、不截断、不补齐数量。
- 8、9、10、16、64等数量使用同一规则，没有针对卧室或“10”的分支。
- 重复记录仍走原重复调和/身份处理，超8条不能绕过身份检查。未新增NMS自动删除，也没有把重叠直接当作同一物体。
- 超过总预算64条直接保留原回复并明确失败，不要求模型删到64条，也不偷偷截取前64条。
- 普通字段错误的修复提示只要求修正该错误，不再把“尚未接受任何对象”的流程状态混入视觉任务。非空提案后的矛盾空清单保护继续保留。

### 下一阶段按单图依赖决定

[local_observation.sh](scripts/local_observation.sh)在二维阶段部分失败后仍执行三维阶段；[assembly_multiscene.py](world_pipeline/completion/assembly_multiscene.py)逐图检查：

```text
G_2d存在 + 二维summary=completed → 该图可进入WildDet3D
缺G_2d/缺summary/二维失败       → 该图blocked_dependency，保留原因
```

即使失败图遗留了G_2d文件，也不能凭文件存在误启动三维。全部图均阻塞时不加载任何推理模型。失败和已完成阶段保留，不通过删除标记或随机重试追求成功。

整体退出码继续反映失败：部分图失败可返回1，但其它有效图的三维与全部报告仍保留。这样既不伪报成功，也不让一图取消其它图。

### 报告与执行状态保持一致

[local_observation.py](world_pipeline/completion/local_observation.py)明确显示二维失败、三维失败和上游依赖阻塞及原因，不再将失败卧室显示为无错误的not_observed。失败阶段遗留的三维文件不会发布为已完成观测；缺三维仍保持未知，不填零、不制造三维框。

## 2. 调研采用的原则及边界

| 原始资料 | 相关方法/事实 | 本次借鉴 |
|---|---|---|
| [DETR：End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | 将检测表述为实例集合预测，以匹配约束区分实例 | 区分实例集合语义与文本分页；不把返回条数当视觉有效性。本次没有引入DETR或复现其训练算法 |
| [POPE：Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/) | 对象幻觉与视觉指令有关，需另行评估存在性 | 字段通过和模型自报完成不能作为准确率；保留质量未认证标记，不靠增补数量证明完整。本次没有声称跑过POPE基准 |
| [WildDet3D官方推理文档](https://github.com/allenai/WildDet3D/blob/main/docs/INFERENCE.md) | geometric模式逐提示框返回一个三维预测 | 三维阶段严格依赖对应图的有效二维结果，保留ID/框/类别对应；不将三维返回数量当存在性验证 |
| [Qwen3-VL-8B-Instruct官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 同一模型可执行图文与定位任务 | 继续使用部署的本地检查点和原处理器，不换模型，不以新的解码参数试图掩盖协议错误 |

这些资料支持的是实例语义、条件输入与评价边界。条数分页和逐图容错是根据本项目失败证据实现的工程修复，不宣称其本身带来论文级检测准确率提升。

## 3. 实际验证

- 部署的qwen3vl环境中完成 **99项CPU检查**，覆盖观察器、重复处理、调度、失败报告、原世界流程与多场景共享代码。
- 逐字段回放上次实际五图保存请求/回复：二维数量为 **16 / 8 / 8 / 10 / 3**。四份原已完成二维状态保持相同；卧室直接保留首轮10条，分页8＋2，不调用其第二轮空回复。
- 执行了混合状态调度检查：有效图继续三维，失败图即使有遗留二维文件也阻塞，缺失图阻塞；再次进入不重复执行已记录阶段。全部阻塞时不初始化模型。
- 覆盖总预算、跨块累计预算、重复身份、非法框、空场景、字段修复、失败阶段遗留三维图等边界。
- shell检查覆盖local/run两种模式，二维/三维失败后仍生成报告，整体失败码保留；未启动因果/隐藏世界阶段。

逐项结果、源码哈希、未修改的模型/映射模块、预检目录及CPU回放来源见[验证记录](reports/observation_lossless_validation_20261103.json)。

[CPU回放框图](outputs/local_observation_lossless_cpu_replay_20261103_001_review/index.html)明确标注未进行新Qwen/WildDet3D推理。它只验证数据传输和导出，不是新GPU结果。原始失败运行及其回复未被覆盖。

## 4. 用户执行

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
CAUPANO_TEST_INPUTS=qwen_edit_pano/data/observer3d_indoor_val5_inputs.jsonl \
srun -p debug --nodelist=GPU3 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
  bash qwen_edit_pano/scripts/local_observation.sh local \
  qwen_edit_pano/outputs/local_observation_fixed_002
```

002必须不存在；已用过则换一个新名字。不要用新源码resume旧001，源码绑定会拒绝。脚本不会嵌套提交srun，不需要API Key或新依赖。

结束打印 `local_observation_fixed_002_review_时间_PID/index.html`，含二维框、原图、对应三维框、原始/映射类别、重投影、平面/六方向、ERP状态图和交互3D。ERP仍是状态投影，不是生成RGB全景。

可运行 `bash qwen_edit_pano/scripts/local_observation.sh test` 检查核心回归；GPU推理由用户执行，本次未运行。

## 5. 未解决的检测质量问题

厨房只识别吧凳的旧漏检以及部分on_floor语义错误仍存在。本次保持原检测提案，未用额外全图生成覆盖它，也未按常见厨房类别人工补物体。二维/三维完成只表示该阶段运行并导出，不能视为完整性与物理真实性验收。

应先用本次实际三维结果核对观测链路，再单独评价厨房的提案覆盖和实例身份。不能将继续增加对象、扩大token预算或反复重跑作为这两处工程修复的替代。
