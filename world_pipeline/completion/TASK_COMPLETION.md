# 回到 CauPano 的局部世界补全任务

2026-10-11。新代码已实现，GPU 训练和效果验证由用户手动执行。

## 1. 为什么旧路线开始偏离

旧 `complete.py` 验证的是“官方预训练模型能否续写 Qwen 预测框”，不是已经训练完成的 CauPano Completion。三张图的测试只能暴露接口问题，不能证明或否定可见条件下的补全任务。

官方 AR 配置启用 BERT 文本条件，采用自己的类别和统计量；其数据代码从排序后的场景随机取前缀。相机实际看到的对象集合一般不是这个前缀。把观测对象重排不能补上遗漏的训练任务。普通 AR 主干没有对象序号位置编码，推理时仅改变同一组 context 的排列也不产生可靠的补全能力。

旧桥接还假设相机离地 1.5 米，把相机坐标转为未标定的场景坐标，并涉及 OBB/AABB、全长/半长、官方细分类到 mpcat40 等转换。现有试验中关闭文本、改尺寸约定会改变输出，但这并不等于建立了正确的观察条件分布。

Qwen 预测框本身也有误差。旧室内客厅样本的 sofa/table 中心与 GT 约差 1.16/0.80 米；旧 lounge 还反复查询 chair/armchair/seating/furniture 等重叠类别。不能靠下游续写修复这些输入。旧结果有物体浮空、在已可见区域增加镜子等现象；本轮没有增加针对某张图的删除、贴地或缩放规则。

## 2. 当前实现究竟学习什么

```text
训练：现有 GT G_full
       ├─ local_evidence == visible → observed
       └─ local_evidence == unobserved → hidden

context = 全部 observed + hidden[:k]
target  = hidden[k]，或 k == len(hidden) 时的 END

推理：G_obs → 一次生成一个 hidden 对象 → END / 对象容量
输出：原始 G_obs 对象逐字段保留 + generated hidden
```

hidden 固定按相机局部 x、z、y、track_id 排序；track_id 仅用于相同坐标时打破排序平局，不进入模型。观测集合不被截成官方空间前缀。训练按场景均匀抽样，再均匀抽 k；验证枚举每个场景全部 k，先在场景内平均，再跨场景平均。验证是 teacher-forced 条件损失，不能替代 rollout 指标。

复用官方 `AutoregressiveTransformer`、类别头和位置/尺寸 DMLL heads，没有另外设计大型 Scene Transformer。具体适配如下：

| 项目 | 新任务实现 |
|---|---|
| 类别 | Adapter 的 37 个 mpcat40 类别 + START/END |
| 位置 | `center_local_m`，相机 x 右、y 上、z 前，米 |
| 尺寸 | `size_aabb_local_m`，当前相机轴 AABB 的完整边长 |
| 朝向 | 官方 API 保留零角度占位；不训练/预测语义 yaw |
| 文本 | 从任务训练开始 `text_condition=false`，不构造 BERT |
| room tensor | 全 1 的固定接口占位，不声称预测了房间边界 |
| 训练损失 | 类别 CE + 非 END 的位置/尺寸 DMLL；不回归 END 框 |
| 容量 | 从现有 Adapter Bundle 读取 160；容量终止与 END 分开报告 |
| 标准化 | 只拟合训练集，保存到 checkpoint；推理恢复为米 |

因此这是 **相机局部 AABB 对象状态补全**，不是官方原生 OBB 任务的无修改复现，也不是已经实现了 Room Geometry Completion。它与当前 Encoder 的实际需求一致。

`validity` 表示字段有值，不代表 Qwen 或 Completion 的估计一定准确；`padding mask` 在现有 `tensorize` 中生成。输出没有新增 Encoder 特征。

新任务的无文本设置是经过训练的条件设计，与在官方带文本 checkpoint 上临时关闭条件不同。用户可选文本仍可走现有 Qwen-Image-Edit 文本输入路径；这个 Completion v1 尚不响应用户文本中的结构干预。

## 3. 统一训练、推理的数值定义

本地官方 `autoregressive/losses/__init__.py` 的 DMLL 使用 `exp(log_scale)`，`autoregressive/base.py::sample_from_dmll` 使用 `ELU(raw)+1.0001`。新任务分支使用前者一致的采样，复用已有预测 head 参数。第三方源文件没有修改。这是一处代码定义差异，不能仅凭此断言历史权重所有异常的原因。

现有 GT 中含远距离、跨房间可在 ERP 中出现的对象。训练 center_x 的带余量范围约 100 米；官方 256 bins 对应约 0.39 米的损失积分宽度。因此新任务仍用官方 DMLL 实现，但使用 4096 bins，约 0.024 米。它是概率积分分辨率，不是声称达到该测量精度；head 结构不变。

尺寸下界为训练最小正尺寸的一半，上界为训练最大尺寸的 1.1 倍，采样逆变换后保持正值。位置边界取训练极值外扩 10%。这些是固定训练统计量，不随验证图调整。此次全部验证对象均在训练范围内。新域的框仍可能越界，模型对训练范围外的生成能力有限；不能把边界截断结果解释为真实房间尺寸。

## 4. 数据和实验隔离

实际读取 `data/oracle_supplement_epoch006_v1/samples.jsonl`：

- 1262 train / 168 val；16 个训练建筑、4 个验证建筑，建筑不重叠。
- 56731 个对象：16449 visible、40282 unobserved；所需几何字段均有效。
- 单场景最多 151 个对象，没有为了官方 121 容量而截断。
- 保留 Oracle 的对象集合、审批范围和 split；不是重新做逐样本视觉审批。

`data/hidden_completion_v1/` 已在本次完成 CPU 导出。没有下载新建筑或重新训练 Qwen。实际训练命令默认 **从头初始化 M3DLayout 架构**，以避免官方 Matterport 预训练建筑可能与当前 val 重叠导致解释混乱。

如果另做迁移实验，可给 `train_hidden` 加 `--init-official .../autoregressive_59000.pth`，在新的 output 目录运行。它只加载名称和形状兼容的参数，类别相关不兼容参数重新初始化，BERT/dependency 参数不加载，完整清单写入 `run.json`。它是初始化实验，仍需要训练。官方预训练数据与本验证建筑的重叠未核验，不能把该组叫作严格未见建筑泛化结果。

## 5. 执行顺序

全部命令从项目根目录执行，使用已安装的 qwen3vl 环境。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# 第一步：训练独立 Completion；不是 Encoder/Adapter 训练。
bash qwen_edit_pano/world_pipeline/completion/run_hidden.sh train
```

20,000 steps、batch 16、lr 1e-4 是首轮可复现训练预算，不是已验证的收敛最优值。每 1000 步及结束时验证，保存 `last.pt` 和最优验证损失的 `best.pt`。脚本单次申请 2 小时，实际是否足够取决于 GPU；若到时中断，执行：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_hidden.sh resume
```

恢复 optimizer、随机状态和步数，累计目标仍是 20,000 步。`HIDDEN_NODE=GPUx` 可指定空闲节点，不要占用正在训练 Adapter 的 GPU。若需要重做首轮，设置新的 `HIDDEN_RUN`；本轮 prepare 已完成，不需重复执行。

```bash
# 第二步：在全部 168 个 val 的 GT 可见对象上 rollout，固定 seed=0。
bash qwen_edit_pano/world_pipeline/completion/run_hidden.sh val

# 第三步：CPU 汇总，结果写入 outputs/hidden_completion_run001/val/evaluation.json。
bash qwen_edit_pano/world_pipeline/completion/run_hidden.sh evaluate
```

指标按建筑和 seed 报告：同类别 AABB IoU≥0.25 的匹配 precision/recall、隐藏数量 MAE、匹配对象中心误差、END 比率。匹配先最大化有效匹配数，再最大化 IoU。零预测时 precision 为 null，recall 仍惩罚遗漏；高 END 率不是质量证明。单张图有多个合理隐藏世界，单一 GT 的这些指标只衡量参考结构一致性，不能独自判断合理性。零补全基线 recall=0、count MAE=平均隐藏数量，可直接与报告中的 target/samples 比较。

不要按三张图挑 seed 或修改布局规则。小规模查看可用 `sample_hidden --val-data ... --per-building 3 --seeds 0 1 2 --output 新目录`，它确定性取每栋前 3 个，仅作诊断；正式指标仍报告全部验证集。

GT-observed 分支有学习证据后，再接 Qwen G_obs。已有样本可直接检查接口：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_hidden.sh predict \
  qwen_edit_pano/outputs/observer3d_indoor_val3_run001/X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c/G_obs.json
```

这是旧 Observer 输出的兼容性测试，不是新 Observer 的质量结果。`predict` 保存三个 seed，不选择最好的一张。需要重跑 Observer 时，使用原 `observe --inputs ... --model 本地snapshot --output 新目录`；当前已按 37 类统一查询，chair/armchair/dining_chair 不再分开查询。没有加入按图片定制的近似框过滤、坐标缩放或镜子删除。

## 6. 如何判断下一步

1. **GT-observed 补全不成立**：检查训练/验证损失、END 过早/容量终止、数量和几何指标。先解决 Completion，不调 Qwen prompt 掩盖问题。
2. **GT-observed 有效，但 Qwen-observed 失败**：这是 Observer 分布差异的证据。下一阶段用 2D 实例定位锚定 3D 查询、度量几何或真实预测误差训练；不要假设单图 VLM 米制框已足够准确。
3. **补全状态合理，但 ERP 不响应**：回到独立 Oracle 的 Correct/Constant/Shuffled 对照，不把责任直接归到 Completion。

当前对象条件不包含可见空白区域或遮挡信息，因此不能严格保证所有新增对象与原图可见区域一致；本版通过 GT visible/hidden 的训练定义学习统计规律，不具备投影遮挡的硬保证。后续仅在跨样本证据支持时增加观测条件，而不是对三张图的输出写修补规则。

输出 `G_full.json` 已与现有 `oracle_model.tensorize` 兼容，但这不代表已经验证完整 ERP 推理链。当前 Oracle 推理入口主要读取 GT Bundle；把预测图接入 ERP 采样入口仍需单独实现，并在 Oracle 成立后做同图同 seed 对照。本轮没有伪造已完成的端到端结果。

## 7. 核验来源

- [官方代码与训练/推理发布说明](https://github.com/Graphic-Kiliani/M3DLayout-code)
- [官方 AR 配置](https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml)
- 本地版本 `af67255521c7d8fd30a7b4eb83bd40f0f0d0742a` 的 AR dataset、transformer、head、DMLL loss 和 sampler；代码行为优先于旧交接文档。
- 当前数据以 `hidden_completion_v1/metadata.json` 和所引用的 Oracle manifest 为准。

14 项 CPU 测试通过，覆盖真实官方网络的前向/反向、padding 与单样本一致性、训练/推理 context 一致性、空观测、END 几何屏蔽、观测对象保留、导出 tensorize，以及 Observer 类别桥接。另外在临时目录完成训练/保存/续训/加载/rollout/评估的 CPU 集成检查，临时数据已清理；官方权重兼容初始化实测加载 223 个张量，4 个类别相关参数重新初始化，DMLL 尺度采样与逆 CDF 数值一致。GPU 训练、收敛和生成改善尚未验证。
