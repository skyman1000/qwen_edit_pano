# GT-visible run002 分析

本报告仅使用用户确认完成的 `data/m3dlayout_visible_probe_v2` 和 `outputs/m3dlayout_visible_probe_run002`。指标从该目录的原始布局重新计算，没有沿用 run001 统计。此次执行的是 CPU 结果审计，没有运行模型推理或更改模型行为。

## 结论

当前官方预训练 AR 权重在这批样本上不适合直接承担任意 GT 可见集合的缺失布局补全。几何桥接和实际条件编码没有发现相应错误，突出问题是立即 END，以及继续生成时空间恢复质量低。现象与空间前缀训练分布不适配相符；还不能由相关性认定唯一根因，也不能把这批结果推广成所有场景或 M3DLayout 全部方法的结论。

这不否定 CauPano 的显式 World State 路线，不影响独立 Encoder/Adapter Oracle 实验。它限制的是“直接复用这份官方 AR 权重免训练补全任意 G_obs”的具体方案。

## 运行和输入核查

- config 中 bundle 指向 v2，captions 指向 run002/captions。
- 实际 11 个房间，4 栋建筑分别为 2/3/3/3。11×2×3=66 个唯一结果与预期键集合完全一致。
- 从 raw_normalized.json 检查 START、context、生成 token、END 的实际位置和长度，全部与 summary 数量一致；18 个空续写确实是首个新增 token 为 END，不是输出解析丢失。
- 实际 context 与重新编码的 bundle 输入一致；所有条件归一化分量在范围内。文本与对应 caption 完全一致，描述未截断且对应正确图片。
- 生成几何没有非有限值、非正尺寸。代码运行完成和数值合法不能证明布局合理。
- 最大 source/native 角点误差 6.2538e-15 m，最大 stored Local center 误差 5.8411e-15 m。这个结论相对于现有 source/bridge 成立，不是独立确认原始 GT 相机和可见性绝对准确。
- 所有 context 均直接保留。由于 complete_scene 本身拼接 context，此项不是模型学会保持物体的证据。

## 总体结果

| 指标 | prefix | visible |
|---|---:|---:|
| 推理数 | 33 | 33 |
| END 正常结束 | 33 | 33 |
| 首步 END / 0 个新增对象 | 0 | 18 (54.5%) |
| 新增对象数 | 346 | 126 |
| 各组待补 GT 数（重复计入 seeds） | 336 | 336 |
| 同类数量交集 | 180 | 48 |
| 各组 GT 同类 OBB IoU≥0.25 一对一匹配 | 60 | 3 |
| 共同未见 GT 数（重复计入 seeds） | 195 | 195 |
| 共同未见 GT 空间匹配 | 38 | 3 |

各组剩余目标数量相同，但对象身份不同；共同未见集排除了两组输入的并集。共同目标参考匹配率为 19.5% 和 1.5%，不能当成多解合理性的完整评分。prefix 输入的信息内容也与 visible 不同，不是相同信息量的严格控制。

匹配方法为同类别、3D 旋转 OBB IoU≥0.25，先最大化有效匹配数再最大化 IoU；3D 交集使用旋转 xz 矩形交面积乘 y 交叠高度。新世界不必等于单一 GT，细薄对象和严格类别匹配也会影响结果。

## 建筑分组

| 建筑 | 房间数 | prefix 空续写 | visible 空续写 | prefix/visible 新增总数 |
|---|---:|---:|---:|---:|
| 8194nk5LbLH | 2 | 0/6 | 3/6 | 100 / 21 |
| X7HyMhZNoso | 3 | 0/9 | 3/9 | 52 / 29 |
| x8F5xyUWy9e | 3 | 0/9 | 3/9 | 148 / 76 |
| zsNo4HB9uLZ | 3 | 0/9 | 9/9 | 46 / 0 |

问题不是只出现在一个场景或某个 seed。另一方面，prefix 也不能称为整体可靠：例如 gym 的 prefix 新增 6/7/11 个对象，但与自身隐藏 GT 的空间匹配均为 0。

## 代表性案例

- X7HyMhZNoso_a59d092…：prefix 与各自 GT 剩余对象空间匹配 5/6/6；visible 仍生成 7/5/6 件，但匹配 0/1/0。seed0 虽生成 bed，位置显著改变，并出现额外 shelving、tv_monitor。仅看生成数量会误判。
- zsNo4HB9uLZ_ead481…：原生完整场景 21 件，可见 16 件，缺失 5 件；可见索引为 5..20，即排序尾部连续片段。visible 三个 seeds 都结束，prefix 都生成。此例直接体现“后段已知、前段缺失”与前缀续写的区别。共同未见集为 0，不能用共同目标空间指标评估此例；仍可评估停止行为和各自剩余目标。
- 8194nk5LbLH_ae915…：visible 输入 16 件，原生剩余 23 件，新增 7/8/6；并非完全不能补，但布局和隐藏家具恢复有限。不能把相对少生成直接等同于错误，因为 GT 隐藏内容并非唯一可能世界。

对照图同一场景使用相同坐标范围；reference 蓝色为真实可见对象，prefix 蓝色为不同的空间前缀，不能误认为两图固定的是同一组物体。

## 原因分层

### 1. 条件分布不适配：当前最有证据的解释

6 个场景全部 seeds 立即 END，其中 5 个条件含官方排序最后一个对象，另一个含倒数第二个对象。所有含最后一个对象的场景在本批次均直接 END。结果中未保存类别 logits，不能报出 P(END) 或声称确定的停止规则。

官方 dataset 将对象按离散化 x,z,y 排序，然后训练 `context=v[:n_boxes]`、`target=v[n_boxes]`。任意可见集合并不是这个训练前缀：它可能已包括空间尾部，却缺少前部与中部。

当前 autoregressive_transformer 路径没有对象序号位置编码；其 positional encoding 编码的是几何属性。相同集合倒序/打乱不应改变理论上的集合条件，因此不要把“重新排序”当作解决办法。实际数值误差可用首步 logits 排列测试确认。

类别先于几何采样产生。固定输入下首步 END 无法直接归因于后续几何 DMLL sampler 的尺度差异。调整几何采样或强制不许 END 都不能验证模型是否学会补内部缺失。

### 2. 对象与房间范围不一致

原 G_obs 共 131 件，native visible 仅 83 件（63.4%）。遗漏 48 件，其中 28 件跨主房间，其余 20 件是同房间未进入 native inventory 的对象。

两个 x8F5xyUWy9e 样本中，原始观测分别 14/11 件，模型各只有 1 件；其中跨主房间分别 12/6 件。描述涵盖 Local RGB 中通过门口看到的区域，但条件和生成目标仍是相机所属单房间。这是文本与对象范围的干扰因素，需要独立报告，不能把所有失败都归因于 AR。

最终 ERP 世界不能静默删掉跨房间观测。后续应保留完整 G_obs，同时将房间级生成模块的对象范围、文本范围和相机覆盖范围对齐；这属于系统设计问题，不能通过几句 prompt 消除。

### 3. 评测与泛化边界

8 个房间来自官方 mp3d_train，3 个来自 mp3d_test。官方权重精确训练版本尚未确认，但这已经不能作为独立跨建筑泛化评测。prefix 在某些场景高度复原 GT 也不能直接解读为泛化能力。

样本按可配准、不同房间等规则选择，属于诊断子集；不能替换原 val split 或删掉失败房间。当前 GT-hidden 是官方 room inventory 的剩余部分，并非 CauPano ERP-visible G_hidden 的完全等价集合。

## 下一步建议

1. 固定 run002 为本轮有效记录，停止重复 check/prepare/66 次生成来调 seed。保留已验证的尺寸和坐标转换；不再无证据改 half/full 或匹配阈值。
2. 若继续定位，只补一次轻量首步概率诊断：每场景 prefix/visible 的 P(END)、top 类别，以及同集合重排的 logits 差。若检验排序尾部影响，应标注 GT 干预诊断并保持数量控制，不能把删除可见对象作为正式算法。该诊断入口尚未实现。
3. 用户当前不训练 Completion，不能把修复希望寄托在未执行的微调上。现有官方 AR 任意集合补全暂不作为最终主链可靠模块；纯文本生成可保留为对照，独立布局不能未经配准直接拼接 G_obs。
4. 若必须继续免训练 full completion，候选方法必须实际支持任意已知对象集合和同坐标约束，仍先通过 GT-visible 测试。仅有 text-to-layout / prefix completion 接口不够；不承诺更换某个模型即可解决。
5. Oracle Encoder/Adapter 训练与本结果独立。继续判定 Full Correct 是否优于 Constant/Shuffled。Observer 的可靠可见实例、尺度与坐标仍有价值，但不能作为修复本轮 GT 可见集合失败的手段。
6. 若未来允许训练，适配方向才是全部 observed 常驻 context、仅逐步生成 hidden，并学习相应 END。这属于后续实验，不在当前“不训练”约束下擅自执行。

## 可追溯产物

位于 `outputs/m3dlayout_visible_probe_run002/comparison/`：

- `analysis_audit.json`：本轮 bundle 路径/指纹、66 个 raw 输出指纹、实际 END 与数量核查、建筑统计。
- `analysis_metrics.json`：本轮重新计算的逐场景/seed 类别与空间匹配。
- `analyze_saved_results.py`：CPU 指标与图表生成脚本，不加载模型。
- `analysis_scene00.png`、`analysis_scene02.png`、`analysis_scene08.png`：厨房与两个卧室的对照。
