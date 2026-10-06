# 首轮GPU smoke失败分析与修复

来源：用户附件86889180-be4c-4906-8356-9045d3d36323/已粘贴的文本.txt。不归因于数据或qwen3vl环境：日志Python路径为qwen360，7条目标posterior编码成功，固定Edit/LoRA加载成功，base_trainable=0，world trainable=5211395。

1. GPU2：第一次backward checkpoint recompute时OOM。设备44.42GiB，本进程44.33GiB，PyTorch allocated43.58GiB/reserved未用250.77MiB；申请146MiB、free71.88MiB。主要是实际内存不足，不能仅靠empty_cache或碎片环境变量宣称修复。
2. GPU4：第一次backward的CheckpointError，saved/recomputed元数据不一致。实际CPU BF16小型真实Edit+冻结LoRA重现：同一autocast上下文内先no_grad参考forward，再训练forward；cache_enabled=True触发同类metadata错误，False通过。原FP32 CPU测试未覆盖此组合，是测试缺口。

修复：oracle_memory.training_autocast禁用权重cast缓存；训练入口拆开无梯度smoke与训练上下文。保持checkpoint determinism检查、原attention/层数/分辨率/损失/GT不变。不采用禁用元数据检查、强制no_grad整个生成模型或修改字段来掩盖问题。

内存措施：ActivationStorage用saved_tensors_hooks，仅将CUDA、requires_grad且非叶的保存激活同步复制到CPU；冻结权重和叶参数保持原处。反向恢复相同dtype，不量化；默认cpu，可选gpu给充足显存条件。CPU回归测试验证hook保存/恢复数值与梯度，不能证明实际CUDA峰值，需重跑日志确认。增加memory.jsonl的allocated/reserved/free/peak及保存激活复制量；copy_gib为累计复制量，不等于峰值节省。

7项CPU测试通过，包括新BF16 no-grad→train→checkpoint backward与非checkpoint输出/梯度逐元素一致、CPU hooks不改变梯度；原target/reference边界、zero gate、冻结、置换与保存重载测试保留。语法检查通过，原paired训练source hashes无差异。没有执行GPU训练、推理或下载。

保留oracle_full_smoke_v1失败产物，不resume无COMPLETE的目录；新运行用oracle_full_smoke_v2。无需重建oracle_pilot_v1。修复后GPU是否成功尚未实测；先2步smoke成功再重载，不进入200步或全量训练。
