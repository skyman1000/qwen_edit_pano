# 两轮 Observer 失败分析与有限对照

读取实际v1、2samples_v2的raw/status/config。v1客厅重复不完整lighting对象至截断，浴室完整JSON但0–1000坐标不符合0–1契约。v2两图均4096 token截断：客厅在单个对象中重复bbox2d，浴室重复category_id/bbox2d且坐标递增至数千。第二轮未改善，不据此宣称BF16、模型权重或图像识别失效。

第二轮同时改了提示词、尺度和repetition_penalty，不能分离因果。重复惩罚针对token，JSON字段、标点、类别本就需要重复；该设置可能干扰结构，但当前两图对照不足以确认唯一原因。停止推荐compact_v2+1.05，不继续增加惩罚或token上限。

CPU实检：在用户qwen3vl环境，首图768尺寸与v2提示词经过当前分步processor调用，以及官方apply_chat_template(tokenize=True)路径，input_ids、attention_mask、pixel_values、image_grid_thw完全相同。未加载GPU模型。排除了该样本这两条预处理路径差异；不代表穷尽所有运行时原因。模型本地generation_config为采样true、temperature0.7、top_p0.8、top_k20、repetition_penalty1.0；旧入口强制greedy。

本轮只增加可回退诊断开关：baseline_v1恢复首轮提示词原有结构，bbox_scale显式1000；decoding=greedy/checkpoint；惩罚均1.0，其他设置一致，每图seed重置0。checkpoint读取固定snapshot配置中的采样字段，并记录实际参数；不声称实现官方文档中额外的presence_penalty。保留旧运行及compact_v2用于追溯。

解析器修复：拒绝重复JSON键，避免Python json.loads静默覆盖实例。继续拒绝截断、非法坐标、缺失字段；不抢救前缀、不剪裁、不补括号。记录格式通过与准确率的区别。

用户手动执行A/B：同2图，BF16/SDPA、768、4096、seed0、bbox-scale1000、prompt-version baseline_v1、repetition-penalty1.0；A decoding greedy，B decoding checkpoint。两个新目录，不覆盖结果。A/B只比较解码方式；与v2的差异不能归因于单一因素。

决策：先比较自然结束、JSON完整性/重复键、框质量及遗漏。若一个候选两图通过，检查boxes后在11图检验，采样配置还需多seed稳定性检查，不挑选最好一次当成绩。若两者仍退化，停止整图全类别单次JSON路线的参数搜索，先用简短单类别定位探针验证能力，再评估自然语言标签→确定性类别映射、按类/分步定位或受约束解码；这些是后备方案，当前未实现，不直接投入正式GT批量标注。结构约束仅解决格式，不能证明检测正确。
