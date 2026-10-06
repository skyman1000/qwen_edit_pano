# v3 A/B结果检查

读取greedy/checkpoint两组原始输出、status和generation，查看checkpoint两张boxes.jpg，并用原有IoU0.5+类别一致评测器及辅助视觉初审文件运行CPU评分。没有执行GPU任务，没有改变模型参数、提示词、GT、阈值。

- greedy：客厅截断，浴室解析成功，格式通过1/2。
- checkpoint：客厅1174 token、30对象；浴室570 token、14对象；均完整结束，格式通过2/2。仅两个train样本、seed0，不代表稳定性或泛化已验证。
- checkpoint严格评分：客厅TP8/FP22/FN12，浴室TP1/FP13/FN3；合计precision20.45%、recall37.50%、F1 26.47%。输出位于qwen_edit_pano/outputs/observer_ab_v3_checkpoint_metrics/metrics.json。这是现有GT协议下的诊断，不是AP；FP表示无法匹配，不全等于幻觉。

视觉观察：四张画、部分窗、前景植物、茶几、洗手盆框基本对应；黑色投影设备被归为tv_monitor、墙面大片区域被归为board_panel等需要复核。密集灯具和浴室小物品超出当前GT可见实例集合，不能仅凭未匹配判定不存在。

字段差异：客厅GT76=bed，预测sofa，框IoU0.961；GT85=curtain，预测furniture，框IoU0.882；GT50=objects，预测cabinet，框IoU0.791。浴室GT159=clothes，预测towel；GT169灯框与预测灯框IoU0.305，GT163浴缸只覆盖局部mesh区域。差异包含类别、实例粒度、mesh覆盖和真实预测错误；尚未完成原始对象语义/分割资产逐一复核，不应擅自修改GT或等价类别映射。

此前7条approved只表示相机/投影辅助初审，不表示类别与可见框达到完整人工检测标注标准。保留原分数与GT不变。

下一步冻结baseline_v1/scale1000/checkpoint/repetition1.0/768/4096，先全11条seed0，再同2条seed1检查输出稳定性。所有GPU任务由用户srun。当前严格评分仍只接受7条train，其余4条（包含全部val）仍受numerical_flags阻挡；不能宣称验证集成绩。若扩展出现重复或截断，保留失败，先统计比例；不要自动重采样到成功或筛选最好seed。若采样稳定但类别粒度问题持续，再单独设计保留原GT的版本化评测映射/忽略规则；不以这两张临时调规则。

暂无必要继续修改推理逻辑。Observer尚不可作为正式可靠结构标签；3D字段继续未知。Completion训练仍后置，GT G_full oracle路线可独立准备。
