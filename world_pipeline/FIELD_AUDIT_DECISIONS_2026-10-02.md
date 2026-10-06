# 字段审计结论与下一实现边界

已读取用户research/audits/field_contract_v1全部汇总与对象候选。新增trace_field_sources.py并实际CPU执行，产物research/audits/field_contract_source_trace_v1。两图24个GT可见对象全部通过类别TSV与既有exact_segment_set链接一致性检查，并回读原始house_segmentations.zip中的O/C记录核对ID引用。未重新计算segment集合；此结论不认证源标注视觉正确、mesh完整或相机物理精度。

明确证据：GT76源bed→bed，GT85源screen→curtain，GT50源heater→objects，GT51源wall frame→picture；浴室GT159源bathrobe→clothes，GT163源bathtub→bathtub，GT169源lamp→lighting，GT174源sink→sink。不能因预测sofa/furniture/cabinet/towel而改原GT，亦不能全局合并这些类别。bed与sofa的具体视觉歧义保留，不宣称观察者或源标签必定错误。

对象数24与44只对应两图、不是一对一匹配。现有8+1严格匹配不变。二维可见框来自重建mesh与RGB识别两种来源，灯具和浴缸框差异尚不能唯一归因于mesh缺失，需保留投影/遮挡/粒度等可能性。二图结果不足以确立可靠字段准入阈值。

## 可进入实现的决定

- 固定已有mpcat40词表哈希和类别ID含义；保留source raw label为离线元数据，未知可null。暂不合类、不重标。
- Observer继续当前baseline_v1/checkpoint/scale1000配置，不要求预测三维；解析后的框仍为0–1。room候选保持独立，不输入首版Encoder。
- 物体状态候选核心：category_id、center_local_m、size_aabb_local_m及字段有效性掩码。Local二维框是观测证据/Completion输入，不与三维AABB或完整物体投影框混用。
- Completion接口必须覆盖可见对象三维属性估计+镜头外补全。GT_obs中真实完整3D不可当部署匹配输入。GT oracle和未来预测full状态共享tensorizer；前者明确是oracle条件。
- 下一代码工作为独立tensorizer与CPU契约测试，不是Observer生成更多字段。分别验证完整GT输入、删除3D的观测输入、空对象输入、对象顺序变化、数量上限不静默截断、source/ID不进入特征、flip变换与缺失mask。

## 仍需解决，不能宣称已正式冻结

- 当前4个样本Local/ERP像素门槛flag：新版本需显式定义对象集合取符合Local或ERP门槛的并集；把仅分辨率门槛差异与无ERP命中/坐标异常分开。现有导出已经保留并集，flag不能直接删除；用新版本转换/导出并验收，保留旧目录及规则。
- 扩展已有GT覆盖、完成独立val检查；不能用当前11条宣称oracle泛化，也不必等待所有标注绝对完美才实现接口。发现问题按字段/样本记录。
- Edit专用target-only Adapter尚未实现。接口CPU测试之后才能给真实GPU smoke命令；正式oracle固定正式训练checkpoint，与同容量常量分支/GT_obs/GT_full对照，保留原生Local条件。扰乱world只是补充诊断。

现在无需用户再跑Observer或重复字段审计。优先实现tensorizer/可见性版本修订，然后Edit分支接口；需要用户执行时再提供已有入口的srun命令。保持正式训练、旧数据、cache和outputs不变。
