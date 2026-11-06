# 室内三图结果分析（2026-10-11）

检查：原图、Observer JSON、M3D raw token/布局/G_full、现有Matterport GT，以及官方Qwen格式说明和M3D训练代码。未运行GPU，未修改推理行为。

| 场景 | Observer | M3D context | 新增 | 最终 | END |
|---|---:|---:|---:|---:|---|
|休息区 8194nk5|45|35|0|45|是|
|客厅 X7HyMh|2|2|4|6|是|
|浴室 x8F5xy|7|7|3|10|是|

三份G_full通过现有tensorize(max_objects=160)，原始观测对象逐字段保留，新对象无导出跳过。Observer全部completed只表示解析完成。全部END，增加max-new-objects不会改变这些已提前结束的seed0轨迹。

## Observer主要证据

休息区：4 armchair、4 chair、4 seating、8 furniture；其中多个中心相同或相近、尺寸略有差异。tv和tv_monitor中心完全相同，尺寸不同；tv_stand与cabinet也有同位置重复描述。仅精确去重不足，且部分泛类映射成不同mpcat40 ID，会给下游传递互相重复的对象。45不是45个可靠独立实例。8个balloon恰好达到查询上限，completed也不能证明该类穷尽。

客厅：识别沙发和桌子，但漏掉清楚可见的靠垫、花卉、书等；不应把“两类主要家具对”当成完整观察。

浴室：洗手台、镜子等有输出，未充分覆盖左前方家具/右前方大型实体。右前方GT标为bed；此前选图说明中的“马桶”不准确，不应继续当真值使用。类别语义存在数据集标注粒度差别，不能把GT对象数直接用作预测召回分母。

## 对已有GT的几何诊断

仅选每张图中预测查询类别和GT对应mpcat40类别各有一个的对象，人工结合图像判读，非全量匹配评测。中心距离为当前已导出的local坐标下欧氏误差；GT自身仍受既有构建/标注质量影响。

|场景/物体|预测z米|GT z米|中心距离米|
|---|---:|---:|---:|
|休息区 coffee_table|4.260|2.474|1.789|
|休息区 tv|5.300|3.836|1.467|
|客厅 sofa|3.730|2.578|1.163|
|客厅 table|1.720|0.951|0.803|
|浴室 standing_sink|2.140|1.501|0.798|
|浴室 mirror|2.340|1.609|0.756|

六个例子均表现为z偏大，但各轴与物体的比例并不相同，不能据此固定乘0.6或0.7，更不能用验证GT拟合后声称纯RGB推理改善。

使用清单hfov=90与实际768方图进行中心回投影。客厅table中心约(384,612)，落在沙发区域；GT完整桌体中心可能在可见图下方，因桌子被画面截断，不应要求每个3D中心都落在可见2D框内。该图投影仍显示当前米制几何不可靠。回投影使用已知图像相机，不套官方demo默认60度。中心示意图不是完整OBB投影或严格检测指标。

证据文件：outputs/completion_indoor_val3_analysis/*_projected_centers.jpg，unique_category_center_comparison.json。

## Completion主要证据与限制

客厅新增large_plant_container、kitchen_cabinet、tv_monitor、tv_stand。花盆xz投影落在沙发范围，可能是遮挡/支撑关系，也可能是冲突，仅俯视AABB无法判定穿模；电视和电视柜中心z相差约2.51米，不能当作已形成合理电视组合。bathroom新增2 cabinet和picture，缺少真实墙体约束，不能证明与原房间兼容。未使用完整ERP/mesh逐项验证hidden GT，因此不声称这7个新增对象正确或错误。

官方当前AR训练按离散x,z,y排序后取随机前缀，非相机可见集到隐藏集；这可能影响停止与覆盖，但不足以判定本次END由重复对象导致。当前无序号位置编码的模型也不是简单排序即可修复。全1 room mask与官方当前训练/demo一致，不是接口错误。

## 下一步实施优先级

1. 冻结这三张诊断样本，停止换图。Observer先改成可见实例清单（一个实体一个实例，类别+2D框），再用完整原图和实例框/指示信息逐个预测一个3D框。不要仅裁剪后把裁剪相机坐标直接当原相机坐标。采用最具体的支持类别；泛类只用于没有更具体类别的剩余实体。2D框仅辅助定位，不进入Adapter。
2. 增加轻量OBB回投影可视化，记录已知90度内参，排查roll/pitch/yaw解释、深度和尺寸。不能把3D几何精度改善等同于实例去重改善。
3. GT observed → M3D 与新Observer → M3D采用同图同seed对照；先核实官方权重训练尺寸full/half和房间/相机坐标。优先用GT原始OBB转换，不能把GT local AABB强当精确OBB；只用AABB时须标为yaw0近似对照。GT只作诊断，不注入最终单图系统。
4. 若GT context也失败，优先改训练采样：始终保留所有observed，追加已生成hidden，预测下一个hidden或END，复用M3D架构/损失。如果GT好而预测差，先改善Observer几何。若直接Qwen3D仍有明显误差，再考虑RGB深度/几何模块，不需要改Adapter字段。
5. GT Oracle训练保持独立，优先完成Correct/Constant/Shuffled结论；上游结构达到可用程度再接ERP。隐藏结构是一对多，应同时看可见一致性、空间合理性和多seed多样性，而非要求唯一恢复GT或只比较新增数量。

## 外部来源

- Qwen格式： https://github.com/QwenLM/Qwen3-VL/issues/1501
- Qwen代码： https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb
- M3D训练： https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/datasets/m3dlayout_dataset.py

官方说明支持“可直接预测9维框”，不提供本测试准确率保证。CauPano val也不自动证明与官方预训练场景无重叠。本轮为诊断，不为正式泛化实验。
