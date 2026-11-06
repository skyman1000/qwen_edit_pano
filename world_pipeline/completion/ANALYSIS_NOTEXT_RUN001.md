# 无文本对照分析（2026-10-11）

客厅样本X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c，seed0。
两份保存config逐项比较，唯一差异是text由房间类型句变成空串。路径相同不等于文件有历史哈希保证，但当前记录支持一次有效的单变量诊断。

有文本：2观测+4新增（large_plant_container,kitchen_cabinet,tv_monitor,tv_stand），END。
无文本：2观测+5新增（large_plant_container,2 kitchen_cabinet,floor_lamp,desk），END。
无文本改变类别与位置，未证明质量提高。单样本单seed不能给两种条件排优劣；更多对象不等于更合理全局结构。

## 本次新发现：half extent 的强数值证据

采用当前桥接scene_y=local_y+1.5，M3D只输出绕y旋转，所以导出AABB的y边长就是当前按full解释的原始size_y。

|无文本新对象|scene中心y|模型size_y|按full解释底部|按half解释底部|
|---|---:|---:|---:|---:|
|large_plant_container|0.6616|0.6506|0.3363|0.0111|
|kitchen_cabinet 1|0.3407|0.3422|0.1696|-0.0015|
|floor_lamp|0.6438|0.6417|0.3230|0.0021|
|kitchen_cabinet 2|0.3184|0.3176|0.1597|0.0009|
|desk|0.3395|0.3403|0.1693|-0.0008|

有文本的花盆、柜子、电视柜也满足同样关系，按half解释底部约0.0187/0.0057/0.0018米。挂墙电视不应要求落地，未纳入这项比较。

模型生成的多个通常落地类别都呈center_y≈size_y，是size代表半高的强证据，比凭渲染主观观察更有力。但完整官方权重训练数据链路仍未逐条核实，源码demo尺寸使用和部分数据预处理存在不一致，不能直接声称所有来源都已证明统一半边长。

现有complete.py --size-convention half的逻辑恰好成对处理：送入context之前full size除2，输出反归一化之后size乘2；保留原始观测graph，不会缩放其中心或最终尺寸。仅把旧输出乘2不能修正已经送错给网络的context，需重跑补全。

## 最小下一步

保持YAML text_condition true，保留同一observed、seed0、相机高度和预算。分别重跑：
- 原房间类型文本 + --size-convention half → completion_indoor_halftext_run001
- --text "" + --size-convention half → completion_indoor_halfnotext_run001

优先检查对象实际完整尺寸、底部高度、与已有对象的相容性和回投影。不要仅按底部接近0选择模型；这是诊断尺寸约定的证据，不是保证整体布局正确。不要调整相机高度掩盖half/full问题。

尺寸转换核实之后，再决定文本条件消融，多seed比较；Observer实例重复与已有深度偏差仍需独立改进；GT observed对照仍是隔离上游误差的必要诊断。当前不建议立即微调M3D或解冻生成器。

本次没有改默认size-convention、覆盖旧结果或运行GPU。
