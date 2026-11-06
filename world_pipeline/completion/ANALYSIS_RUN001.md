# Completion run001 实际结果分析（2026-10-11）

检查对象：observer3d_val3_run004 → completion_val3_run001，三张图，seed000。
查看了原始局部图、俯视图、观测/补全 JSON、原始 token 和本地官方源码；没有重新运行 GPU 推理。

| 样本前缀 | 观测对象 | M3D context | 新增 | 最终 | END |
|---|---:|---:|---:|---:|---|
|8194nk5|22|20|1|23|是|
|X7HyMh|8|8|0|8|是|
|x8F5xy|11|10|6|17|是|

三份 G_full 均通过现有 tensorize(max_objects=160)，观测对象与 G_obs 逐字段一致。原始末 token 均为 END，没有因为 32 步上限而截断，也没有被导出代码跳过的新对象。格式与程序链路已跑通，尚不能认定全局结构恢复有效。

## 样本与观察问题

- 8194nk5：从玻璃门朝向室外庭院，Observer room_type 为 living room，与主要可见内容不符。chair/dining_chair、door/glass_panel_door、sofa/seating/furniture 有疑似重复实例；当前仅去除相同类别 ID 且九维参数完全相等的框。新增 cabinet 中心 local=[2.93,-0.41,6.14]，位于前方，没有提供后方结构。前方不必然等于可见，仍可能遮挡，不能仅按 z 判断隐藏。
- X7HyMh：完全室外，Observer 正确输出 outdoor。8 个 plant 沿规则序列排列；图中确有一排植被，仅凭排列不能断言幻觉，但实例数与米制深度未经验证。AR 第一轮输出 END。此前仅凭“8 个框、解析完整”推荐该图作为室内补全首测不充分。
- x8F5xy：spa/蒸汽浴风格场景。新增 cabinet、ceiling_lamp、3 chair、picture。5 个新增中心 z<0，1 个位于侧方 z约0.21，确有侧后方生成，但没有 GT 对照，无法证明位置正确。Observer Pass A 列出 bathtub/mirror，Pass B 均返回 []，说明完成状态仅表示流程完成，不能等价于观察完整。

## 尺寸与坐标

第三张新增椅子 local AABB 约 0.32–0.36 × 0.52–0.54 × 0.33–0.36 米。按当前 1.5 米相机高度假设，其底部高于假定地面约 0.28–0.30 米；这是尺寸/地面约定需要核实的信号，不能认定是真实场景中的悬空。当前没有真实房间地面恢复。

官方 Matterport 预处理 preprocessing_part1.py 将输入半轴长度乘 2；dataset.sizes 直接读取 size。但同文件的 footprint helper 又按 half extents 解释。现有源码中存在约定不一致，尚未追踪到官方权重训练数据的完整数值链路。不能因视觉上偏小就把所有输出乘 2，也不能断言 full/half 已彻底核实。需用一份官方布局数值与官方对应几何/渲染核对。

## 补全训练分布

本地 m3dlayout_dataset.py 先按离散空间坐标 x,z,y 排序，再随机截取前 n 个对象预测下一个。相机可见集合通常不是这种前缀。complete_scene 接受 context 不等于预训练权重学会了“所有 observed 固定，任意 hidden 待补全”。这可能影响停止和空间覆盖，但三张样本不能证明因果。

当前配置使用无序号位置编码的 AutoregressiveTransformer，不应声称简单重排输入对象即可解决；关键差别是上下文集合的分布。官方 dataset 与 demo 同样使用全 1 room mask，本次固定 mask 不属于偏离该实现的接口错误，但也不提供真实墙体约束。

## 后续顺序

1. 保留这三张结果，不修改正式 validation split；从相同 validation 建筑另外人工选几张明确室内图作为诊断子集，标明选择标准。
2. Observer 改成逐实例查询：Pass A 输出类别和 2D box/实例标识，Pass B 按实例定位 3D box。2D box 只用于定位、检查与去重，不进入 Adapter，不增加训练字段。先叠加回图检查，不批量展开。
3. 核实官方训练 size 的 full/half、相机/房间坐标与地面放置；不做凭观感的缩放修补。
4. 在上述同一室内样本上比较 GT observed 与 predicted observed 驱动的补全。GT 对照只用于诊断，不混入最终单图推理。GT 稍后需独立转换入口，不能将 G_obs.json 直接传给当前 --observed（它要求 observed.json 的原始角点与类别名）。
5. 各取少量 seeds（如 0,1,2）。若 GT context 同样明显失效，优先适配 observed-context/hidden-target 训练采样，而不是强行禁止 END 或不断增加 prompt。
6. 结构达到可用程度后连接 Qwen-Image-Edit；Oracle Encoder/Adapter 训练继续独立进行。

当前没有证据支持修改训练字段、解冻 Qwen、替换 validation split 或直接把本次 JSON 当作可靠完整世界状态。
