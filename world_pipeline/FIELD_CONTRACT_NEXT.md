# 本阶段接口草案与执行顺序

状态：DRAFT；不是新训练schema，也不是字段可靠性认证。继续沿用已有pilot输出，不强迫模型预测GT专属信息。

## 输入与输出责任

1. Observer只接受Local RGB及允许的相机元数据/用户文本。当前生成room_type候选、objects.category_id和bbox2d；raw尺度1000，转换后共用bbox为0–1。先保持已测试的baseline_v1/checkpoint配置。
2. 现有parse_prediction转换到公共G_obs，保留可见类别、框；米制中心、全物体尺寸、房间归属未知。room候选不等于Matterport房间标签。source/valid是记录属性，不是正确概率。
3. Completion未来必须承担可见物体缺失的三维属性估计，以及镜头外对象预测；完整输出仍允许不确定字段。该模块尚未实现。不得把GT完整3D G_obs当作实际Observer等价输入。
4. Encoder未来只读取固定白名单及有效性掩码；GT oracle与Completion使用同一个tensorizer。GT ID、建筑ID、源文件、绝对世界坐标不能作为学习特征。tensorizer尚未实现。

## 字段决策表

| 字段 | 当前处理 | 冻结前必要证据 |
|---|---|---|
| category_id | 固定词表；未知可null；保留原GT类别 | 查清当前bed/sofa、clothes/towel等是模型错误还是源映射/粒度差异；禁止为提分合类 |
| bbox2d_visible_xyxy_norm | 图像边界0–1，可见实例范围 | 与mesh可见mask区分完整物体范围；审查灯/浴缸框差异，允许无法确定 |
| center_local_m | Observer未知；GT由源OBB转换 | 原点、x右y图像上z前、米制；由未来状态估计产生同义预测 |
| size_aabb_local_m | Observer未知；GT从源OBB角点派生 | 相机轴AABB全尺寸；不能混用OBB半径或可见片段尺寸 |
| room字段 | 可选；不急于接入Encoder | 原类型词表映射、粗region范围与真实墙面区别、缺失与冲突 |
| local_evidence | 记录有无观测证据 | 没检测到不等于不存在；GT可见性与预测声明不能混作同来源 |
| instance ID | GT仅审计，预测样本内临时ID | 不能按顺序配对；最佳IoU仅候选；部件和整物体冲突待审 |

## 本次用户执行

仅运行audit_fields CPU命令，复用gt_world_pilot_v1和observer_ab_v3_checkpoint。无GPU、无新下载、无新推理。产物REPORT.md、field_coverage.json、review_cases.csv/json、samples.json、audit_manifest.json。

## 产物之后具体要做什么

助手据审计清单追溯源对象/类别/segment定义，输出逐问题处理决定；用户无需重标全套。保留原始标签；只有证据充分才建立版本化转换或评测忽略策略。当前Local/ERP阈值问题需另行修订导出策略并生成新版本，不改旧flags。

以上语义与集合规则通过后冻结共用schema和tensorizer，再实现Edit target-only World Encoder+Adapter oracle。GPU命令须等入口、CPU接口检查和具体实验配置存在后给出。不得把本审计完成理解为可以开始正式Completion训练。
