# step500 / step1000 室内对照结果

已查看两次各6样本的 baseline/correct/shuffled（共36张输出）及GT。correct字段契约与训练一致：objects_local_aabb_masked_v1，category_id.value，center_local_m.value，size_aabb_local_m.value，center/size valid mask，加对象padding mask；full条件、max_objects160，位置/尺寸以局部相机米制坐标输入，全边长AABB。没有把JSON全文或Observer预测送入Encoder。两次run_config的训练config与实际训练config完全相同；原oracle源代码哈希、两个checkpoint权重哈希、六条GT图哈希通过。correct world_source匹配自己的ID；shuffled匹配指定跨建筑ID。每条geometry tensor形状1×161×8，含一个全局token位置，实际对象25–90，无截断。

两次6张baseline像素完全一致。World更新影响输出，不是分支没加载。step500第40层平均相对残差范围0.270%–0.361%；step1000为1.077%–1.457%。step1000第20层0.158%–0.207%，59层0.0083%–0.0107%。是隐藏激活相对变化，不是输出质量/GT使用比例；绝对RMS大不等于数值爆炸。

视觉结果：楼梯走廊中GT的封闭墙/门没有恢复，correct1000还改变了楼梯上方开口；浅色客厅GT另一侧厨房/电视区域仍变成大白墙/走廊；卧室GT左侧卫浴通道仍缺失，右侧衣柜布置偏离；另一客厅多排沙发重复没有明显消除。spa和吧台主要有物体外观/边界变化，没有稳定恢复真实布局。500与1000都未显示正确GT明显优于跨建筑错配GT的可靠空间控制，不能以变化变大称为学习进步。

结论：没有发现GT字段、ID或checkpoint输入错配证据；主要可见现象是强基线布局主导、状态影响局部外观而缺少结构纠正。空间注意力没有显式ERP方位锚定，且输入不含墙/房间layout，是合理机制假设，并非定位到唯一根因。训练损失无明显下降、仅4000有放回样本抽取也可能相关。不能保证更多训练或强行放大gate会解决。

下一步建议只做固定噪声/固定timestep的预测与损失对照（见过的train与heldout val均选少量），对同一个noisy latent比较无world/正确GT/错配GT/仅位置错排：确定是否学会读取状态及是否学会位置。必要时同时核对缓存训练条件与在线推理条件；本次CPU审核没有运行此GPU等价实验。不再反复挑样本或立即增加大批训练。确认条件学习瓶颈后，再用新实验验证ERP方向空间约束，保留旧checkpoint与训练字段。

机器可读字段审计：qwen_edit_pano/audit/indoor_result_field_audit_2026-10-11.json。原GT、训练代码、bundle及checkpoint本次未修改。
