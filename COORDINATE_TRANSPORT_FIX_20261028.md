# Qwen 坐标容器恢复：实际错误与执行方法

来源：`outputs/organized_multiscene_named_prepared_20261028_001`。这次实际已经完成 **6/10 世界**：API 五图全部成功，Qwen 客厅一图成功。剩余四图仍有固定语义库存，但未进入物理编译。本轮不改变因果图、概率表、库存、尺寸机制和物理编译器，也不重新执行 API。

## 根因

上一版把坐标改成具名字段，解决了列数歧义，却仍严格要求特定 JSON 容器。这不足以覆盖真实 Qwen 回复：

- 厨房返回 `{"a4_g0":{"roots":[],"hosted":[...]}}`，数值与对象 ID 均已提供；解析器因外层不是 `group_id,roots,hosted` 而拒绝，纠错后仍保留外层包装。
- 卧室返回 `{固定对象ID:{yaw_deg,host_offset_uv,surface_fraction},...}`。ID 是字典键，没有丢失；旧解析器无法识别，重试变成 `roots:[],hosted:[]`。
- 休息区同样返回 ID 字典，包含 14 个对象的有效具名坐标；旧重试只留下 6 个根物体，遗漏其余附着对象。
- 卫生间把同一 ID 同时放入 roots 与 hosted。该对象的宿主已固定，root_offset_m 那条不符合其坐标契约；host_offset_uv 那条符合契约。旧流程整体拒绝后，纠错变成空列表，丢掉了原本有效的附着坐标。

所以直接增加重试、token 或再次改因果模型，都没有针对当前根因。上次 32 项 CPU 测试验证了规范回复和部分数值错误，却没有覆盖这些真实容器变体，这是测试覆盖的缺口。当前文件中的失败 `summary.failures` 仍显示历史 numeric_group 错误，**本轮最新错误应读 `named_coordinate_recovery` 和 `named_recovery_error_v2.txt`**。

没有发现本轮 API 429 再次阻断最终世界的证据：`worlds_api.log` 已是 complete，五份 API G_full 均存在。`srun ... exit code 1` 对应四份 Qwen 恢复失败导致阶段汇总失败，不能据此判断 GPU 硬件异常。

## 修复范围

新增 `world_pipeline/completion/assembly_coordinate_transport.py`：

1. 只对可确定的容器做无损转换：解开精确组 ID 包装；把精确对象键恢复为 id；把 roots/hosted 展平为统一记录。组 ID、对象 ID 冲突拒绝处理，不根据近似名称或顺序绑定。
2. 每条数值记录仍由原具名字段契约严格检查。缺少 yaw/坐标、非有限数值、UV/XYZ 长度错误、错误坐标系、越界 surface_fraction 都不会被填默认值或猜测转换。存在两条不同的合法坐标时，该对象保持待补问；不擅自选一条。重复 ID 中只有符合冻结宿主契约的唯一有效记录时，可以保留它，并记录错误记录。
3. 从上一轮原始回复和纠错回复回收合法记录，逐条留下来源与转换记录。后续纠错空列表不会抹去早先有效数值。
4. 新请求只使用一个 `objects` 数组。每次最多六个待补坐标对象是输出计算预算，不是场景数量目标；仍提供完整世界、全部冻结尺寸和已接受坐标。每批最多两轮，每轮最多一次格式纠正；缺失组距离可独立补问。没有无限重试。
5. 一旦接受有效字段立即保存进度；下一轮只补缺失字段，并提供具体字段错误。原机制、尺寸、对象、宿主、采样相机区域不变。原物理求解器仍可能按原规则调整最终位置；保留原始预测不等于锁死最终优化位置。
6. 数值检查未通过仍保留全量候选，不删对象换成功；格式修复通过也不能视为真实世界恢复或语义质量认证。

| 输入 | 固定隐藏库存 | 本轮可直接复用坐标 | 还需请求坐标 |
|---|---:|---:|---:|
| 厨房 | 33 | 14 | 19 |
| 休息区 | 28 | 18 | 10 |
| 卫生间 | 13 | 8 | 5 |
| 卧室 | 23 | 12 | 11 |

未生成的后续组还需预测组距离；表中“复用”只指格式和类型合法，不表示原数值具有可靠的空间质量。例如已有多个根物体偏移都是零，仍须交给原数值编译和最终可视化检查。

另行核对厨房人工回放：接触及视锥线性约束违反数为零，但原求解器最终仍残留 4 对餐椅相关框重叠，其中餐椅1与餐椅3重合；两者原预测偏移均为零。求解器报告收敛不代表碰撞全部解决，`complete_search` 也为 false。因此本次仍严格保留“不通过”的候选标记，没有降低检查门槛或删除餐椅。这是 CPU 补齐其它组后的具体诊断，不能断言真实 Qwen 补齐后必定同样失败；不据此改写因果概率。简要记录为审计目录的 `kitchen_CPU_numeric_diagnostic/review_brief.json`。

## 执行

本次仅恢复四份 Qwen 坐标，**不需要 API Key**。脚本会复制到新目录，核对 API 五个世界已完成及 Qwen 机制、尺寸均就绪，复用六个成功世界，仅执行 Qwen worlds，然后自动生成可视化。源目录不会被修改。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
unset CAUPANO_RECOVERY_OUTPUT

bash qwen_edit_pano/scripts/organized_multiscene.sh repair-coordinates GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001
```

这里用的是新入口 `repair-coordinates`。不要直接 `resume` 旧目录：旧目录绑定旧源码，并保留失败阶段标记和上次已用完的恢复预算。

运行结束打印新的 `index.html`，包含局部原图、平面图、局部投影、六向投影、3D、2:1 ERP 和编号+类别标签。可映射新增物体沿用同一橙色；不可映射新增物体红色；观测蓝色。

如果先只准备、不提交 GPU：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh prepare-coordinates GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001
```

之后对它打印的新目录运行 `resume-coordinates GPU3 新目录`。这些专用入口会在缺少已完成 API 世界或冻结 Qwen 输入时拒绝执行，避免把不完整实验误认为可仅恢复坐标。

可视化重建：`bash qwen_edit_pano/scripts/organized_multiscene.sh report 新目录`。

验证与实际回复审计：`../research/audits/multiscene_named_recovery_20261028/release_002/`。37 项 CPU 回归全部通过，耗时 355.545 秒；另外验证了不设置 API Key 的脚本路径只执行前置核验、Qwen worlds 和 report。六个原成功 G_full 及来源 JSON 核验未改。测试使用 CPU 人工回复填补剩余字段，不调用真实 Qwen/API；其中三个完整候选通过原数值编译，厨房人工补全候选未通过数值门槛，33 个物体全部保留在候选诊断中。这个结果验证错误处理，不证明真实厨房生成失败或成功；真实模型生成及质量仍由上述用户执行结果检验。
