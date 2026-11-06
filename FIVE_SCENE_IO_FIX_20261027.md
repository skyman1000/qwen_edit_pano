# 五图单输出：实际错误原因与恢复执行

来源运行：`outputs/organized_multiscene_20261027_203624_2435093`。本次修复观察与坐标回复接口、阶段依赖管理，保留 v13 因果机制、库存概率、尺寸冻结及原物理代理求解器。没有训练，没有自动运行真实 GPU/API。

## 实际结果及根因

| 输入 | Qwen | API | 原因 |
| --- | --- | --- | --- |
| 公共休息区 c9e8… | 未进入补全 | 未进入补全 | 共享 Qwen 观察器重复生成 glass vase，8192 token 截断 |
| 客厅 2bb4… | 成功，新增21 | 成功，新增20 | 两路均通过当前数值检查 |
| 卫生间 3a12… | 已抽13件，坐标回复失败 | 成功，新增13 | Qwen 第一次原样返回 null 位置模板，重试只把 yaw 改为0，偏移仍为 null |

休息区的两次原始识别文本逐字相同，后半段不断沿横坐标移动并重复“glass vase”；两次都达到 8192 token、无 EOS。这不是普通输出略长，单纯提高上限会延长重复。旧通用重试又把整段坏文本加入上下文，不利于纠正。

该场景后面的 G_2d、G_obs、context、domain 缺失均为前置失败引发的连锁错误，不是独立的文件路径配置错误。`srun ... exit code 1` 是 Python 阶段返回非零，不说明 GPU3 或权重损坏；TRANSFORMERS_CACHE 的 FutureWarning 不是根因。

卫生间 Qwen 的 13 件库存和尺寸已经产生。位置回复以 EOS 正常结束，第一次540 token、重试553 token，远未达到8192；因此不能把它归为 token 不足，也不能当作因果模型没有选中物体。

另外卫生间两路存在部分提案被拒：跨组依赖或与可见宿主/视野外条件不相容。有效提案已进入共享 D，机制估计和 API 几何仍能完成。这应报告为覆盖警告，不能把有效提案全部丢弃，也不能为了消除警告取消原语义约束。

## 针对性修复

1. **可见实例按块识别。** 每次最多8个新增可见实例，明确 `done`，携带已接受实例以避免重复。仍保留原64实例计算预算，最多8块、每块至多一次格式纠正。它不是8件物体的目标，不限制隐藏库存。未明确完成时不导出残缺 G_2d；保留原始诊断。
2. **截断重试不复制坏长文本。** 仅给紧凑格式错误和当前块要求，原始文本保存在文件中。
3. **数值位置改用分组紧凑行。** 精确固定 ID 和每行的数值列，没有空数值模板可照抄。yaw、根相对偏移、宿主归一化偏移及支撑参数全部严格校验；不填默认0或3m，不接收 null，不删对象。
4. **仍做全世界数值编译。** 分组请求看到全部库存、观测、尺寸、空间计划及已预测组；最终合并后使用原联合宿主放置和全局求解器。没有改变因果概率。新鲜生成时，两后端使用相同坐标接口。
5. **前置失败明确跳过。** 缺少观察或候选域时，记录 blocked_dependency，不初始化模型、不追加 API 请求、不伪造空世界，也不制造十几条重复 FileNotFoundError 堆栈。
6. **区分部分提案警告与最终失败。** 有效部分提案保留并公开覆盖情况；仅因此存在警告时不再让阶段进程非零退出。世界确实未导出时仍报告失败。

## 五张局部图

默认清单已改为 `data/observer3d_indoor_val5_inputs.jsonl`：原三张保留，新增两个局部RGB。

| 输入 | 内容 |
| --- | --- |
| 8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39 | 公共休息区 |
| X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c | 客厅 |
| x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6 | 卫生间 |
| zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a | 卧室局部 |
| 8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029 | 厨房及吧台 |

已逐图看过RGB。项目另一个清单把 `zsNo…ed864…` 标成 bedroom，但该局部图实际是沙发；因此本次用真正可见床的 `ead481…`。场景名称仅供预览，不给模型额外输入 GT 场景标签。生成读取局部图和 HFOV，未使用完整 GT 世界。

## 推荐执行：恢复并扩展，复用已成功结果

确认 `SILICONFLOW_API_KEY` 已设置，执行：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG

bash qwen_edit_pano/scripts/organized_multiscene.sh repair-generate GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_20261027_203624_2435093
```

它创建新的 `outputs/organized_multiscene_recovery_时间_PID/`，不会写入原目录：

- 复用客厅 Qwen、客厅 API、卫生间 API 三份成功状态，逐字节保持；不重新调用模型选择更好的样本。
- 复用两个已完成场景的观察、候选、CPT、库存及尺寸。卫生间 Qwen 只重新预测三个组的坐标行，最多6次格式尝试，随后编译原13件库存。
- 休息区从失败的可见实例识别重新开始，旧坏回复在新副本的 recovery_history 中保留。
- 卧室和厨房完整执行新观察、该图独立候选、机制和几何。

最终仍是**每图、每后端一个完整世界**。五图×Qwen/API，共10份计划状态，其中3份复用、其余7份待恢复或生成。内部概率估计仍有多次调用，不等于输出多个世界。

这一恢复批次包含复用、恢复与新输入，不能当作十份全部新生成的严格配对模型能力实验。需要全新、相同接口的五图测试时可另开：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh generate GPU3
```

原目录请保留用于溯源；新目录的 `recovery_manifest.json` 校验原来源文件及受保护副本。若确要重做完整测试，使用新输出目录，不用旧 `resume` 反复挑选成功样本。

只做恢复准备检查、不启动模型：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh prepare-repair GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_20261027_203624_2435093
```

## 结果和可视化

脚本末尾打印新目录 `index.html`，包含逐世界状态、阶段错误/依赖表，以及原图、编号＋类别平面图、原相机和六方向投影、三维图、2:1 ERP 图。仍为观测蓝、可映射新增橙、未映射新增红；灰色ERP不是生成隐藏RGB。

恢复记录：

- `recovery_manifest.json`：复用世界、重置接口、新增图及哈希。
- `recovery_history/phase_records/`：原失败和部分提案状态。
- `scenes/<ID>/observation/qwen/observer2d/chunk_*`：逐块观察原始调用。
- `phase_records/*`：当前阶段 complete / partial / blocked_dependency / failed。
- `world_01/compact_recovery_calls/`：原卫生间Qwen坐标恢复。
- `world_01/compact_numeric_stage.json`：数值行合并结果。

测试入口：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh test
```

已验证原始重复/空模板故障、分块结束与重复检查、不输出残缺观察、13件库存和冻结尺寸不变、紧凑行类型及ID、前置阻断不调用模型、保留原成功状态与CPT、五图输入及单世界模式。CPU人工坐标只验证流程，不代表真实新模型质量；本次真实重跑仍由用户执行。
