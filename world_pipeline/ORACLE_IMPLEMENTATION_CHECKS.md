# Oracle实现检查记录

- checkpoint-epoch001/COMPLETE.json：completed_epochs1、optimizer_step1881；保留该固定基线。
- oracle_data实际CPU执行：research/audits/oracle_bundle_devcheck_v2，7train/0val；hash、paired校准来源、cache、建筑split检查通过。
- oracle_train --preflight实际CPU执行通过；没有加载真实模型权重到GPU。
- oracle_test五项通过：未知与零区分/常量输入/超限拒绝，target-only残差与padding mask，zero gate逐元素基线相同及temporary/persistent一致，冻结LoRA小型真实Diffusers模型的checkpoint backward与条件捕获，门控开启后的Encoder梯度/保存重载/对象顺序置换/撤掉world回到基线。
- 五个oracle Python模块语法检查通过。
- 当前qwen_edit_pano.common.source_hashes与checkpoint训练记录比较，无差异；没有改旧训练Python、cache、输出或环境。
- 尚未运行真实GPU smoke、目标posterior编码、world训练或重载生成；实际显存、速度和收益未知。5个CPU测试不等于20B接入已在GPU验证。

运行说明：RUN_ORACLE.md。所有GPU命令由用户手动srun。当前代码是可执行pilot入口，正式oracle收益评测仍需合格独立val与扩展GT。
