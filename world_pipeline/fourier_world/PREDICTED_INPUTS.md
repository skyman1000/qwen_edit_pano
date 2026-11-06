# 预测状态到全景：单图三路诊断

> 最新用户阶段：Adapter 尚未训练好，用户没有运行本轮厨房/客厅三路全景；当前只推进状态补全。本文的生成命令是预留技术入口，暂不执行，也不因此自行启动训练。当前进度见[2026-10-21 交接](../../../research/CODEX_HANDOFF_2026-10-21.md)。

2026-10-21 更新：`--scene` 也支持 `relational_layout` 输出的样本目录
（包含 `input.json/G_obs.json/G_full.json/local.png`），直接验证其中原图与观测来源，
不必复制图或补造 `config.json`。当前厨房、客厅两例的真实状态已通过 CPU 接口检查，
均为 16 个观测、18 个完整对象。手动运行入口及方法分析见
[当前整体路线审查](../../../research/audits/panorama_completion_route_2026-10-21/REPORT.md)。
下面 2026-10-20 的计数和对象方向属于当时样例，不是当前厨房/客厅的统计。

2026-10-20。`box_world_run008_height_anchor` 实际输出17个WildDet3D观测框及3个新框（椅子、边桌、灯）。相机y下边界参照已按模型声明生效，3个新增框均在视野外，无涉及新物体的AABB重叠。当前可以用于粗原型的端到端诊断，不代表完整物理场景或跨图泛化已验收。

原 `fourier_world.inference` 面向Oracle bundle及固定GT selection。新 `predicted_inference` 直接读取完成目录中的预测 `G_obs.json/G_full.json` 与所绑定的原图，不改Oracle入口、数据、权重或补全算法，不读取场景GT。

三组使用同一原始1024 RGB、同一Panorama/World权重、普通任务prompt、分辨率、采样参数和相同种子。不会把新增物体文字写入prompt来冒充Adapter效果：

| 输出 | World条件 | 本例有效token（含global） |
|---|---|---|
| baseline.png | 关闭World，保留Panorama LoRA和全景padding | 不使用World |
| observed.png | 预测G_obs，17对象 | 18 |
| completed.png | 预测G_full，20对象 | 21 |

`observed`与`completed`都以`tensorize(..., condition='full')`读取传入图，保证后者的新增对象不会被“只保留可见对象”过滤。来源、ID、描述、关系、房间假设不进入Encoder。每组独立重建CPU generator并直接使用`--seed`，这与历史Oracle按样本ID派生seed的规则不同，不能与旧结果混作逐像素对照。

加载时检查预测stage/scope/坐标schema、观测字段和数值前缀不变、词表、原图绑定、对象容量，以及checkpoint原有bundle/基座/scale绑定。只读取bundle元数据、全局类别词表与所需基座配置，不读取bundle中的samples或场景GT。使用现有qwen360环境和本地缓存，不安装、不下载。

以下checkpoint是已存在且绑定epoch015/scale0.5的Fourier World step750，选它用于固定组合诊断，不表示它是最优模型；不可直接换成新Panorama epoch却沿用这个World权重。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

# CPU预检，不加载生成模型、不创建输出目录。
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -m \
  qwen_edit_pano.world_pipeline.fourier_world.predicted_inference \
  --scene qwen_edit_pano/outputs/box_world_run008_height_anchor \
  --checkpoint qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1/checkpoint-step000750 \
  --output qwen_edit_pano/outputs/box_world_run008_pano_diagnostic \
  --preflight

# 由用户手动提交GPU任务；GPU1可自行替换。
srun -p debug --nodelist=GPU1 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=04:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u -m \
  qwen_edit_pano.world_pipeline.fourier_world.predicted_inference \
  --scene qwen_edit_pano/outputs/box_world_run008_height_anchor \
  --checkpoint qwen_edit_pano/outputs/oracle_supplement_epoch015_scale05_fourier_full_v1/checkpoint-step000750 \
  --steps 28 --cfg 4 --seed 0 \
  --output qwen_edit_pano/outputs/box_world_run008_pano_diagnostic
```

检查`comparison.png`与三张完整ERP：先看原图内容是否保持，再重点比较`completed`对`observed`是否在相机右方约60～66度增加所声明的椅子、桌子、灯。本例三者中心、完整尺寸见保存的G_full；不要求图像重现未知真实GT。输入特征另存为`observed_input.safetensors/completed_input.safetensors`供追踪。门控残差或画面发生变化只说明分支起作用，不证明类别、数量和空间位置正确。

若三路仅改变纹理、墙面或风格，优先检查Adapter的对象控制，不继续调Qwen来掩盖问题。若能遵循新增对象，再用少量不同局部图验证补全泛化。灯高于桌面的0.1m间距仍是粗camera-y假设，当前不引入支撑优化器；G_full也不是所有房间物体已穷尽的认证。

本次仅完成CPU接口/权重绑定验证和测试；真实GPU三路结果仍待用户执行。

官方用法核验：[Diffusers v0.37 QwenImage文档](https://huggingface.co/docs/diffusers/v0.37.0/en/api/pipelines/qwenimage)。保持既有本地PairedPanoPipeline的FP32 VAE、BF16生成模型、target-only全景padding和scoped BF16 World，不额外使用全局autocast，也不改变文本CFG机制。
