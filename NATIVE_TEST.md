# 原生 Edit：已有局部图 → ERP

这条入口独立于 train.py/inference.py 的纯文本适配。使用官方 QwenImageEditPlusPipeline，将现有 skybox2 图像同时送入原生视觉语言和 VAE 条件路径。没有 LoRA、World Adapter、自定义 circular padding，不需要训练或文本缓存。

默认从 benchmark_assets/mp3d_stitched1092/reference.jsonl 选 4 个样本，按建筑轮流选择；从原始 ZIP 直接读取 skybox2 JPEG。采用已有 PanFusion 转换器的 skybox2=F 约定，不重新投影局部输入。GT caption 被丢弃，GT 图像只复制到结果目录。

默认 1024×2048、40 steps、CFG=4、negative prompt=" "、bf16、model CPU offload。40 steps 是本次独立原生测试设置，不修改旧推理脚本的 28 steps。可通过 CLI 覆盖。原生 Edit 会按它自己的规则缩放图像条件，本入口不修改此行为。

## 执行

本次只创建了代码，没有执行语法测试、模型下载、推理或 sbatch。以下均由用户执行。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
eval "$(conda shell.bash hook)"
conda activate qwen360
mkdir -p qwen_edit_pano/outputs

# 可选静态检查，不加载模型。
python -m py_compile qwen_edit_pano/native_test.py
bash -n qwen_edit_pano/scripts/native_test.sh
python -m qwen_edit_pano.native_test --help

# 模型尚未下载时，在联网节点执行；沿用 HF_HUB_CACHE / HF_HOME 默认缓存，不写入项目内。
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
'from qwen_edit_pano.common import model_snapshot, DEFAULT_MODEL; print(model_snapshot(DEFAULT_MODEL))'

# 先生成一张明确配对的样本。
sbatch qwen_edit_pano/scripts/native_test.sh \
  --sample-id sT4fr6TAbpF_8112f9278bdf4dfbad53e140b846620c

# 第一张流程正常后，再测试分布在不同建筑中的 8 个样本。
sbatch qwen_edit_pano/scripts/native_test.sh --limit 8
```

若完整权重在其他本地目录，使用同一环境变量指向它：

```bash
QWEN_MODEL=/absolute/path/to/Qwen-Image-Edit-2511 \
  sbatch --export=ALL qwen_edit_pano/scripts/native_test.sh --limit 4
```

默认脚本离线加载模型；本地权重目录应包含 model_index.json 及 transformer、text_encoder、tokenizer、processor、VAE、scheduler 等必要文件。

默认提示词位于 prompts/native_erp.txt：要求同相机位置、完整球面、2:1 ERP、输入视图朝前居中、保留可见内容并补全背面、接缝连续。该约束是请求，不代表模型一定做到。替换提示词使用 `--prompt-file /path/to/prompt.txt`，实际文本会保存到 generation_config.json。

自定义图片也可使用 `--image /absolute/path/local.jpg`；默认提示词假定 90° 前向视图，若照片 FOV 不同，应提供相应 prompt-file。这个入口不估计相机参数。

## 输出与判断

结果目录：qwen_edit_pano/outputs/native_edit_<JOB_ID>/。日志位于同一 outputs 目录。

每个样本包含：

- input.png：实际送入 pipeline 的 RGB。
- input_source.bin：ZIP 中原始 JPEG 字节（自定义输入时为其原始文件字节）。
- generated_erp.png：原生模型输出，没有事后拉伸或接缝修补。
- gt_erp.png：已有 stitched GT；自定义图模式没有此文件。
- sample.json：ZIP 成员、哈希、seed、耗时和显存。

根目录记录 generation_config.json、generated.jsonl、COMPLETE.json。输出必须是新目录，不覆盖已有结果；中断后用新作业目录重跑。COMPLETE 只表示生成结束和尺寸检查通过，不代表标准球面投影或局部保持通过。

检查输入内容是否保留、左右边界是否连续、天顶/地面是否合理，最好放入全景查看器观察。仅有 2:1 尺寸不足以证明 ERP 正确。镜头外允许合理的不同布局，不能仅以其与唯一 GT 不一致就判失败。本脚本不自动计算重投影指标或认证全景质量。
