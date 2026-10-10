# GT 文本的 LLM 忠实改写实验

本次只改变文字表达。输入是上一轮从 G_full 计算出的同一份 spatial 事实投影，仍然不含精确尺寸、垂直坐标和房间关系；不是把完整 G_full 的信息都传给 LLM。类别、数量、相机方向、距离档必须保留。LLM 不看 Local 或 GT ERP、不修正 GT 标签、不补全场景。Observer、Completion、GT 构建、World Adapter 和 Panorama LoRA 均不修改。

已有结果：`qwen_edit_pano/outputs/gt_text_epoch006_2samples_v1`，2 个 train 样本、seed 0、epoch006、28 steps、CFG 4。普通任务文本 23 tokens，带规则 GT 的文本 285–358 tokens，未触及 1024 文本预算。浴室样例有对象层面的响应，但布局/形状仍有问题。不能仅凭这两条 train 图否定文本路线，也不能声称精确三维控制已成立。

新增两个模式 `llm_correct`、`llm_shuffled`。同一 source 只改写一次，正确和错配模式复用同一 caption；错配来源沿用上一轮映射。与已有 baseline/correct/shuffled 对照，总共新增 4 张图。入口会检查 checkpoint、数据哈希、样本、源文本、错配映射、seed、采样设置与上一轮一致。代码检查不证明语义忠实。

## 1. CPU 测试（用户执行）

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export TXT_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python
export OBS_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

"$TXT_PY" -m unittest qwen_edit_pano.world_pipeline.test_gt_text qwen_edit_pano.world_pipeline.test_gt_text_rewrite -v
```

测试覆盖解析拒绝、正确/错配 caption 来源，以及 seed、事实、源映射变更时拒绝混用；不验证模型能力。助手未执行本次测试或推理，下面命令由用户执行。

## 2. 生成改写草稿：独立 GPU 作业

默认复用已经缓存的 Qwen3-VL-8B-Instruct，作为文本改写器，不调用 Observer 入口。用已有 qwen3vl 环境，BF16、SDPA、纯文本输入；不升级 qwen360，不自动下载模型，不与 Edit 模型同驻显存。

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=01:00:00 \
  "$OBS_PY" -u -m qwen_edit_pano.world_pipeline.gt_text_rewrite \
  --reference-run qwen_edit_pano/outputs/gt_text_epoch006_2samples_v1 \
  --model Qwen/Qwen3-VL-8B-Instruct --max-new-tokens 2048 \
  --output qwen_edit_pano/outputs/gt_text_rewrite_drafts_v1
```

输出 `rewrites.json` 包含每个来源的 `source_text`、`caption`、`raw_output`、哈希、编译模型信息及原实验设置。这里只对两个目标及它们的错配来源改写，不跑全量数据。若 JSON 失败或输出未结束，保留原输出并报错，不把失败的内容默默替换成规则文本。

检查 source_text 与 caption：类别和数量是否都保留，back 是否仍指相机身后，是否把相机左右写成 ERP 画面左右，是否新增“相邻卧室/门后/靠墙/木质”等无根据内容。JSON 成功只代表能解析，不能当作事实核验通过。发现问题先保留这一版供分析，不直接生成图。若人工改写 caption，应另存为新文件，并把 compiler.mode 改为 human_edited，记录修改原因，避免声称它是模型原始输出。

若要换为其他大模型，可用同一模块 `--export-only`，不需要 GPU：

```bash
"$TXT_PY" -m qwen_edit_pano.world_pipeline.gt_text_rewrite \
  --reference-run qwen_edit_pano/outputs/gt_text_epoch006_2samples_v1 \
  --export-only --output qwen_edit_pano/outputs/gt_text_rewrite_external_v1
```

将文件内 `instruction` 和每个 `source_text` 分别作为 system/user 输入给所选模型，把其返回的 caption 字符串填入相应项，补全 compiler 中的模型名称、版本和生成设置。不要更改源事实、参考设置或来源映射。后续 `--rewrite-file` 指向该文件即可。此入口不会调用外部 API，也没有自动上传图片或数据。

## 3. 只检查最终 prompt，不加载模型

```bash
"$TXT_PY" -m qwen_edit_pano.world_pipeline.gt_text_oracle \
  --split train --limit 2 --seeds 0 --text-format spatial \
  --steps 28 --cfg 4 --offload model \
  --rewrite-file qwen_edit_pano/outputs/gt_text_rewrite_drafts_v1/rewrites.json \
  --modes llm_correct llm_shuffled --prepare-only \
  --output research/audits/gt_text_rewrite_plan_v1
```

检查该目录 `prompts.json`。输出路径必须是新目录；准备目录与推理目录分开。和旧规则文本一样，实际 token 预算在 Edit 加载 tokenizer 后检查，超长报错而不静默截断。

## 4. 新增 4 张 ERP

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=96G --qos=normal --time=04:00:00 \
  "$TXT_PY" -u -m qwen_edit_pano.world_pipeline.gt_text_oracle \
  --checkpoint qwen_edit_pano/outputs/paired_full_63637/checkpoint-epoch006 \
  --split train --limit 2 --seeds 0 --text-format spatial \
  --steps 28 --cfg 4 --offload model \
  --rewrite-file qwen_edit_pano/outputs/gt_text_rewrite_drafts_v1/rewrites.json \
  --modes llm_correct llm_shuffled \
  --output qwen_edit_pano/outputs/gt_text_epoch006_llm_2samples_v1
```

输出每个 ID 的 `seed-0/llm_correct.png`、`llm_shuffled.png`，另存 Local、GT ERP 和全部生成元数据。旧三组结果不覆盖，不需重跑。沿用同一软件环境与硬件更利于比较；同 seed 不是跨硬件绝对逐像素一致的保证。

## 判读与停止条件

对同一 ID 横向比较旧 baseline、correct、shuffled 和新 llm_correct、llm_shuffled：Local 是否保真，镜头外对象类别/数量/方向是否更接近 GT，是否出现几何畸变、重复对象或接缝退化。不要只评价画面好看。若改写漏掉事实或修正了原始标签，这已经是另一个信息筛选/标签清理实验，不能算表达方式收益。

若忠实改写优于规则文本，再用更多已验收样本和多 seed 验证，最终需要独立建筑 val。若只改变物体类别却不能控制方向/布局，或改写无稳定收益，停止反复调文案，转向验证结构注入的 GT Encoder/Adapter oracle。失败不能单独证明 Adapter 必须有效；成功也不能单独证明 Adapter 没必要。

当前流程支持全景 GT 的 oracle 上界诊断，部署阶段仍只有 Local RGB 与可选文字，需要 Observer/Completion 给出状态。GT 类别噪声另行追踪，不允许文本模型根据常识私自修正后继续称作原 GT。
