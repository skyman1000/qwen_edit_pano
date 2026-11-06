# 局部图片 → Qwen3-VL 3D 框 → M3DLayout 补全

两个入口已实现，模型由你手动运行，不改现有 Encoder/Adapter 训练。

已完成 CLI 帮助、语法、CPU 坐标/旋转/逆变换及现有 `tensorize` 兼容性检查；尚未实测模型加载和 GPU 推理。

2026-10-10 输出截断修复：用户已成功加载模型，但 plant 回答连续续写并截断，导致 JSON 未闭合。当前默认改用模型 checkpoint 的采样配置（此 snapshot 为 temperature=0.7、top_p=0.8、top_k=20），每类最多 8 个实例用于小测试；可通过 `--max-per-category` 调整。完全相同的 canonical 类别和九维框去重，不做近似去重。原始回答全部保留。

某一类输出不是完整 JSON 时，写入 `boxes_<category>.status.json`，该类不加入结构并继续后面的类别/图片。样本 `status.json` 的 partial 表示有失败、去重或数量限制，不能当作完整观测。不会补写截断框或自动反复采样到成功。更新后的 GPU 效果尚待用户重跑。

2026-10-10 类别与输出结构修复：`appliance → appliances`、`misc. → misc`，统一大小写、空格/连字符和可识别复数；原词表及 Adapter 类别 ID 不变。未知类别记录并跳过，不猜测映射。`categories.normalized.json` 保存实际查询类别、名称转换和跳过项。3D 回答支持单个对象、对象列表及 `objects`/`boxes` 包装；缺字段、null、非数值、非有限值、尺寸非正的框逐项跳过。无有效对象时状态为 `empty`。

CPU 验证：7 项回归测试通过，包含现有 `tensorize(max_objects=160)`；回放 run002 已保存回答得到 11 个框并识别 1 个截断回答，回放 run003 的 9 个框回答得到 15 个去重框。run003 的 `appliance` 和 `misc.` 已正确归一化。回放未生成此前崩溃后缺失的回答，也未运行 GPU 推理。可手动复查：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python -m unittest qwen_edit_pano.world_pipeline.completion.test_observer_responses -v
```

- `observe`：先识别可见类别，再逐类别查询 3D 框，输出 `observed.json` 和 `G_obs.json`。
- `complete`：加载官方 AR 权重，保留观测对象并续写，输出多个 seed 的 `G_full.json` 和俯视图。

首版直接测试官方预训练模型，不训练 Completion，也不运行 Qwen-Image-Edit。最终 `G_full.json` 使用现有 Encoder schema；接入生成器仍需独立的预测状态推理入口，旧 `oracle_inference.py` 只接受 GT Bundle。

## 1. Qwen3-VL：先运行这一步

### 直接测试现有验证集的 3 张图片（推荐）

已准备 `qwen_edit_pano/data/observer3d_val3_inputs.jsonl`：从当前 Oracle Bundle 中选 3 栋验证建筑，各取 1 张，只引用已存在的 local RGB，不重新下载数据。实际 Bundle 有 168 条 val、4 栋验证建筑，与旧交接中的 3 栋不同；本次只测试选定的 3 张。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -u -m qwen_edit_pano.world_pipeline.completion.observe \
  --inputs qwen_edit_pano/data/observer3d_val3_inputs.jsonl \
  --model /data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --decoding checkpoint --max-per-category 8 \
  --output qwen_edit_pano/outputs/observer3d_val3_run004
```

这条命令不需要填写图片路径，处理的只有清单里的 3 张图。以下单张入口用于以后自行选图。

### 自己指定单张图片

在项目根目录执行，使用现有 qwen3vl 环境及已缓存模型。将 `LOCAL_IMAGE` 改成要测试的局部图片。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export OBS_PY=/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python
export LOCAL_IMAGE=/绝对路径/你的局部图片.png

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=02:00:00 \
  "$OBS_PY" -u -m qwen_edit_pano.world_pipeline.completion.observe \
  --image "$LOCAL_IMAGE" \
  --model /data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --output qwen_edit_pano/outputs/observer3d_run001
```

上面已填写服务器实际缓存 snapshot 路径。脚本强制离线，并把 Hub ID 先解析成本地 snapshot 后再加载；不会请求远端 processor 配置。没有缓存时先自行下载。Slurm 分区和资源按实际机器调整，不指定训练正在占用的 GPU。

输出在 `qwen_edit_pano/outputs/observer3d_run001/<图片文件名去掉后缀>/`：

```text
local.jpg                   输入图片预览
categories.txt              可见类别原始回答
boxes_<category>.txt         每个类别的原始 3D 回答
*.prompt.txt                对应提示词
observed.json               原始框与转换后的 local 角点
G_obs.json                  当前 Encoder 格式
```

使用已有 Local 清单批量运行，把 `--image` 替换为：

```text
--inputs <现有 observer_inputs.jsonl 路径> --limit 10
```

清单沿用现有 `id / split / local_rgb` 字段；其他字段可存在，脚本只读取局部图片，不读 GT。省略 `--limit` 运行整个清单。单张任意图片输出中的 `split=val` 只是兼容字段，不代表它属于正式验证集。

## 2. M3DLayout：准备一次环境与官方权重

按用户选择，Observer 与 M3DLayout 共用 `qwen3vl`，不修改 `qwen360`。2026-10-10 已检查此环境的包元数据：torch 2.6.0、torchvision 0.21.0、transformers 4.57.1、NumPy 2.5.2；已有 matplotlib、PyYAML、einops、scipy；缺少 pytorch-fast-transformers 和 ninja。共用环境的实际模型运行尚未验证。

保留现有核心包，不运行官方整份 requirements，不执行之前的 Transformers 4.40.2 降级命令。以下由用户手动执行，只补缺失包：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
conda activate /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl
mkdir -p qwen_edit_pano/third_party
git clone https://github.com/Graphic-Kiliani/M3DLayout-code.git \
  qwen_edit_pano/third_party/M3DLayout-code

python -m pip install --no-deps ninja
MAX_JOBS=4 python -m pip install --no-deps --no-build-isolation pytorch-fast-transformers

PYTHONPATH=qwen_edit_pano/third_party/M3DLayout-code python -c "from transformers import Qwen3VLForConditionalGeneration; from scene_synthesis.networks.autoregressive import build_network; print('Qwen3-VL and M3DLayout imports OK')"
```

`--no-deps` 避免 pip 自动替换已有依赖；`--no-build-isolation` 使用当前 Torch 编译扩展。fast-transformers 需要 C++ 编译器，检测到 nvcc 时还会构建 CUDA 扩展。导入检查不加载权重、不申请 GPU；它通过也不代表整个 GPU 补全已验证。如果编译或导入失败，应按具体错误适配，不通过安装整份 requirements 解决。当前 NumPy 2.x / Transformers 4.57.1 与官方依赖组合有差异，保留后需要实际推理验证。

从[官方 README 的 Download Weights](https://github.com/Graphic-Kiliani/M3DLayout-code#download-weights)下载 AR 权重。官方提供 [Google Drive](https://drive.google.com/drive/folders/19GMTUk92fC6FB_dHu0aUw5Whc4j2KEWS) 和百度网盘入口。放到：

```text
qwen_edit_pano/third_party/M3DLayout-code/weights/autoregressive_59000.pth
```

不需要下载几 TB 的场景资产。官方网络还会加载 `bert-base-cased` 和可能的 torchvision 预训练参数；首次运行需能访问对应缓存/下载服务。源码通过 `gradio_demo.py` 的 `DATASET_STATS_AR` 读取官方归一化和类别顺序，不导入 Gradio 页面或渲染器。

## 3. 执行补全

继续在项目根目录，激活 `qwen3vl`，把 `OBSERVED_JSON` 改成第 1 步生成的文件。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
conda activate /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl
export OBSERVED_JSON=qwen_edit_pano/outputs/observer3d_run001/你的图片文件名/observed.json

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --qos=normal --time=02:00:00 \
  python -u -m qwen_edit_pano.world_pipeline.completion.complete \
  --observed "$OBSERVED_JSON" \
  --weights qwen_edit_pano/third_party/M3DLayout-code/weights/autoregressive_59000.pth \
  --camera-height 1.5 \
  --max-new-objects 32 \
  --max-objects 160 \
  --seeds 0 1 2 3 \
  --output qwen_edit_pano/outputs/completion_run001
```

`--max-objects` 要与正在训练 World 分支绑定的 `BUNDLE.json` 一致；已核实当前 `oracle_supplement_epoch006_v1` 使用 160，上述命令已按它设置。脚本未传参数时默认仍为 128。`--max-new-objects` 限制追加数量，已有对象也计入总容量。

默认只加一句来自 Qwen room type 的提示，不额外生成描述。要对比完全不用文本，在上述命令加 `--text ""`，并换一个 output。也可以显式传 `--text "The room is a living room."`。

每个 seed 保存：

```text
seed000/
  raw_normalized.json       M3DLayout 原始输出，包含 START/context/新 token
  G_full.json               observed + 新对象；当前 Encoder 字段
  layout.json               local 坐标的中心和完整角点
  layout_topdown.png         蓝色 observed，橙色 generated，黑三角相机
  summary.json               对象数量、END、跳过的无效或非目标类别
```

先看 `layout_topdown.png` 的新增对象是否合理，再看不同 seed 的差异。观测框在最终 G_full 中原样保留；M3DLayout 内部使用的 yaw-only 近似不会覆盖它们。

## 4. 首版采用的坐标假设

这是可运行的**预训练模型补全探测**，尚未做针对相机视野的补全训练。

- Qwen 原始框按 x 向右、y 向下、z 向前解释，转换到当前 local 的 y 向上。
- 默认角度是官方仓库回复中的归一化格式，乘 π 转成弧度。若所用模型实际输出弧度，observe 加 `--angle-unit radians`。不按每张图数值自动猜角度。
- 完整 9-DoF 角点先转 local。M3DLayout context 用覆盖这些角点的 yaw-only 框，保留原框供最终导出。
- 官方布局使用 y 向上；首版将相机放在 `[0, camera_height, 0]`，默认 1.5 米。`--scene-offset-xz X Z`、`--scene-yaw-deg A` 可显式改变布局中的相机位置和轴方向；转换输出时使用逆变换。
- 这项场景放置假设不等于恢复了原房间坐标或重力方向。明显倾斜的照片、错误相机高度或官方坐标先验可能影响补全效果。
- 默认尺寸按完整边长处理，对应官方 Matterport 预处理中的 full-length 记录和 demo dimensions；若你所用权重的数据约定为半边长，complete 显式加 `--size-convention half`。
- room mask 为固定全 1，不由 Qwen 生成。

类别桥位于 `geometry.py` 的 `CATEGORY_GROUPS`。它是可编辑的多对一映射，粗类别近似会影响效果。Qwen 观测中不在 M3DLayout 词表的类别不作为其 context，但仍保留在最终 graph；生成的 rug 等不属于当前 Encoder 对象集合的类别在 summary 留档，不强行改成家具。

必要处理只有 JSON 解析、类别/坐标转换、START/END 和无效框处理，没有预检审批链或大型 QC 流程。重跑同一 output 会覆盖同名文件；比较实验用不同目录。

## 5. 对应当前 Adapter 的字段

`G_full.json` 的每个对象包含：

```text
category_id.value
center_local_m.value
size_aabb_local_m.value
center_local_m.valid
size_aabb_local_m.valid
```

padding mask 由现有 `oracle_model.tensorize` 生成，不由模型预测。source、track_id 和 local_evidence 仅追踪来源；新生成对象的 local_evidence 为 unknown，不声称已证明在镜头外。

导出使用现有 `objects_local_aabb_masked_v1` 对应的数据表示，尺寸始终由 local 角点 max-min 计算。格式相容不等于预测结构正确，也不等于 ERP 已改善。本次代码只实现上游观察、补全与导出。

## 6. 客厅尺寸约定对照（half，2026-10-11）

直接复用已生成的客厅observed.json，不重跑Qwen3-VL。官方YAML保持text_condition=true；仅在无文本组传空字符串。现有complete.py的half分支在归一化之前将context完整边长除2，在反归一化之后将生成尺寸乘2，中心、角度和官方bounds不缩放；最终原始观测对象保持不变。全局默认仍为full，本次是验证half假设的独立实验。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/completion/run_size_ablation.sh
```

顺序运行half+房间类型文本、half+无文本，固定同一客厅样本、seed0、相机高度1.5、追加预算32、Adapter容量160。分别写completion_indoor_halftext_run001和completion_indoor_halfnotext_run001。命令失败则停止；可分别用脚本参数text或notext单独运行。重复运行某组会覆盖该组同名文件，旧full实验不覆盖。

两组都完成后，在项目根目录执行CPU汇总：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -m qwen_edit_pano.world_pipeline.completion.summarize_size_ablation
```

汇总同时读取已有full+text/full+notext，以及新half两组，打印新增对象的完整local AABB尺寸和相对假定地面的底部高度。该高度不是GT，挂墙/支撑对象不应要求落地；同时查看布局与类别合理性。

验证：用CPU替代网络返回运行实际complete.main，full/half两路径的context尺寸、生成尺寸、观测对象逐字段保留、tensorize兼容均通过。用参数记录器替代srun验证两组命令及空文本参数正确传递；报告流程通过。未加载真实模型、未启动GPU、未验证half组生成效果。
