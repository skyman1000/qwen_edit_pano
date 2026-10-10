# 保留成功样本，补充新建筑

> 2026-10-10 最新执行决定：用户明确要求随机抽查后纳入全部496条仅像素阈值标记样本。已随机检查6条（seed=20261010），本次实验使用934条无标记＋496条阈值标记，共1262 train / 168 val；另53条多视角不一致排除。`run_gt_supplement.sh bundle` 已接入 `gt_supplement_experiment_selection_v1.json`，可直接运行，不再要求下文所述的逐条 approved 流程。原 review_decisions、bridge、GT 和 flags 均保留；这属于用户授权的抽查实验，不是整批逐条视觉认证。后文“验收和训练”的人工验收步骤为此前默认流程，当前以本段为准。

已核对 repaired BUILD_RESULT：1196 train / 164 val 源状态成功，94 train / 10 val 失败；17 栋都有成功样本。退出码 1 是部分质量失败，不是整批丢失。保留成功的修复产物及其 provenance；不重试旧失败 ID。

新增官方 train 的 759xd9YjKW5（61）、EDJbREhghzL（69），官方 val 的 8194nk5LbLH（19）。按候选数量补缺并留余量，不根据模型生成效果选建筑，也不声称新数据更高质量。候选合计 1326 train / 183 val，16 train / 4 val 建筑；最终成功和验收数量可能更少，不承诺恰好达到原 1290 / 174。

已生成 paired_gt_repaired_retained_v1（1360 个成功旧源）和 paired_gt_supplement_v1（1509 候选），并校验：官方建筑划分、ID唯一、无旧104失败ID、保留所有1360成功ID、新增130/19。所有中间目录独立；不要删除旧源目录，软链接仍依赖它们。

## 用户执行顺序

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh plan

srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh download

srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh build
```

只下载新增三栋的 8 类 GT ZIP，复用已有 skybox。下载沿用原下载器的校验/重试/已有文件处理行为。build 复用旧成功源阶段，新建筑使用现有严格几何检查和 lean catalog，不新增 Encoder 字段，不自动进行修复。build 中断后执行同一命令；完成后质量失败写入 BUILD_RESULT，不作为整批异常退出。真正的签名/程序错误仍会报错；不要忽略任意非零退出。已记录失败项不再重算。

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.world_pipeline.build_status \
  --root qwen_edit_pano/data/gt_sources_supplement_v1
```

整批 build 结束、BUILD_RESULT.json 已生成后，创建成功子集（只执行一次）：

```bash
srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --qos=normal --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh select

srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh export
```

导出自动判断已有目录并加 --resume，使用全部成功清单（train/val limit 0），产物 gt_world_supplement_v1。仍可能出现导出坐标/投影质量 flags；源构建成功不能代替它们。不要直接沿用旧 run_gt_14train3val.sh 的 export/bundle/train 路径。

## 验收和训练

检查 gt_world_supplement_v1/EXPORT_COMPLETE.json、gt_manifest.jsonl 和投影预览。review_decisions.json 初始 pending。实际查看后，可用已有命令逐样本或重复 --id 记录；不要把占位 ID 原样执行，不要自动全批准：

```bash
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python \
  -m qwen_edit_pano.world_pipeline.review_gt \
  --root qwen_edit_pano/data/gt_world_supplement_v1 \
  --id 实际已检查的样本ID --decision approved \
  --notes '填写实际检查到的对齐、实例投影和坐标情况'
```

有 flags 的样本保持隔离。完成验收后：

```bash
srun -p debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --qos=normal --time=01:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh bundle

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh train-full

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh train-constant
```

冻结 epoch006 Panorama LoRA 和基础模型，只训练 Encoder/Adapter。full 使用真实完整GT；constant为同容量恒定条件对照。同为1000 optimizer steps、accumulation4、lr1e-4、warmup100、seed0，每100步保存。不是训练新 Panorama LoRA，也不是以两步 smoke 作为正式初始化。沿用原全量 paired/cache 绑定，不把新的 GT 候选清单冒充原 LoRA 训练清单。

GPU --mem=64G 是主机内存，不是显存；选择之前成功跑 oracle 的GPU节点，可加 --nodelist=对应节点。时限内未完成时，脚本支持 `train-full 已保存的checkpoint 新输出目录`（constant 同理），不要误认为重复首次命令会自动恢复训练。

同一验证集评估：

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh eval-full

srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_supplement.sh eval-constant
```

full评估 baseline/correct/shuffled；constant评估correct（固定条件），使用28步CFG4和相同seed。limit183覆盖本清单最多183个val候选，实际取已验收bundle中的val。恢复训练换了输出目录时，eval-full/eval-constant 同样接受 `checkpoint 新评估输出目录`。

当前只执行了清单生成/校验、重复 plan 幂等检查和脚本语法检查；没有下载、新建筑构建、导出或GPU运行。

2026-10-10 打包容量：入选样本最大151个对象，8条超过128；本次 bundle 使用 --max-objects 160，不截断对象，不改变特征字段。
