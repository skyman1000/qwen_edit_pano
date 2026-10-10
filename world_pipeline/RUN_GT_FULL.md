# 补齐 GT 原始资产、派生状态和可恢复导出

2026-10-06 已核实：旧数据来自 `qwen_pano.caupano.tools.prepare_scan`、`build_scan_dataset`，底层复用 camera/house/region/object_link/canonical、ERP alignment、geometry、world_state_builder。旧链确实已经实现 GT 构建；之前仅构建了部分样本，并非整套7521训练数据都有状态。

当前 paired_full_v1 为7521 train / 925 val，72栋建筑。标准raw目录的9类依赖均齐全的目标建筑为6栋；另66栋缺少GT包。磁盘已有的第7栋GT建筑属于其他划分。本轮不下载、不构建test建筑。存在性统计不等于ZIP CRC或几何审核通过。

新增 `expand_gt_sources.py` 按当前官方配对清单遍历全部样本，直接调用原CPU解析/构建模块，不使用旧caption或旧split作为入选条件。旧模块不修改，旧输出只读复用；新文件在 `qwen_edit_pano/data/gt_sources_full_v1`。

原canonical index会读取undistorted color/depth/normal的ZIP成员，geometry还使用实测depth、color作投影核验。因此原链不是只下载house和region两个包就能跑通。以下明确下载8类缺少的配套资产；已有skybox保持复用。数据体量可能较大，这是保留原核验链的实际依赖，不是重新下载已完成的skybox。暂未实现删减原始模态的轻量派生链。

## 执行前

**当前存储阻塞：2026-10-06实测项目文件系统只余约7.6GiB。先释放足够空间或确定新的存储位置，再执行下面的download/build/export。** 现有32个导出目录约0.229GiB，粗略按同等大小推算8446条仅最终导出就约60GiB（未含原始包、geometry、mesh、中断备份，且样本大小会变）。原始包总需求未测得，不能把60GiB当作全部预算。助手没有删除旧输出/缓存，也没有执行下载。

不要让旧版 `prepare_gt` 作业和新版本同时写同一输出目录。旧版没有写锁，新增锁无法约束已启动的旧进程。若旧作业仍在运行，先让其结束，或自行停止它再重启。新全量目录与 `gt_world_expanded_v1` 分开，旧结果不删除。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/world_pipeline/run_gt_full.sh plan
```

助手已执行上述CPU清点，生成 `gt_sources_full_v1/inventory.json`、`scans.txt`（72栋）、`missing_scans.txt`（66栋）。只清点，不下载、不生成mesh。

## 1. 用户手动补下载

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --mem=8G --qos=normal --time=24:00:00 --pty \
  bash qwen_edit_pano/world_pipeline/run_gt_full.sh download
```

脚本调用现有download_mp_py3.py新增的 `--scan-list`，仅遍历当前72栋train/val建筑。保留下载器原条款确认，按终端提示自行确认。已有ZIP检查CRC后跳过；未完成.part按服务端Range支持情况续传，服务器不支持时该文件重新下载。超时/中断后重跑相同命令。没有下载任务由助手启动。

## 2. 用户手动派生完整源状态（CPU，无GPU）

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_full.sh build
```

这一步以7521/925全量配对身份为目标，不再按旧world_state是否存在过滤。旧的合格canonical与每视点阶段用只读符号链接复用，缺少的调用原模块生成。不会向旧链接路径调用写入模块。

恢复粒度：每栋建筑的canonical解析链、每视点的alignment/geometry/world阶段。完成阶段写校验记录；同命令重跑校验后跳过。中断阶段保留为`.interrupted-*`后重做，不能在一次Open3D投影内部按射线继续。配置/代码/完成产物被改动会拒绝混用。原ZIP恢复检查使用size/mtime，派生产物使用SHA256；不能把这些检查说成对所有原ZIP内容再次做SHA256。

查看 `build_progress.json`、`logs/`、`BUILD_RESULT.json`。缺原始文件或几何核验失败会记录、继续其他样本，最终以非零退出；不会把失败样本自动放行。`all_requested_built=true`只表示阶段构建完整，不是GT视觉认证。24小时是每次申请预算，不承诺全量在一天内完成。

## 3. 用户手动导出当前相机坐标下的 GT

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=24:00:00 \
  bash qwen_edit_pano/world_pipeline/run_gt_full.sh export
```

仅源状态全量构建成功后启动，读取新world-root，写新 `qwen_edit_pano/data/gt_world_full_v1`。若仍缺身份，`--require-all-pairs`拒绝把部分导出伪装成全量。第一次自动新建，后续同命令自动带`--resume`。每完成一个样本有DATA_COMPLETE.json和文件哈希，清单原子更新；恢复会校验并跳过完整样本，保留再重建未完成样本。完整样本的review_decisions保留，重新构建的样本需要重新审核。

该阶段不改变类别、坐标、尺寸及可见性阈值，不删除既有告警。数值flags仍保留供后续核验；全量导出不等于全部可训练。

## 可选：只继续旧251/39导出

此命令只是兼容恢复旧任务，不补齐全量源状态；切勿与上述新全量导出混淆：

```bash
srun -p debug --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=48G --qos=normal --time=04:00:00 \
  /data-nfs/gpu1-2/u13529658780/.conda/envs/qwen360/bin/python -u \
  -m qwen_edit_pano.world_pipeline.prepare_gt \
  --pairs qwen_edit_pano/data/paired_full_v1 \
  --train-limit 0 --val-limit 0 --threads 4 --resume \
  --output qwen_edit_pano/data/gt_world_expanded_v1
```

支持已核对的旧版导出脚本哈希迁移。旧样本需通过图结构哈希、来源哈希、图像/NPZ可读性和身份核验才转为可复用checkpoint。不能在同一export目录切换world-root、样本集合或数值协议；因此全量使用新目录。

## 已验证与未执行

CPU回归覆盖：真实旧阶段复用、旧导出迁移、中断后恢复、未完成目录保留、文件损坏拒绝、并发写锁、已有审核记录保留、下载scan-list限定且不访问网络；原几何/Observer测试回归通过。下载器原HTTP Range/CRC实现保持不变。没有执行全量下载/新建筑mesh构建/全量导出，也没有启动GPU任务。

完成后再汇总train/val建筑覆盖、构建失败、投影告警、对象数量，建立审核后的oracle bundle；原Panorama LoRA/Encoder训练代码和正在训练的缓存不改动。
