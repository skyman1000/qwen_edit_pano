# 五图恢复批次：坐标格式与 API 429 修复

核对来源：`outputs/organized_multiscene_recovery_20261027_225835_2897903`。这次根据实际原始回复定位故障，没有修改 v13 的因果图、概率估计、库存采样、尺寸机制或物理求解器。

## 实际发生了什么

| 输入 | Qwen 隐藏库存 | 可直接保留的合法坐标 | API |
|---|---:|---:|---|
| 厨房 `8194…ae915…` | 33 | 7 | 机制阶段 HTTP 429；已有 2 份有效回答缓存 |
| 休息区 `8194…c9e8…` | 28 | 3 | 机制阶段 HTTP 429；已有 52 份有效回答缓存 |
| 客厅 `X7…2bb493…` | 已完成 21 | 全部复用 | 已完成 20 |
| 卫生间 `x8…3a127…` | 13 | 7 | 已完成 13 |
| 卧室 `zs…ead481…` | 23 | 9 | 机制阶段 HTTP 429；尚无有效回答缓存 |

数字是已经采样保存的隐藏物体数，不是本次修复新生成的实验结果。Qwen 的新一轮回复不是全 null，而是把不同宿主类型的坐标行写成同样长度：根物体要求 5 列、top 附着要求 4 列、shelf/seat 要求 5 列，有的回复却分别给了 5、5、6 列。旧验证报错把长度与非有限数值混在一句中，导致问题表象不清晰。不能简单删除最后一列，因为无法确定模型表达的是宿主 UV 还是绝对 XYZ。

API 的三个场景失败于机制估计，后续缺少 API 机制文件及世界是依赖阻断，不是生成了零物体。`srun ... exit code 1` 是汇总阶段返回未完成状态，本次证据没有指向 GPU 硬件错误。[硅基流动官方限流说明](https://docs.siliconflow.cn/docs/userguide/faqs/rate-limit-and-upgradation)将 HTTP 429 定义为限流，建议客户端控制频率并退避重试；限制按账号及模型计算，可能涉及分钟、小时或日配额。

## 修复行为

新增 `world_pipeline/completion/assembly_named_geometry.py`：

- 坐标回复使用 `roots`、`hosted` 和具名字段 `root_offset_m`、`host_offset_uv`、`surface_fraction` 等，不再用不同长度的紧凑行；不提供 null 数值模板。
- 从原始回复和格式纠正回复逐对象核验，保留首个无歧义的合法整行及已合法组距离。不会截列、猜测缺失数字或覆盖已合法坐标。
- 只请求缺失/无效对象的坐标；一次回复中合法对象立即保存，下一轮只问剩余对象。每组最多两轮，每轮最多一次 JSON/成员格式纠正。耗尽仍失败，保留进度，不无限重抽。
- 原观测、语义库存、尺寸、宿主、相机区域、机制概率不变；仍由原有联合布局与数值编译器处理全部物体。物理数值未通过时保留完整候选，不删除物体换取成功。
- 保留的是模型已合法的原始坐标字段；原物理编译器依然可能按既定规则联合调整最终位置。这里没有新增坐标锁定或取消原求解步骤。
- 保存 `named_numeric_progress_v2.json`、`named_numeric_stage_v2.json`、`named_recovery_calls_v2/` 与源字段路径。旧版已经做过 compact recovery 的卫生间也可进入这一独立、有上限的恢复阶段。

新增 `assembly_api_pacing.py`，仅主线五图入口使用，原 `world_api.py` 和独立 RGB 直接生成基线未改：

- 默认请求起始间隔 3 秒；使用相同 Key、模型、地址且接入此封装的进程共享间隔和冷却时间。其它 Key 也可能共用账号配额，客户端无法自动识别账号关联。
- 只重试被 HTTP 429 拒绝的请求，最多追加 3 次，默认退避 20/40/80 秒，并尊重 `Retry-After`；每个逻辑请求累计等待上限 300 秒。超过等待预算直接记录失败，不提前越过服务端要求。
- 超时、断流及模型 JSON 错误不进入此传输重试；格式纠正仍是原有独立、有限的机制。
- 每次网络尝试独立留档，成功内容发布到原有逻辑缓存路径；已有 `.accepted.json` 验证通过后直接读取，不重新抽取有效机制回答。不记录 Key 或服务器错误正文。
- 本次恢复复制到新目录，原目录不改，3 个成功世界逐字节复用。失败阶段重新进入；完整观测、提案、Qwen CPT 等继续复用。

## 执行

从项目工作区执行。环境中仍需 `SILICONFLOW_API_KEY`，模型固定为硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`。

本次已经完成 CPU 准备的恢复目录是 `outputs/organized_multiscene_named_prepared_20261028_001`，可直接执行：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
bash qwen_edit_pano/scripts/organized_multiscene.sh resume GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001
```

这个新目录已经绑定修复后的代码，且有 `presentation_001/index.html` 展示原有三个成功世界。这里只做了复制、核验与可视化，没有新模型调用。完整运行后脚本会再生成新展示页面。

如需自行从原始失败批次重新建立另一个恢复目录，使用以下命令：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
unset CAUPANO_RECOVERY_OUTPUT
export CAUPANO_API_MIN_INTERVAL=3
export CAUPANO_API_429_RETRIES=3
export CAUPANO_API_429_BACKOFF=20
export CAUPANO_API_WAIT_BUDGET=300

bash qwen_edit_pano/scripts/organized_multiscene.sh repair-generate GPU3 \
  qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903
```

使用 `repair-generate` 生成新恢复目录，不要直接 `resume` 那个旧目录：旧目录绑定旧源码且保留了失败阶段标记。脚本末尾打印新目录及 `index.html`；页面包含原局部图、平面图、局部投影、六向投影、静态与交互 3D、2:1 ERP。编号+类别、观测蓝色/新增可映射橙色/新增不可映射红色沿用既定显示规则。

如同时运行直接生成基线等任务，建议错开同一硅基流动账号/模型的 API 阶段。持续 429 仍需查看平台实际配额；这段退避不会消除小时或日额度限制。

仅重建页面时：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh report 新恢复目录
```

CPU 测试入口：

```bash
bash qwen_edit_pano/scripts/organized_multiscene.sh test
```

实际失败审计与验证保存在 `../research/audits/multiscene_named_recovery_20261028/release_001/`。32 项 CPU 测试全部通过（198.323 秒），其中四个实际失败场景均完成了人工回复恢复与导出链路回放，并核验原始来源 JSON 未变。CPU 人工回复回放只验证接口、缓存、保留规则和导出链路，不证明真实 Qwen/API 已修复全部数值或语义问题。真正的生成质量需要上面的用户执行结果再评估。
