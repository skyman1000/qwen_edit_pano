# 仅局部可见观测：执行与验收

用户已停止扩展“直接RGB生成结构化完整世界”的对照。该历史实验保留，不再补跑。本实验仅检查感知入口：

```text
局部 RGB + HFOV
    → Qwen 可见实例、类别、2D 框
    → WildDet3D 对这些实例预测3D框
    → G_obs 与仅观测可视化
```

没有隐藏程序提案、H/A/V概率估计、世界库存采样、补全布局求解或API调用；无需 `SILICONFLOW_API_KEY`。这不是“无因果完整世界”的新基线，而是可见状态提取的诊断实验。

## 查看已有五图观测

实际已生成：`outputs/local_observation_review_20261028_001/index.html`。

来源：原数值恢复批中的休息区、客厅、卫生间、卧室，以及最新 `kitchen_observer_20261028_144049` 的厨房。二维/三维观测数量分别为16/16、6/6、7/7、8/8、3/3。厨房仅识别吧凳的漏检问题原样展示，没有加入中岛等人工补造观测。

重绘到新目录，纯CPU，无新模型调用：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/local_observation.sh report \
  --source qwen_edit_pano/outputs/organized_numeric_recovery_20261028_001 \
  --replacement qwen_edit_pano/outputs/kitchen_observer_20261028_144049 \
  --out qwen_edit_pano/outputs/local_observation_review_20261028_002
```

`--out`必须是不存在的新目录。已有001无需重跑。`--replacement`只明确替换同ID、同原图像素、同HFOV的观测，不按物体数量或结果好坏挑选。

报告只读取 `observation/` 中的识别与三维结果，以及 `suite.json`，不读取 `completion/` 或从完整世界筛选蓝色对象。`G_2d.json`、`G_obs.json`原字节复制，所有对象须为预测的 `scope=observed`、`local_evidence=visible`，且2D与3D实例ID、类别和二维框对应。遇到full/hidden/GT状态会拒绝，不静默过滤。

## 重新提取五张图，只运行观察器

由用户执行GPU任务：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
bash qwen_edit_pano/scripts/local_observation.sh run GPU3
```

默认项目的 `data/observer3d_indoor_val5_inputs.jsonl` 五图。自动先Qwen二维识别，再使用独立WildDet3D环境估计三维，只生成观测结果与报告。没有API Key要求，不进入任何补全阶段。

只测厨房：

```bash
CAUPANO_TEST_INPUTS=qwen_edit_pano/data/observer3d_kitchen_retest_inputs_20261028.jsonl \
  bash qwen_edit_pano/scripts/local_observation.sh run GPU3
```

自选新输出目录：`run GPU3 qwen_edit_pano/outputs/local_observation_new_name`。已有运行可 `resume GPU3 OBSERVATION_ONLY_ROOT`，只接受这个新入口创建的观察器批次，源码绑定检查保留，已完成/失败的阶段不随机重试到成功。

`prepare GPU3 [NEW_ROOT]`只做CPU元数据准备，没有真实推断。`run`的真实视觉效果仍由实际模型决定，本次没有修改二维识别算法，不能承诺重跑会消除厨房漏检。

## 输出及验收

入口打印两个路径：观测数据根目录和独立报告目录 `..._review_时间_PID/index.html`。报告与来源分开，避免改动来源观测。

- `G_2d.json`：类别、ID、二维可见框。
- `G_obs.json`：对应对象的预测相机中心和全尺寸AABB；不冒充真实测量。
- 原局部图、编号+类别二维框、观测3D重投影、XZ平面、静态/交互3D、六方向、2:1 ERP状态投影。
- 全部物体为蓝色可见预测；补全新增数恒为0。周围未知空间留空，不生成隐藏纹理。
- `instances.csv/json`：每个对象的类别、二维框、中心、尺寸和落地预测。
- 若没有完整3D结果，只展示真实二维框和未知三维字段，不填零坐标或制造三维图。
- `validation.json`：来源文件哈希、零新增、未读取补全文件、标签与链接检查。

先看原图与二维框是否漏检/重复，再看三维重投影和尺度是否吻合。两者数量相同只证明保留了相同ID，不证明识别完整或三维准确。相机重力、地面高度和真实遮挡仍未独立验证。

## 代码检查

```bash
bash qwen_edit_pano/scripts/local_observation.sh test
```

测试使用真实保存状态作为结构样例及明确标记的临时绘图stub，覆盖拒绝隐藏/GT/缺失几何、2D/3D身份绑定、只读观测原字节保留、缺3D不造坐标、替换图像匹配，以及无Key入口只启动2D/3D/报告。测试不调用真实模型。

## 第二批：新增五张局部图

清单：`data/observer3d_local_extra5_inputs_20261028.jsonl`。输入预览：`outputs/local_observation_extra5_inputs_20261028_001/index.html`。

1. `X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e`：用户指定的灰色双沙发、白色茶几、金属落地灯客厅。
2. `zsNo4HB9uLZ_ed8645386a0a4de2bfc486e0a1d02468`：白沙发、抱枕、装饰画客厅（原数据曾误标卧室，以原图为准）。
3. `8194nk5LbLH_9bdde31adaa1443bb206b09bfa3c474c`：透过玻璃可见的健身设备及楼梯，检查复杂可见性和部分遮挡。
4. `x8F5xyUWy9e_39d82699c9a44bb9b5349d52faf1464d`：桑拿房和木质坐凳。
5. `x8F5xyUWy9e_03de84eb12c24a93bdfd88e46a6db25a`：浴缸区的局部视角，包含镜面和遮挡。

五图均为项目现有val局部RGB，1024×1024、HFOV=90°；ID与像素均不重复上一批五图。仅用RGB、相机参数和ID/哈希输入观察器，不传上述人工选图说明，不读GT状态。预览是原图，不是检测结果。灰沙发图是早期研究使用过的开发图，此处按用户要求加入，不称为从未见过的独立留出测试。

执行这批五图：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
unset CAUPANO_PROGRAM_CATALOG
CAUPANO_TEST_INPUTS=qwen_edit_pano/data/observer3d_local_extra5_inputs_20261028.jsonl \
  bash qwen_edit_pano/scripts/local_observation.sh run GPU3
```

仍是Qwen识别可见实例、WildDet3D估计三维的同一感知配置，不调用API、不做隐藏补全。不改变原观察器和因果模型，也不复用历史识别结果代替这次提取。

本批输入已实际做CPU prepare-only及源码/输入核验，保存在 `outputs/local_observation_extra5_preflight_20261028_001`，尚未运行真实GPU。上述run命令会创建新批次并在结束时打印独立可视化index.html。核验记录在输入预览目录的 `validation.json`。
