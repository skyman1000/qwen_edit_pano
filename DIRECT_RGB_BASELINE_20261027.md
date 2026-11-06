# RGB直接生成结构化世界：独立对照实验

这个对照可行，也有必要补充：直接让冻结 Qwen/API 从原局部 RGB 预测一个完整周围世界，与当前完整流程比较。它是**无显式功能因果模块、无 WildDet3D 的 RGB→世界状态基线**，不是把已有 G_obs 或候选程序换个提示重新生成。

## 定义与科学解释

```text
直接法：原局部RGB + 相机HFOV → 冻结模型一次联合预测 → 结构化世界 → 只读诊断与可视化

完整流程：RGB → 可见实例/官方WildDet3D → 固定O+
        → 功能H/A/V/R机制与库存 → 尺寸/位置 → 数值编译 → 世界状态
```

直接法不调用可见实例观察器、WildDet3D、场景程序库、显式 SCM、CPT 估计或数值布局求解器。模型自身仍包含预训练常识，不能称作“没有任何隐式因果知识”。输出也不需要功能假设、长解释或概率表。

直接模型同时估计可见与视野外实例。因此其蓝色框是它**自己声称可见**的对象，没有独立 G_obs 来保证观测保持。若把主线 G_obs 喂给它，就不再是本次要求的 RGB-only 基线。

这个实验同时改变了观察器、显式功能机制、几何来源、求解与调用预算，能比较**整条方法链与直接预测**。不能只凭它更差就说“因果模块有效”。因果部分的单独作用仍须结合固定 O+/D 下的 H 断边或其他匹配消融分析。

## 直接运行五张图

确认 `SILICONFLOW_API_KEY` 已设置，执行：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

bash qwen_edit_pano/scripts/direct_rgb_world.sh run GPU3
```

默认读取项目五图 `data/observer3d_indoor_val5_inputs.jsonl`：公共休息区、客厅、卫生间、卧室局部、厨房。每张图每个后端输出**一个完整世界**，总计划10份。无需 WildDet3D Python 环境或检查点。

两后端：

- 本地冻结 Qwen3-VL-8B-Instruct。
- 硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`，`https://api.siliconflow.cn/v1`。

共享原图、相机 HFOV、提示、输出格式和请求解码配置：temperature=.4、top_p=.9、top_k=50、8192输出token预算。图像通过已有统一 JPEG92/max1536 接口发送，记录 canonical_image_sha256。API 的请求含真实 image_url；不支持图像或发生服务错误时保留失败，不改用文本观测偷换输入。本地种子记录，服务端 API 不传未受支持的 seed；两套后端内部视觉编码/采样无法宣称完全相同。

每图每后端一个初始请求，最多一次格式纠正；默认共10个初始请求、至多20个请求。无传输自动重试，不为追求更好的几何重抽样。多次 API/格式失败也不变成“空世界”。

脚本末尾打印 `outputs/direct_rgb_world_时间_PID/index.html`。不用等待或调用原因果实验，即可独立生成和查看基线。

## 与当前五图完整流程并排比较

现有五图完整流程目录可作为参照：

```text
qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903
```

如果希望输入清单自动绑定这个目录，先设置：

```bash
export CAUPANO_DIRECT_REFERENCE_SUITE=qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903
bash qwen_edit_pano/scripts/direct_rgb_world.sh run GPU3
```

**生成时只从 reference 的 suite.json 读取 RGB 路径、ID、HFOV 和哈希，不读取 G_obs、隐藏库存、D 或 CPT。** 脚本结束后会做只读比较。如果完整流程尚未完成，表内保留未运行/失败；待它结束后再做最终比较，不用重跑直接模型。

已生成基线后，单独比较：

```bash
bash qwen_edit_pano/scripts/direct_rgb_world.sh report \
  qwen_edit_pano/outputs/这里替换为实际direct_rgb_world目录 \
  qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903
```

比较先检查局部RGB像素、尺寸和HFOV相同，拒绝把不同视角混为方法对比。要求完整流程是每图一个输出，不在多个候选中挑选最好的参照。共同输入以外的新图会明确标为不在参照清单，而非暗中匹配。

每个输入内，两种方法、两个后端的平面图/3D使用共同坐标范围。完整流程有数值诊断候选但未导出 G_full 时，也可展示该候选，并显式标为“未完成世界”，不算作通过。

若参考方法仍在写入文件，报告会标记快照不稳定；等待生成完成再重复只读报告即可。报告本身不调用模型、不写入因果实验源数据。

## 输出合同和后处理边界

模型输出一个 JSON：

```text
objects: 每行11列
[id, name, category, evidence, cx, cy, cz, width, height, depth, yaw_deg]

evidence: visible / outside_view
relations: [source_id, on|inside|attached_to|near, target_id]，没有声明则空数组
```

关系是模型直接给出的描述性空间/支撑关系，不是额外运行的因果机制。

坐标原点为输入相机，X右、Y图像上、Z前，单位米；尺寸为固有全尺寸，yaw绕相机Y。为兼容现有显示，按 yaw 确定性转换为相机 AABB：

\[
s_x=|\cos\psi|w+|\sin\psi|d,\quad s_y=h,\quad
s_z=|\sin\psi|w+|\cos\psi|d.
\]

这只是表示转换。模型原中心、固有尺寸、yaw、身份和成员原样保留，不进行位置优化、尺寸缩放、裁到相机外、语义删减或缺坐标补零。固定128个对象只是运算容量，无最少数或后方配额，也不因未映射就丢对象。类别映射使用与主线新增对象一致的 `semantic_lexicon_v1`。

只把非有限/非正尺寸、无效字段、重复ID、未知关系端点等当作结构错误纠正。几何重叠、视野冲突或不合理尺度保留为直接模型的表现，不偷偷修复。类型合法不等于世界合理；空输出也单独报告。

## 可视化及指标

沿用现有三色和编号＋类别标签：观测声明蓝，可映射新增橙，未映射新增红。

- 原局部RGB。
- 世界状态XZ平面图。
- 原视角重投影和六方向透视投影。
- 静态3D及可旋转HTML三维图。
- 1600×800 ERP状态投影、局部RGB-ERP投影及完整框叠加。
- 汇总大图、标签表、metrics.csv及逐框诊断。

灰色 ERP 区域未观测，没有生成隐藏 RGB。三维图是状态框，不是恢复出的真实网格或建筑。

两条方法统一计算：可见声明/隐藏数量、后半空间及中央后方数量、声称隐藏但与原视锥相交、声称可见却完全在视锥外、相机落入框、包含隐藏对象的AABB重叠、声明宿主对重叠、非声明宿主重叠、未映射类别、最大水平中心距离。

这些是共同的**只读代理诊断**。AABB重叠不等于真实碰撞；模型声明的宿主关系不证明物理支撑正确。直接法没有固定O+，不能报告与主线相同定义的“观测保持率”。未使用GT或盲评时，也不能把数量多、导出多或框不重叠称作语义准确率。

主线的物理代理检查通过门槛与直接法的格式门槛不同，报告分别显示其原状态，不把两种导出状态合并成同一“成功率”。调用预算也不同；需要同时考虑效果、失败模式和成本。

## 准备、继续及测试

只准备，不需要 Key，也不加载模型：

```bash
bash qwen_edit_pano/scripts/direct_rgb_world.sh prepare GPU3
```

继续未尝试输入：

```bash
bash qwen_edit_pano/scripts/direct_rgb_world.sh resume GPU3 \
  qwen_edit_pano/outputs/这里替换为实际direct_rgb_world目录
```

已完成或失败的输入不自动追加抽样。中断且无完成记录的尝试也保留为中断失败，不消失后偷偷重跑。

CPU测试：

```bash
bash qwen_edit_pano/scripts/direct_rgb_world.sh test
```

代码入口：`world_pipeline/completion/direct_rgb_world.py`；只读比较：`direct_rgb_review.py`。本次只增加独立文件，没有修改已有因果/恢复生成代码，正在执行的原实验不受影响。

验证覆盖：不依赖WildDet预检；不调用观察、因果或优化函数；两后端主提示与图像相同；API请求真正携带图像且日志不存Key；几何错误不被修复/删除；类别未知对象保留；固定格式重试预算；中断及失败不重抽；参照只能给RGB元数据；比较RGB/HFOV一致且来源不变；全部可视化链接。记录在 `../research/audits/direct_rgb_baseline_20261027/release_003/`。

真实直接法 GPU/API 尚未由本次实现运行，不能把CPU合成测试当成实验结果。新实验应与已完成的完整流程按同图、同坐标和相同指标对照后，再判断效果。
