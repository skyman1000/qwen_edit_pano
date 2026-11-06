# 实际产物审计：qwen_world_simple_run001

依据：outputs/qwen_world_simple_run001/review/metrics.json、12份Observer状态、12份hidden_plan.txt/proposal.txt、9份成功G_full；与qwen_upstream_val12_run003/review_pred/metrics.json比较。2026-10-13。没有运行GPU推理。

## 结果

|指标|旧run003|simple_run001|
|---|---:|---:|
|2D实例|189|189（复用）|
|有效格式3D框|188|184|
|复制旧数值示例的框|74|0|
|同类别2D IoU≥0.25匹配|54/150 GT|54/150 GT|
|2D匹配后的中心误差均值|1.608m|1.409m|
|2D匹配后的尺寸MAE|0.389m|0.412m|
|同类别3D AABB IoU≥0.25匹配|3|2|
|隐藏候选|36次全部为空|9/12成功，47个对象|

中心误差均值只覆盖有有效3D的2D匹配对，旧新覆盖不同，不是严格同集合改善证明。GT对象清单有筛选，未匹配2D实例不能全部认定为幻觉。

9份G_full均原样保留输入G_obs。47个新增对象中心全部位于输入90度相机视锥内，0个在相机背后；按实际图片宽高、HFOV和Local坐标计算。中心位于视锥内不等于物体可见，遮挡需要其他证据；当前程序没有提供遮挡验证，因此这些结果没有证明视野外补全。

具体例子：zsNo4HB9uLZ_38e0c09ac7a748dbadea6471861b30c3的hidden_plan要求在可见墙面增加画、灯和植物；原图墙面可见且基本空白。输出植物y=0.95，所谓高处灯具y=-0.45，与Local y向上的约定存在语义矛盾。x8F5xyUWy9e_39d82699c9a44bb9b5349d52faf1464d的towel与mirror有完全相同中心和尺寸；不能用JSON成功认定布局合理。

成功候选与GT隐藏框的同类3D IoU匹配为0/47；分母270个GT仅覆盖9个成功样本，不能当作全部12张的指标。GT是一个真实实现，不是所有合理假设的唯一答案；应同时评价可见证据一致性和几何合理性。

## 原因分类

1. 任务提示偏离：语义阶段“design/useful additions”诱导装修；保护对象列表没有保护空墙和空地。语义阶段缺少相机信息和输出类别集合。责任在此前任务设计，不只是模型不听话。
2. 三次导出失败分别为flooring、thermostat、rug不在既定词表。不能任意映射来凑通过，更不能把rug加入当前Adapter类别接口。新语义阶段直接使用同一允许词表；仍保留失败记录。
3. 五个3D失败来自一实例查询返回多个框或嵌套多框。此前报TypeError不清晰，现改为明确歧义错误；不擅自取第一个框。继续标未知。
4. 原生camera y向下到Local y向上的转换已有测试；隐藏阶段直接输出Local坐标，不应再全局翻转y。文本与数值矛盾不能靠无依据符号翻转修复。
5. 格式有效、observed_preserved只保证数据接口，不保证物理、可见性和任务成功。新报告加入中心视锥诊断，不删除或移动模型结果。

## 官方核验

- Qwen3-VL官方3D cookbook：https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb 。支持图中目标3D定位，不是隐藏世界补全验证。文字写弧度，渲染却乘180再转弧度；本项目明确采用渲染约定，不能宣称官方单位说明毫无歧义。
- M3DLayout配置：https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml 。text_condition与room mask同时开启，有专用类别表。不能据此将其视为任意可见对象集合的即插即用补全器。本轮问题来自Qwen输出，无需引入M3DLayout修复。
- LayoutGPT：https://layoutgpt.github.io/ 。使用in-context布局示例，3D实验以房间类型和floor plan size为条件。当前两次Qwen调用不是该方法复现，不继承论文性能结论。

## 下一步：仅隔离测试off-camera子任务

代码只修改语义任务定义、两阶段提供相机参数/坐标说明、语义阶段限制类别；保留已有Observer、模型、预算、种子和样本。暂不测试视野内遮挡补全，不强制所有对象z<0，不把错误框搬到背面。此实验是完整任务的一个子问题，不声称产出已完整G_full。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export SIMPLE_WORLD_RUN=qwen_edit_pano/outputs/qwen_world_simple_run002
export SIMPLE_WORLD_OBSERVER=qwen_edit_pano/outputs/qwen_world_simple_run001/observer

bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh complete-gt
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh complete
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh review
```

GT与Pred均每图一个seed、上限12对象，共24个候选。只把GT可见对象输入GT条件，GT隐藏对象不进入提示。共用现有图像描述，因此对照仅隔离结构化G_obs误差，不是完美语义Oracle。脚本默认沿用用户GPU2，可export SIMPLE_WORLD_NODE指定其他节点。

报告为run002/review_gt/review.html和run002/review/review.html。检查diagnostics.centers_in_input_fov及原始计划、坐标和原图；低视锥比例不是质量充分条件，还需看尺度、重复和空间关系。若GT条件仍明显违反可见证据或坐标约定，应停止把这条8B纯文本几何路径当可靠补全，不继续累加提示补丁。若GT条件合理而Pred不合理，再集中改Observer，优先定位/深度依据；不新增Adapter字段。
