# 室内验证诊断：执行与核验（2026-10-11）

## 本轮只改变输入样本

已人工查看原验证建筑中的局部图，选定下列三张；均在 oracle_supplement_epoch006_v1/samples.jsonl 的 val 中，图片 SHA256 与原 observer 清单一致。未修改正式 train/val 划分，未参考新样本生成结果选样。

- 8194nk5LbLH_c9e8dc09263e4d0da77d16de0ecddd39：室内休息区，有椅子、桌子、电视。
- X7HyMhZNoso_2bb4931911e4412cb51f6437a82c5b6c：室内客厅，有沙发、桌子。
- x8F5xyUWy9e_3a127fd2c5f34d978cba253145dae7d6：室内浴室，有洗手台、镜子、马桶等。

清单：qwen_edit_pano/data/observer3d_indoor_val3_inputs.jsonl。
预览：indoor_val3_selected.jpg。

本轮保持按类别查询、每类最多8实例、seed0、full size解释、相机高度1.5米等旧设置。目的仅为增加符合室内任务的诊断样本，不把换图当作修复Observer重复框或几何标定。前一轮建议的按2D实例查询尚未实现，本轮没有暗中改变方法。

## 执行

从项目根目录手动执行，不需要安装或切换环境：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_indoor_val3.sh observe
```

看 observer3d_indoor_val3_run001 各样本的 categories.normalized.json、observed.json、status.json，对照原图检查房间类型、主要家具、明显重复和失败查询。completed 只代表解析流程完成；partial 要看实际跳过内容；failed_categories 或 empty 的样本先修复Observer，不能把空观测当作成功再评价补全。

观测可用后：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_indoor_val3.sh complete
```

依次运行3个GPU作业，使用已有qwen3vl环境/权重/BERT缓存；遇到命令失败停止。输出 completion_indoor_val3_run001/<id>/seed000。检查 summary.json、layout_topdown.png 和 G_full.json；config.json 记录 context 数量及未送入M3D的类别。重复同一命令会覆盖同名结果。

只做 seed0 保持与上一轮相同预算；下一轮可另设输出目录对同一图测试seed0/1/2。当前无可靠GT对照时不宣称生成对象确为隐藏物体或全局结构准确。

## 联网和本地源码核验

官方资料：
- https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/3d_grounding.ipynb
- https://github.com/QwenLM/Qwen3-VL/issues/1501
- https://github.com/Graphic-Kiliani/M3DLayout-code
- https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/networks/autoregressive/autoregressive_transformer.py
- https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/datasets/m3dlayout_dataset.py

本地M3D commit：af67255521c7d8fd30a7b4eb83bd40f0f0d0742a。在线源码与本地行号/版本可能不同，实际调用以本地代码为准。

1. Qwen官方支持9维3D框；issue1501回复说明不要求prompt显式提供内参，输出角度归一化[-1,1]，顺序roll,pitch,yaw。cookbook文字写radians但绘图先乘180度；当前乘pi的转换与绘图计算一致。坐标为米的预测不等于尺度有保证。回投影需要内参，当前face2元数据hfov90，不应套用演示默认60度。
2. M3D complete_scene 确实添加START、保留输入context并逐个追加到END或预算。max_boxes是追加采样次数，含可能的END。已有三次作业成功，说明当前环境与这条推理路径已实际兼容；不必重装。
3. 官方训练与demo使用全1 mask；无需为了修复本轮结果额外生成room mask。
4. 官方AR训练是空间排序后随机前缀，不是相机可见集条件下的hidden补全。不能把API支持任意context视为任务已适配。未带序号位置编码的当前模型也不是简单排序输入即可修复。
5. 当前Observer泛类/细类重复查询与仅精确去重仍可能重复计数。按实例查询是后续独立改动，不是已完成工作。
6. full/half边长的权重数据链路，以及相机到房间地面/重力的转换尚未完整核实。当前默认是探索性假设。不要直接乘2或强行落地来美化结果；三张室内样本可做流程诊断，不能用于宣布米制几何准确。
7. Matterport也属于M3DLayout的数据来源。CauPano val并不自动等价于官方预训练权重的未见场景；正式泛化实验需核对官方训练场景与本项目验证建筑是否重叠，本轮只称诊断。
8. G_full 导出与Encoder字段相容，但只包含对象状态，不包含恢复出的完整房间几何；也不等于现有GT-only oracle_inference入口已能读预测状态。ERP接入仍是后续工作。

## 顺序与结论边界

先室内Observer → 查看对象 → M3D补全 → 查看布局；随后核对几何与GT observed对照，必要时改实例查询或hidden completion训练采样；最后接生图。

前几步的问题是样本选择和对成功的解释过强，不代表已跑出的JSON全都无效。不能承诺联网查过就不存在预测误差或未知兼容问题；可确认已检查的接口、已通过的CPU验证和已有GPU运行事实。
