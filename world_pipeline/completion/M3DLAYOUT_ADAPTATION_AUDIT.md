# 官方预训练适配核验（2026-10-11）

## 实际运行配置

本轮 completion_indoor_val3_run001 三份config保存的text依次为living room、living room、bathroom的完整房间类型句。complete.py默认构造该句，并传入complete_scene(text=text or None)。没有关闭text_condition；只有显式--text ""才绕过文本分支。

使用官方m3dlayout_autoregressive.yaml、autoregressive_transformer（非PE）、autoregressive_59000.pth，严格加载state_dict，仅删除旧position_ids buffer。使用BERT文本分支。当前third_party源码tokenizer入口被本地改成google-bert/bert-base-cased；BertModel入口仍为bert-base-cased，这是模型解析入口调整，不是关闭文本或变更网络。未据此修改用户文件。

官方gradio_demo.DATASET_STATS_AR实际词表101个对象+start/end=103；其注释写102+2=104，与literal不一致。当前代码动态读列表长度，使用input_dims110/n_classes103，避免手写错误。mpcat40仅用于导出给Adapter，没有替换M3D网络类别顺序。

坐标/尺寸/角度使用官方demo bounds归一化，数值公式正确不代表相机→布局的物理坐标已标定。full/half、重力、地面、水平原点和yaw的适配仍需证据。

sample_num_points121用于训练序列与dependency head配置；我方按121+1构建dependency head。Adapter max_objects160是另一层容量，并未改M3D网络配置。当前context35/2/7，追加预算32，即最多67/34/39个context+追加token（其中可含END），不触及121，且实际全部提前END，容量不是这轮已观察错误的解释。将来更大context必须独立管理M3D训练长度范围，不能直接以160当作M3D能力上限。

## 排序

官方按round(translations/0.5)后x,z,y优先排序，再抽随机前缀。当前AutoregressiveTransformer无对象序号位置编码，属性的FixedPositionalEncoding作用于xyz/size/yaw，attention_type full；预测读取固定query token。因此在eval下，对同一组对象同步排列所有字段，query理论上置换不变（浮点舍入除外）。这是代码结构推论，没有运行当前权重的排列对照。PE变体与此不同，本轮未使用PE。

真正差异是集合组成：完整排序[A,B,C,D]，训练前缀只能[]/[A]/[A,B]/[A,B,C]/[A,B,C,D]，实际可见[B,D]即使排序仍不是前缀。排序不能补出缺失A,C，也不能把这种条件分布改为训练分布。不要删除已见对象以强行拼前缀。

## 文本

text_condition=true说明使用文本分支，不代表必须给完整世界描述。训练代码从global/large/small descriptions保留room_type/summary，并采样0–3其他句，还可能重写和加约束。本轮房间类型句不是缺文本，但信息偏少。仅可见图描述与训练完整场景文本的分布仍可能有差异，不能预设加长必然改善。

可做低成本对照：固定同一份context、同一组seed和其他参数，只改--text：房间类型 vs 简短可见内容描述。文本只陈述局部证据，不虚构背后家具，不声明可见对象是完整清单，不使用only these objects等封闭世界表述。BERT条件不是通用指令执行器，不能依赖一句“忽略错误框/补全隐藏区”修复结构。不要加入未统一参照系的左右关系。

## 有证据的改进顺序

1. 修Observer重复实例，并回投影检查3D；保持Adapter字段不变。
2. 核实官方预训练几何约定，使用GT observed诊断隔离上游误差；不要把AABB冒充OBB。
3. 固定context做短文本对照，结合用户可选文本（如果存在）统一条件。所有比较均报告完整seed集合，不挑最好结果。
4. 仅当上述仍不足时，用GT observed+hidden做completion采样适配：context=所有observed+hidden[:k]，target=hidden[k]/END。hidden内部可沿用官方空间排序；无需为了已有context顺序换模型。文本从局部可得证据和可选用户文本构造，不泄露隐藏GT。
5. 保留M3D词表/结构/损失先做最小微调，若需要新增类别或网络标志再单独实验。未知词不强行归并，不编造语义。

目标依然是Local RGB+可选文本→observed→plausible full state→冻结Qwen-Pano经Adapter生成ERP。文本是M3D条件的补充，不替换可见结构约束；geometry精度和多样性都需看，不只看新增数量。

## 来源

https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/config/m3dlayout_autoregressive.yaml
https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/networks/autoregressive/autoregressive_transformer.py
https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/scene_synthesis/datasets/m3dlayout_dataset.py
https://github.com/Graphic-Kiliani/M3DLayout-code/blob/main/gradio_demo.py

本轮只读核验与写报告，没有改动推理代码、权重、训练字段或运行GPU。
