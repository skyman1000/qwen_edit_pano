# 从同一局部 RGB 重新做 API / Qwen 世界补全对照

更新时间：2026-10-24。本入口只做预测世界状态与可视化，不训练、不接 Adapter、不生成 RGB 全景。旧代码、旧结果和旧续跑脚本不修改。

## 先澄清旧结果

1. `semantic_world_run013/summary.json` 确实记录 Qwen 自行补出 4 件、原观测 17 件、4 件均通过原视锥外数值检查。但它用的是另一张图 `zsNo4HB9uLZ_ead481533f834704bd489d3d44b6a03a`，协议为 `planned`，不是当前案例、观测或 SCM 的受控对照。因此不能据此认定同图上的 Qwen 能力退化。
2. `generative_world_qwen8b_layout_run007_compare002_recovered` 的 13/13/8 件来自复用 `generative_world_deepseek_run007` 的语义模型后，由 Qwen 做布局，再 CPU 恢复导出。它不证明 Qwen 独立提出了这份丰富清单。
3. `typed_mechanisms_v5_qwen_resume_run001` 继续复用 `target_support_v4_qwen_full_run001` 的域：C、S 各 1 状态，两组各 1 件。两组均为 P(不添加)=0.6、P(添加)=0.4，因此空/单件/单件/两件的概率依次为 0.36/0.24/0.24/0.16。前三为 0/1/1 件是精确的程序结果，不是可视化吞掉了对象。固定旧域续跑无法增加候选类别或实例。
4. 旧图为 `p(C)p(S|C)p(E|C,S)∏p(G_i|C,S,G_parent)`。C/S 单状态时，观测 E 在这个域内无法重新加权隐藏组。程序记录的隐藏更新 TV 约 0。这不意味着图像完全没用：图像早已用于提议域，但不能把这种逆向提议本身叫因果验证。
5. 较早的 `official` Qwen 配置是 temperature=.7/top_p=.8/top_k=20，并有本地 presence penalty；较新的 `matched_v1` 是 .4/.9/50、无此惩罚。历史结果不是全部相同提示词与参数，更不是同一实验不断重复。

## 新实验到底相同什么

```text
同一张原始 RGB、同一 HFOV
  ├─ API 识别 2D + 可见语义 → WildDet3D → API 自己的 G_obs
  │    → 同构联合场景 SCM，由 API 提议域、条件机制和布局 → G_full/API
  └─ Qwen 识别 2D + 可见语义 → 同一 WildDet3D → Qwen 自己的 G_obs
       → 同构联合场景 SCM，由 Qwen 提议域、条件机制和布局 → G_full/Qwen
```

两端使用同一份 2D 提示词、后续提示词模板、解析器、SCM 分解、几何求解和有限计算预算。2D 请求的图片 JPEG92/max1536 字节及提示词完全相同。后续填入的观测、ID、候选和机制由各端独立产生，完整提示词不可能逐字相同。3D 使用同一官方权重、逐实例 geometric+label 模式和原始图片像素内参。

公开采样参数均为 temperature=.4、top_p=.9、top_k=50、max_tokens=8192，禁用 API thinking，本地用非 thinking 的 Qwen3-VL-8B-Instruct。API frequency_penalty=0，本地 repetition_penalty=1，均不施加重复惩罚。种子 3407 用于本地、外部概率推断与首次请求；接口修复的第二次本地请求用 3408。API 不发送未获保证的 seed。不能宣称服务端分词器、视觉预处理、随机性或内部解码完全相同。

“同一个因果模型”在这里准确地说是**同一生成结构与实现**。域及数值 CPT 分别由两端生成，不能声称是完全相同的数值模型。若冻结 G_obs、域和 CPT 后只换布局模型，才是另一种更窄的布局对照；本次不将它混入端到端结果。

两条链路自己的 G_obs 都会固定不变。由于两端 G_obs 可不同，本次结果反映端到端系统差异，不能单独归因于补全模型。报告展示两个 Observer 的原始框、原始类别、3D 投影和对象数，禁止凭同名 ID 跨端配对。

## 新的联合配置生成模型

本版本是有限候选的、假设驱动的联合世界 SCM：

```text
C：潜在用途/居住偏好 → Z：完整周围场景配置 → E：局部正观测语义事件
                                   └────→ H：该配置的完整隐藏对象程序
```

分解为 `p(C)p(Z|C)p(E|Z)p(H|Z)`。H 是 Z 所对应完整清单的确定性展开。外部程序保存条件概率表，执行 `p(C,Z,H|E=signature)` 精确枚举、前向采样、逆 CDF 噪声溯因和 `do(C=...)` / `do(Z=...)` 机制替换诊断。

重要变化不在于节点数增减，而在于**采样的单位是完整的联合世界配置**。同一配置中的家具、变体、宿主和空间组织共同成立，不再给每个组独立插入一份可能性更高的“不添加”，从而把整套场景拆空。几何编译器内部沿用旧的模式编号格式，但每个已选联合配置固定选 mode=1；mode=0 没有语义概率，不会被独立采样。模型真的提出空清单时仍完整保留，不过滤空世界来制造提升。

有限预算为至多 3 个配置、每配置至多 4 个空间组/24 个新对象；不是强制数量、方向配额或房间设计目标。若模型只提出 1 个配置，程序照实记录，后验也无法在配置之间更新，绝不捏造多样性。24 件上限只是一次请求与几何搜索的计算容量，不保证或要求生成 24 件。

`Z` 的每个状态是一整套具体场景程序，因此可以表达对象间联合相关性，但表达能力受有限候选覆盖限制。它不是每个对象都有独立可干预机制的完整物理 SCM；物体级删除/增加的任意反事实不在本版本中。这里的配置级干预诊断不能直接充当实验数据上的因果收益。

## 物体之间的逻辑与空间关系如何生效

- 具体对象身份与变体在语义阶段固定。新对象必须是原图视野外的独立实例；同类不同实例允许存在，不能把原观测移走再命名为隐藏对象。
- `support_target` 绑定确切宿主实例。支撑与相对锚点形成可执行依赖顺序，拒绝自支撑、未知引用和循环。对象支撑时的高度由宿主顶面和自身半高计算；宿主和被支撑物共同生成、共同布局。
- `requires` 是声明的存在依赖，只能引用同一完整世界中实际存在的实例。它不会自动把“常见共现”升级成必要关系。其语义是否正确仍需看具体变体与图像，并非仅引用合法即可证明因果。
- `near` 与相机坐标下的左右/前后关系实际进入共同布局目标，作为软项；不添加硬编码的“近必须小于某距离”或最大位移。固定高度下的上下关系会报告残差，不能谎称全都优化通过。
- 原观测坐标与完整 AABB 尺寸固定；新增尺寸由模型提出，求解器只调整 XZ 位置。无真实墙、地面或重力测量时，不把显示边界当房间边界；地面支撑仅用预测物体底面的相机 Y 水平先验。壁挂/悬挂装置可以保留显式 Y，但安装未验证会显示 `layout_needs_review`。
- 新框在原输入相机视锥外；不是对局部可见区域继续补物，也不根据不可见区的“未检测到”推断物体不存在。
- 布局失败保留完整语义候选、清单和原权重，不删难摆的物体、不重归一化幸存候选。普通数值导出成功与语义、物理正确性分开报告。

`causal_model.json` 是概率模型；`object_relations.json` 是具体对象的支撑/存在/空间关系。这两者不能混称“学到了因果图”。

细类别不受 Adapter 的 30 多类限制。原始普通名词及具体变体保留；能精确映射的类别附上现有粗类别 ID，未映射对象保留 null 和原始名称，照常进入 G_full、3D 和 ERP 状态图。本阶段不导出 Adapter 子集。

## 联网核对的依据与边界

| 工作/官方实现 | 本次借鉴 | 没有据此宣称 |
|---|---|---|
| [SD-SCM / EMNLP 2025](https://aclanthology.org/2025.emnlp-main.107/)、[官方代码](https://github.com/lbynum/sequence-driven-scms) | 用户指定结构、冻结 LM 条件机制、外部采样/条件/干预 | 其论文没有解决本任务的视觉后验。本版本用同一接口询问主观 CPT；官方实现用 LM 候选续写分数，不是逐行复现 SD-SCM |
| [WildDet3D 官方推理说明](https://github.com/allenai/WildDet3D/blob/main/docs/INFERENCE.md) | 2D 前端 + 原像素 geometric:label 单实例 3D；原始 corners 转相机完整 AABB | 单图深度和尺寸已准确；IoU=0 的原始匹配会保留并报告，不假装绑定可靠 |
| [Holodeck 官方代码](https://github.com/allenai/Holodeck) | 对象语义与关系先确定，再做几何布局 | 普通空间约束等同因果模型；从语言设计房间等同从局部观测补全 |
| [LayoutVLM / CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.html) | 模型提出语义/布局表示，外部优化物理代理目标 | 其全部成果都不需要微调；本实现复现了它的性能 |
| [ATISS 官方实现](https://github.com/nv-tlabs/ATISS)、[DiffuScene 官方实现](https://github.com/tangjiapeng/DiffuScene)、[InstructScene 官方实现](https://github.com/chenguolin/InstructScene) | 从部分对象继续生成、联合对象集或语义图与布局分工，是值得借鉴的场景表示思路 | 它们本身是本项目已接入的免训练因果模型；本次没有安装、训练或部署这些模型 |
| [CausalStruct 论文](https://arxiv.org/html/2509.15249v1) | 区分关系顺序、依赖及优化，保留具体实例关联 | 将其文本场景/3DGS流程原样搬到本项目，或通过关系边证明真实因果效应 |
| [Qwen 官方 grounding 示例](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb) | 显式规定统一 0..1000 二维框接口，再按原图换算像素 | 把 Qwen 的框坐标直接当 WildDet3D 原图像素；外部网页未能完整解析大 notebook，不把未见正文的细节作为证据 |
| [硅基流动模型中心](https://www.siliconflow.cn/models)、[更新公告](https://api-docs.siliconflow.cn/docs/release-notes/overview) | 当前 DeepSeek-V4.1-Flash 支持图像输入；2026-10-21 ID 不变、后端升级到 1017 | 固定模型 ID 能锁定服务端权重版本；故记录实验创建时间、实际响应 model 和用量 |

所有概率都是选定有限域内的 LM 主观先验/似然。域本身由同一局部证据提出，因此这不是独立校准的全场景贝叶斯后验，也没有定义像素 RGB 似然。因果收益、隐藏对象真实性、物理合理性和未来全景控制收益目前均未验证。对象更多也不自动代表更合理。

## 执行

从项目父目录执行；不再调用历史 `research/audits/*/run.sh`：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/fresh_compare.sh preflight
bash qwen_edit_pano/scripts/fresh_compare.sh test
read -rsp 'SiliconFlow API Key: ' SILICONFLOW_API_KEY
export SILICONFLOW_API_KEY
export SILICONFLOW_MODEL='deepseek-ai/DeepSeek-V4.1-Flash'
bash qwen_edit_pano/scripts/fresh_compare.sh run GPU1
```

GPU1 换成实际可用节点。默认是当前验证集 `X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e/local.png`，HFOV=90。会自动创建全新 `qwen_edit_pano/outputs/fresh_rgb_compare_日期_时间_PID/`，两个分支全部从 RGB 重做，无历史观测、清单或 CPT 复用。各阶段进度同时写屏幕和该分支日志；API 分支失败时仍执行 Qwen 分支，最后如实报告不完整状态。

如换图，需要同步使用其真实输入 HFOV，不能随意用默认值：

```bash
export CAUPANO_IMAGE='/absolute/path/to/local.png'
export CAUPANO_HFOV='90'
bash qwen_edit_pano/scripts/fresh_compare.sh run GPU1
```

输出布局：

```text
experiment.json / input.png
api/observer2d/{G_2d.json,boxes2d.png,answer.request.json,...}
api/observer3d/{G_obs.json,native_boxes.json,projection_boxes.png,...}
api/completion/{causal_model.json,joint_programs.json,inference.json,...}
api/completion/world_01/{G_full.json,G_out.json,object_relations.json,numeric_review.json,world/index.html,...}
qwen/...                           # 完全独立的同结构链路
review_001/index.html               # 自动生成的两端对照页面
review_001/review.json              # 实际请求参数、2D输入、WildDet权重一致性检查
```

世界候选按各自有限域的后验排序，最多 3 个；2:1 图为状态方向可视化，不是生成的 RGB 全景。报告包含原图 2D、WildDet3D 重投影、3D 交互框、XZ 和完整对象字段。重复分析不会调用模型，会新增报告编号：

```bash
bash qwen_edit_pano/scripts/fresh_compare.sh review /absolute/path/to/fresh_rgb_compare_实际目录
```

不提供假定存在的 resume/config 路径；同阶段已存在时保留成果、拒绝覆盖。真正需要阶段重试时先根据具体 raw/summary 确定原因；本轮不要混用旧域续跑。

## 复核范围

CPU 测试使用明确的合成回答，不冒充模型输出。覆盖两条独立观测链、真实 API 客户端的离线请求格式、精确后验与干预、支撑与引用循环、未映射类别、全清单保留、跨组软关系梯度、布局失败的概率质量保留、观测前缀不变、可视化以及 2:1 比例。只读预检查验证了本机 Qwen/WildDet 权重与 WildDet 已安装的 corners 坐标契约；未构造 GPU 模型、提交 GPU 任务或调用付费 API。

本版本先验证清晰、可审计的端到端生成。后续若要区分补全模型能力与 Observer 影响，应固定同一 G_obs 再独立生成域/机制；若要证明因果结构的收益，还需在固定模型与预算下加入普通联合补全基线。不能用一次 API/Qwen 比较替代上述证据。
