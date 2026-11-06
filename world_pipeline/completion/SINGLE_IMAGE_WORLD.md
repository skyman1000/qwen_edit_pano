2026-10-18最新执行顺序修正：先验证局部观测状态，尚未通过，不继续补全。

用户指出的优先级是正确的。run002只证明流程能完成，未证明观测状态正确；其类别/2D主要对象大体对应，不等于三维位置和大小通过验证。下面旧run/plan说明保留作历史入口，当前先运行observe。

官方技术报告3.2.5提到训练数据统一到虚拟相机坐标；cookbook用原生九维框和自然语言描述定位，没有将真实K放入示例推理prompt。公开说明不足以确认本项目任意FOV下，原始框能仅翻转y轴就变成真实相机坐标。不能据此断言输出一定是虚拟坐标，也不能断言加入内参文字就一定完成了转换。

已新增observe入口，局限在本文件的观测分支：原图不画框、不裁剪；用类别、外观描述和图像方位指认目标，请求原生bbox_3d，保留label/九维原数值。不要求模型重定义为项目AABB或新的角度单位。原图与复用实例图逐字节核对；复用只带类别/描述/2D，旧三维估计不进入新结果。

先检查一个床和两盏床头灯，对应run002的observed_0000、observed_0010、observed_0011。它们只是三个明确目标的接口/能力检查，不代表18个对象均验证正确：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_single_image_world.sh observe \
  --hfov 90 \
  --reuse-observation qwen_edit_pano/outputs/single_image_qwen_run002/G_obs.json \
  --observed-ids observed_0000 observed_0010 observed_0011 \
  --output qwen_edit_pano/outputs/single_image_observer_native_run001
```

只复用现有实例身份，三维重新推理。以observe-preflight替换observe可做CPU预检；助手已执行。去掉--observed-ids可查询全部复用实例；去掉两个复用参数则从--image重新生成可见清单。首次先用以上三个目标，不立即扩成完整世界。

输出geometry/*.txt为原始回答，G_obs.json保留native_bbox_3d/native_label。由于原生相机到项目相机的映射尚未确认，center_local_m/size_aabb_local_m保持null，observation_verified=false、camera_mapping_verified=false；没有把未知写成0或把九维数值标成已校准。observer_reprojection.png中红框为2D目标，绿框为在“原生相机等于真实相机”假设下的三维投影，诊断假设写入reprojection_diagnostic.json。即使绿框吻合，也不能单独证明米制深度/尺寸正确。没有自动生成G_full或调用completion。

官方依据及修正原因见research/audits/observer_official_contract_2026-10-18/REPORT.md。新观测推理尚未由用户执行，不能称几何已经修好。后续须先核对目标身份、坐标解释与独立深度/尺寸参考，再推进视野外补全。

以下为历史两阶段入口说明。

2026-10-18：单图片、冻结Qwen的两阶段状态基线。

目标是Local RGB→G_obs→G_full。两阶段均使用现有Qwen3-VL-8B-Instruct，模型只加载一次，无训练。2026-10-18的run001实测出现重复生成，当前v2改为“短实例清单→逐物体几何”，不再让一次回答同时枚举并填写全部物体三维框。这个入口直接读取图片，不依赖旧Observer结果、场景GT、GT观测清单、完整ERP或完整图caption；读取已有vocabulary.json仅用于类别映射。它不执行全景图像生成，也不证明预测几何准确。

从项目根目录执行，GPU与下载由用户手动操作：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_single_image_world.sh run \
  --hfov 90 \
  --output qwen_edit_pano/outputs/single_image_qwen_run002
```

默认图片是已检查的卧室Local，完整路径见single_image_world.DEFAULT_IMAGE。该skybox face2已知水平FOV90度。可以加--image指定其他图片，但其--hfov必须提供相应已知或估计值；不能把所有普通照片都当90度。输出必须是新目录；run002之后的重复实验更换run003等，不覆盖旧结果。

CPU预检（不加载模型）：

```bash
bash qwen_edit_pano/world_pipeline/completion/run_single_image_world.sh preflight \
  --hfov 90 --output qwen_edit_pano/outputs/single_image_qwen_run002
```

助手已执行默认输入预检与CPU测试，没有代跑GPU。默认观察清单上限32、最多新增12个对象，是首次单图诊断的上限，不是要求模型补足，也不保证候选已穷尽真实场景。--max-observed/--max-hidden可调整，总对象上限仍为160；达到容量会在summary标记。--max-new-tokens默认4096是总的单请求上限，实际清单最多2048、可见单物体框最多512、未见单物体布局最多384 tokens。每个失败请求最多重试一次，记录两次原始回答；单物体几何失败保留该对象但几何未知，不能把它当准确三维布局。

解码按照官方Instruct建议：do_sample=true、temperature0.7、top_p0.8、top_k20、repetition_penalty1、presence_penalty1.5，初始seed3407，重试3408。Transformers没有直接传入本实现的presence_penalty参数，由已有GeneratedPresencePenalty作为logits processor执行。重复的完整JSON对象连续出现三次会提前停止并标记失败，不补括号、不把截断前缀当成功结果。模型仍可能产生重复、幻觉或错误几何，此保护不等于精度改进证明。

可见对象采用官方bbox_3d九维结构做单目标查询，再由代码转换成项目y向上、相机轴AABB。官方3D cookbook的角度文字说明与绘图处理存在不一致；本实现明确在提示中规定角度/pi，并记录native_angle_unit及native_bbox_3d，便于核查，没有擅自换历史实验单位。未见对象不是grounding真值，按语义计划逐个预测项目坐标下的中心和尺寸，始终标为假设。

核心对象字段：

| 字段 | 含义 |
|---|---|
| category_id | 当前37类映射；具体名称保留raw_category/description |
| center_local_m | 相机局部三维中心，x右/y图像上/z前，米 |
| size_aabb_local_m | 相机轴AABB完整xyz边长，米；不是半尺寸或物体固有宽高深 |
| track_id | 区分实例，数量由不同对象条目体现 |
| bbox2d_visible_xyxy_norm | 可见部分2D框，用于核对图像；状态内0..1，两次模型请求均0..1000 |
| value/valid/source等 | 数值是否存在及来源。valid=true不代表几何正确；未知保持null/valid=false |
| completion_role/local_evidence | 已观测、可见漏检补回、未见假设、或无法分类的新增对象 |

当前World Encoder读类别、中心、尺寸及valid/mask；不读取朝向、材质、关系或外观。当前状态足以表达对象盒布局，不等于包含建筑拓扑、完整mesh与材质的完整三维场景。

输出：

- observation.txt：观察阶段的紧凑实例清单原始回答。
- G_obs.json：从输入图片估计的可见状态。
- observed_overlay.png：可见2D定位叠加，先检查实例及类别。
- completion.txt：补全阶段的紧凑计划原始回答。
- geometry/*.txt：逐物体几何的原始回答；可见框与未见布局分别处理。
- *.generation.json、geometry/*.generation.json：实际生成token数、EOS/长度/重复停止原因。
- G_hidden.json：明确标为unseen的新增假设。
- G_visible_added.json：第二阶段发现的可见漏检。
- G_unresolved_added.json：未明确区分可见/未见的新增提议，保留待查，不伪装成已确认隐藏。
- G_full.json：原G_obs对象原样保留，追加上述提议的完整候选集合；quality_status=not_validated。
- summary.json、issues.json：数量、数值覆盖、原观测保持及字段问题。completed仅表示软件流程完成；partial表示存在字段问题。两者都不是场景质量分数。

外层JSON同时支持数组和约定的objects/additions包装，避免先前数组用字符串索引的错误。非法数值保留为未知并记录原值；不交换轴、不翻符号、不猜类别或凭空填数使结果通过。不支持的物体类别保持未知并记录，不能认为已正确控制其具体语义。输出中仍可能有重复或不合理物体，需结合实际图片人工核查。

CPU测试覆盖：真实旧数组和本轮循环输出、未知/非法几何、显式相机轴转换、相同2D尺度、短请求工作流、原观测保持、可见漏检/未见假设/不确定新增分离、EOS恰好落在token上限时不误判，以及当前World tensorize兼容。

2026-10-18实测run002：流程完成但status=partial。18个观测对象，其中12个有数值几何、6个未知。新增1个可见漏检候选door，未见提议为0；不能称完整场景补全成功。按照本次显式角度单位与90度相机，将12个观测三维框回投影，其中7个与其目标2D框完全不相交。自一致性诊断不等于GT几何评分；但该结果不能作为可靠三维状态。详见research/audits/single_image_run002_2026-10-18/REPORT.md和reprojection_review.jpg。

下一步只隔离未见语义规划，复用本次G_obs的类别、描述和2D框，屏蔽其不可靠米制数值。不重跑Observer或逐物体3D，不输出或覆盖G_full。三个seed是三个独立备选方案，不合并成物体全集；仍需逐项核对是否重复、是否真能处于未见区域以及是否符合当前房间。没有强行规定最少新增数量，也没有“非空即成功”的判据。

```bash
bash qwen_edit_pano/world_pipeline/completion/run_single_image_world.sh plan \
  --source qwen_edit_pano/outputs/single_image_qwen_run002 \
  --output qwen_edit_pano/outputs/single_image_unseen_plan_run001
```

默认三次短语义请求，seed3407/3408/3409；模型加载一次，每次最多2048token，至多8个提议。原始回答位于seed*/plan.txt，解码结果plan_decoded.json，诊断diagnostic.json。空回答和失败都保留，不重抽到满意，不自动改标可见对象。CPU预检将上述plan替换为plan-preflight，不加载模型。旧run002及全部失败记录保持原样。
