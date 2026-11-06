> **2026-11-06 恢复说明：当前默认方法为交接前 v13 H/A/V，执行入口固定为 `scripts/world_completion.sh`。joint、panorama_plan、simple、scope、stable 等后续推荐只作历史记录。恢复依据和当前命令见 [RESTORE_ORIGINAL_V13_20261106.md](RESTORE_ORIGINAL_V13_20261106.md)。本页历史记录可能过时，不应覆盖该恢复说明。**

# CauPano 项目交接：局部 RGB → 合理的隐藏世界状态

更新日期：2026-10-24，Asia/Shanghai。面向没有历史对话上下文的 Codex。

2026-10-28 **仅观测第二批输入已准备，未执行GPU**：用户要求再测5图、包含“客厅+茶几+落地灯”、Qwen观测即可。已实际看原图找到 `X7HyMhZNoso_d0468b580cd146778bae5e486bf3e50e` 灰色双沙发/白茶几/金属落地灯；另选四图为白沙发客厅zsNo...ed864（旧label卧室错误不采用）、玻璃后健身区8194...9bdde31、桑拿x8...39d826、浴缸区x8...03de84。新清单 `data/observer3d_local_extra5_inputs_20261028.jsonl`，只含原RGB/ID/split/哈希/尺寸/HFOV，所有val、1024²、90°，与上一批5图ID和像素均无重复；灰沙发是历史开发图，不称独立未见留出。输入预览 `outputs/local_observation_extra5_inputs_20261028_001/index.html`（非预测）。CPU实际prepare-only `outputs/local_observation_extra5_preflight_20261028_001`，真实GPU/模型尚未执行；不改任何观察器/SCM/脚本，沿用Qwen2D→WildDet3D出3D、无API/Key/隐藏补全。执行 `CAUPANO_TEST_INPUTS=qwen_edit_pano/data/observer3d_local_extra5_inputs_20261028.jsonl bash qwen_edit_pano/scripts/local_observation.sh run GPU3`。更多说明见LOCAL_OBSERVATION_ONLY文档第二批部分。不要补跑已取消的直接RGB结构化世界基线。

2026-10-28 **用户最新范围调整与仅观测实验**：用户明确停止“直接RGB→结构化完整世界”对照，不继续补跑/修该基线，历史保留；未来图像级无因果RGB全景对照是另一个任务，本轮不实现。新增独立 `local_observation.py` 与 `scripts/local_observation.sh`：report只读suite与observation/，不读completion/、不从完整世界筛蓝色对象，G_2d/G_obs字节复制；只允许predicted/observed/visible且2D/3D身份一致，缺3D保留未知，不造坐标。run/prepare/resume只调Qwen2D→WildDet3D→观测报告，无Key/API、H/A/V、隐藏库存或布局求解，旧生成源码/脚本不改。已有五图观测报告 `outputs/local_observation_review_20261028_001/index.html`，厨房用最新144049源，2D/3D为16/16、6/6、7/7、8/8、3/3，新增恒0，厨房漏检不掩盖。执行见 [LOCAL_OBSERVATION_ONLY_20261028.md](LOCAL_OBSERVATION_ONLY_20261028.md)。已查看用户上传 `360°全景场景因果推理流程图-1.png`，原图不修改；总体主线可用，但当前V与R为联合选择，S为安装依赖而非通用关系；储物方位0.20+0.35+0.50+0.05=1.10非法；示例0.4/0.25/0.08不是默认K4全有效行的估计；当前H家族混合＋逐角色强调，不是任意功能组合；机制阶段文字观测，最终RGB全景和有纹理俯视渲染未接通，须标未来/概念。详见 [CAUSAL_FLOWCHART_REVIEW_20261028.md](CAUSAL_FLOWCHART_REVIEW_20261028.md)。本轮不代用户运行真实GPU提取，不改观察器预测算法或SCM。

2026-10-28 **最新实际结果与组会比较已整理**：用户新跑 `outputs/kitchen_observer_20261028_144049`，2D恢复路径实际调用2次，输出3个吧凳；WildDet及两后端完整方法均导出，Qwen新增15/API37，原数值检查通过。**仍有观测漏检**：中岛/橱柜/灶具等明显物体未入G_obs，on_floor=false及“吧凳在中岛表面”的解释错误，没有地面锚点；不能称厨房完整观测已修好。直接RGB实际 `direct_rgb_world_20261027_233610_3110337`：API四图非空、厨房截断失败；Qwen三图截断失败、卫浴/卧室为空，因此直接基线为4非空+2空+4失败，未齐全。新只读页面 `outputs/meeting_comparison_20261028_001/index.html`：旧数值恢复中四图＋明确用新厨房替换，20格四列，保留失败，16份实际保存状态（含两空）的同尺度平面/投影/3D/ERP、类别标签、CSV、来源。`model_explained.html` / `MODEL_EXPLAINED.md` 为详细讲稿，`generation_pipeline.svg` / `conditional_scm.svg` 为流程与SCM图，`probability_mechanisms.png`为真实概率图；`model_evidence.json`有实际票数及抽样。新增 `meeting_world_comparison.py`、`meeting_model_notes.py`，不改任何生成源码，无新模型调用。厨房共同域9非空程序/50候选实例/27空间计划/5角色/6H，K4每后端244个机制短提问但只一世界；Qwen选social2件、API24件导致22件数量差。Qwen多数存在机制H不敏感，不能声称已验证因果收益；完整流程对直接RGB同时改变观测/求解/预算，不是纯因果消融。报告验证包含原始文件哈希、链接、标签/ERP尺寸、实际保存噪声重放及边缘概率复算。下方为历史过程。

2026-10-28 **厨房观察器根因修正；下一步先 observe，勿盲跑全流程**：两次真实厨房重测 `organized_multiscene_20261028_104505_836961` / `110616_948850` 都卡在2D：首轮8条吧凳含4条完全重复副本，整批拒绝后纠错为空且done=false。首轮489 tokens正常EOS，原始两轮回复哈希完全相同；WildDet/因果/API均未执行，下游只是blocked_dependency。早期厨房“成功”实为重复→空done=true的零观测假成功；上一轮非空→空守卫只阻止假成功，没有修好候选恢复。**本次仅修观察接口，不改SCM/几何/API**：保留合法候选及副本审计→原图全量清单复核→同类高重叠实例显式身份判断；不按IoU自动删物体、不填坐标、不设数量配额。新增 `assembly_observer_recovery.py` 和 `scripts/organized_multiscene.sh observe GPU3 NEW_OUTPUT`：仅Qwen2D，无API Key，导出observation_index.html；2D失败立即停下游。看框可接受后对该新目录resume，才继续WildDet和两后端各一世界，需要现有硅基流动Key。34项CPU测试通过，四个旧成功图G_2d回放一致，旧十份数值状态字节不变；**尚无本次真实新视觉推断，不能称厨房已修好或质量提升**。单图输入清单不变。详见 [OBSERVER_RECONCILIATION_FIX_20261028.md](OBSERVER_RECONCILIATION_FIX_20261028.md)，审计 `../research/audits/observer_reconciliation_20261028/release_001/`。下方旧generate指令为历史建议，以此分阶段入口为准。

2026-10-28 **实际数值恢复完成及重要验收限制**：`outputs/organized_numeric_recovery_20261028_001` 已CPU实际恢复两份保存布局，10/10数值通过、八份原成功G_full字节不变，零新模型调用；`numeric_before_after/index.html`同轴对比；13数值/库存测试＋37旧管线用例（完成状态旧断言修正后复测）＋2观察一致性新例，共52不同用例核对。**不能说10/10有效补全实验**：复查厨房原图及chunk_00_0/1发现2D首轮非空吧凳重复被拒绝，纠错变空被当成功，3D观测因此也0，厨房两后端状态都缺少可见物体。结果页面新增acceptance_review提示，四图八状态可继续评估，厨房需独立重做观测及依赖推断。assembly_io_recovery加入“拒绝非空候选后空纠错不能成功导出零观测”的守卫；真正首次空图仍允许，无数量配额。本次冻结输入/CPT没补造或篡改。单图原始输入子集 `data/observer3d_kitchen_retest_inputs_20261028.jsonl`；后续可 `CAUPANO_TEST_INPUTS=... bash scripts/organized_multiscene.sh generate GPU3` 重测该一图，用户自己执行GPU/API。不要继续改因果模型掩盖输入质量问题。细节见最新根因报告。

2026-10-28 **最新根因已经通过固定真实状态对照定位**：来源 `organized_multiscene_recovery_20261028_094728_566800` 实际8/10成功，剩Qwen休息区28物体/5重叠、厨房33物体/23重叠；JSON全部已解析，当前为blocked_numeric，旧summary里的numeric_group是历史错误。见 [NUMERIC_SEARCH_ROOT_CAUSE_20261028.md](NUMERIC_SEARCH_ROOT_CAUSE_20261028.md)。原articulated_repair碰撞搜索每分支一轮增一条分离约束，却固定只8轮；Qwen多个根初始坐标为零。固定全部实际观测/库存/CPT/布局/尺寸/朝向/宿主/区域/物理约束，仅将预算8→64，两例均第11轮、10条分离约束达到零重叠并过原数值检查。因此本次只改碰撞搜索预算为min(128,max(8,2n))及终止诊断；不改因果模型、不重抽、不降低门槛。新增 `assembly_numeric_recovery.py` 和 `scripts/organized_numeric_recovery.sh SOURCE_ROOT [NEW_OUTPUT]`，CPU从保存坐标直接重编译两例，八成功字节复用，原失败文件夹在新目录numeric_solver_history完整保存并保护哈希，自动report。**这次不需Qwen/API/GPU/Key，不要再运行repair-coordinates**。源码fork仅数值恢复显式允许assembly_spatial变化，普通接口恢复仍拒绝静默更换数值源码。对照和测试记录 `../research/audits/geometry_root_cause_20261028/diagnosis_001/`。

2026-10-28 **最新实际运行 6/10 完成；仅修复 Qwen 容器解析**：来源 `organized_multiscene_named_prepared_20261028_001`，API五世界全部完成、Qwen客厅完成；其余四图卡在JSON外层包装、对象ID字典、roots/hosted重复分区，原回复已有合法坐标，旧重试却变空列表。详细见 [COORDINATE_TRANSPORT_FIX_20261028.md](COORDINATE_TRANSPORT_FIX_20261028.md)。新增 `assembly_coordinate_transport.py` 仅无损规范化精确组/对象ID容器，严格保留原数值、宿主契约与冲突检查，复用厨房14/33、休息区18/28、卫生间8/13、卧室12/23坐标；剩余字段按最多6对象/批、两轮有限补问。SCM/概率/尺寸/物理编译未改，六成功世界冻结。新增脚本 `repair-coordinates GPU3 qwen_edit_pano/outputs/organized_multiscene_named_prepared_20261028_001`，fork新目录、检查所有API已完成，只跑Qwen worlds并自动report，**本次不用API Key**；`prepare-coordinates`仅准备，`resume-coordinates`运行已准备新目录。不要普通resume旧目录。审计与CPU回归在 `../research/audits/multiscene_named_recovery_20261028/release_002/`。上一版32测试未覆盖真实容器变体，本次补实际回复回放；不要把CPU通过或API导出解释为因果模型收益认证。

2026-10-28 **五图恢复再次定位与修复**：见 [MULTISCENE_NAMED_COORDINATE_FIX_20261028.md](MULTISCENE_NAMED_COORDINATE_FIX_20261028.md)。来源 `organized_multiscene_recovery_20261027_225835_2897903`：仍有3成功世界；四个Qwen保存库存厨房33/休息区28/卫生间13/卧室23，当前故障是 compact numeric rows 的宿主行列数混淆，不是空库存。新增 `assembly_named_geometry.py` 使用具名字段，保留合法整行与组距离，只有缺失/无效坐标进入有限补问；不截列、不补默认值、不重抽尺寸、SCM或库存，原物理编译不改。新增 `assembly_api_pacing.py` 仅主线封装，三份API机制的HTTP429按3秒间隔＋最多3次有上限退避恢复，已有accepted缓存复用；原 `world_api.py`/RGB直接基线未改。恢复入口fork支持从本恢复源继续建立新目录，旧成功世界与JSON保护核验。运行 `bash qwen_edit_pano/scripts/organized_multiscene.sh repair-generate GPU3 qwen_edit_pano/outputs/organized_multiscene_recovery_20261027_225835_2897903`，仍需API Key，默认相同五图各后端一个世界并自动可视化。不要直接resume旧源码绑定目录。此修复是I/O及传输恢复，不构成因果模型质量改善证据；真实GPU/API由用户执行，CPU回放及审计见 `../research/audits/multiscene_named_recovery_20261028/`。

2026-10-27 **新增RGB直接生成基线**：用户明确要求无因果模块、无WildDet3D，原局部图直接给Qwen/API生成结构化完整世界。见 [DIRECT_RGB_BASELINE_20261027.md](DIRECT_RGB_BASELINE_20261027.md)，独立入口 `scripts/direct_rgb_world.sh run GPU3`，默认同五图，每图每后端1世界；DeepSeek固定硅基流动V4.1-Flash。新 `direct_rgb_world.py` 不依赖主流程/WildDet预检；直接输出11列对象数值行与描述性关系，最多1次格式纠正，无观察器/G_obs/D/CPT/布局优化、无删对象或移动/缩尺寸来换通过。只做共享语义映射及yaw→AABB确定表示转换，几何错误保留为诊断。新 `direct_rgb_review.py` 复用三色编号类别可视化，支持 `report DIRECT_ROOT CAUSAL_SUITE` 按原RGB像素/相机核验后统一范围比较；完整流程未通过候选也明确展示，门槛不同不合并成功率。可设置CAUPANO_DIRECT_REFERENCE_SUITE，生成输入只读suite中RGB/ID/HFOV，不读观测或隐状态。默认1初始/至多2格式请求，10初始/至多20请求（五图两模型），API真实发image_url，种子不传服务端，不自动文本降级。这个对照同时移除了视觉几何/因果/求解及调用预算，不能单独证明因果收益，H-cut仍保留。测试与准备在 `../research/audits/direct_rgb_baseline_20261027/`；真实基线GPU/API未运行。原五图因果恢复运行现有 `outputs/organized_multiscene_recovery_20261027_225835_2897903`，本次仅读取metadata，没有修改原生成文件或中断其任务。

2026-10-27 **真实三图错误修复＋扩展五图**：见 [FIVE_SCENE_IO_FIX_20261027.md](FIVE_SCENE_IO_FIX_20261027.md)。实际 `organized_multiscene_20261027_203624_2435093` 为single模式；休息区观察器两次8192token重复花瓶并截断，所有后续缺文件是连锁；客厅Qwen21/API20均成功，卫生间API13成功，Qwen13件尺寸已定但初次位置全null、重试仅yaw0。不是空库存或简单token上限不足。新增 `assembly_io_recovery.py`：可见实例按最多8条/块、显式done、原64实例计算预算、最多8块×2格式尝试，不复制截断坏文本、不输出未完成G_2d；数值按固定group的typed numeric rows输出，不再给null模板，保留原dim和全局编译。`assembly_multiscene.py`增加前置gate，缺依赖不启动模型；有效部分提案是警告而非整个阶段致命失败。默认数据改indoor_val5，保留原3图＋实际看过RGB的卧室zsNo…ead481…、厨房8194…ae915…。误标bedroom的zsNo…ed864…实际是沙发，没有采用该错误标签。推荐 `scripts/organized_multiscene.sh repair-generate GPU3 outputs/OLD_SUITE_ROOT`（实际从工作区执行源路径须带qwen_edit_pano/）；新增fork新目录、成功3世界及CPT/语义/尺寸复用、卫生间Qwen仅坐标恢复、原休息区观察重做、2新图全流程。每图每后端1状态，总10份计划（3复用＋7待生成/恢复），仍要API Key。不改SCM、权重、概率平滑及物理求解器。恢复批不是全新配对能力实验；需要全新五图可 `generate GPU3`。原目录不改，不启动真实GPU/API。最新代码和验证在 `../research/audits/multiscene_io_fix_20261027/`；旧prepare因代码哈希变更不能直接resume。

2026-10-27 **明确单图单世界的实际输出目标**：用户质疑三个分支/四候选是否偏离任务。它们是研究评估配置，不是交付要求；推荐入口改为 `scripts/organized_multiscene.sh generate GPU3`：每个输入、每个后端只运行 end_to_end、只生成一个完整状态并保留完整可视化，默认三张输入×两后端共六份。`prepare-single` 无模型调用；原 `run` 保留三分支四候选的评估模式。没有改变 H/A/V 概率或几何方法，内部概率估计的draws与最终世界数分开。已同步多图文档；旧prepare_002记录因入口代码变更哈希过时，不用于继续。当前只补输出配置，不自动启动GPU/API。

2026-10-27 **三组新图测试与组会可视化已实现**：见 [MULTISCENE_V13_GROUP_MEETING_20261027.md](MULTISCENE_V13_GROUP_MEETING_20261027.md)，入口 `scripts/organized_multiscene.sh run GPU3`。默认项目 indoor_val3 三张不同局部RGB（公共休息区、另一客厅、卫生间），新预测共享Qwen2D＋官方WildDet3D观测，每图重新提案独立D，不沿用开发图目录；原v13主机制/CPT、尺寸冻结和缺距恢复不改。两后端为Qwen3-VL-8B与硅基流动DeepSeek-V4.1-Flash，同图共享O/D、同方法配置；这次新测试需要API Key。三分支×两后端×四世界×三图，计划72世界；新图没有旧geometry_control和历史旧H probe。支持resume复用已完成阶段、不追加抽样挑成功。49项集成CPU回归与最新15项多图/可视化回归通过；109代码哈希一致。只做了prepare，**真实三图GPU/API生成未运行**，最终prepare-only为 `../research/audits/multiscene_v13_20261027/prepared_002/`（001因标签改进代码哈希已过时，不用于执行）。已有真实开发图8个主线候选已重绘至 `outputs/organized_presentation_20261027_002/index.html`，含原RGB、编号+类别XZ图、原图重投影、六方向投影、静态/交互3D、1600×800 ERP状态图与局部RGB-ERP叠加、CSV及组会讲稿/幻灯片。取消紫色：观测蓝、所有可映射新增橙、未映射新增红；旧观测若只有粗类，写sofa[furniture]而不改原ID。未知ERP灰色，未生成隐藏RGB。已有2929来源JSON未变，8世界投影方向/全部ERP比例检查通过、115本地链接有效。原26成功G_full及2732受保护副本仍未变；生成模型/原求解器没有修改。最新验证 `../research/audits/multiscene_v13_20261027/release_002/release_validation.json`；旧风格presentation_001仅保留历史，最终使用002。

2026-10-27 **真实距离恢复结果已核验**：运行 `organized_range_repair_20261027_183331_2074997`，六例用六次 Qwen 请求恢复 19 个距离，32/32 导出；26 个原成功状态逐字节相同，概率/语义库存/尺寸保持，5,664 份来源 JSON 审计前后未变。本轮生成源码未改、没有再调用模型。主线 Qwen `[21,7,9,23]`，后方29/60、中央后方11、后方根12/26；API `[21,8,7,21]`，后方7/57、中央0、后方根4/25。API 四次无显式 rear 的模型内概率约.07325，且控制分支后方可以生成，不加后方/数量配额。新恢复 Qwen world04 边桌声称靠远沙发，实际 gap3.452m，rear 单对象下界3.0519m：距离修复不能解决，优先核对服务目标/关系类型/同装配空间计划粒度；电视观看距离不能套边桌阈值。API 边桌原有错误引用与Qwen电脑深度.05m等仍保留待查。全部 summary 的 layout_needs_review 主要涉及未认证墙面/层板/坐面安装，不能当成新数值失败。建议固定因果骨架与完整装配，必要时只局部处理功能—空间绑定和形态轴，再做新图独立候选的免训练验证。详见 [V13_RANGE_RECOVERY_RESULT_REVIEW_20261027.md](V13_RANGE_RECOVERY_RESULT_REVIEW_20261027.md)；逐对象审计与六例图片在 `../research/audits/range_repair_result_review_20261027/actual_001/index.html`。下文“真实恢复尚由用户执行”为前一实现阶段状态。

2026-10-27 **v13 距离字段局部恢复已实现**：见 [V13_RANGE_ONLY_REPAIR_20261027.md](V13_RANGE_ONLY_REPAIR_20261027.md)，入口 `scripts/organized_range_repair.sh repair GPU3 outputs/organized_completion_20261027_152654_1370933`（从工作区执行时源路径须带 `qwen_edit_pano/`）。没有修改 H/A/V/E/S/R 机制或原概率表：只有缺失 range 时保留初次有效回复、另询问紧凑距离补丁，严格校验 ID/有限正数/不夹带其他字段，再合并原字段。六份实际初始回复共 19 个缺失距离；恢复入口复制到新目录，26 个原成功 G_full 逐字节复用，语义世界和冻结尺寸不变，原 summary/失败回复仍保留。当前只需六个 Qwen 初始请求、最多 12 次格式尝试，API 待修零，故本次不需要 API Key，也不是新一轮两后端能力对比。继续用原数值编译器，数值不通过保留完整候选；不自动改服务对象、不改可疑尺寸、不填默认距离/朝向。未来完整布局回复不增加距离调用。真实模型恢复尚由用户执行；CPU 人工距离回归不代表真实质量改善。验证与 prepare-only 页面在 `../research/audits/organized_range_repair_20261027/`。当前模型说明、概率真实作用示例及其未校准/未识别因果局限均在新文档中。下文“生成源码未改”为前一只读验收阶段状态。

2026-10-27 **v13 实际验收复核**：实际 `organized_completion_20261027_152654_1370933`，见 [V13_WORLD_STATE_ACCEPTANCE_20261027.md](V13_WORLD_STATE_ACCEPTANCE_20261027.md)。主线 Qwen 计划 `[21,7,9,23]` 仅 2/4 导出，API `[21,8,7,21]` 4/4；全部分支 Qwen 10/16、API 16/16，26 个完成状态零残留框碰撞。Qwen 六失败的**最初原因均为 range_m:null**，最初 yaw/偏移已填，整份修复回复又复制成空模板，不能只看最后 yaw 错误。原库存 geometry_control 两路均 4/4，支持联合布局进展。API 第四世界边桌到声明远沙发 gap2.544m、当前 right 方位的解析/凸优化下界2.5436m，单调数值求解无法改善；但它距另一可见沙发仅.475m，实际位置可能合理，首先核对服务对象绑定，不能据错误引用否定整世界或事后自动改解释。API尺寸较协调，Qwen开电脑深度.05m等仍有形态疑点；无统一资产正面定义不能认证电视朝向。新机制 true76/80、75/80，旧H新接口均58/64，对比旧64/64，不把变化全归功于H。建议固定主体模型，仅局部处理缺失距离字段、关系引用与具体尺度，再做不同输入验证；不加物体/后方配额、不扩成CAD物理项目。本轮生成源码未改，无新模型/API/训练，4,341 份来源 JSON 未变，22 实际状态可视化在 `../research/audits/v13_result_review_20261027/actual_001/index.html`；六失败精确字段已保存 `missing_ranges_plan.json`，没有填默认距离或朝向。下文 v13“真实尚未执行”等为实现阶段历史状态。

2026-10-27 **v13 执行代码已实现**：见 [ORGANIZED_FUNCTIONAL_MODEL_V13_20261027.md](ORGANIZED_FUNCTIONAL_MODEL_V13_20261027.md)，入口 `scripts/organized_completion.sh compare GPU3 V12_RUN [SEED]`。固定实际 v12 同图丰富域；H 改为五种软功能组织假设，对称交换存在状态标签/顺序，概率实际采样，另有旧 H+新存在接口诊断。尺寸先估计并冻结，再联合估计朝向与宿主归一化局部坐标；MILP 联合分离同宿主物品，保留原严格全局检查。四分支 end_to_end/layout_control/function_free/geometry_control，最后一个精确重放 v12 的原 layout_control 库存与方位、重新估计几何；默认两后端同方法同配置、硅基流动 DeepSeek-V4.1-Flash。另有 `replay V12_RUN` 纯 CPU 固定旧尺寸重放：原 Qwen 四失败的碰撞对 4/8/1/4→0/0/0/0，API 四次仍通过，1,606 原 JSON 未变；不代表错误尺寸或真实世界语义已修复。236 项完整 CPU 回归及额外 CLI 错误路径测试通过，最终 prepare-only/103 哈希通过。无真实 v13 GPU/API 调用、未训练，不声称新 H/概率/尺寸质量已验证。验证目录 `../research/audits/organized_v13_20261027/release_001/`，原尺寸重放 `../research/audits/organized_v13_20261027/old_size_replay_001/index.html`。默认本图每后端 228 个机制/接口初始请求，外加最多 32 个几何初始请求，预算和失败都公开，不设数量/后方目标。下文为前一阶段研究与历史结果。

2026-10-27 **v12 后续联网调研与优化顺序**：新增 [V12_RESEARCH_OPTIMIZATION_PLAN_20261027.md](V12_RESEARCH_OPTIMIZATION_PLAN_20261027.md)，核验 S-AOG/HSG、FactoredScenes、SD-SCM 局限、LLM 选择偏置与不确定度、Holodeck/LayoutGPT、ReSpace、SceneSmith、SceneEval/FloorplanQA 及 ATISS/MiDiffusion。核心建议：保留完整功能装配骨架，先检验 H 的 possible-affordance 语义和全真存在接口，再在冻结库存上改进尺寸/同宿主联合坐标；最后独立验证新提案及跨图效果。API 四次都未选明确 rear，按冻结 p=.4285317 的 IID 模型概率约 .10665，四样本不足以支持新增后方配额。此为研究设计，生成源码未改、无新 GPU/API/训练，不将假设机制或学习式论文借鉴冒充已验证能力。

2026-10-27 **v12 实际结果已复核**：`compositional_completion_20261027_105105_496519` 为固定 v5 同图语义目录模式，没有重新提案。端到端 Qwen `[21,8,9,23]`、API `[21,8,7,24]`，各 4/4 导出；后半空间分别 30/61、11/60 条新增记录，中央后方 11、0。相同库存布局对照逐项确认语义状态和文字提示相同，Qwen 因 4/8/1/4 对残留碰撞全部阻断，API 4/4 通过。两后端 128 次程序存在判断全部 true，所有 H/角色的存在率均 .9；Qwen h1/h2/h3 变体行相同，H 条件作用弱，不能把内容恢复归功于因果收益。原输出有 0.4m 高写字桌、过薄电脑、声称沙发旁但距离数米的关系。保留 v12 装配框架，下一步定位存在机制饱和与尺寸/宿主局部坐标接口，不增数量或后方配额，不推倒 DAG。本轮仅审计，生成代码未改，无新 GPU/API/训练。详见 [V12_ACTUAL_RESULT_REVIEW_20261027.md](V12_ACTUAL_RESULT_REVIEW_20261027.md)；20 个实际/失败状态的统一可视化 `../research/audits/v12_result_review_20261027/actual_001/index.html`，5,730 份本轮来源 JSON 未变。下方“真实 v12 尚未执行”为实现阶段历史状态。

2026-10-27 最新 **v12 完整功能程序组合生成**：用户要求回看好几个版本以前的丰富案例并交付实际修改。已重新核对 v5/v7/v8；v5 API `[23,23,23,22]` 四次导出，Qwen `[24,23,23,14]` 仅一次导出；v8 两路均四次导出、内容以完整功能装配为单位。已实现 `assembly_compositional.py`：H→各角色的实际库存存在 A_r，(H,A_r)→完整装配及相机计划 V_r，再→E/S/R→新几何；存在 Beta 先验固定独立于候选数，选项用中性别名并在 H 间配对 seed/order；同一功能程序允许多个区域/实例。**建议先执行同图丰富语义目录对照**：使用 v5 的全部 15 个已生成语义程序/70 条实例，按现行接口重新验证；重新估计全部 H/A/V/R 概率和新尺寸/布局，不复用旧 CPT 或旧几何，不宣称新自主提案能力。也已实现重新提案模式：清空 CAUPANO_PROGRAM_CATALOG 后，两后端从固定局部图像/观测重新提出完整功能程序，再使用同一 H/A/V/R/X 机制；不读取旧语义目录。两模式须分别报告，固定观察器不重跑。保留相机方向、宿主、视锥和支撑代理，默认两后端都使用相同文字观测几何；可选匹配 RGB 布局。入口 `scripts/compositional_completion.sh`，三分支×两后端，API 固定硅基流动 DeepSeek-V4.1-Flash。详见 [COMPOSITIONAL_FUNCTIONAL_PROGRAM_V12_20261027.md](COMPOSITIONAL_FUNCTIONAL_PROGRAM_V12_20261027.md)。225 项 CPU 回归通过，真实 prepare-only/97 哈希、43 个合法中心空间计划及合成执行可视化通过，来源 6,083 JSON 未变。旧丰富案例全量可视化在 `../research/audits/compositional_v12_20261027/actual_history_001/index.html`。**真实 v12 尚未执行，未训练、无新 GPU/API，未声称质量提升。** 下面“本轮保持主生成代码不变”等为上一轮诊断阶段历史状态。

2026-10-27 最新 **v11 实际根因审计**：`holistic_completion_20261026_235517_1710340` 仅完成 Qwen 四分支 16 世界；API 提案/识别遇 HTTP 500，API 五分支和 pooled Qwen 共 24 世界依赖阻断，不能作后端质量比较。Qwen 端到端 `[1,1,2,2]`、E[N]=1.81，6 条新增中两条在左后方，中央后方无对象。6 单位/11 实例的 D 因 API 两个提案缺失和 Qwen 接口/观测重述而缩水；Qwen 仅贡献 1 实例。直接库存 48/48 单选、47/48 选择随机菜单第一项，全部 left；原始 148 次选择没有 rear，声明 rear 的 .026604 概率全部来自平滑尾项。选择菜单省略了原有 description/spatial_reason/placement，工作区“相机对面”信息未传入实际决策。端到端都是无宿主地面物体，不能把内容少归因于新支撑约束。**本轮保持主 DAG/概率/生成源码不变**，新增固定 D 的格式/信息/选项顺序及配对 H 小对照，见 [V11_RESULT_ROOT_CAUSE_REVIEW_20261027.md](V11_RESULT_ROOT_CAUSE_REVIEW_20261027.md)；审计 `../research/audits/v11_result_review_20261027/actual_001/index.html`，28 个实际世界同范围对比，5,261 份来源 JSON 未变。诊断代码在同目录 `selection_probe.py`，无新模型调用；不能再把 CPU 214 项通过解读为 LM 已可靠实现联合采样。以下“真实 v11 尚未验证”等以本条和原始记录为准。

2026-10-26 最新实现 **v11 联合功能库存与全景空间组织 SCM**：借鉴真实 v9.1 的“方位随方案采样且受保护”，保留 v10 同功能可共存装配，改为 `H,O_positive,D -> joint V=(B,R)`。无固定宿主的未指定装配展开合法 left/right/rear 备选，在联合机制中真实赋概率，不再让 near 数值搜索选其方位；同一对象跨方位保留实例身份。原深度关系不被改写，固定观测宿主仍固定。主库存用正面事实，增加同文字/规范本地 seed 的 RGB 对照；API 不传 seed。增加 observed_host_name 身份核对、有限 rigid_box 中心支撑代理及主状态朝向/固有尺寸输出。空集、重复、失败全部保留，不设数量/后方目标。入口 `scripts/holistic_completion.sh`，五分支×两后端；详见 [HOLISTIC_FUNCTIONAL_SPATIAL_V11_20261026.md](HOLISTIC_FUNCTIONAL_SPATIAL_V11_20261026.md)。214 项 CPU 回归通过、真实 prepare-only 与 92 项哈希匹配；无新 GPU/API/训练，**真实 v11 质量尚未验证**。旧 4,280 份 JSON 未变；原 API 9 对象固定尺寸重放通过支撑代理，Qwen 20 对象重放被阻断但完整候选未删。下面“不改生成版本”等为前一轮审计的历史状态。

2026-10-26 最新真实 **v10 质量审计**：用户已运行 `assembly_set_completion_20261026_195138_370210`。端到端 Qwen `[0,0,0,20]`、API `[9,15,10,10]`；累计后方记录分别 3/20、0/44，全部五分支 40 次虽导出但不代表世界合理。真实 D=11 单位/21 实例（旧回复重放 19/45 不是实际新候选）；仅一个明确 rear 装配，Qwen/API 存在率 .156/.148；7 个 unspecified 交给近邻数值搜索，媒体装配还被正 Z 深度顺序限制。Qwen RGB 条件最大 H 四次全空，整体 P(empty)=.688；同域 tag-only 更丰富但阶段 seed 不同，不能当成相同噪声反事实或质量胜出。还确认了遥控器宿主错误绑定 picture frame、Qwen 20 物体样本全部 12 个承载物仅边缘接触、功能桌灯中心距约 3 m、固有尺寸与观测地面误差等问题。详见 [V10_WORLD_QUALITY_REVIEW_20261026.md](V10_WORLD_QUALITY_REVIEW_20261026.md)。本轮**不改生成版本**，建议先验证库存 RGB 证据处理，再优化联合功能空间 R，而不是提高数量/后方配额。只读审计在 `../research/audits/v10_world_review_20261026/actual_002/index.html`，24 世界同坐标可视化、全部 40 次逐项检查；4,280 份来源 JSON 未变，无新 GPU/API/训练。注意 `expected_hidden_from_reference` 已包含 uniform tail，不能重复相加。以下“v10 尚未执行”等句均为历史阶段状态。

2026-10-26 最新：真实 v9.1 `panorama_completion_20261026_181035_3984837` 已复核，端到端 Qwen `[7,6,9,9]`、API `[7,7,9,9]`，全部 32 次布局导出；同域直接库存 E[N]=15.43/12.85，功能机制仅 8.0052/8.5716。根因包含同功能/模式的一选一槽位（两侧沙发的阅读装配不能共存）、茶几被额外活动区门控、粗 H 的图像信息瓶颈及独立组件连带落选。已实现 **v10 观测条件联合装配集合 SCM**：O,H,D→联合 B，取消功能槽互斥；候选按依赖闭合单位校验，合法跨组宿主组件可合并，最终采样库存不删物体；保留相机区域和数值契约。五分支×两后端（增加选择阶段 RGB 消融），联合选择初始请求数/输出上限匹配直接对照。详见 [JOINT_ASSEMBLY_SET_V10_20261026.md](JOINT_ASSEMBLY_SET_V10_20261026.md)，入口 `scripts/assembly_set_completion.sh`。204 项 CPU 回归通过，真实观测 prepare-only 通过；原回复仅作候选重放得到 19 个去重单位/45 条实例，**未执行真实 v10 GPU/API，不宣称生成质量提升**。来源 12,690 份 JSON 未变，实际对比与验证在 `../research/audits/assembly_set_v10_20261026/actual_v91_001/index.html`。不训练、不加数量/后方配额，不把这套条件生成图当作已识别的物理因果真相。下列条目为历史状态。

2026-10-26 最新内容退化复核：真实 v9 `panorama_completion_20261026_141314_2619530` 已运行。端到端采样 Qwen `[9,9,9,9]`、API `[9,9,9,12]`；候选域从 v8 57 个实例/16 个装配降至 26/8，本地四个提案阶段全失败、共同域全部来自 API。媒体装配从 7 个缩为 3 个；API 激活渠道数略增，仍发生数量下降，不能笼统解释为存在率保守。后方记录从 v8 的 1/6 增至 15/18，说明空间修复有效但内容退化。已仅做 **v9.1 提案接口修复**：单一 schema、已有 spatial_reason 绑定旧 reason、保留 group 内关系、同时报告缺字段及具体区域冲突；不改 DAG/概率/数量目标。191 项 CPU 回归通过；原回复重放找回 3 个 Qwen 完整方案；冻结域八条分支方程/概率/同 seed 库存完全不变。未执行修复后真实 GPU/API。详见 [V9_CONTENT_REGRESSION_REVIEW_20261026.md](V9_CONTENT_REGRESSION_REVIEW_20261026.md)；实际 v5–v9 对比在 `../research/audits/v9_content_review_20261026/result_001/index.html`。本轮保留 10,159 份来源 JSON。下方各版本条目为历史阶段记录，“未执行”状态以本段和最新原始产物为准。

2026-10-26 最新空间审计与实现：真实 v8 `mechanism_completion_20261026_120551_1879296` 端到端累计 Qwen 50 条新增记录仅 1 条中心在后半空间、API 56 条仅 6 条，中央后方分别 0/1。新组贡献已约 46%/53%，不是统一调高数量的问题。6 个新组中 4 个保留现有相机深度关系时不可能全组置于后方。已实现 **v9 功能—空间程序**：V 包含显式相机方位 R，V→R→X 进入真实概率与受保护几何；functional_links 的 serves 与 near 分开，后方无配额。详见 [PANORAMIC_SPATIAL_MODEL_V9_20261026.md](PANORAMIC_SPATIAL_MODEL_V9_20261026.md)，入口 `scripts/panorama_completion.sh`。实际 v8 只读角度审计在 `../research/audits/panorama_v9_20261026/actual_v8_001/index.html`。不训练，不复用旧 D/CPT，不将合成后方 fixture 当作真实 v9 效果。

2026-10-26 v9 验证：全套 CPU 回归 186/186 通过；真实观测 prepare-only 已完成，83 项代码哈希匹配；八分支、16 次人工合成世界验证及 428 处本地 HTML 链接检查通过。真实 v8 的 2,173 份 JSON 未变。**真实 v9 GPU/API 尚未执行**，尚不能声称补全质量已改善。详见 `../research/audits/panorama_v9_20261026/release_002/release_validation.json`。

2026-10-26 上一阶段 **v8 免训练参考收缩**：保留功能 SCM 的核心变量，以 D-only 冻结模型参考机制定义激活平滑基分布，概率真实进入采样；增加 O,D→联合库存的直接对照，并修复完整备选整份拒绝、移除数值回复中冗余 region 字段。详见 [NO_TRAINING_MECHANISM_V8_20261026.md](NO_TRAINING_MECHANISM_V8_20261026.md)，入口 `scripts/mechanism_completion.sh`。四分支×两后端，固定硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`；不训练。174 项 CPU 回归通过；真实 v8 已执行并在以上 v9 空间审计中复核。直接库存对照保留 H 作为匹配几何上下文且总调用预算不同，报告中明确标注；不得宣称已证明因果增益。旧结果仅复用观测，新 D/机制必须重新估计。

2026-10-26 用户最新硬约束：**不训练，继续优化因果推理模型。** 不训练/微调网络，也不从场景训练集拟合装配库或概率先验。使用冻结 Qwen/硅基流动 DeepSeek 做推断阶段条件采样、显式概率计算与机制诊断。当前路线和已完成的平滑敏感性分析见 [NO_TRAINING_CAUSAL_DIRECTION_20261026.md](NO_TRAINING_CAUSAL_DIRECTION_20261026.md)。此前建议中的训练/学习式装配库仅为历史备选，不是当前计划。新诊断不改生成代码，无新 GPU/API 调用；本轮没有发布新生成版本。

2026-10-26 最新实证：真实 v7 `realization_completion_20261025_224043_2053447` 已只读核验。两路端到端各 4/4 导出；全部六分支为 22/24，两个失败是 Qwen 固定库存布局的 `region=front`。下一步建议**暂时固定 v7 核心方法，先作跨场景/概率/功能验证，不立即造 v8**。单张开发图没有证明因果质量增益；Qwen 完整库存分布与保边缘 H 消融相同，API 联合库存 TV≈0.0712。提案的整份拒绝丢了 3 个单独合法的同伴方案，属于接口候选覆盖问题，未修改旧域/旧表。详见 [V7_FREEZE_AND_VALIDATION_20261026.md](V7_FREEZE_AND_VALIDATION_20261026.md) 和 `../research/audits/v7_freeze_review_20261026/result_002/index.html`。本轮生成代码未改、无新 GPU/API 调用；1,959 个来源 JSON 和 323 个代码哈希未变。以下“真实 v7 尚未运行”等句为历史阶段状态，以本段及原始结果为准。

2026-10-25 当前最新为 **v7**：已复核真实 v6 `activation_completion_20261025_202940_1307023`，数量下降主要来自办公存在率与候选装配变化，不恢复数量配额。新增“可见组延伸/新功能组”两个可共存渠道，分离识别与诊断 H 域，并修复方向软惩罚、功能组共用几何方向、近邻优化与提案失败取消后续计划等根因。详见 [ANCHORED_REALIZATION_MODEL_V7_20261025.md](ANCHORED_REALIZATION_MODEL_V7_20261025.md)，入口 `scripts/realization_completion.sh`。真实 v7 语义模型尚未运行；原 v6 库存 CPU 重放不等于新模型效果。下列 v6/v5 更新为历史阶段记录。

2026-10-25 当前最新：已基于真实 v5 根因实现 **v6 联合功能区域激活→条件装配变体→实例存在/安装→条件几何**。详见 [JOINT_ACTIVATION_CAUSAL_MODEL_V6_20261025.md](JOINT_ACTIVATION_CAUSAL_MODEL_V6_20261025.md)，入口 `scripts/activation_completion.sh`。增加真实库存概率干预、匹配 H 支持、保边缘库存 H 边消融、严格残留碰撞门槛、数值缺失完整反馈和绑定 MP 37 类对象词表的表外着色。没有引入未经测量的房间容量 L；尚未运行真实 v6 GPU/API 生成，不能把 CPU fixture 当成效果验证。API 仍固定硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`。

2026-10-25 先前实证复核：实际 v5 输出 `modular_completion_20261025_162311_4103717` 已完成只读审计。发现角色近乎全激活、H 主要影响替代项、本地数值 null 失败、提示词关系 schema 冲突、部分成功计数误报，以及共享候选来源的能力混杂。详见 [MODULAR_V5_RESULT_REVIEW_20261025.md](MODULAR_V5_RESULT_REVIEW_20261025.md)。以下 v3/v4/v5 段落为当时记录，版本与“尚无结果”状态均以本段和实际输出为准。

2026-10-25 后续最新：已核对实际 DeepSeek 对照 `sdscene_compare_20261025_151733_3744172`。本地有非空提案，但库存选择 14/15 为空；API 为 0/15。已实现纯功能变量、可选联合装配语法 v5、真实 H→装配边消融，以及垂向关系投影与只读观测坐标。详见 [FUNCTIONAL_ASSEMBLY_MODEL_V5_20261025.md](FUNCTIONAL_ASSEMBLY_MODEL_V5_20261025.md)，新入口 `scripts/modular_completion.sh`，固定硅基流动 `deepseek-ai/DeepSeek-V4.1-Flash`。133 项 CPU 回归通过；尚无真实 v5 GPU/API 生成结果。

2026-10-25 最新更新：已核对 `population_completion_20261025_115554_2925396` 的实际结果，修复具体类别计数问题，实现 SD-SCM 启发的经验条件机制 v4 与本地/API 共同观测、共同候选、固定库存布局对照。请先读 [SD_SCENE_MODEL_V4_20261025.md](SD_SCENE_MODEL_V4_20261025.md)，执行入口为 `scripts/sdscene_compare.sh`。122 项 CPU 回归通过；尚无真实 v4 GPU/API 推断结果，不能把合成 fixture 当成效果验证。

2026-10-25 后续更新：已实现完整库存概率模型 v3 与三路同候选可视化入口 `scripts/population_completion.sh`，请结合 [POPULATION_CAUSAL_MODEL_V3_20261025.md](POPULATION_CAUSAL_MODEL_V3_20261025.md) 和当前源码核验。下文保留原交接记录，不代表 v3 的最新模型定义。

**本文由 AI 根据历史对话、服务器源码和实验产物整理，可能存在错误、遗漏或过时信息。项目中的不少代码和研究解释也由 AI 生成，逻辑与实现都可能有问题。请独立思考，以当前源码、配置、原始模型回复、日志和可视化为依据；对研究方法与技术事实查阅论文和官方实现。交接文档提供线索，不替代判断。**

## 1. 用户真正要完成的任务

从一张局部室内 RGB 图像，推断与观测相容、语义与空间布局合理的完整 360° 世界状态。未来把世界状态注入 World Adapter 和 Qwen-Pano，生成 2:1 ERP 全景图。

单张图不能唯一确定镜头外的真实世界，因此输出是合理的完整世界假设，可以有多个候选。

当前工作重点是：

1. 识别图像中已经可见的物体，建立观测状态 `G_obs`。
2. 推断原始输入视野之外的周围物体及其三维状态，得到新增状态 `ΔG`。
3. 保留 `G_obs`，形成 `G_full = G_obs ∪ ΔG`。
4. 可视化实际放置的物体框、周围布局和 360° 方向分布。

例如图像只拍到一张床，补全可以假设侧方或后方存在衣柜、椅子、床头柜等，并赋予合理的类别、中心与尺寸。物体种类和数量由具体场景假设决定。

用户很在意以下几点：

- 隐藏世界补全应覆盖周围空间，不能退化成继续给输入画面中的家具添加东西。
- 推断已有生活场景，保留合理的生活物品、储物和空间组织；不能把任务改成最简功能房间设计。
- 对象数量与对象合理性一起评估。用户曾觉得补全太少；后来出现很多遥控器等重复小物体，也不算改善。
- 因果生成机制是研究主线。程序通过更多检查，不等于方法更正确。
- 每次改动要有具体问题证据、方法理由和验证结果，避免反复叠加补丁。
- 用户手动执行 GPU/API 实验，需要提供完整命令和物体可视化。

**当前阶段只做世界状态与可视化，尚未接通 Adapter，也不生成 RGB ERP。** 可视化中的 2:1 图片是世界状态方向图。

## 2. 项目背景和服务器位置

- 主仓库：<https://github.com/skyman1000/qwen_edit_pano>
- 工作区：`/data-nfs/gpu1-2/u13529658780/DIT360`
- 主项目：`/data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano`
- 当前补全代码：`qwen_edit_pano/world_pipeline/completion/`
- 当前执行入口：`qwen_edit_pano/scripts/assembly_compare.sh`
- 当前方法说明：`qwen_edit_pano/world_pipeline/completion/WORLD_COMPLETION_CURRENT.md`
- 研究审计：工作区下 `research/audits/`

用户之前复现过 DiT360，使用 Qwen-Image-Edit-2511 做全景 LoRA，并实现 World Encoder / Gated Cross-Attention Adapter。GT World 训练与全景生成是已有研究基础。关于 Adapter 控制是否有效，仍需要正确 World、打乱 World、无 World 等对照；组合画质改善不能单独归因于 Adapter。

早期项目还包括 `dit360`、`qwen360`、工作区中的 `qwen_pano`。本阶段以 **qwen_edit_pano 的当前工作树**为准。

截至本次交接，服务器 Git HEAD 为 `5302ceb`，提交说明是“encoder和adapter训练代码上传”；大量近期补全文件为未跟踪文件，另有其他研究任务的修改。单看 GitHub 或该提交无法得到服务器最新实现。保留现有工作树和实验产物。

执行 shell 工具时使用 `login:false`。本环境的登录 shell 初始化可能触发无关 Slurm 输出。

运行环境：

```text
Qwen / CPU 验证：
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python

WildDet3D：
/data-nfs/gpu1-2/u13529658780/.conda/envs/wilddet3d/bin/python
```

目前由用户手动运行 GPU/API 命令。接手 Codex 可以先读源码和日志、修改代码、运行 CPU 验证、生成新的审计报告。大型依赖安装、训练与 GPU/API 批量任务按用户授权执行。

## 3. 当前完整流程

```text
局部 RGB
  ├─ 本地 Qwen3-VL-8B-Instruct：可见名称、类别、2D 框
  └─ SiliconFlow API：同样的可见物体识别任务
          ↓
各分支都由 WildDet3D 预测可见物体 3D 框
          ↓
冻结的 G_obs
          ↓
Qwen / API 提出场景组织假设 Z
          ↓
同一框架下提出联合隐藏世界程序 F
          ↓
结构采样，确定新实例、宿主与安装依赖
          ↓
模型提出新增对象尺寸、姿态与相对布局参数
          ↓
程序进行实例化、支撑树布局优化和视锥判断
          ↓
G_full / 明确标注的阻塞候选
          ↓
交互 3D 框 + XZ 俯视图 + 2:1 ERP 状态方向图
```

API 模型由项目程序调用；WildDet3D、状态转换、结构模型与几何求解均由项目程序组织。API 分支依然使用 WildDet3D。

当前配置中的 API 名称是 `deepseek-ai/DeepSeek-V4.1-Flash`，通过 SiliconFlow 兼容接口访问。运行时以 `experiment.json` 和实际响应记录为准。密钥来自 `SILICONFLOW_API_KEY` 环境变量。

观测三维估计是粗基线，中心深度可能有偏差。当前先固定它，以隔离补全方法的问题。

## 4. 世界状态和坐标约定

核心字段是类别、三维中心 `(x,y,z)`、**完整 AABB 尺寸** `(sx,sy,sz)`。单位为米：

```text
x：相机右方
y：图像上方
z：相机前方
frame = local_view_x_right_y_image_up_z_forward_metres
schema = caupano.local_world.v0.1-pilot1
```

对象记录使用：

```text
track_id
category_id
raw_category / description
center_local_m
size_aabb_local_m
bbox2d_visible_xyxy_norm
in_primary_room
```

一些字段采用 `value/valid/source/...` 结构。以 `world_pipeline/common.py` 的状态检查和真实 `G_obs.json` 为准。

- 新增前后 `G_obs` 对象记录应完整保留，不为获得可行布局而移动或重估它们。
- 新增实例 ID 与已观测 `obs_*` 身份分开。同类别新实例可以存在，类别相同本身不能判定重复。
- 当前目标是新增完整 AABB 位于原始输入视锥之外；尚未实现“虽然在视锥内，但被真实几何遮挡”的可见性推断。
- 相机 Y 近似竖直、地板高度、层板高度和壁挂位置都包含假设，AABB 不是物体实际形状。
- 模型输出 intrinsic 宽高深和 yaw，编译器转换为相机坐标完整 AABB；受支撑物体偏移按宿主身份确定坐标系。
- `in_front_of` 在当前数值关系里表示更大的相机 Z；“电视面朝沙发”不自动具有该深度排序。

Adapter 训练采用有限类别映射，约 30 多类。当前补全使用开放名词，未映射对象保留在世界状态并以颜色标注。以后接 Adapter 时，再单独处理其输入词表与字段匹配。

## 5. 当前实际实现的因果框架

当前版本标识：

```text
protocol                  conditional_inventory_installation_scm_v2
model_variant             anchored_joint_delta
response_interface        observed_anchors_joint_world_delta_v2
geometry_parameterization hidden_region_domains_parent_relative_v2
```

### 5.1 联合语义生成

固定上下文 `B = RGB + G_obs + camera` 用于提出与观测相容的有限假设域。它属于近似逆向推断，不将图像解释成家具产生的物理原因。

离散机制：

```text
Z → F → E_i
E_i → S_i ← S_host
```

- `Z`：活动、生活方式和场景组织假设。
- `F`：一个完整的联合隐藏世界增量，家具、生活物品、合理的建筑附属物一起提出，包含变体、功能组、宿主和定性关系。
- `E_i`：该新实例属于当前库存。
- `S_i`：声明的宿主安装依赖是否满足；不是实际物理安装认证。

模型输出 `observed_refs` 引用已存在的观测锚点，`new_objects` 仅定义新实例。受支撑对象通过精确 ID 引用宿主。

当前一次提出 1～2 个场景组织假设，各自 1～2 个联合程序。最多 4 个语义组、24 个新实例，这是执行容量；没有对象数量下限或必加类别。

已去掉活跃路径中独立的小物品 `T` 生成阶段。`assembly_inventory.py` 中仍有旧实现和兼容逻辑，实际入口选用 `assembly_delta.py`。

对有限候选采用公开的均匀参考采样。没有定义 RGB 似然或校准的视觉后验，也没有从数据识别真实房间的因果结构。这个框架受 SD-SCM 启发，但不是对其原始 LM 概率采样实现的完整复现。

### 5.2 位置与宿主依赖

主流程先用模型给出的尺寸/相对布局初始化，再做几何优化。最新增加 `assembly_pose.py`，将选定布局拟合为可回放的确定性位置方程，例如：

```text
根对象：P_i = 假设的根位置参数
宿主对象：P_i = P_host + 相机轴相对偏移
桌面物体：y_i = y_host + (sy_host + sy_i)/2
层板/座面：y_i = y_host - sy_host/2 + fraction*sy_host + sy_i/2
```

这是**优化后拟合的位置依赖描述与敏感性检查**，尚未替代主流程的布局求解，也不构成从数据识别出的动力学模型。

实际干预记录包括：

- `host_interventions.json`：移除宿主库存，保留子物体库存并让其安装依赖变为未满足；另有固定噪声下切换 Z 的模型内查询。
- `pose_interventions.json`：精确回放选定布局；显式移动一个预测宿主，依赖子物体随动，无关对象与观测不变。不会重新调用模型或重新优化，不模拟物体掉落。

普通空间关系图、优化梯度、模型自评或生成了一张图，都不能单独证明因果推理有效。当前核心研究问题仍是：在同模型、同前端、同输出表示和同布局后端下，这些依赖机制是否比普通补全更能产生合理隐藏世界。

## 6. 最新实际实验：接手后优先检查这里

整理本文时发现，用户已经执行了上一轮修改后的新实验。这里只读取了配置、摘要和出错对象，**尚未开展本轮根因修复与完整视觉评估**。

相对主项目根目录：

```text
outputs/assembly_rgb_compare_20261024_210523_172787
outputs/assembly_rgb_compare_20261024_211032_189937
```

| 实验设计 | API | Qwen |
|---|---|---|
| `210523_172787`：各自预测观测 | 10 个观测；新增 10/10/8；三份导出，摘要均为 0 对残留 AABB 交叠 | `furniture_z0` 失败，尚未生成完整候选世界 |
| `211032_189937`：固定同一 G_obs | 10 个观测；新增 13/10/9；三份导出，摘要均为 0 对残留 AABB 交叠 | 同一 `furniture_z0` 错误 |

最新 Qwen 错误：

```text
ValueError: alternatives[1].new_objects[6](wall_hooks_001).affordances
must be a subset of ['hanging', 'seat', 'shelf', 'top']
```

检查到初次与 repair 回复都包含：

```json
{
  "key": "wall_hooks_001",
  "category": "hook",
  "variant": "wall-mounted coat hooks",
  "attachment": {"kind": "wall", "host": null},
  "affordances": ["hang"]
}
```

接口当前将 `attachment.kind` 的 `hang` 与 `affordances` 的 `hanging` 分开命名。下一位 Codex 应从这里核对提示词、类型语义、规范化及原始回复，判断如何形成一致接口，并补真实回归用例。这个错误本身还不足以证明需要修改 Z/F 的因果生成结构。

直接读取：

```text
<实验>/qwen/completion/summary.json
<实验>/qwen/completion/calls/furniture_z0.json
<实验>/qwen/completion/calls/furniture_z0_repair.json
<实验>/qwen/completion/calls/furniture_z0*.validation.json
<实验>/qwen/completion/calls/furniture_z0*.request.json
<实验>/qwen/completion/error_traceback.txt
```

最新两组物体对比页：

```text
outputs/assembly_rgb_compare_20261024_210523_172787/two_experiments.html
```

API 已导出，但多数世界仍标为 `layout_needs_review`。摘要中的零交叠不能替代对对象选择、空间关系、安装假设和周围组织的审核。

## 7. 上一轮问题、修改与验证证据

上一轮的两个源实验是：

```text
outputs/assembly_rgb_compare_20261024_192949_4040842
outputs/assembly_rgb_compare_20261024_193808_4067827
```

- API：独立观测组提议 15/15/14，全被视锥条件阻塞；固定观测组导出 15/15/15。
- Qwen：两组都提议 15/23/9；15 与 9 因 `region="center"` 被拒，修正回复仍重复原值；23 件导出但有 35 对 AABB 交叠。
- 23 件中实际有 8 个遥控器、4 本书和 4 个零食碗。
- 请求记录证明两端仅 hypotheses 阶段附图；后续对象生成/布局为文字请求，而 provider 每次新建 messages。

针对这些证据，上一轮完成：

1. 三个生成阶段都附同一观测图板，保存图像哈希。
2. `center/unspecified` 解释为未确定隐藏方向，枚举 `left/right/rear` 假设；不固定映射为 rear。数值选择记录在 `direction_search`。
3. 固定观测宿主上的物体尝试左、右、后、上、下五个视锥面；不只按当前 X 符号选一个水平面。
4. 联合生成前提供观测支持域：整个可能接触区域若在输入视锥内，就不能承载一个完整位于视锥外的新框。非空区域仅是必要条件。
5. 新增位置方程和干预回放；重复变体仅标注审核，保留完整实例。
6. 澄清相机深度关系与物体朝向的区别。

验证目录位于工作区下：

```text
research/audits/placement_mechanisms_2026-10-24/
  before/                         修改前相关源码
  tests.log                       61 项 CPU 测试通过
  preflight.log                   两个环境预检通过
  verification.json               验证与源码哈希
  replay_saved_layouts.py          原始回复 CPU 回放脚本
  replay_original_replies_verified/index.html
```

CPU 回放复用 12 份真实模型世界的语义清单、尺寸和原始相对布局提议；优化新增位置，保留 G_obs 和全部新实例。11 份满足视野外框条件，剩一份仍有不成立的观测宿主假设；12 张状态方向图均为 2:1，180 个本地链接检查无缺失。

这组 CPU 回放没有重新生成语义；不能当成新版模型的补全质量结果。新版实际生成结果已出现，见第 6 节。

## 8. 历史实验中值得记住的事实

| 历史目录 | 用途与解释 |
|---|---|
| `outputs/generative_world_deepseek_run007_layout_resume001` | 找到的早期成功 API，新增 13/13/8，周边阅读、储物、窗户/窗帘等组合较丰富，可作视觉参考 |
| `outputs/fresh_rgb_compare_20261024_132221_2406817` | 完整本地 Qwen 成功新增 5/5；API 为 13/14/13 |
| `outputs/generative_world_qwen8b_layout_run007_compare002_recovered` | 用户记得的多物体版本：语义来自 API，Qwen 负责布局，随后 CPU 恢复；不能视为 Qwen 独立语义补全成功 |
| `outputs/scm_world_val_living_run004` | 旧计数方法中 Qwen 曾输出 11/7/9，但有重复/重叠几何中心，应结合框图检查 |
| `outputs/assembly_rgb_compare_20261024_173421_3554424` 与 `174806_3614663` | 曾出现重新生成已观测物体；首份 Qwen 度量回复中 10 个尺寸精确复制 G_obs，对身份与增量边界的修复由此而来 |

历史可视化整理：

```text
research/audits/anchored_delta_2026-10-24/history.html
research/audits/anchored_delta_2026-10-24/observation_copy_audit.json
```

这些版本的观测、方法和随机输出不同，适合找线索，不能直接当成受控消融。API 对象数量没有单调下降；需要判断覆盖了哪些区域、增加了哪些合理实例、是否出现机械重复。

## 9. 当前代码导航

以下路径相对 `qwen_edit_pano/world_pipeline/completion/`：

| 文件 | 当前职责 |
|---|---|
| `assembly_experiment.py` | 初始化、重做实验、来源哈希、阶段 CLI、API/Qwen provider |
| `assembly_world.py` | 当前完整补全主流程、导出与状态页 |
| `assembly_delta.py` | 活跃联合隐藏世界机制、提示词、解析、语义诊断 |
| `assembly_inventory.py` | E/S 机制、结构采样、库存与安装干预；保留旧内容生成逻辑 |
| `assembly_model.py` | 旧模型及共享的类型、宿主、关系解析；最新 affordances 错误在此校验 |
| `assembly_observation.py` | 冻结观测宿主的必要接触/视锥支持域 |
| `assembly_parametric.py` | 新增尺寸/相对布局接口、未知方向候选搜索 |
| `assembly_spatial.py` | 宿主坐标旋转、支撑树位移传播、AABB 代理布局求解 |
| `assembly_layout.py` | 共用实例化、编译、交叠与关系诊断接口 |
| `assembly_pose.py` | 优化后拟合位置方程、回放与局部位移干预 |
| `assembly_review.py` | 物体 3D/XZ/ERP 对比页，两组实验汇总 |
| `fresh_observer.py` | 两端共用的可见 2D 识别约定、观测实例 ID |
| `fresh_wilddet.py` | WildDet3D 调用与三维状态转换 |
| `world_generation.py` | 本地 Qwen 的匹配解码与请求记录 |
| `world_api.py` | SiliconFlow API、请求/响应和用量记录 |
| `world_scm.py` | 有限结构模型检查、forward 与 do 执行 |
| `world_scm_runner.py` | 共用请求解析与一次完整纠错策略 |
| `semantic_layout_visual.py` | 交互三维、XZ、ERP 状态图 |
| `test_assembly*.py`、`test_fresh.py` | 当前入口的 CPU 回归与合成端到端验证 |

目录中还有许多历史方案。先沿 `scripts/assembly_compare.sh → assembly_experiment.py → assembly_world.py` 追踪实际调用，再判断历史模块是否仍参与执行。

## 10. 对比实验与产物的读法

两类实验：

1. **独立观测**：同一 RGB，两端各自预测 2D，再各自用 WildDet3D 得到 G_obs；反映端到端差异。
2. **固定观测**：复制同一份 G_obs 和前端记录给两端；主要比较补全与布局差异。

两端共享框架和公开解码配置：temperature=0.4、top_p=0.9、top_k=50、max_tokens=8192、中性重复惩罚。局部随机种子通常为 3407，纠错 3408，实验入口记录实际种子。API 不保证与本地模型完全相同的随机过程、tokenizer 或服务内部处理。

它们生成的候选、对象数、机制参数和具体结构会不同，因此“同一框架”不能理解成同一份数值世界模型。固定观测对照也不等于因果消融。

实验目录主要包含：

```text
experiment.json
input.png
api/ 或 qwen/
  observer2d/   G_2d.json、grounding.json、boxes2d.png
  observer3d/   G_obs.json、投影图与估计记录
  completion/
    summary.json、case.json、G_obs.json
    calls/                      请求、原始回复、纠错与校验记录
    causal_model.json、mechanism_graph.json/svg
    scene_program_mechanisms.json、structural_samples.json
    observed_support_domains.json
    world_XX/
      semantic_world.json
      semantic_review.json
      execution.json、numeric_review.json
      host_interventions.json
      pose_model.json、pose_interventions.json
      G_full.json、G_out.json
      world/scene_3d.html、layout_xz.png、erp_state_map.png
```

具体阶段失败时，部分文件尚未生成。数值阻塞候选写入 `G_full_candidates.json` 和 `candidate_world/`，会明确标注。

- `world_state_ready` 的现有成功作用域是数值候选，不是语义/物理/因果质量认证。
- `semantics_verified`、`geometry_verified`、`causal_gain_validated` 等字段仍需按实际实验理解。
- 部分世界成功时批次可正常返回并保留 failures；全部失败时返回非零退出码。
- `new_output` 只接受主项目 `data/outputs` 或工作区 `research/audits` 下的新目录。命令应从工作区运行。
- 源码哈希保护会拒绝在初始化后代码变更的旧目录中继续生成。使用新目录的 restart/retry，并读取来源记录。

## 11. 可直接使用的命令

先运行不涉及模型推理的验证：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/assembly_compare.sh test
bash qwen_edit_pano/scripts/assembly_compare.sh preflight
```

重新生成最新两组的可视化页，保留旧页面：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh review \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_210523_172787 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_211032_189937
```

**完成最新 affordances 问题的分析、修复和 CPU 回归后**，用户可复用现有观测重做两组补全：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
  read -rsp 'SiliconFlow API Key: ' SILICONFLOW_API_KEY
fi
export SILICONFLOW_API_KEY

bash qwen_edit_pano/scripts/assembly_compare.sh retry2 GPU3 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_210523_172787 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_211032_189937
```

GPU3 换成可用节点。脚本新建带时间戳的目录，重做联合补全和布局，不复用旧语义回复。

若用户选择连前端一起从 RGB 全部重做：

```bash
bash qwen_edit_pano/scripts/assembly_compare.sh run2 GPU3
```

`run2` 先做各自观测，再做固定观测；默认固定 Qwen 的观测分支。打开命令打印的新 `two_experiments.html` 查看实际放置结果。

## 12. 调研线索与下一步工作

已查阅的主要资料：

- [Language Models as Causal Effect Generators / SD-SCM](https://aclanthology.org/2025.emnlp-main.107/)；[官方实现](https://github.com/lbynum/sequence-driven-scms/blob/main/sdscm.py)。借鉴指定图、条件机制与外部执行，核对其概率与反事实语义。
- [DoWhy：图与机制](https://www.pywhy.org/dowhy/main/user_guide/gcm_based_inference/introduction.html)。图之外必须有实际条件机制；本项目没有安装或依赖 DoWhy。
- [Human-Centric Indoor Scene Synthesis Using Stochastic Grammar](https://yzhu.io/publication/scenesynthesis2018cvpr/paper.pdf)；[官方代码](https://github.com/SiyuanQi-zz/human-centric-scene-synthesis)。借鉴功能组合、支持地址与属性表示；其原方法有学习过程。
- [Holodeck 官方提示](https://github.com/allenai/Holodeck/blob/main/ai2holodeck/generation/prompts.py)与[宿主物品放置](https://github.com/allenai/Holodeck/blob/main/ai2holodeck/generation/small_objects.py)。借鉴精确宿主和语义/几何分工；房间设计目标、数量规则和跳过失败对象的策略不直接适用。
- [LayoutVLM](https://arxiv.org/html/2412.02193v2)；[官方代码](https://github.com/sunfanyunn/LayoutVLM)。借鉴语义关系、初始化和几何优化；整篇工作不能统称免训练方案。
- [DiffuScene 官方补全代码](https://github.com/tangjiapeng/DiffuScene/blob/master/scene_synthesis/networks/diffusion_ddpm.py)。借鉴保留已知对象、生成余下对象的任务边界；当前没有接入其训练模型。
- [CausalStruct](https://arxiv.org/html/2509.15249v1)。参考对象依赖、布局和渲染反馈，独立审核其因果定义及与本任务的差异。
- [SceneComplete](https://scenecomplete.github.io/)。其 RGB-D 单视图任务侧重已观测对象的完整形状，与本项目的视野外周围世界假设有所区别。

用户还提过 FairScene、CausalScene，以及师兄给的 `qwen_edit_pano/1.py`。它们属于可继续核验的参考线索，当前活跃框架没有因引用这些名称就接入相应方法。

建议接手顺序：

1. 检查工作区规则、当前源码与最新输出，确认第 6 节之后是否又有新实验。
2. 定位最新 Qwen 的 hang/hanging 接口问题，读取原始与纠错回复，给出语义一致的修复和回归验证。
3. 逐份看最新 API 的三维布局及清单，解释 `layout_needs_review` 的实际原因；比较周围覆盖、对象选择、宿主和关系，而不只比较数量。
4. 让同一验证样例的 API/Qwen 流程稳定后，冻结接口与布局后端，扩展到更多验证图。
5. 设计同一模型下普通联合补全与依赖机制补全的对照。将模型能力差异、观测误差、结构假设和布局后端的影响分别评估，再决定是否修改因果生成结构。

新的研究改动应回答：它怎样改变隐藏世界的生成分布或结构依赖，为什么有助于当前任务，用什么结果判断有效。用户希望推进合理的完整世界推断，不希望继续围绕检查器不断改版本。
