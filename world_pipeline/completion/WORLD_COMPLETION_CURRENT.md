# 当前世界补全：固定观测锚点与联合隐藏世界增量

日期：2026-10-24。本文件是当前执行说明，历史文档保留用于追溯。入口仍为 `scripts/assembly_compare.sh`。

当前字段：

```text
protocol                  conditional_inventory_installation_scm_v2
model_variant             anchored_joint_delta
response_interface        observed_anchors_joint_world_delta_v2
geometry_parameterization hidden_region_domains_parent_relative_v2
```

## 最新两组：20261024_192949 与 193808

这里记录本轮新结果，下面 17 点的结果保留作历史证据。

| 输入设计 | API | Qwen |
|---|---|---|
| 各自预测观测 `192949_4040842` | 提议 15/15/14，原程序全部因视锥条件阻塞 | 提议 15/23/9；23 对象导出，15 与 9 因 `region="center"` 阻塞 |
| 固定同一观测 `193808_4067827` | 提议并导出 15/15/15 | 同样提议 15/23/9，同样两份 `center` 失败 |

本轮直接核验请求文件，发现两端只有 `hypotheses` 附图；联合对象生成和布局均没有图像。两端 provider 每次都创建新 messages，没有前一请求的聊天记忆。因此后续生成依赖文字摘要。这是需要修复的信息传递问题，不是增加因果节点可以解决的问题。现已在三个阶段传入同一观测图板；请求中记录 canonical_image_sha256。

23 件 Qwen 世界实际包含 **8 个遥控器、4 本书、4 个零食碗**，不能用总数证明补全质量。程序保留全部实例和理由，在 `semantic_review.json` 与可视化页中标出相同宿主上的重复变体，供审核；不设类别数量上限，不自动去重。原始类别中出现 furniture/everyday belonging 等宽泛词，提示已明确具体名词；原回复仍保留。

原始回复 CPU 回放见 `research/audits/placement_mechanisms_2026-10-24/replay_original_replies_verified/index.html`。没有重新生成语义、调用 API 或加载模型。保持原清单、尺寸和 G_obs，以原相对偏移初始化并允许求解器优化新增位置；未知方向枚举多个假设，固定观测宿主尝试完整视锥的五个面：

- Qwen 两份原失败世界可编译成视野外候选；15 件仍有 4 对交叠，9 件为 0 对。23 件仍有 35 对交叠，未被包装成语义或物理成功。
- API 独立观测组恢复前两份布局；第三份仍有不可行的观测宿主假设。固定观测组三份保留。
- 共回放 12 份实际模型世界，原观测及全部新增实例均保留；11 份满足视野外框条件。**这只是布局修复验证，不是新版生成质量结果。**

观测支持域来自冻结的 WildDet3D AABB 与原相机参数。若一个宿主全部可能的接触位置都位于输入视锥内，它不能承载一个完整位于视锥外的新框。这个必要条件在联合生成前提供；不满足时纠正整个提议，不静默改宿主、搬走观测或删掉新增物体。非空支持域只说明尚未排除该假设，不证明实际有表面、空间或相应功能。

## 先看真实结果，不能将所有问题归因于“因果图太简单”

| 来源 | 实际情况 |
|---|---|
| 找到的早期成功 API `generative_world_deepseek_run007_layout_resume001` | 新增 13 / 13 / 8；包含阅读、储物、窗户/窗帘等周边组合 |
| 曾成功的完整本地 Qwen `fresh_rgb_compare_20261024_132221_2406817` | 新增 5 / 5，成功导出 |
| 用户提到的 Qwen `...run007_compare002_recovered` | 13 / 13 / 8；其语义机制来自 API，Qwen 负责布局，不能当作 Qwen 独立补全 |
| 上一轮各自观测 `assembly_rgb_compare_20261024_173421_3554424` | API 提议 10 / 16 / 13，导出 10 / 13，16 件方案因一条位于可见沙发上的毯子失败；Qwen 提议 13 / 10 / 9，均卡在布局 |
| 上一轮固定观测 `assembly_rgb_compare_20261024_174806_3614663` | API 导出 9 / 12 / 15；Qwen 同样卡在布局 |

不能由这几个不同观测/方法/随机输出断言“API 数量持续下降”。但早期方案对周边区域和对象类型的覆盖更丰富，值得保留作参考，而不是因为后来程序复杂就抛弃。真实物体图与来源见 `research/audits/anchored_delta_2026-10-24/history.html`；它是历史参考页，不是受控优劣实验。

上一轮 Qwen 的根因不只是一条角度校验：

- 它反复输出 0°、±45°，未遵守视锥外方位接口。原回答和纠错回答没有解决问题。
- 首个清单重新列出了两张沙发、四个抱枕、脚凳、托盘、落地灯、相框，与已观测库存高度重合。布局中的 13 个对象里，10 个完整尺寸精确匹配已有观测（容差 1e-12），伴有观测位置分量复制。这是重述观测的强证据，不能靠放宽方位校验后把它们搬到背后来充当成功补全。
- 同类别物体并不等于重复实例，所以程序不按类别删除物体。真正应解决的是场景生成接口把“完整布置”和“新增实例”混在一起。
- API 的 16 件失败方案有新毯子安装在可见沙发上，数值框完整落在输入视锥内。这里应修正隐藏世界假设，不能把错误检查取消。

## 这轮改动针对生成机制

保留明确的库存/安装语义，改变其上游的联合生成单元：

```text
固定上下文 B = RGB + G_obs + 相机参数
        ↓ 用于提出相容的假设域（近似逆向推断）
Z：共存活动、生活方式与场景组织假设
        ↓
F：完整的联合隐藏世界增量
   ├─ 观测锚点：仅通过 obs ID 引用，永远不重新生成
   ├─ 新实例：家具、支持物、小物品、合理的建筑附属物一起提出
   └─ 功能组、类型化宿主、定性空间关系一起确定
        ↓
E_i：该实例是否属于当前库存
S_i：该实例声明的安装前提是否满足 ← S_host
        ↓
相对几何参数 → 程序实例化和求解 → G_obs ∪ ΔG
```

代码为 `assembly_delta.py`。不再单独执行“家具 F → 小物品 T”来填满每一组；原来的 T 变量在此路径没有作用，已从模型中删除。没有增加新的潜变量节点。活动假设仍实际决定 F 的条件域；F 联合决定 E；宿主安装实际决定下游 S。

冻结模型输出 `observed_refs` 与 `new_objects`。它可以把已见沙发作为某功能区的锚点，无需再生成一张“新沙发”才能描述该区域。受支撑物品按宿主引用派生放置组。新实例类别开放，最大总容量仍为 24，没有最少数量、必加类别或“更多就是更好”的采样权重；空候选如实保留。

旧 API 方案中值得恢复的是“不同尺度对象与周边组织共同生成”的方式，不是旧版本未经校准的图像似然或自评概率。当前仍用透明的均匀参考策略选择有限候选；不声称它等于照片后验。

## 研究依据与没有实现的内容

- [SD-SCM](https://aclanthology.org/2025.emnlp-main.107/)及[官方代码](https://github.com/lbynum/sequence-driven-scms/blob/main/sdscm.py)支持指定结构、冻结 LM 条件机制与外部采样/干预。其候选 token 概率实现不同于本项目的均匀参考采样；论文未验证本任务的单图视觉后验。
- [S-AOG](https://yzhu.io/publication/scenesynthesis2018cvpr/paper.pdf)同时区分功能分组、支持地址与对象属性，并讨论过多两两关系带来的困难。这里借鉴联合组合与明确引用，不复制其学习式 MRF/MCMC，也不将普通关系边命名为真实因果边。
- [DiffuScene 官方补全实现](https://github.com/tangjiapeng/DiffuScene/blob/master/scene_synthesis/networks/diffusion_ddpm.py)的 `p_sample_loop_complete` 明确保留已知对象，只生成剩余部分。借鉴的是固定观测与生成增量的边界；没有安装、训练或加载 DiffuScene。
- [Holodeck 官方提示与接口](https://github.com/allenai/Holodeck/blob/main/ai2holodeck/generation/prompts.py)区分物体、宿主和关系。本项目不照搬房间设计目标、固定类别清单或强制填满房间的指令。
- [LayoutVLM](https://arxiv.org/html/2412.02193v2)结合数值初始化、语义关系和几何优化，说明空间推理不能只依赖一次性 xyz。这里继续使用已有优化器，不将其当作因果识别。
- 本轮进一步核对了 [LayoutVLM 实际运行代码](https://github.com/sunfanyunn/LayoutVLM/blob/main/src/layoutvlm/layoutvlm.py)、[Holodeck 宿主放置实现](https://github.com/allenai/Holodeck/blob/main/ai2holodeck/generation/small_objects.py)和 [DoWhy 图与机制说明](https://www.pywhy.org/dowhy/main/user_guide/gcm_based_inference/introduction.html)。借鉴的是明确的父对象引用和可执行机制，不复制无法放置就跳过物体的策略。
- [CausalStruct](https://arxiv.org/html/2509.15249v1)结合对象依赖、渲染评估和场景优化；它没有替本任务识别局部观测对应的唯一隐藏世界。LLM 投票与优化反馈也不自动成为真实数据中的因果效应。
- [SceneComplete 官方任务说明](https://scenecomplete.github.io/)处理 RGB-D 下被观测对象的完整形状，不等同于本任务的视野外周围家具假设。本轮没有引入该系统。

这仍是**假设式条件生成 SCM**。最明确的可执行依赖是活动假设到候选域，以及宿主安装依赖；用途到家具的真实因果关系并未从观测数据中识别。`host_interventions.json` 同时记录固定噪声下的宿主库存干预，以及改变 Z 后重新执行联合库存机制的模型内结果。后者没有生成坐标反事实，也不声称新 Z 与实际 RGB 相容。执行出干预结果不能证明因果收益。

研究主线应看隐藏世界质量：是否提供互相相容、合理覆盖周围空间的完整假设。当前代码没有充分证据证明加入这些机制已优于普通补全，需要后续同前端、同模型、同输出表示、同布局后端的消融与多个验证例，而不是继续增加节点来解释每次异常。

## 当前因果模型的具体边界

有限离散机制保持 `Z → F → E_i → S_i ← S_host`。`Z` 是活动/生活组织假设；`F` 是共同产生对象实例、变体、宿主和关系的完整隐藏世界程序；`E` 是库存归属；`S` 仅表示声明的宿主依赖是否满足。功能关系、邻近和支撑不能一律当成因果边。

连续位置另有可执行的假设机制，导出到每份世界的 `pose_model.json`：

```text
无宿主的根对象：P_i = 假设的根位置参数
有宿主的对象：  P_i = P_host + 相机轴相对偏移
桌面物体高度：  y_i = y_host + (sy_host + sy_i) / 2
层板/座面高度： y_i = y_host - sy_host/2 + fraction*sy_host + sy_i/2
世界与相机：    V_i = AABB 与输入视锥相交测试
```

相对位置参数由选定的度量布局拟合；这是条件于既定库存与拟合参数的运动依赖假设，不能冒充从数据识别出的物理因果律。优化器耦合多个位置，不把优化梯度解释成因果边。

`pose_interventions.json` 实际回放原布局，并以显式记录的位移测试 `do(P_host=...)`。依赖子物体随宿主移动，无关对象和观测不变；不重新优化场景、不调用模型、不模拟掉落，也不宣称干预后物理可行。现有 `host_interventions.json` 继续测试宿主库存移除与 Z 切换。

这让图与实际执行对应起来，但**没有证明因果先验提升了补全**。当前优先验证世界的相容性、对象与区域覆盖以及布局合理性；接口稳定后冻结实现，在多个验证图上比较同模型的普通联合补全与依赖机制补全。不能用 API/Qwen 差异代替因果消融。

## 几何职责变简单

代码为 `assembly_parametric.py`。

- LM 可选 `left/right/rear/unspecified`。实测的 center 不提供隐藏方位，因此作为未知方向保留多个候选，绝不映射成固定后方。程序预览方向组合并有界细化，记录所有预览数量、细化候选与选择理由；这是几何搜索，不是图像后验排序。
- LM 只填新增对象的完整尺寸、相对偏移和姿态。观测的绝对三维坐标不再作为数值样例发给它；编译器内部仍完整使用 G_obs。
- 根对象偏移相对组中心；受支撑物体偏移相对宿主，坐标系由宿主身份确定，程序处理旋转。观察宿主没有已知 yaw 时采用相机轴。
- 保留支撑树优化与接触/视锥代理检查。左右区域保持左右方向，不是将所有物体随机放到背后。
- 固定观测宿主上的物体检查左、右、后、上、下五个视锥面，结合实际高度和尺寸判断可行性；不再由当前 X 符号武断决定唯一方向。线性不可行与优化收敛分别记录，不能把零位移当作求解成功。
- 没有测量过的房间边界不添加硬限制；地板/壁挂/层板等假设仍保留不确定性。AABB 通过不代表真实物理已验证。

若数值布局失败，保留并可视化完整 `G_full_candidates.json`，清楚标记 blocked；不把它叫成成功的 G_full。这样可以直接看见失败的是哪个物体，而不是只看到物体数量减少或文字错误。解析阶段没有足够数值时不伪造框。

至少一份数值候选导出时，作业正常返回，并保留 `partial_layouts_needs_review` 与所有失败记录；全部未导出或阶段异常时仍返回失败。退出码不承担语义/因果质量认证。

## 执行

不再换脚本入口。默认新建目录，保留原模型/API 名称和解码参数。两组各自保留原有的观测设计，两端使用同一新模型结构与几何接口。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/assembly_compare.sh test

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
  read -rsp 'SiliconFlow API Key: ' SILICONFLOW_API_KEY
fi
export SILICONFLOW_API_KEY

bash qwen_edit_pano/scripts/assembly_compare.sh retry2 GPU3 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_192949_4040842 \
  /data-nfs/gpu1-2/u13529658780/DIT360/qwen_edit_pano/outputs/assembly_rgb_compare_20261024_193808_4067827
```

GPU3 换成可用节点。这里复用已完成的 2D/WildDet3D 观测，只重做两端补全与布局。若连前端也从 RGB 全部重做，用 `bash qwen_edit_pano/scripts/assembly_compare.sh run2 GPU3` 替换最后一条。

打开新目录中打印的 `two_experiments.html`：直接对比 API/Qwen 的 3D 物体框、XZ 俯视布局、360° 状态方向图。物体清单和机制记录折叠显示。此阶段未接 Adapter，没有生成 RGB ERP。

下一批结果优先看：新增对象是否重述了观测、周围功能/储物/建筑要素是否合理、支持关系是否有具体宿主、不同假设是否确实不同、位置是否真的在输入视野之外。不要只比较对象总数。若相同接口下 Qwen 仍持续重述观测而 API 能给出合理增量，应把冻结 8B 的能力或提示适配作为待验证解释，停止继续堆因果节点。
