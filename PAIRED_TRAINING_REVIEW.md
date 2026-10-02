# Local RGB → ERP 配对训练审核（2026-09-30）

更新：已新增 paired_train.py / paired_inference.py 配对入口及条件缓存、朝向校准工具，CPU接口测试通过；尚未执行真实模型 GPU 验证。执行顺序见 RUN_PAIRED.md。本文件保留此前方案审核，增强方案以新运行文档为准。现有 train.py 仍是 text-only 基线。

## 1. 数据事实与下载

原 train.jsonl：10359 个唯一 (scene_id, source_view_id)，90 栋建筑。当前 1605 个有完整六面 skybox；77 栋建筑缺少 skybox ZIP。此统计读取 ZIP 目录，没有读取全部 JPEG 做 CRC 或几何验证。详见 audit/skybox_coverage_before_download.json。

执行：
```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
bash qwen_edit_pano/scripts/download_skybox.sh
```

默认输出仍为 benchmark_assets/Matterport3D_raw；可用 MP3D_RAW_ROOT 覆盖。脚本调用原 download_mp_py3.py，只指定 matterport_skybox_images，不下载整个 1.3TB release，不更改原下载器。原下载器保留交互式条款确认，跳过已有 ZIP；下载器现支持 .part + 元数据、超时重试和服务器支持时的字节续传；忽略 Range 时安全重下该 ZIP。旧版无元数据的随机 tmp 文件不会自动采用。下载后全量 CRC 检查可能耗时。损坏的已有 ZIP 不会自动覆盖：查看报告后将确认损坏的文件移开，再重跑。

手动等价下载命令：
```bash
python download_mp_py3.py -o benchmark_assets/Matterport3D_raw --id ALL --type matterport_skybox_images
python qwen_edit_pano/audit_skybox_coverage.py --verify-crc --require-full
```

只有 matched_all_six_faces == total_unique_erp == 10359 且无无效 ZIP，才可宣称文件覆盖完整。还必须验证 skybox 与 polished ERP 朝向、镜像、视野及可见内容对应。脚本不会把文件匹配冒充几何验收，也不生成可直接开训的配对 manifest。

## 2. 训练任务

输入是一个局部图（首轮优先 skybox 水平面）、通用全景扩展指令或用户可用文本；GT 始终是现有 polished ERP。只对 ERP 目标计算生成损失。局部图片用作 conditioning，不作为额外 perspective 监督样本，因此仍为 panorama-only，不是 DiT360 mixed training。

原始传感器 RGB 与 skybox face 不是同一种成像数据。后者适合初始配对，前者存在内参、畸变、相机中心偏移问题，不可直接当成精准同中心裁剪。完成 skybox 阶段不能直接宣称适用于任意相机 FOV、俯仰与视差；须随后单独评估真实局部 RGB。

GT ERP 只进入目标 VAE 编码和监督分支；局部可见图进入原生 Edit 的视觉语义编码和 reference VAE 编码。不能把完整 ERP、完整全景描述或 oracle G_full 送进声称仅局部输入的评测。GT 的可见局部作为输入是合法配对监督。

## 3. 必要接口适配

以已安装 Diffusers 0.37.0 QwenImageEditPlusPipeline 为准：保留其局部图预处理、图文模板和图像相关 prompt embeddings；reference VAE 使用原生确定性编码，target VAE 保留旧训练采样方式和归一化。仅 target 加噪，将 target tokens 与 reference tokens 拼接，正确传 img_shapes 和 zero_cond_t 分支，最后只截取 target velocity 计算 loss。参考 token 不加目标噪声、不接受 ERP loss、不被 scheduler 更新。

当前 circular.py 明确禁止 reference tokens。必须改为仅 target 网格 circular padding，复制目标边缘 token 的原 RoPE，reference 段保持自身位置编码；同步更新 target/reference 边界、zero_cond_t 调制索引、prediction crop 和 inference scheduler 状态。不能把整个 token 序列视为一张 ERP。

原文本缓存不可复用。新缓存至少绑定模型与 processor 版本、局部图片内容、方向/翻转变换、prompt 与模板；不能只按全景 ID 或 caption 建缓存。CFG 的负分支沿用原生图像条件行为，并检查训练/推理一致性。外部 Qwen3-VL 与 Edit 内部 Qwen2.5-VL 编码器是两个不同模块。

## 4. Loss：第一阶段保持现有定义

z_t=(1-sigma)z_0+sigma*epsilon；v_target=epsilon-z_0。
保留现有 qwen_pano/losses.py 的 flow、cube、yaw 计算与 weighting：
L=L_flow+0.5 L_cube+0.5 L_yaw。

epoch 从零计：0、1 仍计算并记录几何项但不加总；epoch>=2 加入，保持旧 RNG 消耗逻辑。cube_target=C(epsilon)-C(z_0)；yaw_target=R(epsilon)-R(z_0)，角度60/180/300。不要自行将上游算术改写成不同计算顺序。

这些是 latent velocity 的几何变换监督，不是 RGB cubemap 重建，也不是重新跑一次旋转输入的等变约束。保留 cube/yaw 可继承 panorama 训练偏置，不能保证局部保真、真实三维一致性或因果约束。

第一版不加新损失，lambda_seam 保持0。若配对/接口无误而局部仍漂移，再单独消融“预测干净 ERP 解码后重投影到局部视角”的 masked reconstruction/perceptual loss；要求已知映射、处理 polished 差异，不应对任意未对齐原始 RGB 做逐像素 loss。新权重需实验确定，不在基线中擅自引入。

## 5. 增强与文本：不能盲目照搬

旧数据增强：0.5 水平翻转，ERP 随机 roll randint(W//3,W)。若条件固定 face2 且指令要求该视野在 ERP 中央，只 roll GT 会制造错误监督。

建议已有局部图优先方案：先 face2 固定朝向验证；正式训练从已校验方向的水平 skybox faces 中选一个，将 GT ERP 旋转到该局部朝向在中央，再同步水平翻转。每个 ERP 每 epoch 仍抽一个局部，不把四面展开为四倍 epoch 长度。必须先验证文件 face 到 ERP 的方向及镜像映射，不能凭编号臆测。此方案将旧的连续随机 yaw 改为离散朝向增强，应明确记录，不能声称增强完全相同。

如果必须保留原连续 yaw 分布，需从六面 skybox 按该随机朝向生成对应局部视图（在线投影即可，不重建训练集），同时应用一致翻转与 ERP roll；不能同时承诺任意连续 yaw、固定原局部图片、局部总在中央。几何 yaw loss 的60/180/300与输入增强不是同一个环节，可原样保留。

第一版用固定扩展指令，避免依赖包含镜头外信息的完整 caption；未来可选文本另设有/无文本协议，不能只在完整 caption 上训练后宣称无文本任务已验证。采用固定指令是相对旧文本条件的必要任务改变。

## 6. 保留参数与边界

目标1024×2048，25epoch，batch1，accum4，LR5e-5，rank/alpha64/64，dropout0.05，原 LoRA target modules；AdamW betas0.9/0.999、wd1e-5、eps1e-6；3237步 warmup 后恒定 LR；原 flow timestep sampling、seed0、padding1、cube/yaw0.5；检查点 optimizer/scheduler/RNG/完成标记、日志逻辑保持。VAE/base transformer 冻结，LoRA 可训练。完整10359行、world1且长度不扩张时才对应旧64750次更新；子集不能声称更新预算一致。

旧正式脚本推理28 steps、CFG4、negative空格、per-ID seed0；此前原生测试40steps，应分别标记，正式对照需统一。原生参考图预处理和额外 token 带来的显存/运行时间不同，不能由推理成功推断训练一定适配现有 GPU。

配对覆盖完整不等于独立测试。原正式全量训练与已有 benchmark 存在重叠；不能将重叠图像作为未见样本泛化证据。保持全量协议时明确称训练分布重建评估；若需要建筑级独立泛化，另立固定 split 并如实报告数量，不能同时声称使用全部10359训练且在其中做独立测试。

## 7. World 分支阶段与验收

A：先完成 native RGB conditioning + panorama LoRA，确认比原生 Edit 更好。
B：冻结 A 的生成基线，训练 World Encoder/Adapter；oracle G_full 只作为明确标注的上限对照。
C：训练/评估 G_obs→G_full Completion，逐步使用预测结构训练 renderer，检验预测噪声下效果。
D：接入只看 Local RGB 的外部 Qwen3-VL，完成真实端到端评估。

结构化 JSON 转文本是免训练基线，不提供可靠几何约束保证；新增结构 tensor 注入一般需要训练。World Adapter 与 panorama LoRA 分别解决结构接口和全景生成能力，不应混为同一个已完成模块。G_full 是目标 ERP 可见的世界状态假设，不应包含全屋所有不可见房间；单图尺度/位置要带不确定性。因果模块不在当前实现范围。

开训门槛：完整配对与可视化朝向验证；一个 batch loss finite，LoRA 有梯度，base/VAE 冻结；至少覆盖非零 LR 更新以确认参数变化；保存、重载、断点 RNG 与 scheduler；相同图文/参考图接口推理2:1 ERP；检查中心保真、左右接缝、极区和多方向透视投影，不能仅看输出尺寸。还要做换图/遮蔽图条件对照证明模型使用了局部图。World 阶段再做真实/打乱/常量结构消融。

这些是验收要求，不是本轮已通过的训练结果。

## 8. 核对依据

- 本地 download_mp_py3.py、原 train.jsonl、qwen_pano/data.py、losses.py。
- 正式 checkpoint-epoch025 的 training_config.json、pano_config.json、COMPLETE.json。
- https://github.com/niessner/Matterport/blob/master/data_organization.md
- https://github.com/Insta360-Research-Team/DiT360
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
