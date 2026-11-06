# 2026-10-11 已训练 Oracle 的诊断

## 已查到的事实

旧 preview4 已生成12张输出。选取规则 candidates[:limit]，四条全来自8194nk5LbLH：两条室内朝庭院门窗、两条健身房，不能称作全是室外。未覆盖住宅房间类型；shuffled 使用列表中的相邻样本，可能同房间/相似世界，属于偏弱对照。旧结果仍保留，不能以换样本后的结果替代完整验证报告。

目视对照：第一条baseline已有重复座椅/大面积浅地面，与GT室内后方结构不同；健身房场景主要结构错误在baseline也存在；correct对局部家具/边界有变化，但没有稳定恢复GT布局的证据。correct与baseline的归一化像素MAE为0.0227–0.0395；与shuffled为0.0196–0.0371。这只是输出敏感性，不能作为质量提升指标。

CPU链路检查：

- 已训练配置中6个oracle源文件当前哈希均一致，原模型/训练/推理代码没有为此次诊断改写。
- 最终checkpoint哈希和权重有限性通过；保存的是训练后的World分支。epoch006是冻结Local RGB生成基线。
- 从现有val中按Local和GT预览选定6个明确室内样本，跨3栋建筑，每栋2个：楼梯走廊、客厅、spa房、室内吧台、卧室、客厅。未查看这些样本的新生成结果后再选择。
- 6条的paired身份与provenance一致；原source对象catalog哈希一致；类别一致；重新计算的局部center和完整AABB尺寸与G_full逐项相等。六方向RGB NCC最低0.819–0.966。两个样本有此前用户已允许的纯小物体阈值标记，无多视角不一致标记。
- 训练后Encoder改变中心坐标得到不同token；不能据此推断整个扩散模型已学会空间控制。
- 现有oracle_test 7项CPU测试通过，包含零门控、reference边界、checkpoint backward/activation offload、重载等接口验证。小模型测试不能替代20B实际GPU逐步对齐检查。
- 新诊断hook在小张量上保持输出逐元素相同，新增记录relative_delta_rms；不修改原分支计算。

机器可读结果：qwen_edit_pano/audit/oracle_chain_audit_2026-10-11.json。

## 未证明的部分及研究限制

没有执行任何本轮GPU计算、全模型训练/推理单步等价测试或固定噪声验证损失。因此目前不能证明“全部中间环节都正确”，也不能锁定单一失败原因。

当前G_full是经过筛选的对象状态，不是完整可渲染世界。Encoder特征仅category/center/AABB/valid masks；墙、地板、天花板、纹理材质和显式房间布局没有输入。Adapter是基于target hidden query的全局对象cross-attention，没有显式ERP方向与对象方向的空间attention bias。空间对应只能从重建学习；这是可解释的能力限制，不是已确认代码bug。

1000步accum4是4000次有放回样本抽取，约3.17个数据量，不能称作充分收敛。loss没有明显下降，且没有固定噪声的无world验证损失对照。不要仅提高lr、gate倍率或加长训练就宣称解决。

World同时进入正/负CFG分支，这是现有策略；对条件强度的影响尚未消融，不应擅自改为只进正分支并将结果与原实验混为一谈。

## 用户执行

首先固定训练好的step1000、模型、prompt、seed、步数，仅改善样本覆盖与错误GT对照：

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=08:00:00 \
  bash qwen_edit_pano/world_pipeline/run_indoor_diagnostic.sh 1000
```

输出oracle_indoor6_step001000_v1，每条local.png、gt_erp.png、baseline.png、correct.png、shuffled.png及JSON。shuffled采用固定的跨建筑置换，三组相同Local和seed，总18张生成图。correct/shuffled JSON增加各层所有调用的相对残差统计，原来大绝对RMS不再单独解释。此命令不训练任何参数。

如1000步结果仍差，可运行同一固定清单的500步对照，不需要重训：

```bash
srun -p debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --qos=normal --time=08:00:00 \
  bash qwen_edit_pano/world_pipeline/run_indoor_diagnostic.sh 500
```

每条命令写独立目录；已存在目录拒绝覆盖。本轮仅实现推理诊断，没有修改训练checkpoint、bundle、训练清单或LoRA。

## 后续决策

- correct优于baseline且比跨建筑shuffled合理：扩大到固定完整val，并做同预算constant对照，验证是否有真实状态信息收益。
- correct与shuffled几乎一样：检查条件使用强度及同噪声预测差，再考虑显式空间对应设计；不要将Encoder token变化当作生成器正确使用了几何。
- correct明显破坏局部/比baseline差：先比较500步和归一化残差，判断训练漂移/过强更新，再制定新版本训练方案。
- baseline也明显不佳：单独评估冻结Panorama基线，不把所有基础生成缺陷归因于Adapter。

候选后续优化是方向/距离的空间attention约束或保留现有字段的ERP空间表示，而不是立即叠加Observer/Completion或添加更多字段；任何架构修改需新实验验证，本轮未实现。
