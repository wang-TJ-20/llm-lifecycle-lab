# 项目实施路线

项目目标不是堆叠算法，而是让中文学习者通过可读代码、受控实验和真实产物，
完整走通一个双语小模型从数据到可交互模型的生命周期。

每个阶段采用同一交付标准：

> 原理教程 + CPU 微型实验 + Smoke + GPU Reference + 前后对照报告 + 可下载产物

状态含义：

- **已实现**：功能代码和 CPU 自动验证完成。
- **待 GPU**：功能可以运行，但没有正式训练结果，不能声称能力改善。
- **待平台**：本地发布物已完成，等待外部平台状态。
- **未开始**：尚无可以验收的实现。

## 1. Pretrain 最终交付

| 内容 | 状态 | 证据 |
| --- | --- | --- |
| Native-60M-Base 权重与 Tokenizer 发布包 | 已实现 | `build/modelscope/native-60m-base-v1`，本地不提交 Git |
| Model Card、数据、配置、hash、能力边界 | 已实现 | [发布指南](./MODEL_RELEASE_GUIDE.md) |
| 下载、续写与基础评测命令 | 已实现 | `run_published_model.py`；公开下载待组织审核 |
| 统一 `report.md/json` | 已实现 | 发布报告及 [Base 能力成绩单](./experiments/native-60m-capabilities-v1.md) |
| 公开 ModelScope 仓库 | 待平台 | 组织 `llmlifecyclelab` 审核中 |
| clean provenance 新版权重 | 已实现 | `runs/native-60m-v11-001`；新规范 11 pass / 0 fail |

当前 checkpoint 是可复现的历史基线，但不是 clean rerun。不得修改旧 manifest
伪装成 clean。

clean 重跑已完成：新增
[`native-60m-baseline-v1.1`](../../configs/reference/native-60m-baseline-v1.1.yaml)
规范冻结干净源码快照，**v1 规范与 dirty-Git 例外原样保留**。
`runs/native-60m-v11-001` 在该规范下验收 **11 pass / 0 fail**，
包括此前唯一失败的 `runtime-provenance`。

## 2. 统一能力评测

| 内容 | 状态 |
| --- | --- |
| Base：中英文 BPB、固定续写、重复率 | 已实现 |
| 训练文本重合与后缀记忆探针 | 已实现 |
| SFT：指令、格式、问答、多轮规则评分 | 已实现 |
| Alignment：chosen/rejected 排序、可验证奖励、能力保持 | 已实现 |
| 随机、规则及阶段前基线 | 已实现 |
| 同协议纵向成绩单 | 已实现 |
| 大样本探针套件 lifecycle-v3（140 案例，每类 30+） | 已实现 | `build_evaluation_suite.py`；构建时泄漏检查为 0 |

详细口径见[统一能力评测](./CAPABILITY_EVALUATION_GUIDE.md)。
这些是小型诊断探针，不是通用排行榜；不命中记忆探针也不证明没有泄漏。

## 3. SFT 闭环

| 内容 | 状态 |
| --- | --- |
| `messages` 数据、模板分组隔离与评测集门禁 | 已实现 |
| assistant-only masking、有效监督 token 计权 | 已实现 |
| Base 权重初始化与 SFT 完整恢复 | 已实现 |
| CPU 微型 Base → SFT → 同协议评测 | 已实现 |
| CLI 续写/多轮对话 | 已实现 |
| 旧合成双语 SFT 数据与数据卡 | 历史复现 | `build_posttraining_data.py` + `data/posttraining_data_card.json` |
| 旧合成 Native-60M-Instruct 训练与报告 | 历史结果 | `runs/native-sft-60m-001`；见 [GPU 后训练报告](./experiments/native-60m-lifecycle-gpu-v1.md) |
| OASST1 + MSVAMP 公开 SFT 配方、切分与 CPU 加载 | 已实现 | [`public-60m-v1`](./PUBLIC_POSTTRAINING_GUIDE.md) |
| 公开数据 Native-60M-Instruct CUDA 训练 | 待 GPU | `native-sft-public-60m.yaml` |
| Instruct 权重公开发布 | 待平台 | 训练产物已就绪，未建发布包 |
| 教材第 08–10 章（SFT、DPO、GRPO） | 已实现 |

操作见 [Native SFT 最小闭环](./NATIVE_SFT_GUIDE.md)。
微型三步实验只证明机制和恢复正确，不证明 60M 模型已经会遵循指令。

旧合成数据 60M SFT 已在单张 RTX 4090 上完成 3 个 epoch（626 步，94.6 s），
同协议指令成功率 0.000 → 0.500，语料 BPB 上升 0.072。
这只是**在未见模板探针上的历史结果**，不等于公开数据路线已经完成，
也不等于获得可用的通用 Instruct 模型。

## 4. Native 与 Transfer 双路线

| 内容 | 状态 |
| --- | --- |
| Native → 标准 HF Llama/Qwen3 格式 | 已实现 |
| Tokenizer、chat template、logits、KV Cache、greedy 一致性 | 已实现 |
| 固定 revision 的 Qwen3-0.6B-Base 本地快照 | 已实现 |
| Qwen 与 Native 共用 SFT 数据、训练引擎及能力探针 | 已实现 |
| PEFT LoRA、adapter-only checkpoint 与精确恢复 | 已实现 |
| 随机微型 Qwen3 CPU 机制实验 | 已实现 |
| 下载真实 Qwen 权重并运行 LoRA Smoke | 已实现 | revision `da87bfb6…`；`runs/qwen3-lora-60m-001` |
| QLoRA | 已实现 | `training_method: qlora`；`runs/qwen3-qlora-60m-001` |
| Native Full SFT 与 Qwen LoRA 正式对照 | 已实现 | [报告第 5 节](./experiments/native-60m-lifecycle-gpu-v1.md#5-qwen-迁移与-qlora) |

详细命令见 [HF 导出与 Qwen 迁移](./TRANSFER_GUIDE.md)。
两条路线共享题目，但报告记录 `model_route`、Tokenizer 和后端；
Tokenizer 不同的结果分开统计，不能伪装为完全同口径的 delta。

## 5. 偏好学习、推理和部署

建议顺序固定为 DPO → GRPO/RLVR → 量化/推理 → 服务。

| 内容 | 状态 |
| --- | --- |
| DPO 数据、response-only log-prob 与 pair loss | 已实现 |
| 冻结 SFT reference 分数、恢复门禁、按偏好对计权 | 已实现 |
| Native DPO CPU 微型训练与精确恢复 | 已实现 |
| 旧合成 DPO 数据、训练和能力保持报告 | 历史结果 | `runs/native-dpo-60m-001`；QA 回退已记录 |
| 旧 on-policy rejected 与 NLL 正则（RPO） | 历史结果 | `runs/native-dpo-onpolicy-001`；QA 0.000 → 0.250 |
| HelpSteer3 公开 DPO 配方、切分与 CPU 加载 | 已实现 | `native-dpo-public-60m.yaml` |
| 公开数据 DPO CUDA 训练 | 待 GPU | 依赖公开 SFT step 112 |
| Qwen LoRA DPO | 未开始 |
| 可验证奖励 GRPO/RLVR 数据、目标与 CPU 精确恢复 | 已实现 |
| 旧合成 GRPO/RLVR 数据、GPU 训练和能力保持报告 | 历史结果 | `runs/native-grpo-60m-001`；dev 奖励零结果 |
| 旧 GRPO 可解性预筛 | 历史结果 | `runs/native-grpo-solvable-001`；零方差 0.59~0.82 → 0.06~0.44 |
| MSVAMP 公开 GRPO 配方、SFT 零组交集与 CPU 加载 | 已实现 | `native-grpo-public-60m.yaml` |
| 公开数据 GRPO CUDA 训练 | 待 GPU | 依赖公开 DPO step 21 |
| HF FP32/dynamic INT8 状态、吞吐、RSS 与稳定性基准 | 已实现 |
| Native/HF 非流式 OpenAI-compatible API | 已实现 |
| 同源本地 Chat/Completion 界面 | 已实现 |

DPO 操作与公式见 [Native DPO 最小闭环](./NATIVE_DPO_GUIDE.md)，
GRPO 见 [Native GRPO / RLVR 最小闭环](./NATIVE_GRPO_GUIDE.md)。
GRPO 只接受可程序验证的 exact、integer 或 JSON 奖励，不引入 LLM Judge。

## 受控实验

结构消融（深窄/浅宽、QK-Norm）的四个 pipeline 配置已就绪，但**尚未在 GPU 上
成组运行**，因此仍不输出结论。其余四项需要先补数据或实现：

- 中英文数据比例：需要新的数据配方与 Packing，当前双语音料比例固定。
- Tokenizer 词表大小：需要重训 Tokenizer 并重新 Packing。
- Full SFT 与 LoRA：Qwen 侧已有 LoRA / QLoRA，缺少同底座的 full 对照。
- SFT 数据质量与数量的分离对照：需要按同一模板集构造多档规模的数据。

每个实验必须固定非实验变量，先声明资格门槛、随机种子、训练 token 预算和
评测协议。未通过数据、恢复及 provenance 门禁的 run 不进入横向比较。

以下实验需要正式 GPU 训练，因此目前不输出结论：

1. 固定参数量的深窄与浅宽。
2. QK-Norm 开关。
3. 中英文数据比例。
4. Tokenizer 词表大小。
5. Full SFT 与 LoRA。
6. SFT 数据质量与数量的分离对照。

每个实验必须固定非实验变量，先声明资格门槛、随机种子、训练 token 预算和
评测协议。未通过数据、恢复及 provenance 门禁的 run 不进入横向比较。

## 下一步

旧合成 SFT、DPO、GRPO 以及 Qwen LoRA 与 QLoRA 的 GPU 训练已完成，
成绩单见 [GPU 后训练与迁移报告](./experiments/native-60m-lifecycle-gpu-v1.md)。
推荐的新路线已经完成公开数据准备、配置和 CPU 验证，CUDA 训练尚未执行；
执行顺序与 fail-fast 门槛见
[公开数据 SFT、DPO 与 GRPO 路线](./PUBLIC_POSTTRAINING_GUIDE.md)。

针对其中两个负面结果的第二轮修复已完成，记录在
[负面结果修复](./experiments/native-60m-negative-result-fixes-v1.md)：

1. **GRPO**：诊断更正为"题太难"——87.4% 的 prompt 模型 4 次采样全错。
   可解性预筛后零方差比例 0.59~0.82 → **0.06~0.44**，训练奖励 0.25~0.37 → 0.48~0.78。
2. **DPO**：改用 on-policy rejected 消除风格混淆，并加入 NLL 正则（RPO）。
   `qa.success` 0.000 → **0.250**，能力崩塌机制已缓解。
3. **探针**：`lifecycle-v2` 把偏好探针换成近似错误对，
   Base 上 0.833（v1 为饱和的 1.000），不再无法区分。

**但两者都还没有转化为能力探针上的提升。** 原因是探针样本量太小：
`instruction` 与 `qa` 各只有 4 个 case，0.25 的差异等于 1 个 case 翻转，
落在噪声内；所有后训练阶段的 v2 偏好准确率都是 0.667，未见收益。

其余待办按优先级：

- ~~扩大探针样本量~~ 已完成：`lifecycle-v3`（140 案例，每类 30+）。
  重测后确认：SFT 仍是唯一大跃迁；原 DPO 使 format 崩到 0.000；
  on-policy + NLL 的 DPO 取得最高 preference.accuracy 0.6250。
- GRPO 扫描 `kl_beta`（第二轮 KL 0.198 偏高）。
- DPO 扫描 `nll_coefficient`（当前只试了 1.0，NLL 仍缓慢上升）。
- 结构消融（深窄/浅宽 × QK-Norm）成组 GPU 运行与横向比较。
- Qwen LoRA DPO；Qwen full 与 LoRA 同底座对照。
- Instruct 权重发布包与公开仓库（依赖平台审核）。

在探针样本量扩大前，不产生新的后训练能力结论。
