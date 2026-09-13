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
| clean provenance 新版权重 | 待 GPU | v1 保留并披露 dirty-Git 一次性例外 |

当前 checkpoint 是可复现的历史基线，但不是 clean rerun。不得修改旧 manifest
伪装成 clean；未来有 GPU 时可新建 v1.1。

## 2. 统一能力评测

| 内容 | 状态 |
| --- | --- |
| Base：中英文 BPB、固定续写、重复率 | 已实现 |
| 训练文本重合与后缀记忆探针 | 已实现 |
| SFT：指令、格式、问答、多轮规则评分 | 已实现 |
| Alignment：chosen/rejected 排序、可验证奖励、能力保持 | 已实现 |
| 随机、规则及阶段前基线 | 已实现 |
| 同协议纵向成绩单 | 已实现 |

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
| 正式双语 SFT 数据治理与数据卡 | 未开始 |
| Native-60M-Instruct 训练、报告与发布 | 待 GPU |
| 教材第 08–10 章（SFT、DPO、GRPO） | 已实现 |

操作见 [Native SFT 最小闭环](./NATIVE_SFT_GUIDE.md)。
微型三步实验只证明机制和恢复正确，不证明 60M 模型已经会遵循指令。

## 4. Native 与 Transfer 双路线

| 内容 | 状态 |
| --- | --- |
| Native → 标准 HF Llama/Qwen3 格式 | 已实现 |
| Tokenizer、chat template、logits、KV Cache、greedy 一致性 | 已实现 |
| 固定 revision 的 Qwen3-0.6B-Base 本地快照 | 已实现 |
| Qwen 与 Native 共用 SFT 数据、训练引擎及能力探针 | 已实现 |
| PEFT LoRA、adapter-only checkpoint 与精确恢复 | 已实现 |
| 随机微型 Qwen3 CPU 机制实验 | 已实现 |
| 下载真实 Qwen 权重并运行 LoRA Smoke | 未执行 |
| QLoRA | 待 GPU |
| Native Full SFT 与 Qwen LoRA 正式对照 | 待 GPU |

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
| 正式 DPO 数据、训练和能力保持报告 | 待 GPU |
| Qwen LoRA DPO | 未开始 |
| 可验证奖励 GRPO/RLVR 数据、目标与 CPU 精确恢复 | 已实现 |
| 正式 GRPO/RLVR 数据、GPU 训练和能力保持报告 | 待 GPU |
| HF FP32/dynamic INT8 状态、吞吐、RSS 与稳定性基准 | 已实现 |
| Native/HF 非流式 OpenAI-compatible API | 已实现 |
| 同源本地 Chat/Completion 界面 | 已实现 |

DPO 操作与公式见 [Native DPO 最小闭环](./NATIVE_DPO_GUIDE.md)，
GRPO 见 [Native GRPO / RLVR 最小闭环](./NATIVE_GRPO_GUIDE.md)。
GRPO 只接受可程序验证的 exact、integer 或 JSON 奖励，不引入 LLM Judge。

## 受控实验

这些实验需要正式 GPU 训练，因此目前不输出结论：

1. 固定参数量的深窄与浅宽。
2. QK-Norm 开关。
3. 中英文数据比例。
4. Tokenizer 词表大小。
5. Full SFT 与 LoRA。
6. SFT 数据质量与数量的分离对照。

每个实验必须固定非实验变量，先声明资格门槛、随机种子、训练 token 预算和
评测协议。未通过数据、恢复及 provenance 门禁的 run 不进入横向比较。

## 下一步

当前路线中明确规划的 CPU 功能已实现。后续不等待 GPU 时，优先补正式
SFT/DPO/GRPO 数据治理、数据卡和更多 verifier 测试；这些工作完成前不产生
新的 60M 能力结论。

正式 SFT、LoRA、DPO、结构消融和权重发布在 GPU 或平台条件具备后补齐。
