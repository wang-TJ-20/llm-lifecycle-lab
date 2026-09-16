# 60M CUDA 后训练与 Qwen 迁移 · native-60m-lifecycle-gpu-v1

[项目首页](../README.md) | [项目实施路线](./test.md)

本文记录第一次在单张 RTX 4090 上用固定 60M 配方走完
Pretrain → SFT → DPO / GRPO 后训练，以及 Qwen3-0.6B-Base 真实权重的
LoRA 与 QLoRA 迁移。所有后训练阶段使用同一份 `lifecycle-v1` 探针和
`native-chat-v1` 提示协议，因此报告之间可以直接相减。

> 设备为共享 GPU，同一张卡上还有其它进程；吞吐和显存只描述本次环境，
> 不代表硬件极限。

## 1. 结果与结论

| 阶段 | run_id | steps | 监督/rollout token | 耗时 | CUDA 峰值显存 |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT | `native-sft-60m-001` | 626 | 149,644 | 94.6 s | 1.872 GiB |
| DPO | `native-dpo-60m-001` | 111 | 61,809 | 48.2 s | 1.977 GiB |
| GRPO | `native-grpo-60m-001` | 136 | 127,610 | 1,384.2 s | 3.520 GiB |
| Qwen LoRA | `qwen3-lora-60m-001` | 626 | 177,307 | 377.6 s | 7.267 GiB |
| Qwen QLoRA | `qwen3-qlora-60m-001` | 626 | 177,307 | 525.7 s | 5.572 GiB |

SFT 与 DPO 的 dev loss 都稳定下降，说明 assistant-only 监督和冻结 reference
偏好目标在 CUDA BF16 下按预期工作。GRPO 的**未见模板 dev 奖励没有改善**，
这是本次最重要的负面结果。

## 2. Base 起点

SFT 的父权重是发布包 `build/modelscope/native-60m-base-v1`，即
`native-60m-baseline-v1`（step 5649）。该 run 保留已披露的 dirty-Git
provenance 例外，SFT 的 `initialization.json` 把
`parent_provenance_status` 原样记为 `accepted-with-provenance-waiver`，
没有把它改写成 clean。

同一次的 clean 重跑 `runs/native-60m-v11-001` 用新规范
[native-60m-baseline-v1.1](../../configs/reference/native-60m-baseline-v1.1.yaml)
验收为 **11 pass / 0 fail**，v1 规范本身未改动。

## 3. 数据

`scripts/build_posttraining_data.py` 确定性生成，不下载外部语料、不使用模型生成：

| 数据集 | 记录 | train / dev / test | 分组 |
| --- | ---: | --- | --- |
| SFT | 19,210 | 13,332 / 3,846 / 2,032 | 27 / 9 / 6 个模板组 |
| DPO | 11,266 | 7,075 / 2,516 / 1,675 | 26 / 9 / 6 |
| GRPO | 7,038 | 4,338 / 1,620 / 1,080 | 26 / 9 / 6 |

按 `template_id` 分组切分，dev/test 是**未见模板**，所以 dev loss 与 train loss
的差距主要反映模板泛化，而不是过拟合噪声。

生成过程中修掉三处会导致 Data Manifest 加载器拒绝数据的去重缺陷：
原始实现按采样参数去重，而不同参数可能渲染出相同文本；且加载器按**单条记录**
（即单语言）去重，因此只在中英文之一发生碰撞也算重复。修复后按每种数据集实际
落盘的字段、逐语言去重。

数据卡见 `data/posttraining_data_card.json`（许可 Apache-2.0）。

## 4. 能力成绩单（同一协议，41 项指标）

Base / SFT / DPO / GRPO 均使用 `native-chat-v1`，报告
`runs/evaluations/lifecycle-chat-v1`：

| 指标 | Base | SFT | DPO | GRPO |
| --- | ---: | ---: | ---: | ---: |
| instruction.success | 0.000 | **0.500** | 0.500 | 0.250 |
| qa.success | 0.000 | **0.500** | 0.000 | 0.250 |
| multiturn.success | 0.000 | **0.500** | 0.500 | 0.500 |
| verifiable.reward | 0.000 | **0.438** | 0.313 | 0.313 |
| preference.accuracy | 1.000 | 1.000 | 1.000 | 1.000 |
| corpus.bpb.all | 2.035 | 2.107 | 2.102 | 2.108 |
| continuation.repeated_trigram | 0.242 | 0.213 | 0.192 | 0.229 |

- **SFT 明显生效**：指令、问答、多轮和可验证奖励从 0 提升到 0.44~0.50，
  代价是语料 BPB 上升 0.072（能力保持的小幅退化）。
- **DPO 保住指令与多轮，但 QA 回退到 0**：偏好探针在 Base 上已经是 1.000，
  已饱和，因此无法证明 DPO 收益；QA 回退说明当前合成偏好对没有提供正向信号。
- **GRPO dev 奖励持平**：0.104 → 0.101，而近似 KL 从 0 升到 0.042。
  组内零方差比例长期在 0.59~0.82，说明大量 prompt 的 4 个回答奖励全同，
  优势恒为 0，学习信号不足。这是本次明确的零结果。

## 5. Qwen 迁移与 QLoRA

固定 revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` 的真实
`Qwen/Qwen3-0.6B-Base` 已下载并校验，随后在同一份 SFT 数据上训练：

| | LoRA | QLoRA (NF4 + 双量化) |
| --- | ---: | ---: |
| 可训练参数 | 1,146,880 | 1,146,880 |
| 最终 train loss | 1.8778 | 1.8742 |
| 最优 dev loss | 3.2512 | 3.5745 |
| CUDA 峰值显存 | 7.267 GiB | **5.572 GiB（-23%）** |
| 耗时 | 377.6 s | 525.7 s |

QLoRA 用 23% 的显存换到接近的 train loss，代价是约 39% 的额外耗时和
略差的 dev loss。QLoRA 需要 CUDA 与 `bitsandbytes`，配置中的
`quantization` 会被写入 run binding，与 bitsandbytes 版本一起做门禁；
4-bit 权重在加载时由 `device_map` 固定，不允许再用 `.to()` 搬移。

**Qwen 两项在指令探针上均为 0**，而语料 BPB 为 1.218 / 1.385，远好于 Native。
原因是 Qwen 的 Tokenizer 与 Native 不同，`native-chat-v1` 的控制串对 Qwen
不是它预训练见过的形式；且 LoRA 的 dev loss 从 3.25 升到 4.90，说明它没有泛化到
未见模板。按[统一能力评测](./CAPABILITY_EVALUATION_GUIDE.md)的约定，
Native 与 Qwen 的分数**并列查看，不相减**。

## 6. 证据

- SFT / DPO / GRPO 四阶段成绩单：`runs/evaluations/lifecycle-chat-v1`
- Base 基线：`runs/evaluations/base-chat-v1`、`base-plain-v1`
- Qwen LoRA 与 QLoRA 对照：`runs/evaluations/qwen3-lora-vs-qlora-v1`
- clean provenance v1.1 验收：`runs/native-60m-v11-001`

[项目首页](../README.md) | [项目实施路线](./test.md)
