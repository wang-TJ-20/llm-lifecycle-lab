# Native-60M Base 能力成绩单 v1

这是统一能力评测层的第一份 Base 测量，使用现有
`native-60m-baseline-v1/step-00005649`，没有重新训练或修改权重。
它与[历史 CUDA Reference](./native-60m-baseline-v1.md)并列保留，
不替代旧指标，也不修改已构建的 Base-v1 发布包。

## 运行条件

- 输入：本地 `native-60m-base-v1` 发布包，先校验全部发布文件 hash。
- 权重 SHA-256：`78cf72522ad87e71aca4977ce5708aa3ac3ffaa3db32a04514511593f8b49664`。
- 评测：`lifecycle-v1`，28 个双语手写案例，加 8 个语料记忆探针。
- 设备：CPU、float32、单线程；greedy 生成，固定 32 个新 token。
- 指令提示：`plain-v1`；文本续写固定使用 `BOS + 原文`。
- 语料：Tokenizer 绑定的 `bilingual-60m-v1`，完整扫描 train/dev/test。

复现命令需使用新的输出目录：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --output runs/evaluations/base-capabilities-rerun \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json
```

完整结果：

- [机器可读报告与逐题输出](./results/native-60m-capabilities-v1/report.json)
- [完整指标表与固定续写](./results/native-60m-capabilities-v1/report.md)
- [指标口径及基线说明](../CAPABILITY_EVALUATION_GUIDE.md)

## 主要观察

| 项目 | 英文 | 中文 | 解释 |
| --- | ---: | ---: | --- |
| 手写文本 BPB | 1.5294 | 2.5892 | 各 2 段短文本，中文更差 |
| 续写重复三元组比例 | 0.0000 | 0.4833 | 各 4 个提示，中文明显退化 |
| 指令规则成功 | 0/2 | 0/2 | 抽取与情感分类 |
| JSON 规则成功 | 0/1 | 0/1 | 严格对象、键和值类型 |
| 简单问答成功 | 0/2 | 0/2 | 小规模上下文问答 |
| 多轮逐轮成功 | 0/3 | 0/3 | 使用模型自身历史回答 |
| 偏好排序成功 | 1/2 | 2/2 | 仅 4 对，不能据此判断对齐质量 |

英文续写已经呈现故事式模式，中文重复与不准确输出较多。
在固定指令协议上，当前 Base 尚不能稳定输出规则要求的答案。
这是后续 SFT 应该改变的行为，但三步 Smoke 不足以证明它已改变。

## 重合与记忆

本次有 16 条提示满足最小长度要求，未发现它们作为规范化子串出现在指定训练语料中。
另外 12 条短提示不参与该检查，不能将它们算作“已排除泄漏”。
指定 dev/test 中的 45,343 条记录未发现规范化全文与 train 完全相同。

train 与 test 各抽取中英文各 2 条记忆探针，32 token 前缀之后的
12 token 精确后缀匹配均未命中。样本极少，且精确匹配标准严格，
**不能据此证明模型没有记忆或数据不存在泄漏**。

## 如何用于后续对照

后续 SFT checkpoint 使用相同测试集、Tokenizer、代码、设备、精度和生成设置，
并将本报告传给 `--baseline`。若改用 `native-chat-v1`，必须先用该模板
重新评测 Base；评测器会拒绝跨模板直接计算提升。

这里的 BPB 不包含 EOS，来自几段完整的手写文本；旧 Reference 使用冻结的
Packed 窗口，二者不是同一指标口径。不能把旧 dev loss 与本报告 loss 拼成训练曲线。

当前只建立了 Base 的能力起点。SFT 已有 CPU 微型机制验证，
正式 60M SFT、DPO、GRPO 的效果仍需后续独立训练与同协议成绩单。

[返回统一能力评测](../CAPABILITY_EVALUATION_GUIDE.md)
