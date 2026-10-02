# SFT 数据门禁

本文定义 Native-60M SFT 的 Stage 1：在 Stage 0 已冻结 canonical Base 初始化后，
冻结唯一允许进入 SFT 的公开数据契约，并对真实物化产物执行 fail-fast 校验。

这不是 SFT 训练完成声明。Recipe、gate、canonical source 和 prepared splits
已冻结并通过真实物化验证；Trainer、能力评测和正式配置也已实现。实际训练统一使用
[Native-60M SFT 训练手册](./SFT_TRAINING_GUIDE.md)。

## 1. 冻结文件

```text
configs/data/sft-public-balanced-v1.yaml
configs/gates/native-60m-sft-v1-data.yaml
```

Gate 同时固定 Stage 0 Base gate 和 Stage 1 recipe 的文件 SHA-256。任何来源、选样、
许可、切分或 Base 初始化变化都必须创建新版本，不能原地修改后继续训练。

Recipe 使用新业务名称 `sft-public-balanced-v1`，但保留
`selection_namespace: public-60m-v3`。该 namespace 是稳定哈希选样的一部分，
用于复现已验证的 14,799 条 canonical SFT source，不能随重命名改变。

## 2. 公开来源

| 来源 | 固定 revision | 用途与数量规则 | 许可 |
| --- | --- | --- | --- |
| OASST1 | `fdf72ae...e3399b` | 中英文各 350；拒绝 synthetic；只取 rank 0 assistant | Apache-2.0 |
| Dolly 15k | `bdd27f4...ff1a` | general 2,050、classification 450、short QA 250 | CC-BY-SA-3.0 |
| HC3-Chinese | `09a687b...31c2` | 2,750 条 human answer | CC-BY-SA-4.0 |
| SQuAD | `7b6d24c...b1f` | 3,000 条短答案，加 250 integer / 250 string JSON view | CC-BY-SA-4.0 |
| CMRC2018 | `137f2c4...86aa` | 3,000 条短答案，加 500 string JSON view | CC-BY-SA-4.0 |
| MSVAMP | `301e2b3...0add` | 前 800 group 生成中英文 SFT；后 200 group 保留 | Apache-2.0 |

每个来源还固定上游文件路径与文件 SHA-256，只允许
`provider: huggingface` 的公开数据。组合许可固定为：

```text
Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0
```

`HelpSteer3` 明确保留给 DPO，不能进入 SFT。SFT 不允许合成答案。

## 3. 数据身份与切分

Canonical source 的冻结身份为：

| 字段 | 值 |
| --- | --- |
| 记录类型 | `sft` |
| 记录数 | 14,799 |
| source SHA-256 | `9ba984310ef0996237908b5843c67d869796fef37e312d2b811f23a647fa772b` |
| dataset ID | `sft-public-balanced-v1` |
| split seed | 42 |
| group key | `source_id` |
| 比例 | train/dev/test = 0.8/0.1/0.1 |

门禁不会只读取 manifest。它从 canonical source 重新计算确定性 group split，并要求
prepared JSONL 的记录及顺序完全一致，同时检查 manifest 中的文件哈希、记录数和
group 数。这样同一上游样本产生的普通/structured view 不会跨 split。

## 4. 对话与训练数据质量

每条 prepared 记录必须满足：

- 使用 `RecordKind.SFT`，包含 `id`、`source_id`、`language` 和 `messages`；
- `source_id` 前缀、`metadata.source` 与 `metadata.transform` 必须匹配 recipe；
- 语言只能是 `en` 或 `zh`，并且 train/dev/test 每个 split 都必须覆盖两种语言；
- 对话只能是可选的首条 system，加严格交替的 user/assistant，并以 assistant 结束；
- 内容不得包含 Native control token；
- 使用冻结的 `bilingual-60m-v1` / `native-chat-v1` Tokenizer 后不得超过 512 token；
- 禁止隐式截断，只统计 assistant body 和 `<|im_end|>` 的监督 token；
- 按 `NFKC-whitespace-collapse-v1` 归一化后，不允许重复 conversation；
- 禁止 synthetic 记录。

Train split 还必须精确匹配：

| 指标 | 冻结值 |
| --- | ---: |
| examples | 11,882 |
| supervised tokens | 475,264 |
| English / Chinese examples | 5,938 / 5,944 |
| general | 4,114 |
| short QA | 5,292 |
| classification | 363 |
| numeric | 1,298 |
| structured | 815 |

## 5. 执行方式

只检查 recipe、来源策略和两个 gate 的冻结哈希，不要求本地存在数据：

```bash
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --contract-only
```

真实数据物化后，检查 Stage 0、source、manifest、splits、Tokenizer 和逐条质量：

```bash
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --inputs-only
```

准备启动正式 SFT run 前使用：

```bash
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --preflight
```

`--preflight` 在全部输入检查通过后额外要求 Git 工作树 clean。机器读取可增加
`--json`。退出码 `0` 表示所选 scope 通过，`1` 表示门禁失败，`2` 表示规范或参数
错误。

在已物化数据的当前工作目录中，`--inputs-only` 预期 `14/14` checks 通过。
`data/` 被 Git 忽略，因此新 clone 只有 `--contract-only` 能直接运行；
新机器必须先按 [SFT 训练手册](./SFT_TRAINING_GUIDE.md#3-物化冻结数据)物化或同步
数据，再要求 `--inputs-only` 通过。

## 6. 不属于本门禁

Stage 1 负责在加载时检查 lifecycle-v4 与 SFT 数据的 group、prompt 和 passage
泄漏，但不定义模型质量通过线。正式训练配置和完成态验收分别由
`configs/pipelines/native-60m-sft-v1.yaml` 与
`configs/gates/native-60m-sft-v1-run.yaml` 冻结。Stage 0 与 Stage 1 通过仍不等于
正式 run 已验收；必须完成全量 dev 并通过 `verify_sft_run.py --dev`。
