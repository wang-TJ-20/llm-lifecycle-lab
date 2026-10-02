# SFT 初始化门禁

本文定义 Native-60M SFT 的 Stage 0：在选择 SFT 数据、能力评测和训练超参之前，
先把唯一允许使用的 Base 初始化输入冻结为 canonical Base-v1。

这不是 SFT 训练手册。Stage 1 已通过独立的
[SFT 数据门禁](./SFT_DATA_GATE.md)冻结数据契约；数据物化、Trainer、能力评测、
正式配置和完成态门禁均已实现。实际执行统一使用
[Native-60M SFT 训练手册](./SFT_TRAINING_GUIDE.md)。

## 1. 冻结边界

冻结规范位于：

```text
configs/gates/native-60m-sft-v1-base-init.yaml
```

它固定以下内容：

| 范围 | 冻结内容 |
| --- | --- |
| 权重 | `models/base-v1-final/model.pt`，SHA-256 `4cdbd642...f9398` |
| 模型 | `tiny-60m`、62,927,616 参数、词表 16,384、最大序列长度 512 |
| Tokenizer | `bilingual-60m-v1`、`native-chat-v1` 及文件/manifest 哈希 |
| Base 来源 | `native-60m-base-v1-s42`、step 45,191、规范化配置哈希 |
| 训练证据 | run、checkpoint、预算、metrics、独立 test 和模型 manifest |
| 许可 | 权重与 Tokenizer 使用 `CC-BY-SA-4.0` |
| SFT 初始化 | 只继承模型权重；不继承 optimizer、scheduler、RNG 或 Base step |
| 新 run 要求 | SFT 从 step 0 开始，启动前 Git 工作树必须 clean |

`package_artifacts` 和 `evidence_artifacts` 同时固定文件大小与 SHA-256。
Pipeline 另外同时检查：

1. YAML 文件的字节哈希；
2. 解析后的规范化 `RunConfig` 哈希；
3. 归档 `resolved_config.yaml` 的规范化哈希。

因此，字节变化和实际训练语义变化都不能静默进入 SFT。

## 2. Base 质量前提

门禁重新读取冻结证据，而不是只信任文档结论：

- source run 必须为 `completed`，最终 step 必须为 45,191；
- token coverage 必须在 `[1.0, 1.001]`；
- 最终 dev loss 不高于 step-0 baseline 的 50%；
- 最终 dev loss 不高于最佳 dev loss 的 1.02 倍；
- 英文和中文 dev loss 都必须优于 step-0 baseline；
- 独立 test 必须来自 step 45,191，指标有限且语言 token 加权一致；
- 权重必须能严格加载到当前 `NativeTransformer`，参数量必须精确匹配。

Base 源 run 的 `dirty: true` 是一次已记录的历史偏离，只对这个已冻结 Base 输入有效。
它不改变后续规则，也不允许新的 SFT run 在 dirty worktree 上启动。

## 3. 执行方式

开发期间检查冻结输入：

```bash
python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --inputs-only
```

准备启动 SFT run 时，先提交代码和全部已冻结配置，再执行：

```bash
python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --preflight
```

机器读取可增加 `--json`。退出码约定：

| 退出码 | 含义 |
| --- | --- |
| `0` | 全部门禁通过 |
| `1` | 输入、证据、质量、加载或 Git 门禁失败 |
| `2` | 规范格式错误、缺字段或文件无法解析 |

`--inputs-only` 不检查当前 Git 状态，适合实现和审查期间运行。
它通过不代表可以开始训练；正式启动前必须使用 `--preflight`。

## 4. Stage 0 不负责的范围

以下内容不属于 Base 初始化规范，不能通过修改 Stage 0 来引入：

- 对话格式化、loss mask、截断与 packing 规则；
- SFT 学习率、batch、训练预算、checkpoint 和恢复策略；
- 指令遵循、中英文能力、事实性与安全性评测集及通过阈值；
- SFT Pipeline 配置和完成态 run 验收规范。

公开来源、版本、许可、切分和数据质量规则由 Stage 1 数据门禁管理；训练与验收由
`configs/pipelines/native-60m-sft-v1.yaml` 和
`configs/gates/native-60m-sft-v1-run.yaml` 管理。不得修改本门禁来掩盖 Base
输入漂移；如果 canonical Base 本身需要替换，应创建新的 gate ID 和冻结文件。
