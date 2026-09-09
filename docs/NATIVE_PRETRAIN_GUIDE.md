# Native 10M/60M Pretrain 训练指南

本文只聚焦 Native 路线的训练操作，说明如何使用 `llmlab` 完成以下流程：

```text
原始 JSONL
  -> 数据校验与确定性切分
  -> Byte-level BPE Tokenizer
  -> 不可变磁盘 Token Packing
  -> Doctor 真实 batch 检查
  -> Native Pretrain
  -> checkpoint 恢复
  -> dev/test 独立评测
```

当前 Native 路线只实现 Pretrain，不包含 SFT、DPO、GRPO、导出和服务。

模型结构、注意力实现、KV Cache 和配置缩放方式请看 [Native 10M/60M 模型结构说明](./NATIVE_MODEL_GUIDE.md)。

## 1. 适用范围与配方

| 配方 | 参数量 | 默认设备 | 序列长度 | 用途 |
| --- | ---: | --- | ---: | --- |
| `smoke-10m` | 9,915,200 | CPU/MPS/CUDA 自动选择 | 128 | 验证数据、模型、梯度、checkpoint 和评测链路 |
| `tiny-60m` | 62,927,616 | CUDA | 512 | 进行有意义的预训练实验 |

两个配方共用同一个 `NativeTransformer` 实现，只通过配置缩放。

10M Smoke 只用于验证工程链路，不代表模型质量。60M 配方的正式目标环境是 Linux 和单张 24GB NVIDIA GPU。

## 2. 环境准备

在仓库根目录执行：

```bash
uv sync --extra training --extra dev
uv run llmlab --version
uv run llmlab doctor
```

如果要从内置公开数据 recipe 开始，安装时再加入 `--extra public-data`。

最低要求：

- Python 3.11 或更高版本。
- `uv.lock` 未被绕过。
- 10M Smoke：CPU、MPS 或 CUDA。
- 60M Learn：Linux、CUDA，建议至少 24GB 显存。

基础 `doctor` 只检查运行环境。带配置的 `doctor` 还会检查模型、Tokenizer、数据和真实 forward/backward。

## 3. 准备 Pretrain 数据

数据准备已经拆成独立的 [Pretrain 数据准备 Runbook](./DATA_PREPARATION_RUNBOOK.md)，支持：

- 拉取 SimpleStories 英文与 Wikipedia 中文公开 recipe，并确定性组合。
- 单独使用英文或中文公开数据进行对照实验。
- 接入自己的 `id/text/source` JSONL。
- 统一执行 schema 校验、确定性分组切分和 Data Manifest 固化。

继续本文前，应准备好以下任一 Data Manifest，并在训练 Tokenizer 后生成对应的
Packed Manifest：

```text
data/prepared/bilingual-smoke-v1/data_manifest.json
data/prepared/bilingual-60m-v1/data_manifest.json
data/prepared/<custom-dataset-id>/data_manifest.json
```

默认 Native pipeline 使用前两条中英双语公开数据路径。单语对照或自有数据需要复制 pipeline 配置并修改 `data.manifest` 和 `model.tokenizer`。

## 4. 训练 Native Tokenizer

10M 与 60M 默认都使用 16,384 词表：

```bash
uv run llmlab tokenizer train \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --output data/tokenizers/bilingual-smoke-v1 \
  --tokenizer-id bilingual-smoke-v1 \
  --vocab-size 16384 \
  --min-frequency 1
```

60M 路线：

```bash
uv run llmlab tokenizer train \
  --manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --output data/tokenizers/bilingual-60m-v1 \
  --tokenizer-id bilingual-60m-v1 \
  --vocab-size 16384 \
  --min-frequency 2
```

Tokenizer 使用 NFKC normalization、Byte-level pre-tokenization 和 BPE，并冻结
`native-chat-v1` 协议：

```text
基础：
<|pad|> <|bos|> <|eos|> <|unk|>

消息边界：
<|im_start|> <|im_end|>

预留：
<|reserved_0|> ... <|reserved_7|>
```

模板使用 `<|im_start|>{role}\n{content}<|im_end|>\n` 表示一条消息。预留 token
当前不绑定 Tool 或 Thinking 语义，后续只有在新版本协议中才能启用。训练代码会拒绝
缺少该模板或 token 映射不一致的旧 Tokenizer，避免在 Pretrain 后修改词表导致
checkpoint 失效。

检查编码和解码：

```bash
uv run llmlab tokenizer inspect \
  data/tokenizers/bilingual-smoke-v1 \
  --text "语言模型 learns from text."
```

输出目录包含 `tokenizer.json` 和 `tokenizer_manifest.json`。Manifest 会记录：

- Tokenizer 内容 SHA-256。
- 训练所用 Data Manifest SHA-256。
- train split SHA-256。
- BPE 参数和 special token ID。

20K 双语 Smoke recipe 必须使用 `min_frequency=1` 才能得到完整的 16,384 词表；226K
双语 60M recipe 使用 `min_frequency=2`。其他语料可能得到不同的实际词表。训练阶段
要求 Tokenizer 实际词表与模型 `vocab_size` 完全一致，不会静默接受不匹配。

Tokenizer 输出目录禁止覆盖。需要重训时使用新的目录或先明确处理旧产物。

### 4.1 生成不可变 Packed 数据

Smoke：

```bash
uv run llmlab data pack \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-smoke-v1 \
  --output data/packed/bilingual-smoke-v1-seq128 \
  --sequence-length 128
```

60M：

```bash
uv run llmlab data pack \
  --manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output data/packed/bilingual-60m-v1-seq512 \
  --sequence-length 512
```

每个 split 会生成 token ID、语言 ID 和 UTF-8 byte weight 三个二进制数组。
`packed_manifest.json` 将其绑定到 Data Manifest SHA-256、Tokenizer SHA-256 和
`sequence_length`，并记录每个数组的大小与 SHA-256。训练通过 `numpy.memmap`
按样本读取，不把完整 token stream 常驻内存。输出目录禁止覆盖，生成过程中只写临时
目录，全部成功后才原子发布。Native Pretrain 和 Doctor 均要求该产物，不再保留内存
packing fallback。

当前固定数据的验收基线：

| 配方 | Train 监督 token | Train 样本 | 全部 packed 文件 |
| --- | ---: | ---: | ---: |
| `bilingual-smoke-v1-seq128` | 4,623,620 | 36,407 | 约 53 MiB |
| `bilingual-60m-v1-seq512` | 46,184,530 | 90,381 | 约 500 MiB |

## 5. 检查模型配置

10M：

```bash
uv run llmlab model inspect \
  --config configs/models/smoke-10m.yaml
```

预期参数量：

```text
model_id: smoke-10m
parameters: 9,915,200
layers=4 hidden=320 heads=5 kv_heads=1
```

60M：

```bash
uv run llmlab model inspect \
  --config configs/models/tiny-60m.yaml
```

预期参数量：

```text
model_id: tiny-60m
parameters: 62,927,616
layers=8 hidden=768 heads=12 kv_heads=4
```

比较 8K、12K 和 16K 词表对模型参数预算的影响：

```bash
uv run llmlab model inspect \
  --config configs/models/smoke-10m.yaml \
  --compare-vocab-size 8192 \
  --compare-vocab-size 12288 \
  --compare-vocab-size 16384
```

当前 16K 词表的 token embedding/LM head 参数占比在 10M 配方中为 52.88%，在
60M 配方中为 20.00%。16K 继续作为中英文共享基线，但 8K/12K 必须作为独立对照，
不能只凭模型总参数量判断词表大小。

模型配置中的 `vocab_size` 必须与前一步生成的 Tokenizer Manifest 一致。更详细的结构解释见 [Native 10M/60M 模型结构说明](./NATIVE_MODEL_GUIDE.md)。

## 6. Pipeline 配置

现有配置：

- 10M Smoke：`configs/pipelines/native-smoke.yaml`
- 60M Learn：`configs/pipelines/native-v1.yaml`

运行路径相对于执行 `llmlab` 时的当前工作目录解析。建议始终从仓库根目录运行。

关键字段：

| 字段 | 说明 |
| --- | --- |
| `model_route` | 10M 为 `native-smoke`，60M 为 `native-learn` |
| `model.config` | Native 模型结构配置 |
| `model.tokenizer` | Tokenizer 目录 |
| `data.manifest` | 已准备数据的 Manifest |
| `data.packed_manifest` | 与数据、Tokenizer、序列长度绑定的 Packed Manifest |
| `training.device` | `auto`、`cpu`、`mps` 或 `cuda` |
| `training.dtype` | `float32`、`bfloat16` 或 `float16` |
| `training.sequence_length` | 必须不超过模型最大长度 |
| `training.micro_batch_size` | 单次 forward 的样本数 |
| `training.gradient_accumulation_steps` | 每次 optimizer step 累积的 micro batch 数 |
| `training.max_steps` | 按 optimizer step 指定预算 |
| `training.max_train_tokens` | 按监督 token 指定预算 |
| `training.num_epochs` | 按完整数据轮数指定预算 |
| `training.learning_rate` | AdamW 初始学习率 |
| `training.warmup_steps` | warmup optimizer step 数 |
| `training.checkpoint_interval` | checkpoint 保存间隔 |
| `training.eval_interval` | dev 评测间隔 |
| `training.eval_batches` | 每次评测最多使用的 batch 数 |
| `training.log_interval` | 指标写入间隔 |

验证配置：

```bash
uv run llmlab config validate configs/pipelines/native-smoke.yaml
uv run llmlab config validate configs/pipelines/native-v1.yaml
```

三种预算字段必须且只能设置一个。10M 默认只跑 2 步，用来验证链路；60M 默认
`num_epochs: 1.0`。按当前 46,184,530 个 train 监督 token、batch size 1 和
16 步梯度累积计算，计划为 5,649 个 optimizer steps；最后一个完整累积 step
使估算覆盖率为 100.0033%。

## 7. 训练前 Doctor

10M：

```bash
uv run llmlab doctor \
  --config configs/pipelines/native-smoke.yaml
```

60M：

```bash
uv run llmlab doctor \
  --config configs/pipelines/native-v1.yaml
```

带配置的 Doctor 会检查：

- Python、依赖、磁盘和输出目录。
- 设备是否满足当前 profile。
- Pipeline 与 Model Config 是否一致。
- Data Manifest、split hash 和 dev split。
- Tokenizer hash、来源数据 hash 和词表大小。
- Packed Manifest 的数组 hash，以及 Data Manifest、Tokenizer、序列长度绑定。
- 一个真实 batch 的 forward、loss、backward 和梯度有限性。

只要出现 `FAIL` 就不要开始训练。`WARN` 表示当前路线尚未覆盖的检查或非阻断风险。

## 8. 启动训练

### 8.1 10M Smoke

```bash
uv run llmlab train pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --run-id native-smoke-001
```

Smoke 通过至少意味着：

- 模型可以实例化并执行 causal forward/backward。
- loss 和梯度为有限值。
- baseline 和训练后 dev 指标可计算。
- checkpoint 可以完整保存。

它不证明模型已经获得可用的语言能力。

### 8.2 60M Learn

```bash
uv run llmlab train pretrain \
  --config configs/pipelines/native-v1.yaml \
  --run-id native-60m-001
```

60M 使用 CUDA 和 bfloat16。训练前先根据显存调整：

1. 优先降低 `micro_batch_size`。
2. 再提高 `gradient_accumulation_steps` 维持有效 batch。
3. 仍然不足时降低 `sequence_length`。

不要为了跑通而同时改变模型、数据、Tokenizer 和优化参数，否则无法定位结果变化来源。

### 8.3 60M 结构消融

以下四条配方构成 2×2 对照，除模型结构外保持数据、Tokenizer、seed 和训练参数一致：

```bash
# 宽浅，QK-Norm 关闭
uv run llmlab train pretrain \
  --config configs/pipelines/native-v1.yaml \
  --run-id native-60m-baseline

# 宽浅，QK-Norm 开启
uv run llmlab train pretrain \
  --config configs/pipelines/native-60m-qk-norm.yaml \
  --run-id native-60m-qk-norm

# 深窄，QK-Norm 关闭
uv run llmlab train pretrain \
  --config configs/pipelines/native-60m-deep-narrow.yaml \
  --run-id native-60m-deep-narrow

# 深窄，QK-Norm 开启
uv run llmlab train pretrain \
  --config configs/pipelines/native-60m-deep-narrow-qk-norm.yaml \
  --run-id native-60m-deep-narrow-qk-norm
```

比较时先在相同宽深结构内判断 QK-Norm，再在相同 QK-Norm 设置内判断深窄结构，
最后才分析交互效应。不要把四组同时与不同学习率、token 预算或数据版本组合。

## 9. 训练指标

终端按 `log_interval` 输出训练进度，完整记录写入：

```text
runs/<run_id>/metrics.jsonl
```

第一行是随机初始化模型的 `baseline`。主要字段：

| 指标 | 说明 |
| --- | --- |
| `train_loss` | 当前 optimizer step 的 token-weighted causal LM loss |
| `eval_loss` | 固定 dev batch 上的 causal LM loss |
| `eval_perplexity` | `exp(eval_loss)`，为避免溢出只在计算时截断指数 |
| `train_bits_per_byte` | 训练 NLL 按 UTF-8 字节归一化 |
| `eval_bits_per_byte` | dev NLL 按 UTF-8 字节归一化 |
| `eval_en_*` / `eval_zh_*` | 相同 eval 样本上的中英文分桶 loss、perplexity、token 和 bits-per-byte |
| `gradient_norm` | clip 前的全局梯度范数 |
| `learning_rate` | 当前 optimizer step 实际使用的学习率 |
| `tokens_per_second` | 当前 step 的监督 token 吞吐 |
| `tokens_seen` | 已训练的监督 token 累计值 |
| `epochs_seen` | `tokens_seen / train split 监督 token` |
| `target_token_coverage` | 实际 token 对预算目标的覆盖比例 |
| `cuda_memory_allocated_bytes` | CUDA 当前已分配显存 |
| `cuda_max_memory_allocated_bytes` | CUDA 峰值已分配显存 |

16,384 词表下，随机模型 loss 通常接近：

```text
ln(16384) ≈ 9.70
```

应主要比较同一 run 的 step 0 baseline、训练过程和最终 dev 指标。不要仅根据 train loss 判断效果；train loss 下降而 dev loss 上升通常意味着过拟合或数据分布不一致。

## 10. Run 与 Checkpoint 产物

```text
runs/<run_id>/
├── resolved_config.yaml
├── run_manifest.json
├── data_snapshot.json
├── packed_data_snapshot.json
├── training_budget.json
├── runtime_environment.json
├── tokenizer_manifest.json
├── tokenizer/
│   └── tokenizer.json
├── model_config.json
├── model_manifest.json
├── metrics.jsonl
├── latest_checkpoint.json
├── training_result.json
├── checkpoints/
│   └── step-00000002/
│       ├── checkpoint_metadata.json
│       ├── trainer_state.json
│       ├── optimizer_state.pt
│       └── model/
│           ├── config.json
│           └── model.pt
└── evaluations/
```

Checkpoint 包含：

- 模型权重。
- optimizer 和 scheduler 状态。
- Torch CPU/CUDA RNG 状态。
- global step 和累计 token 数。
- 确定性 batch stream 的 epoch 与 offset。
- run config、Tokenizer 和模型路线兼容信息。

已有 run ID、Tokenizer 目录和 checkpoint 均禁止静默覆盖。

`runtime_environment.json` 记录 Python/PyTorch 版本、平台、实际 accelerator、CUDA/GPU
信息和 Git commit/dirty 状态。正式 60M 发布流程与自动验收见
[Native 60M Pretrain Reference Run](../reference_runs/native-60m-pretrain-v1/README.md)。

## 11. 恢复训练

从当前 run 的最新 checkpoint 恢复：

```bash
uv run llmlab train pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --resume-run native-smoke-001
```

指定 checkpoint：

```bash
uv run llmlab train pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --resume-run native-smoke-001 \
  --resume-checkpoint checkpoints/step-00000001
```

恢复训练要求使用与原 run 完全相同的配置 hash、模型路线和 Tokenizer hash，适用于
中断后继续执行原定训练预算。

当前不支持在同一 run 中修改 `max_steps`、`max_train_tokens` 或 `num_epochs`，也不支持
从较早 checkpoint 覆盖已经存在的后续 checkpoint。此类实验应创建新的 run；
“从 checkpoint 分叉为新 run”将在后续阶段实现。

## 12. 独立评测

评测 dev：

```bash
uv run llmlab eval pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002 \
  --split dev
```

输出 JSON：

```bash
uv run llmlab eval pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002 \
  --split test \
  --json
```

评测前会检查 checkpoint 的 config hash、模型路线、stage 和 Tokenizer hash。标准 run 中的报告同时写入：

```text
runs/<run_id>/evaluations/pretrain-<split>-step-<step>.json
```

当前版本同时报告整个双语 split 的 aggregate loss、perplexity、bits-per-byte，以及
`eval_en_*`、`eval_zh_*` 分桶指标。评测样本按语言确定性轮转选择，但仍保留原始 packed
上下文，不重排或重切文档。aggregate loss 可以由中英文 loss 按监督 token 数加权还原；
比较实验时必须同时检查两种语言，不能只用 aggregate 掩盖单语退化。

## 13. 常见失败

| 错误 | 原因与处理 |
| --- | --- |
| `output directory already exists` | 数据或 Tokenizer 目录已存在；使用新版本目录，禁止隐式覆盖 |
| `pretraining requires a non-empty dev split` | 数据分组过少；增加数据或调整 split 比例后重新 prepare |
| `tokenizer vocab_size ... does not match model vocab_size` | 实际 BPE 词表不足 16,384 或使用了错误 Tokenizer |
| `tokenizer was trained from a different Data Manifest` | Tokenizer 与当前预训练数据不是同一版本 |
| `packed pretraining data is incompatible` | Packed Manifest 与当前数据、Tokenizer 或序列长度不一致；重新 pack 到新目录 |
| `packed output already exists` | Packed 产物不可覆盖；复用已校验版本或使用新版本目录 |
| `training.device=cuda but CUDA is unavailable` | 当前环境不能运行 60M 配方；使用正确 GPU 环境 |
| `resume config does not match` | 修改了原 run 配置；恢复必须使用完全相同的 resolved config |
| `checkpoint already exists` | 尝试从旧 checkpoint 覆盖后续历史；新建 run |
| `training loss became non-finite` | 学习率、精度、数据或梯度异常；保留 failure artifact 后排查 |
| `input sequence exceeds max_sequence_length` | `sequence_length` 或生成长度超过模型配置 |

## 14. 当前限制

- 仅支持单进程、单设备训练。
- Token packing 已磁盘化，但当前为单进程顺序分词，尚未实现并行预处理或分片加载。
- 60M CUDA 配方尚未在当前 macOS 工作区完成 24GB GPU 实测。
- 未实现 gradient checkpointing、DDP 和分布式数据加载。
- Native 模型尚未接入 SFT、DPO、GRPO 和 HF 导出。
- 10M/60M 是教学模型，不应与同参数量工业预训练模型直接比较能力。

## 15. 最短操作顺序

```bash
uv sync --extra public-data --extra training --extra dev

uv run llmlab data fetch \
  --recipe simplestories-smoke-v1 \
  --output data/raw/simplestories-smoke-v1 \
  --accept-license MIT

uv run llmlab data fetch \
  --recipe wikipedia-zh-smoke-v1 \
  --output data/raw/wikipedia-zh-smoke-v1 \
  --accept-license CC-BY-SA-3.0

uv run llmlab data mix \
  --mixture bilingual-smoke-v1 \
  --input data/raw/simplestories-smoke-v1/source.jsonl \
  --input data/raw/wikipedia-zh-smoke-v1/source.jsonl \
  --output data/raw/bilingual-smoke-v1

uv run llmlab data prepare \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --output data/prepared/bilingual-smoke-v1 \
  --dataset-id bilingual-smoke-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --group-by source_id

uv run llmlab tokenizer train \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --output data/tokenizers/bilingual-smoke-v1 \
  --tokenizer-id bilingual-smoke-v1 \
  --vocab-size 16384 \
  --min-frequency 1

uv run llmlab data pack \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-smoke-v1 \
  --output data/packed/bilingual-smoke-v1-seq128 \
  --sequence-length 128

uv run llmlab doctor --config configs/pipelines/native-smoke.yaml

uv run llmlab train pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --run-id native-smoke-001

uv run llmlab eval pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002
```
