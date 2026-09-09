# LLM Lifecycle Lab

> 从预训练到后训练，观察每一步，验证每次改变。

LLM Lifecycle Lab 是一个面向中文用户和初学者的 LLM 全生命周期学习与实验项目。项目使用支持中英文训练的原生 10M/60M 模型讲解完整训练链路，并计划通过 Qwen3-0.6B-Base 验证后训练协议向真实预训练模型的迁移。项目文档和 CLI 操作说明优先使用中文。

项目已经完成工程地基、Native 10M/60M 模型、BPE Tokenizer 和 Pretrain 训练闭环。SFT、DPO、GRPO 与 Qwen 迁移路线尚未实现。

## 核心目标

- 让训练过程可观察，而不只是提供可运行脚本。
- 让代码、配置、数据、checkpoint 和评测结果可追溯。
- 通过统一基线比较每个训练阶段带来的改善与退化。
- 通过受控故障学习数据、监督信号、奖励和导出协议。
- 最终让学习者能够替换自己的数据并交付可验证的模型。

## 首版主线

```text
Data
  -> Tokenizer
  -> Pretrain
  -> SFT
  -> DPO
  -> GRPO
  -> Evaluation
  -> Export & Serving
```

首版只覆盖纯文本 LLM，正式支持目标为 Linux + 单张 24GB NVIDIA GPU，同时提供不承诺模型质量的 CPU/MPS smoke 配方。

## 当前可用

项目要求 Python 3.11 或更高版本。

```bash
uv sync --extra public-data --extra training --extra dev

uv run llmlab doctor
uv run llmlab model inspect --config configs/models/smoke-10m.yaml
uv run llmlab config validate configs/pipelines/native-smoke.yaml
uv run llmlab config validate configs/pipelines/native-v1.yaml
```

拉取固定版本的英文 SimpleStories 与中文 Wikipedia Smoke 数据，生成双语确定性切分：

```bash
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
```

训练 16K BPE Tokenizer、生成不可变磁盘 packing，并运行两步 10M CPU/MPS Smoke：

```bash
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
  --run-id native-smoke-example

uv run llmlab eval pretrain \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-example/checkpoints/step-00000002
```

训练自动记录随机初始化 baseline、训练/验证 loss、perplexity、bits-per-byte、中英文
分桶指标、训练预算覆盖率、梯度范数、学习率、吞吐、checkpoint 和恢复所需状态。
Tokenizer 的实际词表大小必须与模型配置一致。

当前 CLI 仅实现 Native Pretrain 及其评测，不包含 SFT、DPO、GRPO、`export` 和 `serve`。

## 文档

- [Native 10M/60M 模型结构说明](./docs/NATIVE_MODEL_GUIDE.md)
- [Pretrain 数据准备 Runbook](./docs/DATA_PREPARATION_RUNBOOK.md)
- [Native 10M/60M Pretrain 训练指南](./docs/NATIVE_PRETRAIN_GUIDE.md)
- [Native 60M Pretrain Reference Run](./reference_runs/native-60m-pretrain-v1/README.md)

## 当前状态

- [x] Contracts、Artifacts、ModelProtocol 与 Doctor
- [x] JSONL 数据校验、确定性切分与 Data Manifest
- [x] 中英文公开数据 recipe、确定性混合与来源追踪
- [x] Native BPE Tokenizer
- [x] 不可变磁盘 Token Packing、训练预算与中英文分桶评测
- [x] Native 10M/60M 模型与 Pretrain
- [ ] SFT 与单样本监督追踪
- [ ] DPO 与 GRPO
- [ ] 统一评测、导出与服务

## 名称说明

- 项目名：`LLM Lifecycle Lab`
- 仓库名：`llm-lifecycle-lab`
- Python 包名：`llm_lifecycle_lab`
- CLI：`llmlab`

名称强调三件事：LLM 是当前明确范围，Lifecycle 表示从数据和预训练到后训练与交付，Lab 表示项目以可复现实验而不是功能堆叠为核心。
