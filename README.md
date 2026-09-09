# LLM Lifecycle Lab

面向中文用户的 LLM 学习与实验项目：从随机初始化的自有 10M/60M 模型开始，
用可追溯的数据、配置、checkpoint 和双语评测观察训练过程。

当前已实现 Native 模型、BPE Tokenizer、不可变磁盘 Packing、Pretrain、恢复和评测。
SFT、DPO、GRPO、Qwen 迁移、导出和服务尚未实现；仓库不附带训练好的权重。

## 三份文档

| 文档 | 内容 |
| --- | --- |
| [数据介绍与准备](./docs/DATA_GUIDE.md) | 中英文数据来源与许可、公开/自有数据、切分、Tokenizer、Packing、校验与迁移 |
| [自有模型介绍](./docs/NATIVE_MODEL_GUIDE.md) | 10M/60M 结构、参数预算、张量形状、RMSNorm/RoPE/GQA、前向与 KV Cache 实验 |
| [Pretrain 训练文档](./docs/NATIVE_PRETRAIN_GUIDE.md) | Conda 安装、Smoke/60M 训练、配置、指标、恢复、评测、Reference 验收与排错 |

首次实践顺序：安装环境 -> 按数据文档准备 Smoke 数据 -> 运行两步训练与评测。
原理学习可先看模型文档，其中模型实验不需要下载语料。

## 快速开始

从仓库根目录执行。Linux CPU/CUDA 请先按
[环境准备](./docs/NATIVE_PRETRAIN_GUIDE.md#2-环境准备) 选择 PyTorch wheel。

```bash
conda create -n llm-lifecycle-lab python=3.11 pip -y
conda activate llm-lifecycle-lab
python -m pip install -r requirements.txt
python scripts/doctor.py
```

按 [数据文档](./docs/DATA_GUIDE.md#4-第一次实践10m-双语-smoke) 完成下载、混合、切分、
Tokenizer 和 Packing 后：

```bash
python scripts/doctor.py --config configs/pipelines/native-smoke.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --run-id native-smoke-001
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002
```

已有同名数据或 run 时不要覆盖；有效数据可以校验复用，新训练使用新 run ID。
10M Smoke 支持 CPU/MPS/CUDA，用于验证链路，不代表语言能力。
60M 目标环境为 Linux + 单张 24GB NVIDIA GPU，完整 CUDA 参考实验尚未完成。

## 目录

```text
configs/       模型、Pipeline 和 Reference 验收 YAML
docs/          数据、自有模型、Pretrain 三份文档
scripts/       Python 操作入口
src/           共享实现
tests/         单元与集成测试
.github/       CI
data/          本地训练数据，不进入 Git
runs/          本地实验产物，不进入 Git
```

依赖以 `pyproject.toml` 为来源，`requirements.txt` 提供 Conda/Pip 安装入口；
`uv.lock` 保留严格锁环境用途。Python 脚本与旧 `llmlab` 命令共享实现，旧入口继续兼容。

## 开发验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

权重、数据和历史运行不是缓存，不应随普通构建缓存一起删除。完整目录职责与清理边界见训练文档。
