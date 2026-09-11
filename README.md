# LLM Lifecycle Lab

面向中文用户的 LLM 学习与实验项目：从随机初始化的自有 10M/60M 模型开始，
用可追溯的数据、配置、checkpoint 和双语评测观察训练过程。

当前已实现 Native 模型、BPE Tokenizer、不可变磁盘 Packing、Pretrain、恢复和评测。
SFT、DPO、GRPO、Qwen 迁移、导出和服务尚未实现；仓库不附带训练好的权重。
10M 与 60M 统一使用 `model_route: native`，由 Pipeline 的 `run_profile` 区分
Smoke、教学训练和 Reference 验收规模。

## 学习与查阅

**按顺序学习**：从 [实践系列目录](./docs/tutorials/README.md) 开始。
第一篇 [从一次参数更新开始](./docs/tutorials/01-first-parameter-update.md)
不需要下载语料或使用 GPU，沿实际脚本解释前向、loss、反向和更新，并附单变量实验。
后续按数据、Tokenizer、模型、预训练、评测和实验复现继续展开。

**按需查阅**：完整操作步骤、配置说明和排错集中维护在以下三份指南中。

| 文档 | 内容 |
| --- | --- |
| [数据介绍与准备](./docs/DATA_GUIDE.md) | 中英文数据来源与许可、公开/自有数据、切分、Tokenizer、Packing、校验与迁移 |
| [自有模型介绍](./docs/NATIVE_MODEL_GUIDE.md) | 10M/60M 结构、参数预算、张量形状、RMSNorm/RoPE/GQA、前向与 KV Cache 实验 |
| [Pretrain 训练文档](./docs/NATIVE_PRETRAIN_GUIDE.md) | Conda 安装、Smoke/60M 训练、配置、指标、恢复、评测、Reference 验收与排错 |

完整训练的首次实践顺序：安装环境 -> 按数据文档准备 Smoke 数据 -> 运行两步训练与评测。

## 快速开始

从仓库根目录执行。Linux CPU/CUDA 请先按
[环境准备](./docs/NATIVE_PRETRAIN_GUIDE.md#2-环境准备) 选择 PyTorch wheel。

```bash
conda create -n llm-lifecycle-lab python=3.11 pip -y
conda activate llm-lifecycle-lab
python -m pip install -r requirements.txt
python scripts/doctor.py
```

不下载语料，先观察一次模型前向、梯度、参数更新与 KV Cache：

```bash
python scripts/model_experiment.py
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

第一份冻结基线为
[`native-60m-baseline-v1`](./configs/reference/native-60m-baseline-v1.yaml)：
宽浅 62.93M、QK-Norm 关闭、1 epoch、46,184,530 个目标监督 token。
它锁住配置、输入产物、源码和训练依赖，尚不代表已验证的模型权重。
已有配套 60M 数据时，可先运行只读校验：

```bash
python scripts/verify_reference.py \
  --spec configs/reference/native-60m-baseline-v1.yaml --inputs-only
```

正式环境检查、带门控的训练和最终双语验收见
[固定 60M 基线](./docs/NATIVE_PRETRAIN_GUIDE.md#92-固定-60m-基线)。

## 目录

```text
configs/       模型、Pipeline 和 Reference 验收 YAML
docs/          三份操作与原理指南，以及 tutorials/ 连续实践文章
scripts/       面向学习者的 Python 操作入口
src/           共享实现
tests/         单元与集成测试
.github/       CI
data/          本地训练数据，不进入 Git
runs/          本地实验产物，不进入 Git
```

`requirements.txt` 只安装第三方运行依赖；`scripts/*.py` 会直接加载仓库的 `src/`
共享实现，无需把项目安装进 Conda 环境。`pyproject.toml` 保留项目元数据和可选打包入口，
`uv.lock` 用于严格锁环境。阅读代码时从 [scripts/README.md](./scripts/README.md) 和对应
脚本的 `main()` 开始，再沿脚本直接导入的核心函数进入算法实现。

## 开发验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

权重、数据和历史运行不是缓存，不应随普通构建缓存一起删除。完整目录职责与清理边界见训练文档。
