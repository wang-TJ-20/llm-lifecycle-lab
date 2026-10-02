<div align="center">

# LLM Lifecycle Lab

**从一次参数更新开始，亲手理解语言模型的训练过程**

原生 PyTorch 小模型 · 中英文数据 · 七篇实践教程 · Base → SFT 可复现链路

[在线阅读](https://wang-tj-20.github.io/llm-lifecycle-lab/) · [公开权重](https://modelscope.cn/models/wzt777/native-60m-base-v1) · [教程目录](#教程目录) · [快速开始](#快速开始) · [项目进展](#项目进展)

</div>

---

## 项目介绍

一个随机初始化的模型，怎样逐步学会预测下一个 token？训练时，参数为什么会改变？
loss 下降能说明什么？进程中断后，怎样继续同一次实验？

**LLM Lifecycle Lab 是一个面向中文读者的 LLM 学习与实验项目。**
我们把可读的代码、连续的教程和最小对照实验放在同一个仓库，
从自有小模型出发，串起数据准备、Tokenizer、Transformer、预训练、评测与恢复。
读者可以先在 CPU 上观察机制，再进入真实双语数据与 60M 教学训练。

适合能阅读基础 Python、希望进一步理解模型训练的学习者。
不要求先掌握全部 Transformer 公式；每篇沿着
**问题 → 例子与图解 → 原理与代码 → 实验 → 结果与边界** 展开。

### 在这里可以学到什么

- **看懂一次更新**：从张量形状进入 forward、loss、backward 和 optimizer step。
- **把文本变成训练样本**：准备中英文数据，按原文分组切分，训练 BPE 并生成 Packing。
- **理解模型内部**：沿代码验证 RMSNorm、RoPE、GQA、SwiGLU、因果注意力与 KV Cache。
- **读懂训练结果**：区分 step、监督 token 和预算，结合双语指标与生成解释模型表现。
- **做可比较的实验**：固定输入和配置，检查 checkpoint、数据位置与中断恢复。

> 七篇预训练主线、Native-60M Base-v1 的 8-epoch CUDA 训练及发布均已完成；
> SFT 数据、训练、恢复和评测链路已冻结，正式 SFT-v1 尚未运行。
> 权重与 Tokenizer 可从 [ModelScope](https://modelscope.cn/models/wzt777/native-60m-base-v1)
> 下载，也保存在 `models/base-v1-final/`。它是 Base 模型，不是指令微调后的聊天模型。

## 选择你的起点

| 路径 | 运行条件 | 适合做什么 |
| --- | --- | --- |
| 离线机制实验 | CPU；安装依赖后无需下载语料 | 观察参数更新，验证微型训练、评测和恢复 |
| 10M 双语 Smoke | CPU / Apple Silicon MPS / CUDA；需准备公共语料 | 跑通真实数据上的两步训练与评测 |
| 已发布 Native-60M Base | CPU 可加载；CUDA 可加速生成；约 252 MB | 下载权重、校验来源并观察 Base 续写 |
| 60M 训练复现与 Reference | 目标为 Linux + 单张 24GB NVIDIA GPU；需配套数据 | 从零复现训练、结构对照与固定基线验收 |
| Native-60M SFT-v1 | 本地 MPS 门禁 + 远端 Linux/CUDA；需冻结 SFT 数据 | 从已发布 Base 进行指令微调、恢复与同协议评测 |

**第一次接触，先看第一篇并运行离线实验。**
10M 用于快速验证，60M 是效果研究的主基线；无需先购买 GPU 才开始学习。
Reference 比普通教学训练有更严格的输入、环境和结果要求。

## 教程目录

七篇主线均已完成，可在 [文档站](https://wang-tj-20.github.io/llm-lifecycle-lab/#/tutorials/)
连续阅读，也可直接打开仓库 Markdown。

| 章节 | 核心问题与实验 |
| --- | --- |
| [01 从一次参数更新开始](./docs/tutorials/01-first-parameter-update.md) | 参数怎样改变？用单变量实验观察梯度、更新与一致性。 |
| [02 准备中英文训练数据](./docs/tutorials/02-bilingual-training-data.md) | 数据从哪里来？通过原文分组对照检查评测泄漏。 |
| [03 让模型读懂文本的表示](./docs/tutorials/03-tokenizer-and-packing.md) | 文本怎样变成 ID？比较词表预算、规范化和窗口组织。 |
| [04 搭建自己的小型 Transformer](./docs/tutorials/04-small-transformer.md) | 各组件怎样连接？验证 Attention、RoPE、因果性与 Cache。 |
| [05 跑通一次预训练](./docs/tutorials/05-first-pretraining.md) | batch、累积、学习率和预算怎样共同决定训练过程？ |
| [06 判断模型到底学到了什么](./docs/tutorials/06-evaluating-a-model.md) | 总体 loss 变好是否足够？核对双语覆盖、指标与生成。 |
| [07 让实验可以恢复和比较](./docs/tutorials/07-resume-and-compare.md) | 保存权重为什么不够？比较连续训练与中断恢复的状态。 |

阅读顺序与实验约定见 [实践系列导读](./docs/tutorials/README.md)。
教程解释“为什么”，操作指南维护完整参数与排错步骤。

## 快速开始

### 1. 准备环境

新机器上克隆仓库并创建环境；已有仓库或同名环境时直接进入、激活即可。
后续命令均从仓库根目录执行。

```bash
git clone https://github.com/wang-TJ-20/llm-lifecycle-lab.git
cd llm-lifecycle-lab
conda create -n llm-lifecycle-lab python=3.11 pip -y
conda activate llm-lifecycle-lab
```

macOS Apple Silicon 可直接安装下面的依赖；Linux CPU/CUDA 请先按
[环境准备](./docs/NATIVE_PRETRAIN_GUIDE.md#2-环境准备) 选择对应的 PyTorch wheel。

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/doctor.py
```

Python 要求为 3.11 或更高。脚本直接加载仓库中的 `src/`，
**不需要 `pip install -e .` 或手动设置 `PYTHONPATH`**。
基础 Doctor 不替代真实训练前的配置化检查。

### 2. 下载并验证公开 Base

发布仓库为
[`wzt777/native-60m-base-v1`](https://modelscope.cn/models/wzt777/native-60m-base-v1)。
以下命令把权重、Tokenizer、模型配置和 provenance 下载到 Git 忽略的 `build/`：

```bash
python -m pip install "modelscope-hub==0.1.8"
ms download wzt777/native-60m-base-v1 \
  --repo-type model \
  --local-dir build/native-60m-base-v1
```

在 macOS 上校验全部发布文件：

```bash
(cd build/native-60m-base-v1 && shasum -a 256 -c SHA256SUMS)
```

Linux 使用 `sha256sum -c SHA256SUMS`。加载时必须让模型配置、权重与 Tokenizer
来自同一发布目录：

```python
from pathlib import Path

from llm_lifecycle_lab.model.native import (
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.tokenizer import NativeTokenizer

model_dir = Path("build/native-60m-base-v1")
tokenizer = NativeTokenizer.from_directory(model_dir)
model = NativeTransformer(load_native_model_config(model_dir / "config.json"))
model.load(model_dir)
model.eval()

print(model.parameter_count)  # 62927616
print(tokenizer.vocab_size)   # 16384
```

该包使用项目自有加载器，不兼容 Hugging Face `AutoModelForCausalLM`。
完整来源、指标、限制与许可见
[`models/base-v1-final/README.md`](./models/base-v1-final/README.md)。

### 3. 不下载数据，先观察一次参数更新

```bash
python scripts/model_experiment.py
```

默认在 CPU 上运行随机初始化的 10M 模型，完成前向、loss、反向和一次参数更新，
再检查 KV Cache 与模型保存加载的一致性。
输入为 `[2, 16]`，logits 为 `[2, 16, 16384]`，有效预测目标为 30 个。

重点检查 `weight_update_max > 0`、`cache_matches_full=True` 和
`checkpoint_matches_full=True`。随机 token 的 loss 不用于判断语言能力。
逐步解释见 [第一篇](./docs/tutorials/01-first-parameter-update.md)。

### 4. 跑通微型训练、评测与恢复

```bash
python scripts/pretrain_experiment.py --mode train
python scripts/pretrain_experiment.py --mode evaluate
python scripts/pretrain_experiment.py --mode resume
```

三个模式各自在临时目录生成双语模板数据、训练 Tokenizer，并运行配套微型模型。
`train` 检查三步训练与产物；`evaluate` 核对双语指标和生成；
`resume` 比较连续训练与受控中断恢复的状态。

**这些命令不修改已有 `data/` 或 `runs/`，退出后自动清理临时产物。**
这里的微型模型不是正式 10M/60M 配方，模板数据也不用于证明语言能力。
完整解读见 [第五至七篇](./docs/tutorials/05-first-pretraining.md)。

### 5. 在真实双语数据上运行 Smoke

先按 [数据指南](./docs/DATA_GUIDE.md#4-第一次实践10m-双语-smoke)
确认许可，完成下载、混合、切分、Tokenizer 训练和 Packing，再执行：

```bash
python scripts/doctor.py --config configs/pipelines/native-smoke.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --run-id native-smoke-001
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002 \
  --split dev
```

结果保存在 `runs/native-smoke-001/`，包括配置、预算、环境、指标和 checkpoint。
已有同名 run 时换新 ID，不要删除旧记录来绕开覆盖保护。
恢复只用于完成原预算，不能给已经结束的两步 Smoke 直接追加训练。

60M 当前采用“本地 MPS smoke → 本地 MPS pilot → 远端 CUDA Base”三级门禁，执行入口见
[Native-60M Base 从零训练手册](./docs/BASE_TRAINING_GUIDE.md)。

## 模型与数据

### 两档模型，同一份实现

Native 是自有的 decoder-only Transformer，从随机权重训练，不是裁剪或微调 Qwen。
结构采用 RMSNorm、RoPE、SwiGLU、GQA 和权重绑定，支持因果前向与 KV Cache 生成。
模型和训练循环基于 PyTorch，BPE 使用 Hugging Face `tokenizers`。

| 配置 | 参数量 | 层数 × Hidden | Q / KV 头 | 最大长度 |
| --- | ---: | --- | --- | ---: |
| [smoke-10m](./configs/models/smoke-10m.yaml) | 9,915,200 | 4 × 320 | 5 / 1 | 256 |
| [tiny-60m](./configs/models/tiny-60m.yaml) | 62,927,616 | 8 × 768 | 12 / 4 | 512 |

两档默认词表均为 16,384，QK-Norm 关闭，但各自 Tokenizer 的 ID 映射不能互换。
模型长度上限不等于训练长度：10M Smoke 默认训练 seq128，60M 为 seq512。
两者共用 `model_route: native`，具体结构由模型配置选择，
实践级别由 `run_profile` 区分。

### 已发布的 Base-v1

`Native-60M Base v1` 从随机初始化训练 8 epochs，共处理 369,480,328 个监督 token，
最终 step 为 45,191。固定 1,024-window test 评测得到总体 loss 2.731877、
英文 loss 1.955555、中文 loss 3.419416。中文指标仍明显弱于英文，因此该版本应作为
可复现实验基线，而不是通用双语助手。

| 项目 | 位置 |
| --- | --- |
| 公开下载 | [ModelScope: `wzt777/native-60m-base-v1`](https://modelscope.cn/models/wzt777/native-60m-base-v1) |
| 仓库发布包 | [`models/base-v1-final/`](./models/base-v1-final/) |
| 完整训练证据 | [`results/native-60m-base-v1-s42/`](./results/native-60m-base-v1-s42/) |
| 执行结果 | [`docs/BASE_TRAINING_RESULTS.md`](./docs/BASE_TRAINING_RESULTS.md) |
| 权重 SHA-256 | `4cdbd642cb878c0e7f5de4b7988751e92de533afa1317256ace20aea078f9398` |

### 默认覆盖中文与英文

| 语言 | 数据来源 | 内容与来源许可 |
| --- | --- | --- |
| 英文 | [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | 合成故事；MIT |
| 中文 | [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 中文百科；CC-BY-SA-3.0 |

Recipe 固定来源、版本与选取方式，Manifest 记录实际产物和 hash。
默认流程显式按 `source_id` 切分，Tokenizer 只在 train 上学习，
dev/test 使用同一词表独立编码。单语配置用于对照，不是默认学习路线。
使用数据前需自行核对来源许可，格式校验和 hash 不能代替内容质量、隐私与去重检查。

```mermaid
flowchart TD
  accTitle: 从中英文数据到可检查的训练实验
  accDescr: 数据按来源组切分，训练 BPE 并生成 Packing，接入 Native 预训练，再进行固定范围评测与检查点恢复对照。
  A["中英文数据 · 分组切分"] --> B["BPE Tokenizer"]
  B --> C["Packing · 训练窗口"]
  C --> D["Native · 预训练"]
  D --> E["固定范围评测"]
  D --> F["Checkpoint · 恢复"]
```

各阶段的完整代码、张量形状与校验边界见 [模型指南](./docs/NATIVE_MODEL_GUIDE.md)
和 [数据指南](./docs/DATA_GUIDE.md)。

## 项目进展

| 状态 | 内容 |
| --- | --- |
| 已完成 | 七篇中文教程、Docsify 在线阅读站与 CPU 离线实验 |
| 已实现 | Native 10M/60M、双语 BPE、磁盘 Packing、Pretrain、评测与 checkpoint 恢复 |
| 已发布 | Native-60M Base-v1；CUDA 训练门禁、独立 test、发布包与 ModelScope 回拉验证均通过 |
| 已冻结 | SFT Base、公开数据、assistant-only Trainer、quota sampler、评测协议与正式 run 门禁 |
| 已验证 | SFT 数据 14/14 门禁、本地 8-step smoke、精确恢复和完整命令链 |
| 当前准备 | 本地 100-step pilot 后，在远端执行 8M-token SFT-v1 |
| 后续方向，尚未实现 | DPO、GRPO、Hugging Face 导出与模型服务 |

Base-v1 从随机权重开始，只使用
[`native-60m-base-local-smoke.yaml`](./configs/pipelines/native-60m-base-local-smoke.yaml)、
[`native-60m-base-local-pilot100.yaml`](./configs/pipelines/native-60m-base-local-pilot100.yaml)
和 [`native-60m-base-v1.yaml`](./configs/pipelines/native-60m-base-v1.yaml)。
固定边界与复现命令见 [Base 从零训练手册](./docs/BASE_TRAINING_GUIDE.md)，
实测结果与已知 provenance 偏离见
[Base-v1 执行结果](./docs/BASE_TRAINING_RESULTS.md)。

当前精确恢复对照覆盖 CPU；不承诺所有设备和精度逐位一致。
评测受配置中的样本预算限制，不默认代表全量 dev/test。
这些边界与实验结论一起记录，不用 Smoke 指标代替正式模型效果。

## 文档与代码

| 入口 | 用途 |
| --- | --- |
| [数据介绍与准备](./docs/DATA_GUIDE.md) | 来源、许可、公开/自有数据、切分、Tokenizer、Packing 与迁移 |
| [自有模型介绍](./docs/NATIVE_MODEL_GUIDE.md) | 模型结构、参数预算、前向、生成与结构消融 |
| [Base 从零训练手册](./docs/BASE_TRAINING_GUIDE.md) | Base-v1 的本地门禁、远端 CUDA 配方与复现步骤 |
| [Base-v1 执行结果](./docs/BASE_TRAINING_RESULTS.md) | 最终指标、训练证据、发布包与已知限制 |
| [SFT 训练手册](./docs/SFT_TRAINING_GUIDE.md) | 数据物化、本地门禁、远端训练/恢复、dev、sealed test 与归档 |
| [SFT 初始化门禁](./docs/SFT_INITIALIZATION_GATE.md) | 冻结 Base-v1 初始化身份、质量证据与 clean-run 要求 |
| [SFT 数据门禁](./docs/SFT_DATA_GATE.md) | 冻结公开来源、许可、切分、对话协议与数据质量要求 |
| [Pretrain 训练文档](./docs/NATIVE_PRETRAIN_GUIDE.md) | 环境、训练、评测、恢复、Reference 验收与排错 |
| [脚本阅读指南](./scripts/README.md) | 从每个脚本的 `main()` 进入核心实现 |
| [文档站维护](./docs/SITE_GUIDE.md) | 本地预览、GitHub Pages 发布与新增章节 |

```text
configs/       模型、Pipeline、Reference 与阶段门禁配置
docs/          七篇教程、操作指南与阅读站
scripts/       可直接运行的实践入口
src/           模型、数据与训练共享实现
tests/         单元、集成与文档站检查
.github/       CI
data/          本地语料与数据产物，Git 忽略
runs/          本地训练记录与权重，Git 忽略
models/        经审核发布的模型包；大文件使用 Git LFS
results/       经筛选提交的 canonical 实验证据
```

从脚本进入实现，从教程理解实验；权重、数据和历史 run 不是可以随意清理的缓存。
严格锁环境使用 `uv.lock`，具体安装方式见 [环境准备](./docs/NATIVE_PRETRAIN_GUIDE.md#2-环境准备)。

## 参与贡献

欢迎通过 [Issue](https://github.com/wang-TJ-20/llm-lifecycle-lab/issues)
报告问题、讨论学习中的疑问，或提交
[Pull Request](https://github.com/wang-TJ-20/llm-lifecycle-lab/pulls)
改进代码、教程与实验。

- 报错请附命令、配置、Python/PyTorch 版本、设备和脱敏日志。
- 修改行为时补充对应测试，并同步受影响的教程与示例输出。
- 提交实验结果时记录数据版本、预算和对照条件，区分观察与推测。
- 不提交训练语料、凭据或未经审核的本地运行产物；正式发布权重必须使用 Git LFS，
  并同时提交 provenance、校验和、模型卡与许可文件。

<details>
<summary>开发验证命令</summary>

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

[CI](./.github/workflows/ci.yml) 在 Python 3.11 CPU 环境检查
requirements 与锁环境两条安装路径。
文档站浏览器检查另见 [文档站维护](./docs/SITE_GUIDE.md#浏览器检查)。

</details>

## 参考与致谢

- [Happy-LLM](https://github.com/datawhalechina/happy-llm)：从基础概念到 LLM 训练实践的系统性教程。
- [MiniMind](https://github.com/jingyaogong/minimind)：小语言模型的结构、训练与实验实践。

感谢这些项目公开分享教学与实践资料。本 README 借鉴其课程导航与实践入口的组织方式；
本项目的能力、命令和验证状态以本仓库实现为准。

## 开源许可

源代码采用 [Apache License 2.0](./LICENSE)。Native-60M Base-v1 的
`model.pt`、`tokenizer.json` 和 `tokenizer_manifest.json` 采用
[CC-BY-SA-4.0](./models/base-v1-final/LICENSE_MODEL)；公开数据与第三方资源
仍各自遵循原有许可。具体边界见
[`models/base-v1-final/NOTICE.md`](./models/base-v1-final/NOTICE.md)。
