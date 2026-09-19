# 脚本阅读指南

`scripts/` 是面向学习者的运行入口。每个脚本都直接包含：

1. 命令行参数；
2. 当前任务的执行顺序；
3. 对核心函数的直接调用；
4. 结果输出和预期错误处理。

脚本不会再经过统一 CLI 分发。`_project_path.py` 只负责让脚本找到仓库中的 `src/`
目录，不包含业务逻辑。

脚本顶部说明统一使用中英文双语，中文在前、英文在后；流程说明和实现位置保持一致。

## 建议阅读顺序

| 实践步骤 | 先读的脚本 | 核心实现 |
| --- | --- | --- |
| 准备公开预训练与后训练数据 | `data.py` | `src/llm_lifecycle_lab/data/` |
| 训练 Tokenizer | `train_tokenizer.py` | `src/llm_lifecycle_lab/tokenizer/native.py` |
| 检查 Tokenizer | `inspect_tokenizer.py` | `NativeTokenizer.encode/decode` |
| 理解模型规模 | `inspect_model.py` | `src/llm_lifecycle_lab/model/native/` |
| 实践一次参数更新 | `model_experiment.py` | 脚本中直接展示 forward、loss、backward、step 和 Cache 检查 |
| 离线预训练、评测与恢复对照 | `pretrain_experiment.py` | 临时双语数据与真实 `run_native_pretraining` / `evaluate_native_pretraining` |
| 训练模型 | `train_pretrain.py` | `src/llm_lifecycle_lab/training/pretrain.py` |
| 评测模型 | `eval_pretrain.py` | `evaluate_native_pretraining` |
| 固定提示续写 | `generate_pretrain.py` | 校验 checkpoint/Tokenizer/配置绑定后执行确定性贪心生成 |
| 绘制训练曲线 | `plot_training_curves.py` | 从 `metrics.jsonl` 生成无额外依赖的 SVG |
| 冻结基线与验收 | `verify_reference.py` | `src/llm_lifecycle_lab/reference.py` |
| 构建模型发布包 | `build_model_release.py` | 安全解包、绑定 Tokenizer、生成报告与文件 hash |
| 下载并验证公开模型 | `run_published_model.py` | ModelScope 下载、完整性校验、续写与 Smoke 评测 |
| 统一能力评测及纵向比较 | `evaluate_capabilities.py` | `evaluation/` 中的协议、评分、语料检查与报告 |
| SFT 训练与恢复 | `train_sft.py` | `data/sft.py` 与 `training/sft.py` |
| 离线 SFT 闭环 | `sft_experiment.py` | 临时 Base → SFT → 同协议评测及恢复对照 |
| 续写与多轮对话 | `chat.py` | Native/HF 后端、显式上下文策略 |
| Native → HF 导出 | `export_hf.py` | 标准模型、Tokenizer 及 CPU 一致性验收 |
| CPU 推理与量化基准 | `benchmark_inference.py` | FP32/dynamic INT8 状态、吞吐、RSS 与 greedy 对照 |
| 固定 Qwen 快照 | `prepare_transfer.py` | revision 与本地文件 hash 门禁 |
| Qwen Full/LoRA SFT | `train_transfer.py` | 共用 SFT 数据、评测与训练引擎 |
| 离线 Qwen/LoRA 验证 | `transfer_experiment.py` | 随机微型架构、冻结及恢复，不下载权重 |
| Native DPO | `train_dpo.py` | response log-prob、冻结 reference 与 pair loss |
| 离线 DPO 闭环 | `dpo_experiment.py` | 临时 Pretrain → SFT → DPO 与恢复对照 |
| Native GRPO/RLVR | `train_grpo.py` | 程序化奖励、组内优势、KL 与 rollout |
| 离线 GRPO 闭环 | `grpo_experiment.py` | 自适应合成奖励与精确恢复对照 |
| 本地 OpenAI-compatible API | `serve_openai.py` | Native/HF 单模型 completion 与 chat HTTP 接口 |

以 `train_tokenizer.py` 为例，先读 `main()`，可以看到输入 manifest、训练参数和输出；
然后直接跳转到 `train_native_tokenizer()`，查看 BPE 初始化、训练数据迭代和文件保存。

不下载数据也可以先运行 `python scripts/model_experiment.py`。随机 token 只用于理解
张量形状、梯度和更新过程，不代表语言能力。模型的构建与前向在 `transformer.py`，
完整生成循环在 `generation.py`，数学组件在 `layers.py` 和 `attention.py`。

逐步讲解与单变量实验见 [从一次参数更新开始](../docs/tutorials/01-first-parameter-update.md)，
连续学习路线见 [实践系列目录](../docs/tutorials/README.md)。

第五至七篇共用临时实验，无需下载数据或 GPU：

```bash
python scripts/pretrain_experiment.py --mode train
python scripts/pretrain_experiment.py --mode evaluate
python scripts/pretrain_experiment.py --mode resume
```

每个模式独立准备模板双语数据和微型模型，退出时自动清理全部实验产物。
`resume` 只在临时 run 中注入受控异常，不影响已有训练。
实验用于机制检查，不代表真实双语模型能力或完整 CUDA 验证。

`eval_pretrain.py` 默认使用训练配置绑定的数据并把报告写回 run。跨语料诊断时，
同时传入 `--data-manifest`、`--packed-manifest`；可用 `--eval-batches` 扩大
固定评测窗口。checkpoint 仍校验原训练配置和 Tokenizer，覆盖评测数据不会改写
run 内的冻结报告。

## 60M 基线入口

`verify_reference.py` 的三种模式互斥，必须选择其中一种：

| 参数 | 检查范围 |
| --- | --- |
| `--inputs-only` | 配置、有效默认值、源码、真实输入文件与预算；不代表可开始训练 |
| `--preflight` | 输入检查加 Linux/CUDA/BF16、GPU 显存、Git 和固定依赖版本 |
| `--run <路径>` | 已完成 run 的预算、产物、环境记录、最终双语改善与 dev/test 报告 |

正式基线训练使用 `train_pretrain.py --reference-spec <规范路径>`，其 `main()` 中直接
先做门控，再调用训练流程。门控失败不会创建或恢复 run；普通教学训练不强制该参数。
完整命令见 [训练文档](../docs/NATIVE_PRETRAIN_GUIDE.md#92-固定-60m-基线)。

第一次完整 60M CUDA 运行的指标、曲线和验收边界见
[60M CUDA Reference v1](../docs/experiments/native-60m-baseline-v1.md)。
该 run 的 dirty-Git provenance 失败被显式记录并作为一次性例外接受。

## 模型发布入口

`build_model_release.py` 从单个最终 checkpoint 归档构建不可覆盖的发布目录。
它只接受预期的模型、checkpoint 元数据和 trainer state，拒绝危险归档路径或额外文件；
随后绑定 Tokenizer、复制 Reference 证据、实际加载权重并生成发布报告。

```bash
python scripts/build_model_release.py \
  --checkpoint-archive step-00005649-model.tar.gz \
  --repository-id llmlifecyclelab/native-60m-base-v1
```

发布后使用 `run_published_model.py` 在新目录回下载。没有 ModelScope 依赖时先安装
`requirements-modelscope.txt`；本地发布包验证使用 `--use-local`，不访问网络。
完整流程见 [Native 模型发布指南](../docs/MODEL_RELEASE_GUIDE.md)。

## 能力评测与后训练

`evaluate_capabilities.py run` 直接读取本地 checkpoint 或发布包，
生成带显式基线、语言分层和逐题结果的 `report.json/md`。
`compare` 仅接受同一协议和分母的成绩单，不把缺失检查当成零分。
完整命令见[统一能力评测](../docs/CAPABILITY_EVALUATION_GUIDE.md)。

`data.py posttrain-recipes` 查看固定公开后训练配方，
`fetch-posttrain` 一次物化 SFT、DPO、GRPO 三份规范化 source。命令必须显式接受
Apache-2.0 与 CC-BY-4.0，随后每个阶段独立执行 `prepare --group-by source_id`。
`check-posttrain` 在 CPU 上加载全部 split，并检查评测泄漏与跨阶段 source_id 交集。
完整 hash、规模和 CUDA 顺序见
[公开数据后训练](../docs/PUBLIC_POSTTRAINING_GUIDE.md)。

`train_sft.py` 继承 Base 权重与 Tokenizer，但重置新阶段的训练状态；
`--resume-run` 才恢复同一次 SFT 的完整状态。
assistant-only 掩码在数据层构造，训练循环复用现有 `TrainingEngine`。

```bash
python scripts/sft_experiment.py --mode train
python scripts/sft_experiment.py --mode resume
```

实验自动创建并清理微型数据，不下载语料、不修改已有 run，不证明语言能力。
SFT 数据契约和正式数据接入见 [Native SFT 最小闭环](../docs/NATIVE_SFT_GUIDE.md)。

## HF、Transfer 与 DPO

安装可选依赖：

```bash
python -m pip install -r requirements-hf.txt
```

`export_hf.py` 将 Native 权重映射到标准 Llama/Qwen3 类，并强制检查
Tokenizer、logits、KV Cache 与 greedy 生成一致性。`chat.py` 读取 Native
或 HF 后端，Base 默认续写，SFT/DPO 才默认对话。

`prepare_transfer.py` 只接受固定的 Qwen commit SHA；`train_transfer.py`
从 hash 校验后的本地快照运行 Full/LoRA SFT。无需下载即可先检查机制：

```bash
python scripts/transfer_experiment.py --mode resume
```

Native DPO 复用同一训练引擎，但按偏好对而非回答 token 归一化：

```bash
python scripts/dpo_experiment.py --mode resume
```

两个实验均使用随机/合成微型数据，只证明功能和恢复，不代表模型能力。
详见 [HF 导出与 Qwen 迁移](../docs/TRANSFER_GUIDE.md)及
[Native DPO 最小闭环](../docs/NATIVE_DPO_GUIDE.md)。

GRPO 只接受显式的 exact、integer、JSON verifier：

```bash
python scripts/grpo_experiment.py --mode resume
```

该 fixture 会根据初始随机模型的采样构造可区分奖励，只验证算法链路。
公式、数据边界和恢复约束见
[Native GRPO / RLVR 最小闭环](../docs/NATIVE_GRPO_GUIDE.md)。

## CPU 推理与量化

`benchmark_inference.py` 只接受 hash 校验通过的独立 HF 模型，在固定中英文提示上
记录 FP32 与 dynamic INT8 的状态大小、token/s、进程 RSS 和 greedy 序列一致性。
输出目录不可覆盖：

```bash
python scripts/benchmark_inference.py \
  --model build/hf/native-60m-base-v1 \
  --output runs/benchmarks/native-60m-cpu-rerun \
  --max-new-tokens 16 \
  --repeats 3 \
  --threads 1
```

INT8 状态更小不代表在当前 CPU 上更快。指标口径和真实 60M 结果见
[HF 模型 CPU 推理与量化](../docs/CPU_INFERENCE_GUIDE.md)。

## 本地 HTTP 服务

`serve_openai.py` 在加载并校验模型后提供 `/v1/models`、`/v1/completions`
和 `/v1/chat/completions`，同时在 `/` 提供同源本地聊天界面：

```bash
python scripts/serve_openai.py \
  --backend hf \
  --checkpoint build/hf/native-60m-base-v1 \
  --model-id native-60m-base-v1
```

当前只支持非流式、`n=1` 的最小参数集，生成在单模型锁内串行执行。
接口、请求示例和生产边界见
[最小 OpenAI-compatible API](../docs/OPENAI_API_GUIDE.md)。
