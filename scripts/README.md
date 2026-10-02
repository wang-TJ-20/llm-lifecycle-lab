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
| 准备数据 | `data.py` | `src/llm_lifecycle_lab/data/` |
| 训练 Tokenizer | `train_tokenizer.py` | `src/llm_lifecycle_lab/tokenizer/native.py` |
| 检查 Tokenizer | `inspect_tokenizer.py` | `NativeTokenizer.encode/decode` |
| 理解模型规模 | `inspect_model.py` | `src/llm_lifecycle_lab/model/native/` |
| 实践一次参数更新 | `model_experiment.py` | 脚本中直接展示 forward、loss、backward、step 和 Cache 检查 |
| 离线预训练、评测与恢复对照 | `pretrain_experiment.py` | 临时双语数据与真实 `run_native_pretraining` / `evaluate_native_pretraining` |
| 训练模型 | `train_pretrain.py` | `src/llm_lifecycle_lab/training/pretrain.py` |
| 评测模型 | `eval_pretrain.py` | `evaluate_native_pretraining` |
| 冻结基线与验收 | `verify_reference.py` | `src/llm_lifecycle_lab/reference.py` |
| 校验 SFT Base 初始化 | `verify_sft_init.py` | `src/llm_lifecycle_lab/sft_initialization.py` |
| 校验 SFT 数据 | `verify_sft_data.py` | `src/llm_lifecycle_lab/sft_data_gate.py` |
| 训练/恢复 SFT | `train_sft.py` | `src/llm_lifecycle_lab/training/sft.py` |
| 全量 SFT dev/test | `eval_sft.py` | `evaluate_native_sft` |
| Base/SFT 能力对照 | `evaluate_capabilities.py` | `src/llm_lifecycle_lab/evaluation/` |
| 离线 SFT 闭环 | `sft_experiment.py` | 临时 Base → SFT → 评测 → 精确恢复 |
| 验收正式 SFT run | `verify_sft_run.py` | `src/llm_lifecycle_lab/sft_run_gate.py` |

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

## SFT Stage 0 入口

`verify_sft_init.py` 只校验 SFT 的 canonical Base 初始化边界，不执行 SFT：

```bash
python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --inputs-only
```

`--inputs-only` 检查 Base 发布包、来源证据、质量门禁、Tokenizer 契约和权重严格
加载。真正启动新 run 前改用 `--preflight`，额外要求当前 Git 工作树 clean。
完整冻结范围见 [SFT 初始化门禁](../docs/SFT_INITIALIZATION_GATE.md)。

## SFT Stage 1 入口

`verify_sft_data.py` 将数据契约冻结与真实输入准入分开：

```bash
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --contract-only
```

`--contract-only` 不要求下载数据，用于检查 Base gate、公开数据 recipe、固定版本/
文件哈希、许可、排除来源和切分策略。数据物化后改用 `--inputs-only`，逐项检查
canonical source、prepared splits、Tokenizer、对话协议、长度、重复数据与冻结的
train 统计。正式启动 run 前使用 `--preflight`，额外要求 Git 工作树 clean。

当前工作目录的真实 source 与 prepared splits 已通过 `--inputs-only`；新 clone
因为 `data/` 被 Git 忽略，仍需先执行 `data.py fetch-sft` 和 `data.py prepare`。
完整命令见 [SFT 训练手册](../docs/SFT_TRAINING_GUIDE.md)。

## SFT 训练与评测入口

`train_sft.py` 从 canonical Base 权重开始新 stage，或恢复同一 run 的模型、
optimizer、scheduler、RNG 和 quota sampler 状态。`eval_sft.py` 只允许冻结的
完整 dev/test split：

```bash
python scripts/train_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --run-id native-60m-sft-v1-s42

python scripts/eval_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --checkpoint runs/native-60m-sft-v1-s42/checkpoints/step-00014020 \
  --split dev
```

`evaluate_capabilities.py` 对 Base 与 SFT 使用同一版本化 protocol，并拒绝比较
protocol hash 不同的成绩单。`sft_experiment.py` 使用临时微型数据验证完整机制，
不读取正式数据，也不留下 run。

## SFT 完成态入口

`verify_sft_run.py --dev` 检查正式配置锁、8M token 预算、确定性 coverage、
step-0 到最终 step 的中英文 dev 改善、完整 checkpoint 和 1,459 条全量 dev。
只有该模式通过后才能运行一次 test，再使用 `--sealed-test` 验收 1,458 条报告：

```bash
python scripts/verify_sft_run.py \
  --spec configs/gates/native-60m-sft-v1-run.yaml \
  --run runs/native-60m-sft-v1-s42 \
  --dev
```

完整远端训练、恢复、能力对照、sealed test 和归档步骤只维护在
[SFT 训练手册](../docs/SFT_TRAINING_GUIDE.md)。
