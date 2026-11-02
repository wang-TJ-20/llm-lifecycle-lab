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
| A10 Base 开发验收 | `verify_a10_base.py` | 单卡身份、完成预算、有限指标、dev 改善与 checkpoint |
| A10 恢复对照 | `a10_resume_experiment.py` | 独立中断 run、原预算恢复、完整状态逐位比较 |
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

当前硬件实践入口为 [A10 从零重跑手册](../docs/A10_TRAINING_GUIDE.md)。
`verify_a10_base.py --phase smoke|pilot|base --run <路径>` 检查新 A10 配置和
开发结果，原子保存 `a10-development-gate.json`；不读取 test，不替代 SFT 门禁。
`a10_resume_experiment.py` 在第 5 步受控中断并恢复第 4 步 checkpoint；
严格 CUDA 逐位对照须对 baseline 和恢复同时启用手册中的确定性设置。

## SFT 入口（待冻结）

A10 Base 验收通过后，将冻结独立的 SFT Base/data/run 门禁。
`verify_sft_init.py`、`verify_sft_data.py`、`verify_sft_run.py` 的通用机制
已实现，但具体配置路径待 A10 SFT 阶段确定。
