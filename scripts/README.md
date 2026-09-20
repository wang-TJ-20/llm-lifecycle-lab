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
