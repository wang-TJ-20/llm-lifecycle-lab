# A10 SFT 准备与手动训练

本阶段只继承已验收的 A10 Base 权重，不继承优化器、Scheduler 或训练 step。
输入已物化并冻结；**SFT smoke、pilot、恢复验证和正式训练尚未启动，由用户执行。**
本次首轮正式预算经确认采用 1M assistant 监督 token。

## 1. 冻结边界

| 项目 | 当前值 |
| --- | --- |
| Base 来源 | `native-60m-a10-base-v1-s42`，step 45,191，clean source |
| 权重 SHA-256 | `0c3290d6c76bbf775aba20c1d543a062e26aafb987a7d15bde1b652e4dbbb46c` |
| Tokenizer SHA-256 | `7455ac1168f36a3d7dca03fee28784f97899fe233f496484a11e5da2e44a2f32` |
| 初始化包 | `models/a10-base-v1-initialization/`，从归档重建，Git 忽略 |
| Base 门禁 | `configs/gates/native-60m-a10-sft-v1-base-init.yaml` |
| 数据门禁 | `configs/gates/native-60m-a10-sft-v1-data.yaml` |
| 正式 run 门禁 | `configs/gates/native-60m-a10-sft-v1-run.yaml` |
| 公开源计划 / 实际冻结配方 | `configs/data/a10-sft-public-plan.yaml` / `sft-public-a10-v1.yaml` |
| 原文 SHA-256 | `be5fcceb3d01c6015686bdcb85c2ae14a5380cf75e3e40326c4d8737080170aa` |

初始化包保存模型、Tokenizer、保守来源许可声明、provenance、Base 证据与校验和。
初始化门禁固定真实来源状态：clean 来源不伪造 dirty 偏离记录；dirty 来源必须有显式记录。
源代码的 Apache-2.0 许可不自动适用于模型权重；发布 SFT 权重还需整理数据归属和模型卡。

## 2. 数据和预算

来源为 OASST1、Dolly、HC3-Chinese 人类回答、SQuAD、CMRC2018、MSVAMP。
固定每个来源的 commit、文件 hash 和转换方法，不下载 HelpSteer3 偏好数据。
许可为 Apache-2.0、CC-BY-SA-3.0、CC-BY-SA-4.0，已由用户确认。
结构化 JSON 样本来自公开标注答案的确定性格式转换，不是模型生成回答。

共 14,800 条候选，NFKC/空白归一化后删除 1 条重复对话，保留 14,799 条。
按 `source_id`、seed 42、0.8/0.1/0.1 做确定性分组切分：

| 划分 | 对话数 | en / zh | assistant 监督 token |
| --- | ---: | ---: | ---: |
| train | 11,862 | 5,969 / 5,893 | 471,098 |
| dev | 1,434 | 692 / 742 | 57,014 |
| test | 1,503 | 738 / 765 | 60,653 |

继承 Base 的 512 长度上限，超长或协议错误样本拒绝，不进行隐式截断。
监督只覆盖 assistant 内容和结束标记。全量扫描三份划分做 schema、去重和泄漏检查，
准备阶段没有执行 test 模型评测，也不据 test 分数选择训练参数。
MSVAMP 的上游文件名含 `test`，不等于本项目的 sealed test。

采样以 assistant token 配额平衡中英文，各占 50%，不是按对话条数平衡。
当前存在 9 个任务与语言组合，中文没有分类样本：每个英文组合权重 0.1，
每个中文组合权重 0.125，不虚构 `classification.zh`。
极短数值答案会重复采样较多，因此 **1M 不等于均匀遍历训练集 2.1 次**。

正式配置采用单卡 NVIDIA A10、CUDA BF16、micro-batch 1、累积 16、
学习率 `1e-4`、weight decay `0.01`、warmup 50。
确定性采样预算解析为 **9,379 optimizer steps**，预测处理 **1,000,421** 个监督 token，
名义预算覆盖率约 100.0421%；接受区间为 [0.99, 1.01]。
这只是不运行模型的预算计算，不是实测训练结果。

## 3. 已有本地数据时

下面的命令仅校验，不训练：

```bash
.venv-a10/bin/python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-a10-sft-v1-base-init.yaml --inputs-only
.venv-a10/bin/python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-a10-sft-v1-data.yaml --inputs-only
```

先检查并提交本次代码、配置、文档和测试，启动入口会拒绝 dirty-tree。
本轮没有自动暂存、提交或上传：

```bash
git status --short
git add .gitignore README.md configs docs scripts src tests
git diff --cached --stat
git commit -m "Prepare gated A10 SFT inputs and manual launchers"
```

## 4. 用户手动启动

准备阶段当前执行环境使用 `torch 2.7.1+cu126`，但 CUDA 可见设备数为 0，
因此硬件 preflight 尚未通过。先进入已绑定单张 A10 的 GPU 终端；
仅安装 CUDA 版 PyTorch 不会自动分配或绑定 GPU。
可先检查 `nvidia-smi` 和 `torch.cuda.is_available()`。

所有命令在仓库根目录执行。先检查环境和门禁，不创建训练 run：

```bash
.venv-a10/bin/python scripts/train_a10_sft.py --phase smoke --check-only
```

再按顺序执行，观察 loss、梯度、GPU 内存、速度和 checkpoint：

```bash
.venv-a10/bin/python scripts/train_a10_sft.py \
  --phase smoke --run-id native-60m-a10-sft-v1-smoke-s42
.venv-a10/bin/python scripts/a10_sft_resume_experiment.py
.venv-a10/bin/python scripts/train_a10_sft.py \
  --phase pilot100 --run-id native-60m-a10-sft-v1-pilot100-s42
```

smoke 为 8 steps，pilot 为 100 steps。恢复实验在 step 5 受控中断，从 step 4 恢复，
与已完成的 smoke 比较最终权重、优化器/Scheduler/RNG、TrainerState 和数据流。
两条路径都启用确定性 CUDA；不宣称跨设备逐位等价。

检查上述结果后，启动正式训练。入口会检查 smoke、pilot 和精确恢复证据，
但自动检查不能替代对速度、内存或生成效果的人工判断：

```bash
.venv-a10/bin/python scripts/train_a10_sft.py \
  --phase formal --run-id native-60m-a10-sft-v1-s42
```

原预算内中断可恢复；不能用恢复接口追加训练预算：

```bash
.venv-a10/bin/python scripts/train_a10_sft.py \
  --phase formal --resume-run native-60m-a10-sft-v1-s42
```

## 5. 完成后验收

正式训练中的 dev 使用全部 1,434 条对话；smoke/pilot 每次只取 16 条分层样本。
正式结束后先执行全量 dev，核对 run 门禁和同协议 Base/SFT 能力对照，
不依据 test 结果挑配置：

```bash
.venv-a10/bin/python scripts/eval_a10_sft.py --run runs/native-60m-a10-sft-v1-s42
.venv-a10/bin/python scripts/verify_sft_run.py \
  --spec configs/gates/native-60m-a10-sft-v1-run.yaml \
  --run runs/native-60m-a10-sft-v1-s42
```

dev 验收通过后，才执行一次全量 1,503 条 test：

```bash
.venv-a10/bin/python scripts/eval_a10_sft.py \
  --run runs/native-60m-a10-sft-v1-s42 --sealed-test
.venv-a10/bin/python scripts/verify_sft_run.py \
  --spec configs/gates/native-60m-a10-sft-v1-run.yaml \
  --run runs/native-60m-a10-sft-v1-s42 --sealed-test
```

入口用独占 `sealed-test-started.json` 标记和已有评测报告拒绝重复 test。
若执行中断，保留标记和日志先排障，不自动重新打开 test。
通用 `eval_sft.py` 是机制接口；正式 A10 路线使用上述受保护入口。

## 6. 在其他机器重建输入

先拉取 A10 LFS 权重，再运行只准备、不训练的脚本：

```bash
git lfs pull
.venv-a10/bin/python scripts/prepare_a10_sft.py \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-SA-3.0 \
  --accept-license CC-BY-SA-4.0
```

脚本拒绝悄悄覆盖冻结输入；核对实际内容后，使用
`configs/reference/a10-sft-{source,data}-manifest.json` 保留初始冻结的 manifest 身份，
创建时间和记录的原始源路径不改写成新机器的时间与路径。
当前机器重复运行同样只做校验，不创建模型训练。
`--refreeze-untrained` 仅用于明确调整未启动的执行预算，任何本路线 run 出现后禁止使用；
已有正式训练应建立新版本，不能修改原配置来绕过锁。
