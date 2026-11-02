# Pretrain 训练文档

本文负责安装环境、选择配置、启动训练、读取指标、恢复 checkpoint 和独立评测。
数据下载、切分、Tokenizer 和 Packing 命令集中在
[数据介绍与准备](./DATA_GUIDE.md)；模型结构集中在 [自有模型介绍](./NATIVE_MODEL_GUIDE.md)。
60M 正式训练的实际执行命令、预算和门禁以
[A10 从零重跑手册](./A10_TRAINING_GUIDE.md)为准。

初次实践顺序：本文第 2 节安装环境 -> A10 手册完成数据准备 -> 本文第 3～5 节
运行 A10 smoke 与开发评测。smoke、pilot 通过后，再进入正式 Base。

## 1. 当前能力与目录

已实现 Native-60M、BPE、Pretrain、checkpoint 与 dev/test 评测。
SFT 通用训练和门禁机制已实现，A10 SFT 的配置与门禁尚待独立冻结；
DPO、GRPO、Qwen 迁移、HF 导出和服务尚未实现。

| 目录 | 职责 | 是否应保留 |
| --- | --- | --- |
| `src/llm_lifecycle_lab/` | 模型、数据、训练、校验的实际实现 | 必须 |
| `scripts/` | 可直接阅读和运行的参数、流程与结果输出 | 必须 |
| `configs/models/` | 模型结构配置 | 必须 |
| `configs/pipelines/` | 数据、模型、预算、优化器等运行配置 | 必须 |
| `configs/reference/` | 机器可读 Reference 验收规范 | 使用 Reference 时必须 |
| `tests/`、`.github/` | 回归测试、GitHub CI | 开发时保留 |
| `data/` | 原始/切分文本、Tokenizer、packed 数据 | 当前训练依赖，Git 忽略 |
| `runs/` | checkpoint、配置快照和指标 | 实验记录，应归档 |
| `.venv/` | 可选的 uv Python 环境 | 正在使用时保留 |

文档统一使用 `python scripts/...`。每个脚本会从仓库位置自动加载 `src/` 下的共享实现，
无需把项目安装进 Python 环境；它们使用同一套参数解析器、YAML、hash 和训练逻辑。

## 2. 环境准备

### 2.1 进入仓库

新机器上：

```bash
git clone https://github.com/wang-TJ-20/llm-lifecycle-lab.git
cd llm-lifecycle-lab
```

已有仓库不需要重新 clone，进入该仓库根目录即可。后面的相对路径都基于根目录，
不是基于 `scripts/` 或 YAML 所在目录。

### 2.2 创建 Conda 环境

```bash
conda create -n llm-lifecycle-lab python=3.11 pip -y
conda activate llm-lifecycle-lab
python --version
python -c "import sys; print(sys.executable)"
```

Python 应为 3.11 或更高，解释器路径应位于该 Conda 环境内。环境已存在时只执行
`conda activate`。如果 shell 未初始化 Conda，可运行 `conda init zsh`（Bash 用
`conda init bash`），重开终端后再激活。

不要在已激活 uv `.venv` 的 shell 中混装另一个环境。Conda 负责 Python/依赖隔离，
不自动提供 NVIDIA 驱动，也不替代项目的数据与 checkpoint 校验。

### 2.3 按设备选择 PyTorch

| 平台 | 选择 |
| --- | --- |
| macOS Apple Silicon | 可运行不依赖数据的机制实验，不适用 A10 Pipeline |
| Linux 仅 CPU | 先按下方命令安装 CPU wheel，适合机制实验，不适用 A10 Pipeline |
| Linux NVIDIA A10 | 先检查 `nvidia-smi`，按 A10 手册安装固定 CUDA 依赖 |

Linux CPU：

```bash
python -m pip install 'torch>=2.4,<3' \
  --index-url https://download.pytorch.org/whl/cpu
```

Linux CUDA：先执行 `nvidia-smi`，再从
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 获取对应 CUDA 的 Pip
安装命令。项目接受 PyTorch `>=2.4,<3`；不需要 torchvision/torchaudio，也不要仅根据
Conda 中是否安装 CUDA 包来判断驱动可用。

### 2.4 安装依赖并验证

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/data.py recipes
python scripts/doctor.py
```

`requirements.txt` 只安装训练和公开数据所需的第三方依赖，并固定
`tokenizers==0.21.4`。预先安装且满足版本范围的 PyTorch 不会被主动升级。
各 `scripts/*.py` 入口会根据自身位置加载 `src/`，无需 editable 安装或手动修改
`PYTHONPATH`。

**检查结果**：`pip check` 无依赖冲突；recipes 命令正常；基础 Doctor 无 FAIL。
基础 Doctor 不检查所有训练依赖和数据，开始训练前还必须运行带 `--config` 的 Doctor。

GPU 信息检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16:", torch.cuda.is_bf16_supported())
PY
```

60M 默认使用 `cuda + bfloat16`，应确认 CUDA/BF16 可用。macOS 上 CUDA=False 是正常的，
不能据此把 60M 配方冒充为已完成 CUDA 验证。

### 2.5 锁环境与开发工具

Conda + requirements 是兼容范围安装，不是完整传递依赖锁。严格复现时可使用：

```bash
uv sync --locked --no-install-project --extra training --extra public-data --extra dev
uv run --no-sync python scripts/doctor.py
```

选择 uv 后，后续 `python ...` 均相应写为 `uv run --no-sync python ...`，避免调用
另一套解释器或自动安装当前项目。
也可用 `uv export --frozen` 导出锁依赖后安装到独立 Conda 环境，但其 CUDA 依赖仍需驱动兼容。
同 seed 不保证不同 CPU/GPU、PyTorch 版本间结果逐位一致，训练会记录实际环境。

开发测试使用：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

## 3. 训练前检查产物与配置

### 3.1 数据准备到哪一步

首次运行请完整执行 [A10 数据准备](./A10_TRAINING_GUIDE.md#数据)。
已有数据时不要重复 fetch/prepare/tokenize/pack，先核对下列文件：

```text
data/prepared/bilingual-60m-a10-v1/data_manifest.json
data/prepared/bilingual-60m-a10-v1/{train,dev,test}.jsonl
data/tokenizers/bilingual-60m-a10-v1/tokenizer.json
data/tokenizers/bilingual-60m-a10-v1/tokenizer_manifest.json
data/packed/bilingual-60m-a10-v1-seq512/packed_manifest.json
data/packed/bilingual-60m-a10-v1-seq512/各 split 的三个二进制数组
```

smoke、pilot 和正式 Base 使用同一套 A10 数据与 Tokenizer。
数据配套关系和迁移说明见 [数据产物复用](./DATA_GUIDE.md#8-产物校验复用与迁移)。

### 3.2 选择 Pipeline

| Pipeline | 模型/设备 | 预算 | 每次 eval 上限 |
| --- | --- | --- | ---: |
| `native-60m-a10-base-smoke.yaml` | 60M，A10 CUDA BF16 | 8 optimizer steps | 16 batches |
| `native-60m-a10-base-pilot100.yaml` | 60M，A10 CUDA BF16 | 100 optimizer steps | 128 batches |
| `native-60m-a10-base-v1.yaml` | 60M，A10 CUDA BF16 | 8 epochs | 1024 batches |

这些 Pipeline 的 `model_route` 都是 `native`，模型结构均由
`configs/models/tiny-60m.yaml` 指定。`run_profile` 表示实践级别，不表示参数量。
首次选择 `configs/pipelines/native-60m-a10-base-smoke.yaml`。
不要修改已经运行过的配置或 checkpoint，应使用新配置和新 run ID 开始实验。

### 3.3 先验证，再做真实 batch 检查

```bash
python scripts/validate_config.py configs/pipelines/native-60m-a10-base-smoke.yaml
python scripts/doctor.py --config configs/pipelines/native-60m-a10-base-smoke.yaml
```

`validate_config` 只做配置级校验；Doctor 进一步检查：

1. Python、依赖、设备、可写输出路径和磁盘空间。
2. Pipeline 与模型 `provider/model_id/model_route` 等元数据，以及 `run_profile`
   对应的设备和资源要求。
3. Data Manifest、split 内容 hash 与 dev split。
4. Tokenizer 内容/来源 hash、实际词表和模型词表一致性。
5. Packed 数组 hash 以及数据、Tokenizer、序列长度三者绑定。
6. 一个真实 batch 的前向、loss、反向和梯度有限性。

**检查结果**：应有 `real-batch ... passed`，最后 `0 failed`。任何 FAIL 都先解决。
`WARN` 是需理解的非阻断风险，不等于已完成相应能力验证。Doctor 不进行持久训练或保存 checkpoint。

## 4. 运行 A10 Smoke

### 4.1 启动

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-smoke.yaml \
  --run-id native-60m-a10-tutorial-smoke-001
```

`run-id` 对应 `runs/native-60m-a10-tutorial-smoke-001/`，不能覆盖已有目录。
重复实践请使用新 ID；省略该参数会由项目生成 ID，以终端输出为准。

终端应依次出现：

```text
plan mode=max_steps max_steps=8 ...
baseline step=0 eval_loss=...
step=1 train_loss=... eval_loss=...
...
step=8 train_loss=... eval_loss=...
Completed pretraining run native-60m-a10-tutorial-smoke-001
```

这些只是输出结构，loss 数值以实际运行为准。16K 词表的随机初始化 loss 通常接近
`ln(16384) = 9.70`，smoke 之后不承诺有语言能力，也不要求每步 loss 单调下降。

### 4.2 检查完成状态

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("runs/native-60m-a10-tutorial-smoke-001")
manifest = json.loads((run / "run_manifest.json").read_text())
result = json.loads((run / "training_result.json").read_text())
print("status:", manifest["status"])
print("steps:", result["global_step"])
print("tokens:", result["tokens_seen"])
print("coverage:", result["target_token_coverage"])
print("checkpoint:", result["final_checkpoint"])
assert manifest["status"] == "completed"
assert result["global_step"] == 8
assert (run / "checkpoints/step-00000008/model/model.pt").is_file()
PY
```

**通过标准**：status=completed，global_step=8，loss/梯度有限，baseline/训练后评测存在，
最终 checkpoint 可加载。它验证的是工程闭环，不是模型质量。

## 5. 独立评测与指标解读

### 5.1 使用同一份配置评测

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-smoke.yaml \
  --checkpoint runs/native-60m-a10-tutorial-smoke-001/checkpoints/step-00000008 \
  --split dev

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-smoke.yaml \
  --checkpoint runs/native-60m-a10-tutorial-smoke-001/checkpoints/step-00000008 \
  --split test \
  --json
```

checkpoint 参数指向 `step-.../`，不是内部 `model.pt`。命令检查 config hash、
路线、stage 和 Tokenizer hash；不能临时换评测配置的设备、batch 或样本数来绕过检查。

报告写入：

```text
runs/native-60m-a10-tutorial-smoke-001/evaluations/pretrain-dev-step-00000008.json
runs/native-60m-a10-tutorial-smoke-001/evaluations/pretrain-test-step-00000008.json
```

**重要范围**：当前 eval 受 `training.eval_batches × micro_batch_size` 限制，
在 split 中按主导语言确定性轮转选取 packed 样本，不是默认全量评测。中英文 loss 在这些
相同样本内按 label 的语言计算，不会重新排列上下文。缺少有效语言标签或样本太少时，
不能保证同时出现两种语言指标。报告的 `sample_count` 和 `eval_tokens` 才是实际覆盖量。

若需扩大覆盖，应在新训练开始前设置更大的 `eval_batches`。当前 CLI 没有独立的评测
预算覆盖参数，不应修改已完成 run 的配置后声称仍属同一实验。

### 5.2 查看完整曲线

```bash
python - <<'PY'
import json
from pathlib import Path

for line in Path("runs/native-60m-a10-tutorial-smoke-001/metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    print(row["step"], row.get("event", "train"),
          row.get("train_loss"), row.get("eval_loss"),
          row.get("eval_en_loss"), row.get("eval_zh_loss"))
PY
```

| 指标 | 应怎样理解 |
| --- | --- |
| `train_loss` | 当前 optimizer step 的 token 加权日志值，不是整个 epoch 平均 |
| `eval_loss` | 固定抽取的 dev 样本上的因果 LM loss |
| `eval_perplexity` | `exp(min(loss, 20))`；指数被截断，loss 本身不截断 |
| `train_bits_per_byte` / `eval_bits_per_byte` | NLL 对规范化文本 UTF-8 字节权重归一化 |
| `eval_en_*` / `eval_zh_*` | 两个语言桶的 loss、perplexity、token 数和 bits-per-byte |
| `learning_rate` / `gradient_norm` | 该 step 的学习率与 clip 前梯度范数 |
| `tokens_per_second` | 当前训练 step 的监督 token 吞吐，不是包含准备/评测的端到端吞吐 |
| `tokens_seen` / `epochs_seen` | 累计监督 token，以及相对 train 总监督 token 的轮数 |
| `target_token_coverage` | 实际监督 token / 计划 token，可略高于 1 |
| `cuda_max_memory_allocated_bytes` | CUDA 已分配显存峰值，不等于 `nvidia-smi` 显示的全部占用 |

应同时比较 step 0 baseline、后续 dev、英文和中文结果。只有 train loss 下降不足以证明改善。
当所有监督 token 都有 en/zh 标签时，aggregate loss 可由两桶 token 加权还原；
float32 后端允许微小误差。存在未标记 token 时不能只用两桶还原总体。

## 6. 60M 正式实践

### 6.1 检查硬件与数据

先完成 [A10 数据与开发门禁](./A10_TRAINING_GUIDE.md)，再在干净提交上执行：

```bash
python scripts/validate_config.py configs/pipelines/native-60m-a10-base-v1.yaml
python scripts/doctor.py --profile tiny-60m-a10 \
  --config configs/pipelines/native-60m-a10-base-v1.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --run-id native-60m-a10-base-new-s42
```

默认 `micro_batch_size=1`、梯度累积 16、seq512、BF16。
有效 batch 为 16 个序列，每个完整序列提供 511 个监督 token。

```text
train 监督 token:        46,184,530
train packed 样本:       90,381
每 epoch micro batches: 90,381
8 epochs optimizer steps: ceil(90,381 × 8 / 16) = 45,191
最大监督 token/step:     16 × 511 = 8,176
```

最后一个 step 保持完整累积，可能进入下一轮少量数据，预算是估算而非硬截断 token 上限。
以 `training_budget.json` 和 `training_result.json` 的计划/实际覆盖为准。
本轮 8 epochs 约处理 369.5M 监督 token，仍是有限的预训练预算，不代表通用助手能力。

### 6.2 训练后的评测

本轮正式预算的最终 checkpoint 为 step 45,191。先验收开发指标，通过后只执行一次 test：

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --checkpoint runs/native-60m-a10-base-new-s42/checkpoints/step-00045191 \
  --split dev

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --checkpoint runs/native-60m-a10-base-new-s42/checkpoints/step-00045191 \
  --split test
```

改变预算后不得照抄上述 step，以训练输出的 `final_checkpoint` 为准。
已有 A10 归档的 test 已执行，不应重复评测或将新实验结果套用到已有归档。

## 7. 自定义实验配置

### 7.1 创建独立配置

```bash
cp configs/pipelines/native-60m-a10-base-smoke.yaml configs/pipelines/custom-smoke.yaml
```

在新文件中修改对应字段，保留其余必需配置。自有 seq128 数据示例：

```yaml
model:
  provider: native
  model_id: tiny-60m
  architecture: dense-decoder
  config: configs/models/tiny-60m.yaml
  tokenizer: data/tokenizers/custom-v1
data:
  manifest: data/prepared/custom-v1/data_manifest.json
  packed_manifest: data/packed/custom-v1-seq128/packed_manifest.json
training:
  sequence_length: 128
```

这是替换 `model/data` 两段并修改 `training.sequence_length`，不是完整 Pipeline。
保留其他训练字段。假设 Tokenizer 实际词表为 16,384；
其他词表须同步创建匹配的模型配置和 ID。不要在原 run 开始后编辑其配置。

### 7.2 常用训练字段

| 字段 | 含义与约束 |
| --- | --- |
| `device` | `auto/cpu/mps/cuda`；auto 顺序为 CUDA -> MPS -> CPU |
| `dtype` | `float32/bfloat16/float16`；float16 仅 CUDA，当前恢复有已知限制 |
| `sequence_length` | 必须与 packed 一致，且不大于模型最大长度 |
| `micro_batch_size` | 单次 forward 样本数，影响显存 |
| `gradient_accumulation_steps` | 每个 optimizer step 的 micro batch 数 |
| `learning_rate` / `weight_decay` | AdamW 参数；一维 norm 参数不做 weight decay |
| `warmup_steps` / `min_lr_ratio` | warmup 后 cosine 调度；warmup 必须小于解析后的总 steps |
| `gradient_clipping` | 全局梯度范数裁剪阈值 |
| `checkpoint_interval` | 按 optimizer step 保存 checkpoint，结束时必保存 |
| `eval_interval` / `eval_batches` | dev 评测频率与最多 batch 数 |
| `log_interval` | 写日志频率；建议与 eval 周期对齐，避免评测结果未写入该步日志 |

预算只能三选一，替换时删除旧字段，而不是叠加：

```yaml
# 固定 optimizer steps
max_steps: 100
```

```yaml
# 监督 token 目标，折算为完整 optimizer steps，实际可略超出
max_train_tokens: 1000000
```

```yaml
# 按 train 监督 token 数估算轮次，可为小数
num_epochs: 1.0
```

全部属于 `training:` 下的字段。改完执行：

```bash
python scripts/validate_config.py configs/pipelines/custom-smoke.yaml
python scripts/doctor.py --config configs/pipelines/custom-smoke.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/custom-smoke.yaml \
  --run-id custom-smoke-001
```

显存不足先减小 micro batch，必要时提高累积步数。默认 60M 的 micro batch 已为 1；
若还不足可在新配置中降低序列长度，但必须重新 pack 到新目录并更新路径。
梯度累积不会消除单个样本的显存需求。改变这些条件后要重新确认预算和可比性。

## 8. Checkpoint 与中断恢复

### 8.1 保存哪些内容

```text
runs/<run_id>/
  resolved_config.yaml
  run_manifest.json
  data_snapshot.json
  packed_data_snapshot.json
  training_budget.json
  runtime_environment.json
  model_config.json
  model_manifest.json
  tokenizer_manifest.json
  tokenizer/tokenizer.json
  metrics.jsonl
  training_result.json
  latest_checkpoint.json
  checkpoints/step-XXXXXXXX/
    checkpoint_metadata.json
    trainer_state.json
    optimizer_state.pt
    model/config.json
    model/model.pt
  evaluations/
```

模型权重、optimizer、scheduler、Torch CPU/CUDA RNG、step/token 计数及数据流 epoch/offset
一并保存。`training_result.json` 在成功完成后才写入；普通训练异常通常记录 `failure.json`，
但进程强杀等情况不保证能写失败文件。完整 checkpoint 采用目录原子发布，不覆盖旧记录。

注意 run 内的 `tokenizer_manifest.json` 在根目录，不在 `tokenizer/` 目录内；
不要直接把 `runs/<id>/tokenizer` 当成标准 `NativeTokenizer.from_directory` 的目录。

### 8.2 恢复未完成的 run

仅当已经有完整 checkpoint，且其 step 小于原计划上限时：

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --resume-run native-60m-a10-base-new-s42
```

默认读取 `latest_checkpoint.json`。明确选择 checkpoint 时：

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --resume-run native-60m-a10-base-new-s42 \
  --resume-checkpoint checkpoints/step-00005000
```

第二条只适用于 step 5,000 之后不存在会被覆盖的后续 checkpoint 的情况。
`--resume-checkpoint` 必须配合 `--resume-run`；`--run-id` 与 `--resume-run` 互斥。

恢复前检查：

1. 原配置 hash、模型、Tokenizer 和数据配套不变。
2. checkpoint 目录完整，确实属于该 run。
3. 设备与依赖保持一致，尤其不要临时修改 `dtype/device`。
4. 恢复目标未到预算上限，不会覆盖已有后续 checkpoint。

**已经完成的 Smoke 不能再 resume 追加训练**。当前不支持在同一 run 延长预算，
也没有从旧 checkpoint 分叉的新 run 命令。新 ID 默认从随机初始化开始，不等于续训。
改变预算、数据或模型时应明确规划新的实验。

Checkpoint 已保存 FP16 GradScaler 和 MPS RNG；精确恢复等价测试目前仍只覆盖
CPU 路径，不能据此承诺全部设备和精度逐位一致。

## 9. 60M 正式训练

60M 正式训练在单卡 A10 上执行，使用
[`native-60m-a10-base-v1.yaml`](../configs/pipelines/native-60m-a10-base-v1.yaml)。
完整执行顺序、预算、门禁和验收要求见
[A10 从零重跑手册](./A10_TRAINING_GUIDE.md)，实际进度见
[A10 执行记录](./A10_TRAINING_RESULTS.md)。

正式训练前需完成的门控：

1. **Doctor 预检**：使用 `tiny-60m-a10` profile 检查单卡 A10、CUDA BF16、至少 21.9 GiB 可见显存。
2. **干净工作树**：正式 Base 必须在无未提交变更的干净提交上执行。
3. **数据与 Tokenizer**：使用独立的 `bilingual-60m-a10-v1` 数据和 Tokenizer。
4. **开发门禁**：smoke 和 pilot 通过后才能启动正式 Base。

训练完成后验收要求：

- token 覆盖率在 `[1.0, 1.01]` 范围内
- 总体 dev loss 相对 baseline 减半
- 中英文 dev loss 均改善
- final dev loss 不超过 best 的 1.02 倍
- 最终 checkpoint 重载 dev 评测与训练记录一致

## 10. 故障定位

| 现象 | 处理顺序 |
| --- | --- |
| `No module named ...` | 检查 `sys.executable`、Conda 激活、第三方依赖安装及仓库 `src/` 是否完整 |
| `config file does not exist` | 确认从仓库根目录运行，并检查配置路径 |
| `different Data Manifest` / packed 不兼容 | 对照数据文档确认配套关系，禁止手改 hash |
| Tokenizer/model 词表不一致 | 读取实际词表，修正独立模型配置或重新训练 Tokenizer |
| CUDA unavailable / BF16 不支持 | 检查驱动、PyTorch wheel 与硬件，不将 60M 当 CPU Smoke |
| CUDA OOM | 调小 micro batch；若已是 1，考虑新序列长度并重新 pack |
| run/checkpoint 已存在 | 新实验换 run ID；未完成 run 用 resume，不删除历史来掩盖冲突 |
| `resume config does not match` | 使用原 resolved config；恢复不允许改变原预算 |
| `already reached or exceeded max_steps` | checkpoint 已完成原预算，不能继续追加 |
| loss/gradient non-finite | 保留失败产物，检查数据、学习率、精度；不要忽略错误继续运行 |
| 首条训练日志等待较久 | 会先验证 hash 并遍历评测样本，当前尚未优化这部分启动开销 |
| 只出现一种语言 eval | 检查语言标签和评测覆盖；不要用 aggregate 代替双语验证 |
| GitHub loss 最末位不同 | 后端 float32 累加有误差；分桶一致性使用容差而非逐位相等 |

## 11. 保留边界与后续完善

目前仅单进程单设备，不支持 DDP、gradient checkpointing 和 `torch.compile`。
不要直接用 `torchrun` 让多个进程写同一 run。历史 checkpoint 与实验结果不应当作普通缓存删除。

本地可再生成的 `dist/`、`build/`、`.pytest_cache/`、`.ruff_cache/` 和源码
`__pycache__/` 可在无任务运行时清理；`data/`、`.venv/`、`runs/`、`.git/` 不作默认清理对象。

对照 MiniMind 的
[预训练实现](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/train_pretrain.py)
和 [恢复工具](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/trainer_utils.py)，
后续优先顺序为：

1. 已保存并恢复 FP16 GradScaler/MPS RNG；补齐 CUDA/MPS 连续训练与中断恢复对照，
   不把现有 CPU 精确恢复测试扩大为所有设备的保证。
2. 当前梯度累积已按有效监督 token 总数归一化，与 token 加权日志一致；
   在真实 A10 batch、padding 和尾批下继续验证数值与吞吐。
3. 分离稳定数据内容身份与时间/路径来源，完善跨机器复现。
4. 独立评估 SDPA 快速路径、预取、compile 和可视化。

MiniMind 的 scaler、workers、pin memory 与可选 SwanLab 可供借鉴，但不能因此移除
本项目现有 hash、baseline、双语评测和不可覆盖 checkpoint，也不能把未验证优化计为收益。
