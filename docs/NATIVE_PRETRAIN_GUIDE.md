# Pretrain 训练文档

本文负责安装环境、选择配置、启动训练、读取指标、恢复 checkpoint、独立评测和
60M Reference 验收。数据下载、切分、Tokenizer 和 Packing 命令集中在
[数据介绍与准备](./DATA_GUIDE.md)；模型结构集中在 [自有模型介绍](./NATIVE_MODEL_GUIDE.md)。

初次实践顺序：本文第 2 节安装环境 -> 数据文档完成 Smoke 数据准备 -> 本文第 3～5 节
训练与评测。只有通过 Smoke 后，再进入 60M；不需要一次运行全部消融配置。

## 1. 当前能力与目录

已实现 Native 10M/60M、BPE、Pretrain、checkpoint 与 dev/test 评测。
SFT、DPO、GRPO、Qwen 迁移、HF 导出和服务尚未实现。

| 目录 | 职责 | 是否应保留 |
| --- | --- | --- |
| `src/llm_lifecycle_lab/` | 模型、数据、训练、校验的实际实现 | 必须 |
| `scripts/` | 在当前 Python 进程中调用共享实现的操作入口 | 必须 |
| `configs/models/` | 模型结构配置 | 必须 |
| `configs/pipelines/` | 数据、模型、预算、优化器等运行配置 | 必须 |
| `configs/reference/` | 机器可读 Reference 验收规范 | 使用 Reference 时必须 |
| `tests/`、`.github/` | 回归测试、GitHub CI | 开发时保留 |
| `data/` | 原始/切分文本、Tokenizer、packed 数据 | 当前训练依赖，Git 忽略 |
| `runs/` | checkpoint、配置快照和指标 | 实验记录，应归档 |
| `.venv/` | 可选的 uv Python 环境 | 正在使用时保留 |

`llmlab` 与 `python -m llm_lifecycle_lab` 是兼容入口。文档统一使用 `python scripts/...`；
它们使用同一套参数解析器、YAML、hash 和训练逻辑，不通过 shell 调用另一个 CLI。

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
| macOS Apple Silicon | 直接执行下一节 requirements 安装，使用 macOS wheel；Smoke 自动优先 MPS |
| Linux 仅 CPU | 先按下方命令安装 CPU wheel，适合 Smoke，不适合 60M 正式运行 |
| Linux NVIDIA GPU | 先检查 `nvidia-smi`，选择与驱动兼容的 CUDA wheel；60M 目标为单张 24GB GPU |

Linux CPU：

```bash
python -m pip install 'torch>=2.4,<3' \
  --index-url https://download.pytorch.org/whl/cpu
```

Linux CUDA：先执行 `nvidia-smi`，再从
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 获取对应 CUDA 的 Pip
安装命令。项目接受 PyTorch `>=2.4,<3`；不需要 torchvision/torchaudio，也不要仅根据
Conda 中是否安装 CUDA 包来判断驱动可用。

### 2.4 安装项目并验证

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m llm_lifecycle_lab --version
python scripts/doctor.py
```

`requirements.txt` 用 `-e .[training,public-data]` 从 `pyproject.toml` 安装实际依赖，
并固定 `tokenizers==0.21.4`。预先安装且满足版本范围的 PyTorch 不会被主动升级。
editable 安装让 Python 能导入 `src/` 下的包，无需手动修改 `PYTHONPATH`。

**检查结果**：`pip check` 无依赖冲突；版本命令正常；基础 Doctor 无 FAIL。
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
uv sync --locked --extra training --extra public-data --extra dev
uv run python scripts/doctor.py
```

选择 uv 后，后续 `python ...` 均相应写为 `uv run python ...`，避免调用另一套解释器。
也可用 `uv export --frozen` 导出锁依赖后安装到独立 Conda 环境，但其 CUDA 依赖仍需驱动兼容。
同 seed 不保证不同 CPU/GPU、PyTorch 版本间结果逐位一致，训练会记录实际环境。

开发测试使用：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

## 3. 训练前检查产物与配置

### 3.1 数据准备到哪一步

首次运行请完整执行 [数据文档第 4 节](./DATA_GUIDE.md#4-第一次实践10m-双语-smoke)。
已有数据时不要重复 fetch/prepare/tokenize/pack，先核对下列文件：

```text
data/prepared/bilingual-smoke-v1/data_manifest.json
data/prepared/bilingual-smoke-v1/{train,dev,test}.jsonl
data/tokenizers/bilingual-smoke-v1/tokenizer.json
data/tokenizers/bilingual-smoke-v1/tokenizer_manifest.json
data/packed/bilingual-smoke-v1-seq128/packed_manifest.json
data/packed/bilingual-smoke-v1-seq128/各 split 的三个二进制数组
```

60M 使用各自的 `bilingual-60m-v1` 与 `bilingual-60m-v1-seq512`。
数据配套关系和迁移说明见 [数据产物复用](./DATA_GUIDE.md#8-产物校验复用与迁移)。

### 3.2 选择 Pipeline

| Pipeline | 模型/设备 | 预算 | 每次 eval 上限 |
| --- | --- | --- | ---: |
| `native-smoke.yaml` | 10M，自动选择 CUDA/MPS/CPU | 2 optimizer steps | 2 batches |
| `native-v1.yaml` | 60M，CUDA BF16 | 1 epoch | 8 batches |
| `native-60m-reference.yaml` | 60M，CUDA BF16 | 1 epoch，更严格发布验收 | 64 batches |
| `native-60m-qk-norm.yaml` | 宽浅 + QK-Norm | 同 60M 基线 | 8 batches |
| `native-60m-deep-narrow.yaml` | 深窄，无 QK-Norm | 同 60M 基线 | 8 batches |
| `native-60m-deep-narrow-qk-norm.yaml` | 深窄 + QK-Norm | 同 60M 基线 | 8 batches |

首次选择 `configs/pipelines/native-smoke.yaml`。模型详细参数不在本节重复，见模型文档。

### 3.3 先验证，再做真实 batch 检查

```bash
python scripts/validate_config.py configs/pipelines/native-smoke.yaml
python scripts/doctor.py --config configs/pipelines/native-smoke.yaml
```

`validate_config` 只做配置级校验；Doctor 进一步检查：

1. Python、依赖、设备、可写输出路径和磁盘空间。
2. Pipeline 与模型 `provider/model_id/model_route` 等元数据。
3. Data Manifest、split 内容 hash 与 dev split。
4. Tokenizer 内容/来源 hash、实际词表和模型词表一致性。
5. Packed 数组 hash 以及数据、Tokenizer、序列长度三者绑定。
6. 一个真实 batch 的前向、loss、反向和梯度有限性。

**检查结果**：应有 `real-batch ... passed`，最后 `0 failed`。任何 FAIL 都先解决。
`WARN` 是需理解的非阻断风险，不等于已完成相应能力验证。Doctor 不进行持久训练或保存 checkpoint。

## 4. 运行两步 Smoke

### 4.1 启动

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --run-id native-smoke-001
```

`run-id` 对应 `runs/native-smoke-001/`，不能覆盖已有目录。重复实践请使用
`native-smoke-002` 等新 ID；省略该参数会由项目生成 ID，以终端输出为准。

终端应依次出现：

```text
plan mode=max_steps max_steps=2 ...
baseline step=0 eval_loss=...
step=1 train_loss=... eval_loss=...
step=2 train_loss=... eval_loss=...
Completed pretraining run native-smoke-001
```

这些只是输出结构，loss 数值以实际运行为准。16K 词表的随机初始化 loss 通常接近
`ln(16384) = 9.70`，两步之后不承诺有语言能力，也不要求每步 loss 单调下降。

### 4.2 检查完成状态

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("runs/native-smoke-001")
manifest = json.loads((run / "run_manifest.json").read_text())
result = json.loads((run / "training_result.json").read_text())
print("status:", manifest["status"])
print("steps:", result["global_step"])
print("tokens:", result["tokens_seen"])
print("coverage:", result["target_token_coverage"])
print("checkpoint:", result["final_checkpoint"])
assert manifest["status"] == "completed"
assert result["global_step"] == 2
assert (run / "checkpoints/step-00000002/model/model.pt").is_file()
PY
```

**通过标准**：status=completed，global_step=2，loss/梯度有限，baseline/训练后评测存在，
最终 checkpoint 可加载。它验证的是工程闭环，不是模型质量。

## 5. 独立评测与指标解读

### 5.1 使用同一份配置评测

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002 \
  --split dev

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-smoke.yaml \
  --checkpoint runs/native-smoke-001/checkpoints/step-00000002 \
  --split test \
  --json
```

checkpoint 参数指向 `step-.../`，不是内部 `model.pt`。命令检查 config hash、
路线、stage 和 Tokenizer hash；不能临时换评测配置的设备、batch 或样本数来绕过检查。

报告写入：

```text
runs/native-smoke-001/evaluations/pretrain-dev-step-00000002.json
runs/native-smoke-001/evaluations/pretrain-test-step-00000002.json
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

for line in Path("runs/native-smoke-001/metrics.jsonl").read_text().splitlines():
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

先完成 [60M 数据准备](./DATA_GUIDE.md#5-切换为-60m-双语数据)，再在 Linux/CUDA 环境执行：

```bash
python scripts/validate_config.py configs/pipelines/native-v1.yaml
python scripts/doctor.py --config configs/pipelines/native-v1.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-v1.yaml \
  --run-id native-60m-001
```

默认 `micro_batch_size=1`、梯度累积 16、seq512、BF16。
有效 batch 为 16 个序列，每个完整序列提供 511 个监督 token。

```text
train 监督 token:        46,184,530
train packed 样本:       90,381
每 epoch micro batches: 90,381
optimizer steps:        ceil(90,381 / 16) = 5,649
最大监督 token/step:     16 × 511 = 8,176
```

最后一个 step 保持完整累积，可能进入下一轮少量数据，预算是估算而非硬截断 token 上限。
以 `training_budget.json` 和 `training_result.json` 的计划/实际覆盖为准。
当前约 46.2M token 仅约为每个模型参数 0.73 token，是教学预算，不代表充分预训练。

### 6.2 训练后的评测

默认最终 checkpoint 为 step 5,649：

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-v1.yaml \
  --checkpoint runs/native-60m-001/checkpoints/step-00005649 \
  --split dev

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-v1.yaml \
  --checkpoint runs/native-60m-001/checkpoints/step-00005649 \
  --split test
```

改变预算后不得照抄上述 step，以训练输出的 `final_checkpoint` 为准。
当前仓库尚未完成 60M 全程 CUDA 参考运行，不提供虚构的耗时、成本或质量预期。

## 7. 自定义实验配置

### 7.1 创建独立配置

```bash
cp configs/pipelines/native-smoke.yaml configs/pipelines/custom-smoke.yaml
```

在新文件中修改对应字段，保留其余必需配置。自有 seq128 数据示例：

```yaml
model:
  provider: native
  model_id: smoke-10m
  architecture: dense-decoder
  config: configs/models/smoke-10m.yaml
  tokenizer: data/tokenizers/custom-v1
data:
  manifest: data/prepared/custom-v1/data_manifest.json
  packed_manifest: data/packed/custom-v1-seq128/packed_manifest.json
```

这是替换 `model/data` 两段，不是完整 Pipeline。假设 Tokenizer 实际词表为 16,384；
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
  --config configs/pipelines/native-v1.yaml \
  --resume-run native-60m-001
```

默认读取 `latest_checkpoint.json`。明确选择 checkpoint 时：

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-v1.yaml \
  --resume-run native-60m-001 \
  --resume-checkpoint checkpoints/step-00000500
```

第二条只适用于 step 500 之后不存在会被覆盖的后续 checkpoint 的情况。
`--resume-checkpoint` 必须配合 `--resume-run`；`--run-id` 与 `--resume-run` 互斥。

恢复前检查：

1. 原配置 hash、模型、Tokenizer 和数据配套不变。
2. checkpoint 目录完整，确实属于该 run。
3. 设备与依赖保持一致，尤其不要临时修改 `dtype/device`。
4. 恢复目标未到预算上限，不会覆盖已有后续 checkpoint。

**已经完成的两步 Smoke 不能再 resume 追加训练**。当前不支持在同一 run 延长预算，
也没有从旧 checkpoint 分叉的新 run 命令。新 ID 默认从随机初始化开始，不等于续训。
改变预算、数据或模型时应明确规划新的实验。

精确恢复测试目前覆盖 CPU 路径；FP16 GradScaler 和 MPS RNG 尚未纳入保存，
不能据此承诺全部设备/精度逐位恢复。

## 9. 结构消融与 Reference 验收

### 9.1 四臂结构实验

依次运行第 3 节列出的基线、QK-Norm、深窄、深窄 QK-Norm 四份 pipeline，每份使用
独立 run ID。数据、Tokenizer、seed、预算和评测保持一致。
先比较相同结构内的 QK-Norm，再比较相同 QK-Norm 下的宽浅/深窄组合，最后分析交互。
深窄同时改变 hidden/层数/MLP/GQA，不是只改变一个层数参数。

token 预算相同不等于 FLOPs 或耗时相同。应一起比较双语指标、token 吞吐、峰值显存、
总耗时，不预设更复杂结构一定更好。

### 9.2 Reference 是另一份独立配方

机器验收规范已集中到
[`configs/reference/native-60m-pretrain-v1.yaml`](../configs/reference/native-60m-pretrain-v1.yaml)，
对应 `configs/pipelines/native-60m-reference.yaml`，不能拿普通 `native-v1` 的 run 冒充。

**当前限制**：规范固定了原 Data Manifest 文件 hash，而 manifest 含时间戳和来源绝对路径。
在另一台机器重新 prepare 相同文本也可能得到不同 hash。须使用原配套不可变产物；
否则该历史 v1 规范不适用，不能通过跳过校验或把要求改成当前输出就宣称复现成功。
稳定内容身份的跨机器重建支持仍待完善。

符合固定输入与硬件条件时执行：

```bash
git status --short
python scripts/doctor.py --config configs/pipelines/native-60m-reference.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --run-id native-60m-pretrain-v1

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-pretrain-v1/checkpoints/step-00005649 \
  --split dev --json

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-pretrain-v1/checkpoints/step-00005649 \
  --split test --json

python scripts/verify_reference.py \
  --spec configs/reference/native-60m-pretrain-v1.yaml \
  --run runs/native-60m-pretrain-v1
```

`git status --short` 应无输出，正式运行须来自干净 commit。验收检查 completed 状态、
固定配置/数据/模型/Tokenizer/packed hash、5,649 steps、目标 token 46,184,530、
覆盖率 `[1.0, 1.01]`、双语指标、相对 baseline 的 dev 改善、checkpoint 完整性、
独立 dev/test 报告及 Linux/CUDA GPU 不少于 22 GiB。
Reference 的 64 batches 也只是抽样范围，不是全量评测。

实际时间取 `training_result.json`，显存取 `metrics.jsonl`。成本应依据 GPU 计费与
实际占用时长计算，不引用其他项目的 SFT 耗时作为本项目 Pretrain 成本。

## 10. 故障定位

| 现象 | 处理顺序 |
| --- | --- |
| `No module named ...` | 检查 `sys.executable`、Conda 激活、根目录 editable 安装 |
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

1. 补齐 FP16 GradScaler/MPS RNG 恢复，并用连续训练对中断恢复检验。
2. 明确不同有效 token 数下的梯度累积归一化：当前 micro batch 均值等权，
   不一定等于全 token 均值，虽然日志是 token 加权；改动须独立验证训练语义。
3. 分离稳定数据内容身份与时间/路径来源，完善跨机器 Reference。
4. 完成固定 60M CUDA 实验后，再评估 SDPA 快速路径、预取、compile 和可视化。

MiniMind 的 scaler、workers、pin memory 与可选 SwanLab 可供借鉴，但不能因此移除
本项目现有 hash、baseline、双语评测和不可覆盖 checkpoint，也不能把未验证优化计为收益。
