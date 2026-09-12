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

这些 Pipeline 的 `model_route` 都是 `native`。规模由 `run_profile` 表达：
`smoke` 对应 10M 快速验证，`learn` 对应 60M 教学训练，`reproduce` 对应 60M
Reference 验收；具体模型维度仍由 `model.config` 指定。

旧版 `native-smoke` / `native-learn` 配置和 run/checkpoint 契约不再兼容。已有数据、
Tokenizer 和 Packing 通过 hash 校验后仍可复用；不要手改历史 run 或 checkpoint，
应使用新配置和新 run ID 重新开始。

首次选择 `configs/pipelines/native-smoke.yaml`。模型详细参数不在本节重复，见模型文档。

### 3.3 先验证，再做真实 batch 检查

```bash
python scripts/validate_config.py configs/pipelines/native-smoke.yaml
python scripts/doctor.py --config configs/pipelines/native-smoke.yaml
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
第一次 60M 全程 CUDA [Reference 运行](./experiments/native-60m-baseline-v1.md)
已完成 5,649 steps。它记录了实际耗时、显存和双语指标，
项目已在保留 dirty-Git provenance 例外的前提下接受它作为正式 Reference。

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

checkpoint 现已保存 FP16 GradScaler 和 MPS RNG，旧 checkpoint 缺少字段时仍可读取。
精确连续训练对恢复的自动测试目前覆盖 CPU float32；
CUDA scaler 与 MPS RNG 有独立状态往返测试，不能据此承诺跨设备或跨依赖版本逐位一致。

## 9. 结构消融与 Reference 验收

### 9.1 四臂结构实验

依次运行第 3 节列出的基线、QK-Norm、深窄、深窄 QK-Norm 四份 pipeline，每份使用
独立 run ID。数据、Tokenizer、seed、预算和评测保持一致。
先比较相同结构内的 QK-Norm，再比较相同 QK-Norm 下的宽浅/深窄组合，最后分析交互。
深窄同时改变 hidden/层数/MLP/GQA，不是只改变一个层数参数。

token 预算相同不等于 FLOPs 或耗时相同。应一起比较双语指标、token 吞吐、峰值显存、
总耗时，不预设更复杂结构一定更好。

### 9.2 固定 60M 基线

本轮使用
[`configs/reference/native-60m-baseline-v1.yaml`](../configs/reference/native-60m-baseline-v1.yaml)，
对应 `configs/pipelines/native-60m-reference.yaml`。历史
[`native-60m-pretrain-v1.yaml`](../configs/reference/native-60m-pretrain-v1.yaml)
保留原验收要求；新规范增加独立配置锁、源码锁、环境版本锁和最终双语改善检查。
不能拿普通 `native-v1.yaml` 的 learn run 冒充正式基线。

| 固定项 | `native-60m-baseline-v1` |
| --- | --- |
| 模型 | 62,927,616 参数；8 层、hidden 768、12 Q heads / 4 KV heads、MLP 2048 |
| 词表与上下文 | 16,384；训练长度 512；embedding/head 绑定 |
| 初始化与结构 | `initializer_range=0.02`；QK-Norm 关闭；不加入新初始化或结构消融 |
| 数据 | 英文 SimpleStories 100,000 条、中文 Wikipedia 126,000 条 |
| Packing | train 90,381 / dev 11,432 / test 11,055 个样本 |
| 预算 | seed 42；1 epoch；46,184,530 个目标监督 token；5,649 optimizer steps |
| Batch | micro batch 1、梯度累积 16；完整无 padding 时每步 8,176 个监督 token |
| 优化器 | AdamW；LR 0.0003；betas 0.9/0.95；weight decay 0.1；clip 1.0 |
| 调度与评测 | warmup 50；cosine 最低 LR 比例 0.1；每 500 steps 评测和保存；最终步也执行 |
| 评测范围 | 确定性双语分层抽样，每个 dev/test suite 64 个 packed 样本 |
| 环境 | Linux、CUDA BF16、首个可见 GPU 不少于 22 GiB、干净 Git commit |
| Python 与训练依赖 | Python 3.11；NumPy 2.4.6；PyYAML 6.0.3；tokenizers 0.21.4；torch 2.14.0 |

步数按完整梯度累积向上取整，实际 token 可略超 1 epoch；验收覆盖率仍限定在
`[1.0, 1.01]`。64 batches 在 micro batch 为 1 时等于 64 个样本，不是全量评测，
也不是每种语言各 64 个样本。

`freeze.execution_sha256` 同时锁住 Pipeline、模型配置及训练器的有效默认值；
`freeze.source_sha256` 锁住 `src/llm_lifecycle_lab/**/*.py` 的相对路径和文件内容，
与 checkout 的绝对路径、时间戳无关。文档与脚本不参与该源码摘要。
不要在原基线上调学习率、初始化或默认值后重算 hash 放行，应另建实验版本。
第一次完整 CUDA [Reference 运行](./experiments/native-60m-baseline-v1.md)
已获得训练和双语评测结果。其自动检查为 10 pass / 1 `runtime-provenance`
fail；项目已记录一次性例外并接受该结果，不要求重跑。
规范仍保留 `require_clean_git: true`，未来 Reference 继续执行该门禁。

### 9.3 输入与环境门控

**不可变输入**：保留以下三个目录的完整字节内容及目录内部相对路径：

```text
data/prepared/bilingual-60m-v1/
data/tokenizers/bilingual-60m-v1/
data/packed/bilingual-60m-v1-seq512/
```

可以把整套产物复制到另一台机器的相同仓库相对位置，不需要修改 manifest 中作为
来源记录的旧绝对路径。校验会实际读取三个 prepared split、Tokenizer 和全部九个
packed 数组，不只比较 manifest 自报的 hash。

Data Manifest 含时间戳和来源绝对路径，重新 prepare 相同文本也可能产生不同 hash。
没有原配套产物时，新建数据与基线版本，不能修改本规范假装复现；稳定内容身份的
跨机器重建支持不在本轮范围内。

任何已安装训练依赖的机器都可先检查输入，不会创建 run：

```bash
python scripts/verify_reference.py \
  --spec configs/reference/native-60m-baseline-v1.yaml \
  --inputs-only --json
```

`scope=inputs-only` 且 `ok=true` **只表示输入冻结通过**，不表示 CUDA 就绪或模型质量合格。

正式环境沿用 `uv.lock` 的 Python 3.11 分支。可以使用第 2 节的 Conda 环境，
安装以下固定训练依赖；不要安装项目自身：

```bash
conda create -n llm-60m-baseline python=3.11 pip -y
conda activate llm-60m-baseline
python -m pip install "torch==2.14.0"
python -m pip install "numpy==2.4.6" "PyYAML==6.0.3" "tokenizers==0.21.4"
python -m pip check
```

Linux 的 PyTorch wheel 与 NVIDIA 驱动须兼容，按第 2.3 节核对。以上固定了训练直接依赖，
不是全部传递依赖；完整锁环境可改用已安装的 uv：

```bash
uv sync --locked --no-install-project --python 3.11 \
  --extra training --extra public-data --extra dev
```

使用 uv 时，后续命令的 `python` 替换为 `uv run --no-sync python`。不能通过只安装
NumPy 2.4.6 到 Python 3.13 来满足 Python 3.11 要求。
验收比较 PyTorch 发布版本，允许 `+cu...` 后缀；实际 CUDA/cuDNN、GPU 和 Python patch
记录在 `runtime_environment.json`。这不是跨硬件逐位一致性保证，目标 CUDA 环境仍待实测。

在确认并提交本次代码的干净 checkout 上检查，Git 有任何未提交变更都不能发布正式基线。
不要为了清空状态删除自己的工作：

```bash
git status --short
python scripts/verify_reference.py \
  --spec configs/reference/native-60m-baseline-v1.yaml \
  --preflight --json
python scripts/doctor.py --config configs/pipelines/native-60m-reference.yaml
```

两个检查都须通过再训练。`--preflight` 检查冻结输入和真实环境；Doctor 额外检查
资源、输出目录和真实 batch。macOS/无 CUDA/版本不符/dirty Git 的失败是正常门控，
不得降低规范要求。退出码 `0` 表示该 scope 通过，`1` 表示验收不通过，`2` 表示参数、
配置或产物读取错误。

### 9.4 正式训练与验收

训练命令必须带 `--reference-spec`，入口会再次门控；失败时不会创建或恢复 run。

```bash
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --reference-spec configs/reference/native-60m-baseline-v1.yaml \
  --run-id native-60m-baseline-v1

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-baseline-v1/checkpoints/step-00005649 \
  --split dev --json

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-baseline-v1/checkpoints/step-00005649 \
  --split test --json

python scripts/verify_reference.py \
  --spec configs/reference/native-60m-baseline-v1.yaml \
  --run runs/native-60m-baseline-v1 --json
```

未完成 run 恢复时将 `--run-id` 换成 `--resume-run`，保留 `--reference-spec`，并使用
同一源码、配置、输入和环境；不要在普通训练命令里绕过门控后宣称符合基线。
已有完成 run 不覆盖，重复实验使用新的 run ID，最后验收指向实际 run。

最终验收要求：completed 状态、所有冻结项一致、预算完成、指标有限且中英文分桶与
聚合一致、checkpoint 完整，以及最终 checkpoint 的独立 dev/test 报告各 64 个样本。
除历史的 best dev 改善外，**最终 dev 聚合、英文、中文 loss 都必须低于 step 0**；
不能用中途 best 或英文改善掩盖最终中文退化。各语言 loss、perplexity、bits-per-byte
和 token 数都需保留，不预设一个未经正式训练验证的绝对质量阈值。

实际时间取 `training_result.json`，显存取 `metrics.jsonl`。成本应依据 GPU 计费与
实际占用时长计算，不引用其他项目的 SFT 耗时作为本项目 Pretrain 成本。

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

1. 增加 CUDA FP16 与 MPS 的端到端连续训练/恢复对照；当前只完成状态往返单测。
2. 分离稳定数据内容身份与时间/路径来源，完善跨机器 Reference。
3. 以已接受的 60M Reference 为基线开展结构消融，再评估 SDPA、预取和 compile。
4. 在 Pretrain 证据稳定后进入 SFT，不提前叠加 DPO/GRPO。

MiniMind 的 scaler、workers、pin memory 与可选 SwanLab 可供借鉴，但不能因此移除
本项目现有 hash、baseline、双语评测和不可覆盖 checkpoint，也不能把未验证优化计为收益。
