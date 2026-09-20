# Native-60M Base 从零训练手册

本文是当前唯一的 Base 训练入口。旧 Base 实验结果以及 SFT、DPO、GRPO 路线已从
工作树移除；归档只用于追溯，不作为本轮输入。

本轮只回答一个问题：**Native-60M 能否从随机权重开始，在同一套冻结数据和
held-out 上稳定学到更低的双语语言模型损失。** 在 Base 门禁通过前，不设计或运行
任何后训练阶段。

## 1. 固定边界

- 模型：`configs/models/tiny-60m.yaml`，随机初始化，seed 42。
- Tokenizer：`data/tokenizers/bilingual-60m-v1`。
- 数据：`data/prepared/bilingual-60m-v1`。
- Packing：`data/packed/bilingual-60m-v1-seq512`。
- 本地配置：`configs/pipelines/native-60m-base-local-smoke.yaml`。
- 远端配置：`configs/pipelines/native-60m-base-v1.yaml`。
- 本轮只评测同一 manifest 的 dev；通过开发门禁后，test 只运行一次。
- 不从任何历史 checkpoint 初始化，不复用历史 run ID。
- 配置、数据、Tokenizer 或源码变化后必须使用新 run ID。

现有数据包含公开的 SimpleStories 英文数据和中文 Wikipedia。它适合验证训练
与优化闭环，但英文数据是合成故事体，因此本轮结果只是一条可复现 Base 基线，
不代表通用双语模型已经完成。若后续改用非合成或更多领域的数据，必须建立新
recipe 和共同评测 anchor，不能把两套 held-out 的绝对 loss 直接比较。

## 2. 为什么不再使用旧门禁

旧 Base 的主要问题已经确定：

1. 1 epoch 只有约 46.2M token，即 0.73 token/parameter，明显欠训练。
2. 后续实验把不同语料各自的 held-out loss 放在同一绝对阈值下比较，模型效应和
   数据难度无法分离。
3. 旧计划同时展开 Base、SFT、DPO、GRPO，Base 结论未稳定时下游复杂度已经进入。

新流程只保留两个阶段：本地 8-step 链路验证，以及远端 8-epoch Base 训练。
所有收益判断都使用同一 run 的 step-0 baseline。

当前状态：

| 阶段 | 状态 | 结果 |
| --- | --- | --- |
| 本地 Native-60M MPS smoke | 通过 | 8 steps；dev loss 9.834474 → 8.748201；en/zh 均下降 |
| 远端 Base-v1 | 未运行 | 等待从干净提交执行 |

## 3. 本地 MPS 门禁

从仓库根目录执行：

```bash
set -euo pipefail

.venv/bin/python scripts/validate_config.py \
  configs/pipelines/native-60m-base-local-smoke.yaml

.venv/bin/python scripts/doctor.py \
  --config configs/pipelines/native-60m-base-local-smoke.yaml

test ! -e runs/native-60m-base-local-smoke-s42
.venv/bin/python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-local-smoke.yaml \
  --run-id native-60m-base-local-smoke-s42
```

本地门禁：

```bash
.venv/bin/python - <<'PY'
import json
import math
from pathlib import Path

run = Path("runs/native-60m-base-local-smoke-s42")
manifest = json.loads((run / "run_manifest.json").read_text())
rows = [
    json.loads(line)
    for line in (run / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
baseline = next(row for row in rows if row.get("event") == "baseline")
final = next(row for row in reversed(rows) if row.get("step") == 8)

assert manifest["status"] == "completed"
for key in ("eval_loss", "eval_en_loss", "eval_zh_loss"):
    assert math.isfinite(float(baseline[key]))
    assert math.isfinite(float(final[key]))
assert float(final["eval_loss"]) < float(baseline["eval_loss"])
assert math.isfinite(float(final["train_loss"]))
assert math.isfinite(float(final["gradient_norm"]))
print("PASS: local Native-60M MPS smoke")
PY
```

任何断言失败都停止。不要修改阈值，也不要启动远端训练。

## 4. 远端 Base-v1

远端必须使用 Linux、CUDA BF16、干净提交和同一批数据文件。先同步仓库中的
Tokenizer、prepared manifest 及全部 packed 数组，再执行：

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

test -z "$(git status --porcelain=v1)"
python -m pip check
ruff check .
pytest -q

python scripts/validate_config.py configs/pipelines/native-60m-base-v1.yaml
python scripts/doctor.py --config configs/pipelines/native-60m-base-v1.yaml

test ! -e runs/native-60m-base-v1-s42
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-v1.yaml \
  --run-id native-60m-base-v1-s42
```

固定预算为 8 epochs，约 369.5M supervised token 和 45,191 optimizer steps。
每 1,000 step 在固定的 1,024 个 dev 窗口上评测；每 5,000 step 保存 checkpoint。

## 5. Base 开发门禁

训练完成后只读取 `metrics.jsonl` 和 `training_result.json`：

```bash
python - <<'PY'
import json
import math
from pathlib import Path

run = Path("runs/native-60m-base-v1-s42")
manifest = json.loads((run / "run_manifest.json").read_text())
result = json.loads((run / "training_result.json").read_text())
rows = [
    json.loads(line)
    for line in (run / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
baseline = next(row for row in rows if row.get("event") == "baseline")
evaluated = [row for row in rows if "eval_loss" in row and "train_loss" in row]
final = evaluated[-1]
best = min(float(row["eval_loss"]) for row in evaluated)

assert manifest["status"] == "completed"
assert result["global_step"] == 45191
assert 1.0 <= float(result["target_token_coverage"]) <= 1.01
for row in (baseline, final):
    for key in ("eval_loss", "eval_en_loss", "eval_zh_loss"):
        assert math.isfinite(float(row[key]))
assert float(final["eval_loss"]) <= 0.50 * float(baseline["eval_loss"])
assert float(final["eval_en_loss"]) < float(baseline["eval_en_loss"])
assert float(final["eval_zh_loss"]) < float(baseline["eval_zh_loss"])
assert float(final["eval_loss"]) <= 1.02 * best
print("PASS: Base-v1 development gate")
PY
```

门禁失败时保留全部证据并停止。只允许从中间 checkpoint 做同配置诊断，不允许
降低阈值或切换数据后继续复用同一 run ID。

门禁通过后，才对最终 checkpoint 运行一次 test：

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-v1.yaml \
  --checkpoint runs/native-60m-base-v1-s42/checkpoints/step-00045191 \
  --split test \
  --json > runs/native-60m-base-v1-s42/final-test.json
```

## 6. 停止规则

- 本地 MPS 门禁失败：修训练链路，不上远端。
- 远端数值异常、token coverage 不合格或任一语言未改善：停止在 Base。
- final dev 比最佳 checkpoint 回退超过 2%：从已有 checkpoint 诊断训练日程，
  不直接进入后训练。
- Base 通过只代表语言模型训练闭环成立；在定义新的下游能力门禁前，不启动 SFT。

旧路线恢复入口：

```text
archive/pre-base-restart-20260920
```
