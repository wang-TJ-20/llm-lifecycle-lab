# Native-60M SFT 训练手册

本文是 Native-60M SFT-v1 的**唯一执行入口**。远端 Linux/CUDA 训练、恢复、
开发集验收、能力对照、一次性 sealed test 和产物回传均以本文命令为准。
[SFT 初始化门禁](./SFT_INITIALIZATION_GATE.md)与
[SFT 数据门禁](./SFT_DATA_GATE.md)只解释冻结契约，不重复训练步骤。

当前状态：

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| Stage 0 Base 初始化 | 已冻结并通过 | 只允许 `models/base-v1-final` |
| Stage 1 公开数据 | 已物化并通过 | 14,799 条 source；train/dev/test 为 11,882/1,459/1,458 |
| 本地 8-step MPS smoke | 已通过 | 固定 dev 子集 loss 5.249586 → 5.089118 |
| 本地 100-step pilot | 尚未执行 | 远端正式训练前必须独立运行 |
| 远端 8M-token SFT-v1 | 尚未执行 | 本文不声明任何正式 SFT 结果 |

本地 smoke 只证明 Base → SFT、assistant-only loss、quota sampler、checkpoint 和恢复
链路可运行，不代表正式能力改善。

## 1. 冻结边界

| 范围 | 唯一输入 |
| --- | --- |
| Base | `models/base-v1-final`；权重 SHA-256 `4cdbd642...f9398` |
| Tokenizer | `bilingual-60m-v1`；SHA-256 `7455ac11...4a2f32` |
| 数据 recipe | `configs/data/sft-public-balanced-v1.yaml` |
| prepared manifest | `data/prepared/sft-public-balanced-v1/data_manifest.json` |
| 能力评测 | `configs/evaluation/lifecycle-v4.yaml` |
| 本地 smoke | `configs/pipelines/native-60m-sft-v1-local-smoke.yaml` |
| 本地 pilot | `configs/pipelines/native-60m-sft-v1-local-pilot100.yaml` |
| 正式训练 | `configs/pipelines/native-60m-sft-v1.yaml` |
| 完成态验收 | `configs/gates/native-60m-sft-v1-run.yaml` |

正式配置使用 Linux/CUDA BF16、sequence length 512、micro batch 16、
gradient accumulation 4、学习率 `2e-5` 和 8,000,000 个 assistant 监督 token。
冻结数据与 sampler 将预算解析为 14,020 optimizer steps；确定性实际处理
8,004,634 个监督 token，coverage 为 `1.00057925`。

采样按任务与语言的监督 token quota 执行。任何数据、权重、Tokenizer、suite、
sampler、预算或超参数变化都不是 SFT-v1 重跑，必须创建新的配置和 gate ID。

## 2. 环境与仓库同步

所有命令从仓库根目录执行。远端必须 checkout 将要归档的同一提交：

```bash
set -euo pipefail

git clone https://github.com/wang-TJ-20/llm-lifecycle-lab.git
cd llm-lifecycle-lab
git checkout <SFT_V1_COMMIT>
git lfs install
git lfs pull

python3.11 -m pip install --user uv
uv sync --frozen --extra training --extra public-data --extra dev
source .venv/bin/activate

python -m pip check
test -z "$(git status --porcelain=v1)"
```

`git lfs pull` 后必须存在真实的 `models/base-v1-final/model.pt`，不能是 LFS pointer。
先运行 Base 输入门禁：

```bash
python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --inputs-only
```

退出码非 `0` 时停止，不下载数据、不启动训练。

## 3. 物化冻结数据

`data/` 被 Git 忽略。新机器必须从冻结 recipe 重新物化，或从已验收机器同步
`data/raw/sft-public-balanced-v1/` 和
`data/prepared/sft-public-balanced-v1/`。不能混用两种来源，也不能覆盖已有目录。

从公开来源重新物化：

```bash
set -euo pipefail

test ! -e data/raw/sft-public-balanced-v1
python scripts/data.py fetch-sft \
  --recipe configs/data/sft-public-balanced-v1.yaml \
  --output data/raw/sft-public-balanced-v1 \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-SA-3.0 \
  --accept-license CC-BY-SA-4.0

test ! -e data/prepared/sft-public-balanced-v1
python scripts/data.py prepare \
  --input data/raw/sft-public-balanced-v1/source.jsonl \
  --output data/prepared/sft-public-balanced-v1 \
  --dataset-id sft-public-balanced-v1 \
  --kind sft \
  --license "Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0" \
  --seed 42 \
  --group-by source_id \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1
```

物化后执行完整输入门禁：

```bash
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --inputs-only
```

预期 `14/14` checks 通过。关键身份：

```text
source records: 14799
source sha256: 9ba984310ef0996237908b5843c67d869796fef37e312d2b811f23a647fa772b
train/dev/test: 11882 / 1459 / 1458
train supervised tokens: 475264
```

任何数量、SHA-256、许可、split 或逐条质量检查失败都停止。不要改 gate 中的期望值
来适配当前机器产物。

## 4. 本地 8-step smoke

在 Apple Silicon 上执行：

```bash
set -euo pipefail

python scripts/validate_config.py \
  configs/pipelines/native-60m-sft-v1-local-smoke.yaml

test ! -e runs/native-60m-sft-v1-local-smoke-s42
python scripts/train_sft.py \
  --config configs/pipelines/native-60m-sft-v1-local-smoke.yaml \
  --run-id native-60m-sft-v1-local-smoke-s42
```

固定门禁：

```bash
RUN=runs/native-60m-sft-v1-local-smoke-s42 STEP=8 python - <<'PY'
import json
import math
import os
from pathlib import Path

run = Path(os.environ["RUN"])
step = int(os.environ["STEP"])
manifest = json.loads((run / "run_manifest.json").read_text())
result = json.loads((run / "training_result.json").read_text())
rows = [
    json.loads(line)
    for line in (run / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
baseline = next(row for row in rows if row.get("event") == "baseline")
final = next(row for row in reversed(rows) if row.get("step") == step)

assert manifest["status"] == "completed"
assert result["global_step"] == step
for key in ("eval_loss", "eval_en_loss", "eval_zh_loss"):
    assert math.isfinite(float(baseline[key]))
    assert math.isfinite(float(final[key]))
    assert float(final[key]) < float(baseline[key])
for key in ("train_loss", "gradient_norm"):
    assert math.isfinite(float(final[key]))
print("PASS: local SFT smoke")
PY
```

该门禁只使用训练时固定 dev 子集，不读取 sealed test。任一断言失败都保留 run 后停止。

## 5. 本地 100-step pilot

Pilot 必须从 Base 权重重新开始，不能继承 smoke：

```bash
set -euo pipefail

python scripts/validate_config.py \
  configs/pipelines/native-60m-sft-v1-local-pilot100.yaml

test ! -e runs/native-60m-sft-v1-local-pilot100-s42
python scripts/train_sft.py \
  --config configs/pipelines/native-60m-sft-v1-local-pilot100.yaml \
  --run-id native-60m-sft-v1-local-pilot100-s42

python scripts/eval_sft.py \
  --config configs/pipelines/native-60m-sft-v1-local-pilot100.yaml \
  --checkpoint \
    runs/native-60m-sft-v1-local-pilot100-s42/checkpoints/step-00000100 \
  --split dev \
  --json | tee runs/native-60m-sft-v1-local-pilot100-s42/final-dev-summary.json
```

用上一节相同门禁检查 `RUN=runs/native-60m-sft-v1-local-pilot100-s42 STEP=100`。
另外要求完整 dev 报告存在：

```bash
test -f \
  runs/native-60m-sft-v1-local-pilot100-s42/evaluations/sft-dev-step-00000100.json
```

Pilot 未通过时不得启动远端正式训练。不能根据 pilot 输出修改 SFT-v1 阈值。

## 6. 正式训练前门禁

先提交全部代码、配置、测试与文档。数据和 run 保持 Git 忽略。远端执行：

```bash
set -euo pipefail
source .venv/bin/activate

test -z "$(git status --porcelain=v1)"
python -m pip check
ruff check .
pytest -q

python scripts/validate_config.py configs/pipelines/native-60m-sft-v1.yaml
python scripts/doctor.py \
  --config configs/pipelines/native-60m-sft-v1.yaml

python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --preflight

python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --preflight
```

Doctor 允许 warning，但不能有 failure。两个 `--preflight` 都必须退出 `0`；
它们会拒绝 dirty worktree。训练启动后，`preflight.json` 会保存本次门禁证据。

## 7. 远端正式训练

为本次执行选择唯一 run ID：

```bash
export RUN_ID=native-60m-sft-v1-s42
test ! -e "runs/$RUN_ID"

python scripts/train_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --run-id "$RUN_ID" \
  2>&1 | tee "sft-${RUN_ID}.log"
```

Trainer 每 250 step 在固定 dev 子集上评测，每 500 step 保存完整 checkpoint，
最终必须到达 step 14,020。不要使用相同 run ID 重启已完成 run。

### 中断恢复

恢复前必须 checkout 初次启动时的同一提交、使用同一数据目录，并重新通过两个
`--preflight`。默认从 `latest_checkpoint.json` 指向的完整 checkpoint 恢复：

```bash
test -z "$(git status --porcelain=v1)"

python scripts/verify_sft_init.py \
  --spec configs/gates/native-60m-sft-v1-base-init.yaml \
  --preflight
python scripts/verify_sft_data.py \
  --spec configs/gates/native-60m-sft-v1-data.yaml \
  --preflight

python scripts/train_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --resume-run "$RUN_ID" \
  2>&1 | tee -a "sft-${RUN_ID}.log"
```

只在 `latest_checkpoint.json` 损坏且已人工确认另一个 checkpoint 完整时，才显式加
`--resume-checkpoint runs/$RUN_ID/checkpoints/step-XXXXXXXX`。恢复不能修改配置、
预算或 run ID。

## 8. 完整 dev 评测与验收

训练完成后先运行全量 dev assistant-only loss：

```bash
python scripts/eval_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --checkpoint "runs/$RUN_ID/checkpoints/step-00014020" \
  --split dev \
  --json | tee "runs/$RUN_ID/final-dev-summary.json"
```

再运行完成态开发门禁：

```bash
python scripts/verify_sft_run.py \
  --spec configs/gates/native-60m-sft-v1-run.yaml \
  --run "runs/$RUN_ID" \
  --dev
```

它要求：

- 正式 pipeline、Stage 0/1 gate 和 lifecycle-v4 文件/hash 全部匹配；
- run 为 `reproduce/native/sft/completed`，启动时 clean preflight 已保存；
- 预算精确为 8M target、14,020 steps、8,004,634 actual supervised tokens；
- step-0 到最终 step 的总体、英文、中文 dev loss 全部下降；
- 最终模型、optimizer、trainer state、Tokenizer 和 metadata 完整；
- 全量 1,459 条 dev 报告存在且双语加权一致；
- 初始运行环境为 Linux/CUDA 且源码 clean。

任一检查失败都不得运行 sealed test。

## 9. Base 与 SFT 能力对照

能力对照使用同一个 lifecycle-v4、同一代码提交、同一环境、同一 device/thread、
`native-chat-v1` prompt protocol 和 generation 参数。先评 Base，再评 SFT：

```bash
test ! -e results/capability/native-60m-base-v1-lifecycle-v4
python scripts/evaluate_capabilities.py run \
  --checkpoint models/base-v1-final \
  --suite configs/evaluation/lifecycle-v4.yaml \
  --output results/capability/native-60m-base-v1-lifecycle-v4 \
  --device cuda \
  --threads 1 \
  --max-new-tokens 32 \
  --prompt-protocol native-chat-v1

test ! -e "results/capability/${RUN_ID}-lifecycle-v4"
python scripts/evaluate_capabilities.py run \
  --checkpoint "runs/$RUN_ID/checkpoints/step-00014020" \
  --tokenizer "runs/$RUN_ID/tokenizer" \
  --suite configs/evaluation/lifecycle-v4.yaml \
  --baseline \
    results/capability/native-60m-base-v1-lifecycle-v4/report.json \
  --output "results/capability/${RUN_ID}-lifecycle-v4" \
  --device cuda \
  --threads 1 \
  --max-new-tokens 32 \
  --prompt-protocol native-chat-v1

test ! -e "results/capability/${RUN_ID}-comparison"
python scripts/evaluate_capabilities.py compare \
  --reports \
    results/capability/native-60m-base-v1-lifecycle-v4/report.json \
    "results/capability/${RUN_ID}-lifecycle-v4/report.json" \
  --output "results/capability/${RUN_ID}-comparison"
```

脚本会拒绝协议 hash 不一致的比较。各指标分开解释，不汇总成一个总分；能力探针
用于归因，不替代全量 dev loss 和 sealed test。

## 10. 一次性 sealed test

只有开发门禁通过且决定接受当前 checkpoint 后，才执行一次：

```bash
test ! -e \
  "runs/$RUN_ID/evaluations/sft-test-step-00014020.json"

python scripts/eval_sft.py \
  --config configs/pipelines/native-60m-sft-v1.yaml \
  --checkpoint "runs/$RUN_ID/checkpoints/step-00014020" \
  --split test \
  --json | tee "runs/$RUN_ID/final-test-summary.json"

python scripts/verify_sft_run.py \
  --spec configs/gates/native-60m-sft-v1-run.yaml \
  --run "runs/$RUN_ID" \
  --sealed-test
```

Sealed-test scope在全部 dev 条件之外，额外要求 1,458 条完整 test 报告。Test 结果
只能用于最终记录，不能回头调学习率、预算、采样权重或选择 checkpoint；若据此改动，
必须建立 SFT-v2 并重新冻结。

## 11. 归档与回传

门禁通过后生成不包含自身的校验清单：

```bash
find "runs/$RUN_ID" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "runs/$RUN_ID/SHA256SUMS"

find "results/capability/${RUN_ID}-lifecycle-v4" \
     "results/capability/${RUN_ID}-comparison" \
     -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "results/capability/${RUN_ID}-SHA256SUMS"
```

回传完整 run，而不是只复制 `model.pt`：

```bash
rsync -a --info=progress2 \
  "runs/$RUN_ID/" \
  <LOCAL_HOST>:<LOCAL_REPO>/runs/"$RUN_ID"/

rsync -a --info=progress2 \
  "results/capability/${RUN_ID}-lifecycle-v4/" \
  <LOCAL_HOST>:<LOCAL_REPO>/results/capability/"${RUN_ID}-lifecycle-v4"/

rsync -a --info=progress2 \
  "results/capability/${RUN_ID}-comparison/" \
  <LOCAL_HOST>:<LOCAL_REPO>/results/capability/"${RUN_ID}-comparison"/
```

本地执行 `sha256sum -c runs/$RUN_ID/SHA256SUMS`。归档至少保留
`run_manifest.json`、`resolved_config.yaml`、`preflight.json`、`initialization.json`、
`training_budget.json`、`training_result.json`、`metrics.jsonl`、完整 final checkpoint、
dev/test 报告、runtime provenance、Tokenizer 和能力成绩单。

## 12. 停止规则

| 情况 | 处理 |
| --- | --- |
| 任一 Stage 0/1/完成态 gate 失败 | 停止；保留输出，定位根因 |
| source、split、Tokenizer 或 suite hash 漂移 | 不改期望值；重新同步或建立新版本 |
| 正式启动/恢复时 Git dirty | 提交或移走无关改动后重跑 preflight |
| OOM 或吞吐不足 | 停止当前 SFT-v1；不能原地改 batch/累积后沿用同一 run |
| loss、梯度或评测出现非有限值 | 保留 checkpoint 和日志，停止 |
| 最终总体、英文或中文 dev 任一未优于 baseline | Dev gate 失败，不运行 test |
| 完整 dev 报告缺失或样本数不是 1,459 | 不运行 test |
| sealed test 已存在 | 不重复评测，不据其调参 |
| 训练代码、配置、数据或预算需要变化 | 创建 SFT-v2 配置、gate 和新 run ID |

正式 SFT-v1 尚未运行，因此本文只冻结可执行流程和通过条件，不提前填写最终结果。
