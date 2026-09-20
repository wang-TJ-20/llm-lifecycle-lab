# Native-60M 公开数据后训练续跑手册

本文是远程服务器的唯一续跑入口。R0 交叉评测和 R1 两个 large-SFT arm
**已经完成，不得重跑**。执行者应从第 4 节 D0 诊断开始，然后按
SFT-v3、DPO、GRPO 的门禁顺序继续。

SFT-v3 析因矩阵已执行完毕（四臂全部未通过硬门禁，第 8–10 节因此未运行），
结果记录与机制归因见
[Native-60M 公开数据后训练 v3](./experiments/native-60m-public-posttrain-v3.md)；
更早的历史结果与完整指标见
[Native-60M 公开数据后训练 v2](./experiments/native-60m-public-posttrain-v2.md)。

## 1. 当前状态与结论

已完成：

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| Base-v1 / Base-repeat / Base-v2 | 完成 | Base-v2 与 repeat 存在领域 tradeoff |
| R0：3 × 2 dev 交叉评测 | 完成 | 旧 Base 门禁比较口径失配 |
| R1：两个 parent baseline | 完成 | 协议 hash 相同 |
| R1：两个 old-data/2M SFT | 完成，均失败 | objective、BPB 通过，严格能力项近零 |
| DPO / GRPO | 未运行 | SFT 无 winner，按门禁正确停止 |

R1 关键结果：

| SFT arm | dev loss | instruction | format | QA | verifiable |
| --- | --- | ---: | ---: | ---: | ---: |
| Base-repeat + old-data/2M | 5.0826 → 4.1860 | 0 | 0 | 0 | 0 |
| Base-v2 + old-data/2M | 3.9896 → 3.5078 | 1 | 0 | 0 | 1 |

失败主因不是训练链路：

- `public-60m-v2` train 有 894,771 个 SFT supervised token；
- HC3 与 Dolly 的长回答占绝大部分 token；
- MSVAMP 短整数回答只占极小比例；
- 没有足够的公开短答案和 JSON/CSV 格式监督；
- SFT loss 按 supervised token 归一，短回答仅增加记录数仍会被长回答淹没。

因此本轮不降低门禁、不直接跑 DPO/GRPO，也不立即重训 Base-v3。先运行
SFT-v3 数据/预算析因矩阵。

## 2. 固定实验契约

- 模型：`configs/models/tiny-60m.yaml`，62.93M 参数。
- SFT-v3 parent：只使用 Base-v2。
- Tokenizer：`data/tokenizers/bilingual-60m-v1`。
- seed 42、context 512、BF16、`native-chat-v1`。
- 开发门禁：`configs/evaluation/lifecycle-v3.yaml`。
- sealed test：`configs/evaluation/lifecycle-v4.yaml`，只允许最终候选评测一次。
- SFT/DPO/GRPO source group 交集必须为 0。
- SFT-v3 使用公开人工标注数据；结构化视图只对公开答案做确定性 JSON
  序列化，不生成新答案。
- 所有输出目录不可覆盖；改变数据、预算、学习率或 parent 必须使用新 run ID。

SFT-v3 的优化 token 配额固定为：

| task family | supervised-token quota |
| --- | ---: |
| general | 40% |
| short_qa | 25% |
| classification | 10% |
| numeric | 15% |
| structured | 10% |

任务族内部的 en/zh 配额分别为：general 17.5%/22.5%、short QA
10%/15%、classification 10%/0%、numeric 7.5%/7.5%、structured 5%/5%；
总计 en/zh 各 50%。中文 classification 没有引入许可清晰的公开标注源，因此不
伪造该数据，而由其他任务族补齐语言配额。

`supervised-token-quota` 根据累计 supervised token 选择下一任务×语言层，并持久化
每层的 epoch、offset 和累计 token。恢复训练必须得到相同的后续 batch 序列。

## 3. 远程预检

从仓库根目录执行：

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

python -m pip check
test -z "$(git status --porcelain=v1)"
git rev-parse HEAD
ruff check .
pytest -q

test -f scripts/analyze_capability_outputs.py
test -f configs/evaluation/lifecycle-v4.yaml
test -f configs/pipelines/native-sft-public-60m-v3-basev2-2m.yaml
test -f configs/pipelines/native-sft-public-60m-v3-basev2-8m.yaml
test -f configs/pipelines/native-sft-public-60m-v2-basev2-8m.yaml

BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
test -f "$BASE_V2/model/model.pt"
test -f runs/evaluations/r1-parent-basev2/report.json
test -f runs/gates/native-sft-public-60m-r1-basev2-large-s42.json
jq -e '.ok == false' \
  runs/gates/native-sft-public-60m-r1-basev2-large-s42.json
```

确认旧 R1 两个 SFT run 完成，且中间 checkpoint 仍在：

```bash
for run_id in \
  native-sft-public-60m-r1-repeat-large-s42 \
  native-sft-public-60m-r1-basev2-large-s42
do
  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  found=0
  for step in 50 100 150 200 250 300 350 383; do
    checkpoint=$(printf 'runs/%s/checkpoints/step-%08d' "$run_id" "$step")
    if test -f "$checkpoint/model/model.pt"; then
      found=$((found + 1))
    fi
  done
  test "$found" -gt 0
  printf '%s: %s diagnostic checkpoints available\n' "$run_id" "$found"
done
```

若中间 checkpoint 已清理，D0 只评测仍存在的 checkpoint，并在结果记录中注明；
不得为补 D0 重跑旧 SFT。

## 4. D0：旧 SFT checkpoint 轨迹诊断

D0 不训练、不选 winner、不改变硬门禁。它回答两个问题：

1. strict exact success 是否曾在中途出现后消失；
2. 模型是否已包含答案但总是附加解释，或 JSON 始终不可解析。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json
mkdir -p runs/evaluations/sft-r1-checkpoint-diagnostics

diagnose_r1() {
  local model_name=$1
  local run_id=$2
  local parent_report=$3
  local step checkpoint output
  local -a report_args=()

  for step in 50 100 150 200 250 300 350 383; do
    checkpoint=$(printf 'runs/%s/checkpoints/step-%08d' "$run_id" "$step")
    test -f "$checkpoint/model/model.pt" || continue
    output=$(printf \
      'runs/evaluations/sft-r1-checkpoint-diagnostics/%s-step-%08d' \
      "$model_name" "$step")
    test ! -e "$output"

    python scripts/evaluate_capabilities.py run \
      --checkpoint "$checkpoint" \
      --tokenizer data/tokenizers/bilingual-60m-v1 \
      --suite configs/evaluation/lifecycle-v3.yaml \
      --output "$output" \
      --device cuda \
      --threads 1 \
      --max-new-tokens 32 \
      --prompt-protocol native-chat-v1 \
      --pretrain-manifest "$COMMON_PRETRAIN_MANIFEST" \
      --baseline "$parent_report"

    report_args+=(--report "$output/report.json")
  done

  test "${#report_args[@]}" -gt 0
  python scripts/analyze_capability_outputs.py \
    "${report_args[@]}" \
    --output \
      "runs/evaluations/sft-r1-checkpoint-diagnostics/${model_name}-summary.json"
}

diagnose_r1 \
  repeat \
  native-sft-public-60m-r1-repeat-large-s42 \
  runs/evaluations/r1-parent-repeat/report.json

diagnose_r1 \
  basev2 \
  native-sft-public-60m-r1-basev2-large-s42 \
  runs/evaluations/r1-parent-basev2/report.json
```

解释规则：

| 现象 | 结论 |
| --- | --- |
| exact success 中途明显更高、后期归零 | 存在预算/日程效应，但旧 arm 仍未过最终门禁 |
| answer-contained 上升、exact 仍低、extra-text 高 | 主要是短输出与格式控制缺失 |
| JSON parseable 始终接近 0 | 需要显式 structured 监督 |
| 所有诊断始终近 0 | 旧数据在 2M 内没有形成可测迁移 |

无论 D0 出现哪种结果，都继续第 5–7 节的预注册矩阵；不得选择旧失败 checkpoint
进入 DPO。

## 5. 冻结 sealed test

`lifecycle-v4` 与 v3 使用相同任务族和分母，但 surface offset 为 1000，且构建时
对 `public-60m-v3` 三阶段 source 做过泄漏检查。固定文件 hash：

```bash
echo \
  'cbc4c6e0216ac6186a607f9de9c2081b86ff0e347deac21d71f74203e9dd72e5  configs/evaluation/lifecycle-v4.yaml' \
  | shasum -a 256 -c -
```

在第 10 节确定最终模型前，不得运行、查看或据此调整配方。

## 6. 物化 public-60m-v3

### 6.1 下载并固定规范化 source

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

test ! -e data/raw/public-60m-v3
python scripts/data.py fetch-posttrain \
  --recipe public-60m-v3 \
  --output data/raw/public-60m-v3 \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-SA-3.0 \
  --accept-license CC-BY-SA-4.0 \
  --accept-license CC-BY-4.0

jq -e '
  .recipe_id == "public-60m-v3"
  and .stages == [
    {
      "manifest": "sft/source_manifest.json",
      "record_kind": "sft",
      "records": 14799,
      "source_sha256":
        "9ba984310ef0996237908b5843c67d869796fef37e312d2b811f23a647fa772b"
    },
    {
      "manifest": "dpo/source_manifest.json",
      "record_kind": "dpo",
      "records": 2200,
      "source_sha256":
        "565ebaf82e70eb44e5548149fa963342503e4f979d86c932fa3cf7fcf0ed43bb"
    },
    {
      "manifest": "grpo/source_manifest.json",
      "record_kind": "grpo",
      "records": 400,
      "source_sha256":
        "9e1f60bc945e39d5b2fc05a43c3a3cf3dd374bcc437c58773f3522d7156aad93"
    }
  ]
' data/raw/public-60m-v3/bundle_manifest.json
```

### 6.2 按 source_id 分组切分

```bash
python scripts/data.py prepare \
  --input data/raw/public-60m-v3/sft/source.jsonl \
  --output data/prepared/sft-public-60m-v3 \
  --dataset-id sft-public-60m-v3 \
  --kind sft \
  --license 'Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0' \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v3/dpo/source.jsonl \
  --output data/prepared/dpo-public-60m-v3 \
  --dataset-id dpo-public-60m-v3 \
  --kind dpo \
  --license 'Apache-2.0 AND CC-BY-4.0' \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v3/grpo/source.jsonl \
  --output data/prepared/grpo-public-60m-v3 \
  --dataset-id grpo-public-60m-v3 \
  --kind grpo \
  --license 'Apache-2.0' \
  --seed 42 \
  --group-by source_id
```

### 6.3 完整数据门禁

```bash
mkdir -p runs/qualifications
DATA_V3_CHECK=runs/qualifications/public-60m-v3-data-check.json
test ! -e "$DATA_V3_CHECK"

python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v3/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v3/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v3/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16 \
  --json > "$DATA_V3_CHECK"

jq -e '
  .ok == true
  and .stage_source_id_overlap == {
    "dpo-grpo": 0,
    "sft-dpo": 0,
    "sft-grpo": 0
  }
  and .sft.splits.train.examples == 11882
  and .sft.splits.train.supervised_tokens == 475264
  and .sft.splits.train.task_families.general.examples == 4114
  and .sft.splits.train.task_families.short_qa.examples == 5292
  and .sft.splits.train.task_families.classification.examples == 363
  and .sft.splits.train.task_families.numeric.examples == 1298
  and .sft.splits.train.task_families.structured.examples == 815
  and .dpo.splits.train.pairs == 1741
  and .grpo.splits.train.prompts == 318
' "$DATA_V3_CHECK"
```

若任何数量、hash、长度、泄漏或 tokenizer 检查失败，停止。不得修改断言继续。

## 7. SFT-v3 析因矩阵

### 7.1 四个固定 arm

| arm | 数据 | 预算 | LR | 作用 |
| --- | --- | ---: | ---: | --- |
| A0，已完成 | old v2 | 2M | 2e-5 | 历史基线 |
| A1 | old v2 | 8M | 2e-5 | 只测预算 |
| B0 | balanced v3 | 2M | 2e-5 | 只测数据/采样 |
| B1 | balanced v3 | 8M | 2e-5 | 数据与预算组合 |

四臂都从同一个 Base-v2 初始化。A0 不重跑；其余三臂执行：
新 arm 将 `checkpoint_interval` 统一设为 2,000、`eval_interval` 设为 250；
这是防止短答案配额导致 step 数增多后耗尽磁盘的观测频率调整，不改变 batch、
梯度、学习率或 token 预算。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate
mkdir -p runs/gates

python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

root = Path("configs/pipelines")
paths = {
    "old8": root / "native-sft-public-60m-v2-basev2-8m.yaml",
    "new2": root / "native-sft-public-60m-v3-basev2-2m.yaml",
    "new8": root / "native-sft-public-60m-v3-basev2-8m.yaml",
    "lr": root / "native-sft-public-60m-v3-basev2-8m-lr5e5.yaml",
}
configs = {
    name: yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, path in paths.items()
}
assert len({value["model"]["init_checkpoint"] for value in configs.values()}) == 1
assert len({value["seed"] for value in configs.values()}) == 1
assert configs["old8"]["training"] == configs["new8"]["training"]
assert configs["new2"]["data"] == configs["new8"]["data"]

new2 = deepcopy(configs["new2"]["training"])
new8 = deepcopy(configs["new8"]["training"])
new2.pop("max_train_tokens")
new8.pop("max_train_tokens")
assert new2 == new8

base = deepcopy(configs["new8"]["training"])
lr = deepcopy(configs["lr"]["training"])
base.pop("learning_rate")
lr.pop("learning_rate")
assert base == lr
assert configs["new8"]["training"]["learning_rate"] == 2e-5
assert configs["lr"]["training"]["learning_rate"] == 5e-5
print("PASS: SFT-v3 control matrix is internally consistent")
PY

BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
BASE_REPORT=runs/evaluations/r1-parent-basev2/report.json
OLD_DATA_CHECK=runs/qualifications/public-60m-v2-data-check-r1.json
V3_DATA_CHECK=runs/qualifications/public-60m-v3-data-check.json
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json

run_gate_allow_fail() {
  if "$@"; then
    return 0
  else
    local status=$?
  fi
  test "$status" -eq 1 && return 0
  return "$status"
}

run_sft_arm() {
  local config=$1
  local run_id=$2
  local data_check=$3
  local checkpoint

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$BASE_V2"

  python scripts/train_sft.py \
    --config "$config" \
    --init-checkpoint "$BASE_V2" \
    --run-id "$run_id"

  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  jq -e '.code.dirty == false' "runs/$run_id/runtime_environment.json"
  checkpoint=$(jq -er '.final_checkpoint' \
    "runs/$run_id/training_result.json")
  test -f "$checkpoint/model/model.pt"

  python scripts/evaluate_capabilities.py run \
    --checkpoint "$checkpoint" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda \
    --threads 1 \
    --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$COMMON_PRETRAIN_MANIFEST" \
    --baseline "$BASE_REPORT"

  run_gate_allow_fail python scripts/check_stage_gate.py sft \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$BASE_REPORT" \
    --data-check "$data_check" \
    --output "runs/gates/$run_id.json"
}

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev2-8m.yaml \
  native-sft-public-60m-r2-old8m-basev2-s42 \
  "$OLD_DATA_CHECK"

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v3-basev2-2m.yaml \
  native-sft-public-60m-r2-balanced2m-basev2-s42 \
  "$V3_DATA_CHECK"

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v3-basev2-8m.yaml \
  native-sft-public-60m-r2-balanced8m-basev2-s42 \
  "$V3_DATA_CHECK"
```

硬门禁保持不变：dev objective 下降至少 5%；instruction ≥4、format ≥3、
QA ≥4、verifiable ≥12；instruction/format/QA 的 en/zh 各 ≥1；BPB
不超过 parent 的 1.02 倍；跨阶段 source group 交集为 0。

### 7.2 学习率敏感性资格

只有 B1 未通过硬门禁但满足以下全部条件，才运行 `5e-5`：

- instruction、format、QA、verifiable 成功数之和至少为 6；
- instruction、format、QA 中至少两个类别非零；
- 成功数之和严格高于 old-data/8M；
- B1 的 BPB retention 检查通过。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
BASE_REPORT=runs/evaluations/r1-parent-basev2/report.json
V3_DATA_CHECK=runs/qualifications/public-60m-v3-data-check.json
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json

run_gate_allow_fail() {
  if "$@"; then return 0; else local status=$?; fi
  test "$status" -eq 1 && return 0
  return "$status"
}

run_sft_arm() {
  local config=$1
  local run_id=$2
  local data_check=$3
  local checkpoint

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"
  python scripts/doctor.py \
    --config "$config" --init-checkpoint "$BASE_V2"
  python scripts/train_sft.py \
    --config "$config" --init-checkpoint "$BASE_V2" --run-id "$run_id"
  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  jq -e '.code.dirty == false' "runs/$run_id/runtime_environment.json"
  checkpoint=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  test -f "$checkpoint/model/model.pt"
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$checkpoint" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$COMMON_PRETRAIN_MANIFEST" \
    --baseline "$BASE_REPORT"
  run_gate_allow_fail python scripts/check_stage_gate.py sft \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$BASE_REPORT" \
    --data-check "$data_check" \
    --output "runs/gates/$run_id.json"
}

test ! -e runs/qualifications/sft-v3-lr-sensitivity.json
python - <<'PY'
import json
from pathlib import Path

root = Path("runs")
old = json.loads(
    (root / "evaluations/native-sft-public-60m-r2-old8m-basev2-s42/report.json")
    .read_text()
)
balanced = json.loads(
    (root / "evaluations/native-sft-public-60m-r2-balanced8m-basev2-s42/report.json")
    .read_text()
)
gate = json.loads(
    (root / "gates/native-sft-public-60m-r2-balanced8m-basev2-s42.json")
    .read_text()
)
names = (
    "instruction.success",
    "format.success",
    "qa.success",
    "verifiable.reward",
)

def successes(report, name):
    metric = report["metrics"][name]
    return round(metric["value"] * metric["count"])

old_counts = {name: successes(old, name) for name in names}
new_counts = {name: successes(balanced, name) for name in names}
bpb_ok = next(
    check["ok"]
    for check in gate["checks"]
    if check["name"] == "corpus-bpb-retention"
)
core_nonzero = sum(new_counts[name] > 0 for name in names[:3])
run_lr = (
    not gate["ok"]
    and sum(new_counts.values()) >= 6
    and core_nonzero >= 2
    and sum(new_counts.values()) > sum(old_counts.values())
    and bpb_ok
)
result = {
    "run_lr": run_lr,
    "old_counts": old_counts,
    "balanced_counts": new_counts,
    "core_nonzero": core_nonzero,
    "bpb_retention": bpb_ok,
}
Path("runs/qualifications/sft-v3-lr-sensitivity.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
PY

if jq -e '.run_lr == true' \
  runs/qualifications/sft-v3-lr-sensitivity.json >/dev/null
then
  run_sft_arm \
    configs/pipelines/native-sft-public-60m-v3-basev2-8m-lr5e5.yaml \
    native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42 \
    "$V3_DATA_CHECK"
else
  echo "Skip LR sensitivity: pre-registered trend condition not met"
fi
```

### 7.3 选择 SFT winner

```bash
set -euo pipefail

GATES=(
  runs/gates/native-sft-public-60m-r1-basev2-large-s42.json
  runs/gates/native-sft-public-60m-r2-old8m-basev2-s42.json
  runs/gates/native-sft-public-60m-r2-balanced2m-basev2-s42.json
  runs/gates/native-sft-public-60m-r2-balanced8m-basev2-s42.json
)
if test -f \
  runs/gates/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42.json
then
  GATES+=(
    runs/gates/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42.json
  )
fi

if ! jq -e -s 'any(.[]; .ok == true)' "${GATES[@]}" >/dev/null; then
  echo "STOP: no SFT-v3 candidate passed; do not run DPO or GRPO" >&2
  exit 1
fi

CANDIDATES=(
  --candidate
  runs/native-sft-public-60m-r1-basev2-large-s42
  runs/evaluations/native-sft-public-60m-r1-basev2-large-s42/report.json
  runs/gates/native-sft-public-60m-r1-basev2-large-s42.json
  --candidate
  runs/native-sft-public-60m-r2-old8m-basev2-s42
  runs/evaluations/native-sft-public-60m-r2-old8m-basev2-s42/report.json
  runs/gates/native-sft-public-60m-r2-old8m-basev2-s42.json
  --candidate
  runs/native-sft-public-60m-r2-balanced2m-basev2-s42
  runs/evaluations/native-sft-public-60m-r2-balanced2m-basev2-s42/report.json
  runs/gates/native-sft-public-60m-r2-balanced2m-basev2-s42.json
  --candidate
  runs/native-sft-public-60m-r2-balanced8m-basev2-s42
  runs/evaluations/native-sft-public-60m-r2-balanced8m-basev2-s42/report.json
  runs/gates/native-sft-public-60m-r2-balanced8m-basev2-s42.json
)
if test -f \
  runs/gates/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42.json
then
  CANDIDATES+=(
    --candidate
    runs/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42
    runs/evaluations/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42/report.json
    runs/gates/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42.json
  )
fi

test ! -e runs/gates/sft-r2-winner.json
python scripts/select_stage_winner.py \
  --stage sft \
  "${CANDIDATES[@]}" \
  --output runs/gates/sft-r2-winner.json
```

winner 排序仍为：四类任务成功数之和降序、BPB 升序、SFT dev loss 升序。

## 8. DPO：pure 与 NLL 0.10

只有 `sft-r2-winner.json` 存在时执行。两臂使用同一个 SFT winner；单臂失败不
阻断另一臂，两臂都失败则回退 SFT winner。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

root = Path("configs/pipelines")
paths = [
    root / "native-dpo-public-60m-v3-pure.yaml",
    root / "native-dpo-public-60m-v3-nll010.yaml",
]
configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
values = []
for config in configs:
    value = deepcopy(config)
    values.append(value["training"].pop("nll_coefficient"))
assert configs[0]["data"] == configs[1]["data"]
assert configs[0]["model"] == configs[1]["model"]
left = deepcopy(configs[0])
right = deepcopy(configs[1])
left["training"].pop("nll_coefficient")
right["training"].pop("nll_coefficient")
assert left == right
assert values == [0.0, 0.1]
print("PASS: DPO arms differ only by nll_coefficient")
PY

SFT_CKPT=$(jq -er '.winner.final_checkpoint' runs/gates/sft-r2-winner.json)
SFT_REPORT=$(jq -er '.winner.capability_report' runs/gates/sft-r2-winner.json)
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json

run_gate_allow_fail() {
  if "$@"; then return 0; else local status=$?; fi
  test "$status" -eq 1 && return 0
  return "$status"
}

run_dpo_arm() {
  local config=$1
  local run_id=$2
  local checkpoint

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"
  python scripts/doctor.py --config "$config" --init-checkpoint "$SFT_CKPT"
  python scripts/train_dpo.py \
    --config "$config" --init-checkpoint "$SFT_CKPT" --run-id "$run_id"
  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  jq -e '.code.dirty == false' "runs/$run_id/runtime_environment.json"
  checkpoint=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  test -f "$checkpoint/model/model.pt"
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$checkpoint" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$COMMON_PRETRAIN_MANIFEST" \
    --baseline "$SFT_REPORT"
  run_gate_allow_fail python scripts/check_stage_gate.py dpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$SFT_REPORT" \
    --output "runs/gates/$run_id.json"
}

run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v3-pure.yaml \
  native-dpo-public-60m-r2-pure-s42
run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v3-nll010.yaml \
  native-dpo-public-60m-r2-nll010-s42

if jq -e -s 'any(.[]; .ok == true)' \
  runs/gates/native-dpo-public-60m-r2-pure-s42.json \
  runs/gates/native-dpo-public-60m-r2-nll010-s42.json >/dev/null
then
  python scripts/select_stage_winner.py \
    --stage dpo \
    --candidate \
      runs/native-dpo-public-60m-r2-pure-s42 \
      runs/evaluations/native-dpo-public-60m-r2-pure-s42/report.json \
      runs/gates/native-dpo-public-60m-r2-pure-s42.json \
    --candidate \
      runs/native-dpo-public-60m-r2-nll010-s42 \
      runs/evaluations/native-dpo-public-60m-r2-nll010-s42/report.json \
      runs/gates/native-dpo-public-60m-r2-nll010-s42.json \
    --output runs/gates/dpo-r2-winner.json
else
  echo "Both DPO arms failed; keep SFT winner"
fi
```

DPO 门禁不变：preference loss 改善；独立 preference 至少增加 3 个成功；
已有能力每类最多下降 1 个；BPB 不超过 parent 的 1.02 倍。

## 9. GRPO：资格检查与 KL 扫描

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

root = Path("configs/pipelines")
paths = [
    root / "native-grpo-public-60m-v3-kl004.yaml",
    root / "native-grpo-public-60m-v3-kl008.yaml",
    root / "native-grpo-public-60m-v3-kl012.yaml",
]
configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
betas = []
normalized = []
for config in configs:
    value = deepcopy(config)
    betas.append(value["training"].pop("kl_beta"))
    normalized.append(value)
assert normalized[0] == normalized[1] == normalized[2]
assert betas == [0.04, 0.08, 0.12]
print("PASS: GRPO arms differ only by kl_beta")
PY

if test -f runs/gates/dpo-r2-winner.json; then
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/dpo-r2-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/dpo-r2-winner.json)
else
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/sft-r2-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/sft-r2-winner.json)
fi

QUAL_DIR=runs/qualifications/native-grpo-public-60m-r2-parent-s42
test ! -e "$QUAL_DIR"
if python scripts/qualify_grpo.py \
  --config configs/pipelines/native-grpo-public-60m-v3-kl004.yaml \
  --init-checkpoint "$GRPO_PARENT_CKPT" \
  --source-groups 64 \
  --minimum-mixed-group-fraction 0.25 \
  --output "$QUAL_DIR"
then
  GRPO_QUALIFIED=1
else
  status=$?
  test "$status" -eq 1 || exit "$status"
  GRPO_QUALIFIED=0
fi
```

资格失败时不运行任何 GRPO arm。资格通过时：

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

if test -f runs/gates/dpo-r2-winner.json; then
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/dpo-r2-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/dpo-r2-winner.json)
else
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/sft-r2-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/sft-r2-winner.json)
fi

if jq -e '.ok == true' \
  runs/qualifications/native-grpo-public-60m-r2-parent-s42/qualification.json \
  >/dev/null
then
  GRPO_QUALIFIED=1
else
  GRPO_QUALIFIED=0
fi

run_gate_allow_fail() {
  if "$@"; then return 0; else local status=$?; fi
  test "$status" -eq 1 && return 0
  return "$status"
}

run_grpo_arm() {
  local config=$1
  local run_id=$2
  local checkpoint

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"
  python scripts/doctor.py \
    --config "$config" --init-checkpoint "$GRPO_PARENT_CKPT"
  python scripts/train_grpo.py \
    --config "$config" \
    --init-checkpoint "$GRPO_PARENT_CKPT" \
    --run-id "$run_id"
  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  jq -e '.code.dirty == false' "runs/$run_id/runtime_environment.json"
  checkpoint=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  test -f "$checkpoint/model/model.pt"
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$checkpoint" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest data/prepared/bilingual-60m-v2/data_manifest.json \
    --baseline "$GRPO_PARENT_REPORT"
  run_gate_allow_fail python scripts/check_stage_gate.py grpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$GRPO_PARENT_REPORT" \
    --output "runs/gates/$run_id.json"
}

if test "$GRPO_QUALIFIED" -eq 1; then
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v3-kl004.yaml \
    native-grpo-public-60m-r2-kl004-s42
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v3-kl008.yaml \
    native-grpo-public-60m-r2-kl008-s42
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v3-kl012.yaml \
    native-grpo-public-60m-r2-kl012-s42

  if jq -e -s 'any(.[]; .ok == true)' \
    runs/gates/native-grpo-public-60m-r2-kl004-s42.json \
    runs/gates/native-grpo-public-60m-r2-kl008-s42.json \
    runs/gates/native-grpo-public-60m-r2-kl012-s42.json >/dev/null
  then
    python scripts/select_stage_winner.py \
      --stage grpo \
      --candidate \
        runs/native-grpo-public-60m-r2-kl004-s42 \
        runs/evaluations/native-grpo-public-60m-r2-kl004-s42/report.json \
        runs/gates/native-grpo-public-60m-r2-kl004-s42.json \
      --candidate \
        runs/native-grpo-public-60m-r2-kl008-s42 \
        runs/evaluations/native-grpo-public-60m-r2-kl008-s42/report.json \
        runs/gates/native-grpo-public-60m-r2-kl008-s42.json \
      --candidate \
        runs/native-grpo-public-60m-r2-kl012-s42 \
        runs/evaluations/native-grpo-public-60m-r2-kl012-s42/report.json \
        runs/gates/native-grpo-public-60m-r2-kl012-s42.json \
      --output runs/gates/grpo-r2-winner.json
  fi
fi
```

GRPO 三臂除 `kl_beta=0.04/0.08/0.12` 外一致。全部失败时回退 DPO winner；
无 DPO winner 时回退 SFT winner。

## 10. 最终成绩单与 sealed test

先生成同一 v3 协议的纵向成绩单，再确定最后一个通过门禁的模型：

```bash
set -euo pipefail

REPORTS=(
  runs/evaluations/r1-parent-basev2/report.json
  "$(jq -er '.winner.capability_report' runs/gates/sft-r2-winner.json)"
)
if test -f runs/gates/dpo-r2-winner.json; then
  REPORTS+=(
    "$(jq -er '.winner.capability_report' runs/gates/dpo-r2-winner.json)"
  )
fi
if test -f runs/gates/grpo-r2-winner.json; then
  REPORTS+=(
    "$(jq -er '.winner.capability_report' runs/gates/grpo-r2-winner.json)"
  )
fi

test "$(
  jq -r '.protocol_sha256' "${REPORTS[@]}" |
    sort -u | wc -l | tr -d ' '
)" -eq 1
test ! -e runs/evaluations/public-posttrain-r2-v3
python scripts/evaluate_capabilities.py compare \
  --reports "${REPORTS[@]}" \
  --output runs/evaluations/public-posttrain-r2-v3

if test -f runs/gates/grpo-r2-winner.json; then
  FINAL_GATE=runs/gates/grpo-r2-winner.json
elif test -f runs/gates/dpo-r2-winner.json; then
  FINAL_GATE=runs/gates/dpo-r2-winner.json
else
  FINAL_GATE=runs/gates/sft-r2-winner.json
fi

FINAL_CKPT=$(jq -er '.winner.final_checkpoint' "$FINAL_GATE")
FINAL_V3_REPORT=$(jq -er '.winner.capability_report' "$FINAL_GATE")
test ! -e runs/evaluations/public-posttrain-r2-sealed-v4

python scripts/evaluate_capabilities.py run \
  --checkpoint "$FINAL_CKPT" \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v4.yaml \
  --output runs/evaluations/public-posttrain-r2-sealed-v4 \
  --device cuda \
  --threads 1 \
  --max-new-tokens 32 \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v2/data_manifest.json

test -f "$FINAL_V3_REPORT"
jq -e '.protocol.suite_id == "lifecycle-v4"' \
  runs/evaluations/public-posttrain-r2-sealed-v4/report.json
```

sealed v4 只做一次无条件报告，不用于回退、调参或重选 winner。最终结论仍由
预注册 v3 门禁决定。

## 11. 何时进入 Base-v3

满足任一条件才进入 Base-v3：

1. B1 未达到第 7.2 节趋势资格，且没有任何 SFT arm 通过；
2. B1 达到趋势资格，但 `5e-5` 敏感性臂仍未通过；
3. 数据或训练链路经修复后可复现，能力仍接近零。

Base-v3 仍需遵守：

- 450M–480M 唯一 train token，按 source_id stable hash + token quota 采样；
- 与约 460M seen token 的 repeat arm 做 compute-matched 对照；
- 固定现有 tokenizer；
- 多域公开非合成中英文数据；
- 从随机初始化训练，不从已衰减到最低 LR 的 Base-v2 接续；
- sealed test 不参与配方选择。

## 12. 停止与回退矩阵

| 条件 | 动作 |
| --- | --- |
| D0 checkpoint 缺失 | 记录缺失，只诊断现存 checkpoint |
| public-60m-v3 hash/数量/长度/泄漏失败 | 停止并修复数据 |
| 单个 SFT arm 失败 | 保留报告，继续矩阵 |
| 全部 SFT arm 失败 | 停止 DPO/GRPO，进入 Base-v3 |
| 单个 DPO arm 失败 | 继续另一个 DPO arm |
| 两个 DPO arm 都失败 | 回退 SFT winner |
| GRPO qualification 失败 | 不运行 GRPO |
| 三个 GRPO arm 都失败 | 回退 GRPO parent |
| 任一阶段仅 objective 改善 | 不视为能力收益 |
| protocol hash 不一致 | 停止比较 |

## 13. 必须回传的产物

```text
runs/evaluations/sft-r1-checkpoint-diagnostics/
runs/qualifications/public-60m-v3-data-check.json
runs/qualifications/sft-v3-lr-sensitivity.json
runs/native-sft-public-60m-r2-*/
runs/evaluations/native-sft-public-60m-r2-*/
runs/gates/native-sft-public-60m-r2-*.json
runs/gates/sft-r2-winner.json
runs/native-dpo-public-60m-r2-*/
runs/gates/dpo-r2-winner.json                     # 仅 DPO 有 winner 时
runs/qualifications/native-grpo-public-60m-r2-parent-s42/
runs/native-grpo-public-60m-r2-*/
runs/gates/grpo-r2-winner.json                    # 仅 GRPO 有 winner 时
runs/evaluations/public-posttrain-r2-v3/
runs/evaluations/public-posttrain-r2-sealed-v4/
```

每个训练 run 至少保留 `resolved_config.yaml`、`run_manifest.json`、
`runtime_environment.json`、`training_budget.json`、`training_result.json`、
`metrics.jsonl`、`initialization.json`、数据 summary 和最终 checkpoint。
结果必须写回实验记录，不能只保留终端输出。
