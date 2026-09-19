# Native-60M 公开数据后训练续跑手册

本文是 `public-60m-v2` 已完成 Base 训练后的唯一执行计划。远程服务器应从本文
**R0** 开始续跑，不重新物化数据，也不重新训练 Base-v1、Base-repeat 或
Base-v2。

历史结果与问题归因见
[Native-60M 公开数据后训练 v2](./experiments/native-60m-public-posttrain-v2.md)。
第一次小数据实验保留为冻结对照，见
[Native-60M 公开数据后训练 v1](./experiments/native-60m-public-posttrain-v1.md)。

## 1. 当前结论与目标

已完成的三个 Base checkpoint：

| 权重 | train token | token/parameter | 状态 |
| --- | ---: | ---: | --- |
| Base-v1 | 46.18M | 0.73 | 明显训练不足，仅作历史锚点 |
| Base-repeat | 369.48M | 5.87 | 同一 v1 语料重复 8 epoch |
| Base-v2 | 510.61M | 8.11 | 扩充公开语料训练 1 epoch |

旧 Base 门禁把不同 held-out 语料上的绝对 loss 直接比较，因此其失败只表示
“未通过旧协议”，不能证明 Base-v2 英文能力退化。下一轮不再使用该门禁，按以下
顺序执行：

```text
R0：三个现有 Base × 两套 dev held-out 交叉评测
  -> R1：Base-repeat / Base-v2 的统一 capability baseline
  -> 两个 large-SFT arm（同数据、同预算、同超参数）
  -> SFT 硬门禁与自动选优
  -> DPO pure / DPO+NLL
  -> GRPO train-only rollout 资格检查
  -> GRPO KL 0.04 / 0.08 / 0.12
  -> 同协议纵向成绩单
```

R0 是诊断，不根据其相对排名停止低成本 SFT。只有命令失败、产物不完整或指标
非数值时停止。R1 的两个 SFT arm 都失败时停止后训练，转入尚未实现的 Base-v3
设计。

## 2. 固定实验契约

- 模型：`configs/models/tiny-60m.yaml`，62.93M 参数。
- Tokenizer：`data/tokenizers/bilingual-60m-v1`，所有实验臂固定。
- seed 42、context 512、BF16、`native-chat-v1`。
- capability suite：`configs/evaluation/lifecycle-v3.yaml`，greedy，
  `max_new_tokens=32`，单线程。
- 所有 R1 capability 评测统一传入
  `data/prepared/bilingual-60m-v2/data_manifest.json`。这保证 Base、SFT、DPO、
  GRPO 的 protocol hash 一致。
- SFT 只跑 Base-repeat 与 Base-v2 两个 large arm，均使用
  `public-60m-v2` 和 2,000,000 assistant supervised token。
- DPO 两臂除 `nll_coefficient` 外一致；GRPO 三臂除 `kl_beta` 外一致。
- SFT/DPO/GRPO 的 source group 交集必须为 0。
- 所有输出目录不可覆盖。修改数据、父 checkpoint、预算或超参数必须使用新的
  recipe/config/run ID。
- DPO 和 GRPO 都是可选收益阶段。DPO 全部失败时回退 SFT winner；GRPO
  qualification 或全部 arm 失败时回退其父模型。

本轮不做以下事情：

- 不再运行 Base-v1 或 small-SFT arm。
- 不把 Base-repeat 与 Base-v2 宣称为 compute-matched；两者分别看过
  369.48M 与 510.61M token。
- 不在 R0 或中途选型时读取 `test` split。旧 v1/v2 test 已被使用，只能视为
  历史诊断数据。
- 不根据旧 test badcase 调低现有 SFT/DPO/GRPO 门禁。
- 不使用合成后训练数据。

## 3. 远程预检

以下命令均从仓库根目录执行。已记录的远程环境是 `/root/.venv60m`：

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

python -m pip check
test -z "$(git status --porcelain=v1)"
git rev-parse HEAD
ruff check .
pytest -q
```

确认本轮代码至少包含交叉评测参数和新增 SFT 配置：

```bash
python scripts/eval_pretrain.py --help | grep -q -- '--data-manifest'
python scripts/eval_pretrain.py --help | grep -q -- '--packed-manifest'
python scripts/eval_pretrain.py --help | grep -q -- '--eval-batches'
test -f configs/pipelines/native-sft-public-60m-v2-repeat-large.yaml
```

### 3.1 验证已有 checkpoint

```bash
BASE_V1=runs/native-60m-v11-001/checkpoints/step-00005649
BASE_REPEAT=runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191
BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453

for checkpoint in "$BASE_V1" "$BASE_REPEAT" "$BASE_V2"; do
  test -f "$checkpoint/checkpoint_metadata.json"
  test -f "$checkpoint/model/model.pt"
  test -f "$checkpoint/model/config.json"
done

jq -e \
  '.run_id == "native-60m-v11-001" and .step == 5649' \
  "$BASE_V1/checkpoint_metadata.json"
jq -e \
  '.run_id == "native-60m-base-repeat-control-v2-s42" and .step == 45191' \
  "$BASE_REPEAT/checkpoint_metadata.json"
jq -e \
  '.run_id == "native-60m-base-expanded-v2-s42" and .step == 62453' \
  "$BASE_V2/checkpoint_metadata.json"

for run_id in \
  native-60m-base-repeat-control-v2-s42 \
  native-60m-base-expanded-v2-s42
do
  jq -e '.status == "completed"' "runs/$run_id/run_manifest.json"
  jq -e '.code.dirty == false' "runs/$run_id/runtime_environment.json"
done
```

### 3.2 验证已有数据

```bash
test -f data/prepared/bilingual-60m-v1/data_manifest.json
test -f data/packed/bilingual-60m-v1-seq512/packed_manifest.json
test -f data/prepared/bilingual-60m-v2/data_manifest.json
test -f data/packed/bilingual-60m-v2-seq512/packed_manifest.json
test -f data/prepared/sft-public-60m-v2/data_manifest.json
test -f data/prepared/dpo-public-60m-v2/data_manifest.json
test -f data/prepared/grpo-public-60m-v2/data_manifest.json

mkdir -p runs/qualifications
DATA_RECHECK=runs/qualifications/public-60m-v2-data-check-r1.json
test ! -e "$DATA_RECHECK"

python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v2/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v2/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v2/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16 \
  --json > "$DATA_RECHECK"

jq -e '
  .ok == true
  and .stage_source_id_overlap == {
    "dpo-grpo": 0,
    "sft-dpo": 0,
    "sft-grpo": 0
  }
  and .sft.splits.train.examples == 10935
  and .sft.splits.train.supervised_tokens == 894771
  and .dpo.splits.train.pairs == 6712
  and .dpo.splits.train.response_tokens == 1514923
  and .grpo.splits.train.prompts == 1288
' "$DATA_RECHECK"

jq -e '
  .ok == false
  and .train_tokens == 510611452
  and ([.checks[] | select(.ok | not) | .name]
       == ["train-token-range"])
' data/prepared/bilingual-60m-v2/qualification.json
```

Base-v2 的 `510,611,452` token 超出旧资格上限 2.12%，是已登记的历史偏差。
本轮只把现有权重作为 exploratory parent，不把该偏差改写成“资格通过”，也不
重新生成数据。

## 4. R0：现有 Base 交叉评测

R0 将三个 checkpoint 分别放到 v1、v2 的 **dev** held-out 上，以 1,024 个固定
batch 评测，共产生 6 个 JSON。`--config` 始终使用 checkpoint 所属 run 的
`resolved_config.yaml`，外部 manifest 只改变评测数据，不改变 checkpoint
身份校验，也不会覆盖 run 内旧报告。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

BASE_V1=runs/native-60m-v11-001/checkpoints/step-00005649
BASE_REPEAT=runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191
BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
R0_DIR=runs/evaluations/base-cross-r0

test ! -e "$R0_DIR"
mkdir -p "$R0_DIR"

eval_cross_dev() {
  name=$1
  config=$2
  checkpoint=$3
  manifest=$4
  packed_manifest=$5

  python scripts/eval_pretrain.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --data-manifest "$manifest" \
    --packed-manifest "$packed_manifest" \
    --eval-batches 1024 \
    --split dev \
    --json > "$R0_DIR/$name.json"

  jq -e '
    .split == "dev"
    and .external_data == true
    and .eval_batches == 1024
    and .sample_count == 1024
    and (.eval_loss > 0 and .eval_loss < 100)
    and (.eval_en_loss > 0 and .eval_en_loss < 100)
    and (.eval_zh_loss > 0 and .eval_zh_loss < 100)
    and (.eval_bits_per_byte > 0 and .eval_bits_per_byte < 100)
  ' "$R0_DIR/$name.json"
}

eval_model() {
  name=$1
  config=$2
  checkpoint=$3

  eval_cross_dev "$name-on-v1" "$config" "$checkpoint" \
    data/prepared/bilingual-60m-v1/data_manifest.json \
    data/packed/bilingual-60m-v1-seq512/packed_manifest.json

  eval_cross_dev "$name-on-v2" "$config" "$checkpoint" \
    data/prepared/bilingual-60m-v2/data_manifest.json \
    data/packed/bilingual-60m-v2-seq512/packed_manifest.json
}

eval_model basev1 \
  runs/native-60m-v11-001/resolved_config.yaml \
  "$BASE_V1"
eval_model repeat \
  runs/native-60m-base-repeat-control-v2-s42/resolved_config.yaml \
  "$BASE_REPEAT"
eval_model basev2 \
  runs/native-60m-base-expanded-v2-s42/resolved_config.yaml \
  "$BASE_V2"

printf '%s\n' \
  $'model\tanchor\tsamples\tloss\ten_loss\tzh_loss\tbpb\tmanifest_sha256' \
  > "$R0_DIR/summary.tsv"

for name in \
  basev1-on-v1 basev1-on-v2 \
  repeat-on-v1 repeat-on-v2 \
  basev2-on-v1 basev2-on-v2
do
  model=${name%-on-*}
  anchor=${name##*-on-}
  jq -r --arg model "$model" --arg anchor "$anchor" '
    [
      $model,
      $anchor,
      .sample_count,
      .eval_loss,
      .eval_en_loss,
      .eval_zh_loss,
      .eval_bits_per_byte,
      .data_manifest_sha256
    ] | @tsv
  ' "$R0_DIR/$name.json" >> "$R0_DIR/summary.tsv"
done

test "$(find "$R0_DIR" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')" -eq 6
cat "$R0_DIR/summary.tsv"
```

解释规则：

| 结果 | 可支持的结论 |
| --- | --- |
| Base-v2 在 v1、v2 dev 都胜过 repeat | Base-v2 是当前更强父模型候选 |
| Base-v2 在 v2 胜、v1 败 | 存在领域 tradeoff，不能称为整体改善或退化 |
| Base-v2 在两套 dev 都败 | 优先怀疑配方或优化日程，但仍执行两个低成本 SFT arm |
| 64-window 与 1,024-window 结论翻转 | 旧 Base 门禁样本不足，不再使用 |

R0 不产生新的硬门禁，也不读取 test。它的作用是补齐同一 held-out 内的可比矩阵，
而不是从已消费的 test 反向拟合阈值。

## 5. R1：两个父模型的统一 capability baseline

两个 parent 都使用同一个 `lifecycle-v3` suite、同一个 v2 pretrain manifest 和
完全相同的生成参数。输出目录必须不存在。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

BASE_REPEAT=runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191
BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json

eval_parent() {
  checkpoint=$1
  output=$2

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
    --pretrain-manifest "$COMMON_PRETRAIN_MANIFEST"
}

eval_parent "$BASE_REPEAT" runs/evaluations/r1-parent-repeat
eval_parent "$BASE_V2" runs/evaluations/r1-parent-basev2

test "$(
  jq -r '.protocol_sha256' \
    runs/evaluations/r1-parent-repeat/report.json \
    runs/evaluations/r1-parent-basev2/report.json |
    sort -u | wc -l | tr -d ' '
)" -eq 1
```

## 6. R1：large-SFT 父模型筛选

### 6.1 对照完整性检查

两份 SFT 配置除父 checkpoint 外必须逐字段一致：

```bash
python - <<'PY'
from copy import deepcopy
from pathlib import Path

import yaml

paths = [
    Path("configs/pipelines/native-sft-public-60m-v2-repeat-large.yaml"),
    Path("configs/pipelines/native-sft-public-60m-v2-basev2-large.yaml"),
]
configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
inits = []
for config in configs:
    value = deepcopy(config)
    inits.append(value["model"].pop("init_checkpoint"))
    config.clear()
    config.update(value)

assert configs[0] == configs[1], "SFT arms differ beyond init_checkpoint"
assert inits[0] != inits[1], "SFT parents must differ"
training = configs[0]["training"]
assert training["max_train_tokens"] == 2_000_000
assert training["learning_rate"] == 2e-5
assert configs[0]["seed"] == 42
print("PASS: SFT configs differ only by parent checkpoint")
PY
```

### 6.2 训练、评测和门禁

门禁返回码 `1` 表示该 arm 未通过，函数会保留报告并继续另一个 arm；返回码 `2`
表示命令或产物错误，立即停止。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate
mkdir -p runs/gates

BASE_REPEAT=runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191
BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json
DATA_CHECK=runs/qualifications/public-60m-v2-data-check-r1.json

run_gate_allow_fail() {
  if "$@"; then
    return 0
  else
    status=$?
  fi
  if test "$status" -eq 1; then
    return 0
  fi
  return "$status"
}

run_sft_arm() {
  config=$1
  run_id=$2
  parent_checkpoint=$3
  parent_report=$4

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$parent_checkpoint"

  python scripts/train_sft.py \
    --config "$config" \
    --init-checkpoint "$parent_checkpoint" \
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
    --baseline "$parent_report"

  run_gate_allow_fail python scripts/check_stage_gate.py sft \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$parent_report" \
    --data-check "$DATA_CHECK" \
    --output "runs/gates/$run_id.json"
}

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-repeat-large.yaml \
  native-sft-public-60m-r1-repeat-large-s42 \
  "$BASE_REPEAT" \
  runs/evaluations/r1-parent-repeat/report.json

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev2-large.yaml \
  native-sft-public-60m-r1-basev2-large-s42 \
  "$BASE_V2" \
  runs/evaluations/r1-parent-basev2/report.json

test "$(
  jq -r '.protocol_sha256' \
    runs/evaluations/r1-parent-repeat/report.json \
    runs/evaluations/r1-parent-basev2/report.json \
    runs/evaluations/native-sft-public-60m-r1-repeat-large-s42/report.json \
    runs/evaluations/native-sft-public-60m-r1-basev2-large-s42/report.json |
    sort -u | wc -l | tr -d ' '
)" -eq 1
```

SFT 硬门禁保持代码中的预注册定义：

- dev objective 相对 step 0 至少下降 5%；
- instruction 至少 4 个成功、format 至少 3 个、QA 至少 4 个；
- verifiable 至少 12 个成功；
- instruction、format、QA 的 en/zh 各至少成功 1 个；
- `corpus.bpb.all` 不超过对应 parent 的 1.02 倍；
- `public-60m-v2` 跨阶段 source group 交集全部为 0。

### 6.3 选择 SFT winner

如果两个 arm 都失败，下面的显式检查终止流程。不得继续 DPO/GRPO，也不得现场
降低门槛；按第 10 节进入 Base-v3 设计。

```bash
set -euo pipefail

SFT_REPEAT_GATE=runs/gates/native-sft-public-60m-r1-repeat-large-s42.json
SFT_BASEV2_GATE=runs/gates/native-sft-public-60m-r1-basev2-large-s42.json

if ! jq -e '.ok == true' "$SFT_REPEAT_GATE" >/dev/null &&
   ! jq -e '.ok == true' "$SFT_BASEV2_GATE" >/dev/null
then
  echo "STOP: both R1 SFT arms failed; do not run DPO or GRPO" >&2
  exit 1
fi

test ! -e runs/gates/sft-r1-winner.json
python scripts/select_stage_winner.py \
  --stage sft \
  --candidate \
    runs/native-sft-public-60m-r1-repeat-large-s42 \
    runs/evaluations/native-sft-public-60m-r1-repeat-large-s42/report.json \
    "$SFT_REPEAT_GATE" \
  --candidate \
    runs/native-sft-public-60m-r1-basev2-large-s42 \
    runs/evaluations/native-sft-public-60m-r1-basev2-large-s42/report.json \
    "$SFT_BASEV2_GATE" \
  --output runs/gates/sft-r1-winner.json

SFT_WINNER_RUN=$(jq -er '.winner.run_dir' runs/gates/sft-r1-winner.json)
SFT_WINNER_CKPT=$(jq -er '.winner.final_checkpoint' \
  runs/gates/sft-r1-winner.json)
SFT_WINNER_REPORT=$(jq -er '.winner.capability_report' \
  runs/gates/sft-r1-winner.json)

case "$SFT_WINNER_RUN" in
  *-repeat-*)
    ACTIVE_BASE_REPORT=runs/evaluations/r1-parent-repeat/report.json
    ;;
  *-basev2-*)
    ACTIVE_BASE_REPORT=runs/evaluations/r1-parent-basev2/report.json
    ;;
  *)
    echo "unknown SFT winner parent: $SFT_WINNER_RUN" >&2
    exit 2
    ;;
esac

printf 'SFT winner: %s\ncheckpoint: %s\n' \
  "$SFT_WINNER_RUN" "$SFT_WINNER_CKPT"
```

winner 排序为：四类任务成功数之和降序、BPB 升序、SFT dev loss 升序。

## 7. DPO：pure / NLL 0.10

两个 DPO arm 都从同一个 SFT winner 初始化。任一 arm 门禁失败仍继续另一 arm；
两者都失败则不生成 DPO winner，并直接用 SFT winner 做 GRPO qualification。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate
mkdir -p runs/gates

SFT_WINNER_CKPT=$(jq -er '.winner.final_checkpoint' \
  runs/gates/sft-r1-winner.json)
SFT_WINNER_REPORT=$(jq -er '.winner.capability_report' \
  runs/gates/sft-r1-winner.json)
COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json
test ! -e runs/gates/dpo-r1-winner.json

run_gate_allow_fail() {
  if "$@"; then
    return 0
  else
    status=$?
  fi
  if test "$status" -eq 1; then
    return 0
  fi
  return "$status"
}

run_dpo_arm() {
  config=$1
  run_id=$2

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$SFT_WINNER_CKPT"

  python scripts/train_dpo.py \
    --config "$config" \
    --init-checkpoint "$SFT_WINNER_CKPT" \
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
    --baseline "$SFT_WINNER_REPORT"

  run_gate_allow_fail python scripts/check_stage_gate.py dpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$SFT_WINNER_REPORT" \
    --output "runs/gates/$run_id.json"
}

run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v2-pure.yaml \
  native-dpo-public-60m-r1-pure-s42

run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v2-nll010.yaml \
  native-dpo-public-60m-r1-nll010-s42

DPO_PURE_GATE=runs/gates/native-dpo-public-60m-r1-pure-s42.json
DPO_NLL_GATE=runs/gates/native-dpo-public-60m-r1-nll010-s42.json

if jq -e '.ok == true' "$DPO_PURE_GATE" >/dev/null ||
   jq -e '.ok == true' "$DPO_NLL_GATE" >/dev/null
then
  python scripts/select_stage_winner.py \
    --stage dpo \
    --candidate \
      runs/native-dpo-public-60m-r1-pure-s42 \
      runs/evaluations/native-dpo-public-60m-r1-pure-s42/report.json \
      "$DPO_PURE_GATE" \
    --candidate \
      runs/native-dpo-public-60m-r1-nll010-s42 \
      runs/evaluations/native-dpo-public-60m-r1-nll010-s42/report.json \
      "$DPO_NLL_GATE" \
    --output runs/gates/dpo-r1-winner.json
else
  echo "Both DPO arms failed; GRPO parent remains the SFT winner"
fi
```

DPO 门禁要求：

- dev preference loss 低于 step 0；
- v2 dev pair 数量不少于 200 时，accuracy 的 95% Wilson 下界大于 0.5；
- 独立 preference 相对 SFT 至少多 3 个成功；
- instruction、format、QA、multiturn、verifiable 各最多下降 1 个 case；
- BPB 不超过 SFT 的 1.02 倍。

## 8. GRPO：资格检查与 KL 扫描

### 8.1 确定父模型并做 train-only qualification

有 DPO winner 时优先从 DPO 开始，否则从 SFT winner 开始。qualification 只读取
train source group，不读取 dev/test，也不更新权重。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

if test -f runs/gates/dpo-r1-winner.json; then
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/dpo-r1-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/dpo-r1-winner.json)
else
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/sft-r1-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/sft-r1-winner.json)
fi

QUALIFICATION_DIR=runs/qualifications/native-grpo-public-60m-r1-parent-s42
test ! -e runs/gates/grpo-r1-winner.json
test ! -e "$QUALIFICATION_DIR"

if python scripts/qualify_grpo.py \
  --config configs/pipelines/native-grpo-public-60m-v2-kl004.yaml \
  --init-checkpoint "$GRPO_PARENT_CKPT" \
  --source-groups 64 \
  --minimum-mixed-group-fraction 0.25 \
  --output "$QUALIFICATION_DIR"
then
  GRPO_QUALIFIED=1
else
  status=$?
  if test "$status" -eq 1; then
    GRPO_QUALIFIED=0
    echo "GRPO qualification failed; skip every GRPO arm"
  else
    exit "$status"
  fi
fi
```

资格要求为：总体 success rate 严格位于 `(0, 1)`；128 个 prompt group 中至少
32 个有混合奖励；en、zh 各至少一个混合奖励 group；verifier parse failure 为
0。

### 8.2 只在 qualification 通过时运行三个 arm

本节从 winner 与 qualification 文件重新恢复状态，可以在新的 shell 中独立执行。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

if test -f runs/gates/dpo-r1-winner.json; then
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/dpo-r1-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/dpo-r1-winner.json)
else
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/sft-r1-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/sft-r1-winner.json)
fi

QUALIFICATION_REPORT=runs/qualifications/native-grpo-public-60m-r1-parent-s42/qualification.json
if jq -e '.ok == true' "$QUALIFICATION_REPORT" >/dev/null; then
  GRPO_QUALIFIED=1
else
  status=$?
  if test "$status" -eq 1; then
    GRPO_QUALIFIED=0
  else
    exit "$status"
  fi
fi

COMMON_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json
mkdir -p runs/gates

run_gate_allow_fail() {
  if "$@"; then
    return 0
  else
    status=$?
  fi
  if test "$status" -eq 1; then
    return 0
  fi
  return "$status"
}

run_grpo_arm() {
  config=$1
  run_id=$2

  test ! -e "runs/$run_id"
  test ! -e "runs/evaluations/$run_id"
  test ! -e "runs/gates/$run_id.json"

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$GRPO_PARENT_CKPT"

  python scripts/train_grpo.py \
    --config "$config" \
    --init-checkpoint "$GRPO_PARENT_CKPT" \
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
    --baseline "$GRPO_PARENT_REPORT"

  run_gate_allow_fail python scripts/check_stage_gate.py grpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$GRPO_PARENT_REPORT" \
    --output "runs/gates/$run_id.json"
}

if test "$GRPO_QUALIFIED" -eq 1; then
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v2-kl004.yaml \
    native-grpo-public-60m-r1-kl004-s42
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v2-kl008.yaml \
    native-grpo-public-60m-r1-kl008-s42
  run_grpo_arm \
    configs/pipelines/native-grpo-public-60m-v2-kl012.yaml \
    native-grpo-public-60m-r1-kl012-s42

  GRPO_004_GATE=runs/gates/native-grpo-public-60m-r1-kl004-s42.json
  GRPO_008_GATE=runs/gates/native-grpo-public-60m-r1-kl008-s42.json
  GRPO_012_GATE=runs/gates/native-grpo-public-60m-r1-kl012-s42.json

  if jq -e '.ok == true' "$GRPO_004_GATE" >/dev/null ||
     jq -e '.ok == true' "$GRPO_008_GATE" >/dev/null ||
     jq -e '.ok == true' "$GRPO_012_GATE" >/dev/null
  then
    python scripts/select_stage_winner.py \
      --stage grpo \
      --candidate \
        runs/native-grpo-public-60m-r1-kl004-s42 \
        runs/evaluations/native-grpo-public-60m-r1-kl004-s42/report.json \
        "$GRPO_004_GATE" \
      --candidate \
        runs/native-grpo-public-60m-r1-kl008-s42 \
        runs/evaluations/native-grpo-public-60m-r1-kl008-s42/report.json \
        "$GRPO_008_GATE" \
      --candidate \
        runs/native-grpo-public-60m-r1-kl012-s42 \
        runs/evaluations/native-grpo-public-60m-r1-kl012-s42/report.json \
        "$GRPO_012_GATE" \
      --output runs/gates/grpo-r1-winner.json
  else
    echo "All GRPO arms failed; keep the GRPO parent"
  fi
fi
```

GRPO 门禁要求 dev reward 相对 step 0 至少提高 0.05、zero-variance group 不超过
0.75、approx KL 不超过 0.10、各任务最多下降 1 case、BPB 不超过父模型 1.02
倍，并且 prompt pass 与 rollout token coverage 均有独立的正数记录。

## 9. 最终同协议成绩单

成绩单固定从 active Base 和 SFT winner 开始，只追加实际通过门禁的 DPO/GRPO
winner。所有报告必须具有同一个 protocol hash。

```bash
set -euo pipefail
source /root/.venv60m/bin/activate

SFT_WINNER_RUN=$(jq -er '.winner.run_dir' runs/gates/sft-r1-winner.json)
SFT_WINNER_REPORT=$(jq -er '.winner.capability_report' \
  runs/gates/sft-r1-winner.json)

case "$SFT_WINNER_RUN" in
  *-repeat-*)
    ACTIVE_BASE_REPORT=runs/evaluations/r1-parent-repeat/report.json
    ;;
  *-basev2-*)
    ACTIVE_BASE_REPORT=runs/evaluations/r1-parent-basev2/report.json
    ;;
  *)
    echo "unknown SFT winner parent: $SFT_WINNER_RUN" >&2
    exit 2
    ;;
esac

REPORTS=(
  "$ACTIVE_BASE_REPORT"
  "$SFT_WINNER_REPORT"
)

if test -f runs/gates/dpo-r1-winner.json; then
  REPORTS+=("$(jq -er '.winner.capability_report' \
    runs/gates/dpo-r1-winner.json)")
fi

if test -f runs/gates/grpo-r1-winner.json; then
  REPORTS+=("$(jq -er '.winner.capability_report' \
    runs/gates/grpo-r1-winner.json)")
fi

test "$(
  jq -r '.protocol_sha256' "${REPORTS[@]}" |
    sort -u | wc -l | tr -d ' '
)" -eq 1

test ! -e runs/evaluations/public-posttrain-r1
python scripts/evaluate_capabilities.py compare \
  --reports "${REPORTS[@]}" \
  --output runs/evaluations/public-posttrain-r1
```

最终模型定义为最后一个通过门禁的 winner；如果 DPO/GRPO 都无收益，SFT winner
就是最终模型。不能用“阶段训练完成”替代“阶段门禁通过”。

## 10. 两个 SFT arm 都失败时：Base-v3 设计

本节是下一轮设计约束，当前仓库尚无对应 recipe/config，**不得把本节改写成临时
命令直接开跑**。

1. 使用新的 recipe ID；唯一 train token 固定为 450M-480M，资格失败即停止。
2. 按 `source_id` 稳定 hash 与 token quota 采样，不按输入文件顺序或 record
   count 猜 token 预算。
3. 使用 revision、文件 hash、许可证固定的多域公开非合成数据；降低单一英文
   Wikipedia 域占比，并为英文和中文同时补充非百科域。
4. 固定现有 tokenizer。若要更换 tokenizer，必须单独建实验臂，不能与数据配方
   收益混在一起。
5. 第一轮同时训练约 460M seen token 的 Base-v3 与 10-epoch Base-v1 repeat，
   形成 compute-matched 对照。
6. 配方选定后从随机初始化训练 2 epoch，约 900M-960M seen token；不得从已经
   衰减到最低学习率的 1-epoch checkpoint 接续。
7. 所有对照臂使用同一优化日程。若 warmup 改为总 step 的约 1%，必须从头训练
   并同步修改全部 arm。
8. `checkpoint_interval` 放粗至约 5,000 step，`eval_interval` 可独立保持约
   1,000 step；开跑前完成磁盘预算。
9. 新建至少 1,024 固定窗口的 dev anchor，报告按语言、来源的 BPB 和置信区间。
   Base loss 只拦截灾难性回退，最终父模型仍由 compute-matched SFT 能力选择。
10. 新建 sealed test anchor。它不参与配方、阈值、早停或 winner 选择，只在最终
    候选冻结后评测一次。

## 11. 停止与回退矩阵

| 条件 | 动作 |
| --- | --- |
| R0 命令失败、产物不完整或指标非数值 | 停止并修复评测链路 |
| R0 仅显示某个 Base 相对较差 | 记录诊断，继续两个 SFT arm |
| 单个 SFT arm 失败 | 保留报告，继续另一个 SFT arm |
| 两个 SFT arm 都失败 | 停止 DPO/GRPO，进入 Base-v3 设计 |
| 单个 DPO arm 失败 | 保留报告，继续另一个 DPO arm |
| 两个 DPO arm 都失败 | 回退 SFT winner，允许 GRPO qualification |
| GRPO qualification 失败 | 不训练任何 GRPO arm |
| 三个 GRPO arm 都失败 | 回退 DPO winner；无 DPO winner 时回退 SFT winner |
| 任一阶段只改善自身 objective | 不视为能力收益 |
| protocol hash 不一致 | 报告不可比较，停止选优 |

## 12. 必须回传的产物

```text
runs/evaluations/base-cross-r0/*.json
runs/evaluations/base-cross-r0/summary.tsv
runs/qualifications/public-60m-v2-data-check-r1.json
runs/evaluations/r1-parent-*/{report.json,report.md}
runs/native-{sft,dpo,grpo}-public-60m-r1-*/
runs/evaluations/native-{sft,dpo,grpo}-public-60m-r1-*/
runs/gates/*-r1-*.json
runs/qualifications/native-grpo-public-60m-r1-parent-s42/qualification.json
runs/evaluations/public-posttrain-r1/{report.json,report.md}
```

每个训练 run 至少保留：

```text
resolved_config.yaml
run_manifest.json
runtime_environment.json
training_budget.json
training_result.json
metrics.jsonl
initialization.json
*_data_summary.json
```

GRPO 额外保留 `grpo_budget_summary.json`。checkpoint、optimizer state 和 packed
arrays 可以留在远端，但必须记录绝对存储位置和 SHA-256。最终结果写回
`docs/experiments/native-60m-public-posttrain-v2.md`，不得只保留终端输出。
