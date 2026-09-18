# Native-60M 公开数据训练与后训练执行手册

本文给出 Native-60M 下一轮 Base -> SFT -> DPO -> GRPO 实验的完整执行顺序。
每个阶段先训练、再评测、最后运行硬门禁；门禁失败立即停止，不能用下游训练掩盖
上游能力缺失。

第一次 `public-60m-v1` 实验保留为冻结对照，结果见
[公开数据 60M 后训练 v1](./experiments/native-60m-public-posttrain-v1.md)。
本轮只使用固定 revision、上游文件 SHA-256 和许可证明确的公开数据。

## 1. 本轮回答什么

v1 已确认：

| 阶段 | 冻结事实 | 结论 |
| --- | --- | --- |
| Base-v1 | 62.93M 参数只看过 46.18M train token，即 0.73 token/param | Base 明显训练不足 |
| SFT-v1 | 883 条 train；核心 capability 指标全为 0 | 小数据只拟合文本，未建立任务能力 |
| DPO-v1 | 448 对 train；独立 preference 只多 1/32 | 没有可靠偏好收益 |
| GRPO-v1 | 绝大多数 group 奖励零方差，KL 漂移 | 父模型不具备可优化信号 |

本轮按以下顺序执行：

```text
CPU 数据物化和资格检查
  -> Base-repeat 计算量对照
  -> Base-v2 唯一语料扩容
  -> Base x SFT 数据规模四臂
  -> SFT 硬门禁与自动选优
  -> DPO pure / DPO+NLL
  -> GRPO train-only rollout 资格检查
  -> KL 0.04 / 0.08 / 0.12
  -> 同协议纵向评测
```

DPO 和 GRPO 不是必经阶段。DPO 全部失败时保留 SFT winner；GRPO 可以直接从
通过门禁的 SFT 开始。GRPO 资格检查失败时不得启动 GRPO 训练。

## 2. 固定实验契约

- 模型：`configs/models/tiny-60m.yaml`，62.93M 参数。
- Tokenizer：`data/tokenizers/bilingual-60m-v1`，所有实验臂固定不变。
- seed 42、context 512、BF16、`native-chat-v1`。
- capability suite：`configs/evaluation/lifecycle-v3.yaml`，greedy，
  `max_new_tokens=32`，单线程。
- 数据按 `source_id` 分组切分；SFT/DPO/GRPO source group 交集必须为 0。
- Base-v2 必须有 3-5 亿唯一 train token；en/zh 各 40%-60%；任一来源不超过
  50%；synthetic token 不超过 30%。
- SFT 四臂统一 2,000,000 assistant supervised token。
- DPO 两臂除 `nll_coefficient` 外完全一致。
- GRPO 三臂除 `kl_beta` 外完全一致。
- 输出目录必须预先不存在。修改数据、父 checkpoint、预算或超参数必须使用新
  run ID。

阶段产物至少保留：

```text
resolved_config.yaml
run_manifest.json
runtime_environment.json
training_budget.json
training_result.json
metrics.jsonl
initialization.json
*_data_summary.json
evaluations/*/report.json
evaluations/*/report.md
```

GRPO 还必须保留 `grpo_budget_summary.json`，其中 prompt pass 与 rollout token
coverage 分开记录。

## 3. 环境与代码预检

以下命令均从仓库根目录执行：

```bash
set -euo pipefail
conda activate llm-lifecycle-lab
python -m pip check
git status --porcelain=v1
git rev-parse HEAD
python scripts/data.py recipes
python scripts/data.py posttrain-recipes
pytest -q
ruff check .
```

正式训练使用的 commit 必须包含本文、v2 数据 recipes、11 份 v2 pipeline、
`qualify_pretrain_data.py`、`qualify_grpo.py`、`check_stage_gate.py` 和
`select_stage_winner.py`。`git status --porcelain=v1` 必须无输出。

## 4. 物化 Base-v2 数据

### 4.1 固定来源

| recipe | 数据 | 固定文件 | 上游 SHA-256 | 记录上限 | 许可 |
| --- | --- | --- | --- | ---: | --- |
| `wikipedia-en-primary-60m-v2` | Wikimedia English | `20231101.en/train-00000-of-00041.parquet` | `382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677` | 150,000 | CC-BY-SA-3.0 |
| `wikipedia-en-secondary-60m-v2` | Wikimedia English | `20231101.en/train-00019-of-00041.parquet` | `99c23125b0dd115d68f4a921376880e307bdfcbf3bfd66ae4ed90e853e1dadac` | 150,000 | CC-BY-SA-3.0 |
| `wikipedia-zh-60m-v2` | Wikimedia Chinese | `20231101.zh/train-00000-of-00006.parquet` | `853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56` | 220,000 | CC-BY-SA-3.0 |
| `simplestories-60m-v1` | SimpleStories | `data/train-00000-of-00007.parquet` | `ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4` | 100,000 | MIT |

Wikimedia 三个分片固定 revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`；SimpleStories 固定 revision
`e63b8adc3b1a1bdc7cac5b500d150b71346b0628`。

前三个来源是非合成百科语料；SimpleStories 提供叙事覆盖。配方使用 round-robin
合并，并把 `source_recipe_id` 写入每条记录。最终资格以固定 tokenizer 的真实
token 计数为准，不以压缩文件大小估算。

### 4.2 下载、混合、切分、资格检查和 packing

```bash
python scripts/data.py fetch \
  --recipe wikipedia-en-primary-60m-v2 \
  --output data/raw/wikipedia-en-primary-60m-v2 \
  --accept-license CC-BY-SA-3.0

python scripts/data.py fetch \
  --recipe wikipedia-en-secondary-60m-v2 \
  --output data/raw/wikipedia-en-secondary-60m-v2 \
  --accept-license CC-BY-SA-3.0

python scripts/data.py fetch \
  --recipe wikipedia-zh-60m-v2 \
  --output data/raw/wikipedia-zh-60m-v2 \
  --accept-license CC-BY-SA-3.0

python scripts/data.py fetch \
  --recipe simplestories-60m-v1 \
  --output data/raw/simplestories-60m-v1 \
  --accept-license MIT

python scripts/data.py mix \
  --mixture bilingual-60m-v2 \
  --input data/raw/wikipedia-en-primary-60m-v2/source.jsonl \
  --input data/raw/wikipedia-zh-60m-v2/source.jsonl \
  --input data/raw/wikipedia-en-secondary-60m-v2/source.jsonl \
  --input data/raw/simplestories-60m-v1/source.jsonl \
  --output data/raw/bilingual-60m-v2

python scripts/data.py prepare \
  --input data/raw/bilingual-60m-v2/source.jsonl \
  --output data/prepared/bilingual-60m-v2 \
  --dataset-id bilingual-60m-v2 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --seed 42 \
  --group-by source_id

python scripts/qualify_pretrain_data.py \
  --manifest data/prepared/bilingual-60m-v2/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output data/prepared/bilingual-60m-v2/qualification.json

jq -e '.ok == true' \
  data/prepared/bilingual-60m-v2/qualification.json

python scripts/data.py pack \
  --manifest data/prepared/bilingual-60m-v2/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output data/packed/bilingual-60m-v2-seq512 \
  --sequence-length 512
```

`qualification.json` 是开跑前硬门禁。它逐条重新 tokenize train split，并记录
数据 manifest hash、tokenizer hash、各语言、各来源和 synthetic token 比例。
任一阈值失败时先调整新 recipe ID，不训练 Base-v2。

## 5. 物化公开后训练 v2 数据

### 5.1 固定数据量

`public-60m-v2` 的规范化输出已冻结：

| 阶段 | 公开来源 | 总记录 | 规范化输出 SHA-256 |
| --- | --- | ---: | --- |
| SFT | OASST1、Dolly-15K、HC3-Chinese、MSVAMP warmup | 13,600 | `e71bc675698cf72346c2669c9f5cb1003e27414d818ce2dc4ab25a2b1a766063` |
| DPO | HelpSteer3、HH-RLHF helpful、CValues helpful | 8,400 | `07333995bd3df1b38cb5d4bdfd5377ec4920162813bffd344db56caac35a30cb` |
| GRPO | 与 SFT 隔离的 MSVAMP source groups | 1,600 | `505f8bdf2d4a3ac008174382e8e134d61c0d2bd29d3c7cedc443c45dd5dbfa9e` |

SFT en/zh 各 6,800；DPO en/zh 各 4,200；GRPO en/zh 各 800。详细 revision、
上游文件 hash、筛选规则、来源计数和任务计数由
`public-60m-v2.yaml` 与生成的 `source_manifest.json` 共同记录。

seed 42、按 `source_id` 切分后的冻结规模为：

| 阶段 | train | dev | test | train 目标 token |
| --- | ---: | ---: | ---: | ---: |
| SFT | 10,935 | 1,283 | 1,382 | 894,771 assistant token/epoch |
| DPO | 6,712 | 844 | 844 | 1,514,923 response token/epoch |
| GRPO | 1,288 prompt / 644 group | 150 / 75 | 162 / 81 | 1 prompt pass |

### 5.2 下载与切分

```bash
python scripts/data.py fetch-posttrain \
  --recipe public-60m-v2 \
  --output data/raw/public-60m-v2 \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-4.0 \
  --accept-license CC-BY-SA-3.0 \
  --accept-license CC-BY-SA-4.0 \
  --accept-license MIT

python scripts/data.py prepare \
  --input data/raw/public-60m-v2/sft/source.jsonl \
  --output data/prepared/sft-public-60m-v2 \
  --dataset-id sft-public-60m-v2 \
  --kind sft \
  --license "Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0" \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v2/dpo/source.jsonl \
  --output data/prepared/dpo-public-60m-v2 \
  --dataset-id dpo-public-60m-v2 \
  --kind dpo \
  --license "Apache-2.0 AND CC-BY-4.0 AND MIT" \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v2/grpo/source.jsonl \
  --output data/prepared/grpo-public-60m-v2 \
  --dataset-id grpo-public-60m-v2 \
  --kind grpo \
  --license Apache-2.0 \
  --seed 42 \
  --group-by source_id

python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v2/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v2/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v2/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16 \
  --json > data/prepared/public-60m-v2-data-check.json

jq -e '.ok == true' data/prepared/public-60m-v2-data-check.json
```

四臂实验还需要冻结的 v1 small SFT 数据和 v1 三阶段 data check。若远端尚未
物化，按本文附录 A 执行一次。

## 6. Phase 1：Base 对照和 Base-v2

### 6.1 计算量对照

该臂从随机初始化开始，复用 Base-v1 的 46,184,530 train token，只训练 8 个
epoch。它回答“旧 Base 是否主要缺计算”，不能作为唯一语料扩容结果。

```bash
python scripts/doctor.py \
  --config configs/pipelines/native-60m-base-repeat-control-v2.yaml

python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-repeat-control-v2.yaml \
  --run-id native-60m-base-repeat-control-v2-s42
```

### 6.2 正式 Base-v2

```bash
python scripts/doctor.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml

python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --run-id native-60m-base-expanded-v2-s42

BASE_V2_CKPT=$(jq -er '.final_checkpoint' \
  runs/native-60m-base-expanded-v2-s42/training_result.json)

mkdir -p runs/evaluations
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --checkpoint "$BASE_V2_CKPT" \
  --split dev \
  --json > runs/evaluations/base-v2-pretrain-dev.json

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --checkpoint "$BASE_V2_CKPT" \
  --split test \
  --json > runs/evaluations/base-v2-pretrain-test.json

python scripts/check_stage_gate.py base \
  --run-dir runs/native-60m-base-expanded-v2-s42 \
  --dev-evaluation runs/evaluations/base-v2-pretrain-dev.json \
  --test-evaluation runs/evaluations/base-v2-pretrain-test.json \
  --baseline-test-evaluation \
    docs/experiments/results/native-60m-baseline-v1/evaluations/pretrain-test-step-00005649.json \
  --output runs/gates/base-v2.json
```

Base 门禁固定为：

- `target_token_coverage` 在 `[1.00, 1.01]`；
- overall/en/zh dev loss 分别不高于 2.9480、2.0119、3.8698；
- overall/en/zh test loss 均不得比冻结 Base-v1 test 恶化超过 1%；
- run 完成且 runtime provenance 为 clean Git。

Base 门禁失败时不运行 SFT 四臂。

## 7. Phase 2：Base x SFT 四臂

### 7.1 先生成两个 Base capability baseline

```bash
BASE_V1_CKPT=$(jq -er '.final_checkpoint' \
  runs/native-60m-baseline-v1/training_result.json)
BASE_V2_CKPT=$(jq -er '.final_checkpoint' \
  runs/native-60m-base-expanded-v2-s42/training_result.json)

python scripts/evaluate_capabilities.py run \
  --checkpoint "$BASE_V1_CKPT" \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-v2-basev1 \
  --device cuda --threads 1 --max-new-tokens 32 \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json

python scripts/evaluate_capabilities.py run \
  --checkpoint "$BASE_V2_CKPT" \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-v2-basev2 \
  --device cuda --threads 1 --max-new-tokens 32 \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v2/data_manifest.json
```

### 7.2 统一执行函数

```bash
run_sft_arm() {
  config=$1
  run_id=$2
  parent_ckpt=$3
  parent_report=$4
  pretrain_manifest=$5
  data_check=$6

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$parent_ckpt"
  python scripts/train_sft.py \
    --config "$config" \
    --init-checkpoint "$parent_ckpt" \
    --run-id "$run_id"

  ckpt=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$ckpt" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$pretrain_manifest" \
    --baseline "$parent_report"

  python scripts/check_stage_gate.py sft \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$parent_report" \
    --data-check "$data_check" \
    --output "runs/gates/$run_id.json"
}
```

执行 A/B/C/D：

```bash
run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev1-small.yaml \
  native-sft-v2-basev1-small-s42 \
  "$BASE_V1_CKPT" \
  runs/evaluations/public-v2-basev1/report.json \
  data/prepared/bilingual-60m-v1/data_manifest.json \
  data/prepared/public-60m-v1-data-check.json

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev2-small.yaml \
  native-sft-v2-basev2-small-s42 \
  "$BASE_V2_CKPT" \
  runs/evaluations/public-v2-basev2/report.json \
  data/prepared/bilingual-60m-v2/data_manifest.json \
  data/prepared/public-60m-v1-data-check.json

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev1-large.yaml \
  native-sft-v2-basev1-large-s42 \
  "$BASE_V1_CKPT" \
  runs/evaluations/public-v2-basev1/report.json \
  data/prepared/bilingual-60m-v1/data_manifest.json \
  data/prepared/public-60m-v2-data-check.json

run_sft_arm \
  configs/pipelines/native-sft-public-60m-v2-basev2-large.yaml \
  native-sft-v2-basev2-large-s42 \
  "$BASE_V2_CKPT" \
  runs/evaluations/public-v2-basev2/report.json \
  data/prepared/bilingual-60m-v2/data_manifest.json \
  data/prepared/public-60m-v2-data-check.json
```

SFT 硬门禁要求 dev objective 至少下降 5%，instruction 至少 4/32、format 至少
3/24、QA 至少 4/32、verifiable 至少 12/112，三个任务的 en/zh 各至少命中
1 题，且 `corpus.bpb.all` 不超过父 Base 的 1.02 倍。

只从通过门禁的 arm 自动选择 winner：

```bash
python scripts/select_stage_winner.py \
  --stage sft \
  --candidate runs/native-sft-v2-basev1-small-s42 \
    runs/evaluations/native-sft-v2-basev1-small-s42/report.json \
    runs/gates/native-sft-v2-basev1-small-s42.json \
  --candidate runs/native-sft-v2-basev2-small-s42 \
    runs/evaluations/native-sft-v2-basev2-small-s42/report.json \
    runs/gates/native-sft-v2-basev2-small-s42.json \
  --candidate runs/native-sft-v2-basev1-large-s42 \
    runs/evaluations/native-sft-v2-basev1-large-s42/report.json \
    runs/gates/native-sft-v2-basev1-large-s42.json \
  --candidate runs/native-sft-v2-basev2-large-s42 \
    runs/evaluations/native-sft-v2-basev2-large-s42/report.json \
    runs/gates/native-sft-v2-basev2-large-s42.json \
  --output runs/gates/sft-winner.json

SFT_WINNER_CKPT=$(jq -er '.winner.final_checkpoint' \
  runs/gates/sft-winner.json)
SFT_WINNER_REPORT=$(jq -er '.winner.capability_report' \
  runs/gates/sft-winner.json)
SFT_WINNER_RUN=$(jq -er '.winner.run_dir' runs/gates/sft-winner.json)

case "$SFT_WINNER_RUN" in
  *basev1*)
    ACTIVE_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v1/data_manifest.json
    ACTIVE_BASE_REPORT=runs/evaluations/public-v2-basev1/report.json
    ;;
  *basev2*)
    ACTIVE_PRETRAIN_MANIFEST=data/prepared/bilingual-60m-v2/data_manifest.json
    ACTIVE_BASE_REPORT=runs/evaluations/public-v2-basev2/report.json
    ;;
  *) exit 2 ;;
esac
```

选择顺序预注册为：四个任务成功数之和降序、BPB 升序、SFT dev loss 升序。

## 8. Phase 3：DPO pure / DPO+NLL

```bash
run_dpo_arm() {
  config=$1
  run_id=$2

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$SFT_WINNER_CKPT"
  python scripts/train_dpo.py \
    --config "$config" \
    --init-checkpoint "$SFT_WINNER_CKPT" \
    --run-id "$run_id"

  ckpt=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$ckpt" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$ACTIVE_PRETRAIN_MANIFEST" \
    --baseline "$SFT_WINNER_REPORT"

  python scripts/check_stage_gate.py dpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$SFT_WINNER_REPORT" \
    --output "runs/gates/$run_id.json"
}

run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v2-pure.yaml \
  native-dpo-v2-pure-s42

run_dpo_arm \
  configs/pipelines/native-dpo-public-60m-v2-nll010.yaml \
  native-dpo-v2-nll010-s42
```

DPO 门禁要求 dev preference loss 低于 step 0；v2 dev pair 的 95% Wilson 下界
大于 0.5；独立 preference 相对 SFT 至少多 3/32；各任务最多下降 1 case；
BPB 不超过 SFT 的 1.02 倍。

```bash
if python scripts/select_stage_winner.py \
  --stage dpo \
  --candidate runs/native-dpo-v2-pure-s42 \
    runs/evaluations/native-dpo-v2-pure-s42/report.json \
    runs/gates/native-dpo-v2-pure-s42.json \
  --candidate runs/native-dpo-v2-nll010-s42 \
    runs/evaluations/native-dpo-v2-nll010-s42/report.json \
    runs/gates/native-dpo-v2-nll010-s42.json \
  --output runs/gates/dpo-winner.json
then
  GRPO_PARENT_CKPT=$(jq -er '.winner.final_checkpoint' \
    runs/gates/dpo-winner.json)
  GRPO_PARENT_REPORT=$(jq -er '.winner.capability_report' \
    runs/gates/dpo-winner.json)
else
  GRPO_PARENT_CKPT=$SFT_WINNER_CKPT
  GRPO_PARENT_REPORT=$SFT_WINNER_REPORT
fi
```

## 9. Phase 4：GRPO 资格检查和 KL 扫描

### 9.1 train-only rollout 门禁

```bash
python scripts/qualify_grpo.py \
  --config configs/pipelines/native-grpo-public-60m-v2-kl004.yaml \
  --init-checkpoint "$GRPO_PARENT_CKPT" \
  --source-groups 64 \
  --minimum-mixed-group-fraction 0.25 \
  --output runs/qualifications/native-grpo-v2-parent-s42

jq -e '.ok == true' \
  runs/qualifications/native-grpo-v2-parent-s42/qualification.json
```

工具按 `sha256(source_id)` 固定选择 64 个 train source group，每组取 en/zh 两个
prompt，共 128 个 prompt；每个 prompt 采样 8 次。它复用正式 GRPO rollout 和
reward 实现，不读取 dev/test 样本、不构造 optimizer、不修改 checkpoint。

资格门禁要求：

- `0 < success_rate < 1`；
- mixed-reward prompt group 至少 32/128；
- en、zh 各至少一个 mixed-reward group；
- verifier parse failure 为 0。

### 9.2 三个 KL 臂

```bash
run_grpo_arm() {
  config=$1
  run_id=$2

  python scripts/doctor.py \
    --config "$config" \
    --init-checkpoint "$GRPO_PARENT_CKPT"
  python scripts/train_grpo.py \
    --config "$config" \
    --init-checkpoint "$GRPO_PARENT_CKPT" \
    --run-id "$run_id"

  ckpt=$(jq -er '.final_checkpoint' "runs/$run_id/training_result.json")
  python scripts/evaluate_capabilities.py run \
    --checkpoint "$ckpt" \
    --tokenizer data/tokenizers/bilingual-60m-v1 \
    --suite configs/evaluation/lifecycle-v3.yaml \
    --output "runs/evaluations/$run_id" \
    --device cuda --threads 1 --max-new-tokens 32 \
    --prompt-protocol native-chat-v1 \
    --pretrain-manifest "$ACTIVE_PRETRAIN_MANIFEST" \
    --baseline "$GRPO_PARENT_REPORT"

  python scripts/check_stage_gate.py grpo \
    --run-dir "runs/$run_id" \
    --capability-report "runs/evaluations/$run_id/report.json" \
    --parent-capability-report "$GRPO_PARENT_REPORT" \
    --output "runs/gates/$run_id.json"
}

run_grpo_arm \
  configs/pipelines/native-grpo-public-60m-v2-kl004.yaml \
  native-grpo-v2-kl004-s42
run_grpo_arm \
  configs/pipelines/native-grpo-public-60m-v2-kl008.yaml \
  native-grpo-v2-kl008-s42
run_grpo_arm \
  configs/pipelines/native-grpo-public-60m-v2-kl012.yaml \
  native-grpo-v2-kl012-s42

python scripts/select_stage_winner.py \
  --stage grpo \
  --candidate runs/native-grpo-v2-kl004-s42 \
    runs/evaluations/native-grpo-v2-kl004-s42/report.json \
    runs/gates/native-grpo-v2-kl004-s42.json \
  --candidate runs/native-grpo-v2-kl008-s42 \
    runs/evaluations/native-grpo-v2-kl008-s42/report.json \
    runs/gates/native-grpo-v2-kl008-s42.json \
  --candidate runs/native-grpo-v2-kl012-s42 \
    runs/evaluations/native-grpo-v2-kl012-s42/report.json \
    runs/gates/native-grpo-v2-kl012-s42.json \
  --output runs/gates/grpo-winner.json
```

GRPO 门禁要求 dev reward 相对 step 0 至少提高 0.05、zero-variance group 不超过
0.75、KL 不超过 0.10、各任务最多下降 1 case、BPB 不超过父模型 1.02 倍，并且
prompt pass 与 rollout token coverage 均已记录。

## 10. 最终纵向成绩单

只把实际通过门禁的阶段加入比较。SFT 一定存在；DPO/GRPO 报告按 winner 文件是否
存在追加：

```bash
REPORTS=(
  "$ACTIVE_BASE_REPORT"
  "$SFT_WINNER_REPORT"
)

if test -f runs/gates/dpo-winner.json; then
  REPORTS+=("$(jq -er '.winner.capability_report' runs/gates/dpo-winner.json)")
fi

if test -f runs/gates/grpo-winner.json; then
  REPORTS+=("$(jq -er '.winner.capability_report' runs/gates/grpo-winner.json)")
fi

python scripts/evaluate_capabilities.py compare \
  --reports "${REPORTS[@]}" \
  --output runs/evaluations/public-posttrain-v2
```

每个阶段的 `report.json`、`report.md`、gate JSON、winner JSON 和训练小文件均需
回传。checkpoint、optimizer state 和 packed arrays 留在远端，并记录存储位置与
SHA-256。

## 11. 停止条件

| 条件 | 动作 |
| --- | --- |
| Base-v2 数据资格失败 | 修改新 recipe，不训练 |
| Base-v2 硬门禁失败 | 不跑 SFT 四臂 |
| 四个 SFT arm 均失败 | 停止，扩充或重配 SFT |
| DPO 两臂均失败 | 回退 SFT winner，允许做 GRPO 资格检查 |
| GRPO qualification 失败 | 不训练 GRPO |
| GRPO 三臂均失败 | 最终模型回退 SFT 或 DPO winner |
| 任一阶段只改善自身 loss | 不视为能力收益 |

首轮是单 seed 工程决策实验。若要声称稳定收益，对最终链路追加至少两个 seed，
但不得根据首轮 test badcase 修改既有门禁。

## 附录 A：补齐冻结的 v1 后训练数据

```bash
python scripts/data.py fetch-posttrain \
  --recipe public-60m-v1 \
  --output data/raw/public-60m-v1 \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-4.0

python scripts/data.py prepare \
  --input data/raw/public-60m-v1/sft/source.jsonl \
  --output data/prepared/sft-public-60m-v1 \
  --dataset-id sft-public-60m-v1 \
  --kind sft --license Apache-2.0 --seed 42 --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v1/dpo/source.jsonl \
  --output data/prepared/dpo-public-60m-v1 \
  --dataset-id dpo-public-60m-v1 \
  --kind dpo --license CC-BY-4.0 --seed 42 --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v1/grpo/source.jsonl \
  --output data/prepared/grpo-public-60m-v1 \
  --dataset-id grpo-public-60m-v1 \
  --kind grpo --license Apache-2.0 --seed 42 --group-by source_id

python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v1/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v1/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16 \
  --json > data/prepared/public-60m-v1-data-check.json
```

若这些目录已由冻结实验物化，直接运行最后一条检查命令，不覆盖现有目录。
