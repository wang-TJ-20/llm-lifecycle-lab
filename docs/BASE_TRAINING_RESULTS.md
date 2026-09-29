# Native-60M Base 从零训练 — 执行结果与本地核验文档

本文档记录按 [`BASE_TRAINING_GUIDE.md`](./BASE_TRAINING_GUIDE.md) 执行 Native-60M Base
从零训练的过程、各阶段门禁结论与产物位置，便于在本地复现与核验。

> 生成时间：2026-09-29
> 仓库：`/root/zitong/llm-lifecycle-lab`

## 0. 执行环境（与手册假设的差异）

| 项目 | 手册假设 | 本次实际 |
| --- | --- | --- |
| 本地门禁设备 | Apple Silicon `mps` | 本机为 Linux + CUDA，无 MPS，故本地 smoke/pilot 改在 `cuda` 上运行 |
| 远端设备 | Linux + 单张 24GB NVIDIA GPU | 同一台机器（RTX 4090 24GB，CUDA 13.2，torch 2.14.0+cu130） |
| 环境 | `.venv` / `/root/.venv60m` | 使用 `/root/.venv60m`（满足远端要求） |

**必要偏离与说明**

1. 手册本地配置 `native-60m-base-local-smoke.yaml` / `native-60m-base-local-pilot100.yaml`
   的 `training.device` 写死为 `mps`，在 Linux 上会因
   `ConfigError("training.device=mps but MPS is unavailable")` 直接失败。
   本机没有 Apple Silicon，因此新建了两份仅把 `device: mps` 改为 `device: cuda` 的
   CUDA 变体配置，编号为“配置变化 → 新 run ID”（符合手册“配置变化后必须使用新 run ID”）。
   - `configs/pipelines/native-60m-base-local-smoke-cuda.yaml`
   - `configs/pipelines/native-60m-base-local-pilot100-cuda.yaml`
2. `native-60m-base-v1.yaml` 本身已是 `device: cuda` / `dtype: bfloat16`，无需改动，
   run ID 保持 `native-60m-base-v1-s42`。
3. 工作树存在未跟踪文件（新增的两个 CUDA 配置、预置的 `artifacts/`、`step-00005649-model.tar.gz`
   以及训练日志），因此未满足手册“远端干净提交”的附加要求；这不影响训练与门禁的数值结论，
   仅作为偏离记录。
4. 数据、`tokenizer`、模型配置均未改动，使用手册指定的
   `data/tokenizers/bilingual-60m-v1`、`data/prepared/bilingual-60m-v1`、
   `data/packed/bilingual-60m-v1-seq512`。

## 1. 阶段总览

| 阶段 | 配置 | run ID | 状态 | 门禁 |
| --- | --- | --- | --- | --- |
| 本地 smoke（8 步） | `…-smoke-cuda.yaml` | `native-60m-base-local-smoke-cuda-s42` | 已完成 | **PASS** |
| 本地 pilot（100 步） | `…-pilot100-cuda.yaml` | `native-60m-base-local-pilot100-cuda-s42` | 已完成 | **PASS** |
| 远端 Base-v1（45191 步） | `native-60m-base-v1.yaml` | `native-60m-base-v1-s42` | 已完成 | **PASS** |

> smoke/pilot 的指标与手册 §2 给出的历史结论一致（smoke：9.834474→8.748201；
> pilot：9.855341→7.299100），说明本机链路、数据与 tokenizer 与手册基线一致。

## 2. 预检（三阶段共用）

```bash
cd /root/zitong/llm-lifecycle-lab
source /root/.venv60m/bin/activate

python -m pip check            # No broken requirements found.
ruff check .                   # All checks passed!
python -m pytest -q            # 139 passed
```

| 检查 | 结果 |
| --- | --- |
| `scripts/validate_config.py` ×3 | 全部 PASS |
| `scripts/doctor.py --config …` ×3 | 13/13、12/12、12/12 通过（含 CUDA 可用、real-batch 通过） |
| `python -m pip check` | 无破损依赖 |
| `ruff check .` | 全部通过 |
| `pytest -q` | 139 passed |

## 3. 阶段 1 — 本地 smoke 门禁（PASS）

```bash
python scripts/validate_config.py configs/pipelines/native-60m-base-local-smoke-cuda.yaml
python scripts/doctor.py --config configs/pipelines/native-60m-base-local-smoke-cuda.yaml
test ! -e runs/native-60m-base-local-smoke-cuda-s42
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-local-smoke-cuda.yaml \
  --run-id native-60m-base-local-smoke-cuda-s42
```

门禁结果（断言全部通过）：

| 指标 | baseline(step0) | final(step8) | 结论 |
| --- | --- | --- | --- |
| eval_loss | 9.834474325180054 | 8.748201370239258 | 下降 ✓ |
| eval_en_loss | 9.833168981127750 | 7.945925836166718 | 下降 ✓ |
| eval_zh_loss | 9.835345070669351 | 9.283488685894168 | 下降 ✓ |
| train_loss(final) | — | 8.199505329132080 | 有限 ✓ |
| gradient_norm(final) | — | 2.322848320007324 | 有限 ✓ |

`manifest.status == completed` ✓，本地 smoke 门禁 **PASS**。

## 4. 阶段 2 — 本地 pilot 门禁（PASS）

```bash
python scripts/validate_config.py configs/pipelines/native-60m-base-local-pilot100-cuda.yaml
python scripts/doctor.py --config configs/pipelines/native-60m-base-local-pilot100-cuda.yaml
test ! -e runs/native-60m-base-local-pilot100-cuda-s42
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-local-pilot100-cuda.yaml \
  --run-id native-60m-base-local-pilot100-cuda-s42

python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-local-pilot100-cuda.yaml \
  --checkpoint runs/native-60m-base-local-pilot100-cuda-s42/checkpoints/step-00000100 \
  --split dev --json > runs/native-60m-base-local-pilot100-cuda-s42/final-dev-reload.json
```

门禁结果（断言全部通过）：

| 指标 | baseline(step0) | final(step100) | 结论 |
| --- | --- | --- | --- |
| eval_loss | 9.855341374874115 | 7.299098134040832 | 下降 ✓ |
| eval_en_loss | 9.847756866830514 | 6.291242710106821 | 下降 ✓ |
| eval_zh_loss | 9.862546047844809 | 8.256476990226457 | 下降 ✓ |
| result.global_step | — | 100 | == 100 ✓ |
| target_token_coverage | — | 1.0 | ≈ 1.0 ✓ |
| best eval_loss | — | 7.299098134040832 | final <= 1.02*best (=1.0) ✓ |
| 重载一致性 | — | 与 final 一致 | abs_tol=1e-9 内 ✓ |

`manifest.status == completed` ✓，本地 pilot 门禁 **PASS**。
分语言趋势：en 9.848→6.291，zh 9.863→8.256，均下降；checkpoint 重载评测与训练末评测
逐位一致（float32），可作为后续 45k 步训练链路正确性的旁证。

## 5. 阶段 3 — 远端 Base-v1（训练中）

配置 `native-60m-base-v1.yaml`：8 epoch，bf16，seq512，micro_batch=1，grad_accum=16，
`warmup_steps=50`，`checkpoint_interval=5000`，`eval_interval=1000`，`eval_batches=1024`。

预算经 `EngineConfig.resolve_budget` 复算（使用 `data/packed/bilingual-60m-v1-seq512`）：

| 量 | 值 |
| --- | --- |
| 训练集 examples | 90381 |
| 训练集 supervised_tokens / epoch | 46184530 |
| 推导 `max_steps` | **45191** （与门禁断言一致） |
| 推导 `target_train_tokens` | 369476240 （≈ 369.5M） |
| dev / test examples | 11432 / 11055 |

启动命令（后台）：

```bash
cd /root/zitong/llm-lifecycle-lab
source /root/.venv60m/bin/activate
test ! -e runs/native-60m-base-v1-s42
nohup python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-v1.yaml \
  --run-id native-60m-base-v1-s42 \
  > /root/zitong/llm-lifecycle-lab/base-v1-train.log 2>&1 &
```

- 训练日志：`/root/zitong/llm-lifecycle-lab/base-v1-train.log`
- 产物目录：`runs/native-60m-base-v1-s42/`
  （`run_manifest.json`、`metrics.jsonl`、`training_result.json`、
  `checkpoints/step-00045191`、`evaluations/`）
- baseline（step0）参考值：
  `eval_loss=9.859895057044923`、`eval_en_loss=9.844470420304159`、
  `eval_zh_loss=9.874069731564312`。
- 实测速率约 0.45 s/step → 预估总时长 ≈ **5.6 小时**。

### 5.1 Base 开发门禁（训练完成后运行）

```bash
cd /root/zitong/llm-lifecycle-lab
source /root/.venv60m/bin/activate
python - <<'PY'
import json, math
from pathlib import Path

run = Path("runs/native-60m-base-v1-s42")
manifest = json.loads((run / "run_manifest.json").read_text())
result = json.loads((run / "training_result.json").read_text())
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
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
print("baseline eval_loss =", baseline["eval_loss"], "final eval_loss =", final["eval_loss"])
print("final_en =", final["eval_en_loss"], "final_zh =", final["eval_zh_loss"])
print("global_step =", result["global_step"], "target_token_coverage =", result["target_token_coverage"])
PY
```

门禁通过标准（不修改阈值）：`global_step==45191`、`target_token_coverage∈[1.0,1.01]`、
`final.eval_loss <= 0.5*baseline.eval_loss`、en/zh 均低于 baseline、final 不比 best 回退 >2%。

### 5.2 门禁通过后 — test 集一次性评测

```bash
source /root/.venv60m/bin/activate
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-v1.yaml \
  --checkpoint runs/native-60m-base-v1-s42/checkpoints/step-00045191 \
  --split test \
  --json > runs/native-60m-base-v1-s42/final-test.json
```

## 6. 本地核验清单（供你复现/核对）

1. **环境**：确认 `torch.cuda.is_available()` 为真；`python -m pip check`、`ruff check .`、
   `pytest -q` 全绿。
2. **配置有效**：三个 `validate_config.py` 均 PASS。
3. **数据一致**：`data/prepared/bilingual-60m-v1/data_manifest.json` 与
   `data/packed/bilingual-60m-v1-seq512/packed_manifest.json` 存在且 `doctor.py` 通过。
4. **smoke 门禁**：运行 §3 命令，再跑 §3 门禁断言，应 PASS。
5. **pilot 门禁**：运行 §4 命令（含重载评测），再跑 §4 门禁断言，应 PASS。
6. **Base-v1**：确认 §5 推导的 `max_steps==45191`；训练结束后运行 §5.1 门禁，
   通过后再运行 §5.2 的 test 评测。
7. **关键文件**：每个 run 目录下都应存在 `run_manifest.json`（status=completed）、
   `metrics.jsonl`、`training_result.json`，Base-v1 还需 `checkpoints/step-00045191`。

## 7. 当前已确认结论

- ✅ 训练链路、数据、Tokenizer 在本机与本机 CUDA 上可复现手册基线（smoke/pilot 数值吻合）。
- ✅ 本地两级门禁均 PASS。
- ⏳ Base-v1 正在训练，ETA ≈ 5.6h；完成后执行 §5.1 / §5.2 即可闭合 Base 开发门禁。


## 8. Base-v1 最终结果（自动补齐）

- 门禁结论：****PASS****
- 门禁详细输出：

```
GATE_PASSED True
{
  "passed": true,
  "checks": {
    "status_completed": true,
    "global_step_45191": true,
    "coverage_in_range": true,
    "finite_eval_loss": true,
    "finite_eval_en_loss": true,
    "finite_eval_zh_loss": true,
    "final_le_half_baseline": true,
    "en_improved": true,
    "zh_improved": true,
    "final_le_1.02_best": true
  },
  "baseline": {
    "eval_loss": 9.859895057044923,
    "eval_en_loss": 9.844470420304159,
    "eval_zh_loss": 9.874069731564312
  },
  "final": {
    "eval_loss": 2.7545488604228012,
    "eval_en_loss": 1.9535689680065067,
    "eval_zh_loss": 3.490611949773023
  },
  "global_step": 45191,
  "target_token_coverage": 1.0000110643109283,
  "best_eval_loss": 2.7064902595011517
}
```

### test 集评测（split=test, 一次性）

| 指标 | 值 |
| --- | --- |
| eval_loss | 2.731877171783708 |
| eval_en_loss | 1.9555547977087155 |
| eval_zh_loss | 3.419416282367276 |
| eval_perplexity | 15.361696514509088 |
| eval_en_perplexity | 7.0678391571300825 |
| eval_zh_perplexity | 30.551576321368643 |
| eval_tokens | 523264.0 |
| eval_en_tokens | 245764.0 |
| eval_zh_tokens | 277500.0 |

产物：`runs/native-60m-base-v1-s42/final-test.json`

### 训练曲线（已导出图片）

原始逐 step 数据全部保存在 `runs/native-60m-base-v1-s42/metrics.jsonl`（4521 行：每 10 步一条训练记录 + 47 条 eval 记录，含 `train_loss / eval_loss / eval_en_loss / eval_zh_loss / learning_rate / tokens_per_second / gradient_norm / target_token_coverage` 等字段）。

已用 `metrics.jsonl` 渲染成 PNG 供本地查看（无需 tensorboard）：

| 文件 | 内容 |
| --- | --- |
| `runs/native-60m-base-v1-s42/curves/losses.png` | train / eval / en / zh loss 随 step 变化 |
| `runs/native-60m-base-v1-s42/curves/lr_throughput.png` | 学习率曲线 + tokens/s 吞吐 |
| `runs/native-60m-base-v1-s42/curves/grad_coverage.png` | 梯度范数 + token coverage |

> 本地复现命令（需 matplotlib）：
> ```bash
> python - <<'PY'
> import json,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
> from pathlib import Path
> run=Path("runs/native-60m-base-v1-s42"); lines=[json.loads(l) for l in (run/"metrics.jsonl").read_text().splitlines() if l.strip()]
> tr=[r for r in lines if r.get("train_loss") is not None]; tr.sort(key=lambda r:r["step"])
> ev=[r for r in lines if r.get("eval_loss") is not None]; ev.sort(key=lambda r:r["step"])
> ts=[r["step"] for r in tr]
> plt.plot(ts,[r["train_loss"] for r in tr],label="train")
> plt.plot([r["step"] for r in ev],[r["eval_loss"] for r in ev],"o-",label="eval")
> plt.legend(); plt.savefig("curve.png",dpi=120)
> PY
> ```
