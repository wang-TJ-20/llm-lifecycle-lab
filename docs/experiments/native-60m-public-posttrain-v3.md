# Native-60M 公开数据后训练 v3（public-60m-v3 / SFT-v3 析因矩阵）

本文记录按
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 第 3–7 节执行的
v3 轮次结果。**D0 诊断、`lifecycle-v4` 冻结、`public-60m-v3` 物化与 SFT-v3 析因
矩阵三臂已全部执行完毕；四个 SFT arm（含已完成的 A0）均未通过预注册硬门禁，
且 B1 未达到 7.2 节的学习率趋势资格，因此 `5e-5` 敏感性臂、DPO、GRPO、
sealed v4 评测均未运行**（见第 6–8 节）。

上一轮的 R0/R1 结果见
[Native-60M 公开数据后训练 v2](./native-60m-public-posttrain-v2.md)。

## 1. 结论摘要

| 环节 | 状态 | 关键结果 |
| --- | --- | --- |
| 第 3 节 远程预检 | 通过 | `pip check`、clean tree、`ruff`、275 passed、`ok == false` 断言、8+8 个诊断 checkpoint 齐全 |
| 第 4 节 D0 checkpoint 诊断 | 完成 | 两臂 8 个 checkpoint，`json_parseable` 恒为 0/16；`extra_text` ≈ `answer_contained`；basev2 的 `exact_success` 由 0 稳定升到 1 并维持 |
| 第 5 节 sealed test 冻结 | 完成 | `lifecycle-v4.yaml` SHA-256 与文档一致，全程未运行、未读取 |
| 第 6 节 public-60m-v3 物化 | 完成，门禁通过 | 三阶段 source hash 与记录数逐项一致；`check-posttrain` `ok = true`，跨阶段 group 交集全 0 |
| 第 7.1 节 SFT-v3 三臂 | 完成，**三臂门禁全部失败** | old8m 6 项、balanced2m 8 项、balanced8m **11 项** 检查未通过 |
| 第 7.2 节 LR 敏感性资格 | 未达标 | `run_lr = false`，按预注册条件跳过 `5e-5` 臂 |
| 第 7.3 节 SFT winner | **未生成** | 无 `ok == true` 候选，脚本按 §7.3/§12 返回 STOP |
| 第 8–10 节 DPO / GRPO / 成绩单 / sealed v4 | **未执行** | 无 SFT winner，硬性停止 |

终止输出：

```text
STOP: no SFT-v3 candidate passed; do not run DPO or GRPO   (exit=1)
```

## 2. 实验契约与执行环境

固定契约：`configs/models/tiny-60m.yaml`（62.93M）、tokenizer
`data/tokenizers/bilingual-60m-v1`、seed 42、context 512、BF16、
`native-chat-v1`、开发门禁 `configs/evaluation/lifecycle-v3.yaml`、
commit `429e5807e1a440b0faea84870141bd5b4369bb02`（`git status --porcelain=v1`
无输出，全部结果记录时工作区仍为 clean）。

运行环境 `/root/.venv60m`：Python 3.11、torch 2.14.0+cu130、RTX 4090 24G。

### 2.1 执行环境偏差（一项，已登记）

**共享 GPU。** 同一张 RTX 4090 上始终存在其它进程（常驻占用约 15G）。训练日志中
出现 `CUDACachingAllocator` 的分配重试（如 `old8m` 与 `balanced8m` 各有一次
`memory allocation failed with OOM` 后重试成功），训练与评测均正常完成。
因此本文记录的耗时只描述本次环境，不用于跨设备或跨轮次的性能比较
（与 v2 记录 §14.1 第 2 项口径一致）。训练结果本身不依赖该偏差：batch、
梯度累积、学习率、字典 conversational protocol、seed 均未改动。

## 3. 第 4 节：D0 checkpoint 轨迹诊断

对两个已完成 R1 run 的 step-50/100/150/200/250/300/350/383 共 16 个 checkpoint
各自运行 `lifecycle-v3`（`max-new-tokens 32`，`threads 1`，baseline 为对应
parent 报告），再由 `scripts/analyze_capability_outputs.py` 汇总。
16 个 checkpoint 全部存在，无需按 §3 尾部条款降级。

### 3.1 repeat arm（`native-sft-public-60m-r1-repeat-large-s42`）

| checkpoint | exact | contained | extra_text | json_parseable | 平均输出字符 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step-00000050 | 0/112 | 11 | 11 | 0/16 | 54.2 |
| step-00000100 | 0/112 | 16 | 16 | 0/16 | 49.9 |
| step-00000150 | 0/112 | 14 | 14 | 0/16 | 53.6 |
| step-00000200 | 0/112 | 17 | 17 | 0/16 | 51.1 |
| step-00000250 | 0/112 | 15 | 15 | 0/16 | 46.0 |
| step-00000300 | 0/112 | 14 | 14 | 0/16 | 45.7 |
| step-00000350 | 0/112 | 19 | 19 | 0/16 | 46.0 |
| step-00000383 | 0/112 | 19 | 19 | 0/16 | 44.3 |

### 3.2 basev2 arm（`native-sft-public-60m-r1-basev2-large-s42`）

| checkpoint | exact | contained | extra_text | json_parseable | 平均输出字符 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step-00000050 | 0/112 | 46 | 46 | 0/16 | 59.3 |
| step-00000100 | 0/112 | 32 | 32 | 0/16 | 44.7 |
| step-00000150 | 0/112 | 40 | 40 | 0/16 | 31.3 |
| step-00000200 | 1/112 | 42 | 41 | 0/16 | 27.7 |
| step-00000250 | 1/112 | 42 | 41 | 0/16 | 27.9 |
| step-00000300 | 1/112 | 44 | 43 | 0/16 | 29.1 |
| step-00000350 | 1/112 | 45 | 44 | 0/16 | 29.0 |
| step-00000383 | 1/112 | 42 | 41 | 0/16 | 27.9 |

### 3.3 按文档解释规则的结论

| 现象 | 是否出现 | 结论 |
| --- | --- | --- |
| exact success 中途明显更高、后期归零 | 否（repeat 全程 0；basev2 由 0 升到 1 后**保持** 1 到终点） | 不存在预算/日程效应；旧 arm 的失败不是「训过头」 |
| answer-contained 上升、exact 仍低、extra-text 高 | 是（`extra_text ≈ answer_contained`，包含数远高于精确数） | 主要缺口是**短输出与格式控制** |
| JSON parseable 始终接近 0 | 是（两臂全程 0/16） | 需要**显式 structured 监督** |
| 所有诊断始终近 0 | basev2 否（contained 32–46）、repeat 的 exact 是 | 旧数据在 2M 内已形成可测迁移，但未通过门禁 |

按 §4 末段，无论 D0 结果如何都必须继续第 5–7 节的预注册矩阵，且**不得**把任何
旧失败 checkpoint 带入 DPO。本次未这样做。

## 4. 第 5 节：sealed test 冻结

```text
configs/evaluation/lifecycle-v4.yaml: OK
```

`cbc4c6e0216ac6186a607f9de9c2081b86ff0e347deac21d71f74203e9dd72e5` 校验通过。
本轮未到达 §10，因此 **一次也未运行 sealed v4**，符合 §5「最终候选确定前不得运行、
查看或据此调参」。

## 5. 第 6 节：public-60m-v3 物化

### 5.1 规范化 source hash

`scripts/data.py fetch-posttrain --recipe public-60m-v3` 的输出与文档 §6.1 逐项一致：

| 阶段 | 记录 | source SHA-256 |
| --- | ---: | --- |
| SFT | 14,799 | `9ba984310ef0996237908b5843c67d869796fef37e312d2b811f23a647fa772b` |
| DPO | 2,200 | `565ebaf82e70eb44e5548149fa963342503e4f979d86c932fa3cf7fcf0ed43bb` |
| GRPO | 400 | `9e1f60bc945e39d5b2fc05a43c3a3cf3dd374bcc437c58773f3522d7156aad93` |

### 5.2 切分与完整数据门禁

按 `source_id` 分组、seed 42 切分后 `check-posttrain` 结果为 `ok = true`：

| 检查项 | 期望 | 实测 | 结果 |
| --- | --- | --- | --- |
| `stage_source_id_overlap` | sft-dpo / sft-grpo / dpo-grpo 全 0 | 全 0 | 通过 |
| SFT train examples | 11,882 | 11,882 | 通过 |
| SFT train supervised tokens | 475,264 | 475,264 | 通过 |
| task families | general 4,114 / short_qa 5,292 / classification 363 / numeric 1,298 / structured 815 | 同左 | 通过 |
| DPO train pairs | 1,741 | 1,741 | 通过 |
| GRPO train prompts | 318 | 318 | 通过 |

产物：`runs/qualifications/public-60m-v3-data-check.json`。

### 5.3 按 supervised token 计的配额实际占比（train split）

配额声明的是**任务族占比**，但训练按 supervised token 归一，实测占比如下：

| stratum | examples | supervised tokens | 占比 |
| --- | ---: | ---: | ---: |
| general.zh | 2,179 | 273,248 | 57.5% |
| general.en | 1,935 | 122,799 | 25.8% |
| short_qa.zh | 2,706 | 39,096 | 8.2% |
| short_qa.en | 2,586 | 19,285 | 4.1% |
| structured.zh | 410 | 8,360 | 1.8% |
| structured.en | 405 | 5,151 | 1.1% |
| classification.en | 363 | 4,517 | 1.0% |
| numeric.en | 649 | 1,404 | 0.3% |
| numeric.zh | 649 | 1,404 | 0.3% |
| **合计** | **11,882** | **475,264** | 100% |

两个后果在第 7 节的结果中被直接观测到，归因见第 8 节。

## 6. 第 7 节：SFT-v3 析因矩阵

### 6.1 控制矩阵一致性（§7.1 代码块）

```text
PASS: SFT-v3 control matrix is internally consistent
```

`old8`/`new8` 的 `training` 段完全相同、`new2`/`new8` 的 `data` 段完全相同、
`new2`/`new8` 除 `max_train_tokens` 外一致、`new8`/`lr` 除 `learning_rate`
（2e-5 vs 5e-5）外一致；四个 config 的 `init_checkpoint` 与 `seed` 唯一。

### 6.2 四臂结果（A0 为已完成的 old-data/2M）

同一 Base-v2 初始化（`step-00062453`），同 tokenizer、seed、context、BF16、
`native-chat-v1`，parent baseline 统一为
`runs/evaluations/r1-parent-basev2/report.json`
（`instruction/format/qa/verifiable = 0/0/0/0`，`corpus.bpb.all = 1.693502`）。

| arm | 数据 | 预算 | tokens seen | steps | virtual epochs | 耗时 | final loss | best eval loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0（已完成） | old v2 | 2M | 2,002,968 | 383 | 2.24 | 74.5s | 3.2903 | 3.5078 |
| A1 | old v2 | 8M | 8,003,619 | 1,529 | 8.95 | 222.6s | 2.6472 | 3.4477 |
| B0 | balanced v3 | 2M | 2,001,101 | 3,505 | 18.88 | 544.2s | 1.6626 | 3.1951 |
| B1 | balanced v3 | 8M | 8,004,634 | 14,020 | **75.52** | 2,100.4s | **0.3376** | 3.1998 |

硬门禁结果：

| 检查 | 阈值 | A0 | A1 old8m | B0 balanced2m | B1 balanced8m |
| --- | --- | --- | --- | --- | --- |
| dev objective ↓≥5% | ≤3.7901 / ≤3.7311 | 3.5078 通过 | 3.4632 通过 | 3.2390 通过 | **3.9017 失败** |
| instruction.success ≥4 | 32 | **1 失败** | **3 失败** | **2 失败** | **0 失败** |
| format.success ≥3 | 24 | **0 失败** | **0 失败** | **2 失败** | **0 失败** |
| qa.success ≥4 | 32 | **0 失败** | **0 失败** | **0 失败** | **0 失败** |
| verifiable.reward ≥12 | 112 | **1 失败** | **3 失败** | **4 失败** | **0 失败** |
| instruction en / zh ≥1 | 16 / 16 | 0 / 1 | 0 / 3 | 0 / 2 | **0 / 0** |
| format en / zh ≥1 | 12 / 12 | 0 / 0 | 0 / 0 | 2 / 0 | 0 / 0 |
| qa en / zh ≥1 | 16 / 16 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| corpus BPB ≤1.02× parent | ≤1.7274 | 1.713292 通过 | **1.883611（1.112×）失败** | **1.920438（1.134×）失败** | **2.481439（1.465×）失败** |
| 跨阶段 source group 交集 | 0 | 通过 | 通过 | 通过 | 通过 |
| **合计** | | **ok=false** | **ok=false** | **ok=false** | **ok=false** |

### 6.3 学习率敏感性资格（§7.2）

`runs/qualifications/sft-v3-lr-sensitivity.json`：

```json
{
  "balanced_counts": {"format.success": 0, "instruction.success": 0, "qa.success": 0, "verifiable.reward": 0},
  "bpb_retention": false,
  "core_nonzero": 0,
  "old_counts": {"format.success": 0, "instruction.success": 3, "qa.success": 0, "verifiable.reward": 3},
  "run_lr": false
}
```

四项条件全部不满足：B1 四类成功数之和为 0（要求 ≥6）、核心非零类为 0
（要求 ≥2）、不高于 old-data/8M 的 6、BPB retention 失败。
因此脚本输出 `Skip LR sensitivity: pre-registered trend condition not met`，
`native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42` **未运行**。

### 6.4 winner 选择（§7.3）

四个 gate 文件均为 `ok = false`，脚本按 §7.3 输出 STOP 并以 1 退出，
`runs/gates/sft-r2-winner.json` **未生成**（这是 §12 矩阵要求的正确终端状态，
不是遗漏）。

## 7. 停止决策

按 §12「全部 SFT arm 失败 → 停止 DPO/GRPO，进入 Base-v3」以及 §11 第 1 条
「B1 未达到第 7.2 节趋势资格，且没有任何 SFT arm 通过」，本轮的终端状态是：

- 不运行 DPO 两臂、不运行 GRPO qualification 与三个 KL arm；
- 不运行 §10 的 v3 纵向成绩单与 sealed v4 评测（无候选可评）；
- 未生成 `sft-r2-winner.json` / `dpo-r2-winner.json` / `grpo-r2-winner.json`；
- 未降低门禁、未改写 `check_stage_gate.py` 的预注册定义、未改动
  `max_new_tokens` 或 suite 规则、未把旧失败 checkpoint 带入任何后续阶段。

## 8. 归因

与 v2 相比，本轮新增的事实有三条：

1. **换数据没有消除缺口，反而暴露了取样器的 epoch 结构问题。**
   `supervised-token-quota` 的虚拟 epoch 只有
   `estimated_tokens_per_virtual_epoch = 105,994` token、每例平均 8.92 个
   supervised token（见 `sft_data_summary.json`）。因此 8M 预算在 balanced v3 上
   等于 **75.5 个 epoch**（B1），final loss 掉到 0.3376 而 dev objective 反而升到
   3.9017、BPB 退化到 parent 的 1.465 倍——这是过拟合形态，不是欠拟合。
   同一预算在 old v2 上只有 8.9 个 epoch（A1）。**「token 预算」与「epoch 数」
   在本取样器下不可互换**，二者必须同时约束。
2. **配额按 supervised token 归一后，长答案仍然占主体。**
   声明的 general 配额是 40%，但 general.zh 一层的 supervised token 就占 57.5%；
   numeric 两层各占 0.3%、classification 占 1.0%、structured 两层合计 2.9%。
   也就是说，「增加短答案记录数」没有等比转化为 supervised token 权重，
   §1 记录的「短回答仍会被长回答淹没」这一机制在 v3 里依然存在。
3. **D0 已排除「训过头导致能力消失」这一假说。**
   repeat arm 全程 exact=0、basev2 arm 从 step-200 起稳定停在 exact=1，
   都不存在「中途更高后归零」的轨迹；而 `json_parseable` 在 16 个 checkpoint 上
   恒为 0。缺口集中在**输出长度控制**（extra_text ≈ answer_contained）与
   **结构化格式**，与 B0 唯一非零的 `format.success = 2`（全部在 en）方向一致。

综合：这是一种「**phase 只优化自身 objective、能力项不迁移**」的形态。B0 在
2M 预算下四类成功数之和为 8（高于 A1 的 6），说明 balanced v3 在**低 epoch 数**
下确实比 old v2 更能产出精确短输出；但一旦把预算放大到 8M（75 epoch），
所有能力读数归零。下一步必须先解决 **epoch 数与配额口径**，而不是继续加预算
或加 `.optimizer`。

## 9. 承接：进入 Base-v3 的前置条件

§11 第 1 条已满足，因此 SFT-v3 路线的驳回结果是**支持**进入 Base-v3 的。但在开工前
需要一份新的预注册计划，并至少先把三件事写死（本轮结果给出的硬约束）：

1. **budget 的定义方式。** 不能用 `max_train_tokens` 直接配 `supervised-token-quota` 取样器，
   除非先限定 epoch 上限；否则「8M 预算」在 balanced 数据上就是 75 epoch。
   建议改为「epoch 上限 + token 预算」双约束。
2. **配额应按 exemple 或混合口径分配，不能只按 supervised token。** 否则
   numeric/structured/classification 的实际权重会被长答案压到 1% 以下。
3. **中文 classification 仍缺许可清晰的公开标注源。** 本轮按 §2 的规定未伪造该数据，
   语言配额由其它族补齐；若要真正补齐 zh classification，需要新的候选数据源。

Base-v3 仍需遵守 §11 的六条约束（450M–480M 唯一 train token、stable hash + token
quota 采样、与约 460M seen token 的 repeat arm 做 compute-matched 对照、固定现有
tokenizer、多域公开非合成中英双语数据、从随机初始化训练、sealed test 不参与配方选择）。

## 10. 本轮产物

```text
runs/evaluations/sft-r1-checkpoint-diagnostics/                 # 16 个 checkpoint 报告 + 2 份 summary
runs/qualifications/public-60m-v3-data-check.json
runs/qualifications/sft-v3-lr-sensitivity.json
runs/native-sft-public-60m-r2-old8m-basev2-s42/                 # 全套 + checkpoints（含中间 step）
runs/native-sft-public-60m-r2-balanced2m-basev2-s42/
runs/native-sft-public-60m-r2-balanced8m-basev2-s42/
runs/evaluations/native-sft-public-60m-r2-{old8m,balanced2m,balanced8m}-basev2-s42/
runs/gates/native-sft-public-60m-r2-{old8m,balanced2m,balanced8m}-basev2-s42.json
data/raw/public-60m-v3/                                          # bundle_manifest.json + 三阶段 source
data/prepared/{sft,dpo,grpo}-public-60m-v3/
```

未生成（按 §12 逐一核对，均为正确终端状态）：
`runs/gates/native-sft-public-60m-r2-balanced8m-lr5e5-basev2-s42.json`、
`runs/gates/sft-r2-winner.json`、`runs/gates/dpo-r2-winner.json`、
`runs/gates/grpo-r2-winner.json`、
`runs/qualifications/native-grpo-public-60m-r2-parent-s42/`、
`runs/native-dpo-public-60m-r2-*`、`runs/native-grpo-public-60m-r2-*`、
`runs/evaluations/public-posttrain-r2-v3/`
与 `runs/evaluations/public-posttrain-r2-sealed-v4/`。

每个训练 run 均已保留 `resolved_config.yaml`、`run_manifest.json`、
`runtime_environment.json`（`code.dirty == false`）、`training_budget.json`、
`training_result.json`、`metrics.jsonl`、`initialization.json`、`sft_data_summary.json`
与最终 checkpoint。

最终 checkpoint 绝对路径与 SHA-256：

| run_id | 绝对路径（`model/model.pt`） | SHA-256 |
| --- | --- | --- |
| `native-sft-public-60m-r2-old8m-basev2-s42` | `/root/zitong/llm-lifecycle-lab/runs/native-sft-public-60m-r2-old8m-basev2-s42/checkpoints/step-00001529` | `89a631d0c153d6d097a9e6a1eb5d75409987950b32055212eab5c32f36af1f7e` |
| `native-sft-public-60m-r2-balanced2m-basev2-s42` | `/root/zitong/llm-lifecycle-lab/runs/native-sft-public-60m-r2-balanced2m-basev2-s42/checkpoints/step-00003505` | `a36584b91d06d8a65ce532b16f41ae440d69b40c79d167ff6b91aab1b65ec95c` |
| `native-sft-public-60m-r2-balanced8m-basev2-s42` | `/root/zitong/llm-lifecycle-lab/runs/native-sft-public-60m-r2-balanced8m-basev2-s42/checkpoints/step-00014020` | `d51c732a9932775875b8dbd3639bf16e09e9007ad09a1e5ebb0b57b6ad58882d` |
