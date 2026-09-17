# 公开数据 60M 后训练 · native-60m-public-posttrain-v1

[项目首页](../README.md) | [项目实施路线](./test.md) | [公开数据路线](../PUBLIC_POSTTRAINING_GUIDE.md)

本文记录按 [公开数据 SFT、DPO 与 GRPO 路线](../PUBLIC_POSTTRAINING_GUIDE.md)
（`public-60m-v1` 配方）在单张 RTX 4090 上走完的第一次完整后训练：
OASST1 + MSVAMP 的 SFT、HelpSteer3 的 DPO、MSVAMP 后 800 组的 GRPO，
以及四阶段 `lifecycle-v3` 同协议评测。

> 设备为共享 GPU，同一张卡上还有其它进程；训练中出现过显存分配重试日志。
> 耗时与显存只描述本次环境，不代表硬件极限，也不用于跨设备比较。

## 1. 结果与结论

| 阶段 | run_id | steps | 监督 / rollout token | 耗时 | CUDA 峰值显存 | 训练目标（dev，step 0 → 最终） |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SFT | `native-sft-public-60m-001` | 112 | 481,776 | 27.7 s | 5.283 GiB | loss 5.1526 → **4.4969** |
| DPO | `native-dpo-public-60m-001` | 21 | 284,901 | 13.8 s | 7.678 GiB | 偏好损失 0.6915 → **0.6825** |
| GRPO | `native-grpo-public-60m-001` | 161 | 207,799 / 预算 329,728 | 1,623.2 s | 3.388 GiB | 奖励均值 0.0000 → 0.0469（**未达 +0.05**） |

三个阶段的最终 step 与方案一致（112 / 21 / 161）。

- **SFT 生效**：dev loss 从 5.1526 降到 4.4969（en 5.4527 → 4.5983，
  zh 4.9462 → 4.4272），语料 BPB 同时从 2.035306 降到 2.008218，
  没有出现非定向退化。
- **DPO 只有弱正向信号**：dev 偏好损失 0.6915 → 0.6825、reward margin
  0.00342 → 0.02417、能力探针 `preference.accuracy` 0.4375 → 0.46875；
  dev pair accuracy 停在 0.5217 未变，因此只能说目标改善，不能说偏好能力已提升。
- **GRPO 记为预注册的零结果**：dev 奖励均值只从 0 升到 0.0469（差 0.0031 未达
  +0.05），零方差组 0.875 高于 0.75 上限，dev 近似 KL 0.1305 高于 0.10 上限。
  按方案第 6 节，本次不把 loss 或 KL 变化解释为 RL 收益。

## 2. 数据

按 `public-60m-v1` 配方在固定 revision 上物化，输出哈希与方案完全一致：

| 阶段 | 来源 | 记录 | 输出 SHA-256 | train / dev / test |
| --- | --- | ---: | --- | --- |
| SFT | OASST1（en/zh 各 350）+ MSVAMP 前 200 组 | 1,100 | `4286e08a…8110f` | 883 / 116 / 101 |
| DPO | HelpSteer3（en/zh 各 280） | 560 | `d6a28429…f2b0e` | 448 / 46 / 66 |
| GRPO | MSVAMP 后 800 组 | 1,600 | `3cc66aa2…abb22b` | 1,288 / 148 / 164 |

三阶段均按 `--group-by source_id` 切分，`check-posttrain` 报告的跨阶段
`source_id` 交集为 **0**；与 `lifecycle-v3` 的重合门禁通过。
MSVAMP 只用于训练与阶段前后测量，不用于独立测试成绩。

## 3. GRPO 资格门槛

方案第 6 节的三道门槛在训练前固定，未因结果放宽。dev rollout 读数为主，
括号内为同 step 的训练 rollout（`report_*`）读数：

| 门槛 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| 1 首个 eval window（step 25） | `0 < success_rate < 1` 且 `zero_variance < 1` | 0.00781 / 0.9375 | 通过 |
| 2 前 25 step 的非零方差 group | ≥ 4 | 3（step 10 / 20 / 25 各 1 个，step 5 / 15 为 0） | 未通过 |
| 3a dev rollout 奖励均值提升 | ≥ +0.05 | +0.0469（0.0234） | 未通过 |
| 3b `zero_variance_groups` | ≤ 0.75 | 0.875（0.8125） | 未通过 |
| 3c `approx_kl` | ≤ 0.10 | 0.1305（0.0974） | 未通过 |

门槛 3 在两种读数下结论相同，因此 GRPO 记录为零结果。
后续若要继续，需要单独设计难度分层或 `kl_beta` 对照，并使用新的 run ID；
不从 test 筛题，也不把 SFT warmup 的 200 组移入 GRPO。

另需记录一处预算偏差：GRPO 的 `training_budget.json` 按 2048 token/step
估算，而实际约 1,290 token/step，因此 161 步只覆盖 1.26 个 epoch、
63% 的目标 token。step 数与方案一致，epoch 与 token 覆盖率不一致。

## 4. 能力成绩单（同一协议，41 项指标）

Base / SFT / DPO / GRPO 均使用 `native-chat-v1`，报告
`runs/evaluations/public-posttrain-v3`（副本见
[`docs/experiments/results/native-60m-public-posttrain-v1`](./results/native-60m-public-posttrain-v1/report.md)）：

| 指标 | Base | SFT | DPO | GRPO |
| --- | ---: | ---: | ---: | ---: |
| instruction.success | 0.000 | 0.000 | 0.000 | 0.000 |
| format.success | 0.000 | 0.000 | 0.000 | 0.000 |
| qa.success | 0.000 | 0.000 | 0.000 | 0.000 |
| multiturn.success | 0.000 | 0.000 | 0.000 | 0.000 |
| preference.accuracy | 0.4375 | 0.4375 | **0.4688** | 0.4375 |
| verifiable.reward | 0.000 | 0.000 | 0.000 | 0.000 |
| corpus.bpb.all | 2.035306 | 2.008218 | **2.007856** | 2.009478 |
| continuation.repeated_trigram | 0.203442 | 0.204167 | **0.158333** | 0.270833 |

- 四个阶段的 instruction / format / qa / multiturn 均为 0，
  说明 60M 模型在 `lifecycle-v3` 探针上没有可测的指令能力；
  因此这四项既无改善也无退化，不能用来声称能力收益。
- SFT 与 DPO 的语料 BPB 都优于 Base，没有出现非定向退化，
  按方案第 7 节可以进入后续比较。
- GRPO 相对 DPO 出现小幅非定向退化：BPB 2.007856 → 2.009478，
  `preference.accuracy` 0.4688 → 0.4375，`repeated_trigram` 0.1583 → 0.2708。
  它与零结果结论一致，不解释为 RL 收益。

## 5. 证据与边界

- 三阶段 run：`runs/native-sft-public-60m-001`、
  `runs/native-dpo-public-60m-001`、`runs/native-grpo-public-60m-001`
- 四阶段成绩单：`runs/evaluations/public-posttrain-v3`；
  单阶段报告 `public-base-v3`、`public-sft-v3`、`public-dpo-v3`、`public-grpo-v3`
- SFT 的父权重沿用 `build/modelscope/native-60m-base-v1`，其
  `parent_provenance_status` 原样记为 `accepted-with-provenance-waiver`
- 本次运行时 Git 为 `6cff7b8` 且 `dirty: true`（1 项，`.gitignore` 顺序调整，
  不涉及 `src/` 或配置），与 Base 的已披露例外一并保留，未改写为 clean

[项目首页](../README.md) | [项目实施路线](./test.md)
