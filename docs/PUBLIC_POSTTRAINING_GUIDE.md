# Native-60M 公开数据后训练计划

本文是 Native-60M 下一轮公开数据训练与评测的执行计划。默认目标不是再次机械地
跑完 Base -> SFT -> DPO -> GRPO，而是先证明每一阶段具备进入下一阶段的资格。
任一硬门禁失败时立即停止，不用后续训练掩盖上游问题。

第一次 `public-60m-v1` 实验保留为冻结对照，结果见
[公开数据 60M 后训练 v1](./experiments/native-60m-public-posttrain-v1.md)。
旧的 `build_posttraining_data.py` 和不带 `public` 的 pipeline 只用于复现历史
合成数据实验，不再作为新实验入口。

## 0. 先读结论

### 0.1 v1 已经回答了什么

| 阶段 | 冻结事实 | 结论 |
| --- | --- | --- |
| Base-v1 | 62.93M 参数只看过 46.18M train token，即 0.73 token/param；最终 dev loss 为 en 2.1178、zh 4.0735，且最后仍在下降 | Base 明显训练不足，中文能力尤其弱 |
| SFT-v1 | 883 条 train、112 updates、约 48.18 万 assistant token；`lifecycle-v3` 的 instruction/format/QA/multiturn/verifiable 全为 0 | dev loss 下降不等于获得指令能力；数据规模和任务覆盖不足 |
| DPO-v1 | 448 对 train、21 updates；dev pair accuracy 为 24/46，独立 preference 只多 1/32 | 只有弱目标信号，没有可靠的独立偏好收益 |
| GRPO-v1 | 前 25 step 只有 3 个非零方差 group；最终 14/16 dev group 零方差，KL 0.1305 | 父模型没有足够可验证能力，GRPO 属于零结果 |

GRPO 的 161 个 optimizer step 实际已遍历约 2 轮 prompt batch。报告中的
`63% token coverage` 是因为回答平均约 10 token、低于 16 token 的预算上界，
对应约 1.26 个**预算 token epoch**，不是只看过 1.26 轮 prompt。不能靠简单补
step 修复无奖励方差问题。

### 0.2 新路线的顺序

```text
R0 实现与证据门禁
  -> Base 计算量对照
  -> 扩大唯一语料的 Base-v2
  -> Base x SFT 数据规模四臂实验
  -> SFT 资格门禁
      -> DPO pure / DPO+NLL 对照
      -> GRPO 训练前 rollout 资格检查
  -> 只对通过门禁的分支做最终评测
```

DPO 和 GRPO 都不是必须完成的流水线步骤。DPO 失败时保留 SFT，不把失败的 DPO
作为 GRPO 父权重；只要 SFT 已通过可验证能力门禁，GRPO 可以直接从 SFT 分支做
资格检查。

### 0.3 当前提交的可执行边界

截至第一次 v1 结果提交 `3c947ce`，旧基线可复现，但**下一轮 v2 还不能在远程
服务器直接启动**：

| R0 项 | 当前状态 | 开跑前必须达到 |
| --- | --- | --- |
| Base 计算量对照配置 | 待新增 | 同 Base-v1 数据、Tokenizer、架构和 seed，固定约 3.69 亿 seen token |
| Base-v2 扩大语料 | 待新增 | 3-5 亿唯一 train token 的第一里程碑；理想目标 6-12 亿 |
| 固定 Tokenizer 复用 | 代码阻塞 | 预训练代码当前要求 Tokenizer 来源 manifest 等于训练 manifest；需显式支持固定 Tokenizer + 新语料，并记录两者 provenance |
| 全训练源泄漏检查 | 代码阻塞 | 当前 capability corpus audit 只接受 Tokenizer 绑定的旧 manifest；suite builder 也未扫描 pretrain `text` 字段 |
| SFT-v2 | 待新增 | 至少 10,000 条去重后的公开 train 样本，覆盖短答案、格式、分类、QA、多轮和算术 |
| DPO-v2 | 待新增 | 扩大公开偏好数据和 dev 分母，不能只重复 448 对 |
| 通用后训练 recipe | 代码阻塞 | 当前 loader 固定为 OASST1、HelpSteer3、MSVAMP，无法直接加入新来源 |
| GRPO rollout 资格工具 | 待新增 | 在固定 train qualification subset 上只采样、不更新参数，并输出 group 方差 |
| 自动门禁脚本 | 待新增 | 读取训练与 capability JSON，任一硬阈值失败时返回非零退出码 |
| v1 单阶段评测报告 | 远端待同步 | 除 aggregate 外，还需 Base/SFT/DPO/GRPO 四份 `report.json`，保留逐题输出 |

**R0 未全部完成前，不启动新的 CUDA 正式训练。** 本文后续出现的 v2 路径和
配置名是实现契约；只有文件已经进入干净提交、校验命令全绿后，相关训练命令才
是可执行命令。

## 1. 全局实验契约

### 1.1 固定项

除正在研究的变量外，所有实验臂固定：

- 模型：`configs/models/tiny-60m.yaml`，62.93M 参数。
- Tokenizer：`data/tokenizers/bilingual-60m-v1`；Base-v2 第一轮不得重训
  Tokenizer，否则无法隔离数据与训练量收益。
- context：512；seed：42；BF16；优化器及 beta 参数沿用 Base-v1。
- 后训练 prompt protocol：`native-chat-v1`。
- 能力评测：`configs/evaluation/lifecycle-v3.yaml`、greedy、
  `max_new_tokens=32`、`threads=1`。
- 所有数据按 `source_id` 分组切分，禁止同源样本跨 train/dev/test。
- SFT 与 GRPO 的 `source_id` 交集必须为 0。
- suite 与全部 pretrain/SFT/DPO/GRPO 训练源逐条检查；任一源缺失时
  fail-closed。

不要在看到 test 或 probe 结果后改阈值、挑训练题、改 seed 或追加 epoch。
超参数变化必须创建新配置、新 run ID 和新输出目录。

### 1.2 数据准入

SFT、DPO、GRPO 只接受有明确来源、固定 revision、文件 SHA-256 和许可证的公开
数据集。纯规则模板或临时 LLM 生成数据不得作为正式训练主体。每个 recipe 必须
记录：

1. repository、40 位 revision、上游文件路径和 SHA-256；
2. license、语言、是否 synthetic、筛选与去重规则；
3. 规范化输出 SHA-256、记录数、语言数和任务数；
4. train/dev/test 的记录数、group 数和各 split SHA-256；
5. 与 `lifecycle-v3` 及其它阶段 source group 的交集检查结果。

Base-v2 语料还需满足：

- 第一里程碑为 3-5 亿**唯一 train token**，不是把 46.18M token 重复 8 次；
- 英文和中文各占 40%-60% token；
- 至少覆盖叙事、百科/知识、说明文和对话等多个领域；
- 任一单一来源不超过 50%，synthetic token 不超过 30%；
- dev/test 与 train 按文档或 source group 隔离。

SFT-v2 需有至少 10,000 条去重后的 train 样本，en/zh 各不少于 35%，并在数据卡
中分别报告短约束回答、分类、结构化输出、QA、多轮、数学/可验证任务和开放问答
的数量。不得通过重复相同题目或同一模板改写凑规模。

DPO-v2 建议至少 3,000 个 train pair、200 个 dev pair，en/zh 各不少于 30%。
若当前公开来源无法满足，先引入经过许可审查的新来源，不降低分母门槛。

### 1.3 统一产物

每个 run 至少归档以下小文件，不把 checkpoint 权重提交到 Git：

```text
resolved_config.yaml
run_manifest.json
runtime_environment.json
training_budget.json
training_result.json
metrics.jsonl
initialization.json              # 后训练阶段
*_data_summary.json              # 后训练阶段
evaluations/*/report.json
evaluations/*/report.md
```

实验登记表必须包含 run ID、Git commit、config SHA-256、数据 manifest SHA-256、
父权重 SHA-256、实际 step、seen token、prompt pass、rollout token coverage、
最终 checkpoint 和门禁结论。

## 2. Phase R0：补齐实现后再上 GPU

计划新增并提交以下实现。名称可在实现时调整，但文档、配置和命令必须同时更新：

```text
src/llm_lifecycle_lab/data/recipes/*-60m-v2.yaml
src/llm_lifecycle_lab/data/mixtures/bilingual-60m-v2.yaml
src/llm_lifecycle_lab/data/posttraining_recipes/public-60m-v2.yaml
configs/reference/native-60m-pretrain-v2.yaml
configs/pipelines/native-60m-base-repeat-control-v2.yaml
configs/pipelines/native-60m-base-expanded-v2.yaml
configs/pipelines/native-sft-public-60m-v2-*.yaml
configs/pipelines/native-dpo-public-60m-v2-*.yaml
configs/pipelines/native-grpo-public-60m-v2-*.yaml
scripts/qualify_grpo.py
scripts/check_stage_gate.py
```

实现验收必须覆盖：

1. 固定旧 Tokenizer 训练新 pretrain manifest，且两个 manifest hash 都进入 run
   provenance；模型词表和 Tokenizer 内容 hash 仍强校验。
2. capability leakage audit 能扫描所有 pretrain manifest 和后训练 source，
   识别 `text`、`messages`、`prompt` 三类字段；缺文件或解析失败必须报错。
3. 通用 post-training recipe 可以加入多个公开来源，并按来源、任务、语言输出
   统计；不能把来源逻辑写死在文档里。
4. `qualify_grpo.py` 只做 rollout，不调用 optimizer，不读 dev/test，不修改
   checkpoint。
5. `check_stage_gate.py` 从 JSON 读取分母和数值，硬门禁失败返回非零退出码。
6. Base 的 prompt pass 与 GRPO 的 rollout token coverage 分开记录，不能再把
   token 预算 epoch 当作数据遍历 epoch。
7. 新增单元与集成测试，并保证全仓测试与 lint 通过。

本地验收：

```bash
conda activate llm-lifecycle-lab
python -m pip check
python scripts/data.py recipes
python scripts/data.py posttrain-recipes
python scripts/validate_config.py \
  configs/pipelines/native-60m-base-repeat-control-v2.yaml
python scripts/validate_config.py \
  configs/pipelines/native-60m-base-expanded-v2.yaml
pytest -q
ruff check .
```

然后确认工作区干净并记录版本：

```bash
git status --porcelain=v1
git rev-parse HEAD
```

`git status --porcelain=v1` 必须无输出。数据、配置或门禁脚本未进入该 commit 时，
不要在远端手改后开跑。

## 3. Phase 1：Base 诊断与 Base-v2

### 3.1 计算量对照，不冒充数据扩容

`Base-repeat-control` 从随机初始化开始，完全复用 Base-v1 的 46,184,530 train
token、Tokenizer、模型、seed 和优化器，只把预算提高到 8 epoch：

- seen token 目标：369,476,240；
- 预计 optimizer step：45,192；
- 有效 seen token/param：约 5.87；
- 唯一 token/param 仍只有 0.73。

该臂只回答“Base-v1 是否主要因为计算不足”。它不能证明扩大了知识或数据多样性，
也不命名为正式 Base-v2。

R0 完成后的命令：

```bash
python scripts/doctor.py \
  --config configs/pipelines/native-60m-base-repeat-control-v2.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-repeat-control-v2.yaml \
  --run-id native-60m-base-repeat-control-v2-s42
```

### 3.2 扩大唯一语料的正式 Base-v2

`Base-expanded-v2` 使用 3-5 亿唯一 train token 的新 manifest，保持架构、
Tokenizer 和训练超参数不变。第一轮只跑 1 个数据 epoch；如果仍在稳定下降，
后续再把正式预算扩到 6-12 亿 token，不在同一个已完成 run 上追加。

```bash
python scripts/doctor.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml
python scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --reference-spec configs/reference/native-60m-pretrain-v2.yaml \
  --run-id native-60m-base-expanded-v2-s42
```

只用固定 dev 选择 Base；选定后才对该 checkpoint 跑一次 test：

```bash
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --checkpoint runs/native-60m-base-expanded-v2-s42/checkpoints/FINAL_STEP \
  --split dev \
  --json
python scripts/eval_pretrain.py \
  --config configs/pipelines/native-60m-base-expanded-v2.yaml \
  --checkpoint runs/native-60m-base-expanded-v2-s42/checkpoints/FINAL_STEP \
  --split test \
  --json
```

`FINAL_STEP` 从 `training_result.json` 读取，不在训练前猜测。

### 3.3 Base 硬门禁

以 Base-v1 固定 dev 为基线，正式 Base-v2 必须同时满足：

| 检查 | 阈值 |
| --- | --- |
| 完成度 | `target_token_coverage` 在 `[1.00, 1.01]`，无 NaN/Inf |
| overall dev loss | 不高于 2.9480，即相对 3.1032 至少改善 5% |
| en dev loss | 不高于 2.0119，即相对 2.1178 至少改善 5% |
| zh dev loss | 不高于 3.8698，即相对 4.0735 至少改善 5% |
| held-out test | overall/en/zh 均不得比 Base-v1 对应 test 恶化超过 1% |
| provenance | clean Git、数据/packing/config/reference hash 全通过 |

若 repeat-control 通过而 expanded-v2 失败，结论只能是“增加同语料计算量有效”，
不能发布 expanded-v2。若两者都失败，先修改 Base 数据或优化设置，不进入 SFT
四臂实验。

## 4. Phase 2：Base x SFT 数据规模四臂实验

### 4.1 为什么必须四臂

只运行 `Base-v2 + SFT-v2` 无法区分收益来自 Base 还是 SFT 数据。四臂固定为：

| 臂 | Base | SFT 数据 | 回答的问题 |
| --- | --- | --- | --- |
| A | Base-v1 | SFT-v1 small | 新协议下的小数据基线 |
| B | Base-v2 | SFT-v1 small | Base 改善在相同 SFT 下的收益 |
| C | Base-v1 | SFT-v2 large | 扩大 SFT 数据在相同 Base 下的收益 |
| D | Base-v2 | SFT-v2 large | 最终候选 |

历史 A0 的 112-step 结果只作参考。正式 A/B/C/D 必须重新运行并使用相同的
assistant supervised-token 预算；建议首轮统一为 2,000,000 token，配置使用
`max_train_tokens`，不能用相同 epoch 数伪装成相同计算量。

除 Base 和 SFT manifest 外，四臂的 seed、batch、学习率、warmup 比例、
sequence length、评测频率和总 supervised token 完全一致。每个 arm 先 Doctor，
再使用配置中冻结的父 checkpoint 启动：

```bash
python scripts/doctor.py --config CONFIG
python scripts/train_sft.py --config CONFIG --run-id RUN_ID
```

建议 run ID：

```text
native-sft-v2-basev1-small-s42
native-sft-v2-basev2-small-s42
native-sft-v2-basev1-large-s42
native-sft-v2-basev2-large-s42
```

逐指标报告以下差值，不合并成单一能力分：

```text
Base 主效应（small） = B - A
Base 主效应（large） = D - C
SFT 数据效应（Base-v1） = C - A
SFT 数据效应（Base-v2） = D - B
交互效应 = (D - C) - (B - A)
```

### 4.2 SFT 资格门禁

每一臂都以自己的 Base 父权重为 baseline。只有同时满足下表的 arm 才有资格进入
DPO 或 GRPO：

| 指标 | 硬阈值 |
| --- | ---: |
| SFT dev objective | final loss 相对 step 0 至少下降 5% |
| `instruction.success` | 至少 4/32 |
| `format.success` | 至少 3/24 |
| `qa.success` | 至少 4/32 |
| `verifiable.reward` | 至少 12/112 |
| 双语覆盖 | instruction/format/QA 的 en、zh 各至少命中 1 题 |
| `corpus.bpb.all` | 不高于父 Base 的 1.02 倍 |
| 数据与泄漏 | source group 交集为 0，全部源文件存在且 hash 正确 |

`multiturn.success` 继续报告，但 v3 该项历史上不敏感，暂不单独作为硬门禁。
如果没有任何 arm 通过，停止后训练，先修 SFT 数据分布；禁止直接运行 DPO/GRPO。

若多个 arm 通过，先按四个任务指标的总正确题数选择，再用较低的
`corpus.bpb.all` 和 SFT dev loss 依次打破平局。选择规则必须在看结果前写入门禁
脚本。

## 5. Phase 3：DPO 对照

### 5.1 两个实验臂

只从通过 SFT 门禁的同一个 winner checkpoint 出发，数据、seed、beta、batch 和
token 预算相同：

| 臂 | `nll_coefficient` | 目的 |
| --- | ---: | --- |
| DPO-pure | 0.0 | 纯 DPO 对照 |
| DPO-NLL | 0.1 | 防止只压低 rejected 绝对概率导致能力退化 |

首轮不同时扫描 beta、学习率和 NLL 系数，避免不可归因。若 0.1 给出稳定信号，
下一轮再单独扫描 NLL 系数。

```bash
python scripts/doctor.py \
  --config configs/pipelines/native-dpo-public-60m-v2-pure.yaml
python scripts/train_dpo.py \
  --config configs/pipelines/native-dpo-public-60m-v2-pure.yaml \
  --run-id native-dpo-v2-pure-s42

python scripts/doctor.py \
  --config configs/pipelines/native-dpo-public-60m-v2-nll010.yaml
python scripts/train_dpo.py \
  --config configs/pipelines/native-dpo-public-60m-v2-nll010.yaml \
  --run-id native-dpo-v2-nll010-s42
```

### 5.2 DPO 硬门禁

| 检查 | 阈值 |
| --- | --- |
| 训练目标 | final dev preference loss 低于 step 0 |
| dev pair accuracy | v1 的 46 对口径下至少 30/46；若使用 >=200 对的 v2 dev，则 95% Wilson 下界必须 > 0.5 |
| 独立 preference | `lifecycle-v3` 相对 SFT 父权重至少多 3/32 |
| 任务保持 | instruction/format/QA/multiturn/verifiable 每项相对 SFT 最多少 1 个 case |
| 语言建模保持 | `corpus.bpb.all` 不高于 SFT 的 1.02 倍 |

训练 loss、reward margin 或 1/32 的 preference 波动都不能单独判为成功。
两个 DPO arm 都失败时保留 SFT winner，并允许直接从 SFT 做 GRPO 资格检查。

## 6. Phase 4：GRPO 资格检查与 KL 扫描

### 6.1 训练前只做 rollout

从选定父权重对 GRPO train split 中固定的 64 个 `source_id` 抽取 en/zh 两条记录，
共 128 个 prompt。固定 `group_size=8`、temperature 1.0、top-p 1.0 和 rollout
seed；不得读取 dev/test，也不得更新参数：

```bash
python scripts/qualify_grpo.py \
  --config configs/pipelines/native-grpo-public-60m-v2-kl004.yaml \
  --source-groups 64 \
  --output runs/qualifications/native-grpo-v2-parent-s42
```

资格报告必须同时满足：

- `0 < success_rate < 1`；
- 至少 25%，即 32/128 prompt group 内同时出现正、负奖励；
- en、zh 都至少有一个 mixed-reward group；
- 所有 rollout 均能被 verifier 解析，解析失败需单独计数，不能静默当成系统错误。

若 mixed group 比例不足 25%，不启动 GRPO。先只用 train 数据设计课程或调整奖励，
冻结新的 qualification subset 后重新登记；禁止从 dev/test 挑“可解题”。

### 6.2 只在信号合格后扫描 KL

资格通过后，固定其它变量，分别运行：

```text
kl_beta = 0.04
kl_beta = 0.08
kl_beta = 0.12
```

三个 arm 使用相同父权重、prompt 顺序、group size、采样参数和 prompt-pass 预算。
首轮只跑 1 个完整 prompt pass；选出满足门禁的 beta 后，才允许用新 run ID 扩到
最多 2 个 prompt pass。

```bash
python scripts/doctor.py --config CONFIG
python scripts/train_grpo.py --config CONFIG --run-id RUN_ID
```

建议 run ID：

```text
native-grpo-v2-kl004-s42
native-grpo-v2-kl008-s42
native-grpo-v2-kl012-s42
```

### 6.3 GRPO 硬门禁

| 检查 | 阈值 |
| --- | --- |
| dev reward | `report_reward_mean` 相对 step 0 至少提高 0.05 |
| 奖励方差 | `report_zero_variance_groups <= 0.75` |
| 漂移 | `report_approx_kl <= 0.10` |
| 任务保持 | instruction/format/QA/multiturn/preference 每项相对父权重最多少 1 个 case |
| 语言建模保持 | `corpus.bpb.all` 不高于父权重的 1.02 倍 |
| 预算记录 | prompt pass 与 rollout token coverage 都存在且分别解释 |

任一项失败即记零结果，不解释为 RL 收益。多个 beta 通过时先选 dev reward 较高者，
再用较低 KL 打破平局。

## 7. Phase 5：同协议评测

### 7.1 每个阶段立即评测

所有 checkpoint 在同一机器、同一依赖环境中连续评测。下面以环境变量代替已由
门禁选出的实际路径：

```bash
BASE_CKPT=runs/native-60m-base-expanded-v2-s42/checkpoints/FINAL_STEP
SFT_CKPT=runs/SELECTED_SFT_RUN/checkpoints/FINAL_STEP
DPO_CKPT=runs/SELECTED_DPO_RUN/checkpoints/FINAL_STEP
GRPO_CKPT=runs/SELECTED_GRPO_RUN/checkpoints/FINAL_STEP
```

先评测 Base：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint "$BASE_CKPT" \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-v2-base \
  --device cuda \
  --threads 1 \
  --max-new-tokens 32 \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json
```

SFT 以 Base 报告为 baseline：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint "$SFT_CKPT" \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-v2-sft \
  --device cuda \
  --threads 1 \
  --max-new-tokens 32 \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --baseline runs/evaluations/public-v2-base/report.json
```

DPO 与 GRPO 使用同一命令，分别替换 checkpoint、输出目录和前一阶段 baseline。
如果某阶段没有通过门禁，就从比较中省略该阶段，不创建“占位 winner”。

当前 `--pretrain-manifest` 仍必须指向 Tokenizer 绑定的 v1 manifest。R0 必须另行
实现并执行对 Base-v2 全部训练源的 audit；在该实现完成前，上述 capability
报告不能单独证明 Base-v2 无泄漏。

### 7.2 生成纵向成绩单

```bash
python scripts/evaluate_capabilities.py compare \
  --reports runs/evaluations/public-v2-base/report.json \
            runs/evaluations/public-v2-sft/report.json \
            runs/evaluations/public-v2-dpo/report.json \
            runs/evaluations/public-v2-grpo/report.json \
  --output runs/evaluations/public-posttrain-v2
```

只比较实际存在且通过门禁的报告。`compare` 会校验 protocol hash、分母和基线；
不允许手工拼接不同 device、线程、Tokenizer、suite 或 generation 参数的数字。

除 aggregate 外，必须同步每个阶段的 `report.json` 和 `report.md`。逐题
`samples` 用于区分语义错误、格式错误和空输出；只同步 aggregate 无法做
badcase 归因。

## 8. 决策树与停止条件

| 条件 | 动作 |
| --- | --- |
| R0 任一项未完成 | 不上 GPU |
| Base-repeat 改善、Base-expanded 不改善 | 记录计算量收益，修扩大语料；不发布 Base-v2 |
| Base-v2 未过门禁 | 不跑四臂 SFT |
| 四个 SFT arm 均未过门禁 | 停止，扩数据覆盖或调整 SFT 目标 |
| SFT 通过、DPO 失败 | 保留 SFT；GRPO 可直接从 SFT 做资格检查 |
| GRPO qualification 未过 | 不训练 GRPO |
| GRPO 训练门禁未过 | 记零结果，最终模型回退到 SFT 或通过的 DPO |
| 某阶段只改善自身 loss | 不视为能力收益，不进入下一阶段 |

完成首轮探索后，若要对外声称收益，至少对选定链路追加两个 seed。单 seed 和
32/24/112 题的小探针只支持工程决策，不支持稳定的总体能力结论。

## 9. 远程执行与回传清单

每次训练前：

```bash
conda activate llm-lifecycle-lab
python -m pip check
git status --porcelain=v1
git rev-parse HEAD
python scripts/validate_config.py CONFIG
python scripts/doctor.py --config CONFIG
```

执行规则：

1. Doctor 有 `FAIL` 时不训练；已知的后训练 `real-batch WARN` 需单独记录。
2. run 中断只能用完全相同配置恢复：

   ```bash
   python scripts/train_sft.py --config CONFIG --resume-run RUN_ID
   ```

3. 已完成预算的 run 不追加训练；改变数据、父 checkpoint、预算、学习率或目标
   函数时创建新 run ID。
4. 每个阶段训练结束后立即运行能力评测和 `check_stage_gate.py`；gate 非零退出
   时不执行下一阶段。
5. 回传小型证据文件和评测报告；checkpoint、optimizer state 和大数组保留在远端
   并记录 SHA-256 与存储位置。

## 附录 A：冻结的 public-60m-v1 数据

配方：
`src/llm_lifecycle_lab/data/posttraining_recipes/public-60m-v1.yaml`。

| 阶段 | 数据源 | 固定 revision | 文件 SHA-256 | 许可 |
| --- | --- | --- | --- | --- |
| SFT | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e` | Apache-2.0 |
| DPO | `nvidia/HelpSteer3` | `f6d145777bcbde96137596340fab89793acd1031` | `32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb` | CC-BY-4.0 |
| SFT/GRPO | `Mathoctopus/MSVAMP` | `301e2b3b168be70058c89c21c2fbdc9262102add` | `31e136303b56ed1734d3e9fbcdcff4499f974a0857298816c41e1c5c0be6b4bd` | Apache-2.0 |

规范化输出：

| 阶段 | 记录数 | train/dev/test | 输出 SHA-256 |
| --- | ---: | --- | --- |
| SFT | 1,100 | 883/116/101 | `4286e08a52910b027c2c3d0b7ab1b0333faec11792d918d6a3cc7c483be8110f` |
| DPO | 560 | 448/46/66 | `d6a284296bb5b1bfaab2173a28b2d52ffb5669522e8987751a0afd33175f2b0e` |
| GRPO | 1,600 | 1,288/148/164 | `3cc66aa262967b18dee023cbe97336dd875c9db3b995015dedb629fc45abb22b` |

OASST1 只保留 en/zh、全人工、review 通过、ready、未删除且 assistant rank 0
的祖先链。HelpSteer3 只保留非零偏好的 en/zh 单轮 pair。MSVAMP 的前 200 个
稳定 hash source group 进入 SFT，后 800 个进入 GRPO，两者交集为 0。

MSVAMP 上游文件名是 `test_Chinese.json`，但本项目把它明确标记为 training。
不得再用 MSVAMP 报告独立测试成绩。

物化和 CPU 检查命令：

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
  --kind sft \
  --license Apache-2.0 \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v1/dpo/source.jsonl \
  --output data/prepared/dpo-public-60m-v1 \
  --dataset-id dpo-public-60m-v1 \
  --kind dpo \
  --license CC-BY-4.0 \
  --seed 42 \
  --group-by source_id

python scripts/data.py prepare \
  --input data/raw/public-60m-v1/grpo/source.jsonl \
  --output data/prepared/grpo-public-60m-v1 \
  --dataset-id grpo-public-60m-v1 \
  --kind grpo \
  --license Apache-2.0 \
  --seed 42 \
  --group-by source_id

python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v1/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v1/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16
```

输出目录必须预先不存在。revision、上游 hash、筛选规则或输出 hash 漂移时应
fail-fast，不得手改 manifest 绕过。

## 附录 B：路线选择

如果研究问题是“从零训练 60M 模型的完整生命周期”，执行本文 Native 路线。
如果研究问题只是“SFT/DPO/GRPO 方法是否有效”，优先使用固定 revision 的
Qwen3-0.6B-Base 做独立 transfer 实验，避免 Base 能力不足吞掉后训练信号。
两条路线的参数量、Tokenizer 和底座不同，结果分开报告，不能合并为同一对照。
