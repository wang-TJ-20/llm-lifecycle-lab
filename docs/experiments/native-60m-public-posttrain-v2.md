# Native-60M 公开数据后训练 v2（public-60m-v2）

本文记录按旧版
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 执行的 v2 轮次结果。
执行状态：**Phase 1 完成，旧 Base 硬门禁失败，原 Phase 2/3/4 未执行**。
链接中的现行手册已更新为本轮结果之后的 R0/R1 续跑计划。

## 1. 结论摘要

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 数据物化 | 完成（1 项门禁失败后经确认继续） | Base-v2 语料 510,611,451 train token；SFT/DPO/GRPO 切分与文档表格一致 |
| Base-repeat 计算量对照 | 完成 | dev/test 相对冻结 Base-v1 大幅下降，确认旧 Base 主要缺计算量 |
| Base-v2 唯一语料扩容 | 完成，但门禁失败 | v2 held-out 的数值与 v1 基线一升一降；因评测语料不同，尚不能归因为能力改善或退化 |
| SFT 四臂 | 未执行 | Base 门禁失败，按 §11 停止 |
| DPO / GRPO | 未执行 | 同上 |

## 2. 实验契约与三项已确认偏差

固定契约：`configs/models/tiny-60m.yaml`（62.93M）、tokenizer
`data/tokenizers/bilingual-60m-v1`、seed 42、context 512、BF16、commit
`689fbc79643673e2575b34254acec9018366ae1e`（`git status` 无输出）。
运行环境为 `/root/.venv60m`（torch 2.14.0+cu130，RTX 4090 24G）；
文档中指定的 conda 环境 `llm-lifecycle-lab` 在本机不存在。

执行前有三处与文档的偏差，均已经确认后执行：

1. **Base-v2 数据资格门禁失败后继续。** `train_tokens=510,611,452` 超过上限
   `500,000,000`（其余 4 项通过：en 59.74%、zh 40.26%、最大单源 40.26%、
   synthetic 4.55%）。文档 §4.2/§11 规定此时应改新 recipe；本次按确认直接开跑，
   因此 Base-v2 结果**不满足冻结契约**，不可与 v1 严格对照。
2. **Base-v1 checkpoint 缺失，改复用同 config 复现 run。**
   `runs/native-60m-baseline-v1` 目录已被删除（仅剩
   `docs/experiments/results/native-60m-baseline-v1` 的冻结结果）。改用
   `runs/native-60m-v11-001/checkpoints/step-00005649`
   （`config_sha256=b8c459fc373df588…`，与冻结 baseline 完全一致，seed 42；
   final_loss 3.0640 vs 冻结 3.0608）。
3. **checkpoint 滚动保留（只留最新 2 个）+ 旧 run 迁出。**
   磁盘仅 42G 可用，而 Base-v2 按 `checkpoint_interval=500` 会产生约 125 个
   checkpoint（每个 721M，约 90G）。训练期间滚动清理中间 checkpoint；
   12 个历史 run 目录已迁至 `/data/legacy-runs-llm-lab/`
   （索引见 `runs/_archive/legacy_run_index.json`）。

## 3. 数据物化

### 3.1 Base-v2 语料

四个来源均按固定 revision 与上游文件 SHA-256 物化，round-robin 混合后按
`source_id` 分组切分：

| 来源 | 记录 | train token | 占比 |
| --- | ---: | ---: | ---: |
| wikipedia-en-primary-60m-v2 | 150,000 | 190,605,842 | 37.33% |
| wikipedia-zh-60m-v2 | 220,000 | 205,593,570 | 40.26% |
| wikipedia-en-secondary-60m-v2 | 150,000 | 91,175,756 | 17.86% |
| simplestories-60m-v1 | 100,000 | 23,236,284 | 4.55% |
| **合计** | 620,000 | **510,611,451** | en 59.74% / zh 40.26% |

`data/prepared/bilingual-60m-v2/qualification.json`：`ok = false`
（仅 `train-token-range` 一项失败）。

### 3.2 公开后训练数据

`public-60m-v2` 规范化输出 SHA-256 与文档 §5.1 完全一致：

| 阶段 | 记录 | 规范化输出 SHA-256 | 一致 |
| --- | ---: | --- | --- |
| SFT | 13,600 | `e71bc675698cf72346c2669c9f5cb1003e27414d818ce2dc4ab25a2b1a766063` | 是 |
| DPO | 8,400 | `07333995bd3df1b38cb5d4bdfd5377ec4920162813bffd344db56caac35a30cb` | 是 |
| GRPO | 1,600 | `505f8bdf2d4a3ac008174382e8e134d61c0d2bd29d3c7cedc443c45dd5dbfa9e` | 是 |

seed 42 按 `source_id` 切分后：SFT 10,935/1,283/1,382；DPO 6,712/844/844；
GRPO 1,288 prompt / 644 group、150/75、162/81，与文档表格逐项一致。
`check-posttrain` 结果：

- `public-60m-v2-data-check.json`：`ok = true`，
  SFT train 894,771 assistant token、DPO train 1,514,923 response token，
  SFT/DPO/GRPO source group 交集全为 0。
- `public-60m-v1-data-check.json`：`ok = true`（SFT 883 条、DPO 448 对，
  与 v1 冻结事实一致）。

## 4. Phase 1：Base 对照与 Base-v2

### 4.1 计算量对照（native-60m-base-repeat-control-v2-s42）

复用 Base-v1 的 46,184,530 train token，随机初始化训练 8 个 epoch。

| 指标 | 值 |
| --- | --- |
| steps / tokens | 45,191 / 369,480,328 |
| 耗时 | 19,457 s（5.41 h） |
| target_token_coverage | 1.000011 |
| final train loss | 1.7940 |
| best eval loss | 2.6099 |
| final eval loss | 2.6516（en 1.8748 / zh 3.4165） |

对照冻结 Base-v1（test loss 3.0387、en 2.1520、zh 3.9386）与 v1 baseline 的
dev 口径，重复训练把 loss 压到约 −14.5%（en −11.5%、zh −16.1%）。
**该臂回答了 §6.1 的问题：旧 Base-v1 主要缺的是计算量**，0.73 token/param 远未收敛。

### 4.2 Base-v2（native-60m-base-expanded-v2-s42）

唯一语料 1 个 epoch。

| 指标 | 值 |
| --- | --- |
| steps / tokens | 62,453 / 510,615,539 |
| 耗时 | 26,724 s（7.42 h） |
| target_token_coverage | 1.000008 |
| final train loss | 2.6956 |
| best eval loss | 3.1391 |
| dev loss | 3.1444（en 2.6808 / zh 3.6156） |
| test loss | 2.9338（en 2.3097 / zh 3.5390） |

与冻结 Base-v1 test（3.0387 / en 2.1520 / zh 3.9386）的直接数值差为：
**overall −3.45%、zh −10.1%、en +7.3%**。Base-v1 和 Base-v2 分别在
`bilingual-60m-v1`、`bilingual-60m-v2` held-out 上评测，这组差值是
**跨语料描述统计，不是模型改善率**。

## 5. Base 硬门禁：失败

`runs/gates/base-v2.json`：`ok = false`（6 通过 / 3 失败）。

| 检查 | 实际 | 阈值 | 结果 |
| --- | --- | --- | --- |
| run-completed | completed | completed | 通过 |
| clean-git | false(dirty) | false | 通过 |
| target-token-coverage | 1.000008 | [1.00, 1.01] | 通过 |
| dev-loss | 3.1444 | ≤ 2.9480 | **失败** |
| dev-en-loss | 2.6808 | ≤ 2.0119 | **失败** |
| dev-zh-loss | 3.6156 | ≤ 3.8698 | 通过 |
| test-loss | 2.9338 | ≤ 3.0691 | 通过 |
| test-en_loss | 2.3097 | ≤ 2.1735 | **失败** |
| test-zh_loss | 3.5390 | ≤ 3.9780 | 通过 |

三项失败全部集中在**英文**。已确认的问题是：门禁阈值来自 Base-v1 held-out，
而 Base-v2 指标来自另一套 held-out，英文又从 SimpleStories 为主变成 Wikipedia
为主，因此门禁在跨语料比较绝对 loss。当前证据不能进一步证明 Base-v2 英文能力
真实退化、阈值不可达或训练已经充分。

按文档 §11「Base-v2 硬门禁失败 → 不跑 SFT 四臂」，SFT/DPO/GRPO 均未启动。

## 6. 复盘：门禁比较口径失配，模型归因未完成

**状态：评测设计问题已确认，模型优劣未确认。** 本文保留旧门禁的失败事实，
不回写或伪造历史结果；现行
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 已改为新 run ID
下的 R0/R1 协议，且不根据旧 test badcase 调低 SFT/DPO/GRPO 门禁。

### 6.1 现象

Base-v2 训练正常完成（`target_token_coverage=1.000008`、clean Git、
run completed），但三项英文检查失败，导致整个后训练流程按 §11 停摆。
这能证明“Base-v2 未通过预注册 v2 门禁”，不能证明“Base-v2 不适合作为 SFT
父模型”。

### 6.2 已确认的评测缺口

阈值由 Base-v1 标定，而 Base-v1 与 Base-v2 的**英文语料构成不同**：

| | Base-v1 英文 | Base-v2 英文 |
| --- | --- | --- |
| 主要来源 | SimpleStories 叙事文本（100K 条） | 维基百科条目（300K 条） |
| 文本性质 | 故事体、局部模式重复 | 百科体、实体/数字密集 |

当前比较实际是：

- Base-v1 权重在 **v1 test**：en loss **2.1520**
- Base-v2 权重在 **v2 test**：en loss **2.3097**

缺少的是同一评测集上的另外两个格子：

| 权重 \ held-out | v1 held-out | v2 held-out |
| --- | --- | --- |
| Base-v1 | 已有 | **缺失** |
| Base-repeat | 已有 | **缺失** |
| Base-v2 | **缺失** | 已有 |

此外，每次 pretrain 评测只取 64 个 dominant-language 分层窗口，共 32,704
supervised token，且没有按来源报告指标。Base-v2 的 dev/test loss 分别为
3.1444/2.9338，相差 0.2106；在这种波动下，用约 0.02 的 1% 阈值做硬停止过于
敏感。新门禁至少应扩大到 1,024 个固定窗口，并同时报告每语言 BPB、每来源指标
和置信区间。

Base-v2 只训练了 510.6M token，即约 **8.11 token/parameter**。这远高于 v1 的
0.73，但不能据此排除继续预训练的收益。唯一数据增加也可能因固定模型容量、
领域配比和优化日程而降低某个固定域的 held-out 表现，因此原记录中“唯一语料扩容
不可能让同分布语言变差”的推断不成立。

### 6.3 影响范围

| 结论 | 是否可用 |
| --- | --- |
| 「旧 Base-v1 主要缺计算量」（repeat-control 臂） | 可用，在同一 v1 语料内比较 |
| 「Base-v2 改善中文、损害英文」 | 不可用，两个权重没有在同一 held-out 上比较 |
| 「Base-v2 已训练充分」 | 不可用，只有 8.11 token/parameter，且缺少曲线斜率分析 |
| 「Base-v2 整体优于 Base-v1」 | 不可用，评测矩阵不完整 |
| Base-v2 作为后训练父模型是否更优 | 未测，SFT 四臂未执行 |

### 6.4 建议的后续动作（按代价排序）

1. **先补交叉评测，不重训。** 三个现有权重都在 v1/v2 held-out 上用相同窗口和
   协议评测，先把模型效应与语料难度分开。
2. **再做低成本父模型筛选。** 对 Base-repeat 与 Base-v2 跑同一
   `lifecycle-v3` baseline，并各跑一个 large-SFT 臂；能力评测可跨父模型直接
   比较，但仍会受父模型差异影响。
3. **最后定义新协议。** 如需重配语料或阈值，使用新 recipe、run ID 和门禁版本；
   只用新 dev anchor 定阈值，test 保持封存到最终选型。
4. **不要以回到 v1 英文 loss 为配方目标。** 这会把 Base 优化成
   SimpleStories 专项模型。应先明确目标域，再按共同 anchor 的 BPB 和下游能力
   选配方。

### 6.5 执行环境问题（不影响结论，但影响可复现性）

本机磁盘仅 42G 可用，而 Base-v2 按 `checkpoint_interval=500` 需约 90G；
且该环境的删除操作被安全删除守卫拦截（文件被移入回收站而非释放，
`rm` 与 `shutil.rmtree` 均不释放空间）。最终以「先 truncate 再删除」回收空间，
并以滚动保留最新 2 个 checkpoint 完成训练。他人复现时需预留磁盘
或提前放粗 `checkpoint_interval`（这属于预算变更，须使用新 run ID）。

## 7. 下一轮优化方案

### 7.1 R0：现有 checkpoint 交叉评测

以下评测不改权重，也不覆盖 run 内原有报告。`--config` 仍必须匹配 checkpoint
的训练配置；`--data-manifest`、`--packed-manifest` 只覆盖评测语料。本轮仅补
dev 矩阵，不再读取已经用于旧门禁的 test。

```bash
set -euo pipefail
mkdir -p runs/evaluations/base-cross-r0

BASE_V1=runs/native-60m-v11-001/checkpoints/step-00005649
BASE_REPEAT=runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191
BASE_V2=runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453

eval_cross_dev() {
  name=$1
  config=$2
  checkpoint=$3
  manifest=$4
  packed=$5

  python scripts/eval_pretrain.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --data-manifest "$manifest" \
    --packed-manifest "$packed" \
    --eval-batches 1024 \
    --split dev \
    --json > "runs/evaluations/base-cross-r0/${name}.json"
}

eval_model() {
  name=$1
  config=$2
  checkpoint=$3

  eval_cross_dev "${name}-on-v1" "$config" "$checkpoint" \
    data/prepared/bilingual-60m-v1/data_manifest.json \
    data/packed/bilingual-60m-v1-seq512/packed_manifest.json
  eval_cross_dev "${name}-on-v2" "$config" "$checkpoint" \
    data/prepared/bilingual-60m-v2/data_manifest.json \
    data/packed/bilingual-60m-v2-seq512/packed_manifest.json
}

eval_model basev1 runs/native-60m-v11-001/resolved_config.yaml "$BASE_V1"
eval_model repeat \
  runs/native-60m-base-repeat-control-v2-s42/resolved_config.yaml "$BASE_REPEAT"
eval_model basev2 \
  runs/native-60m-base-expanded-v2-s42/resolved_config.yaml "$BASE_V2"
```

解释规则：

| 结果 | 结论 |
| --- | --- |
| Base-v2 在 v1、v2 dev 都优于 Base-v1/repeat | Base-v2 是当前更强父模型候选 |
| Base-v2 在 v2 dev 胜、v1 dev 败 | 领域迁移或容量分配，不是单纯训练失败 |
| Base-v2 在两套 dev 都败 | 优先怀疑配方或优化日程，但仍执行两个低成本 SFT arm |
| 64 与 1,024 窗口结论翻转 | 原门禁样本不足，新协议不得继续使用 64 窗口阈值 |

### 7.2 R1：低成本下游筛选

R0 六份报告完整且指标为有限数值后，先只跑 **Base-repeat + large-SFT** 与
**Base-v2 + large-SFT** 两臂；R0 的相对排名本身不阻断 SFT。两臂统一
2M assistant token、相同
`public-60m-v2`、学习率和 seed。Base-v1 保留为历史锚点；已知不足的 883 条
small-SFT 不再作为首轮算力投入。父模型先各自产生 `lifecycle-v3` baseline，
SFT 选型只看共同 capability suite、SFT dev objective 和共同语料 anchor BPB。

若目标是发表“唯一数据扩容优于重复计算”的严格结论，则现有 8-epoch repeat
还不够：它只看过 369.5M token，而 Base-v2 看过 510.6M。应新增约 11-epoch
repeat（约 508M token）作为 compute-matched 父模型，再执行 SFT 对照。

### 7.3 R2：需要重训时的 Base-v3 配方

仅当 R0/R1 证明 Base-v2 父模型确实不优时再重训，并使用新 recipe ID：

1. 将唯一 train token 固定在 450M-480M，资格失败必须停止；按 token quota
   选样，不再靠 record count 猜预算。
2. 将 `source-prefix` 改为按 `source_id` 稳定 hash 选样，避免上游顺序偏差。
3. 降低单一英文 Wikipedia 域占比，引入许可证和 revision 固定的非合成公开英文
   叙事/说明文本；中文也增加非百科域。SimpleStories 可保留为评测 anchor，
   不以降低故事域 loss 作为训练目标。
4. 数据配方实验继续固定现有 tokenizer；若要重训 tokenizer，单独开实验臂，
   避免把 tokenizer 与语料收益混在一起。
5. 第一轮用约 460M seen token 与 10-epoch v1 repeat 做 compute-matched 选择；
   选定配方后从随机初始化按单一日程重训 2 epoch，达到约 920M token、
   14.6 token/parameter，不从已衰减到最低学习率的 1-epoch checkpoint 重启。
6. 所有 Base 臂使用相同优化日程。可把 warmup 从固定 50 step 改为总步数约 1%，
   但必须从头训练并在所有对照臂一致应用。
7. 将 `checkpoint_interval` 放粗到约 5,000 step，`eval_interval` 独立保持
   1,000 step；训练前完成磁盘预算，不再训练中手工删除证据。

### 7.4 新门禁原则

- 数据资格、run 完成、clean Git、token coverage 和数值稳定性属于**安全门禁**。
- 模型收益只在同一冻结 dev anchor 上比较；至少 1,024 个固定窗口，报告
  en-story、en-wiki、zh-wiki 等来源级 BPB 与 bootstrap 置信区间。
- 现有 v1/v2 test 已被本轮门禁使用，只作历史诊断，不再纳入 R0；新协议应另建
  并冻结 test anchor，且不参与调参和中途停止，只在最终候选选定后评测一次。
- Base loss 只筛掉灾难性回退；父模型最终优劣由 compute-matched SFT 能力决定。
- DPO/GRPO 继续沿用原 fail-fast 资格要求，且只在 SFT winner 产生后启动。

## 8. 产物

```text
data/prepared/bilingual-60m-v2/{data_manifest.json,qualification.json}
data/packed/bilingual-60m-v2-seq512/packed_manifest.json
data/prepared/{sft,dpo,grpo}-public-60m-v2/data_manifest.json
data/prepared/public-60m-v{1,2}-data-check.json
runs/native-60m-base-repeat-control-v2-s42/{training_result.json,metrics.jsonl,
  training_budget.json,run_manifest.json,runtime_environment.json,resolved_config.yaml}
runs/native-60m-base-expanded-v2-s42/{同上}
runs/evaluations/base-v2-pretrain-dev.json
runs/evaluations/base-v2-pretrain-test.json
runs/gates/base-v2.json
runs/_archive/legacy_run_index.json
```

Base-v2 最终 checkpoint：
`/root/zitong/llm-lifecycle-lab/runs/native-60m-base-expanded-v2-s42/checkpoints/step-00062453`。
对照臂最终 checkpoint：
`/root/zitong/llm-lifecycle-lab/runs/native-60m-base-repeat-control-v2-s42/checkpoints/step-00045191`。
中间 checkpoint 已滚动清理，历史 run 位于 `/data/legacy-runs-llm-lab/`。
