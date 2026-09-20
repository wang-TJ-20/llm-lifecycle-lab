# Native-60M 公开数据后训练 v2（public-60m-v2）

本文记录按旧版
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 执行的 v2 轮次结果。
**Phase 1 已完成；现行手册的 R0/R1 续跑亦已执行完毕，两个 large-SFT arm 均未通过
预注册硬门禁，因此 DPO/GRPO 未运行**（见第 9–13 节）。
链接中的现行手册为本次 R0/R1 的执行依据。

## 1. 结论摘要

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 数据物化 | 完成（1 项门禁失败后经确认继续） | Base-v2 语料 510,611,451 train token；SFT/DPO/GRPO 切分与文档表格一致 |
| Base-repeat 计算量对照 | 完成 | dev/test 相对冻结 Base-v1 大幅下降，确认旧 Base 主要缺计算量 |
| Base-v2 唯一语料扩容 | 完成，但门禁失败 | v2 held-out 的数值与 v1 基线一升一降；因评测语料不同，尚不能归因为能力改善或退化 |
| SFT 四臂 | 未执行 | Base 门禁失败，按旧 §11 停止 |
| DPO / GRPO | 未执行 | 同上 |
| R0 交叉评测（现行手册） | 完成 | 3 权重 × 2 held-out，6 份 JSON 齐全；见第 9 节 |
| R1 父模型 baseline（现行手册） | 完成 | 同协议 hash；见第 10 节 |
| R1 large-SFT 两臂（现行手册） | 完成，**两臂门禁均未通过** | dev objective 与 BPB 通过，能力项未达标；见第 11 节 |
| R1 DPO / GRPO（现行手册） | **未执行** | 两个 SFT arm 都失败，按现行 §6.3 / §11 硬性停止 |

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

---

## 9. R0：三个 Base 的 dev 交叉评测（现行手册 §4）

按现行手册从 **R0** 续跑。R0 是诊断，不改权重、不读 test、不产生新门禁。
6 份报告全部为有限数值，`eval_batches = 1024`、`external_data = true`、
`sample_count = 1024`，Manifest 与 v2 §6.2 表格一致（`ok = true`，三对
`stage_source_id_overlap` 全为 0，SFT 10,935 / 894,771、DPO 6,712 / 1,514,923、
GRPO 1,288）。

`--config` 始终使用 checkpoint 所属 run 的 `resolved_config.yaml`，
外部 manifest 只改变评测数据。两套 held-out 的 `data_manifest_sha256`：
v1 = `11a0c0ab…d577b8`，v2 = `ae80609b…6121c5`。

| 权重 | held-out | eval loss | en loss | zh loss | BPB |
| --- | --- | ---: | ---: | ---: | ---: |
| Base-v1 | v1 | 3.199809 | 2.203963 | 4.114944 | 1.189372 |
| Base-v1 | v2 | 5.076120 | 5.054696 | 5.097978 | 2.284374 |
| Base-repeat | v1 | **2.756106** | 1.952536 | 3.494549 | 1.024447 |
| Base-repeat | v2 | 4.746067 | 4.958934 | 4.528888 | 2.135842 |
| Base-v2 | v1 | 3.023885 | 2.446250 | 3.554704 | 1.123981 |
| Base-v2 | v2 | **3.120518** | **2.540050** | **3.712744** | **1.404307** |

按现行手册的解释规则：**Base-v2 在 v2 dev 胜、在 v1 dev 败**，属于
「存在领域 tradeoff，不能称为整体改善或退化」。

- v1 held-out 上 Base-repeat 最好，但它把 v1 语料看过 8 epoch（369.48M token），
  该项优势主要来自重复暴露，不能在 v2 上复现（4.746 vs Base-v2 的 3.121）。
- 同一个 Base-v1 权重在两套 held-out 上的 loss 相差 1.876（3.200 / 5.076），
  远大于旧门禁 ±1% 的判定粒度：**旧 Base 门禁确实在跨语料比较绝对 loss**，
  这一点由 R0 直接证实，v2 §6.2 的归因成立。
- 因此 R0 不给出「哪个 Base 更优」的结论，也不据此停止 SFT。

产物：`runs/evaluations/base-cross-r0/{6 × *.json, summary.tsv}`。

## 10. R1：两个父模型的统一 capability baseline（现行手册 §5）

两个 parent 使用同一 `lifecycle-v3` suite、同一 v2 pretrain manifest、
同一 `native-chat-v1`、greedy、`max_new_tokens=32`、单线程。
两份报告的 `protocol_sha256` 相同：
`b99400ff7074086539d835f689265f466b30f2f5124ff26fba622322ac8bd1fa`（去重计数 = 1）。

| 指标 | Base-repeat | Base-v2 |
| --- | ---: | ---: |
| instruction.success | 0 / 32 | 0 / 32 |
| format.success | 0 / 24 | 0 / 24 |
| qa.success | 0 / 32 | 0 / 32 |
| multiturn.success | 0 / 24 | 0 / 24 |
| verifiable.reward | 0 / 112 | 0 / 112 |
| preference.accuracy | 0.718750 | 0.656250 |
| corpus.bpb.all | 1.986153 | **1.693502** |
| corpus.bpb.en | 1.527985 | **1.340105** |
| corpus.bpb.zh | 2.487817 | **2.080450** |
| continuation.repeated_trigram | 0.062500 | 0.108333 |

两个 parent 在四类生成任务上都是全 0：和 v1 冻结结论
（`public-posttrain-v3` 的 Base/SFT/DPO/GRPO 四项同为 0）一致。
Base-v2 的 BPB 明显更好（1.6935 vs 1.9862），但这来自它在 v2 语料上训练，
而 `corpus.bpb.*` 读的正是 v2 语料——因此 BPB 在本轮只能做**父子同分布对照**
（用于不超过父模型 1.02 倍的回归检查），不能作为跨父模型的能力比较。

## 11. R1：两个 large-SFT arm（现行手册 §6）

### 11.1 预算与执行

两臂配置除 `model.init_checkpoint` 外逐字段一致（脚本断言通过：
`max_train_tokens = 2,000,000`、`learning_rate = 2e-5`、`seed = 42`）。
两臂实际 batch/预算完全相同：

| run_id | parent | steps | 监督 token | 覆盖 | dev loss（step 0 → 最终） | 最终 train loss | 耗时 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `native-sft-public-60m-r1-repeat-large-s42` | Base-repeat | 383 | 2,002,968 | 1.001484 | 5.082605 → **4.185979** | 3.938592 | 73.1 s |
| `native-sft-public-60m-r1-basev2-large-s42` | Base-v2 | 383 | 2,002,968 | 1.001484 | 3.989625 → **3.507844** | 3.290331 | 74.5 s |

两臂 clean Git（`code.dirty = false`）、`run_manifest.status = completed`，
`repository_commit = 6cf411ff77e3234dd35b863df6d352ea90093342`。

### 11.2 门禁结果：两臂均失败

| 检查 | 阈值 | repeat arm | basev2 arm |
| --- | --- | --- | --- |
| dev objective 相对 step 0 ↓ ≥5% | ≤ 4.8285 / ≤ 3.7901 | 4.1860 通过 | 3.5078 通过 |
| instruction.success ≥4 | 32 | **0 失败** | **1 失败** |
| format.success ≥3 | 24 | **0 失败** | **0 失败** |
| qa.success ≥4 | 32 | **0 失败** | **0 失败** |
| verifiable.reward ≥12 | 112 | **0 失败** | **1 失败** |
| instruction/format/qa en 各 ≥1 | 16/12/16 | **0/0/0 失败** | **0/0/0 失败** |
| instruction/format/qa zh 各 ≥1 | 16/12/16 | **0/0/0 失败** | **1/0/0 部分失败** |
| corpus.bpb.all ≤ 1.02× parent | ≤ 2.0259 / ≤ 1.7274 | 1.908904 通过 | 1.713292 通过 |
| 跨阶段 source group 交集 = 0 | 全 0 | 通过 | 通过 |
| **合计** | | **ok = false** | **ok = false** |

产物：`runs/gates/native-sft-public-60m-r1-{repeat,basev2}-large-s42.json`。

### 11.3 归因：objective 达标、能力未达标，且两臂同源

两臂都完成了这类低成本筛选臂应该完成的部分——dev objective 下降 12.1%–17.6%，
BPB 相对父模型没有退化，跨阶段数据隔离为 0——但四类任务的成功数几乎全部为 0。这个「只改善自身 objective」
的形态完全落在现行手册 §11 的最后一行：**不视为能力收益**。

失败不是偶发算子错误，而是**数据分布分离不足**：

1. `public-60m-v2` 的 SFT 训练集 10,935 条里，`open_qa` 3,260 + `baike` 1,768
   + `brainstorming` 837 + `general_qa` 683 占 60% 以上，来源以
   `databricks-dolly-15k`（5,031）与 `HC3-Chinese`（5,019）为主；回答中位数
   **124 字符**、p90 318 字符。
2. `lifecycle-v3` 的 instruction/format/qa/verifiable 全部要求
   **精确串或整数匹配**（如 “Reply only Lima or Bern.”、JSON 键整数值、
   “Write back the number 115838 using digits only.”）。
3. 实测输出形态已学会 assistant 轮次（如 `Lima is Lima.`、
   `1. The key "count" with the integer value 3.`），但不会收敛到被要求的
   最小答案串；长答案先验直接违背精确匹配规则。
4. 同一现象在历史资料里可复现：v1 冻结成绩单 `public-posttrain-v3` 的
   SFT/DPO/GRPO 四项同为 0；仓库内唯一拿到非零值的是合成/模板数据的
   `sft-v3`（0.25 / 0.21 / 0.34 / 0.27），仍远低于 4/3/4/12 的门禁。

因此本轮 SFT 失败应归因于「**公开长答案数据与 lifecycle-v3 精确匹配探针的分布
不匹配 + 2M assistant token 预算下 62.93M 模型不足以学会压缩到最小答案串**」，
而不是某一臂的优化缺陷（两臂 dev/bpb 行为一致），也不是评测链路故障
（同一份报告里 `corpus.bpb.*`、`continuation.*`、`preference.*` 全部产出有效读数）。

## 12. 停止决策

按现行手册 §6.3 的显式检查执行，返回 STOP：

```text
STOP: both R1 SFT arms failed; do not run DPO or GRPO   (exit=1)
```

依据同一节与 §11 矩阵：

- 两个 SFT arm 都失败 → **停止 DPO/GRPO**，不生成 `runs/gates/sft-r1-winner.json`，
  因此也不存在 SFT winner、DPO winner、GRPO winner 与最终成绩单
  `runs/evaluations/public-posttrain-r1`。
- 不现场降低门槛、不改写 `check_stage_gate.py` 的预注册定义、不为拉高成功数
  改动 `max_new_tokens` 或 suite 规则。
- 也不因为「SFT 已经跑完」而把它当成通过：「阶段训练完成」不能替代
  「阶段门禁通过」。

DPO、GRPO qualification 与三个 KL arm 未运行，这是**协议要求的正确终端状态**，
不是遗漏。

## 13. 承接：先执行 SFT-v3，再决定是否 Base-v3

现行
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md)
已根据本轮结果推进到 SFT-v3。新的执行顺序是：

1. 对保留的 R1 step-50 至 step-383 checkpoint 做 D0 诊断，不更新权重；
2. 冻结 `lifecycle-v4` sealed test，最终选型前不读取；
3. 物化 `public-60m-v3`，引入固定 revision/hash 的 SQuAD、CMRC 人工短答案；
4. 对公开标注答案只做确定性 JSON 序列化，不生成语义答案；
5. 用 supervised-token quota 固定 general/short QA/classification/numeric/
   structured 为 40%/25%/10%/15%/10%；
6. 从 Base-v2 运行 old-data/8M、balanced-data/2M、balanced-data/8M，与已完成的
   old-data/2M 形成析因矩阵；
7. 只有 balanced/8M 呈现预注册趋势但仍未过门禁，才运行 `5e-5` 敏感性臂；
8. 只有 SFT 硬门禁通过后才允许 DPO 和 GRPO。

这一步优先于 Base-v3，因为当前证据已定位到后训练数据和 token 权重问题；
直接更换 Base 不能补足短输出和结构化格式监督。只有 SFT-v3 全部候选仍失败，
才进入 compute-matched Base-v3 设计。

## 14. 本轮产物

```text
runs/qualifications/public-60m-v2-data-check-r1.json
runs/evaluations/base-cross-r0/{6 × *.json, summary.tsv}
runs/evaluations/r1-parent-repeat/{report.json,report.md}
runs/evaluations/r1-parent-basev2/{report.json,report.md}
runs/native-sft-public-60m-r1-repeat-large-s42/    （含 metrics.jsonl 等全套）
runs/native-sft-public-60m-r1-basev2-large-s42/    （含 metrics.jsonl 等全套）
runs/evaluations/native-sft-public-60m-r1-repeat-large-s42/{report.json,report.md}
runs/evaluations/native-sft-public-60m-r1-basev2-large-s42/{report.json,report.md}
runs/gates/native-sft-public-60m-r1-repeat-large-s42.json
runs/gates/native-sft-public-60m-r1-basev2-large-s42.json
```

未生成（按 §12 要求逐一核对）：`runs/gates/sft-r1-winner.json`、
`runs/gates/dpo-r1-winner.json`、`runs/gates/grpo-r1-winner.json`、
`runs/qualifications/native-grpo-public-60m-r1-parent-s42/qualification.json`、
`runs/evaluations/public-posttrain-r1/`。

最终 checkpoint 绝对位置与 SHA-256（中间 step-00000050 … step-00000350 保留未删）：

| run_id | 绝对路径（`model/model.pt`） | SHA-256 |
| --- | --- | --- |
| `native-sft-public-60m-r1-repeat-large-s42` | `/root/zitong/llm-lifecycle-lab/runs/native-sft-public-60m-r1-repeat-large-s42/checkpoints/step-00000383` | `8a490087a291e275469c0802bdb36cca881c4a5c0dc4b38686799b4e61d00f45` |
| `native-sft-public-60m-r1-basev2-large-s42` | `/root/zitong/llm-lifecycle-lab/runs/native-sft-public-60m-r1-basev2-large-s42/checkpoints/step-00000383` | `60ae22a57c2d7b21d3b5378c1e612efecbfc45d7502f03de7551a250b44c8c64` |

### 14.1 执行环境偏差（两项，已登记）

1. **手册 §4 的 shell 片段有一处变量名冲突，已修正后执行。**
   原文 `eval_model()` 用 `$name-on-v1` 与 `$name-on-v2` 调用 `eval_cross_dev()`，
   而 `eval_cross_dev()` 内部第一条语句就是 `name=$1`。bash 函数不隔离辅助变量，
   第二次调用实际展开成 `<model>-on-v1-on-v2`，summary 循环会在缺失的
   `basev1-on-v2.json` 上失败。本次把两层的形参分别改名为 `target` / `model_name`
   并加 `local`，输出文件名与手册 §12 要求的命名一致；评测本身的内容未被改动。
2. **共享 GPU。** 同一张 RTX 4090 上还有其它进程，训练中出现过
   `CUDACachingAllocator` 的分配重试日志，训练仍正常完成。
   耗时与显存只描述本次环境，不用于跨设备比较（与 v1 记录口径一致）。
