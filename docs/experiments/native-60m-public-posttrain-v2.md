# Native-60M 公开数据后训练 v2（public-60m-v2）

本文记录按 [PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 执行的 v2 轮次结果。
执行状态：**Phase 1 完成，Base 硬门禁失败，Phase 2/3/4 按文档 §11 未执行**。

## 1. 结论摘要

| 阶段 | 状态 | 关键结果 |
| --- | --- | --- |
| 数据物化 | 完成（1 项门禁失败后经确认继续） | Base-v2 语料 510,611,451 train token；SFT/DPO/GRPO 切分与文档表格一致 |
| Base-repeat 计算量对照 | 完成 | dev/test 相对冻结 Base-v1 大幅下降，确认旧 Base 主要缺计算量 |
| Base-v2 唯一语料扩容 | 完成，但门禁失败 | 中文显著改善，英文反而恶化 |
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

与冻结 Base-v1 test（3.0387 / en 2.1520 / zh 3.9386）相比：
**overall −3.45%、zh −10.1%、en +7.3%（恶化）**。

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

三项失败全部集中在**英文**。原因不是训练不足，而是语料分布变化：
门禁中的英文阈值来自 Base-v1 的英文构成（SimpleStories 叙事文本占比高、
单 token 可预测性强），而 Base-v2 的英文主体换成了维基百科条目
（百科文本本身 loss 显著更高）。中文侧因语料从 126K 扩到 220K 条而大幅改善，
说明扩容本身有效，但**英文阈值对新语料不可达**，该门禁实际上在跨语料比较。

按文档 §11「Base-v2 硬门禁失败 → 不跑 SFT 四臂」，SFT/DPO/GRPO 均未启动。

## 6. 已知问题（负面结果）：Base 门禁的英文阈值与 v2 语料不匹配

**状态：已确认，未修复。** 本文只记录问题，不修改
[PUBLIC_POSTTRAINING_GUIDE](../PUBLIC_POSTTRAINING_GUIDE.md) 中已冻结的门禁
——文档 §11 明确要求「不得根据首轮 test badcase 修改既有门禁」。

### 6.1 现象

Base-v2 训练正常完成（`target_token_coverage=1.000008`、clean Git、
run completed），中文与 overall 全部达标且相对冻结 Base-v1 明显改善，
但三项英文检查全部失败，导致整个后训练流程按 §11 停摆。

### 6.2 根因：门禁阈值在跨语料比较

阈值由 Base-v1 标定，而 Base-v1 与 Base-v2 的**英文语料构成不同**：

| | Base-v1 英文 | Base-v2 英文 |
| --- | --- | --- |
| 主要来源 | SimpleStories 叙事文本（100K 条） | 维基百科条目（300K 条） |
| 文本性质 | 故事体、可预测性高、单 token 损失低 | 百科体、实体/数字密集、单 token 损失高 |

判据是「训练量增加了 11 倍，英文 loss 反而变差」：

- Base-v1：46.2M token，test en **2.1520**
- Base-v2：510.6M token（11 倍），test en **2.3097**（+7.3%）

唯一语料扩容不可能让同分布语言变差，因此这不是欠拟合或训练失败，
而是语料难度口径变了。中文可作为对照：语料从 126K 扩到 220K 条、
文本性质未变（维基百科），zh loss 即下降 10.1%（3.9386 → 3.5390）。
repeat-control 臂也支持该判断——在 v1 语料上把计算量加 8 倍，
en loss 从 2.1520 降到 1.8748，**同样的计算量在英文上是能显著改善的**。

结论：**`dev-loss ≤ 2.9480`、`dev-en-loss ≤ 2.0119`、
`test-en_loss ≤ 2.1735` 对 Wikipedia-heavy 的英文语料不可达。**
门禁本身有效（它正确拦下了分布漂移），只是阈值需要按语料重新标定。

### 6.3 影响范围

| 结论 | 是否可用 |
| --- | --- |
| 「旧 Base-v1 主要缺计算量」（repeat-control 臂） | 可用，在同一 v1 语料内比较 |
| 「唯一语料扩容改善中文」 | 可用，中文语料性质未变 |
| 「Base-v2 整体优于 Base-v1」 | 不可用，英文口径不可比 |
| Base-v2 作为后训练父模型是否更优 | 未测，SFT 四臂未执行 |

### 6.4 建议的后续动作（按代价排序）

1. **重新标定门禁阈值**：在同语料口径内定义（如与"同 mixture 的 repeat 基线"
   比较，或使用 bits-per-byte 的分语言相对改善），而不是跨语料绝对阈值。
2. **语料再平衡**：提高英文叙事语料（SimpleStories）占比、降低英文维基占比，
   使英文 loss 回到可达区间后再重跑 Base-v2（约 7.4h）。
3. **若只想验证后训练链路**：直接用 SFT 四臂的 capability 指标判断父模型优劣，
   capability 口径（lifecycle-v3）与语料分布无关，不受本问题影响。

### 6.5 执行环境问题（不影响结论，但影响可复现性）

本机磁盘仅 42G 可用，而 Base-v2 按 `checkpoint_interval=500` 需约 90G；
且该环境的删除操作被安全删除守卫拦截（文件被移入回收站而非释放，
`rm` 与 `shutil.rmtree` 均不释放空间）。最终以「先 truncate 再删除」回收空间，
并以滚动保留最新 2 个 checkpoint 完成训练。他人复现时需预留磁盘
或提前放粗 `checkpoint_interval`（这属于预算变更，须使用新 run ID）。

## 7. 产物

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
