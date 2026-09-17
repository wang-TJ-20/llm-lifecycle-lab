# 公开数据 SFT、DPO 与 GRPO 路线

本文给出 Native-60M 推荐的后训练路线。SFT、DPO、GRPO 均从固定 revision
的公开数据构建；本地完成下载、校验、切分和 CPU 加载检查，CUDA 服务器只负责
训练与评测。

旧的 `build_posttraining_data.py` 和不带 `public` 的 60M pipeline 仅用于复现
已经记录的合成数据实验，不再作为新训练默认入口。

## 1. 数据源与许可

配方固定在
`src/llm_lifecycle_lab/data/posttraining_recipes/public-60m-v1.yaml`：

| 阶段 | 数据源 | 固定 revision | 文件 SHA-256 | 许可 |
| --- | --- | --- | --- | --- |
| SFT | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e` | Apache-2.0 |
| DPO | `nvidia/HelpSteer3` | `f6d145777bcbde96137596340fab89793acd1031` | `32b52e1d378f8dab1e4c9ae549da49a5d6fc0875aeafe3f9139e6053beb906bb` | CC-BY-4.0 |
| SFT / GRPO | `Mathoctopus/MSVAMP` | `301e2b3b168be70058c89c21c2fbdc9262102add` | `31e136303b56ed1734d3e9fbcdcff4499f974a0857298816c41e1c5c0be6b4bd` | Apache-2.0 |

运行下载命令表示操作者已核对并接受相应许可；脚本要求许可集合精确匹配，
不会静默补充或忽略许可。发布模型或衍生数据时仍需保留数据集归属和许可信息，
尤其是 HelpSteer3 的 CC-BY-4.0 attribution。

MSVAMP 的固定上游文件名是 `test_Chinese.json`，但本配方在每条记录的 metadata
中写入 `intended_use=training`。从采用本配方开始，**不得再用 MSVAMP 或该文件
报告独立测试成绩**。正式效果评测使用未参与训练的 `lifecycle-v3`；
该 suite 只用于泄漏门禁和阶段前后测量，不进入任何训练文件。

## 2. 确定性选择与隔离

| 来源 | 选择规则 | 进入阶段 |
| --- | --- | --- |
| OASST1 | 仅 `en/zh`；整条祖先链均为人工、review 通过、未删除、ready；所有 assistant 都是 rank 0；最多 8 条消息且总字符不超过 500；按稳定 hash 每种语言取 350 条 | SFT |
| HelpSteer3 | 仅单轮 `en/zh`；偏好非零；chosen/rejected 不同；总字符不超过 500；规范化去重后按稳定 hash 每种语言取 280 对 | DPO |
| MSVAMP | 校验恰好 1000 个双语 source group；按稳定 hash 排序 | 前 200 组进入 SFT，后 800 组进入 GRPO |

MSVAMP 的英文和中文版本共用 `source_id`，所以同一题不会跨 split。SFT warmup
与 GRPO 使用互斥的 source group，交集必须为 0。三个阶段 prepare 时均使用
`--group-by source_id`，避免同源记录跨 train/dev/test。

规范化输出的固定 SHA-256：

| 阶段 | 记录数 | 输出 SHA-256 |
| --- | ---: | --- |
| SFT | 1,100 | `4286e08a52910b027c2c3d0b7ab1b0333faec11792d918d6a3cc7c483be8110f` |
| DPO | 560 | `d6a284296bb5b1bfaab2173a28b2d52ffb5669522e8987751a0afd33175f2b0e` |
| GRPO | 1,600 | `3cc66aa262967b18dee023cbe97336dd875c9db3b995015dedb629fc45abb22b` |

revision、上游文件 hash、筛选规则或依赖解析结果发生漂移时，物化会 fail-fast，
不会生成部分可用的正式目录。

## 3. 本地下载与准备

先确认环境及配方：

```bash
conda activate llm-lifecycle-lab
python -m pip check
python scripts/data.py posttrain-recipes
```

下载并规范化三个公开来源。输出目录必须尚不存在：

```bash
python scripts/data.py fetch-posttrain \
  --recipe public-60m-v1 \
  --output data/raw/public-60m-v1 \
  --accept-license Apache-2.0 \
  --accept-license CC-BY-4.0
```

显式校验三个 JSONL，再按来源组切分：

```bash
python scripts/data.py validate \
  --input data/raw/public-60m-v1/sft/source.jsonl \
  --kind sft
python scripts/data.py validate \
  --input data/raw/public-60m-v1/dpo/source.jsonl \
  --kind dpo
python scripts/data.py validate \
  --input data/raw/public-60m-v1/grpo/source.jsonl \
  --kind grpo

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
```

固定结果：

| 阶段 | train | dev | test |
| --- | ---: | ---: | ---: |
| SFT | 883 | 116 | 101 |
| DPO | 448 | 46 | 66 |
| GRPO | 1,288 | 148 | 164 |

若数量或 hash 不一致，不要手改 manifest。检查依赖版本、recipe 和输入文件，
然后在新的空目录重建。

## 4. CPU 加载检查

后训练不需要 packing，也不重新训练 Tokenizer；继续使用 Base 的
`data/tokenizers/bilingual-60m-v1`。以下检查会真实读取全部 split，执行
评测集重合门禁，按 seq512 编码，并确认三个阶段的 `source_id` 交集为 0：

```bash
python scripts/data.py check-posttrain \
  --sft-manifest data/prepared/sft-public-60m-v1/data_manifest.json \
  --dpo-manifest data/prepared/dpo-public-60m-v1/data_manifest.json \
  --grpo-manifest data/prepared/grpo-public-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --evaluation-suite configs/evaluation/lifecycle-v3.yaml \
  --sequence-length 512 \
  --max-new-tokens 16

python scripts/validate_config.py configs/pipelines/native-sft-public-60m.yaml
python scripts/validate_config.py configs/pipelines/native-dpo-public-60m.yaml
python scripts/validate_config.py configs/pipelines/native-grpo-public-60m.yaml
```

将仓库、`data/prepared/*-public-60m-v1`、Tokenizer 和 Base 发布包成套同步到
CUDA 服务器。raw 数据建议一并归档，以便审计和重建。

## 5. 远程 CUDA 训练

三阶段必须顺序执行。每次启动前运行 Doctor；后训练的 `real-batch` 当前显示
`WARN` 是已知边界，数据 manifest、CUDA、BF16、磁盘和父 checkpoint 仍必须通过。

```bash
# 1. SFT: 8 epochs, expected final step 112
python scripts/doctor.py --config configs/pipelines/native-sft-public-60m.yaml
python scripts/train_sft.py \
  --config configs/pipelines/native-sft-public-60m.yaml \
  --run-id native-sft-public-60m-001

# 2. DPO: parent is SFT step 112; 3 epochs, expected final step 21
python scripts/doctor.py --config configs/pipelines/native-dpo-public-60m.yaml
python scripts/train_dpo.py \
  --config configs/pipelines/native-dpo-public-60m.yaml \
  --run-id native-dpo-public-60m-001

# 3. GRPO: parent is DPO step 21; 2 epochs, expected final step 161
python scripts/doctor.py --config configs/pipelines/native-grpo-public-60m.yaml
python scripts/train_grpo.py \
  --config configs/pipelines/native-grpo-public-60m.yaml \
  --run-id native-grpo-public-60m-001
```

配置已绑定以下父 checkpoint：

```text
build/modelscope/native-60m-base-v1
  -> runs/native-sft-public-60m-001/checkpoints/step-00000112
  -> runs/native-dpo-public-60m-001/checkpoints/step-00000021
  -> runs/native-grpo-public-60m-001/checkpoints/step-00000161
```

如果 run 中断，只能用同一配置恢复：

```bash
python scripts/train_sft.py \
  --config configs/pipelines/native-sft-public-60m.yaml \
  --resume-run native-sft-public-60m-001
```

DPO/GRPO 同理替换脚本和 run ID。已完成预算的 run 不追加训练；任何数据、
父 checkpoint、学习率或目标函数变化都创建新的 run ID。

## 6. GRPO 资格门槛

MSVAMP 的 800 个 GRPO source group 没有出现在 SFT 中，但这也意味着 60M policy
可能对它们过难，导致一组 8 次采样全部错误。此时 advantage 全为 0，继续长跑
只会浪费算力。

训练前固定以下门槛，不在看到结果后放宽：

1. 首个 eval window 必须同时满足
   `0 < report_success_rate < 1` 且
   `report_zero_variance_groups < 1`；否则停止并记录为无训练信号。
2. 前 25 step 至少有 4 个 prompt group 出现非零方差；若持续全零，不通过
   “可训练性”资格，不能把 loss 或 KL 变化解释为 RL 收益。
3. 完成 run 后，固定 dev rollout 的 `report_reward_mean` 相对 step 0
   至少提高 `0.05`，`report_zero_variance_groups` 不高于 `0.75`，
   且 `report_approx_kl` 不高于 `0.10`。未同时满足时记录零结果，
   再单独设计难度分层或 `kl_beta` 对照。

这些是本次小模型实验的预注册验收线，不是 GRPO 的通用理论阈值。
禁止根据训练结果从 test 中筛题，也不能把 SFT warmup 的 200 组移入 GRPO
来人为提高成功率。

## 7. 同协议评测

`lifecycle-v3` 只做训练外比较。先评测 Base，再依次把前一阶段报告作为 baseline：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-base-v3 \
  --device cuda \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json

python scripts/evaluate_capabilities.py run \
  --checkpoint runs/native-sft-public-60m-001/checkpoints/step-00000112 \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --suite configs/evaluation/lifecycle-v3.yaml \
  --output runs/evaluations/public-sft-v3 \
  --device cuda \
  --prompt-protocol native-chat-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --baseline runs/evaluations/public-base-v3/report.json
```

DPO 与 GRPO 使用相同命令，分别替换 checkpoint、输出目录和前一阶段 baseline。
最终用 `evaluate_capabilities.py compare` 生成纵向成绩单。只有训练目标改善且
instruction、format、QA、multiturn 和 corpus BPB 的非定向退化在预先声明范围内，
该阶段才进入后续比较；训练 loss 下降本身不构成通过。

本路线第一次完整训练的结果、资格门槛判定与边界记录在
[公开数据后训练报告](./experiments/native-60m-public-posttrain-v1.md)：
SFT dev loss 5.1526 → 4.4969，DPO dev 偏好损失 0.6915 → 0.6825，
GRPO 未通过第 6 节门槛，记为预注册零结果。
