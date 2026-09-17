# Native GRPO / RLVR 最小闭环

这一阶段只处理**可程序验证奖励**（RLVR）。不接入 LLM Judge，也不把格式启发式
包装成人类偏好。目标是先验证 on-policy rollout、组内优势、KL 和恢复链路。

推荐的 60M 配方使用与 SFT warmup 零 source-group 交集的 800 组公开 MSVAMP
题目，完整数据与训练流程见
[公开数据 SFT、DPO 与 GRPO 路线](./PUBLIC_POSTTRAINING_GUIDE.md)。

## 1. CPU 机制实验

```bash
python scripts/grpo_experiment.py --mode train
python scripts/grpo_experiment.py --mode resume
```

脚本在临时目录完成微型 Pretrain、SFT 和三步 GRPO。为了让随机小模型确实产生
有正有负的组内奖励，fixture 会先从初始 SFT 模型采样，再选择其中一个输出作为
合成 exact answer。这是刻意构造的机制测试，不是独立评测集，也不产生能力结论。

验收内容：

- 每个 prompt 采样 4 个回答，固定 prompt/group seed；
- 只使用程序化 verifier 计算 0/1 奖励；
- 组内标准化 advantage，零方差组安全地产生零 advantage；
- policy 相对于冻结父模型计算 KL；
- 父 SFT checkpoint 不被修改；
- 保存最近一批 rollout、输出和奖励；
- 连续训练与中断恢复的权重、优化器、scheduler、RNG 和数据位置一致。

## 2. 数据与 verifier

每行一个可验证问题：

```json
{"id":"en-1","source_id":"math-1","template_id":"integer-add-v1","language":"en","prompt":"What is 12 + 7? Reply with an integer only.","answer":"19","metadata":{"verifier":"integer"}}
{"id":"zh-1","source_id":"math-1","template_id":"integer-add-v1","language":"zh","prompt":"12 加 7 等于多少？只回答整数。","answer":"19","metadata":{"verifier":"integer"}}
```

仅允许三类 verifier：

| verifier | 规则 |
| --- | --- |
| `exact` | 只去除首尾空白后完全一致 |
| `integer` | 整个输出必须是整数，再比较整数值 |
| `json` | 完整解析、拒绝重复键/非有限数，类型和值完全一致 |

没有正则执行、Python 表达式、shell、网络请求或 LLM Judge。
新增 verifier 必须先定义输入域、解析失败行为和针对性测试。

准备数据：

```bash
python scripts/data.py prepare \
  --input data/grpo-source.jsonl \
  --output data/prepared/grpo-bilingual-v1 \
  --dataset-id grpo-bilingual-v1 \
  --kind grpo \
  --license YOUR_DATA_LICENSE \
  --group-by template_id
```

加载器要求每个 split 覆盖中英文，拒绝重复 prompt、跨 split 分组重合、
固定能力评测题重合、无 verifier，以及 prompt 加 rollout 预算后超过上下文。
翻译和改写模板仍需人工归组。

## 3. 单步 on-policy GRPO

对每个 prompt 生成 \(G\) 个回答及奖励 \(r_i\)，在组内计算：

```text
advantage_i = (reward_i - group_mean) / (group_std + epsilon)
```

每批 rollout 都由尚未更新的当前 policy 新鲜生成，然后只进行一次优化。
对 rollout token 计算当前 policy 与冻结 reference 的 log-prob：

```text
KL = exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1
objective = logp_policy * advantage - kl_beta * KL
loss = -mean_per_sequence(objective)
```

prompt 和 padding 不参与 loss。先在每个回答内部按有效 token 求平均，再在
回答之间平均，因此长回答不会仅因 token 更多获得更大权重；
`tokens_seen` 仍记录实际 rollout token。

由于没有 rollout reuse 或同一 rollout 上的多轮 policy 更新，显式计算
`old == current` 的 ratio 只能恒为 1，clipping 也不会产生有效约束。
当前实现因此使用单次 on-policy policy-gradient + reference KL，不伪造
PPO-style clipping；它不等价于带 rollout buffer 的大规模 PPO/GRPO 系统。

## 4. 训练与恢复

公开配置从 Native DPO step 21 开始；实现也允许从 Native SFT checkpoint
开始。reference 始终是该父 checkpoint 的冻结副本：

```bash
python scripts/train_grpo.py \
  --config configs/pipelines/native-grpo-public-60m.yaml \
  --run-id native-grpo-public-60m-001
```

恢复：

```bash
python scripts/train_grpo.py \
  --config configs/pipelines/native-grpo-public-60m.yaml \
  --resume-run native-grpo-public-60m-001
```

初始化记录绑定父权重、Tokenizer、数据、评测集、采样参数、group size、
KL 和 advantage epsilon。attention dropout 必须为 0，保证 policy/reference
打分不混入 dropout 噪声。修改任一项必须创建新实验。

由于 rollout 长度取决于 EOS，训练预算中的 token/epoch 是
`prompts × group_size × max_new_tokens` 上界估算；实际
`tokens_seen` 和 coverage 需以运行产物为准。

## 5. 如何判断是否有效

训练日志包含：

- `report_reward_mean` 与 `report_success_rate`；
- `report_zero_variance_groups`；
- `report_approx_kl`；
- 实际 rollout token、吞吐和梯度范数。

奖励上升不是充分证据。正式运行还必须用训练外的统一能力评测比较：

- 可验证任务奖励和 chosen/rejected 排序；
- 指令、格式、问答和多轮保持；
- Base BPB 与续写退化；
- 中英文分别统计。

## 当前边界

- Native CPU 微型 on-policy GRPO/RLVR 与精确恢复：已验证。
- 公开 MSVAMP 数据物化、SFT/GRPO 零组交集、seq512 加载与泄漏门禁：已验证。
- 公开数据 GPU 训练和阶段前后报告：未执行。
- Qwen/LoRA GRPO：未实现。
- 多 epoch rollout reuse、分布式 rollout、独立 vLLM worker：未实现。
- 当前 verifier 只衡量答案/格式正确性，不代表帮助性、安全性或人类偏好。

[返回项目路线](./test.md)
