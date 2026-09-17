# Native DPO 最小闭环

DPO 使用成对偏好数据，让策略相对于冻结 reference 更偏向 chosen 回答。
当前实现面向 Native 路线，重点是把数据、数学、reference 与恢复契约做正确；
正式偏好能力仍需 GPU 训练和统一成绩单验证。

推荐的 60M 配方从固定 revision 的 HelpSteer3 选择 560 个公开偏好对，
命令与许可见
[公开数据 SFT、DPO 与 GRPO 路线](./PUBLIC_POSTTRAINING_GUIDE.md)。
旧的模板偏好对和 on-policy 数据只用于复现历史结果。

## 1. 先验证机制

不下载数据、不修改已有 run：

```bash
python scripts/dpo_experiment.py --mode train
python scripts/dpo_experiment.py --mode resume
```

脚本在临时目录依次执行微型 Pretrain、SFT 和三步 DPO。它验证：

- DPO 只能从 Native SFT checkpoint 开始；
- policy 和 reference 初始权重相同，reference 始终冻结；
- reference 的 chosen/rejected log-prob 只计算一次并保存；
- SFT 父 checkpoint 不被修改；
- DPO loss 按偏好对平均，而不是按较长回答给予更大权重；
- 中断恢复与连续训练的权重、优化器、scheduler、RNG 和数据位置一致。

合成数字偏好只检查机制，不代表模型获得了真实偏好能力。

## 2. 数据格式

每行一个偏好对：

```json
{"id":"en-1","source_id":"problem-1","template_id":"arithmetic-v1","language":"en","prompt":"What is 2 + 3? Reply with a digit.","chosen":"5","rejected":"9"}
{"id":"zh-1","source_id":"problem-1","template_id":"arithmetic-v1","language":"zh","prompt":"2 加 3 等于多少？只回答数字。","chosen":"5","rejected":"9"}
```

准备 Data Manifest：

```bash
python scripts/data.py prepare \
  --input data/dpo-source.jsonl \
  --output data/prepared/dpo-bilingual-v1 \
  --dataset-id dpo-bilingual-v1 \
  --kind dpo \
  --license YOUR_DATA_LICENSE \
  --group-by template_id
```

正式数据建议按 prompt 来源或模板分组；翻译、改写和同一问题的不同偏好对必须
归到同组。加载器要求每个 split 同时包含中英文，拒绝：

- chosen 与 rejected 相同；
- 规范化后重复的完整偏好对；
- 分组跨 train/dev/test；
- prompt、group 或 template 与固定能力评测集重合；
- 超过序列长度的回答。

当前门禁不能识别所有语义改写，仍需数据卡、来源许可和人工审查。

## 3. 目标函数

对 prompt \(x\)、chosen \(y_w\)、rejected \(y_l\)，先计算回答 token 的
**序列 log-prob 总和**，只覆盖 assistant 正文和结束标记：

```text
margin_policy = log π(y_w|x) - log π(y_l|x)
margin_ref    = log π_ref(y_w|x) - log π_ref(y_l|x)
loss = -log sigmoid(beta * (margin_policy - margin_ref))
```

没有对回答长度取平均，这与标准 DPO 的序列概率定义一致。
prompt、角色头和 padding 不参与 log-prob。`beta` 必须位于 `(0, 1]`。

共享训练引擎区分两种统计单位：

- SFT/Pretrain loss 按有效监督 token 加权；
- DPO loss 按偏好对加权，`tokens_seen` 仍记录实际回答 token。

因此长回答不会只因 token 更多就在 DPO 梯度平均中占更大权重。

## 4. 冻结 reference

新 DPO run 从同一个 SFT checkpoint 创建：

- policy：可训练；
- reference：冻结、CPU float32，仅用于预计算全部 split 的序列 log-prob。

`reference_scores.json` 保存 pair ID hash 和 chosen/rejected 分数，不复制文本。
`initialization.json` 同时绑定：

- SFT checkpoint 元数据及权重/config hash；
- Tokenizer hash；
- DPO Data Manifest 和评测集 hash；
- reference scores hash；
- `beta`。

恢复只读取冻结分数，不重新定义 reference。任一输入改变都拒绝恢复。

## 5. 运行

公开路线完成 SFT step 112 后直接使用已绑定父 checkpoint 的配置：

```bash
python scripts/train_dpo.py \
  --config configs/pipelines/native-dpo-public-60m.yaml \
  --run-id native-dpo-public-60m-001
```

恢复未完成的同一次 run：

```bash
python scripts/train_dpo.py \
  --config configs/pipelines/native-dpo-public-60m.yaml \
  --resume-run native-dpo-public-60m-001
```

已完成预算的 run 不能追加训练。修改 `beta`、学习率、数据或父 checkpoint
属于新实验。`native-dpo-smoke.yaml` 仍可用于 CPU 机制检查，不用于正式效果结论。

训练日志记录 DPO loss、policy 偏好准确率、reference 偏好准确率及 reward margin。
这些训练指标只描述当前偏好对；最终结论必须运行统一能力评测，比较：

- `preference.accuracy` 与可验证奖励是否改善；
- SFT 指令、格式、问答和多轮能力是否保持；
- Base BPB 与续写是否退化。

## 当前边界

- Native DPO 数据、目标、冻结 reference、checkpoint 与 CPU 精确恢复：已验证。
- HelpSteer3 公开数据物化、切分、seq512 加载与泄漏门禁：已验证。
- 公开数据 60M GPU 训练和能力保持报告：未执行。
- Qwen LoRA DPO：未实现。
- DPO 不等于安全对齐；四个固定偏好探针不足以说明整体偏好质量。
- GRPO/RLVR 是下一阶段，只接受程序可验证奖励，不复用 DPO 偏好对冒充奖励。

[返回项目路线](./test.md)
