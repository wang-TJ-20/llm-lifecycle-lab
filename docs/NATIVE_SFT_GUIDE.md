# Native SFT 最小闭环

SFT 并没有把“预测下一个 token”换成另一种学习问题。关键变化是输入变成对话，
而 loss 只监督 assistant 的回答及结束标记。

先有[统一能力评测](./CAPABILITY_EVALUATION_GUIDE.md)，再运行 SFT：
训练前后使用同一组探针，才能区分 loss 下降、格式变化和实际任务改善。

## 1. 不下载数据，先验证机制

```bash
python scripts/sft_experiment.py --mode train
python scripts/sft_experiment.py --mode resume
```

每条命令都在独立临时目录中：

1. 创建中英文微型预训练数据及 Tokenizer，训练三步 Base。
2. 固定 `native-chat-v1` 评测协议，生成阶段前成绩单。
3. 创建另一份中英文数字复写对话数据，加载 Base 权重并训练三步 SFT。
4. 用同一评测器生成 41 项指标及差值。
5. `resume` 模式额外注入中断，对比连续训练和恢复后的全部训练状态。

检查 `parent_weights_preserved`、`sft_weights_updated`、
`same_evaluation_protocol` 均为 true；恢复模式还应输出
`resume_weights_optimizer_scheduler_rng_equal=true`。

退出后自动清理，不修改已有数据、训练记录或 Base-v1 发布包。
这些模板数据只用来检查机制：同一任务模板可能跨 split 出现，
不能据此声称模型具备泛化能力或形成正式 Instruct 权重。

## 2. 哪些 token 参与监督

示意对话：

```text
<|im_start|>user
只输出数字：7<|im_end|>
<|im_start|>assistant
7<|im_end|>
```

监督范围是 assistant 正文 `7` 与它的 `<|im_end|>`。
user、system、角色名称、起始标记、回合之间的换行、padding 的 label 都是 `-100`。
因果偏移仍由共用 loss 函数执行：位置 t 的 logits 预测 label[t+1]，不做第二次偏移。

`data/sft.py` 先按现有完整 chat 模板编码，再依据控制 token 定位角色，
验证角色头部的 token 边界，避免用字符串长度当 token 数。
多轮中的每段 assistant 都受监督，padding 不改变有效 token 分母。
如果罕见的前导空白导致 BPE 跨过角色头部边界，会报错并要求修整样本，
不会误把角色 token 当作答案。

训练引擎按**有效回答 token**累计梯度和训练预算，而不是把长短回答等权平均。
SFT dev loss 同样只度量回答部分，单列中英文，不输出混合整段对话的 BPB。

## 3. 准备自己的双语对话数据

每行一个 JSON 对象，例如：

```json
{"id":"en-example-1","source_id":"example-1","template_id":"copy-number-v1","language":"en","messages":[{"role":"user","content":"Write this number again using digits only: 17."},{"role":"assistant","content":"17"}]}
{"id":"zh-example-1","source_id":"example-1","template_id":"copy-number-v1","language":"zh","messages":[{"role":"user","content":"请重新写出这个数字，只输出数字：17。"},{"role":"assistant","content":"17"}]}
```

这是格式示例，两行不足以形成 train/dev/test。需要足够多的独立对话或模板组，
且三个 split 都覆盖中英文。若目标是测试新模板泛化，应按 `template_id` 分组；
若按 `source_id` 分组，只能声称隔离了来源实例，不能声称隔离了模板。
翻译版本应归在同组。

复用现有准备命令。下面的 `YOUR_DATA_LICENSE` 必须改成真实来源许可：

```bash
python scripts/data.py prepare \
  --input data/sft-source.jsonl \
  --output data/prepared/sft-bilingual-v1 \
  --dataset-id sft-bilingual-v1 \
  --kind sft \
  --license YOUR_DATA_LICENSE \
  --group-by template_id
```

训练启动前核对 Manifest 和全部 split hash，禁止跨 split 的分组重合和规范化
完整对话重复。固定能力评测集中的 group、template_id、精确 user 提示，
以及正文中包含的评测文本或长提示片段被命中时，拒绝启动；
改写题或未标注的同模板题仍需要人工数据审核。

角色规则沿用 `native-chat-v1`：system 仅可位于开头，之后 user/assistant 交替，
SFT 样本必须以非空 assistant 结束，正文禁止嵌入控制 token。
当前版本不打包不同对话、不静默丢弃或截断超长样本。超过配置长度直接报错，
应在上游按完整对话/回合重新整理数据并生成新的 Manifest。

## 4. 从本地 60M Base 开始

准备好对应权重、Tokenizer 和上述数据后：

```bash
python scripts/train_sft.py \
  --config configs/pipelines/native-sft-smoke.yaml \
  --run-id native-sft-smoke-001
```

这是两步 CPU/float32 机制配置，不是正式 60M SFT 效果配方。
`init_checkpoint` 可以是本地发布包或原 Base checkpoint 目录；不依赖组织审核。
Tokenizer 必须与 Base 的内容 hash 一致，不重新训练词表，也不扩展 token ID。

每个新 SFT run 会记录：

- `initialization.json`：父 checkpoint、权重和配置 hash、数据与评测集 hash；
- `sft_data_summary.json`：各 split 的中英文样本数和有效监督 token；
- `training_budget.json`、环境记录、指标与完整 checkpoint。

Base 的 dirty-Git 例外不会被改写为 clean；发布包提供的父来源状态保留在初始化记录里。
原 checkpoint 不含这项声明时记录未知，不自行推断。

## 5. 初始化不是恢复

| 操作 | 继承什么 | 重置什么 |
| --- | --- | --- |
| Base → 新 SFT | 模型结构、权重、Tokenizer | optimizer、scheduler、step、RNG、数据位置、预算 |
| 同一次 SFT 恢复 | 权重及完整训练状态 | 不重新开始预算 |

恢复命令：

```bash
python scripts/train_sft.py \
  --config configs/pipelines/native-sft-smoke.yaml \
  --resume-run native-sft-smoke-001
```

仅适用于未完成预算的中断 run。已完成的两步 Smoke 不能追加训练；
修改预算或学习率属于新实验。恢复时还会核对父权重、数据 Manifest 和评测集，
仅路径相同而内容已变化也会拒绝。

## 6. 训练后重新测量

用 `evaluate_capabilities.py` 测量 SFT checkpoint，并传入相同协议的 Base 报告，
具体命令见[纵向成绩单](./CAPABILITY_EVALUATION_GUIDE.md#6-纵向成绩单与能力保持)。

重点观察：格式与指令是否改善，中文与英文是否一致，基础 BPB 与续写是否退化。
不要用训练集回答 loss 下降替代这些证据。当前已验证 CPU 微型链路与精确恢复；
正式 60M 长跑、GPU 数值对照及 Instruct 权重发布仍需后续数据与算力验证。

[返回项目首页](../README.md)
