# 统一能力评测

评测器独立于训练阶段和 ModelScope 发布流程。同一份版本化探针可以先评测
Base，再评测 SFT、DPO 或其他阶段的 Native checkpoint。它不依赖 LLM Judge，
也不试图用一个总分表示模型的全部能力。

## 1. 第一份成绩单

在仓库根目录运行；需要基础训练依赖，但不需要 ModelScope 登录或 GPU：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --output runs/evaluations/base-plain-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json
```

`--checkpoint` 支持独立发布包或原训练 checkpoint 目录。
原 checkpoint 需另传 `--tokenizer data/tokenizers/bilingual-60m-v1`。
发布包先验证发布清单，所有输入都核对 checkpoint 与 Tokenizer 的内容 hash。
输出目录已存在时拒绝覆盖。

产物为 `report.json` 和 `report.md`。JSON 保存完整协议、测试集快照、逐题输出、
评分、基线、模型 hash 和运行环境；Markdown 用于阅读。

默认 CPU/float32、单线程、greedy、32 个新 token，不根据输出质量重试或挑选种子。
上下文不够时直接失败，不静默截断。完整执行可能需要几分钟，语料扫描需读取
所有 train/dev/test 文本。

## 2. 分清三种验证

| 类型 | 作用 | 不能说明什么 |
| --- | --- | --- |
| 发布 Smoke | 校验下载、加载、前向与固定输出 | 不代表语言能力 |
| Pretrain Reference | 冻结数据窗口、训练预算及历史指标 | 不等于全量 dev/test 或指令能力 |
| Lifecycle 能力探针 | 固定小型任务、规则评分与阶段间比较 | 不等于大型能力榜单或生产验收 |

当前 [lifecycle-v1](../configs/evaluation/lifecycle-v1.yaml) 是项目手写的
28 个中英双语诊断案例。其许可为 Apache-2.0，不应放入训练数据，
同组中文/英文变体也不能拆开后作为训练与评测的独立样本。

## 3. 指标与基线

| 指标 | 计算口径 | 基线 |
| --- | --- | --- |
| corpus loss/BPB | BOS 后预测正文，不含 EOS；NFKC 规范化文本的 UTF-8 字节数 | 词表均匀概率的解析 NLL |
| 续写重复率 | 生成 token 中重复三元组的比例 | 无重复目标 0 |
| 续写非空率 | 去除终止 token 后是否存在正文 | 总是空输出的规则基线 |
| 指令/问答成功率 | 去掉首尾空白后完全匹配，不接受附加说明 | 选项均匀猜测；其他任务使用空输出基线 |
| JSON 成功率 | 完整解析、精确键和值及类型，不接受代码围栏或重复键 | 空输出基线 |
| 多轮成功率 | 使用模型自己的历史回答逐轮评分，并单列全对率 | 空回复基线 |
| 偏好排序准确率 | chosen/rejected 回答 token 平均条件 log-prob；平局 0.5 | 随机排序 0.5 |
| 可验证奖励 | 指令/格式/问答/多轮规则的 0/1 结果 | 对应任务基线 |
| 后缀精确记忆 | 32 token 前缀预测后续 12 token，全等才命中 | 独立均匀 token 的解析命中概率 |
| 语料重合 | 规范化全文相等，以及足够长的探针文本是否为训练文本子串 | 零重合目标 |

每项都有显式分母，任务同时列出中文、英文及合计。不平均不同语言 BPB，
而是将 NLL 累加后除以对应的字节数。Memory 的升降属于诊断事实，
不是越高或越低就一定越好。

空输出基线只是下界，不是强规则解题器；解析随机基线不是随机权重实测。
偏好排序不是奖励模型准确率，规则奖励也不证明模型更安全或更符合人类偏好。
题目很少，0% 或 100% 都只能解释为“这些探针上的结果”。

## 4. 提示与多轮协议

- 续写始终使用 `BOS + 原文`，各阶段保持一致。
- 指令默认使用 `plain-v1`，即显式的 `User:` / `Assistant:` 文本。
- 可使用 `--prompt-protocol native-chat-v1` 测试正式对话模板。
- 多轮会加入模型上一轮的实际回答，不用标准答案代替。
- 空回答与生成的控制串只在下一轮对话序列化时转义；原始输出照常保留和评分。
- 偏好回答按显式 token 边界拼接到提示之后，不计算结束 token 的概率。

Base 与 SFT 可以共享同一组题目，但不能把 `plain-v1` 的 Base 报告和
`native-chat-v1` 的 SFT 报告直接相减。需要用相同协议重新测量 Base。

## 5. 记忆与重合检查的边界

`--pretrain-manifest` 必须与 Tokenizer 记录的预训练 Data Manifest hash 一致，
且全部 split 的 hash 与行数校验通过。扫描所有文本后，记录：

- train 与 dev/test 的规范化全文重合比例；
- 长度至少 24 个规范化字符的评测提示是否出现在训练文本中；
- 首批符合长度要求的 train/test 记忆探针，每种语言每个 split 最多 2 条；
- test 记忆探针若全文出现在 train，则排除，不当作干净泛化样本。

报告不复制训练文本或真实来源 ID，只保存探针的内容 hash、来源 ID hash、
评分和采样协议。提供的语料成员关系不证明训练过程确实消费了该条样本。
检查不覆盖语义近似、改写和未提供的数据；短提示被排除而不是算作无泄漏。
不提供 manifest 时，相关指标是 `not-checked` / `null`，不是 0。

## 6. 纵向成绩单与能力保持

后续阶段使用同一命令，添加阶段前报告：

```bash
python scripts/evaluate_capabilities.py run \
  --checkpoint runs/sft-001/checkpoints/step-00000003 \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output runs/evaluations/sft-plain-v1 \
  --pretrain-manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --baseline runs/evaluations/base-plain-v1/report.json
```

也可按阶段顺序合并多份报告：

```bash
python scripts/evaluate_capabilities.py compare \
  --reports runs/evaluations/base-plain-v1/report.json \
            runs/evaluations/sft-plain-v1/report.json \
  --output runs/evaluations/base-to-sft-v1
```

比较前核对测试集、评分代码、Tokenizer、提示格式、生成预算、精度、设备、
线程数、依赖版本、语料 hash，以及各项分母和基线。任意差异导致拒绝比较，
不会仅因两份报告都有 `loss` 字段就计算提升。

Delta 定义为当前减首份报告。BPB 与重复率负差通常更好，任务分数正差更好。
同时查看原有 BPB、续写和问答的变化来观察能力保持，不发明一个混合总分。
正式 SFT、DPO、GRPO 的训练效果仍需各自真实训练记录，本评测层不替代它们。

[返回项目首页](../README.md)
