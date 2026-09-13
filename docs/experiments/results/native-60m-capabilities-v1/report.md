# 能力评测成绩单

- 模型：`native-60m-baseline-v1/step-00005649`
- 阶段：`pretrain`
- 权重 SHA-256：`78cf72522ad87e71aca4977ce5708aa3ac3ffaa3db32a04514511593f8b49664`
- 协议 SHA-256：`c51428a095ff2b0a392c5ec86ab1c8e19ff6201fb218f81e41ad3b1f599953e1`
- 小型诊断探针，不是通用能力排行榜；不能用少量样例推断总体能力。

| 指标 | 数值 | 分母 | 基线 | 方向 |
| --- | ---: | ---: | ---: | --- |
| continuation.nonempty | 1.000000 | 8 | 0.000000 (rule) | higher |
| continuation.nonempty.en | 1.000000 | 4 | 0.000000 (rule) | higher |
| continuation.nonempty.zh | 1.000000 | 4 | 0.000000 (rule) | higher |
| continuation.repeated_trigram | 0.241667 | 8 | 0.000000 (rule) | lower |
| continuation.repeated_trigram.en | 0.000000 | 4 | 0.000000 (rule) | lower |
| continuation.repeated_trigram.zh | 0.483333 | 4 | 0.000000 (rule) | lower |
| corpus.bpb.all | 2.035306 | 331 | 3.468278 (analytic-random) | lower |
| corpus.bpb.en | 1.529442 | 173 | 3.236994 (analytic-random) | lower |
| corpus.bpb.zh | 2.589195 | 158 | 3.721519 (analytic-random) | lower |
| corpus.loss.all | 5.694680 | 82 | 9.704061 (analytic-random) | lower |
| corpus.loss.en | 4.585057 | 40 | 9.704061 (analytic-random) | lower |
| corpus.loss.zh | 6.751465 | 42 | 9.704061 (analytic-random) | lower |
| format.success | 0.000000 | 2 | 0.000000 (rule) | higher |
| format.success.en | 0.000000 | 1 | 0.000000 (rule) | higher |
| format.success.zh | 0.000000 | 1 | 0.000000 (rule) | higher |
| instruction.success | 0.000000 | 4 | 0.250000 (mixed) | higher |
| instruction.success.en | 0.000000 | 2 | 0.250000 (mixed) | higher |
| instruction.success.zh | 0.000000 | 2 | 0.250000 (mixed) | higher |
| memory.test.suffix_exact | 0.000000 | 4 | 2.6728e-51 (analytic-random) | diagnostic |
| memory.test.suffix_exact.en | 0.000000 | 2 | 2.6728e-51 (analytic-random) | diagnostic |
| memory.test.suffix_exact.zh | 0.000000 | 2 | 2.6728e-51 (analytic-random) | diagnostic |
| memory.train.suffix_exact | 0.000000 | 4 | 2.6728e-51 (analytic-random) | diagnostic |
| memory.train.suffix_exact.en | 0.000000 | 2 | 2.6728e-51 (analytic-random) | diagnostic |
| memory.train.suffix_exact.zh | 0.000000 | 2 | 2.6728e-51 (analytic-random) | diagnostic |
| multiturn.all_correct | 0.000000 | 2 | 0.000000 (rule) | higher |
| multiturn.all_correct.en | 0.000000 | 1 | 0.000000 (rule) | higher |
| multiturn.all_correct.zh | 0.000000 | 1 | 0.000000 (rule) | higher |
| multiturn.success | 0.000000 | 6 | 0.000000 (rule) | higher |
| multiturn.success.en | 0.000000 | 3 | 0.000000 (rule) | higher |
| multiturn.success.zh | 0.000000 | 3 | 0.000000 (rule) | higher |
| overlap.heldout_fraction | 0.000000 | 45343 | 0.000000 (rule) | lower |
| overlap.query_fraction | 0.000000 | 16 | 0.000000 (rule) | lower |
| preference.accuracy | 0.750000 | 4 | 0.500000 (analytic-random) | higher |
| preference.accuracy.en | 0.500000 | 2 | 0.500000 (analytic-random) | higher |
| preference.accuracy.zh | 1.000000 | 2 | 0.500000 (analytic-random) | higher |
| qa.success | 0.000000 | 4 | 0.250000 (mixed) | higher |
| qa.success.en | 0.000000 | 2 | 0.250000 (mixed) | higher |
| qa.success.zh | 0.000000 | 2 | 0.250000 (mixed) | higher |
| verifiable.reward | 0.000000 | 16 | 0.125000 (mixed) | higher |
| verifiable.reward.en | 0.000000 | 8 | 0.125000 (mixed) | higher |
| verifiable.reward.zh | 0.000000 | 8 | 0.125000 (mixed) | higher |

## 口径与边界

- 手写双语诊断集样本很少；不得作为通用语言能力或生产可用性的证明。
- BPB 使用 NFKC 后文本 UTF-8 字节数，不计 EOS；不是旧 Reference 的 Packing BPB。
- 续写重复率只度量退化，不度量事实性、流畅性或帮助程度。
- 多轮输入使用模型自己的历史回答，不注入标准答案；空回复/控制串仅为续轮转义。
- 偏好分数是回答 token 平均条件 log-prob，不含结束符；不是 reward model 准确率。
- verifiable.reward 是任务规则的 0/1 奖励，不代表安全性或人类偏好对齐。
- 空输出规则基线是下界；随机基线是解析期望，不是随机初始化模型实测。
- 记忆探针对固定语料前缀做精确后缀匹配；不命中不能证明模型没有记忆。
- 重合检查只覆盖指定语料的规范化文本/子串，不能证明不存在泄漏。

## 固定续写

- `Once upon a time` → , a boy named Leo found a strange map in his attic. It was old and dusty, with strange symbols. "What could this lead to?" he wondered
- `从前` → 线的线线,是线线的线线,是线线,是线线,在线线,在线线为。  线
- `The little girl` →  named Mia found a shiny stone in her backyard. It was not just any stone; it was a magic stone. She picked it up and felt a warm glow
- `在中国` → 的香港人,為臺灣的香港人,為臺灣的香港人,為臺灣的香港人。  生平  早年  香港中文
- `One day, a boy` →  named Leo found a strange map in his attic. It was old and dusty, but it showed a path to a hidden treasure. He felt a spark of excitement
- `这个故事` → ,是《大魔王》的主角。  故事  《大魔王》是《大魔王》的主角,是《大
- `There was a dragon` →  named Leo. He was a brave knight. He wanted to find the dragon's treasure. One day, he found a shiny stone. "This must be the
- `北京是` → 香港的香港人,為香港人,由香港人,香港人,香港人,香港人,香港人,香港人,香港人,香港
