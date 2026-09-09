# Pretrain 数据准备 Runbook

本文说明如何为 Native 10M/60M Pretrain 准备数据。项目支持两条入口：

```text
英文公开 recipe ─┐
                 ├-> 双语混合 -> 标准 JSONL ┐
中文公开 recipe ─┘                          ├-> 校验 -> 确定性切分 -> Data Manifest
自有数据 --------------------> 标准 JSONL ┘
```

公开数据与自有数据只在“标准 JSONL 之前”不同。进入 `llmlab data prepare` 后，共用同一套 schema、切分、指纹和完整性检查。

训练操作见 [Native 10M/60M Pretrain 训练指南](./NATIVE_PRETRAIN_GUIDE.md)，模型实现见 [Native 10M/60M 模型结构说明](./NATIVE_MODEL_GUIDE.md)。

## 1. 首版公开数据方案

首版为自有 Native 模型准备两份公开语料，并默认将它们组合成一份中英双语训练集：

| 语言 | 数据集 | 定位 | 许可证 |
| --- | --- | --- | --- |
| 英文 | [SimpleStories/SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | 小模型友好的合成英文故事 | MIT |
| 中文 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.zh` | 真实中文百科文章 | CC-BY-SA-3.0 |

英文选择 SimpleStories，是因为它面向小型、可解释语言模型，数据生成代码公开，适合观察语法、叙事结构和局部一致性。相比 TinyStories，它的研究目标之一是改善公式化、重复和编码伪影问题。

中文选择 Wikimedia 官方中文 Wikipedia，而不使用调研中的 Chinese FineWeb Edu：

- Wikipedia 来源、版本和许可边界更容易追溯。
- 固定 shard 约 127MB，适合首次实践。
- 文章覆盖多个知识主题，并包含简体和繁体中文。
- Chinese FineWeb Edu 虽然质量筛选更强，但数据卡同时出现 Apache-2.0 和额外社区/商业授权要求，首版不引入该许可歧义。
- FineWeb2 的中文 shard 单文件约 4.8GB，不符合首版低下载成本目标。

Smoke 配方按记录数 1:1 组合；60M 配方根据同一 16K BPE Tokenizer 下的
train split token 数校准中文配额：

| 配方 | 英文 | 中文 | 总记录数 |
| --- | ---: | ---: | ---: |
| `bilingual-smoke-v1` | 10,000 | 10,000 | 20,000 |
| `bilingual-60m-v1` | 100,000 | 126,000 | 226,000 |

混合使用固定组件顺序和 `round-robin` 规则，每条记录写入 `language=en` 或
`language=zh`。较短的英文组件耗尽后继续写入剩余中文记录。之所以不固定为记录数
1:1，是因为两种语料的篇幅和分词效率不同：100K + 100K 在 train split 中会形成
57.76% 英文 token 和 42.24% 中文 token。126K 中文配额将最终比例校准为
50.31% / 49.69%。这只是训练 token 预算基线，不代表领域分布或能力相等，正式能力
评估仍须按语言分别报告。

数据边界：

- SimpleStories 是 GPT-4o-mini 生成的合成英文故事，不代表真实世界英文分布。
- 中文 Wikipedia 是百科文本，不代表中文对话、文学、新闻或代码分布。
- 两份语料的领域不对称，不能用它们直接比较中英文学习难度。
- 10M/60M 模型只用于教学实验，双语训练不等于具备通用双语能力。
- 公开数据仍可能包含偏见、不当内容和事实错误，训练前应抽样检查。

参考资料：

- [SimpleStories 数据集](https://huggingface.co/datasets/SimpleStories/SimpleStories)
- [SimpleStories 论文](https://arxiv.org/abs/2504.09184)
- [数据生成代码](https://github.com/simple-stories/simple_stories_generate)
- [Wikimedia Wikipedia 数据集](https://huggingface.co/datasets/wikimedia/wikipedia)

## 2. 内置 Recipe

查看当前内置 recipe：

```bash
uv run llmlab data recipes
uv run llmlab data recipes --json
```

首版提供四个单语 source recipe 和两个双语 mixture recipe：

| Recipe | 记录数 | 标准 JSONL 约占用 | 用途 |
| --- | ---: | ---: | --- |
| `simplestories-smoke-v1` | 10,000 | 14MB | 英文 Smoke 组件/单语对照 |
| `wikipedia-zh-smoke-v1` | 10,000 | 13MB | 中文 Smoke 组件/单语对照 |
| `bilingual-smoke-v1` | 20,000 | 27MB | Native 10M 默认双语 Smoke |
| `simplestories-60m-v1` | 100,000 | 142MB | 英文 60M 组件/单语对照 |
| `wikipedia-zh-60m-v1` | 126,000 | 132MB | 中文 60M 组件/单语对照 |
| `bilingual-60m-v1` | 226,000 | 278MB | Native 60M 默认双语实践 |

英文 recipe 固定：

```text
repository: SimpleStories/SimpleStories
revision: e63b8adc3b1a1bdc7cac5b500d150b71346b0628
split: train
file: data/train-00000-of-00007.parquet
file_sha256: ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4
license: MIT
```

中文 recipe 固定：

```text
repository: wikimedia/wikipedia
revision: b04c8d1ceb2f5cd4588862100d08de323dccfbaa
config: 20231101.zh
split: train
file: 20231101.zh/train-00002-of-00006.parquet
file_sha256: ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f
license: CC-BY-SA-3.0
```

选择策略为 `source-prefix`：

- 只从固定 commit 的固定 train shard 读取。
- 忽略长度小于 64 字符的记录。
- 英文按上游顺序保留前 10K 或 100K 条合格记录，中文保留前 10K 或 126K 条。
- 同一语言的 Smoke 数据是 60M 数据的严格子集。
- 不使用上游 test split 参与训练。

这是强调稳定性和低成本的教学样本，不是完整上游数据的统计代表性随机样本。正式数据研究不应据此推导整个语料的分布。

标准化后的预期 SHA-256：

| Recipe | `source.jsonl` SHA-256 |
| --- | --- |
| `simplestories-smoke-v1` | `861ca23380e0d6a718350038e74fedd4f7d824bf9e18c3d907bd01fa88be346e` |
| `simplestories-60m-v1` | `e1ddf88e44e31a0858733c66f9b8ab3eaa7b91db1e73cfb6a97b0a7dd161763e` |
| `wikipedia-zh-smoke-v1` | `6dd9f3f86fbb3101dffea4315881e25c5516cecaf66bf1afcb442c804bae2739` |
| `wikipedia-zh-60m-v1` | `fdf06f43db54c564945725e175d5196f93824a1b411558763ff30d318128a9a0` |
| `bilingual-smoke-v1` | `4a6cb466167566d151dd4f597f29522a5157047b93989917776d8983a6118883` |
| `bilingual-60m-v1` | `af07e4c3a610637239a3295fd4754af316bd74c5d08bc25b63fd04a147339589` |

上游文件和标准化输出任一 hash 不一致，命令都会 fail-fast。

使用 `seed=42`、`group_by=source_id` 和默认 8:1:1 比例时，预期切分为：

| Recipe | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| `simplestories-smoke-v1` | 7,944 | 1,029 | 1,027 |
| `simplestories-60m-v1` | 79,982 | 10,083 | 9,935 |
| `wikipedia-zh-smoke-v1` | 8,004 | 980 | 1,016 |
| `wikipedia-zh-60m-v1` | 100,675 | 12,770 | 12,555 |
| `bilingual-smoke-v1` | 15,948 | 2,009 | 2,043 |
| `bilingual-60m-v1` | 180,657 | 22,853 | 22,490 |

双语 split 内的语言分布也固定：

| Recipe | Train `en/zh` | Dev `en/zh` | Test `en/zh` |
| --- | ---: | ---: | ---: |
| `bilingual-smoke-v1` | 7,944 / 8,004 | 1,029 / 980 | 1,027 / 1,016 |
| `bilingual-60m-v1` | 79,982 / 100,675 | 10,083 / 12,770 | 9,935 / 12,555 |

使用各自 recipe 训练出的 16K Tokenizer 统计 train split，token 权重为：

| Recipe | 英文 token | 中文 token | 英文/中文占比 |
| --- | ---: | ---: | ---: |
| `bilingual-smoke-v1` | 2,294,616 | 2,329,005 | 49.63% / 50.37% |
| `bilingual-60m-v1` | 23,236,284 | 22,948,247 | 50.31% / 49.69% |

使用当前 `uv.lock` 训练 16,384 词表的验收基线：

| Recipe | `min_frequency` | `tokenizer.json` SHA-256 |
| --- | ---: | --- |
| `bilingual-smoke-v1` | 1 | `31a61c82a4039357f1c2493763343c24b79489d5d8e6bdcac566e6cefbaaf1c6` |
| `bilingual-60m-v1` | 2 | `7455ac1168f36a3d7dca03fee28784f97899fe233f496484a11e5da2e44a2f32` |

改变 seed、分组字段、依赖锁或 Tokenizer 参数后，不应继续使用这些验收值。

## 3. 安装数据依赖

只使用自有 JSONL 时不需要 `public-data` extra。

需要公开 recipe 时：

```bash
uv sync --extra public-data --extra training --extra dev
```

`public-data` 固定：

- `huggingface-hub==0.36.2`
- `pyarrow==25.0.1`

首次执行会把约 238MB 的英文 shard 和 127MB 的中文 shard 下载到 Hugging Face
本地缓存。同一语言的 Smoke 与 60M recipe 共用缓存。

## 4. 路线 A：使用公开数据

### 4.1 Native 10M 默认双语 Smoke

分别下载并标准化英文、中文数据：

```bash
uv run llmlab data fetch \
  --recipe simplestories-smoke-v1 \
  --output data/raw/simplestories-smoke-v1 \
  --accept-license MIT

uv run llmlab data fetch \
  --recipe wikipedia-zh-smoke-v1 \
  --output data/raw/wikipedia-zh-smoke-v1 \
  --accept-license CC-BY-SA-3.0
```

显式 `--accept-license` 表示操作者已经检查并接受对应数据集许可证。CLI 不会替用户完成法律判断。

按 recipe 固定顺序轮转混合：

```bash
uv run llmlab data mix \
  --mixture bilingual-smoke-v1 \
  --input data/raw/simplestories-smoke-v1/source.jsonl \
  --input data/raw/wikipedia-zh-smoke-v1/source.jsonl \
  --output data/raw/bilingual-smoke-v1
```

校验并切分：

```bash
uv run llmlab data validate \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --kind pretrain

uv run llmlab data prepare \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --output data/prepared/bilingual-smoke-v1 \
  --dataset-id bilingual-smoke-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --seed 42 \
  --group-by source_id
```

该路径与 `configs/pipelines/native-smoke.yaml` 一致。

### 4.2 Native 60M 默认双语数据

```bash
uv run llmlab data fetch \
  --recipe simplestories-60m-v1 \
  --output data/raw/simplestories-60m-v1 \
  --accept-license MIT

uv run llmlab data fetch \
  --recipe wikipedia-zh-60m-v1 \
  --output data/raw/wikipedia-zh-60m-v1 \
  --accept-license CC-BY-SA-3.0

uv run llmlab data mix \
  --mixture bilingual-60m-v1 \
  --input data/raw/simplestories-60m-v1/source.jsonl \
  --input data/raw/wikipedia-zh-60m-v1/source.jsonl \
  --output data/raw/bilingual-60m-v1

uv run llmlab data validate \
  --input data/raw/bilingual-60m-v1/source.jsonl \
  --kind pretrain

uv run llmlab data prepare \
  --input data/raw/bilingual-60m-v1/source.jsonl \
  --output data/prepared/bilingual-60m-v1 \
  --dataset-id bilingual-60m-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --seed 42 \
  --group-by source_id
```

该路径与 `configs/pipelines/native-v1.yaml` 一致。

### 4.3 单语对照

四个 source recipe 都能独立执行 `data prepare`。例如只准备中文 Smoke：

```bash
uv run llmlab data prepare \
  --input data/raw/wikipedia-zh-smoke-v1/source.jsonl \
  --output data/prepared/wikipedia-zh-smoke-v1 \
  --dataset-id wikipedia-zh-smoke-v1 \
  --kind pretrain \
  --license CC-BY-SA-3.0 \
  --group-by source_id
```

单语数据用于数据与 Tokenizer 对照，不是默认训练路径。中英模型必须使用
`bilingual-smoke-v1` 或 `bilingual-60m-v1` 训练的 Tokenizer 和 checkpoint。

### 4.4 公开来源记录

单语 source 标准化后的每条记录形如：

```json
{
  "id": "<由上游版本、source_id 和正文计算的 SHA-256>",
  "text": "A short story...",
  "source": "hf://datasets/SimpleStories/SimpleStories@e63b.../train",
  "source_id": "<SimpleStories generation_id>"
}
```

双语混合记录额外包含：

```json
{"language":"en"}
{"language":"zh"}
```

`source_manifest.json` 记录：

- recipe ID。
- 上游 repository、commit、split 和 Parquet 文件。
- 上游文件 SHA-256。
- 许可证、语言和是否为合成数据。
- 选择规则、记录数和跳过数。
- `huggingface-hub` / `pyarrow` 版本。
- 标准化 JSONL SHA-256。

`mixture_manifest.json` 记录：

- mixture recipe ID 和 `round-robin` 策略。
- 两个组件的完整 source manifest。
- 语言、许可证组合、记录总数和最终 SHA-256。

`data prepare` 会验证 source/mixture manifest、已安装 recipe 和 `source.jsonl` 的 hash，并把完整来源信息写入最终 Data Manifest 的 `source_metadata`。

## 5. 路线 B：使用自有数据

自有 Pretrain 数据必须是 UTF-8 JSONL，每行一个对象：

```json
{"id":"doc-0001","text":"语言模型根据已有 token 预测下一个 token。","source":"internal-corpus","source_id":"article-001"}
{"id":"doc-0002","text":"The model predicts the next token.","source":"internal-corpus","source_id":"article-002"}
```

必填字段：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `id` | string | 全文件唯一、非空 |
| `text` | string | 非空 |
| `source` | string | 非空，标明数据来源 |

推荐字段：

| 字段 | 用途 |
| --- | --- |
| `source_id` | 同一原始文档或生成批次使用同一值，防止跨 split 泄漏 |
| `document_id` | 如果已有文档级 ID，可直接作为 `--group-by` |

先校验：

```bash
uv run llmlab data validate \
  --input data/custom/pretrain.jsonl \
  --kind pretrain
```

再准备：

```bash
uv run llmlab data prepare \
  --input data/custom/pretrain.jsonl \
  --output data/prepared/custom-v1 \
  --dataset-id custom-v1 \
  --kind pretrain \
  --license Proprietary \
  --seed 42 \
  --group-by source_id
```

如果每行本身就是独立文档，且没有共享来源，可以使用 `--group-by id`。如果同一文档被切成多条记录，不能使用 `id` 分组，否则近重复片段可能进入不同 split。

自有数据的 `--license` 应填写真实授权标识，例如 SPDX ID、内部许可策略 ID 或 `Proprietary`，不要照抄公开 recipe 的 MIT。

## 6. 确定性切分

默认切分比例：

```text
train: 0.8
dev:   0.1
test:  0.1
seed:  42
```

切分以 `--group-by` 的值为单位，通过稳定 hash 分配：

- 输入行顺序变化不会改变已有记录的 split。
- 同一 group 不会同时出现在 train/dev/test。
- 相同数据、seed 和 group key 会得到相同结果。

可显式调整：

```bash
uv run llmlab data prepare \
  --input data/custom/pretrain.jsonl \
  --output data/prepared/custom-v2 \
  --dataset-id custom-v2 \
  --kind pretrain \
  --license Proprietary \
  --seed 42 \
  --group-by document_id \
  --train-ratio 0.9 \
  --dev-ratio 0.05 \
  --test-ratio 0.05
```

Pretrain 要求 dev split 非空。分组数量太少时应增加数据，不应绕过 baseline。

## 7. Prepared Dataset 产物

```text
data/prepared/<dataset_id>/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── data_manifest.json
```

Data Manifest 记录：

- 源 JSONL 的绝对路径、大小和 SHA-256。
- train/dev/test 的记录数、group 数和 SHA-256。
- split seed、group key、license 和处理步骤。
- 使用公开 recipe 时的上游 commit、文件 hash、选择规则和 loader 版本。

这些字段会继续进入训练 run 的数据快照，用于恢复和复现门禁。

## 8. 不可变 Token Packing

训练前必须将 Prepared Dataset 和已冻结的 Tokenizer 物化为固定长度的磁盘数据：

```bash
uv run llmlab data pack \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-smoke-v1 \
  --output data/packed/bilingual-smoke-v1-seq128 \
  --sequence-length 128

uv run llmlab data pack \
  --manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output data/packed/bilingual-60m-v1-seq512 \
  --sequence-length 512
```

每个目录包含：

```text
data/packed/<packed_id>/
├── train.tokens.i32
├── train.languages.i8
├── train.byte_weights.f32
├── dev.tokens.i32
├── dev.languages.i8
├── dev.byte_weights.f32
├── test.tokens.i32
├── test.languages.i8
├── test.byte_weights.f32
└── packed_manifest.json
```

- `tokens.i32` 保存连续 token stream，训练窗口按 `sequence_length - 1` 重叠一个 token。
- `languages.i8` 保存每个 token 的 `en=0`、`zh=1` 标签，供分桶评测使用。
- `byte_weights.f32` 保存内容 token 对 UTF-8 字节数的分摊，BOS/EOS 权重为 0。
- Manifest 绑定 Data Manifest SHA-256、Tokenizer SHA-256、`sequence_length`、数组大小和数组 SHA-256。
- 训练使用只读 `numpy.memmap` 按样本切片，不把完整 token stream 常驻内存。

固定双语数据的验收计数：

| Packed 配方 | Split | 文档 | 监督 token | 样本 | Padding |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoke seq128 | train | 15,948 | 4,623,620 | 36,407 | 69 |
| Smoke seq128 | dev | 2,009 | 587,712 | 4,628 | 44 |
| Smoke seq128 | test | 2,043 | 616,580 | 4,855 | 5 |
| 60M seq512 | train | 180,657 | 46,184,530 | 90,381 | 161 |
| 60M seq512 | dev | 22,853 | 5,841,738 | 11,432 | 14 |
| 60M seq512 | test | 22,490 | 5,648,666 | 11,055 | 439 |

输出目录禁止覆盖。命令先写同目录临时文件，三个 split 和 Manifest 全部成功后再原子
发布。重复运行应使用新版本目录；Data Manifest、Tokenizer 或序列长度任一变化都必须
重新 pack。

## 9. 质量检查

在训练 Tokenizer 前至少检查：

1. `data validate` 无错误。
2. train/dev/test 均非空。
3. `source` 和 license 与真实来源一致。
4. 同源记录通过 `source_id` 或 `document_id` 分组。
5. 随机抽样文本，确认语言、领域和安全边界符合预期。
6. 不包含密钥、个人信息、未授权内部数据或测试集答案。
7. 对公开 recipe，`source_manifest.json` 或 `mixture_manifest.json` 与 Data Manifest 中的 revision/hash 一致。
8. 双语数据分别抽样中英文，并记录各语言的 token 数和 dev loss。

不要把“公开可下载”等同于“适合任何用途”。数据许可证、隐私、内容安全和目标地区法规仍需由使用者审核。

## 10. 训练自有数据

准备好 `data/prepared/custom-v1` 后：

1. 训练独立 Tokenizer：

```bash
uv run llmlab tokenizer train \
  --manifest data/prepared/custom-v1/data_manifest.json \
  --output data/tokenizers/custom-v1 \
  --tokenizer-id custom-v1 \
  --vocab-size 16384
```

2. 复制对应 Native pipeline 配置。
3. 修改：

```yaml
model:
  tokenizer: data/tokenizers/custom-v1
data:
  manifest: data/prepared/custom-v1/data_manifest.json
  packed_manifest: data/packed/custom-v1-seq512/packed_manifest.json
```

4. 使用相同 Data Manifest、Tokenizer 和 pipeline 序列长度执行 `llmlab data pack`。
5. 先执行带配置的 `llmlab doctor`，再开始训练。

不要让自有数据实验复用公开数据的 Tokenizer 路径或 Data Manifest，否则来源边界和结果解释会混淆。

## 11. 常见失败

| 错误 | 处理 |
| --- | --- |
| `requires --accept-license MIT` | 检查数据集许可后显式传入 `--accept-license MIT` |
| `upstream file hash mismatch` | 缓存或上游内容与 recipe 不一致；不要绕过，清理对应缓存后重试 |
| `public recipe output hash mismatch` | loader 或标准化行为漂移；使用锁定依赖并停止实验 |
| `public source hash mismatch` | `source.jsonl` 已被修改；重新 fetch 到新目录 |
| `public mixture inputs do not match` | 输入的单语 source 与 mixture recipe 不一致 |
| `public mixture output hash mismatch` | 组件、顺序或混合逻辑发生漂移；停止实验并检查版本 |
| `output ... already exists` | 产物不可覆盖；使用新版本目录或明确删除无用产物 |
| `duplicate_id` | 自有数据 ID 重复；在 prepare 前修正 |
| dev/test 为空 | group 太少；增加数据或调整比例 |
| `--license must match` | `--license` 必须与 source 或 mixture manifest 一致 |
| `tokenizer was trained from a different Data Manifest` | 不允许用其他数据版本训练出的 Tokenizer 生成 packed 产物 |
| `packed output already exists` | Packed 目录不可覆盖；复用已校验产物或使用新版本目录 |
| `packed array hash mismatch` | 二进制数组已损坏或被修改；停止训练并重新生成 |

## 12. 最短公开数据流程

```bash
uv sync --extra public-data --extra training --extra dev

uv run llmlab data fetch \
  --recipe simplestories-smoke-v1 \
  --output data/raw/simplestories-smoke-v1 \
  --accept-license MIT

uv run llmlab data fetch \
  --recipe wikipedia-zh-smoke-v1 \
  --output data/raw/wikipedia-zh-smoke-v1 \
  --accept-license CC-BY-SA-3.0

uv run llmlab data mix \
  --mixture bilingual-smoke-v1 \
  --input data/raw/simplestories-smoke-v1/source.jsonl \
  --input data/raw/wikipedia-zh-smoke-v1/source.jsonl \
  --output data/raw/bilingual-smoke-v1

uv run llmlab data prepare \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --output data/prepared/bilingual-smoke-v1 \
  --dataset-id bilingual-smoke-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --group-by source_id

uv run llmlab tokenizer train \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --output data/tokenizers/bilingual-smoke-v1 \
  --tokenizer-id bilingual-smoke-v1 \
  --vocab-size 16384 \
  --min-frequency 1

uv run llmlab data pack \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-smoke-v1 \
  --output data/packed/bilingual-smoke-v1-seq128 \
  --sequence-length 128
```
