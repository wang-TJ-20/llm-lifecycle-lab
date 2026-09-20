# 数据介绍与准备

本文负责从原始文本到可训练数据的完整流程：了解数据来源、下载或接入自己的 JSONL、
切分、训练 Tokenizer、生成 Packing，并核对产物。模型原理见
[自有模型介绍](./NATIVE_MODEL_GUIDE.md)，环境安装和训练见
[Pretrain 训练文档](./NATIVE_PRETRAIN_GUIDE.md)。

## 1. 先了解数据流

```text
公开英文 + 公开中文 -> fetch -> mix --+
                                     +-> validate -> prepare -> train_tokenizer -> pack
自有 UTF-8 JSONL --------------------+
```

| 名称 | 作用 | 默认位置 |
| --- | --- | --- |
| Source recipe | 固定上游版本、文件、采样规则、许可和 hash | `src/llm_lifecycle_lab/data/recipes/` |
| Mixture recipe | 固定组件顺序与双语混合规则 | `src/llm_lifecycle_lab/data/mixtures/` |
| Raw | 下载后标准化的文本及来源信息 | `data/raw/` |
| Prepared | 按文档组切分的 train/dev/test JSONL | `data/prepared/` |
| Tokenizer | 将文本编码为模型的 token ID | `data/tokenizers/` |
| Packed | 按固定长度读取的只读二进制 token 数组 | `data/packed/` |

`prepare` 不会自动训练 Tokenizer，`pack` 也不会自动下载原始数据。
必须按这个顺序完成前置产物。上述 `data/` 全部为本地产物，不进入 Git。

## 2. 默认用什么数据

### 2.1 来源与边界

| 语言 | 数据集 | 内容 | 许可 |
| --- | --- | --- | --- |
| 英文 | [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | GPT-4o-mini 生成的英文故事 | MIT |
| 中文 | [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)，`20231101.zh` | 中文百科文章，包含简体与繁体 | CC-BY-SA-3.0 |

选择这两类数据是为了让首个实验的来源、下载量和授权边界可追溯。它们不是通用语料：
英文偏故事，中文偏百科；不能用两种语言的 loss 差异直接推断语言难度，也不能把双语
训练等同于双语助手能力。内容可能包含事实错误、偏见或不适宜文本，使用前仍须抽样检查。

项目只从上游固定 **train shard** 选取文本，再创建自己的 train/dev/test；
不会使用上游 test 来训练。选择策略为 `source-prefix`：按原顺序取前 N 条长度至少
64 字符的合格记录，并非对整个上游语料随机抽样。Smoke 是同语言 60M 配方的子集。

### 2.2 选择规模

| 用途 | 英文记录 | 中文记录 | 总记录 | 混合文本约占用 |
| --- | ---: | ---: | ---: | ---: |
| 10M Smoke | 10,000 | 10,000 | 20,000 | 27 MB |
| 60M 教学预训练 | 100,000 | 126,000 | 226,000 | 278 MB |

60M 不按记录数 1:1 配比。使用当前 16K BPE 后，train 英文/中文 token 占比约
50.31%/49.69%；如果只用 100K 英文 + 100K 中文，会约为 57.76%/42.24%。
配额按 token 校准，不代表两个领域的覆盖或质量相同。

对应 recipe：

| 阶段 | 英文 | 中文 | 混合 |
| --- | --- | --- | --- |
| Smoke | `simplestories-smoke-v1` | `wikipedia-zh-smoke-v1` | `bilingual-smoke-v1` |
| 60M | `simplestories-60m-v1` | `wikipedia-zh-60m-v1` | `bilingual-60m-v1` |

可用 `python scripts/data.py recipes --json` 查看完整定义。

### 2.3 固定来源

| 字段 | 英文 | 中文 |
| --- | --- | --- |
| Repository | `SimpleStories/SimpleStories` | `wikimedia/wikipedia` |
| Revision | `e63b8adc3b1a1bdc7cac5b500d150b71346b0628` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| File | `data/train-00000-of-00007.parquet` | `20231101.zh/train-00002-of-00006.parquet` |
| File SHA-256 | `ca33531b99f3bebb4125e82f56017c223a4900331b55bcf7cde1b0f750d88fd4` | `ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f` |

首次下载约 238 MB 英文 shard 和 127 MB 中文 shard 到 Hugging Face 缓存。
Smoke/60M 复用同一上游文件，但各自的标准化输出目录不同。应额外预留 raw、prepared、
packed 和临时输出的空间，不能只按下载文件大小估算磁盘需求。

## 3. 开始前检查

1. 完成 [Conda 环境准备](./NATIVE_PRETRAIN_GUIDE.md#2-环境准备)，激活环境。
2. 在仓库根目录执行；相对路径都以当前工作目录为基准。
3. 确认许可后再运行带 `--accept-license` 的下载命令，脚本不替操作者作法律判断。
4. 输出目录必须不存在。完整产物已经存在时先校验并复用，不要重复运行生成命令。

```bash
conda activate llm-lifecycle-lab
python -m pip check
python scripts/data.py recipes
```

标准 [requirements.txt](../requirements.txt) 已包含训练及公开数据依赖。
它只安装第三方依赖；各 `scripts/*.py` 会直接加载仓库的 `src/`，不需要安装本项目。

## 4. 第一次实践：10M 双语 Smoke

以下六步按顺序执行。没有运行过训练也能完成它们，Tokenizer 和 packing 主要在 CPU 上处理。

### 步骤 1：获取两种语言

```bash
python scripts/data.py fetch \
  --recipe simplestories-smoke-v1 \
  --output data/raw/simplestories-smoke-v1 \
  --accept-license MIT

python scripts/data.py fetch \
  --recipe wikipedia-zh-smoke-v1 \
  --output data/raw/wikipedia-zh-smoke-v1 \
  --accept-license CC-BY-SA-3.0
```

**检查结果**：两次命令各输出 `records: 10000`、revision 和 SHA-256；
每个输出目录包含 `source.jsonl` 与 `source_manifest.json`。源文件和标准化输出的 hash
都会被校验，下载失败或 hash 不符时停止，不继续执行下一步。

### 步骤 2：确定性混合

```bash
python scripts/data.py mix \
  --mixture bilingual-smoke-v1 \
  --input data/raw/simplestories-smoke-v1/source.jsonl \
  --input data/raw/wikipedia-zh-smoke-v1/source.jsonl \
  --output data/raw/bilingual-smoke-v1
```

混合按 recipe 的组件顺序轮转，每条记录写入 `language=en` 或 `language=zh`。
**检查结果**：输出 `records: 20000`，生成 `source.jsonl` 与 `mixture_manifest.json`；
混合许可为 `CC-BY-SA-3.0 AND MIT`，SHA-256 应为：

```text
4a6cb466167566d151dd4f597f29522a5157047b93989917776d8983a6118883
```

### 步骤 3：校验并按来源组切分

```bash
python scripts/data.py validate \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --kind pretrain

python scripts/data.py prepare \
  --input data/raw/bilingual-smoke-v1/source.jsonl \
  --output data/prepared/bilingual-smoke-v1 \
  --dataset-id bilingual-smoke-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --seed 42 \
  --group-by source_id
```

`validate` 检查 schema、必填字段、重复 ID 等；它不负责文本近重复去重或安全清洗。
`prepare` 使用 group hash 按默认 8:1:1 分配，因此实际数量不是精确 80%/10%/10%。

**检查结果**：

| Split | 记录数 | 英文 | 中文 |
| --- | ---: | ---: | ---: |
| train | 15,948 | 7,944 | 8,004 |
| dev | 2,009 | 1,029 | 980 |
| test | 2,043 | 1,027 | 1,016 |

生成 `train.jsonl`、`dev.jsonl`、`test.jsonl` 和 `data_manifest.json`。
后续只在 train 上训练 Tokenizer 和模型，dev 用于观察和调参，test 用于最终检查。

### 步骤 4：训练 Tokenizer

```bash
python scripts/train_tokenizer.py \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --output data/tokenizers/bilingual-smoke-v1 \
  --tokenizer-id bilingual-smoke-v1 \
  --vocab-size 16384 \
  --min-frequency 1

python scripts/inspect_tokenizer.py \
  data/tokenizers/bilingual-smoke-v1 \
  --text "语言模型 learns from text."
```

**检查结果**：训练输出 `vocab_size: 16384`；目录内有 `tokenizer.json` 和
`tokenizer_manifest.json`；inspect 输出 token ID、数量及解码文本。
本配方使用 `min_frequency=1`，提高它可能导致实际词表不足 16K。

Tokenizer 使用 NFKC + Byte-level BPE。NFKC 会规范化全角等兼容字符，所以解码应与
**规范化后**的输入一致，不承诺保留原始字符形式。Byte-level 覆盖中英文 UTF-8，
但相同字符数未必产生相同 token 数。控制 token 与协议见模型文档。

当前锁环境的 Tokenizer 内容 SHA-256：

```text
31a61c82a4039357f1c2493763343c24b79489d5d8e6bdcac566e6cefbaaf1c6
```

该值是版本基线，不是所有依赖、输入顺序和训练参数下的通用值。词表实际不足时应增加
数据或创建匹配词表的新模型配置，不能只手改 manifest 的 `vocab_size`。

### 步骤 5：生成磁盘 Packing

```bash
python scripts/data.py pack \
  --manifest data/prepared/bilingual-smoke-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-smoke-v1 \
  --output data/packed/bilingual-smoke-v1-seq128 \
  --sequence-length 128
```

**检查结果**：输出路径为 `packed_manifest.json`，train 为 36,407 个样本、
4,623,620 个监督 token。三个 split 的二进制数组合计约 53 MiB。

### 步骤 6：验证整条数据链

```bash
python scripts/doctor.py --config configs/pipelines/native-smoke.yaml
```

应看到 `data-manifest`、`packed-data` 和 `real-batch` 均为 `PASS`，最后 `0 failed`。
Doctor 会校验 hash，并使用真实样本执行 forward/backward；不会保存训练 checkpoint。
全部通过后，进入 [Pretrain 的 Smoke 训练](./NATIVE_PRETRAIN_GUIDE.md#4-运行两步-smoke)。

## 5. 切换为 60M 双语数据

使用独立的 `bilingual-60m-v1` 产物，不覆盖 Smoke。下载、混合、切分步骤：

```bash
python scripts/data.py fetch \
  --recipe simplestories-60m-v1 \
  --output data/raw/simplestories-60m-v1 \
  --accept-license MIT

python scripts/data.py fetch \
  --recipe wikipedia-zh-60m-v1 \
  --output data/raw/wikipedia-zh-60m-v1 \
  --accept-license CC-BY-SA-3.0

python scripts/data.py mix \
  --mixture bilingual-60m-v1 \
  --input data/raw/simplestories-60m-v1/source.jsonl \
  --input data/raw/wikipedia-zh-60m-v1/source.jsonl \
  --output data/raw/bilingual-60m-v1

python scripts/data.py validate \
  --input data/raw/bilingual-60m-v1/source.jsonl \
  --kind pretrain

python scripts/data.py prepare \
  --input data/raw/bilingual-60m-v1/source.jsonl \
  --output data/prepared/bilingual-60m-v1 \
  --dataset-id bilingual-60m-v1 \
  --kind pretrain \
  --license "CC-BY-SA-3.0 AND MIT" \
  --seed 42 \
  --group-by source_id
```

**检查结果**：混合共 226,000 条，SHA-256 为
`af07e4c3a610637239a3295fd4754af316bd74c5d08bc25b63fd04a147339589`。

| Split | 记录数 | 英文 | 中文 |
| --- | ---: | ---: | ---: |
| train | 180,657 | 79,982 | 100,675 |
| dev | 22,853 | 10,083 | 12,770 |
| test | 22,490 | 9,935 | 12,555 |

随后训练该数据集自己的 Tokenizer，再打包为 seq512：

```bash
python scripts/train_tokenizer.py \
  --manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --output data/tokenizers/bilingual-60m-v1 \
  --tokenizer-id bilingual-60m-v1 \
  --vocab-size 16384 \
  --min-frequency 2

python scripts/data.py pack \
  --manifest data/prepared/bilingual-60m-v1/data_manifest.json \
  --tokenizer data/tokenizers/bilingual-60m-v1 \
  --output data/packed/bilingual-60m-v1-seq512 \
  --sequence-length 512
```

**检查结果**：词表 16,384；train 90,381 个 packed 样本、46,184,530 个监督 token；
三个 split 合计约 500 MiB。当前锁环境 Tokenizer SHA-256：

```text
7455ac1168f36a3d7dca03fee28784f97899fe233f496484a11e5da2e44a2f32
```

准备数据不要求 CUDA，但 `native-v1.yaml` 的完整 Doctor/训练要求 Linux + CUDA。
没有 GPU 时可完成本节，到训练前再迁移产物和环境。

## 6. 接入自己的数据

### 6.1 文件格式

放入 `data/custom/pretrain.jsonl`，使用 UTF-8，每行一个独立 JSON 对象：

```json
{"id":"doc-0001","text":"语言模型根据已有 token 预测下一个 token。","source":"licensed-corpus","source_id":"article-001","language":"zh"}
{"id":"doc-0002","text":"The model predicts the next token.","source":"licensed-corpus","source_id":"article-002","language":"en"}
```

这两行只是格式示例，不足以训练 16K Tokenizer，也不保证能切出三个非空 split。

| 字段 | 要求 |
| --- | --- |
| `id` | 必填，非空字符串，全文件唯一 |
| `text` | 必填，非空字符串，是真实正文而非路径 |
| `source` | 必填，非空字符串，记录真实来源 |
| `source_id` / `document_id` | 推荐；同一原始文章的所有片段共用一个值 |
| `language` | 推荐填 `en` 或 `zh`；缺失或其他值会进入未标记桶，不产生对应中英文分桶指标 |

当前不自动识别语言。不要为了凑齐分桶而把中文标成英文。文本授权、隐私去除、质量过滤、
近重复去重需要在准备前处理，不能把 schema 校验当作这些检查的替代。

### 6.2 校验与切分

```bash
python scripts/data.py validate \
  --input data/custom/pretrain.jsonl \
  --kind pretrain

python scripts/data.py prepare \
  --input data/custom/pretrain.jsonl \
  --output data/prepared/custom-v1 \
  --dataset-id custom-v1 \
  --kind pretrain \
  --license Proprietary \
  --seed 42 \
  --group-by source_id
```

`Proprietary` 仅为填写示例，应替换为实际授权标识。无 `source_id` 且每行确实是独立文章时
可用 `--group-by id`；同篇文章的多个片段不能这样分组。所有记录都必须有选定的 group 字段。

同 group 不跨 split；输入重排不会改变 group 的归属，但会改变 split 内顺序和文件 hash。
若改 split 比例，可另用 `--train-ratio 0.9 --dev-ratio 0.05 --test-ratio 0.05`，
并换新输出目录。验证 train/dev/test 均非空，尤其不要跳过 dev baseline。

### 6.3 Tokenizer、Packing 与 Pipeline

以先跑 seq128 Smoke 为例：

```bash
python scripts/train_tokenizer.py \
  --manifest data/prepared/custom-v1/data_manifest.json \
  --output data/tokenizers/custom-v1 \
  --tokenizer-id custom-v1 \
  --vocab-size 16384 \
  --min-frequency 1

python scripts/data.py pack \
  --manifest data/prepared/custom-v1/data_manifest.json \
  --tokenizer data/tokenizers/custom-v1 \
  --output data/packed/custom-v1-seq128 \
  --sequence-length 128
```

接着按 [自定义训练配置](./NATIVE_PRETRAIN_GUIDE.md#7-自定义实验配置) 创建 pipeline，
绑定上面的三个路径。如果实际词表不是 16,384，还必须复制模型 YAML，修改词表与
`model_id` 并同步 pipeline。不能拿公开数据 Tokenizer 冒充自有数据训练的 Tokenizer；
当前训练和 pack 都严格绑定其来源 Data Manifest。

单语对照可独立 prepare 任一 source recipe，并按同样流程创建独立 Tokenizer/pipeline。
单语产物不替换默认双语路线。

## 7. Packing 到底保存了什么

每个 split 先将文档编码为 `BOS + content + EOS`，再连接成一个连续 stream。
窗口长度为 L，步长为 L-1，重叠的一个 token 为下一窗口提供上下文。例如 L=4：

```text
stream:  a b c d e f g
window1: a b c d       targets: b c d
window2:       d e f g targets: e f g
```

每个有效目标只监督一次，尾部不足长度的窗口补 PAD、label 设为 `-100`。
当前文档之间 **不做 attention 隔离**，只用 BOS/EOS 分界；因果注意力可以看到同窗口内
的前文。长文不会简单截掉尾部，但上下文长度仍受窗口限制。

```text
data/packed/<packed_id>/
  packed_manifest.json
  train.tokens.i32
  train.languages.i8
  train.byte_weights.f32
  dev.tokens.i32
  dev.languages.i8
  dev.byte_weights.f32
  test.tokens.i32
  test.languages.i8
  test.byte_weights.f32
```

`tokens` 保存 ID；`languages` 中 en=0、zh=1、未标记=-1；`byte_weights` 将 NFKC
规范化正文的 UTF-8 字节均摊到内容 token，BOS/EOS 为 0。这是 bits-per-byte 的统计权重，
不是每个 BPE token 的精确原文 byte offset。

训练使用只读 `numpy.memmap` 取样，不将全量 tensor 常驻内存。Manifest 绑定数组大小、
SHA-256、来源 Data Manifest、Tokenizer hash 和序列长度。先写临时目录，再整体原子发布。

| 配方 | Split | 样本数 | 监督 token | 尾部 Padding |
| --- | --- | ---: | ---: | ---: |
| Smoke seq128 | train | 36,407 | 4,623,620 | 69 |
| Smoke seq128 | dev | 4,628 | 587,712 | 44 |
| Smoke seq128 | test | 4,855 | 616,580 | 5 |
| 60M seq512 | train | 90,381 | 46,184,530 | 161 |
| 60M seq512 | dev | 11,432 | 5,841,738 | 14 |
| 60M seq512 | test | 11,055 | 5,648,666 | 439 |

## 8. 产物校验、复用与迁移

| 已有产物 | 从哪里继续 |
| --- | --- |
| 仅 raw | 校验来源后从 prepare 开始 |
| prepared | 保留原 manifest，训练对应 Tokenizer |
| prepared + Tokenizer | 直接生成相同数据/词表对应的 packing |
| prepared + Tokenizer + packed | 跑配置化 Doctor，通过后训练 |
| 产物不完整或 hash 不符 | 停止；确认原因后在新版本目录重建，不手改 hash 放行 |

跨机器迁移时成套保留 prepared、Tokenizer、packed 以及各自 manifest。Data Manifest 内
保留来源绝对路径和创建时间；移动原有文件不需要手改这些来源记录。
重新执行 prepare 即使文本相同，也会生成不同的 manifest 文件 hash，必须同步重建下游。
不要混用“旧 Tokenizer + 新 Data Manifest”。

建议把原始来源与制品归档到可靠存储后再做磁盘清理。训练当前仍用 prepared 做
Doctor 校验，不应只留下 packed。删除 raw/Hugging Face 缓存虽可重新下载，却会增加
后续重建成本；删除 Tokenizer 或改其 ID 映射会使原 checkpoint 无法按原协议使用。
