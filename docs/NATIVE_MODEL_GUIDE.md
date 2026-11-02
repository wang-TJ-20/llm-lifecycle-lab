# 自有模型介绍

本文说明 Native-60M 的结构设计、参数预算、张量形状和代码边界，并给出不需要
训练数据就能执行的模型实验。模型是从随机权重开始训练，不是裁剪或微调 Qwen 得到的。

数据来源及 Tokenizer 准备见 [数据介绍与准备](./DATA_GUIDE.md)；环境安装、训练与恢复见
[Pretrain 训练文档](./NATIVE_PRETRAIN_GUIDE.md)。A10 Base 训练结果见
[A10 执行记录](./A10_TRAINING_RESULTS.md)。

## 1. 设计目标

Native 模型的定位不是追求同参数量下的最佳效果，而是作为一个足够真实、又足够可读的教学模型：

- 结构上接近现代 decoder-only LLM。
- 同一份实现支持通过配置缩放；当前保留 A10 使用的 `tiny-60m` 配置。
- 模型统一使用 `model_route: native`；`run_profile` 表示实践级别，不表示参数量。
- 代码边界清晰：模型只负责张量级前向、生成和 checkpoint，不负责 stage-specific loss。
- 能支持真实的 Pretrain 训练、增量生成和 KV Cache。
- 默认词表大小为 16,384，A10 使用配套数据训练的双语 Tokenizer。

模型结构本身与语言无关。“支持中英文”表示 Byte-level Tokenizer 能编码两种语言，
且默认训练数据覆盖两种语言。NFKC 会规范化输入，往返还原针对规范化后的文本，不保证
原字符形式不变；这也不表示模型已具备通用双语助手能力。

当前实现文件：

- 实践入口：[scripts/model_experiment.py](../scripts/model_experiment.py)，直接展示一次参数更新。
- 配置：[src/llm_lifecycle_lab/model/native/config.py](../src/llm_lifecycle_lab/model/native/config.py)
- 注意力：[src/llm_lifecycle_lab/model/native/attention.py](../src/llm_lifecycle_lab/model/native/attention.py)
- RMSNorm：[src/llm_lifecycle_lab/model/native/normalization.py](../src/llm_lifecycle_lab/model/native/normalization.py)
- Block/SwiGLU：[src/llm_lifecycle_lab/model/native/layers.py](../src/llm_lifecycle_lab/model/native/layers.py)
- 主模型：[src/llm_lifecycle_lab/model/native/transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)
- 生成循环与采样：[src/llm_lifecycle_lab/model/native/generation.py](../src/llm_lifecycle_lab/model/native/generation.py)

先读实验脚本，再读主模型的 `__init__()` 和 `forward()`，最后按需进入 Attention
和 Block。关键数学实现旁有中英文说明与张量形状，不需要先理解训练框架。

## 2. 当前模型配置

当前保留的模型由 `NativeTransformer` 实现：

| 配方 | 参数量 | 层数 | Hidden | Q Heads | KV Heads | MLP | Max Seq | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tiny-60m` | 62,927,616 | 8 | 768 | 12 | 4 | 2048 | 512 | A10 预训练 |

对应配置：

- [configs/models/tiny-60m.yaml](../configs/models/tiny-60m.yaml)

`model_route` 只选择 Native 实现。Pipeline 中的
`model.config` 明确选择具体结构，`run_profile` 则决定对应的实践规模和 Doctor 资源要求。

模型最大长度与训练长度是两件事：A10 配方中两者均为 512。
改大配置上限不能证明模型学会了更长上下文。
词表大小相同也不代表 token ID 映射相同，不能交换不同 Tokenizer。

共有默认项：

- `vocab_size = 16384`
- `norm_eps = 1e-6`
- `rope_theta = 10000.0`
- `qk_norm = false`
- `attention_dropout = 0.0`
- `tie_word_embeddings = true`
- `initializer_range = 0.02`

Tokenizer 同时固定 `native-chat-v1` 消息协议。Pretrain 只使用 BOS/EOS，但在正式
训练前就预留 `<|im_start|>`、`<|im_end|>` 和 8 个控制 token 槽位，确保后续
SFT/DPO/GRPO 不需要扩展 embedding 或改变已有 token ID。

`NativeModelConfig` 会在加载时做约束检查，例如：

- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`
- `head_dim` 必须为偶数，才能应用 RoPE
- 所有核心维度必须是正整数；`2.9`、`true` 或字符串 `"2"` 不会自动转为整数
- `norm_eps`、`rope_theta`、`initializer_range` 必须是有限正数，拒绝 `NaN/Inf`
- 直接构造与 YAML/JSON 加载共用校验；YAML 数值不要加引号

### 2.1 词表参数预算

词表会同时决定 token embedding 和 LM head 的参数量；当前两者权重绑定，因此只计
一份矩阵：

| 配方 | 词表 | Token 参数 | 占总参数 |
| --- | ---: | ---: | ---: |
| `tiny-60m` | 16,384 | 12,582,912 | 20.00% |

减小词表会同时改变 embedding 参数预算和文本的 token 数。
项目保留 16K 作为中英文统一基线，同时通过 `inspect_model.py --compare-vocab-size`
显式比较 8K/12K/16K，而不是静默更换默认词表。

### 2.2 A10 架构基线

冻结基线选用宽浅、QK-Norm 关闭的 `tiny-60m`。
其 62,927,616 个参数、16K 词表、512 上下文和 `initializer_range=0.02` 保持不变；
数据、训练预算、源码与验收由 A10 Base 配置
[`native-60m-a10-base-v1.yaml`](../configs/pipelines/native-60m-a10-base-v1.yaml) 固定。
执行顺序见 [60M 正式训练](./NATIVE_PRETRAIN_GUIDE.md#9-60m-正式训练)。
当前没有冻结的结构消融配置。模型实现仍支持 QK-Norm 等显式选项；
独立对照需新建配置并固定数据、Tokenizer、seed、预算与评测，不能由单次 A10 训练
推断哪种结构更好。

## 3. 顶层结构

整体拓扑是标准 decoder-only 结构：

```text
input_ids
  -> token_embedding
  -> N x TransformerBlock
  -> final RMSNorm
  -> tied LM head
  -> logits
```

对应实现见 [NativeTransformer](../src/llm_lifecycle_lab/model/native/transformer.py)。

几个关键点：

- 输入是已经 tokenized 的 `input_ids`，模型不处理原始文本。
- 输出只包含 `logits` 和可选 `cache`。
- 模型本体不计算 Pretrain loss，也不持有 optimizer / scheduler。
- `lm_head.weight` 默认与 `token_embedding.weight` 绑定，减少参数量并与常见 LLM 做法一致。

### 张量形状

设 B 为 batch size，T 为输入长度，D 为 hidden size，V 为词表大小，Hq/Hkv 为
Query/KV heads，Dh=D/Hq。60M 的 Dh=64。

| 位置 | 通用形状 | 60M，B=2、T=16 |
| --- | --- | --- |
| `input_ids` | `[B, T]`，整数 | `[2, 16]` |
| embedding/block 输出 | `[B, T, D]` | `[2, 16, 768]` |
| Query | `[B, Hq, T, Dh]` | `[2, 12, 16, 64]` |
| Key/Value，重复前 | `[B, Hkv, T, Dh]` | `[2, 4, 16, 64]` |
| LM head 输出 `logits` | `[B, T, V]` | `[2, 16, 16384]` |
| 每层 KV cache | 两个 `[B, Hkv, past_T, Dh]` | 不保存重复到 12 头后的 KV |

logits 是未归一化分数，不是文字和概率。Pretrain objective 在外部将位置 t 的分数
与位置 t+1 的标签对齐；训练数据不应提前再做一次标签 shift。

## 4. Transformer Block

单层 block 的结构在 [TransformerBlock](../src/llm_lifecycle_lab/model/native/layers.py)：

```text
residual
  + Attention(RMSNorm(x))
  + MLP(RMSNorm(x_after_attention))
```

它是典型的 pre-norm 残差结构：

1. 先 `RMSNorm`
2. 再 attention
3. 残差相加
4. 再 `RMSNorm`
5. 再 MLP
6. 再一次残差相加

### RMSNorm

`RMSNorm` 实现在 [normalization.py](../src/llm_lifecycle_lab/model/native/normalization.py)：

- 只学习一个 `weight`
- 在 `float32` 中计算均方 `mean(x²)`，归一化后转回输入 dtype
- 不使用均值中心化

这比 LayerNorm 更贴近当前主流 LLM 的归一化方式。

```text
RMSNorm(x) = weight * x / sqrt(mean(x²) + eps)
```

这里不是中心化后的统计方差；不减去均值，也没有单独的 bias。

### SwiGLU

MLP 使用 `SwiGLU`，实现见 [layers.py](../src/llm_lifecycle_lab/model/native/layers.py)：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

对应三组线性层：

- `gate_proj`
- `up_proj`
- `down_proj`

全部不带 bias。

## 5. 注意力实现

注意力实现在 [CausalSelfAttention](../src/llm_lifecycle_lab/model/native/attention.py)。

### 5.1 Grouped Query Attention

这里实现的是 GQA，而不是完全对称的 MHA：

- Query head 数：`num_attention_heads`
- Key/Value head 数：`num_key_value_heads`
- `query_groups = num_attention_heads // num_key_value_heads`

前向时：

1. `q_proj` 生成所有 query heads
2. `k_proj` / `v_proj` 只生成较少的 KV heads
3. 通过 `repeat_key_value()` 把 KV 沿 head 维复制到 query head 数

这样能减少 KV cache 占用，也更接近真实大模型常见做法。

### 5.2 Rotary Position Embedding

RoPE 实现在 [RotaryEmbedding](../src/llm_lifecycle_lab/model/native/attention.py)。

特点：

- 只作用在 Q/K，不作用在 V。
- 模型初始化时按 `max_sequence_length` 一次性生成 float32 cos/sin cache。
- 所有 Transformer layer 共用同一个非持久化 cache，不再逐层重复计算三角函数。
- 前向时按 `past_length + 当前序列位置` 只读切片，并转换到 Q/K 的计算 dtype。
- 使用 `rotate_half()` 完成旋转。

因此在 cache 续写时，位置编码不会从 0 重新开始，而是和已有 prefix 连续。
RoPE cache 不进入 `state_dict`，不会增加 checkpoint 大小或改变旧权重 key。

### 5.3 QK-Norm

`qk_norm: true` 时，每层在 RoPE 之前分别对每个 Query/Key head 的 `head_dim`
执行可学习 RMSNorm。Value 不归一化。它每层只增加 `2 × head_dim` 个参数：

- 宽浅 8 层配方增加 1,024 参数。
- 深窄 19 层配方增加 2,432 参数。

默认 `tiny-60m` 保持关闭，避免把共享 RoPE 的等价优化与 QK-Norm 的模型行为变化
混在一起。开启和关闭 QK-Norm 的 checkpoint 使用不同 `model_id` 与完整模型配置，
不能交叉恢复。

### 5.4 Causal Mask

mask 逻辑见 [build_attention_mask()](../src/llm_lifecycle_lab/model/native/attention.py)：

- 在 CUDA/MPS 上，如果是完整前缀、无外部 mask、且没有 cache，则让
  `scaled_dot_product_attention(..., is_causal=True)` 走内置 causal 路径。
- CPU 保留共享的显式 causal mask；具体性能依赖硬件、模型规模和序列长度。
- `collate_pretraining_batch()` 在 CPU 组 batch 时省略全真的 mask，不需要在 GPU 上
  判断是否全真；存在 padding 时保留原始 mask，不删除样本或改变 labels。
- 需要显式 mask 时，只在模型前向中构造一次，所有层共享，而非每层重复分配。
- 当存在 cache 时，`attention_mask` 的长度必须是 `past_length + current_length`。

公开 `forward()` 接收 `[B, total_T]` mask；内部 Attention 接收已经准备好的
`[B, 1, Tq, Tk]` mask（无 padding 时可沿 batch 广播）。True 表示允许注意，
这是 PyTorch SDPA 的语义。

## 6. KV Cache 与生成

`forward()` 支持：

- `use_cache`
- `cache`
- `logits_to_keep`：默认 `0` 输出全部位置；正整数只对末尾指定数量的位置做 LM head
  投影，不裁剪 KV Cache。该选项属于 Native 前向，不改变通用训练接口。

其中 cache 的结构是：

```text
tuple[layer] -> (key, value)
```

每层一个 `(K, V)`，所以 cache 层数必须与 block 数完全一致，否则直接报错。

对应校验在 [transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)。

### 生成流程

`generate()` 的行为：

- 默认 greedy；如果 `do_sample=True`，支持 `temperature` 和 `top_p`
- 首轮处理整个 prompt，之后只输入最新 token 并复用 cache；每轮仅投影最后位置的 logits
- 支持无 padding 或左侧 padding；拒绝右侧 padding、mask 中间有空洞和全 padding 的行
- mask 必须与输入同形、同设备且只包含 0/1；未提供 mask 时所有输入位置都视为有效
- 如果配置了 `eos_token_id`，所有样本都结束时提前停止
- 已结束样本会被 `pad_token_id` 填充
- `prompt_tokens + max_new_tokens` 不能超过模型 `max_sequence_length`，prompt 长度包含
  左侧 padding 占据的物理位置
- 成功或异常退出后都会恢复模型原来的 train/eval 状态

完整流程在 [generation.py](../src/llm_lifecycle_lab/model/native/generation.py) 的
`generate_tokens()` 中，主模型的 `generate()` 只是直接调用它。训练 `forward()` 仍然
支持右侧 padding；生成的布局限制不改变训练数据格式。

## 7. 初始化与参数量

初始化在 [transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)：

- `Linear` 和 `Embedding` 都用均值 0、标准差 `initializer_range` 的正态分布
- 不单独处理 bias，因为这些层都不带 bias

参数量可以从两处看：

1. `NativeModelConfig.estimated_parameter_count`
2. `NativeTransformer.parameter_count`

前者用于配置级估算，后者是实例化后的真实总参数数。

默认 tied embeddings、QK-Norm 关闭时，可按下式复核：

```text
token parameters = V × D
attention/layer  = 2 × D² + 2 × D × Hkv × Dh
SwiGLU/layer     = 3 × D × intermediate_size
norms/layer      = 2 × D
final norm       = D
```

总数为 token parameters + N × 每层参数 + final norm。启用 QK-Norm 每层额外
`2 × Dh`；关闭权重绑定再增加 `V × D`。RoPE buffer 无可训练参数且不进入 checkpoint。

## 8. 模型边界

Native 模型故意只暴露窄接口，见 [ModelProtocol](../src/llm_lifecycle_lab/model/protocol.py)：

- `forward`
- `generate`
- `trainable_parameters`
- `set_training` / `is_training`
- `to_device`
- `save` / `load`

这意味着以下内容明确不属于模型本体：

- Tokenizer 训练与文本编码
- Pretrain loss 计算
- log-probability 聚合
- optimizer / scheduler
- 数据打包与 batch 顺序
- checkpoint 的 optimizer / RNG / data-stream 状态

这些分别放在：

- Tokenizer：[src/llm_lifecycle_lab/tokenizer/native.py](../src/llm_lifecycle_lab/tokenizer/native.py)
- Pretrain objective：[src/llm_lifecycle_lab/training/stages/pretrain.py](../src/llm_lifecycle_lab/training/stages/pretrain.py)
- Training engine：[src/llm_lifecycle_lab/training/engine.py](../src/llm_lifecycle_lab/training/engine.py)
- Checkpoint manager：[src/llm_lifecycle_lab/training/checkpoint.py](../src/llm_lifecycle_lab/training/checkpoint.py)

## 9. 保存与加载

模型级 checkpoint 保存格式：

```text
model/
├── config.json
└── model.pt
```

保存和加载时会检查：

- 目标目录是否为空
- `config.json` 是否与当前 `NativeModelConfig` 完全一致
- `state_dict` 是否能严格加载

这保证了“模型结构不一致但误用旧 checkpoint”的情况会 fail-fast。

### 9.1 加载训练完成的 Base 权重

从训练 run 的 checkpoint 目录加载最终权重：

```python
from pathlib import Path

from llm_lifecycle_lab.model.native import (
    NativeTransformer,
    load_native_model_config,
)
from llm_lifecycle_lab.tokenizer import NativeTokenizer

model_dir = Path("runs/native-60m-a10-base-v1-s42/checkpoints/step-00045191/model")
tokenizer = NativeTokenizer.from_directory("data/tokenizers/bilingual-60m-a10-v1")
model = NativeTransformer(load_native_model_config("configs/models/tiny-60m.yaml"))
model.load(model_dir)
model.eval()
```

Checkpoint 中的 `model.pt` 是严格加载的 `state_dict`，不是 Hugging Face
`save_pretrained` 目录。它不含 optimizer、scheduler、RNG 或数据流位置，
不能直接用于原 run 的精确恢复。完整训练证据见
[`results/native-60m-a10-base-v1-s42/`](../results/native-60m-a10-base-v1-s42/)。

## 10. 当前设计取舍

这个 Native 模型当前刻意保持简洁：

- 单机单进程，不考虑分布式并行
- 没有 gradient checkpointing
- 没有 HF `save_pretrained` 兼容层
- 没有 MoE、ALiBi、Sliding Window Attention 等扩展
- QK-Norm 只作为显式消融开关，默认基线关闭
- 只支持训练当前实现过的 Native Pretrain 路线

这样做的目的，是让训练逻辑、mask、cache、恢复和指标都可读、可测、可控。

## 11. 逐步验证模型

以下实验在安装好 requirements 后，从仓库根目录执行。前三步不需要公开数据、Tokenizer
或 GPU，只验证模型逻辑；随机 token 不用于评价语言能力。

### 步骤 1：检查模型和词表预算

```bash
python scripts/inspect_model.py --config configs/models/tiny-60m.yaml
python scripts/inspect_model.py \
  --config configs/models/tiny-60m.yaml \
  --compare-vocab-size 8192 \
  --compare-vocab-size 12288 \
  --compare-vocab-size 16384
```

**检查结果**：总参数为 62,927,616。词表比较只是参数估算，
不会自动修改模型、生成新的 Tokenizer 或开始训练。

### 步骤 2：前向、Loss、梯度与参数更新

```bash
python scripts/model_experiment.py
```

打开 [model_experiment.py](../scripts/model_experiment.py) 的 `main()`，可以按顺序读到：

1. 加载 60M 配置并随机初始化模型，创建 `[2, 16]` 的整数 token。
2. 调用 `model(input_ids=...)` 得到 `[2, 16, 16384]` 的 logits。
3. 使用 `logits[:, :-1]` 预测 `input_ids[:, 1:]`，直接计算交叉熵。
4. `loss.backward()` 计算梯度，裁剪梯度，然后 `optimizer.step()` 更新参数。

**检查结果**：每条序列 15 个 next-token 目标，共 30；loss、梯度范数有限，
`weight_update_max > 0` 表示观察的 embedding 行确实发生了更新。
它只在 CPU 上做一步实验，没有训练 run、调度器或真实语料；不是正式 Pretrain 的替代入口。
实际小数值随环境可能变化，不用固定 loss 或要求单步 loss 必须下降作为验收。

### 步骤 3：KV Cache 与保存加载

同一个脚本在更新后继续验证以下行为，无需运行另一份内嵌 Python：

- 完整前向与 prefix + cached suffix 的 logits 在容差内一致。
- 临时保存模型，重新构建并加载后，logits 保持一致。
- 生成 4 个新 token，默认输出形状为 `[2, 20]`，每层 K 为 `[2, 4, 16, 64]`。

结果中的 `cache_matches_full` 和 `checkpoint_matches_full` 应为 `true`；
对应比较失败会直接报错。临时模型文件会自动删除，不留下训练产物。
可添加 `--json` 输出结构化结果，或使用 `--sequence-length` / `--batch-size` 调整规模；
学习实验的长度必须至少为 2，且小于模型最大长度，以保留生成空间。

### 步骤 4：理解 Tokenizer/生成边界

真实文本必须由 checkpoint 对应的 Tokenizer 编码。Native chat-v1 预留协议如下：

| Token | 默认 ID | 用途 |
| --- | ---: | --- |
| `<\|pad\|>` | 0 | padding，不贡献 padding 位置的 loss |
| `<\|bos\|>` / `<\|eos\|>` | 1 / 2 | Pretrain 文档起止 |
| `<\|unk\|>` | 3 | 未知 token 标记 |
| `<\|im_start\|>` / `<\|im_end\|>` | 4 / 5 | 后续消息边界 |
| `<\|reserved_0\|>` ... `<\|reserved_7\|>` | 6 ... 13 | 未绑定具体语义的保留槽 |

消息格式为 `<|im_start|>{role}\n{content}<|im_end|>\n`，system 只能在首条，
其后 user/assistant 交替；正文不能注入控制 token。
Pretrain 只训练原始文本延续，不会因为预留 chat token 就自然获得指令跟随能力。
SFT 之前应使用普通文本续写观察模型，而不是用聊天质量判断 Smoke。

代码阅读路径：`NativeModelConfig` -> `NativeTransformer.forward` -> `TransformerBlock`
-> `CausalSelfAttention` -> `PretrainObjective` ->
[TrainingEngine](../src/llm_lifecycle_lab/training/engine.py)。实际训练步骤进入训练文档。

## 12. 对照 MiniMind：已吸收与待验证

本轮阅读基于 MiniMind 本地 commit `6fc918beb68a0d8c40452338df6319fe168014ba`，
不是对其所有历史版本的评价。依据为
[模型实现](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/model/model_minimind.py)
和 [Tokenizer 脚本](https://github.com/jingyaogong/minimind/blob/6fc918beb68a0d8c40452338df6319fe168014ba/trainer/train_tokenizer.py)。
这里仅记录设计参考与待验证项，当前 Native 模型和 checkpoint 结构不变。

| 观察点 | MiniMind 当前实现 | 本项目的取舍或后续完善 |
| --- | --- | --- |
| RoPE 与 QK-Norm | 模型级共享 cos/sin；默认 Q/K RMSNorm | 已有共享 cache；QK-Norm 保留独立消融，不直接打开基线开关 |
| 无 padding 的注意力 | 满足条件时使用 SDPA `is_causal` | CPU 组 batch 时省略全真 mask；CUDA/MPS 可用 causal 路径，CPU 使用共享显式 mask；仍需测量 CUDA 吞吐 |
| 词表与 MLP 预算 | 6,400 词表；768 hidden 默认 MLP 为 2,432 | 我们 16K 词表、768 hidden、MLP 2,048；缩词表应同时比较中英文压缩率、上下文利用率和质量，不能只比较参数量 |
| 推理 logits | 支持 `logits_to_keep` 以限制输出位置 | 已支持生成时只投影最后位置，训练默认保留全序列，KV Cache 不裁剪 |
| HF 兼容 | `PreTrainedModel`、`PretrainedConfig` 与生成接口 | 后续以独立适配/导出层对接，不在 Native forward 中加入训练阶段分支 |
| 长上下文 | 配置更大 RoPE cache 和可选 YaRN | 512 长度训练不能据此宣称具备 32K 能力；需长文本数据与专门评测后再扩展 |
| MoE | 有专家路由、top-k 和辅助 loss | 暂不引入；它增加变量与算子调度成本，也不能替代当前 Dense Base 的数据与评测改进 |

### 公平比较需要控制什么

- MiniMind 默认 768 hidden 配方的 token embedding 约 4.92M 参数；我们约 12.58M。
  即使总参数都在 60M 附近，注意力 head、MLP 和词表分配也不同，不是权重兼容模型。
- 若同时改变层数、hidden、MLP 和 GQA 比例，得到的是结构组合对照，
  不是仅层数变化的纯消融；独立实验应明确各变量。
- 相同 token 预算不等于相同训练 FLOPs 或时间。需要同时公布中英文 dev/test 指标、
  峰值显存、监督 token/s、总耗时；不同词表还应检查相同文本的 bytes/token 和 bits-per-byte。

### 推荐顺序

1. 冻结并完成宽浅 60M 基线，记录中英文指标、吞吐、显存和耗时，再开始改变实验变量。
2. 在现有输出、梯度和缓存等价测试之外，增加独立数值参考，覆盖 attention 与 FP32/BF16。
3. 单独做残差投影初始化消融，不把新的初始化混入原基线。
4. 在相同数据与预算下做 CUDA profiling，再测量 GQA、优化器或其他性能调整的收益。
5. 最后做词表与容量实验；比较中英文压缩率和 bits-per-byte，新 Tokenizer 使用新训练路线。

结构对照需另行冻结新配置；HF 导出和长上下文另行扩展，不预设 MoE 或更深模型更优。

训练侧的恢复、预算和复现缺口见
[Pretrain 的后续完善](./NATIVE_PRETRAIN_GUIDE.md#11-保留边界与后续完善)。
