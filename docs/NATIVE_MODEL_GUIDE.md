# Native 10M/60M 模型结构说明

本文专门说明 LLM Lifecycle Lab 中 Native 路线模型的结构设计、配置缩放方式和代码边界。

如果你想看“怎么准备数据、怎么训练、怎么恢复 checkpoint、怎么评测”，请看 [Native 10M/60M Pretrain 训练指南](./NATIVE_PRETRAIN_GUIDE.md)。

## 1. 设计目标

Native 模型的定位不是追求同参数量下的最佳效果，而是作为一个足够真实、又足够可读的教学模型：

- 结构上接近现代 decoder-only LLM。
- 10M 和 60M 共用同一份实现，只通过配置缩放。
- 代码边界清晰：模型只负责张量级前向、生成和 checkpoint，不负责 stage-specific loss。
- 能支持真实的 Pretrain 训练、增量生成和 KV Cache。
- 默认使用同一份 16,384 词表双语 Tokenizer，在中英文混合语料上训练。

模型结构本身与语言无关。“支持中英文”具体表示 Tokenizer 能无损编码两种语言，且默认
checkpoint 同时接受中英文训练数据；它不表示 10M/60M 参数规模能够达到通用双语助手能力。

当前实现文件：

- 配置：[src/llm_lifecycle_lab/model/native/config.py](../src/llm_lifecycle_lab/model/native/config.py)
- 注意力：[src/llm_lifecycle_lab/model/native/attention.py](../src/llm_lifecycle_lab/model/native/attention.py)
- RMSNorm：[src/llm_lifecycle_lab/model/native/normalization.py](../src/llm_lifecycle_lab/model/native/normalization.py)
- Block/SwiGLU：[src/llm_lifecycle_lab/model/native/layers.py](../src/llm_lifecycle_lab/model/native/layers.py)
- 主模型：[src/llm_lifecycle_lab/model/native/transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)
- 采样生成：[src/llm_lifecycle_lab/model/native/generation.py](../src/llm_lifecycle_lab/model/native/generation.py)

## 2. 两档模型

两档模型都来自同一个 `NativeTransformer`：

| 配方 | 参数量 | 层数 | Hidden | Q Heads | KV Heads | MLP | Max Seq | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke-10m` | 9,915,200 | 4 | 320 | 5 | 1 | 960 | 256 | CPU/MPS/CUDA smoke |
| `tiny-60m` | 62,927,616 | 8 | 768 | 12 | 4 | 2048 | 512 | 正式 Pretrain 实验 |

对应配置：

- [configs/models/smoke-10m.yaml](../configs/models/smoke-10m.yaml)
- [configs/models/tiny-60m.yaml](../configs/models/tiny-60m.yaml)

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
- 所有核心维度必须为正值

### 2.1 词表参数预算

词表会同时决定 token embedding 和 LM head 的参数量；当前两者权重绑定，因此只计
一份矩阵：

| 配方 | 词表 | Token 参数 | 占总参数 |
| --- | ---: | ---: | ---: |
| `smoke-10m` | 16,384 | 5,242,880 | 52.88% |
| `tiny-60m` | 16,384 | 12,582,912 | 20.00% |

10M 配方超过一半参数位于 embedding，这意味着减小词表会显著改变“模型容量”的含义。
项目保留 16K 作为中英文统一基线，同时通过 `model inspect --compare-vocab-size`
显式比较 8K/12K/16K，而不是静默更换默认词表。

### 2.2 60M 架构消融

项目提供四个约 60M 配方，组成宽浅/深窄与 QK-Norm 的 2×2 对照：

| 模型 | 层数×Hidden | Heads/KV | MLP | QK-Norm | 参数量 |
| --- | --- | --- | ---: | --- | ---: |
| `tiny-60m` | 8×768 | 12/4 | 2048 | 关闭 | 62,927,616 |
| `tiny-60m-qk-norm` | 8×768 | 12/4 | 2048 | 开启 | 62,928,640 |
| `tiny-60m-deep-narrow` | 19×512 | 8/4 | 1344 | 关闭 | 62,574,080 |
| `tiny-60m-deep-narrow-qk-norm` | 19×512 | 8/4 | 1344 | 开启 | 62,576,512 |

深窄组与宽浅基线的参数差小于 0.6%。四条 pipeline 除模型 ID 和模型配置路径外，
数据、Tokenizer、seed、训练预算、优化器和评测设置完全相同，并有单测防止配置漂移。
这使两类变量可以分别比较；在完成同预算 CUDA 实验前，不预设哪种结构更好。

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
- 在 `float32` 中计算方差，再转回输入 dtype
- 不使用均值中心化

这比 LayerNorm 更贴近当前主流 LLM 的归一化方式。

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

mask 逻辑见 [_attention_mask()](../src/llm_lifecycle_lab/model/native/attention.py)：

- 如果是完整前缀、无外部 mask、且没有 cache，直接让 `scaled_dot_product_attention(..., is_causal=True)` 走内置 causal 路径。
- 只要引入 cache 或显式 `attention_mask`，就构造布尔 mask。
- 当存在 cache 时，`attention_mask` 的长度必须是 `past_length + current_length`。

这也是为什么 `forward()` 在增量推理时会强校验 `attention_mask` 形状。

## 6. KV Cache 与生成

`forward()` 支持：

- `use_cache`
- `cache`

其中 cache 的结构是：

```text
tuple[layer] -> (key, value)
```

每层一个 `(K, V)`，所以 cache 层数必须与 block 数完全一致，否则直接报错。

对应校验在 [transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)。

### 生成流程

`generate()` 的行为：

- 默认 greedy；如果 `do_sample=True`，支持 `temperature` 和 `top_p`
- 每步只把最新 token 喂回模型，并复用 cache
- 如果配置了 `eos_token_id`，所有样本都结束时提前停止
- 已结束样本会被 `pad_token_id` 填充
- `prompt_tokens + max_new_tokens` 不能超过模型 `max_sequence_length`

这部分见 [transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py) 和 [generation.py](../src/llm_lifecycle_lab/model/native/generation.py)。

## 7. 初始化与参数量

初始化在 [transformer.py](../src/llm_lifecycle_lab/model/native/transformer.py)：

- `Linear` 和 `Embedding` 都用均值 0、标准差 `initializer_range` 的正态分布
- 不单独处理 bias，因为这些层都不带 bias

参数量可以从两处看：

1. `NativeModelConfig.estimated_parameter_count`
2. `NativeTransformer.parameter_count`

前者用于配置级估算，后者是实例化后的真实总参数数。

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

## 10. 当前设计取舍

这个 Native 模型当前刻意保持简洁：

- 单机单进程，不考虑分布式并行
- 没有 gradient checkpointing
- 没有 HF `save_pretrained` 兼容层
- 没有 MoE、ALiBi、Sliding Window Attention 等扩展
- QK-Norm 只作为显式消融开关，默认基线关闭
- 只支持训练当前实现过的 Native Pretrain 路线

这样做的目的，是让 10M/60M 的训练逻辑、mask、cache、恢复和指标都可读、可测、可控。

## 11. 怎么验证自己理解对了

可以按下面顺序快速自检：

1. 看 [configs/models/smoke-10m.yaml](../configs/models/smoke-10m.yaml) 和 [configs/models/tiny-60m.yaml](../configs/models/tiny-60m.yaml)
2. 运行：

```bash
uv run llmlab model inspect --config configs/models/smoke-10m.yaml
uv run llmlab model inspect --config configs/models/tiny-60m.yaml
uv run llmlab model inspect --config configs/models/tiny-60m-qk-norm.yaml
uv run llmlab model inspect --config configs/models/tiny-60m-deep-narrow.yaml
uv run llmlab model inspect \
  --config configs/models/tiny-60m-deep-narrow-qk-norm.yaml
```

3. 再看训练如何调用它：

- [run_native_pretraining()](../src/llm_lifecycle_lab/training/pretrain.py)
- [TrainingEngine.train()](../src/llm_lifecycle_lab/training/engine.py)

如果你接下来更关心怎么真正跑起来，直接看 [Native 10M/60M Pretrain 训练指南](./NATIVE_PRETRAIN_GUIDE.md)。
