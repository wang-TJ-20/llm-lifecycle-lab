# HF 导出与 Qwen 迁移

项目保留两条模型路线：

- **Native**：从零实现，用于理解结构、训练与受控实验。
- **Transfer**：使用标准 Hugging Face 与 PEFT，连接真实开源模型生态。

两条路线复用 SFT 数据规范和能力探针，但模型、Tokenizer、checkpoint 和报告
保持隔离。不能因为题目相同就忽略 Tokenizer 与后端差异。

## 1. 安装可选依赖

基础 Native 路线不依赖 Transformers。需要导出或 Qwen 时安装：

```bash
python -m pip install -r requirements-hf.txt
```

项目冻结 `transformers==4.53.3`、`peft==0.17.1`、
`accelerate==1.10.1` 和 `safetensors==0.6.2`。版本不一致时主动失败，
避免同一配置在不同实现版本上产生不可解释的变化。

## 2. 将 Native 导出为标准 HF 模型

对已经包含 Tokenizer 的本地发布包：

```bash
python scripts/export_hf.py \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --output build/hf/native-60m-base-v1
```

原训练 checkpoint 需要额外指定 `--tokenizer`。目标目录存在时拒绝覆盖。
导出使用标准 `LlamaForCausalLM`；启用 QK-Norm 的 Native 配置使用标准
`Qwen3ForCausalLM`，不依赖 `trust_remote_code`。

导出不是简单改文件名。命令会重新从磁盘加载 HF 文件，并验证：

- 英文、中文和 NFKC 文本的 Token ID 完全一致；
- 多个输入上的 float32 logits 在容差内一致；
- greedy 生成 token 完全一致；
- `native-chat-v1` 的对话 token 完全一致；
- 使用 KV Cache 的下一个 token logits 与 Native 全序列前向一致。

`hf_manifest.json` 记录全部文件 hash；额外文件、删除或内容变化都会导致加载失败。
本机已对 `step-00005649` 完成上述验证，最大 logits 误差为 0。
生成目录位于 `build/`，默认不提交 Git。

标准加载方式：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "build/hf/native-60m-base-v1"
tokenizer = AutoTokenizer.from_pretrained(
    path, local_files_only=True, trust_remote_code=False
)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, trust_remote_code=False
)
```

Base 模型只用于续写；存在 chat template 不代表它已经具备指令能力。

## 3. CLI 续写与对话

Native Base 默认自动选择续写模式：

```bash
python scripts/chat.py \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --prompt "Once upon a time"
```

HF 导出使用：

```bash
python scripts/chat.py \
  --backend hf \
  --checkpoint build/hf/native-60m-base-v1 \
  --prompt "Once upon a time"
```

后续 Instruct 或 DPO checkpoint 可使用 `--mode chat`。交互模式提供 `/reset`
和 `/exit`。上下文默认溢出即失败；显式设置
`--history-policy drop-oldest` 才会按完整 user/assistant 回合删除最早历史。
不会静默截断半个回合。

## 4. 固定 Qwen 快照

Transfer 路线固定为 `Qwen/Qwen3-0.6B-Base`。先从模型页面取得一个明确的
40 位 commit SHA，不使用漂移的 `main`：

```bash
python scripts/prepare_transfer.py \
  --revision YOUR_40_CHARACTER_COMMIT_SHA \
  --output artifacts/qwen3-0.6b-base
```

下载完成后生成：

- 上游 repo ID、commit、Transformers 版本和全部文件 hash；
- 可校验的本地快照；
- `sft.example.yaml` 示例配置。

后续加载只使用 `local_files_only=True` 和 `trust_remote_code=False`。
模型 revision、Tokenizer revision、依赖版本或文件 hash 任一不一致都会失败，
不会自动回退到网络或随机权重。

当前工作区没有下载真实 Qwen 权重，也没有形成 Qwen 能力结果。

## 5. LoRA SFT

准备双语 SFT Data Manifest 后，检查快照中的示例配置并运行：

```bash
python scripts/train_transfer.py \
  --config artifacts/qwen3-0.6b-base/sft.example.yaml \
  --run-id qwen3-0.6b-lora-sft-001
```

配置只允许 `full` 或 `lora`。LoRA 默认目标为 `q_proj`、`v_proj`，
底座参数全部冻结，checkpoint 只保存 adapter 权重及其底座绑定。
恢复时重新加载完全相同的本地快照，然后校验 adapter 配置和 manifest hash。

不下载模型也可验证机制：

```bash
python scripts/transfer_experiment.py --mode train
python scripts/transfer_experiment.py --mode resume
```

该实验创建一个随机微型 Qwen3，仅验证：

- PEFT adapter 是唯一可训练参数；
- 底座参数逐位不变；
- adapter 确实更新；
- 连续训练与中断恢复的权重、优化器、scheduler、RNG 和数据位置一致。

它不使用 `Qwen/Qwen3-0.6B-Base` 权重，不能产生模型能力结论。

## 6. 统一评测

HF 导出或 Qwen 本地快照：

```bash
python scripts/evaluate_capabilities.py run \
  --backend hf \
  --checkpoint build/hf/native-60m-base-v1 \
  --output runs/evaluations/native-hf-export-v1
```

LoRA checkpoint 还需要提供底座：

```bash
python scripts/evaluate_capabilities.py run \
  --backend hf \
  --checkpoint runs/qwen3-lora/checkpoints/step-00000002 \
  --base-model artifacts/qwen3-0.6b-base \
  --output runs/evaluations/qwen3-lora-v1
```

评测题和规则与 Native 相同，但报告额外绑定 `model_route`、Tokenizer、
Transformers/PEFT 版本和 HF 实现代码。Native 与 Qwen 的 Tokenizer 不同，
BPB 可以并列查看，但不会计算成同一模型阶段的直接 delta。

## 7. QLoRA

配置中把 `training_method` 设为 `qlora`，并给出 4-bit 量化设置：

```yaml
model:
  training_method: qlora
  lora:
    rank: 8
    alpha: 16
    dropout: 0.0
    target_modules: [q_proj, v_proj]
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true
    bnb_4bit_compute_dtype: bfloat16
```

运行方式与 LoRA 相同：

```bash
python scripts/train_transfer.py \
  --config configs/pipelines/qwen3-qlora-60m.yaml \
  --run-id qwen3-qlora-60m-001
```

门禁独立于 LoRA：

- 需要 CUDA 与 `bitsandbytes`；`load_in_4bit` 目前只接受 `true`。
- `bnb_4bit_quant_type` 只允许 `nf4` / `fp4`，
  `bnb_4bit_compute_dtype` 只允许 `bfloat16` / `float16` / `float32`，
  选 `bfloat16` 时会检查设备是否真的支持 BF16。
- 量化设置与 `bitsandbytes` 版本一并写入 run binding，
  加载 checkpoint 时逐项比对，防止用不同量化配置复用同一个 adapter。
- 4-bit 权重在加载时由 `device_map` 固定，禁止再用 `.to()` 搬移；
  评测 QLoRA checkpoint 必须显式 `--device cuda`。

`bitsandbytes` 与硬件强相关，未加入 `requirements-hf.txt`；
缺失时报错并给出安装提示，不静默回退到非量化路径。

## 当前边界

- Native 60M HF 导出与 CPU 一致性：已验证。
- 随机微型 Qwen3 LoRA 训练和精确恢复：已验证。
- 真实 Qwen 快照下载、LoRA/QLoRA GPU 训练：已执行，见
  [GPU 后训练报告](./experiments/native-60m-lifecycle-gpu-v1.md)。
- QLoRA 已在 CUDA 上验证（峰值显存 7.267 → 5.572 GiB）。
- Qwen 侧指令探针为 0：Tokenizer 与 `native-chat-v1` 控制串不匹配，
  且 LoRA 未泛化到未见模板；与 Native 并列查看，不相减。
- Qwen LoRA DPO、Qwen full 与 LoRA 同底座对照：未开始。

[返回项目路线](./test.md)
