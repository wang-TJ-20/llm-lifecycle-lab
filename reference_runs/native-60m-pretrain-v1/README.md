# Native 60M Pretrain Reference Run v1

该目录定义首个 Native 60M 双语预训练 Reference Run。它是待 Linux/CUDA
环境执行的发布配方，不包含伪造的训练指标或模型权重。

## 固定输入

| 项目 | 值 |
| --- | --- |
| Pipeline | `configs/pipelines/native-60m-reference.yaml` |
| 模型 | `tiny-60m`，62,927,616 参数 |
| 数据 | `bilingual-60m-v1` |
| Tokenizer SHA-256 | `7455ac1168f36a3d7dca03fee28784f97899fe233f496484a11e5da2e44a2f32` |
| 序列长度 | 512 |
| Train 监督 token | 46,184,530 |
| 预算 | 1 epoch，5,649 optimizer steps |
| 有效 batch | 16 sequences，最多 8,176 监督 token/step |
| 最低硬件 | Linux，单张 22 GiB 以上 CUDA GPU |

正式运行必须基于干净的 Git commit。`runtime_environment.json` 会自动记录 commit、
dirty 状态、Python/PyTorch 版本、平台、CUDA 版本和 GPU 显存。

## 执行

```bash
git status --short

uv sync --frozen --extra training --extra dev

uv run llmlab doctor \
  --config configs/pipelines/native-60m-reference.yaml

uv run llmlab train pretrain \
  --config configs/pipelines/native-60m-reference.yaml \
  --run-id native-60m-pretrain-v1
```

训练完成后，对同一个最终 checkpoint 独立执行 dev 和 test：

```bash
uv run llmlab eval pretrain \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-pretrain-v1/checkpoints/step-00005649 \
  --split dev \
  --json

uv run llmlab eval pretrain \
  --config configs/pipelines/native-60m-reference.yaml \
  --checkpoint runs/native-60m-pretrain-v1/checkpoints/step-00005649 \
  --split test \
  --json
```

最后执行机器验收：

```bash
uv run llmlab reference verify \
  --spec reference_runs/native-60m-pretrain-v1/reference.yaml \
  --run runs/native-60m-pretrain-v1
```

## 验收门禁

命令必须全部通过：

1. Run 状态为 `completed`，配置 hash 与发布 Pipeline 完全一致。
2. Data、Tokenizer 和 Packed Data 构成一致的 SHA-256 链。
3. 预算为 46,184,530 token，最终 step 为 5,649，覆盖率在 `[1.0, 1.01]`。
4. baseline、最终训练指标和中英文分桶指标存在、有限且可加权还原。
5. 最佳 dev loss 严格低于随机初始化 baseline。
6. 最终 checkpoint 包含模型、optimizer、scheduler、RNG 和数据流位置。
7. 独立 dev/test 报告都存在并包含 aggregate、英文和中文指标。
8. 运行环境为 Linux/CUDA，GPU 显存不少于 22 GiB，Git commit 干净可追溯。

峰值显存取 `metrics.jsonl` 中的 `cuda_max_memory_allocated_bytes`，训练耗时取
`training_result.json` 的 `elapsed_seconds`。云资源成本应在发布报告中根据实际实例
单价和占用时长单独列出；在没有账单依据时不得估算为实测成本。

当前 macOS 工作区只完成了数据、packing、预算和验收代码验证，尚未执行该 CUDA
Reference Run。
