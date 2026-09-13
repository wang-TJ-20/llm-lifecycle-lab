# HF 模型 CPU 推理与量化

本指南使用已经通过一致性验收的 HF 导出模型，在 CPU 上比较 float32 与
PyTorch dynamic INT8。目标是建立可复现的本机诊断，不宣称某种量化方案在
所有硬件上都更快或更省运行内存。

## 1. 前提

安装冻结的 HF 依赖：

```bash
python -m pip install -r requirements-hf.txt
```

输入必须是带 `hf_manifest.json` 的独立 HF 模型。Native checkpoint 应先导出：

```bash
python scripts/export_hf.py \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --output build/hf/native-60m-base-v1
```

benchmark 会先校验 manifest 和所有文件 hash。它不接受未绑定底座的 adapter，
也不会联网补齐缺失文件。

## 2. 运行

```bash
python scripts/benchmark_inference.py \
  --model build/hf/native-60m-base-v1 \
  --output runs/benchmarks/native-60m-cpu-rerun \
  --max-new-tokens 16 \
  --repeats 3 \
  --threads 1
```

输出目录必须不存在。命令固定使用四个中英文提示，先 warmup，再分别记录每次
greedy 生成。`--no-int8` 可只测 float32。

dynamic INT8 使用：

```python
torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
    inplace=True,
)
```

程序从当前 PyTorch 构建支持的 `x86`、`fbgemm`、`onednn`、`qnnpack`
后端中显式选择一个。没有可用后端时立即失败，不生成不完整的 INT8 报告。

## 3. 报告口径

输出目录包含：

- `benchmark.json`：协议、运行时、逐次样本和稳定 digest；
- `benchmark.md`：适合人工查看的模式对照。

主要指标：

| 指标 | 含义 |
| --- | --- |
| `state_dict_bytes` | 将当前模型 state dict 序列化后的字节数 |
| `median_seconds` | 全部固定样本的生成耗时中位数 |
| `median_tokens_per_second` | 每个样本生成 token/s 的中位数 |
| `peak_observed_rss_bytes` | generate 前后采样到的最高进程 RSS |
| `greedy_sequence_match_fraction` | INT8 与 FP32 完整生成 token 序列相同比例 |

状态大小不是发布文件大小，RSS 也不是模型独占内存。RSS 包含 Python、Tokenizer、
PyTorch 和内存分配器，且只在生成前后采样，可能遗漏瞬时峰值。float32 测量后
使用原地量化，避免同时持有完整的两份模型，但分配器仍可能保留已申请的内存。

## 4. 解释结果

先分别判断三件事：

1. 状态是否变小；
2. 当前硬件上的吞吐是否改善；
3. 固定 greedy 输出是否保持。

三者不能互相替代。小模型、batch=1 或缺少优化 kernel 时，INT8 完全可能状态
更小但吞吐更慢。greedy 一致也只说明这些固定提示没有改变 token 序列，
不等于能力无损。

当前 Native-60M 实测见
[CPU FP32 / dynamic INT8 v1](./experiments/native-60m-cpu-inference-v1.md)。

## 5. 边界

- 当前产物是进程内 PyTorch dynamic INT8，不是可移植 HF 权重。
- 未实现 GGUF、GPTQ、AWQ、bitsandbytes、llama.cpp 或 vLLM 导出。
- 当前冻结的 PyTorch 已弃用 `torch.ao` dynamic quantization；迁移到
  `torchao` 时必须建立新协议版本，不能静默改变 v1 结果。
- 不同机器、线程数、量化引擎和依赖版本的报告不能直接计算加速比。

[返回项目路线](./test.md)
