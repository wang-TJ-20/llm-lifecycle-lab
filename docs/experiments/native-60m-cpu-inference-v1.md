# Native-60M CPU 推理与 dynamic INT8 v1

本记录使用已经通过 Native/HF 一致性验收的 `native-60m-base-v1` HF 导出，
比较同一进程中的 float32 与 PyTorch dynamic INT8。没有训练或修改权重。

## 运行条件

- 模型 manifest SHA-256：
  `6d4f92288746135a868f315f3ef9f0a17596cc5975bfb2077851574115567a42`
- 设备：macOS 15.7.1、arm64、10 个逻辑 CPU，单线程。
- 运行时：Python 3.13.9、PyTorch 2.14.0、Transformers 4.53.3。
- 量化：`torch.ao` dynamic qint8 Linear，后端 `qnnpack`，原地转换。
- 提示：4 个固定中英文续写提示。
- 生成：greedy，每个提示最多 16 个新 token，重复 3 次。
- 报告 digest：
  `997d85c40b53f08e5787cccc584fd091659c77ac07b1ccf18c9d1fd43274a16a`
- `benchmark.json` 文件 SHA-256：
  `d6e58633aedc668202ce515b5df40ecea26691f2f6b2d8da49374d5fac0685f8`

复现时必须使用新的输出目录：

```bash
python scripts/benchmark_inference.py \
  --model build/hf/native-60m-base-v1 \
  --output runs/benchmarks/native-60m-cpu-rerun \
  --max-new-tokens 16 \
  --repeats 3 \
  --threads 1
```

原始本机报告位于 `runs/benchmarks/native-60m-cpu-v1/`，默认不提交 Git。

## 结果

| 模式 | state dict 字节 | 中位 token/s | 观测峰值 RSS | greedy 匹配 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 251,740,421 | 176.844 | 631,668,736 | - |
| dynamic INT8 | 113,674,947 | 92.885 | 734,674,944 | 1.000 |

相对 FP32：

- INT8 序列化状态为 `45.16%`，减少约 `54.84%`。
- INT8 吞吐为 `52.52%`，在当前协议下约慢 `47.48%`。
- 12/12 个固定生成序列与 FP32 完全一致。
- 观测 RSS 没有下降，不能声称该路径节省了进程运行内存。

## 结论边界

这次实验支持“dynamic INT8 明显减少模型状态体积”，不支持“在本机更快”或
“进程内存更低”。60M、batch=1、单线程与 qnnpack 是当前结论的重要前提。

RSS 包含运行时和内存分配器，且只在 generate 前后采样。INT8 由已测完的
FP32 模型原地转换，避免同时保留两份完整模型，但转换期间申请过的内存仍可能
留在进程分配器中。因此 RSS 只作为诊断证据，不作为跨模式的模型独占内存。

`torch.ao` dynamic quantization 在当前 PyTorch 版本已标记弃用。未来迁移到
`torchao` 后需要发布 v2 协议并重新建立基线，不能把新后端结果覆盖到本记录。

[返回 CPU 推理指南](../CPU_INFERENCE_GUIDE.md)
