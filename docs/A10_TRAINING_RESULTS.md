# A10 从零重跑执行记录

本记录与历史 RTX 4090 的 Base-v1 分开。历史权重、结果和冻结配置保留归档，
本轮 Base 不从它们初始化。当前操作入口见 [A10 手册](./A10_TRAINING_GUIDE.md)。

## 环境与数据

2026-11-01 在 Worker `1158961` 完成以下检查：

| 项目 | 本轮结果 |
| --- | --- |
| GPU | 单张 NVIDIA A10，23,028 MiB |
| CUDA 可见容量 | 21.988 GiB；驱动保留后低于 nvidia-smi 报告值 |
| 驱动 | 535.161.08 |
| Python | 3.11.2 |
| Torch / CUDA wheel | 2.7.1+cu126 / 12.6 |
| CUDA / BF16 | 可用 |
| 独立环境 | `.venv-a10`，不继承系统 site-packages |
| pip check | 通过 |
| 原有完整 Python 测试 | 独立环境 187 passed，145.91 秒 |
| 新门禁与更新后完整测试 | 202 passed，151.13 秒；ruff check/format 通过 |
| A10 Doctor 修正后完整测试 | 205 passed，151.24 秒 |
| 中英文源数据 | 100,000 条故事 + 126,000 条百科，固定来源 hash 通过 |
| 混合数据 | 226,000 条，重新 round-robin 物化 |
| 分组切分 | 按 source_id；train/dev/test = 180,657 / 22,853 / 22,490 |
| Tokenizer | 重新 train-only BPE，16,384 词表 |
| Packing | seq512 已完成；train 90,381 窗口、46,184,530 监督 token |

混合文本 SHA-256：
`af07e4c3a610637239a3295fd4754af316bd74c5d08bc25b63fd04a147339589`。
本轮 Tokenizer 内容 SHA-256：
`7455ac1168f36a3d7dca03fee28784f97899fe233f496484a11e5da2e44a2f32`。
Tokenizer 与历史内容一致是重新训练后的检查结果，不是复制旧 Tokenizer。
manifest 包含本轮 ID、路径和时间，不能直接沿用历史 manifest hash。

## 训练状态

| 阶段 | 状态 |
| --- | --- |
| 8-step A10 BF16 smoke | 已完成，开发门禁通过 |
| 100-step A10 BF16 pilot | 已完成，开发门禁通过 |
| A10 受控中断恢复 | 确定性 CUDA 对照逐位通过；普通路径不保证逐位一致 |
| 正式 8-epoch A10 Base | 未启动；需要前序验证和干净提交 |
| 新 Base 发布包与质量验收 | 未完成 |
| A10 SFT Base/data/run 门禁 | 新 Base 完成后独立冻结 |
| 正式 A10 SFT | 未启动 |

不能把 CPU 测试、旧 Worker 的合成输入探测或 4090 历史训练当成本轮训练通过。
开发 smoke/pilot 如使用未提交工作树，必须记录 dirty；
正式 Base 需要 clean-tree，不能继承历史 dirty 例外。

启动前 Doctor 的历史通用 `tiny-60m` profile 要求至少 22 GiB，
因此本卡 21.988 GiB 被拒绝，虽然真实批次前后向通过。
已新增限定单卡 A10/CUDA BF16 的 `tiny-60m-a10` profile，最低 CUDA 可见
容量为 21.9 GiB；原通用门槛、训练配方和质量门禁不变。

## 短训练实测

| 指标 | smoke | pilot |
| --- | ---: | ---: |
| optimizer steps | 8 | 100 |
| 实际监督 token | 65,408 | 817,600 |
| baseline dev loss | 9.855544 | 9.855544 |
| final dev loss | 8.295427 | 6.516608 |
| final English dev loss | 7.691859 | 5.477941 |
| final Chinese dev loss | 8.868767 | 7.503255 |
| TrainingResult 耗时 | 6.27 秒 | 41.63 秒 |

Pilot 的日志峰值分配显存约 1.35 GiB，记录 step 的平均吞吐约 23,058
监督 token/秒。耗时不包含此前数据准备和训练入口检查；显存/吞吐不是正式
Base 或 SFT 的承诺。短 dev 评测只覆盖 16 个窗口，不代表模型语言能力已完成。
本轮 8-epoch 预算重新换算为 369,476,240 目标监督 token 和 45,191 steps，
目前没有启动正式 run，也没有读取 test 评分。

普通恢复实验保留在 `runs/native-60m-a10-base-resume-s42`。
数据位置和 token 数一致，但独立连续 run 与中断 run 在中断前就出现数值差异，
因此不能仅凭最终逐位不一致认定 checkpoint 状态丢失。
不得放宽逐位比较阈值后把失败改写为通过。

确定性对照使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和
`torch.use_deterministic_algorithms(True)`，连续 run 为
`native-60m-a10-base-deterministic-smoke-s42`，恢复 run 为
`native-60m-a10-base-deterministic-resume-s42`。
从 step 4 恢复到 step 8、65,408 个监督 token；
权重、optimizer/scheduler/RNG、trainer/数据流三项均逐位一致。
该设置只用于本次恢复对照，没有悄悄改变普通 smoke/pilot 的执行条件。
正式 run 是否启用确定性算法必须在启动前记录，不承诺跨硬件/依赖等价。
