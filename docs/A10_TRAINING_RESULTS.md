# A10 从零重跑执行记录

本记录为单卡 A10 从零重跑的完整执行证据。
当前操作入口见 [A10 手册](./A10_TRAINING_GUIDE.md)。

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
| 正式 8-epoch A10 Base | 已完成，commit `f702145` |
| 新 Base 质量验收 | 已通过 |
| A10 SFT Base/data/run 门禁 | 已冻结；Base 和数据 inputs-only 校验通过 |
| A10 SFT 数据与预算 | 14,799 条对话；首轮 1M 监督 token，9,379 steps |
| A10 SFT smoke/pilot/恢复 | 未启动，待用户手动执行 |
| SFT 启动硬件 preflight | 当前准备会话 CUDA 设备数为 0；须在绑定 A10 的 GPU 终端检查 |
| 正式 A10 SFT | 未启动 |

不能把 CPU 测试或合成输入探测当成本轮训练通过。
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
正式 run 已完成，test 固定子集评测结果见下节。

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

## 正式 Base 结果

正式 8-epoch Base 于 2026-11-02 在 Worker `1158961` 完成并通过验收。

| 指标 | 值 |
| --- | ---: |
| run ID | `native-60m-a10-base-v1-s42` |
| 训练提交 | `f7021451919f86d5de4d197dbe4fcf01ad2ef42b` |
| steps | 45,191 |
| 监督 token | 369,480,328 |
| token coverage | 100.0011% |
| 训练耗时 | 4 小时 39 分钟 |
| baseline dev loss | 9.859973 |
| final dev loss | 2.755610 |
| best dev loss | 2.708168 @ step 33,000 |
| final English dev loss | 1.955765 |
| final Chinese dev loss | 3.490631 |
| final/best 回退 | 1.75% |
| fixed-subset test loss | 2.732666 |
| fixed-subset English test loss | 1.954598 |
| fixed-subset Chinese test loss | 3.421753 |
| final weights SHA-256 | `0c3290d6c76bbf775aba20c1d543a062e26aafb987a7d15bde1b652e4dbbb46c` |

验收检查全部通过：
- 源码干净性、配置身份、数据/Packing/Tokenizer hash 一致；
- train/dev/test 按 source_id 分组无重叠；
- 最终 checkpoint 架构、长度 512、绑定权重和有限值校验通过；
- 重载最终 checkpoint 的 dev 指标与训练最终记录完全一致；
- test 仅评测一次，固定 1024 个分层窗口，不代表全量 test 能力结论。

证据归档在 `results/native-60m-a10-base-v1-s42/`，包含 28 个校验文件及 SHA-256 校验单。
验收报告为 [a10-acceptance.json](file:///mnt/bn/smart-customer-service/users/zitong/project/llm-lifecycle-lab/results/native-60m-a10-base-v1-s42/a10-acceptance.json)。
本轮 Base 独立于任何历史产物，SFT 使用独立冻结的初始化、数据与 run 门禁。

归档维护说明：2026-11-02 按用户要求移除了历史硬件对照字段，并同步更新
`SHA256SUMS` 与 `checksums.json`。A10 训练指标、评测、权重和验收结论未改变；
`accepted_at` 仍表示原始验收时间，校验清单对应维护后的归档。

最终权重已复制到
[final_checkpoint/model/model.pt](file:///mnt/bn/smart-customer-service/users/zitong/project/llm-lifecycle-lab/results/native-60m-a10-base-v1-s42/final_checkpoint/model/model.pt)，
并加入两份校验清单。该文件配置为 Git LFS 跟踪；这表示已准备好提交，不表示已经上传。
训练目录和其他权重仍保持忽略，原始 A10 checkpoint、指标和验收结论未改变。

## 下一步

1. 检查并提交已准备的 SFT 代码、配置、文档和测试；
2. 用户手动运行 SFT smoke、同卡精确恢复验证与 pilot；
3. 检查结果后，用户手动启动 1M 预算的正式 A10 SFT。

独立操作手册见 [A10 SFT 准备与手动训练](./A10_SFT_GUIDE.md)。
本次只完成数据、初始化和预算校验，未启动任何 A10 SFT 训练。
