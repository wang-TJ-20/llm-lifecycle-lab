## 4. 扩大探针后的重测（lifecycle-v3）

v1/v2 的 instruction 与 qa 各只有 4 个 case，0.25 的差异就是 1 个 case 翻转，
落在噪声内，因此第二轮无法判断任何后训练阶段的真实效果。

新增 `scripts/build_evaluation_suite.py` 生成 `configs/evaluation/lifecycle-v3.yaml`：
140 个案例，每个受评类别 30+，构建时逐条比对全部训练源（泄漏数 0），
固定种子可复现。

| 类别 | v1/v2 | v3 |
| --- | ---: | ---: |
| instruction | 4 | 32 |
| qa | 4 | 32 |
| format | 2 | 24 |
| multiturn | 6（轮次） | 24（轮次） |
| preference | 12 | 32 |
| verifiable.reward | 16 | 112 |

### 4.1 v3 结果（六个 checkpoint，同一套件同一协议）

| 指标 | n | Base | SFT | DPO 原 | DPO 新 | GRPO 原 | GRPO 新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| instruction.success | 32 | 0.0000 | 0.2500 | 0.2188 | 0.2500 | 0.2500 | 0.1562 |
| qa.success | 32 | 0.0000 | 0.3438 | 0.3125 | 0.3125 | **0.3750** | 0.3125 |
| format.success | 24 | 0.0000 | 0.2083 | **0.0000** | 0.1667 | 0.2500 | **0.2917** |
| multiturn.success | 24 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| verifiable.reward | 112 | 0.0000 | 0.2679 | 0.2054 | 0.2500 | **0.2857** | 0.2500 |
| preference.accuracy | 32 | 0.4375 | 0.5625 | 0.5938 | **0.6250** | 0.5625 | 0.5938 |
| corpus.bpb.all | 331 | 2.0353 | 2.1071 | 2.1019 | 2.1067 | 2.1079 | 2.1076 |

相对 SFT 的差（1 个 case 的步长：instruction/qa/preference 0.031，format 0.042，
verifiable 0.009）：

```text
instruction   DPO原 -0.031  DPO新  0.000  GRPO原  0.000  GRPO新 -0.094
qa            DPO原 -0.031  DPO新 -0.031  GRPO原 +0.031  GRPO新 -0.031
format        DPO原 -0.208  DPO新 -0.042  GRPO原 +0.042  GRPO新 +0.083
multiturn     全部 0.000
verifiable    DPO原 -0.063  DPO新 -0.018  GRPO原 +0.018  GRPO新 -0.018
preference    DPO原 +0.031  DPO新 +0.063  GRPO原  0.000  GRPO新 +0.031
```

### 4.2 v3 推翻了 v2 的一个结论

v2（n=12）显示 Base 0.833 高于 SFT 0.667，我曾据此说 "SFT 降低了偏好准确率"。
v3（n=32）显示 Base 0.4375 低于 SFT 0.5625，方向完全相反。
v2 那个数字是 12 个样本的假象，不应被引用。

### 4.3 现在能确认的事

1. **SFT 仍是唯一的大跃迁**：所有任务指标从 0 到 0.21~0.34，偏好从 0.4375 到 0.5625。
2. **原始 DPO 确实有害**：`format.success` 从 0.208 崩到 **0.000**（少 5 个 case），
   `verifiable.reward` 少 0.063（少 7 个 case）。
3. **on-policy + NLL 的修复有效**：format 恢复到 0.167，并取得**最高的
   `preference.accuracy` 0.6250**（比 SFT 多 0.063，即 2 个 case）。
   这是 DPO 第一次出现可测的正向收益。
4. **GRPO 两版互有胜负**：原版 verifiable/qa 略好，新版 format 最好（多 2 个 case）
   但 instruction 掉了 3 个 case。没有一版全面占优。
5. **`multiturn.success` 六个 checkpoint 完全相同（0.2500）**，该探针完全不敏感。
6. **语料 BPB 几乎不变**（正负 0.005），后训练没有进一步损害语言建模。

### 4.4 仍然不能确认的事

差异仍是 1~7 个 case 量级。即使 n=32，也不能据此宣称某个后训练阶段"更好"；
只能说方向与量级现在**可分辨**，足以指导下一步，不足以作为结论。

## 5. 下一步

1. **GRPO 控制 KL**：第二轮 KL 0.198 偏高，扫描 `kl_beta`，
   用 v3 的 verifiable/qa 观察权衡。
2. **DPO 扫描 `nll_coefficient`**：当前只试了 1.0，且它已给出最高偏好准确率，
   值得扩大搜索。
3. **替换失效探针**：`multiturn.success` 六个 checkpoint 恒为 0.25，需要更难的多轮题。
4. **训练侧的表层形式多样性**：在不把探针题目本身放进训练数据的前提下，
   为每个任务族增加多种措辞，提升对新表层形式的泛化。
5. 结构消融（深窄/浅宽 与 QK-Norm）成组 GPU 运行。
6. Qwen LoRA DPO；Qwen full 与 LoRA 同底座对照。

## 6. 证据

| 内容 | 位置 |
| --- | --- |
| 采样统计与难度分桶 | `data/onpolicy_sampling_stats.json` |
| on-policy 偏好对 / 可解题集 | `data/dpo-onpolicy-source.jsonl`、`data/grpo-solvable-source.jsonl` |
| 第二轮 DPO | `runs/native-dpo-onpolicy-001` |
| 第二轮 GRPO | `runs/native-grpo-solvable-001` |
| v3 探针套件 | `configs/evaluation/lifecycle-v3.yaml` |
| v3 成绩单 | `runs/evaluations/{base,sft,dpo-orig,dpo-onpolicy,grpo-orig,grpo-solvable}-v3` |

[项目首页](../README.md) | [GPU 后训练报告](./native-60m-lifecycle-gpu-v1.md)
