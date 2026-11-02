# 单卡 A10 从零重跑手册

当前硬件实践主线为 A10：重新物化双语数据、训练 Tokenizer、从随机权重预训练
Native-60M，再冻结新的 SFT 输入。CPU 实验用于理解机制。

## 固定边界

- 模型：`configs/models/tiny-60m.yaml`，62,927,616 参数，词表 16,384。
- 环境：Python 3.11、CUDA、BF16，依赖见 `requirements-a10.txt`。
- 数据与 Tokenizer：`bilingual-60m-a10-v1`，独立目录、train-only BPE。
- 长度：512；micro-batch 1、梯度累积 16。
- smoke、pilot、正式 Base 使用独立 run ID，均随机初始化，不继承旧权重。
- 新 Base 验收后建立独立 SFT 门禁，不能替换旧门禁的 hash 来放行。
- 正式 Base 需要干净提交；未提交的 smoke/pilot 仅为开发验证。

## 环境

从仓库根目录执行。标准 Python 安装包含 venv/ensurepip 时：

```bash
python3 -m venv .venv-a10
.venv-a10/bin/python -m pip install "torch==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu126
.venv-a10/bin/python -m pip install -r requirements-a10.txt
```

当前开发机缺少 ensurepip，系统 pip 支持指定目标 Python，可改用：

```bash
python3 -m venv --without-pip .venv-a10
python3 -m pip --python .venv-a10 install pip==25.2
python3 -m pip --python .venv-a10 install "torch==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu126
.venv-a10/bin/python -m pip install -r requirements-a10.txt
```

不继承系统 site-packages。已有环境先检查，不删除重建。
验证 pip check、CUDA/BF16、pytest、ruff，确认实际 wheel 为 CUDA 12.6。
`requirements-a10.txt` 独立于历史 uv.lock；不能用历史锁环境冒充 A10 实测。

## 数据

继续使用冻结 SimpleStories/Wikipedia recipe，先核对 MIT/CC-BY-SA-3.0 许可。
按 [数据指南](./DATA_GUIDE.md#5-切换为-60m-双语数据)重新 fetch、mix、prepare、
train_tokenizer 和 pack，路径替换规则：

- raw 来源和混合文件放在 `data/raw/a10-v1/`，recipe ID 保持不变。
- prepared 放在 `data/prepared/bilingual-60m-a10-v1/`，dataset ID 同名。
- Tokenizer 放在 `data/tokenizers/bilingual-60m-a10-v1/`，tokenizer ID 同名。
- Packing 放在 `data/packed/bilingual-60m-a10-v1-seq512/`。

按 source_id 分组切分，只用 train 训练 BPE，词表 16,384、min-frequency 2。
下载可以复用校验缓存，但文本物化、切分、BPE、Packing 必须本轮重做。
混合文本预期共 226,000 条，内容 hash：
`af07e4c3a610637239a3295fd4754af316bd74c5d08bc25b63fd04a147339589`。
新 manifest 的时间/ID 不同，不能要求文件 hash 等于历史 manifest。
禁止覆盖已有目录；保留失败产物后再诊断。

## 同卡 smoke 与 pilot

固定 BF16、seq512、micro-batch 1、累积 16，两次均独立随机初始化。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv-a10/bin/python scripts/doctor.py \
  --config configs/pipelines/native-60m-a10-base-smoke.yaml
.venv-a10/bin/python -u scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-smoke.yaml \
  --run-id native-60m-a10-base-smoke-s42
.venv-a10/bin/python scripts/verify_a10_base.py --phase smoke \
  --run runs/native-60m-a10-base-smoke-s42
.venv-a10/bin/python -u scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-pilot100.yaml \
  --run-id native-60m-a10-base-pilot100-s42
.venv-a10/bin/python scripts/verify_a10_base.py --phase pilot \
  --run runs/native-60m-a10-base-pilot100-s42
```

门禁核对单卡 A10、配置、完整预算、有限指标和 checkpoint。
Pilot 要求中英文 dev 均改善，final 相对 best 回退不超过 2%。
报告原子写入 `a10-development-gate.json`，不读取 test，也不是 SFT 门禁。
不能根据结果放宽条件。Doctor 单步不替代真实精度/累积测试；
step 吞吐不包含全部评测和保存时间。

## 正式 Base 与后训练

正式前还要受控中断恢复验证、干净提交和完整依赖/数据检查。
不回退覆盖已完成 run，不改变预算后复用 run ID，不承诺跨硬件逐位一致。
正式 Doctor 显式使用 `tiny-60m-a10`：要求单张 NVIDIA A10、CUDA BF16
以及至少 21.9 GiB CUDA 可见容量。本卡驱动保留内存后为 21.988 GiB，
低于历史通用 profile 的 22 GiB；通用与历史门槛保持不变。

```bash
.venv-a10/bin/python scripts/a10_resume_experiment.py \
  --baseline-run runs/native-60m-a10-base-smoke-s42 \
  --run-id native-60m-a10-base-resume-s42
```

该实验在独立 run 的第 5 步受控中断，保留第 4 步完整 checkpoint，
恢复到原定第 8 步，再比较模型、优化器/调度器/RNG、trainer 与数据流状态。
结果原子写入 `a10-resume-comparison.json`，相同环境不一致时仍保留失败证据；
不要把一次通过扩大为跨设备或所有配置的保证。

普通 CUDA 算法的独立 run 即使同 seed 也可能在中断前产生差异。
本轮严格对照另外启用了 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和
`torch.use_deterministic_algorithms(True)`，不放宽逐位比较容差。
下面在新 run ID 上重新建立确定性 baseline，再做恢复：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-a10/bin/python - <<'PY'
import sys
import torch
from pathlib import Path

sys.path.insert(0, "src")
from llm_lifecycle_lab.config import load_run_config
from llm_lifecycle_lab.training.pretrain import run_native_pretraining

torch.use_deterministic_algorithms(True)
config = load_run_config("configs/pipelines/native-60m-a10-base-smoke.yaml")
run_native_pretraining(
    config,
    run_id="native-60m-a10-base-deterministic-smoke-s42",
    workdir=Path.cwd(),
)
PY
.venv-a10/bin/python - <<'PY'
import sys
import torch

sys.path.insert(0, "scripts")
from a10_resume_experiment import main

torch.use_deterministic_algorithms(True)
raise SystemExit(main([
    "--baseline-run", "runs/native-60m-a10-base-deterministic-smoke-s42",
    "--run-id", "native-60m-a10-base-deterministic-resume-s42",
]))
PY
unset CUBLAS_WORKSPACE_CONFIG
```

已有同名 run 时换新 ID，不删除或覆盖。设置前后属于不同执行条件，
不能将普通 CUDA baseline 与确定性恢复直接做逐位比较；
正式训练要先明确是否采用确定性算法并记录，而不是中途切换。

```bash
test -z "$(git status --porcelain=v1)"
.venv-a10/bin/python scripts/doctor.py \
  --profile tiny-60m-a10 \
  --config configs/pipelines/native-60m-a10-base-v1.yaml
.venv-a10/bin/python -u scripts/train_pretrain.py \
  --config configs/pipelines/native-60m-a10-base-v1.yaml \
  --run-id native-60m-a10-base-v1-s42
.venv-a10/bin/python scripts/verify_a10_base.py --phase base \
  --run runs/native-60m-a10-base-v1-s42
```

正式预算为 8 epochs，实际 token/step 从本轮 Packing 换算，不照抄历史值。
开发门禁要求覆盖率 [1.0, 1.01]、总体 dev loss 减半、双语改善、
final 相对 best 回退不超过 2%。通过后才用 eval_pretrain 对最终 checkpoint
执行一次 test；step 取 training_result，不照抄历史 checkpoint 路径。
评测仍受 eval_batches=1024 限制，不称作全量 test。

归档环境、源码、配置、数据身份、预算、指标、checkpoint、门禁和 dev/test。
新 Base 验收后冻结独立 A10 SFT Base/data/run 门禁，重新编码公开 SFT 数据、
统计 assistant token 和长度，再做 smoke/pilot、恢复、正式 SFT 和同协议能力对照。

当前实测进度见 [A10 执行记录](./A10_TRAINING_RESULTS.md)。
