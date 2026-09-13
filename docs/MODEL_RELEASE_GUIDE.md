# Native 模型发布指南

本文说明如何把已经验收的 Native checkpoint 构建为可审计的 ModelScope
公开模型，并在全新目录中完成回下载验证。

当前目标仓库为：

```text
llmlifecyclelab/native-60m-base-v1
```

ModelScope 组织需要在网页提交申请并等待审核。组织可用前，可以完整构建和验证
本地发布包，但不要临时上传到另一个 namespace 后再覆盖历史。

## 1. 发布边界

公开包包含：

- 最终 `step-00005649` 模型权重和模型配置；
- 与 checkpoint 严格绑定的双语 Tokenizer；
- checkpoint、训练预算、运行环境和 dev/test 评测证据；
- Model Card、结构化报告、固定续写和双语 Smoke 评测；
- `release_manifest.json` 与 `SHA256SUMS`。

公开包不包含原始训练数据、中间 checkpoint、optimizer、scheduler 或 RNG 状态。
完整训练恢复仍依赖训练服务器上的原 run，模型发布包只面向推理和后续阶段初始化。

## 2. 构建发布包

从仓库根目录运行：

```bash
python scripts/build_model_release.py \
  --checkpoint-archive step-00005649-model.tar.gz \
  --repository-id llmlifecyclelab/native-60m-base-v1
```

默认输出：

```text
build/modelscope/native-60m-base-v1/
```

构建器拒绝覆盖已有目录，也会拒绝路径穿越、额外文件或缺失文件的 checkpoint
归档。构建过程中会实际加载权重、核对 Tokenizer hash、运行固定中英文续写和
双语 Smoke 评测，然后才写入发布清单。

## 3. Provenance 声明

该权重来自已接受的 `native-60m-baseline-v1`。训练运行通过冻结输入、预算、
评测、checkpoint、环境和源码内容检查，但启动时 Git 工作区包含 5 个未提交改动，
因此状态保持为：

```text
accepted-with-provenance-waiver
```

不要修改历史环境记录或把它改写成 clean。源码内容仍由
`source_sha256=142361e52641c2dcd5ea7de3f55030bd3e2434a5132fb39abd1303c2bc597729`
固定。

## 4. 创建 ModelScope 资源

1. 登录 [ModelScope 国内站](https://www.modelscope.cn/)。
2. 在[组织申请页](https://www.modelscope.cn/organization/create)申请英文名
   `llmlifecyclelab` 的组织。平台只允许英文字母和数字，创建后不可修改。
3. 审核通过后，在该组织下创建公开模型 `native-60m-base-v1`。
4. 许可选择 Apache-2.0，任务选择文本生成，框架选择 PyTorch。

账号登录、验证码、手机号和组织申请确认必须由账号持有人完成。

## 5. 上传

安装 ModelScope 发布依赖并完成本机登录：

```bash
python -m pip install -r requirements-modelscope.txt
ms login
```

组织和空模型仓库创建完成后，优先按照模型仓库页面给出的 Git 或 `ms upload`
命令上传 `build/modelscope/native-60m-base-v1/` 的全部内容。上传前后分别执行：

```bash
python scripts/run_published_model.py \
  --use-local \
  --local-dir build/modelscope/native-60m-base-v1
```

## 6. 回下载验收

删除或换用新的下载目录，避免命中构建目录：

```bash
python scripts/run_published_model.py \
  --model-id llmlifecyclelab/native-60m-base-v1 \
  --local-dir build/downloads/native-60m-base-v1
```

命令会依次完成下载、全部文件 SHA-256 校验、固定提示续写和内置双语
Smoke 评测。只有回下载产物通过检查后，README 才应把模型标记为已发布。

## 7. 版本策略

- 当前权重固定为 `v1.0.0`。
- ModelScope `master` 指向当前正式版本。
- 未来 clean provenance 重跑使用新版本，不覆盖 v1 的权重或历史证据。
- SFT、DPO 等阶段使用独立模型仓库，不替换 Base 模型。

[返回项目首页](../README.md)
