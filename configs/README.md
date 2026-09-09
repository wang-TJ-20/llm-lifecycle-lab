# 配置说明

目录职责：

- `models/`：Native 模型尺寸与架构参数。
- `pipelines/`：完整、可恢复的 Pretrain 运行配置。

每个 run 都会保存完整的 resolved config。

当前配置：

- `native-smoke.yaml`：Native 10M 两步 CPU/MPS Smoke。
- `native-v1.yaml`：Native 60M CUDA Pretrain。
- `native-60m-reference.yaml`：更严格评测覆盖的 60M 发布配方。
- `native-60m-*.yaml`：QK-Norm 与深窄结构的受控消融。

不启动训练即可校验配置：

```bash
uv run llmlab config validate configs/pipelines/native-v1.yaml
```

Native Pretrain 配置必须声明：

- `model.config`、`model.tokenizer`
- `data.manifest`、`data.packed_manifest`
- 完整的 `training` 配置
- `max_steps`、`max_train_tokens`、`num_epochs` 三种预算之一
