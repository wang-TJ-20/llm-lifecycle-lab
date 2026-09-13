# 最小 OpenAI-compatible API

项目提供一个只依赖 Python 标准库的本地 HTTP 服务，用同一接口加载 Native 或
HF 模型。它用于打通“模型产物 → HTTP 请求 → token 用量与停止原因”链路，
不是生产级推理网关。

## 1. 启动

Native 发布包：

```bash
python scripts/serve_openai.py \
  --checkpoint build/modelscope/native-60m-base-v1 \
  --model-id native-60m-base-v1 \
  --host 127.0.0.1 \
  --port 8000
```

HF 导出：

```bash
python scripts/serve_openai.py \
  --backend hf \
  --checkpoint build/hf/native-60m-base-v1 \
  --model-id native-60m-base-v1 \
  --host 127.0.0.1 \
  --port 8000
```

HF adapter checkpoint 还需要 `--base-model`。默认只监听 loopback，不提供认证；
不要直接绑定公网地址。模型在绑定端口之前完成加载和 hash 校验。

启动后直接打开：

```text
http://127.0.0.1:8000/
```

本地界面提供 Chat/Completion 分段切换、system prompt、max tokens、
temperature、top-p、seed、会话重置和 token usage。Base 阶段自动选择
Completion，SFT/DPO/GRPO 阶段自动选择 Chat；用户仍可显式切换。
切换模式会清空旧会话，避免把两种协议的历史混在同一次请求中。

前端由同一个 Python 进程提供，不需要 npm、前端构建或另起静态服务器。
模型输出只按文本渲染，不执行其中的 HTML 或脚本。

## 2. HTTP 接口

服务实现：

- `GET /`
- `GET /app.css`
- `GET /app.js`
- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

文本续写：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "native-60m-base-v1",
    "prompt": "Once upon a time",
    "max_tokens": 16,
    "temperature": 0
  }'
```

对话：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "native-60m-base-v1",
    "messages": [
      {"role": "system", "content": "回答要简短。"},
      {"role": "user", "content": "你好"}
    ],
    "max_tokens": 16,
    "temperature": 0
  }'
```

Base checkpoint 可以经过 chat template 生成，但这不代表模型已经具备指令能力。
Base 的默认使用方式仍是 `/v1/completions`。

## 3. 支持参数

两个生成接口共用：

| 参数 | 支持范围 |
| --- | --- |
| `model` | 必填，必须与启动时的 `--model-id` 完全相同 |
| `max_tokens` | 正整数；也接受 `max_completion_tokens`，二者不能同时提供 |
| `temperature` | `0..2`；`0` 表示 greedy，默认 `0` |
| `top_p` | `(0, 1]` |
| `seed` | 非负整数，默认 `42` |
| `stream` | 只接受 `false` |
| `n` | 只接受 `1` |
| `stop` | 只接受 `null` 或省略 |

`/v1/completions` 只接受单个字符串 `prompt`。`/v1/chat/completions`
只接受可选首个 system 后严格交替的 user/assistant 消息，并且最后一条必须是
user。工具调用、图片、音频、logprobs、自定义 stop 和批量 prompt 均不支持；
发送未知字段会返回 400，不会静默忽略。

## 4. OpenAI 客户端

项目不依赖 OpenAI SDK。环境中已经安装该 SDK 时，可以把它指向本地服务：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-not-checked",
)
response = client.completions.create(
    model="native-60m-base-v1",
    prompt="Once upon a time",
    max_tokens=16,
    temperature=0,
)
print(response.choices[0].text)
```

服务不读取 API key。参数必须落在上一节的最小子集内。

## 5. 错误与资源边界

- 请求体必须是 `application/json`、UTF-8、无重复 key、无 NaN/Infinity。
- 默认请求体上限为 1 MiB，可用 `--max-request-bytes` 调整。
- prompt 加生成预算超过模型上下文时返回 400，不静默截断。
- 模型 ID 不匹配返回 OpenAI 风格 `model_not_found` 404。
- 每个错误响应包含 `error.message/type/param/code` 和 `X-Request-ID`。
- HTTP server 可同时接收请求，但单模型生成使用进程内锁串行执行，避免 RNG
  和模型状态并发互相影响。

## 6. 非目标

当前没有流式 SSE、continuous batching、KV Cache 跨请求复用、认证、TLS、
限流、指标采集、请求取消或多 worker。也没有把 dynamic INT8 接入服务加载路径。
需要生产部署时应选择成熟推理引擎，并重新验证 Tokenizer、chat template、
采样参数和固定输出，不应把本服务直接暴露到公网。

[返回项目路线](./test.md)
