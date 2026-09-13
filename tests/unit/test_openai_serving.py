from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import torch

from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.model.protocol import GenerationOutput
from llm_lifecycle_lab.serving.openai import (
    OpenAIAPIError,
    OpenAIService,
    create_openai_server,
)


class FakeTokenizer:
    eos_token_id = 2
    chat_end_token_id = 3
    pad_token_id = 0

    def encode(self, text: str, *, add_bos: bool) -> list[int]:
        assert text
        return ([1] if add_bos else []) + [4]

    def decode(self, ids, *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return "generated"


class FakeModel:
    def __init__(self) -> None:
        self.last_config = None

    def generate(self, *, input_ids, attention_mask, config) -> GenerationOutput:
        assert attention_mask is None
        self.last_config = config
        suffix = torch.tensor([[7, config.eos_token_id]])
        return GenerationOutput(
            token_ids=torch.cat((input_ids.cpu(), suffix), dim=1),
            prompt_tokens=input_ids.shape[1],
            generated_tokens=2,
            stop_reason="eos",
        )


class FakeEvaluator:
    def __init__(self) -> None:
        self.identity = {"stage": "sft"}
        self.config = SimpleNamespace(max_sequence_length=32)
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.device = torch.device("cpu")
        self.last_messages = None

    def prompt_ids(self, messages, protocol: str) -> list[int]:
        assert protocol == "native-chat-v1"
        self.last_messages = messages
        return [1, 5, 6]


def make_service(**kwargs) -> OpenAIService:
    return OpenAIService(
        FakeEvaluator(),
        model_id="fixture",
        default_max_tokens=4,
        **kwargs,
    )


def test_completion_and_chat_responses_follow_supported_schema() -> None:
    service = make_service()
    completion = service.create_completion(
        {
            "model": "fixture",
            "prompt": "Once",
            "max_tokens": 2,
            "temperature": 0,
        }
    )
    assert completion["object"] == "text_completion"
    assert completion["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "generated",
            "logprobs": None,
        }
    ]
    assert completion["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert not service.evaluator.model.last_config.do_sample

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    chat = service.create_chat_completion(
        {
            "model": "fixture",
            "messages": messages,
            "max_completion_tokens": 3,
            "temperature": 0.5,
            "top_p": 0.8,
            "seed": 7,
        }
    )
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"] == {
        "role": "assistant",
        "content": "generated",
    }
    assert service.evaluator.last_messages == messages
    config = service.evaluator.model.last_config
    assert config.do_sample
    assert (config.max_new_tokens, config.temperature, config.top_p, config.seed) == (
        3,
        0.5,
        0.8,
        7,
    )


@pytest.mark.parametrize(
    ("payload", "status", "message"),
    [
        (
            {"model": "missing", "prompt": "x"},
            404,
            "model not found",
        ),
        (
            {"model": "fixture", "prompt": "x", "stream": True},
            400,
            "streaming is not supported",
        ),
        (
            {"model": "fixture", "prompt": "x", "n": 2},
            400,
            "only n=1",
        ),
        (
            {"model": "fixture", "prompt": "x", "unknown": 1},
            400,
            "unsupported request field",
        ),
    ],
)
def test_completion_rejects_unsupported_requests(
    payload: dict, status: int, message: str
) -> None:
    with pytest.raises(OpenAIAPIError, match=message) as error:
        make_service().create_completion(payload)
    assert error.value.status_code == status


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "wrong start"}],
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ],
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    ],
)
def test_chat_requires_alternating_messages_ending_in_user(messages: list) -> None:
    with pytest.raises(OpenAIAPIError):
        make_service().create_chat_completion(
            {"model": "fixture", "messages": messages}
        )


def test_service_configuration_and_context_fail_fast() -> None:
    with pytest.raises(ContractError, match="model_id"):
        OpenAIService(FakeEvaluator(), model_id="")
    with pytest.raises(ContractError, match="default_max_tokens"):
        OpenAIService(FakeEvaluator(), model_id="fixture", default_max_tokens=0)
    service = make_service()
    with pytest.raises(ContractError, match="context is full"):
        service.create_completion(
            {"model": "fixture", "prompt": "x", "max_tokens": 31}
        )


def request_json(
    url: str,
    *,
    payload: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_server_returns_openai_json_and_strict_errors() -> None:
    service = make_service(max_request_bytes=1024)
    server = create_openai_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base, timeout=5) as response:
            html = response.read().decode()
            assert response.headers["Content-Type"].startswith("text/html")
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            assert "LLM Lifecycle Lab" in html
        with urllib.request.urlopen(f"{base}/app.css", timeout=5) as response:
            assert response.headers["Content-Type"].startswith("text/css")
            assert b".app-shell" in response.read()
        with urllib.request.urlopen(f"{base}/app.js", timeout=5) as response:
            assert response.headers["Content-Type"].startswith("text/javascript")
            assert b"requestCompletion" in response.read()

        status, models = request_json(f"{base}/v1/models")
        assert status == 200
        assert models["data"][0]["id"] == "fixture"

        status, completion = request_json(
            f"{base}/v1/completions",
            payload=json.dumps(
                {"model": "fixture", "prompt": "Once", "max_tokens": 2}
            ).encode(),
        )
        assert status == 200
        assert completion["choices"][0]["text"] == "generated"

        status, error = request_json(
            f"{base}/v1/completions",
            payload=b'{"model":"fixture","model":"duplicate","prompt":"x"}',
        )
        assert status == 400
        assert error["error"]["code"] == "invalid_request"

        status, error = request_json(
            f"{base}/v1/completions",
            payload=b"{}",
            content_type="text/plain",
        )
        assert status == 415
        assert error["error"]["code"] == "unsupported_media_type"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
