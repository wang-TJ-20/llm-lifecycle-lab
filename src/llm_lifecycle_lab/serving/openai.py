"""A strict, non-streaming subset of the OpenAI HTTP inference API."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from llm_lifecycle_lab.chat import generate_text
from llm_lifecycle_lab.exceptions import ContractError, LLMLabError
from llm_lifecycle_lab.model.protocol import GenerationConfig

_COMMON_FIELDS = {
    "model",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "stream",
    "n",
    "stop",
}
_QUANTIZED_INT_MAX = 2**63 - 1
_STATIC_ROOT = Path(__file__).with_name("static")
_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class OpenAIAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str = "invalid_request",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def _invalid(message: str, *, param: str | None = None) -> OpenAIAPIError:
    return OpenAIAPIError(400, message, param=param)


def _reject_unknown(payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise _invalid(
            f"unsupported request field: {unknown[0]}",
            param=unknown[0],
        )


def _require_model(payload: dict[str, Any], model_id: str) -> None:
    requested = payload.get("model")
    if not isinstance(requested, str) or not requested:
        raise _invalid("model must be a non-empty string", param="model")
    if requested != model_id:
        raise OpenAIAPIError(
            404,
            f"model not found: {requested}",
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )


def _number(
    payload: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    include_minimum: bool = True,
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{name} must be a number", param=name)
    result = float(value)
    lower_ok = result >= minimum if include_minimum else result > minimum
    if not math.isfinite(result) or not lower_ok or result > maximum:
        comparator = ">=" if include_minimum else ">"
        raise _invalid(
            f"{name} must be finite, {comparator} {minimum} and <= {maximum}",
            param=name,
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _invalid(f"{name} must be a positive integer", param=name)
    return value


def _generation_config(
    payload: dict[str, Any],
    *,
    default_max_tokens: int,
) -> GenerationConfig:
    if "max_tokens" in payload and "max_completion_tokens" in payload:
        raise _invalid(
            "provide only one of max_tokens or max_completion_tokens",
            param="max_tokens",
        )
    max_tokens = _positive_int(
        payload.get(
            "max_completion_tokens",
            payload.get("max_tokens", default_max_tokens),
        ),
        (
            "max_completion_tokens"
            if "max_completion_tokens" in payload
            else "max_tokens"
        ),
    )
    temperature = _number(
        payload,
        "temperature",
        0.0,
        minimum=0.0,
        maximum=2.0,
    )
    top_p = _number(
        payload,
        "top_p",
        1.0,
        minimum=0.0,
        maximum=1.0,
        include_minimum=False,
    )
    seed = payload.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= _QUANTIZED_INT_MAX:
        raise _invalid("seed must be a non-negative integer", param="seed")
    if payload.get("stream", False) is not False:
        raise _invalid("streaming is not supported", param="stream")
    if payload.get("n", 1) != 1 or type(payload.get("n", 1)) is not int:
        raise _invalid("only n=1 is supported", param="n")
    if payload.get("stop") is not None:
        raise _invalid("custom stop sequences are not supported", param="stop")
    return GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        seed=seed,
    )


def _finish_reason(stop_reason: str) -> str:
    return "stop" if stop_reason == "eos" else "length"


def _validate_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise _invalid("messages must be a non-empty array", param="messages")
    messages = []
    expected = "user"
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise _invalid(
                f"messages[{index}] must be an object",
                param="messages",
            )
        unknown = set(message) - {"role", "content"}
        if unknown:
            raise _invalid(
                f"unsupported message field: {sorted(unknown)[0]}",
                param="messages",
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _invalid(
                f"messages[{index}].content must be a non-empty string",
                param="messages",
            )
        if index == 0 and role == "system":
            messages.append({"role": role, "content": content})
            continue
        if role != expected:
            raise _invalid(
                f"messages[{index}].role must be {expected}",
                param="messages",
            )
        messages.append({"role": role, "content": content})
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "user":
        raise _invalid("the final message must have role=user", param="messages")
    return messages


class OpenAIService:
    def __init__(
        self,
        evaluator,
        *,
        model_id: str,
        default_max_tokens: int = 48,
        max_request_bytes: int = 1024 * 1024,
    ) -> None:
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or len(model_id) > 256
            or any(ord(character) < 32 for character in model_id)
        ):
            raise ContractError("model_id must be printable text of at most 256 chars")
        if type(default_max_tokens) is not int or default_max_tokens <= 0:
            raise ContractError("default_max_tokens must be a positive integer")
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ContractError("max_request_bytes must be a positive integer")
        self.default_max_tokens = default_max_tokens
        self.max_request_bytes = max_request_bytes
        self.evaluator = evaluator
        self.model_id = model_id
        self.created = int(time.time())
        self._generation_lock = threading.Lock()

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": self.created,
                    "owned_by": "llm-lifecycle-lab",
                }
            ],
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_id,
            "stage": self.evaluator.identity["stage"],
        }

    def create_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown(payload, _COMMON_FIELDS | {"prompt"})
        _require_model(payload, self.model_id)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise _invalid("prompt must be a non-empty string", param="prompt")
        generation = _generation_config(
            payload,
            default_max_tokens=self.default_max_tokens,
        )
        ids = self.evaluator.tokenizer.encode(prompt, add_bos=True)
        with self._generation_lock:
            result = generate_text(
                self.evaluator,
                ids,
                generation=generation,
                chat=False,
            )
        return self._completion_response(result, chat=False)

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        _reject_unknown(payload, _COMMON_FIELDS | {"messages"})
        _require_model(payload, self.model_id)
        messages = _validate_messages(payload.get("messages"))
        generation = _generation_config(
            payload,
            default_max_tokens=self.default_max_tokens,
        )
        ids = self.evaluator.prompt_ids(messages, "native-chat-v1")
        with self._generation_lock:
            result = generate_text(
                self.evaluator,
                ids,
                generation=generation,
                chat=True,
            )
        return self._completion_response(result, chat=True)

    def _completion_response(
        self,
        result: dict[str, Any],
        *,
        chat: bool,
    ) -> dict[str, Any]:
        choice: dict[str, Any] = {
            "index": 0,
            "finish_reason": _finish_reason(result["stop_reason"]),
        }
        if chat:
            choice["message"] = {
                "role": "assistant",
                "content": result["text"],
            }
        else:
            choice["text"] = result["text"]
            choice["logprobs"] = None
        return {
            "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["generated_tokens"],
                "total_tokens": (
                    result["prompt_tokens"] + result["generated_tokens"]
                ),
            },
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


class _OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    service: OpenAIService

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in _STATIC_ROUTES:
            filename, content_type = _STATIC_ROUTES[path]
            try:
                body = (_STATIC_ROOT / filename).read_bytes()
            except OSError as exc:
                self.log_error("cannot load static asset %s: %s", filename, exc)
                self._write_error(
                    OpenAIAPIError(
                        500,
                        "static interface is unavailable",
                        error_type="server_error",
                        code="internal_error",
                    )
                )
            else:
                self._write_bytes(200, body, content_type=content_type)
        elif path == "/health":
            self._write_json(200, self.service.health())
        elif path == "/v1/models":
            self._write_json(200, self.service.list_models())
        else:
            self._write_error(
                OpenAIAPIError(
                    404,
                    f"route not found: {path}",
                    error_type="invalid_request_error",
                    code="route_not_found",
                )
            )

    def do_POST(self) -> None:
        request_id = f"req-{uuid.uuid4().hex}"
        try:
            payload = self._read_json()
            path = urlsplit(self.path).path
            if path == "/v1/completions":
                result = self.service.create_completion(payload)
            elif path == "/v1/chat/completions":
                result = self.service.create_chat_completion(payload)
            else:
                raise OpenAIAPIError(
                    404,
                    f"route not found: {path}",
                    error_type="invalid_request_error",
                    code="route_not_found",
                )
        except OpenAIAPIError as exc:
            self._write_error(exc, request_id=request_id)
        except LLMLabError as exc:
            self._write_error(
                _invalid(str(exc)),
                request_id=request_id,
            )
        except Exception as exc:
            self.log_error("request failed: %s", exc)
            self._write_error(
                OpenAIAPIError(
                    500,
                    "internal server error",
                    error_type="server_error",
                    code="internal_error",
                ),
                request_id=request_id,
            )
        else:
            self._write_json(200, result, request_id=request_id)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise OpenAIAPIError(
                415,
                "Content-Type must be application/json",
                code="unsupported_media_type",
            )
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise _invalid("Transfer-Encoding is not supported")
        length_value = self.headers.get("Content-Length")
        if length_value is None:
            raise OpenAIAPIError(
                411,
                "Content-Length is required",
                code="length_required",
            )
        try:
            length = int(length_value)
        except ValueError as exc:
            raise _invalid("invalid Content-Length") from exc
        if length <= 0:
            raise _invalid("request body must not be empty")
        if length > self.service.max_request_bytes:
            self.close_connection = True
            raise OpenAIAPIError(
                413,
                "request body is too large",
                code="request_too_large",
            )
        raw = self.rfile.read(length)
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _invalid("request body must be valid strict JSON") from exc
        if not isinstance(payload, dict):
            raise _invalid("request body must be a JSON object")
        return payload

    def _write_error(
        self,
        error: OpenAIAPIError,
        *,
        request_id: str | None = None,
    ) -> None:
        self._write_json(
            error.status_code,
            error.payload(),
            request_id=request_id,
        )

    def _write_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._write_bytes(
            status_code,
            body,
            content_type="application/json; charset=utf-8",
            request_id=request_id,
        )

    def _write_bytes(
        self,
        status_code: int,
        body: bytes,
        *,
        content_type: str,
        request_id: str | None = None,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)


def create_openai_server(
    host: str,
    port: int,
    service: OpenAIService,
) -> ThreadingHTTPServer:
    if not isinstance(host, str) or not host:
        raise ContractError("host must be a non-empty string")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ContractError("port must be between 0 and 65535")

    class Handler(_OpenAIHandler):
        pass

    Handler.service = service
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
