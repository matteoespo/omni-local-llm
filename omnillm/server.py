import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from omnillm import __version__
from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidImageError,
    InvalidRequestError,
    ModelNotFoundError,
    UnsupportedFeatureError,
)
from omnillm.core.manager import LocalLLMManager
from omnillm.core.media import sanitize_untrusted_messages
from omnillm.core.types import ChatChunk, ChatResponse, Usage


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    images: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    n: int = Field(default=1, ge=1)
    # Non-OpenAI extensions (send via `extra_body` in the OpenAI SDK) needed to address llama.cpp GGUF models.
    filename: str | None = None
    mmproj_filename: str | None = None


def _resolve_tools(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    """Applies `tool_choice`. Local backends can't force a call, so only "auto" and "none" are honoured."""
    choice = request.tool_choice
    if choice is None or choice == "auto":
        return request.tools
    if choice == "none":
        return None
    raise UnsupportedFeatureError("Only tool_choice 'auto' and 'none' are supported.", feature="tool_choice")


class EmbeddingInputRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


logger = logging.getLogger("omnillm.server")

#: Hosts accepted by default: loopback names only, which blocks DNS-rebinding attacks from web pages.
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
ALLOWED_HOSTS_ENV = "OMNILLM_ALLOWED_HOSTS"

_FEATURE_MESSAGES: dict[str, str] = {
    "streaming": "The backend for this model does not support streaming.",
    "json_mode": "The backend for this model does not support JSON mode.",
    "structured_outputs": "The backend for this model does not support structured outputs (json_schema).",
    "tools": "The backend for this model does not support tool calling.",
    "vision": "The backend for this model does not support image inputs.",
    "embeddings": "The backend for this model does not support embeddings.",
    "tool_choice": "Only tool_choice 'auto' and 'none' are supported by local backends.",
}
_IMAGE_MESSAGE = (
    "Images must be inline base64 or data:image/<type>;base64,<data> URIs. "
    "Local file paths and remote URLs are not accepted."
)


def _client_error(message: str, *, status_code: int = 400, code: str = "invalid_request") -> JSONResponse:
    """An error whose message is written by the server itself (never derived from an exception)."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}},
    )


def _one_line(text: str) -> str:
    """Strips line breaks so request-derived text can't forge extra log lines."""
    return text.replace("\r", " ").replace("\n", " ")


def _error_response(error: Exception) -> JSONResponse:
    """Maps an exception to an OpenAI-style error with a fixed, server-written message.

    Exception text can contain upstream backend responses or internal details, so it is only logged
    (with a request id that is also returned to the client), never sent.
    """
    request_id = f"req_{uuid4().hex[:24]}"
    status_code, error_type, code = 500, "server_error", "internal_error"
    message = "An internal error has occurred."
    if isinstance(error, InvalidImageError):
        status_code, error_type, code, message = 400, "invalid_request_error", "invalid_image", _IMAGE_MESSAGE
    elif isinstance(error, ModelNotFoundError):
        status_code, error_type, code = 404, "invalid_request_error", "model_not_found"
        message = "The requested model was not found on its backend. GET /v1/models lists available models."
    elif isinstance(error, BackendNotFoundError):
        status_code, error_type, code = 404, "invalid_request_error", "model_not_found"
        message = "Unknown model or backend. GET /v1/models lists available models."
    elif isinstance(error, BackendUnavailableError):
        status_code, error_type, code = 503, "server_error", "backend_unavailable"
        message = "The requested backend is currently unavailable."
    elif isinstance(error, UnsupportedFeatureError):
        status_code, error_type, code = 400, "invalid_request_error", "unsupported_feature"
        message = "The backend for this model does not support a feature used by this request."
        for feature, text in _FEATURE_MESSAGES.items():
            if error.feature == feature:
                message = text
    elif isinstance(error, InvalidRequestError):
        status_code, error_type, code = 400, "invalid_request_error", "invalid_request"
        message = "The backend rejected the request (for example, the prompt may exceed the context window)."
    elif isinstance(error, ValueError):
        status_code, error_type, code = 400, "invalid_request_error", "invalid_request"
        message = "Invalid request parameters."

    if status_code >= 500 and code == "internal_error":
        logger.error("Request %s failed", request_id, exc_info=error)
    else:
        logger.warning("Request %s: %s: %s", request_id, type(error).__name__, _one_line(str(error)))
    return JSONResponse(
        status_code=status_code,
        headers={"x-request-id": request_id},
        content={
            "error": {
                "message": f"{message} (request id: {request_id})",
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def host_name(host_header: str) -> str:
    """Hostname part of a Host header: 'example.com:8000' -> 'example.com', '[::1]:8000' -> '::1'."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1 : host.index("]")] if "]" in host else host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host  # bare name, IPv4, or unbracketed IPv6


class HostCheckMiddleware:
    """Rejects requests whose Host header isn't allowed (supports IPv6 and '*.domain' wildcards)."""

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str]):
        self.app = app
        self.allowed = [host_name(h) if not h.startswith("*.") else h.lower() for h in allowed_hosts]

    def _is_allowed(self, name: str) -> bool:
        for pattern in self.allowed:
            if pattern == name or (pattern.startswith("*.") and name.endswith(pattern[1:])):
                return True
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and not self._is_allowed(
            host_name(Headers(scope=scope).get("host", ""))
        ):
            await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _allowed_hosts(allowed_hosts: Sequence[str] | None) -> list[str]:
    if allowed_hosts is not None:
        return list(allowed_hosts)
    configured = os.environ.get(ALLOWED_HOSTS_ENV)
    if configured:
        return [host.strip() for host in configured.split(",") if host.strip()]
    return list(LOOPBACK_HOSTS)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def create_app(manager: LocalLLMManager | None = None, *, allowed_hosts: Sequence[str] | None = None) -> FastAPI:
    """Builds the OpenAI-compatible API.

    `allowed_hosts` limits accepted ``Host`` headers (default: ``$OMNILLM_ALLOWED_HOSTS`` or loopback names),
    which stops malicious web pages from reaching a local server through DNS rebinding. Pass ``["*"]`` to
    accept any host, e.g. behind a reverse proxy that already checks it.
    """
    app = FastAPI(title="Omni-Local-LLM OpenAI API", version=__version__)
    hosts = _allowed_hosts(allowed_hosts)
    if "*" not in hosts:
        app.add_middleware(HostCheckMiddleware, allowed_hosts=hosts)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    llm_manager = manager if manager is not None else LocalLLMManager.from_config()

    @app.get("/v1/models", response_model=None)
    async def list_models() -> JSONResponse | dict[str, Any]:
        try:
            # Backends may be remote HTTP servers: query them off the event loop.
            backend_models = await asyncio.to_thread(llm_manager.list_models)
            aliases = llm_manager.aliases
        except Exception as error:
            return _error_response(error)
        models = [
            {"id": name, "object": "model", "created": 0, "owned_by": f"alias:{target.ref}"}
            for name, target in aliases.items()
        ]
        models.extend(
            {"id": f"{backend}/{model}", "object": "model", "created": 0, "owned_by": backend}
            for backend, model in backend_models
        )
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse | dict[str, Any]:
        if request.n != 1:
            return _client_error("Only n=1 is supported.")
        try:
            messages = sanitize_untrusted_messages(
                [message.model_dump(exclude_none=True) for message in request.messages]
            )
        except InvalidImageError:
            return _client_error(_IMAGE_MESSAGE, code="invalid_image")
        try:
            tools = _resolve_tools(request)
        except UnsupportedFeatureError:
            return _client_error(_FEATURE_MESSAGES["tool_choice"], code="unsupported_feature")

        try:
            response_format_dict: dict[str, Any] | None = None
            json_mode = False
            if request.response_format is not None and request.response_format.type != "text":
                json_mode = True
                response_format_dict = request.response_format.model_dump(by_alias=True, exclude_none=True)

            response = await llm_manager.achat(
                model=request.model,
                messages=messages,
                stream=request.stream,
                json_mode=json_mode,
                response_format=response_format_dict,
                tools=tools,
                temperature=request.temperature,
                max_tokens=request.max_completion_tokens or request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                seed=request.seed,
                filename=request.filename,
                mmproj_filename=request.mmproj_filename,
            )
        except Exception as error:
            return _error_response(error)

        chat_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        if request.stream:
            if not isinstance(response, AsyncIterator):
                return _error_response(RuntimeError("Backend did not return a stream."))

            include_usage = request.stream_options is not None and request.stream_options.include_usage

            async def stream() -> AsyncIterator[str]:
                finish_reason = "stop"
                saw_tool_calls = False
                next_tool_index = 0
                usage: Usage | None = None
                yield _sse(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                try:
                    async for chunk in response:
                        if not isinstance(chunk, ChatChunk):
                            raise InvalidRequestError("Backend returned an invalid stream chunk.")
                        delta: dict[str, Any] = {}
                        if chunk.content:
                            delta["content"] = chunk.content
                        if chunk.tool_calls:
                            calls = []
                            for call in chunk.tool_calls:
                                call = dict(call)
                                if "index" not in call:
                                    call["index"] = next_tool_index
                                next_tool_index = max(next_tool_index, call["index"] + 1)
                                calls.append(call)
                            delta["tool_calls"] = calls
                            saw_tool_calls = True
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if delta:
                            yield _sse(
                                {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                }
                            )
                    yield _sse(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "tool_calls" if saw_tool_calls else finish_reason,
                                }
                            ],
                        }
                    )
                    if include_usage:
                        yield _sse(
                            {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [],
                                "usage": usage.as_openai() if usage is not None else None,
                            }
                        )
                except Exception as error:
                    payload = bytes(_error_response(error).body).decode()
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        if not isinstance(response, ChatResponse):
            return _error_response(RuntimeError("Backend returned a stream."))

        message: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            message["tool_calls"] = list(response.tool_calls)
        payload: dict[str, Any] = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        if response.usage is not None and (usage := response.usage.as_openai()) is not None:
            payload["usage"] = usage
        return payload

    @app.post("/v1/embeddings", response_model=None)
    async def embeddings(request: EmbeddingInputRequest) -> JSONResponse | dict[str, Any]:
        if request.encoding_format != "float":
            return _client_error("Only float encoding_format is supported.")

        inputs = [request.input] if isinstance(request.input, str) else request.input
        if not inputs or any(not isinstance(text, str) or not text for text in inputs):
            return _client_error("Embedding input must contain non-empty strings.")

        try:
            response = await llm_manager.aembed(
                model=request.model,
                input=inputs,
            )
        except Exception as error:
            return _error_response(error)

        data = [
            {
                "object": "embedding",
                "index": item.index,
                "embedding": list(item.embedding),
            }
            for item in response.data
        ]
        payload: dict[str, Any] = {
            "object": "list",
            "data": data,
            "model": request.model,
        }
        if response.usage is not None and response.usage.prompt_tokens is not None:
            payload["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.prompt_tokens,
            }
        return payload

    return app


_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Builds the module-level ``app`` (for ``uvicorn omnillm.server:app``) on first access, so importing this
    module never reads config files."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
