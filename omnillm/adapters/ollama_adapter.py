import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from omnillm.core.base import LLMBackend
from omnillm.core.errors import BackendUnavailableError
from omnillm.core.types import ChatChunk, ChatRequest, ChatResponse, FinishReason, ModelSource, Usage

ollama: Any = None
AsyncClient: Any = None
try:
    import ollama as ollama_module
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:  # Optional dependency.
    pass
else:
    ollama = ollama_module
    AsyncClient = OllamaAsyncClient


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {"value": value}


class OllamaAdapter(LLMBackend):
    """Adapter for the official Ollama Python client."""

    def __init__(
        self,
        client: Any | None = None,
        async_client_factory: Callable[[], Any] | None = None,
    ):
        self._client = client if client is not None else ollama
        self._async_client_factory = async_client_factory
        self._known_models: set[str] = set()

    def _require_client(self) -> Any:
        if self._client is None:
            raise BackendUnavailableError(
                "The Ollama backend requires the optional dependency. Install omni-local-llm[ollama]."
            )
        return self._client

    def _make_async_client(self) -> Any:
        if self._async_client_factory is not None:
            return self._async_client_factory()
        if AsyncClient is None:
            raise BackendUnavailableError(
                "The Ollama backend requires the optional dependency. Install omni-local-llm[ollama]."
            )
        return AsyncClient()

    def pull_model(self, model_name: str, source: ModelSource) -> None:
        del source
        if model_name in self._known_models:
            return

        client = self._require_client()
        try:
            client.show(model_name)
        except Exception as error:
            response_error = getattr(client, "ResponseError", None)
            if response_error is None and ollama is not None:
                response_error = getattr(ollama, "ResponseError", None)
            if response_error is None or not isinstance(error, response_error) or _value(error, "status_code") != 404:
                raise
            client.pull(model_name)
        self._known_models.add(model_name)

    @staticmethod
    def _chat_kwargs(request: ChatRequest) -> dict[str, Any]:
        options = {
            key: value
            for key, value in {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
                "stop": request.stop,
            }.items()
            if value is not None
        }
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": list(request.messages),
            "stream": request.stream,
        }
        if options:
            kwargs["options"] = options
        if request.json_mode:
            kwargs["format"] = "json"
        if request.tools:
            kwargs["tools"] = list(request.tools)
        return kwargs

    @staticmethod
    def _usage(response: Any) -> Usage | None:
        prompt_tokens = _value(response, "prompt_eval_count")
        completion_tokens = _value(response, "eval_count")
        if prompt_tokens is None or completion_tokens is None:
            return None
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @staticmethod
    def _finish_reason(response: Any) -> FinishReason:
        return "tool_calls" if _value(response, "done_reason") == "tool_calls" else "stop"

    @classmethod
    def _response(cls, response: Any) -> ChatResponse:
        message = _value(response, "message", {})
        tool_calls = tuple(_as_dict(call) for call in (_value(message, "tool_calls", []) or []))
        return ChatResponse(
            content=_value(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else cls._finish_reason(response),
            usage=cls._usage(response),
        )

    @classmethod
    def _chunk(cls, response: Any) -> ChatChunk:
        message = _value(response, "message", {})
        tool_calls = tuple(_as_dict(call) for call in (_value(message, "tool_calls", []) or []))
        return ChatChunk(
            content=_value(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=cls._finish_reason(response) if _value(response, "done") else None,
        )

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        self.pull_model(request.model, request.model_source)
        response = self._require_client().chat(**self._chat_kwargs(request))
        if not request.stream:
            return self._response(response)

        def stream() -> Iterator[ChatChunk]:
            for chunk in response:
                yield self._chunk(chunk)

        return stream()

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        await asyncio.to_thread(self.pull_model, request.model, request.model_source)
        response = await self._make_async_client().chat(**self._chat_kwargs(request))
        if not request.stream:
            return self._response(response)

        async def stream() -> AsyncIterator[ChatChunk]:
            async for chunk in response:
                yield self._chunk(chunk)

        return stream()

    def list_models(self) -> list[str]:
        response = self._require_client().list()
        models = _value(response, "models", []) or []
        return [
            _value(model, "model", _value(model, "name"))
            for model in models
            if _value(model, "model", _value(model, "name"))
        ]
