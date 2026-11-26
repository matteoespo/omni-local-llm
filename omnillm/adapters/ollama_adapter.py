import asyncio
import os
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from omnillm.core.base import LLMBackend
from omnillm.core.errors import (
    BackendUnavailableError,
    InvalidRequestError,
    ModelNotDownloadedError,
    ModelNotFoundError,
    OmniLLMError,
)
from omnillm.core.media import normalize_messages_for_ollama
from omnillm.core.tool_calls import to_openai_tool_call
from omnillm.core.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    ModelSource,
    Usage,
)

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

httpx: Any = None
try:
    import httpx as httpx_module
except ImportError:  # Installed with the ollama client.
    pass
else:
    httpx = httpx_module


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


class OllamaAdapter(LLMBackend):
    """Adapter for the official Ollama Python client.

    `host` (default: ``$OLLAMA_HOST`` or localhost) and `timeout` let one manager talk to several Ollama
    servers. Connection failures raise `BackendUnavailableError`, so alias fallbacks can take over.
    """

    def __init__(
        self,
        client: Any | None = None,
        async_client_factory: Callable[[], Any] | None = None,
        *,
        host: str | None = None,
        timeout: float | None = None,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if client is None and ollama is not None and (host is not None or timeout is not None):
            client = ollama.Client(host=host, timeout=timeout)
        self._client = client if client is not None else ollama
        if async_client_factory is None and AsyncClient is not None:

            def async_client_factory() -> Any:
                return AsyncClient(host=host, timeout=timeout)

        self._async_client_factory = async_client_factory
        self._async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = weakref.WeakKeyDictionary()
        self._known_models: set[str] = set()

    def _response_error_type(self) -> Any:
        response_error = getattr(self._client, "ResponseError", None)
        if response_error is None and ollama is not None:
            response_error = getattr(ollama, "ResponseError", None)
        return response_error

    def _translate(self, error: Exception) -> Exception:
        if isinstance(error, OmniLLMError):
            return error
        response_error = self._response_error_type()
        if response_error is not None and isinstance(error, response_error):
            status = _value(error, "status_code") or -1
            message = f"Ollama error: {_value(error, 'error') or error}"
            if status == 404:
                return ModelNotFoundError(message)
            return InvalidRequestError(message) if 400 <= status < 500 else BackendUnavailableError(message)
        if isinstance(error, ConnectionError) or (httpx is not None and isinstance(error, httpx.TransportError)):
            return BackendUnavailableError(f"Cannot reach the Ollama server at {self.host}. Is `ollama serve` running?")
        return error

    @contextmanager
    def _errors(self) -> Iterator[None]:
        try:
            yield
        except Exception as error:
            translated = self._translate(error)
            if translated is error:
                raise
            raise translated from error

    def _require_client(self) -> Any:
        if self._client is None:
            raise BackendUnavailableError(
                "The Ollama backend requires the optional dependency. Install omni-local-llm[ollama]."
            )
        return self._client

    def _make_async_client(self) -> Any:
        if self._async_client_factory is None:
            raise BackendUnavailableError(
                "The Ollama backend requires the optional dependency. Install omni-local-llm[ollama]."
            )
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            client = self._async_clients[loop] = self._async_client_factory()
        return client

    def pull_model(self, model_name: str, source: ModelSource) -> None:
        """Pulls the model if the Ollama server doesn't have it, unless `source.local_files_only` is set."""
        if model_name in self._known_models:
            return

        client = self._require_client()
        try:
            client.show(model_name)
        except Exception as error:
            response_error = self._response_error_type()
            if response_error is None or not isinstance(error, response_error) or _value(error, "status_code") != 404:
                raise
            if source.local_files_only:
                raise ModelNotDownloadedError(
                    f"Ollama model '{model_name}' is not installed and downloads are disabled."
                ) from error
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
                "seed": request.seed,
            }.items()
            if value is not None
        }
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": normalize_messages_for_ollama(request.messages),
            "stream": request.stream,
        }
        if options:
            kwargs["options"] = options
        if request.json_schema is not None:
            kwargs["format"] = request.json_schema
        elif request.is_json:
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
        done_reason = _value(response, "done_reason")
        if done_reason in {"tool_calls", "length"}:
            return done_reason
        return "stop"

    @staticmethod
    def _tool_calls(message: Any) -> tuple[dict[str, Any], ...]:
        return tuple(to_openai_tool_call(call) for call in (_value(message, "tool_calls", []) or []))

    @classmethod
    def _response(cls, response: Any) -> ChatResponse:
        message = _value(response, "message", {})
        tool_calls = cls._tool_calls(message)
        return ChatResponse(
            content=_value(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else cls._finish_reason(response),
            usage=cls._usage(response),
        )

    @classmethod
    def _chunk(cls, response: Any) -> ChatChunk:
        message = _value(response, "message", {})
        done = bool(_value(response, "done"))
        return ChatChunk(
            content=_value(message, "content", "") or "",
            tool_calls=cls._tool_calls(message),
            finish_reason=cls._finish_reason(response) if done else None,
            usage=cls._usage(response) if done else None,
        )

    def _stream(self, first: Any, iterator: Iterator[Any]) -> Iterator[ChatChunk]:
        if first is not None:
            yield self._chunk(first)
        with self._errors():
            for raw in iterator:
                yield self._chunk(raw)

    async def _astream(self, first: Any, iterator: AsyncIterator[Any]) -> AsyncIterator[ChatChunk]:
        if first is not None:
            yield self._chunk(first)
        with self._errors():
            async for raw in iterator:
                yield self._chunk(raw)

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        with self._errors():
            self.pull_model(request.model, request.model_source)
            response = self._require_client().chat(**self._chat_kwargs(request))
            if not request.stream:
                return self._response(response)
            # Fetch the first chunk now so connection/model errors surface here (enabling fallbacks).
            iterator = iter(response)
            first = next(iterator, None)
        return self._stream(first, iterator)

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        with self._errors():
            await asyncio.to_thread(self.pull_model, request.model, request.model_source)
            response = await self._make_async_client().chat(**self._chat_kwargs(request))
            if not request.stream:
                return self._response(response)
            iterator = aiter(response)
            first = await anext(iterator, None)
        return self._astream(first, iterator)

    @classmethod
    def _embedding_response(cls, request: EmbeddingRequest, raw_response: Any) -> EmbeddingResponse:
        embeddings = _value(raw_response, "embeddings", []) or []
        data = [EmbeddingData(index=idx, embedding=tuple(emb)) for idx, emb in enumerate(embeddings)]
        prompt_tokens = _value(raw_response, "prompt_eval_count")
        usage = Usage(prompt_tokens=prompt_tokens) if prompt_tokens is not None else None
        return EmbeddingResponse(data=data, model=request.model, usage=usage)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        with self._errors():
            return self._embed(request)

    def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.pull_model(request.model, request.model_source)
        client = self._require_client()
        if hasattr(client, "embed"):
            response = client.embed(model=request.model, input=list(request.input))
            return self._embedding_response(request, response)
        embeddings: list[Sequence[float]] = []
        for text in request.input:
            res = client.embeddings(model=request.model, prompt=text)
            embeddings.append(_value(res, "embedding", []))
        return self._embedding_response(request, {"embeddings": embeddings})

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        with self._errors():
            return await self._aembed(request)

    async def _aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        await asyncio.to_thread(self.pull_model, request.model, request.model_source)
        client = self._make_async_client()
        if hasattr(client, "embed"):
            response = await client.embed(model=request.model, input=list(request.input))
            return self._embedding_response(request, response)
        embeddings: list[Sequence[float]] = []
        for text in request.input:
            res = await client.embeddings(model=request.model, prompt=text)
            embeddings.append(_value(res, "embedding", []))
        return self._embedding_response(request, {"embeddings": embeddings})

    def list_models(self) -> list[str]:
        with self._errors():
            response = self._require_client().list()
        models = _value(response, "models", []) or []
        return [
            _value(model, "model", _value(model, "name"))
            for model in models
            if _value(model, "model", _value(model, "name"))
        ]
