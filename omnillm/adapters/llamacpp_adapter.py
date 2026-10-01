import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from omnillm.core.base import LLMBackend
from omnillm.core.errors import BackendUnavailableError, InvalidRequestError
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

hf_hub_download: Any = None
try:
    from huggingface_hub import hf_hub_download as hub_download
except ImportError:  # Optional dependency.
    pass
else:
    hf_hub_download = hub_download

Llama: Any = None
try:
    from llama_cpp import Llama as LlamaClass
except ImportError:  # Optional dependency.
    pass
else:
    Llama = LlamaClass


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


def _finish_reason(value: Any) -> FinishReason:
    if value in {"stop", "tool_calls", "length"}:
        return value
    return "stop"


class LlamaCPPAdapter(LLMBackend):
    """Adapter for llama.cpp with one cached in-process model."""

    def __init__(
        self,
        llama_factory: Callable[..., Any] | None = None,
        hub_download: Callable[..., str] | None = None,
    ):
        self._llama_factory = llama_factory if llama_factory is not None else Llama
        self._hub_download = hub_download if hub_download is not None else hf_hub_download
        self._active_key: tuple[str, ModelSource, bool] | None = None
        self._llm: Any | None = None

    def _require_dependencies(self) -> tuple[Callable[..., Any], Callable[..., str]]:
        if self._llama_factory is None or self._hub_download is None:
            raise BackendUnavailableError(
                "The llama.cpp backend requires optional dependencies. Install omni-local-llm[llama-cpp]."
            )
        return self._llama_factory, self._hub_download

    def pull_model(self, model_name: str, source: ModelSource) -> str:
        if not source.filename:
            raise InvalidRequestError("llama.cpp requires a Hugging Face GGUF filename.")
        _, hub_download = self._require_dependencies()
        return hub_download(
            repo_id=model_name,
            filename=source.filename,
            revision=source.revision,
            cache_dir=source.cache_dir,
            local_files_only=source.local_files_only,
        )

    def _load_model(self, request: ChatRequest | EmbeddingRequest, embedding: bool = False) -> Any:
        key = (request.model, request.model_source, embedding)
        if self._active_key == key and self._llm is not None:
            return self._llm

        llama_factory, _ = self._require_dependencies()
        model_path = self.pull_model(request.model, request.model_source)
        runtime_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_gpu_layers": request.runtime.n_gpu_layers if request.runtime.n_gpu_layers is not None else -1,
            "verbose": False,
        }
        if request.runtime.n_ctx is not None:
            runtime_kwargs["n_ctx"] = request.runtime.n_ctx
        if embedding:
            runtime_kwargs["embedding"] = True
        self._llm = llama_factory(**runtime_kwargs)
        self._active_key = key
        return self._llm

    @staticmethod
    def _chat_kwargs(request: ChatRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"messages": list(request.messages), "stream": request.stream}
        for key, value in {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stop": request.stop,
        }.items():
            if value is not None:
                kwargs[key] = value
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.tools:
            kwargs["tools"] = list(request.tools)
        return kwargs

    @staticmethod
    def _usage(response: Any) -> Usage | None:
        usage = _value(response, "usage")
        if usage is None:
            return None
        prompt_tokens = _value(usage, "prompt_tokens")
        completion_tokens = _value(usage, "completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @classmethod
    def _response(cls, response: Any) -> ChatResponse:
        choice = _value(response, "choices", [])[0]
        message = _value(choice, "message", {})
        tool_calls = tuple(_as_dict(call) for call in (_value(message, "tool_calls", []) or []))
        return ChatResponse(
            content=_value(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else _finish_reason(_value(choice, "finish_reason", "stop")),
            usage=cls._usage(response),
        )

    @staticmethod
    def _chunk(response: Any) -> ChatChunk:
        choice = _value(response, "choices", [])[0]
        delta = _value(choice, "delta", {})
        tool_calls = tuple(_as_dict(call) for call in (_value(delta, "tool_calls", []) or []))
        finish_reason = _value(choice, "finish_reason")
        return ChatChunk(
            content=_value(delta, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=_finish_reason(finish_reason) if finish_reason is not None else None,
        )

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        response = self._load_model(request).create_chat_completion(**self._chat_kwargs(request))
        if not request.stream:
            return self._response(response)

        def stream() -> Iterator[ChatChunk]:
            for chunk in response:
                yield self._chunk(chunk)

        return stream()

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        result = await asyncio.to_thread(self.chat, request)
        if not request.stream:
            if not isinstance(result, ChatResponse):
                raise RuntimeError("llama.cpp returned a stream for a non-streaming request.")
            return result

        if not isinstance(result, Iterator):
            raise RuntimeError("llama.cpp returned a non-streaming response for a streaming request.")
        iterator = result

        def next_or_none(iterator: Iterator[ChatChunk]) -> ChatChunk | None:
            try:
                return next(iterator)
            except StopIteration:
                return None

        async def stream() -> AsyncIterator[ChatChunk]:
            while True:
                chunk = await asyncio.to_thread(next_or_none, iterator)
                if chunk is None:
                    break
                yield chunk

        return stream()

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        llm = self._load_model(request, embedding=True)
        raw_response = llm.create_embedding(input=list(request.input), model=request.model)
        raw_data = _value(raw_response, "data", []) or []
        data = [
            EmbeddingData(
                index=_value(item, "index", idx),
                embedding=tuple(_value(item, "embedding", [])),
            )
            for idx, item in enumerate(raw_data)
        ]
        raw_usage = _value(raw_response, "usage")
        prompt_tokens = _value(raw_usage, "prompt_tokens") if raw_usage else None
        usage = Usage(prompt_tokens=prompt_tokens) if prompt_tokens is not None else None
        return EmbeddingResponse(data=data, model=request.model, usage=usage)

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return await asyncio.to_thread(self.embed, request)

    def list_models(self) -> list[str]:
        return [self._active_key[0]] if self._active_key else []
