import asyncio
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

from omnillm.core.base import LLMBackend
from omnillm.core.errors import BackendUnavailableError, InvalidRequestError, ModelNotDownloadedError
from omnillm.core.media import normalize_messages_for_openai
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


class _Lease:
    """Releases a held lock exactly once, however the holder finishes (completion, error, close, or GC)."""

    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._release()


class _LeasedStream(Iterator[ChatChunk]):
    """A chunk iterator that keeps the model lease until it is exhausted, fails, or is closed/collected."""

    def __init__(self, source: Iterator[Any], lease: _Lease, convert: Callable[[Any], ChatChunk]):
        self._source = source
        self._lease = lease
        self._convert = convert

    def __next__(self) -> ChatChunk:
        try:
            return self._convert(next(self._source))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        finally:
            self._lease.release()

    def __del__(self) -> None:
        self._lease.release()


class _AsyncLeasedStream(AsyncIterator[ChatChunk]):
    """Async view of a leased sync stream; holds the async lease until exhausted, failed, or closed/collected."""

    def __init__(self, source: Iterator[ChatChunk], lease: _Lease):
        self._source = source
        self._lease = lease

    @staticmethod
    def _next_or_none(iterator: Iterator[ChatChunk]) -> ChatChunk | None:
        try:
            return next(iterator)
        except StopIteration:
            return None

    async def __anext__(self) -> ChatChunk:
        try:
            chunk = await asyncio.to_thread(self._next_or_none, self._source)
        except BaseException:
            await self.aclose()
            raise
        if chunk is None:
            await self.aclose()
            raise StopAsyncIteration
        return chunk

    async def aclose(self) -> None:
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        finally:
            self._lease.release()

    def __del__(self) -> None:
        close = getattr(self._source, "close", None)
        if callable(close):
            close()
        self._lease.release()


class LlamaCPPAdapter(LLMBackend):
    """Adapter for llama.cpp with one cached in-process model.

    A llama.cpp model instance is not thread-safe, so requests are serialized: one generation (including a
    stream, until it is exhausted or closed) runs at a time. Async callers wait on the event loop rather
    than occupying worker threads.
    """

    def __init__(
        self,
        llama_factory: Callable[..., Any] | None = None,
        hub_download: Callable[..., str] | None = None,
    ):
        self._llama_factory = llama_factory if llama_factory is not None else Llama
        self._hub_download = hub_download if hub_download is not None else hf_hub_download
        self._active_key: tuple[str, ModelSource, bool, str | None] | None = None
        self._llm: Any | None = None
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._async_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    def _acquire(self, track_owner: bool) -> _Lease:
        if not self._lock.acquire(blocking=False):
            if track_owner and self._owner == threading.get_ident():
                raise RuntimeError(
                    "The llama.cpp model is busy with an unfinished stream in this thread. "
                    "Consume or close() the previous stream before starting another request."
                )
            self._lock.acquire()
        self._owner = threading.get_ident() if track_owner else None

        def release() -> None:
            self._owner = None
            self._lock.release()

        return _Lease(release)

    async def _acquire_async(self) -> _Lease:
        loop = asyncio.get_running_loop()
        lock = self._async_locks.get(loop)
        if lock is None:
            lock = self._async_locks[loop] = asyncio.Lock()
        await lock.acquire()
        return _Lease(lock.release)

    def _require_dependencies(self) -> tuple[Callable[..., Any], Callable[..., str]]:
        if self._llama_factory is None or self._hub_download is None:
            raise BackendUnavailableError(
                "The llama.cpp backend requires optional dependencies. Install omni-local-llm[llama-cpp]."
            )
        return self._llama_factory, self._hub_download

    def _fetch(self, model_name: str, filename: str, source: ModelSource) -> str:
        """Uses the local Hugging Face cache when the file is there (no network, works offline); downloads otherwise."""
        _, hub_download = self._require_dependencies()
        kwargs = {
            "repo_id": model_name,
            "filename": filename,
            "revision": source.revision,
            "cache_dir": source.cache_dir,
        }
        try:
            return hub_download(**kwargs, local_files_only=True)
        except Exception as error:
            if source.local_files_only:
                raise ModelNotDownloadedError(
                    f"'{filename}' from '{model_name}' is not in the local Hugging Face cache and downloads are disabled."
                ) from error
        return hub_download(**kwargs, local_files_only=False)

    def pull_model(self, model_name: str, source: ModelSource) -> str:
        if not source.filename:
            raise InvalidRequestError("llama.cpp requires a Hugging Face GGUF filename.")
        if source.mmproj_filename:
            self._fetch(model_name, source.mmproj_filename, source)
        return self._fetch(model_name, source.filename, source)

    def _load_model(self, request: ChatRequest | EmbeddingRequest, embedding: bool = False) -> Any:
        clip_path: str | None = None
        llama_factory, _ = self._require_dependencies()
        if isinstance(request, ChatRequest):
            if request.runtime.clip_model_path:
                clip_path = request.runtime.clip_model_path
            elif request.model_source.mmproj_filename:
                clip_path = self._fetch(request.model, request.model_source.mmproj_filename, request.model_source)

        key = (request.model, request.model_source, embedding, clip_path)
        if self._active_key == key and self._llm is not None:
            return self._llm

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
        if clip_path is not None:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                runtime_kwargs["chat_handler"] = Llava15ChatHandler(clip_model_path=clip_path)
            except Exception:
                runtime_kwargs["clip_model_path"] = clip_path
        self._llm = llama_factory(**runtime_kwargs)
        self._active_key = key
        return self._llm

    @staticmethod
    def _chat_kwargs(request: ChatRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": normalize_messages_for_openai(request.messages),
            "stream": request.stream,
        }
        for key, value in {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stop": request.stop,
            "seed": request.seed,
        }.items():
            if value is not None:
                kwargs[key] = value
        if request.json_schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": request.json_schema}
        elif request.is_json:
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
        tool_calls = tuple(to_openai_tool_call(call) for call in (_value(message, "tool_calls", []) or []))
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
        # Streaming tool-call deltas are partial (index + fragments), so they are passed through untouched.
        tool_calls = tuple(_as_dict(call) for call in (_value(delta, "tool_calls", []) or []))
        finish_reason = _value(choice, "finish_reason")
        return ChatChunk(
            content=_value(delta, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=_finish_reason(finish_reason) if finish_reason is not None else None,
            usage=LlamaCPPAdapter._usage(response),
        )

    def _chat(self, request: ChatRequest, *, track_owner: bool) -> ChatResponse | Iterator[ChatChunk]:
        lease = self._acquire(track_owner)
        try:
            response = self._load_model(request).create_chat_completion(**self._chat_kwargs(request))
        except BaseException:
            lease.release()
            raise
        if not request.stream:
            lease.release()
            return self._response(response)
        return _LeasedStream(iter(response), lease, self._chunk)

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        return self._chat(request, track_owner=True)

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        lease = await self._acquire_async()
        try:
            result = await asyncio.to_thread(self._chat, request, track_owner=False)
        except BaseException:
            lease.release()
            raise
        if not request.stream:
            lease.release()
            if not isinstance(result, ChatResponse):
                raise RuntimeError("llama.cpp returned a stream for a non-streaming request.")
            return result
        if not isinstance(result, Iterator):
            lease.release()
            raise RuntimeError("llama.cpp returned a non-streaming response for a streaming request.")
        return _AsyncLeasedStream(result, lease)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        lease = self._acquire(track_owner=True)
        try:
            llm = self._load_model(request, embedding=True)
            raw_response = llm.create_embedding(input=list(request.input), model=request.model)
        finally:
            lease.release()
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
        lease = await self._acquire_async()
        try:
            return await asyncio.to_thread(self.embed, request)
        finally:
            lease.release()

    def list_models(self) -> list[str]:
        return [self._active_key[0]] if self._active_key else []
