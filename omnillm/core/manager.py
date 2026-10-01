from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from omnillm.core.session import ChatSession

from omnillm.core.base import LLMBackend
from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from omnillm.core.types import (
    AsyncChatResult,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelSource,
    RuntimeOptions,
)

BackendFactory = Callable[[], LLMBackend]


def _create_ollama() -> LLMBackend:
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    return OllamaAdapter()


def _create_llamacpp() -> LLMBackend:
    from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter

    return LlamaCPPAdapter()


def _resolve_response_format(
    response_model: type[Any] | None,
    response_format: dict[str, Any] | None,
    json_mode: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if response_model is not None:
        if hasattr(response_model, "model_json_schema"):
            schema = response_model.model_json_schema()
        elif hasattr(response_model, "schema"):
            schema = response_model.schema()
        else:
            raise ValueError(f"response_model '{response_model}' must be a Pydantic model with model_json_schema().")
        return True, {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(response_model, "__name__", "ResponseModel"),
                "schema": schema,
                "strict": True,
            },
        }
    if response_format is not None:
        fmt_type = response_format.get("type", "")
        is_json = (
            json_mode
            or fmt_type in {"json_object", "json_schema"}
            or "schema" in response_format
            or "properties" in response_format
        )
        return is_json, response_format
    return json_mode, None


def _attach_parsed_model[R: (ChatResult, AsyncChatResult)](result: R, response_model: type[Any] | None) -> R:
    if response_model is None or not isinstance(result, ChatResponse):
        return result
    try:
        parsed = result.parse_as(response_model)
        return cast(
            R,
            ChatResponse(
                content=result.content,
                tool_calls=result.tool_calls,
                finish_reason=result.finish_reason,
                usage=result.usage,
                parsed=parsed,
            ),
        )
    except Exception as error:
        name = getattr(response_model, "__name__", str(response_model))
        raise InvalidRequestError(f"Failed to parse response into {name}: {error}") from error


class LocalLLMManager:
    """The external seam for backend registration, routing, and request validation."""

    def __init__(self, backends: Mapping[str, LLMBackend] | None = None):
        self._backend_factories: dict[str, BackendFactory] = (
            {}
            if backends is not None
            else {
                "ollama": _create_ollama,
                "llama.cpp": _create_llamacpp,
            }
        )
        self._backends = dict(backends or {})

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self._backend_factories, *self._backends)))

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        self._backend_factories[name] = factory
        self._backends.pop(name, None)

    def _get_backend(self, name: str) -> LLMBackend:
        if name in self._backends:
            return self._backends[name]
        if name not in self._backend_factories:
            raise BackendNotFoundError(f"Backend '{name}' is not supported. Choose from {list(self.backend_names)}")
        self._backends[name] = self._backend_factories[name]()
        return self._backends[name]

    def _make_request(
        self,
        backend: str,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        json_mode: bool,
        response_model: type[Any] | None,
        response_format: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | Sequence[str] | None,
        filename: str | None,
        revision: str | None,
        cache_dir: str | None,
        local_files_only: bool,
        n_gpu_layers: int | None,
        n_ctx: int | None,
    ) -> tuple[LLMBackend, ChatRequest]:
        if not model:
            raise ValueError("A model name is required.")
        adapter = self._get_backend(backend)
        if stream and not adapter.capabilities.streaming:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support streaming.")
        is_json, resolved_format = _resolve_response_format(response_model, response_format, json_mode)
        if is_json and not adapter.capabilities.json_mode:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support JSON mode.")
        if (
            response_model is not None or (resolved_format and resolved_format.get("type") == "json_schema")
        ) and not getattr(adapter.capabilities, "structured_outputs", True):
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support structured outputs.")
        if tools and not adapter.capabilities.tools:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support tool calling.")
        return adapter, ChatRequest(
            model=model,
            messages=messages,
            stream=stream,
            json_mode=is_json,
            response_format=resolved_format,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            model_source=ModelSource(filename, revision, cache_dir, local_files_only),
            runtime=RuntimeOptions(n_gpu_layers, n_ctx),
        )

    def chat(
        self,
        backend: str,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> ChatResult:
        adapter, request = self._make_request(
            backend,
            model,
            messages,
            stream=stream,
            json_mode=json_mode,
            response_model=response_model,
            response_format=response_format,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        result = adapter.chat(request)
        return _attach_parsed_model(result, response_model)

    async def achat(
        self,
        backend: str,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> AsyncChatResult:
        adapter, request = self._make_request(
            backend,
            model,
            messages,
            stream=stream,
            json_mode=json_mode,
            response_model=response_model,
            response_format=response_format,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        result = await adapter.achat(request)
        return _attach_parsed_model(result, response_model)

    def _make_embedding_request(
        self,
        backend: str,
        model: str,
        input: str | Sequence[str],
        *,
        filename: str | None,
        revision: str | None,
        cache_dir: str | None,
        local_files_only: bool,
        n_gpu_layers: int | None,
        n_ctx: int | None,
    ) -> tuple[LLMBackend, EmbeddingRequest]:
        if not model:
            raise ValueError("A model name is required.")
        if isinstance(input, str):
            inputs: tuple[str, ...] = (input,)
        elif isinstance(input, Sequence):
            inputs = tuple(input)
        else:
            raise ValueError("Embedding input must be a string or a sequence of strings.")

        if not inputs or any(not isinstance(item, str) for item in inputs):
            raise ValueError("Embedding input must contain at least one string.")

        adapter = self._get_backend(backend)
        if not adapter.capabilities.embeddings:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support embeddings.")

        return adapter, EmbeddingRequest(
            model=model,
            input=inputs,
            model_source=ModelSource(filename, revision, cache_dir, local_files_only),
            runtime=RuntimeOptions(n_gpu_layers, n_ctx),
        )

    def embed(
        self,
        backend: str,
        model: str,
        input: str | Sequence[str],
        *,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> EmbeddingResponse:
        adapter, request = self._make_embedding_request(
            backend,
            model,
            input,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        return adapter.embed(request)

    async def aembed(
        self,
        backend: str,
        model: str,
        input: str | Sequence[str],
        *,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> EmbeddingResponse:
        adapter, request = self._make_embedding_request(
            backend,
            model,
            input,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        return await adapter.aembed(request)

    def create_session(
        self,
        backend: str,
        model: str,
        system_prompt: str | None = None,
        *,
        max_turns: int | None = None,
        max_tokens_budget: int | None = None,
        strategy: Literal["full", "sliding_window"] = "full",
    ) -> "ChatSession":
        from omnillm.core.session import ChatSession

        self._get_backend(backend)
        return ChatSession(
            manager=self,
            backend=backend,
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_tokens_budget=max_tokens_budget,
            strategy=strategy,
        )

    def list_models(self, backend: str | None = None) -> list[tuple[str, str]]:
        names = (backend,) if backend else self.backend_names
        models: list[tuple[str, str]] = []
        for name in names:
            try:
                available_models = self._get_backend(name).list_models()
            except BackendUnavailableError:
                if backend is not None:
                    raise
                continue
            for model in available_models:
                models.append((name, model))
        return models
