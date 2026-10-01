from collections.abc import Callable, Mapping, Sequence
from typing import Any

from omnillm.core.base import LLMBackend
from omnillm.core.errors import BackendNotFoundError, BackendUnavailableError, UnsupportedFeatureError
from omnillm.core.types import (
    AsyncChatResult,
    ChatMessage,
    ChatRequest,
    ChatResult,
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
        if json_mode and not adapter.capabilities.json_mode:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support JSON mode.")
        if tools and not adapter.capabilities.tools:
            raise UnsupportedFeatureError(f"Backend '{backend}' does not support tool calling.")
        return adapter, ChatRequest(
            model=model,
            messages=messages,
            stream=stream,
            json_mode=json_mode,
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
        return adapter.chat(request)

    async def achat(
        self,
        backend: str,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        stream: bool = False,
        json_mode: bool = False,
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
        return await adapter.achat(request)

    def create_session(self, backend: str, model: str, system_prompt: str | None = None):
        from omnillm.core.session import ChatSession

        self._get_backend(backend)
        return ChatSession(manager=self, backend=backend, model=model, system_prompt=system_prompt)

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
