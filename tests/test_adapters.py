import asyncio
import gc
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
from omnillm.adapters.ollama_adapter import OllamaAdapter
from omnillm.core.errors import BackendUnavailableError, InvalidRequestError, ModelNotDownloadedError
from omnillm.core.types import ChatRequest, EmbeddingRequest, ModelSource


class FakeOllamaClient:
    class ResponseError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code

    def __init__(self):
        self.chat_kwargs: dict[str, Any] | None = None
        self.show_calls = 0

    def show(self, model: str) -> None:
        self.show_calls += 1

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_kwargs = kwargs
        return {
            "message": {"content": '{"name": "test"}', "tool_calls": [{"function": {"name": "tool"}}]},
            "prompt_eval_count": 3,
            "eval_count": 2,
        }

    def embed(self, model: str, input: list[str]) -> dict[str, Any]:
        self.embed_kwargs = {"model": model, "input": input}
        return {
            "embeddings": [[0.1, 0.2] for _ in input],
            "prompt_eval_count": 4,
        }

    def list(self) -> dict[str, Any]:
        return {"models": [{"model": "llama3"}]}


class FakeAsyncOllamaClient:
    def __init__(self):
        self.chat_kwargs: dict[str, Any] | None = None
        self.embed_kwargs: dict[str, Any] | None = None

    async def chat(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.chat_kwargs = kwargs

        async def stream() -> AsyncIterator[dict[str, Any]]:
            yield {"message": {"content": "one"}}
            yield {"message": {"content": "two"}, "done": True}

        return stream()

    async def embed(self, model: str, input: list[str]) -> dict[str, Any]:
        self.embed_kwargs = {"model": model, "input": input}
        return {
            "embeddings": [[0.1, 0.2] for _ in input],
            "prompt_eval_count": 4,
        }


def test_ollama_maps_generation_options_and_normalizes_usage():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)

    response = adapter.chat(
        ChatRequest(
            model="test-model",
            messages=[],
            json_mode=True,
            tools=[{"type": "function"}],
            temperature=0.2,
            max_tokens=7,
            top_p=0.9,
            stop="END",
        )
    )

    assert response.content == '{"name": "test"}'
    assert response.tool_calls[0]["function"]["name"] == "tool"
    assert response.usage is not None
    assert response.usage.total_tokens == 5
    assert client.chat_kwargs == {
        "model": "test-model",
        "messages": [],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 7, "top_p": 0.9, "stop": "END"},
        "format": "json",
        "tools": [{"type": "function"}],
    }
    assert adapter.list_models() == ["llama3"]


@pytest.mark.asyncio
async def test_ollama_async_stream_returns_typed_chunks():
    client = FakeOllamaClient()
    async_client = FakeAsyncOllamaClient()
    adapter = OllamaAdapter(client=client, async_client_factory=lambda: async_client)

    stream = await adapter.achat(ChatRequest(model="test-model", messages=[], stream=True))

    assert [chunk.content async for chunk in stream] == ["one", "two"]
    assert async_client.chat_kwargs == {"model": "test-model", "messages": [], "stream": True}


class FakeLlama:
    instances: list["FakeLlama"] = []

    def __init__(self, **kwargs: Any):
        self.init_kwargs = kwargs
        self.chat_kwargs: dict[str, Any] | None = None
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any] | Iterator[dict[str, Any]]:
        self.chat_kwargs = kwargs
        if kwargs["stream"]:
            return iter(
                [
                    {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
                ]
            )
        return {
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }

    def create_embedding(self, input: list[str], model: str | None = None) -> dict[str, Any]:
        self.embed_kwargs = {"input": input, "model": model}
        return {
            "data": [{"index": i, "embedding": [0.5, 0.6]} for i in range(len(input))],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }


def test_ollama_embed():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)

    res = adapter.embed(EmbeddingRequest(model="test-embed", input=["hello", "world"]))

    assert res.embeddings == [[0.1, 0.2], [0.1, 0.2]]
    assert res.usage is not None and res.usage.prompt_tokens == 4
    assert client.embed_kwargs == {"model": "test-embed", "input": ["hello", "world"]}


@pytest.mark.asyncio
async def test_ollama_aembed():
    client = FakeOllamaClient()
    async_client = FakeAsyncOllamaClient()
    adapter = OllamaAdapter(client=client, async_client_factory=lambda: async_client)

    res = await adapter.aembed(EmbeddingRequest(model="test-embed", input=["doc"]))

    assert res.embeddings == [[0.1, 0.2]]
    assert async_client.embed_kwargs == {"model": "test-embed", "input": ["doc"]}


def test_llamacpp_caches_model_and_forwards_generation_options():
    downloads: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return "/models/test.gguf"

    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=download)
    request = ChatRequest(
        model="owner/model",
        messages=[],
        model_source=ModelSource(filename="test.gguf", revision="main"),
        temperature=0.4,
        max_tokens=5,
        top_p=0.8,
        stop=["END"],
    )

    assert adapter.chat(request).content == "done"
    assert adapter.chat(request).content == "done"
    assert len(downloads) == 1
    assert len(FakeLlama.instances) == 1
    assert FakeLlama.instances[0].init_kwargs["n_gpu_layers"] == -1
    assert FakeLlama.instances[0].chat_kwargs == {
        "messages": [],
        "stream": False,
        "temperature": 0.4,
        "max_tokens": 5,
        "top_p": 0.8,
        "stop": ["END"],
    }


@pytest.mark.asyncio
async def test_llamacpp_async_stream_ends_cleanly():
    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=lambda **_: "/models/test.gguf")
    request = ChatRequest(model="owner/model", messages=[], stream=True, model_source=ModelSource(filename="test.gguf"))

    stream = await adapter.achat(request)

    assert [chunk.content async for chunk in stream] == ["hello", " world"]


def test_llamacpp_embed_sets_embedding_mode_and_returns_embeddings():
    downloads: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return "/models/test.gguf"

    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=download)
    request = EmbeddingRequest(
        model="owner/embed-model",
        input=["chunk 1", "chunk 2"],
        model_source=ModelSource(filename="test.gguf"),
    )

    response = adapter.embed(request)

    assert response.embeddings == [[0.5, 0.6], [0.5, 0.6]]
    assert response.usage is not None and response.usage.prompt_tokens == 5
    assert FakeLlama.instances[-1].init_kwargs.get("embedding") is True
    assert FakeLlama.instances[-1].embed_kwargs == {"input": ["chunk 1", "chunk 2"], "model": "owner/embed-model"}


def test_ollama_passes_schema_dict_when_json_schema_present():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    adapter.chat(
        ChatRequest(
            model="test-model",
            messages=[],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
        )
    )

    assert client.chat_kwargs is not None
    assert client.chat_kwargs["format"] == schema


def test_llamacpp_passes_json_object_with_schema_when_json_schema_present():
    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=lambda **_: "/models/test.gguf")
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}}

    adapter.chat(
        ChatRequest(
            model="owner/model",
            messages=[],
            model_source=ModelSource(filename="test.gguf"),
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
        )
    )

    assert FakeLlama.instances[-1].chat_kwargs is not None
    assert FakeLlama.instances[-1].chat_kwargs["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }


def test_ollama_multimodal_normalizes_images():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)

    adapter.chat(
        ChatRequest(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aGVsbG8="}},
                    ],
                }
            ],
        )
    )

    assert client.chat_kwargs is not None
    messages = client.chat_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["content"] == "What is this?"
    assert messages[0]["images"] == ["aGVsbG8="]


def test_llamacpp_multimodal_normalizes_images_and_downloads_mmproj():
    downloads: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        return f"/models/{kwargs['filename']}"

    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=download)
    request = ChatRequest(
        model="owner/llava-model",
        messages=[{"role": "user", "content": "Analyze", "images": [b"raw_bytes"]}],
        model_source=ModelSource(filename="model.gguf", mmproj_filename="mmproj.gguf"),
    )

    adapter.chat(request)

    # Both model and mmproj downloaded
    downloaded_files = [d["filename"] for d in downloads]
    assert "model.gguf" in downloaded_files
    assert "mmproj.gguf" in downloaded_files

    # Formatted to OpenAI image_url content list
    chat_kwargs = FakeLlama.instances[-1].chat_kwargs
    assert chat_kwargs is not None
    msg = chat_kwargs["messages"][0]
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "Analyze"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_ollama_tool_calls_are_openai_shaped_and_history_round_trips():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)
    client.chat = lambda **kwargs: (  # type: ignore[method-assign]
        setattr(client, "chat_kwargs", kwargs)
        or {"message": {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}}
    )

    response = adapter.chat(ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}]))

    call = response.tool_calls[0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert response.finish_reason == "tool_calls"

    # Feeding the OpenAI-shaped call back must give Ollama the object-shaped arguments it requires.
    adapter.chat(
        ChatRequest(
            model="m",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "name": "f", "content": "2"},
            ],
        )
    )
    assert client.chat_kwargs is not None
    sent = client.chat_kwargs["messages"]
    assert sent[1]["tool_calls"] == [{"function": {"name": "f", "arguments": {"a": 1}}}]
    assert sent[2]["tool_name"] == "f"


def test_ollama_maps_length_finish_reason_seed_and_final_stream_usage():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)
    client.chat = lambda **kwargs: (  # type: ignore[method-assign]
        setattr(client, "chat_kwargs", kwargs)
        or iter(
            [
                {"message": {"content": "a"}},
                {
                    "message": {"content": "b"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                },
            ]
        )
    )

    chunks = list(adapter.chat(ChatRequest(model="m", messages=[], stream=True, seed=7)))  # type: ignore[arg-type]

    assert client.chat_kwargs is not None and client.chat_kwargs["options"] == {"seed": 7}
    assert chunks[0].usage is None and chunks[0].finish_reason is None
    assert chunks[1].finish_reason == "length"
    assert chunks[1].usage is not None and chunks[1].usage.completion_tokens == 2


class ConcurrencyTrackingLlama:
    """Fake model that records how many generations overlap."""

    def __init__(self, **kwargs: Any):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        with self.lock:
            self.active -= 1

    def create_chat_completion(self, **kwargs: Any) -> Any:
        if not kwargs["stream"]:
            self._enter()
            time.sleep(0.02)
            self._exit()
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        def stream() -> Iterator[dict[str, Any]]:
            self._enter()
            try:
                for token in ("a", "b", "c"):
                    time.sleep(0.01)
                    yield {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
            finally:
                self._exit()

        return stream()


def _llamacpp_with_tracking_model() -> tuple[LlamaCPPAdapter, list[ConcurrencyTrackingLlama]]:
    models: list[ConcurrencyTrackingLlama] = []

    def factory(**kwargs: Any) -> ConcurrencyTrackingLlama:
        models.append(ConcurrencyTrackingLlama(**kwargs))
        return models[-1]

    return LlamaCPPAdapter(llama_factory=factory, hub_download=lambda **_: "/m.gguf"), models


def _request(stream: bool = False) -> ChatRequest:
    return ChatRequest(model="o/m", messages=[], stream=stream, model_source=ModelSource(filename="m.gguf"))


def test_llamacpp_serializes_concurrent_requests_including_streams():
    adapter, models = _llamacpp_with_tracking_model()

    def worker(i: int) -> None:
        result = adapter.chat(_request(stream=i % 2 == 0))
        if isinstance(result, Iterator):
            list(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(models) == 1
    assert models[0].max_active == 1


def test_llamacpp_unfinished_stream_in_same_thread_raises_instead_of_deadlocking():
    adapter, _ = _llamacpp_with_tracking_model()
    stream = adapter.chat(_request(stream=True))
    assert isinstance(stream, Iterator)
    next(stream)

    with pytest.raises(RuntimeError, match="unfinished stream"):
        adapter.chat(_request())

    stream.close()  # type: ignore[attr-defined]
    assert adapter.chat(_request()).content == "ok"  # type: ignore[union-attr]


def test_llamacpp_abandoned_stream_releases_model():
    adapter, _ = _llamacpp_with_tracking_model()
    stream = adapter.chat(_request(stream=True))
    del stream
    gc.collect()

    assert adapter.chat(_request()).content == "ok"  # type: ignore[union-attr]


def test_llamacpp_load_failure_releases_model():
    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=lambda **_: "/m.gguf")
    with pytest.raises(InvalidRequestError):
        adapter.chat(ChatRequest(model="o/m", messages=[]))  # missing filename
    assert adapter.chat(_request()).content == "done"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_llamacpp_async_streams_are_serialized():
    adapter, models = _llamacpp_with_tracking_model()

    async def consume() -> list[str]:
        stream = await adapter.achat(_request(stream=True))
        assert isinstance(stream, AsyncIterator)
        return [chunk.content async for chunk in stream]

    results = await asyncio.gather(*(consume() for _ in range(4)))

    assert results == [["a", "b", "c"]] * 4
    assert models[0].max_active == 1


def test_ollama_connection_errors_become_backend_unavailable():
    client = FakeOllamaClient()
    client.show = lambda model: (_ for _ in ()).throw(ConnectionError("refused"))  # type: ignore[method-assign]
    adapter = OllamaAdapter(client=client, host="http://gpu-box:11434")

    with pytest.raises(BackendUnavailableError, match="gpu-box:11434"):
        adapter.chat(ChatRequest(model="m", messages=[]))
    with pytest.raises(BackendUnavailableError):
        adapter.embed(EmbeddingRequest(model="m", input=["x"]))


def test_ollama_response_errors_are_classified():
    client = FakeOllamaClient()
    adapter = OllamaAdapter(client=client)

    def chat(**kwargs: Any) -> Any:
        raise FakeOllamaClient.ResponseError(400) if kwargs["model"] == "bad" else FakeOllamaClient.ResponseError(500)

    client.chat = chat  # type: ignore[method-assign]

    with pytest.raises(InvalidRequestError, match="Ollama error"):
        adapter.chat(ChatRequest(model="bad", messages=[]))
    with pytest.raises(BackendUnavailableError, match="Ollama error"):
        adapter.chat(ChatRequest(model="oom", messages=[]))


def test_ollama_stream_errors_surface_at_call_time():
    client = FakeOllamaClient()

    def failing_stream(**kwargs: Any) -> Iterator[dict[str, Any]]:
        raise ConnectionError("refused")
        yield {}  # pragma: no cover

    client.chat = failing_stream  # type: ignore[method-assign]
    adapter = OllamaAdapter(client=client)

    with pytest.raises(BackendUnavailableError):
        adapter.chat(ChatRequest(model="m", messages=[], stream=True))


async def test_ollama_async_stream_errors_surface_at_call_time():
    class FailingAsyncClient:
        async def chat(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            async def stream() -> AsyncIterator[dict[str, Any]]:
                raise ConnectionError("refused")
                yield {}  # pragma: no cover

            return stream()

    adapter = OllamaAdapter(client=FakeOllamaClient(), async_client_factory=FailingAsyncClient)

    with pytest.raises(BackendUnavailableError):
        await adapter.achat(ChatRequest(model="m", messages=[], stream=True))


def test_ollama_host_configures_real_clients():
    adapter = OllamaAdapter(host="http://gpu-box:11434", timeout=30)

    assert adapter.host == "http://gpu-box:11434"
    assert str(adapter._client._client.base_url).startswith("http://gpu-box:11434")


def test_llamacpp_uses_local_cache_before_the_network():
    calls: list[bool] = []
    cached = {"hit": True}

    def download(**kwargs: Any) -> str:
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"] and not cached["hit"]:
            raise FileNotFoundError("not in cache")
        return f"/cache/{kwargs['filename']}"

    adapter = LlamaCPPAdapter(llama_factory=FakeLlama, hub_download=download)
    source = ModelSource(filename="m.gguf")

    assert adapter.pull_model("o/m", source) == "/cache/m.gguf"
    assert calls == [True]  # cache hit: no network request

    calls.clear()
    cached["hit"] = False
    assert adapter.pull_model("o/m", source) == "/cache/m.gguf"
    assert calls == [True, False]  # cache miss: download

    calls.clear()
    with pytest.raises(ModelNotDownloadedError) as excinfo:
        adapter.pull_model("o/m", ModelSource(filename="m.gguf", local_files_only=True))
    assert calls == [True]  # offline mode never touches the network
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_ollama_local_files_only_never_pulls_missing_models():
    client = FakeOllamaClient()
    pulled: list[str] = []

    def show(model: str) -> None:
        raise FakeOllamaClient.ResponseError(404)

    client.show = show  # type: ignore[method-assign]
    client.pull = pulled.append  # type: ignore[attr-defined]
    adapter = OllamaAdapter(client=client)

    with pytest.raises(ModelNotDownloadedError):
        adapter.chat(ChatRequest(model="huge:405b", messages=[], model_source=ModelSource(local_files_only=True)))
    with pytest.raises(ModelNotDownloadedError):
        adapter.embed(
            EmbeddingRequest(model="huge:405b", input=("x",), model_source=ModelSource(local_files_only=True))
        )
    assert pulled == []

    adapter.chat(ChatRequest(model="huge:405b", messages=[]))  # downloads stay on by default for the SDK
    assert pulled == ["huge:405b"]


@pytest.mark.parametrize(
    ("n_ctx", "n_gpu_layers", "expected"),
    [
        (None, None, {}),  # unset: Ollama decides
        (8192, 17, {"num_ctx": 8192, "num_gpu": 17}),  # fit's split plan
        (None, 0, {"num_gpu": 0}),  # CPU only
        (None, -1, {"num_gpu": 999}),  # llama.cpp's "all layers", not Ollama's "auto"
    ],
)
def test_ollama_maps_runtime_options_to_num_ctx_and_num_gpu(n_ctx, n_gpu_layers, expected):
    from omnillm.core.types import RuntimeOptions

    client = FakeOllamaClient()
    runtime = RuntimeOptions(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

    OllamaAdapter(client=client).chat(ChatRequest(model="m", messages=[], temperature=0.1, runtime=runtime))

    assert client.chat_kwargs is not None
    assert client.chat_kwargs["options"] == {"temperature": 0.1, **expected}


async def test_ollama_embeddings_forward_runtime_options_only_when_set():
    from omnillm.core.types import RuntimeOptions

    class Client(FakeOllamaClient):
        def embed(self, **kwargs: Any) -> dict[str, Any]:
            self.embed_kwargs = kwargs
            return {"embeddings": [[0.1]]}

    class AsyncClient(FakeAsyncOllamaClient):
        async def embed(self, **kwargs: Any) -> dict[str, Any]:
            self.embed_kwargs = kwargs
            return {"embeddings": [[0.1]]}

    client, async_client = Client(), AsyncClient()
    adapter = OllamaAdapter(client=client, async_client_factory=lambda: async_client)

    adapter.embed(EmbeddingRequest(model="e", input=("x",)))
    assert "options" not in client.embed_kwargs  # older clients without an options parameter keep working

    adapter.embed(EmbeddingRequest(model="e", input=("x",), runtime=RuntimeOptions(n_ctx=2048)))
    assert client.embed_kwargs["options"] == {"num_ctx": 2048}

    await adapter.aembed(EmbeddingRequest(model="e", input=("x",), runtime=RuntimeOptions(n_gpu_layers=4)))
    assert async_client.embed_kwargs is not None and async_client.embed_kwargs["options"] == {"num_gpu": 4}


def test_ollama_alias_runtime_options_reach_the_server():
    from omnillm import LocalLLMManager

    client = FakeOllamaClient()
    manager = LocalLLMManager({"ollama": OllamaAdapter(client=client)})
    manager.register_alias("qwen", backend="ollama", model="qwen2.5:7b", n_ctx=8192, n_gpu_layers=16)

    manager.chat(model="qwen", messages=[{"role": "user", "content": "hi"}])

    assert client.chat_kwargs is not None
    assert client.chat_kwargs["options"] == {"num_ctx": 8192, "num_gpu": 16}
