from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
from omnillm.adapters.ollama_adapter import OllamaAdapter
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
