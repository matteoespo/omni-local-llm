from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
from omnillm.adapters.ollama_adapter import OllamaAdapter
from omnillm.core.types import ChatRequest, ModelSource


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

    def list(self) -> dict[str, Any]:
        return {"models": [{"model": "llama3"}]}


class FakeAsyncOllamaClient:
    def __init__(self):
        self.chat_kwargs: dict[str, Any] | None = None

    async def chat(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.chat_kwargs = kwargs

        async def stream() -> AsyncIterator[dict[str, Any]]:
            yield {"message": {"content": "one"}}
            yield {"message": {"content": "two"}, "done": True}

        return stream()


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
