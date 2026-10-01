import pytest

from omnillm import ChatResponse, LocalLLMManager
from omnillm.core.errors import BackendNotFoundError, UnsupportedFeatureError
from omnillm.core.types import BackendCapabilities
from tests.fakes import RecordingBackend


def test_manager_builds_typed_request_and_forwards_generation_options():
    backend = RecordingBackend(response=ChatResponse(content="Hello world!"))
    manager = LocalLLMManager({"fake": backend})

    response = manager.chat(
        backend="fake",
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.3,
        max_tokens=12,
        top_p=0.8,
        stop=["END"],
        filename="model.gguf",
        revision="main",
        n_gpu_layers=20,
        n_ctx=4096,
    )

    assert response.content == "Hello world!"
    request = backend.requests[0]
    assert request.temperature == 0.3
    assert request.max_tokens == 12
    assert request.top_p == 0.8
    assert request.stop == ["END"]
    assert request.model_source.filename == "model.gguf"
    assert request.model_source.revision == "main"
    assert request.runtime.n_gpu_layers == 20
    assert request.runtime.n_ctx == 4096


@pytest.mark.asyncio
async def test_manager_async_uses_the_same_request_contract():
    backend = RecordingBackend(response=ChatResponse(content="Async hello"))
    manager = LocalLLMManager({"fake": backend})

    response = await manager.achat("fake", "test-model", [{"role": "user", "content": "hi"}], temperature=0.2)

    assert response.content == "Async hello"
    assert backend.requests[0].temperature == 0.2


def test_manager_rejects_unknown_backend():
    with pytest.raises(BackendNotFoundError):
        LocalLLMManager().chat(backend="invalid", model="test", messages=[])


def test_manager_validates_adapter_capabilities():
    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(tools=False)
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(UnsupportedFeatureError):
        manager.chat("fake", "test", [], tools=[{"type": "function"}])
