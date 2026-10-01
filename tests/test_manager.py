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


def test_manager_embed_normalizes_input_and_forwards_options():
    backend = RecordingBackend()
    manager = LocalLLMManager({"fake": backend})

    response = manager.embed(
        backend="fake",
        model="nomic-embed",
        input="hello",
        filename="model.gguf",
        revision="v1",
        n_gpu_layers=10,
        n_ctx=2048,
    )

    assert response.embeddings == [[0.1, 0.2, 0.3]]
    request = backend.embedding_requests[0]
    assert request.model == "nomic-embed"
    assert request.input == ("hello",)
    assert request.model_source.filename == "model.gguf"
    assert request.model_source.revision == "v1"
    assert request.runtime.n_gpu_layers == 10
    assert request.runtime.n_ctx == 2048


@pytest.mark.asyncio
async def test_manager_aembed_forwards_sequence():
    backend = RecordingBackend()
    manager = LocalLLMManager({"fake": backend})

    response = await manager.aembed("fake", "nomic-embed", ["doc 1", "doc 2"])

    assert response.embeddings == [[0.1, 0.2, 0.3]]
    assert backend.embedding_requests[0].input == ("doc 1", "doc 2")


def test_manager_embed_validates_inputs_and_capabilities():
    backend = RecordingBackend()
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(ValueError, match="A model name is required"):
        manager.embed("fake", "", "text")

    with pytest.raises(ValueError, match="Embedding input must contain at least one string"):
        manager.embed("fake", "model", [])

    backend.capabilities = BackendCapabilities(embeddings=False)
    with pytest.raises(UnsupportedFeatureError, match="does not support embeddings"):
        manager.embed("fake", "model", "text")


def test_manager_structured_output_with_response_model():
    from pydantic import BaseModel

    class Character(BaseModel):
        name: str
        level: int

    backend = RecordingBackend(response=ChatResponse(content='{"name": "Arthur", "level": 10}'))
    manager = LocalLLMManager({"fake": backend})

    response = manager.chat(
        backend="fake",
        model="test-model",
        messages=[{"role": "user", "content": "Create character"}],
        response_model=Character,
    )

    assert isinstance(response, ChatResponse)
    assert response.parsed == Character(name="Arthur", level=10)
    assert response.parse_as(Character) == Character(name="Arthur", level=10)
    request = backend.requests[0]
    assert request.json_mode is True
    assert request.response_format is not None
    assert request.response_format["type"] == "json_schema"
    assert request.json_schema is not None
    assert "properties" in request.json_schema


def test_manager_structured_output_parse_error():
    from pydantic import BaseModel

    from omnillm.core.errors import InvalidRequestError

    class Character(BaseModel):
        name: str
        level: int

    backend = RecordingBackend(response=ChatResponse(content="not json"))
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(InvalidRequestError, match="Failed to parse response into Character"):
        manager.chat(
            backend="fake",
            model="test-model",
            messages=[],
            response_model=Character,
        )


def test_manager_vision_validation_unsupported_backend():
    from omnillm.core.types import BackendCapabilities

    backend = RecordingBackend()
    backend.capabilities = BackendCapabilities(vision=False)
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(UnsupportedFeatureError, match="does not support vision"):
        manager.chat(
            backend="fake",
            model="test-model",
            messages=[{"role": "user", "content": "What is this?", "images": ["data:image/png;base64,AAAA"]}],
        )


def test_manager_forwards_vision_options():
    backend = RecordingBackend()
    manager = LocalLLMManager({"fake": backend})

    manager.chat(
        backend="fake",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        mmproj_filename="mmproj.gguf",
        clip_model_path="/path/to/clip",
    )

    request = backend.requests[0]
    assert request.model_source.mmproj_filename == "mmproj.gguf"
    assert request.runtime.clip_model_path == "/path/to/clip"
