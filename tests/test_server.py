import json

from fastapi.testclient import TestClient

from omnillm import ChatChunk, ChatResponse, LocalLLMManager, Usage
from omnillm.server import create_app
from tests.fakes import RecordingBackend


def make_client(backend: RecordingBackend) -> TestClient:
    return TestClient(create_app(LocalLLMManager({"fake": backend})))


def test_chat_completions_forwards_parameters_and_returns_usage():
    backend = RecordingBackend(response=ChatResponse(content="Mock response", usage=Usage(3, 2)))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "max_tokens": 20,
            "top_p": 0.8,
            "stop": ["END"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("chatcmpl-")
    assert payload["choices"][0]["message"]["content"] == "Mock response"
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    request = backend.requests[0]
    assert request.temperature == 0.3
    assert request.max_tokens == 20
    assert request.top_p == 0.8
    assert request.stop == ["END"]


def test_models_endpoint_lists_models_from_available_backends():
    client = make_client(RecordingBackend(models=["one", "two"]))

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["fake/one", "fake/two"]


def test_streaming_response_has_role_chunks_finish_reason_and_done_marker():
    backend = RecordingBackend(chunks=(ChatChunk(content="Hello"), ChatChunk(content="!", finish_reason="stop")))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake/test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 200
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0])["choices"][0]["delta"] == {"role": "assistant"}
    assert json.loads(events[1])["choices"][0]["delta"] == {"content": "Hello"}
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"


def test_request_errors_are_openai_shaped():
    client = make_client(RecordingBackend())

    unknown_backend = client.post(
        "/v1/chat/completions",
        json={"model": "unknown/test", "messages": [{"role": "user", "content": "hi"}]},
    )
    unsupported_n = client.post(
        "/v1/chat/completions",
        json={"model": "fake/test", "messages": [{"role": "user", "content": "hi"}], "n": 2},
    )

    assert unknown_backend.status_code == 404
    assert unknown_backend.json()["error"]["type"] == "invalid_request_error"
    assert unsupported_n.status_code == 400
    assert unsupported_n.json()["error"]["message"] == "Only n=1 is supported."


def test_embeddings_endpoint_returns_openai_format():
    from omnillm.core.types import EmbeddingData, EmbeddingResponse

    backend = RecordingBackend(
        embedding_response=EmbeddingResponse(
            data=(
                EmbeddingData(index=0, embedding=(0.1, 0.2)),
                EmbeddingData(index=1, embedding=(0.3, 0.4)),
            ),
            usage=Usage(prompt_tokens=6),
        )
    )
    client = make_client(backend)

    response = client.post(
        "/v1/embeddings",
        json={"model": "fake/nomic-embed", "input": ["first", "second"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "fake/nomic-embed"
    assert len(payload["data"]) == 2
    assert payload["data"][0] == {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
    assert payload["data"][1] == {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
    assert payload["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    assert backend.embedding_requests[0].input == ("first", "second")


def test_embeddings_endpoint_validates_input_and_encoding():
    client = make_client(RecordingBackend())

    bad_encoding = client.post(
        "/v1/embeddings",
        json={"model": "fake/test", "input": "text", "encoding_format": "base64"},
    )
    empty_input = client.post(
        "/v1/embeddings",
        json={"model": "fake/test", "input": []},
    )

    assert bad_encoding.status_code == 400
    assert "Only float encoding_format is supported" in bad_encoding.json()["error"]["message"]
    assert empty_input.status_code == 400
    assert "Embedding input must contain non-empty strings" in empty_input.json()["error"]["message"]
