import json

import pytest
from fastapi.testclient import TestClient

from omnillm import ChatChunk, ChatResponse, LocalLLMManager, Usage
from omnillm.core.errors import (
    BackendUnavailableError,
    InvalidRequestError,
    ModelNotDownloadedError,
    ModelNotFoundError,
    UnsupportedFeatureError,
)
from omnillm.server import create_app
from tests.fakes import RecordingBackend


@pytest.fixture(autouse=True)
def _no_server_env(monkeypatch):
    """Keeps a developer's own server settings from leaking into these tests."""
    for name in ("OMNILLM_API_KEY", "OMNILLM_ALLOW_DOWNLOADS", "OMNILLM_ALLOWED_HOSTS"):
        monkeypatch.delenv(name, raising=False)


def local_client(app) -> TestClient:
    return TestClient(app, base_url="http://localhost")


def make_client(backend: RecordingBackend) -> TestClient:
    return local_client(create_app(LocalLLMManager({"fake": backend})))


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


def test_chat_completions_json_schema_response_format():
    backend = RecordingBackend(response=ChatResponse(content='{"result": 123}'))
    client = make_client(backend)

    schema = {"type": "object", "properties": {"result": {"type": "integer"}}}
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/test",
            "messages": [{"role": "user", "content": "compute"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Result", "schema": schema},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == '{"result": 123}'
    request = backend.requests[0]
    assert request.json_mode is True
    assert request.response_format is not None
    assert request.response_format["type"] == "json_schema"
    assert request.json_schema == schema


def test_models_endpoint_lists_models_from_available_backends():
    client = make_client(RecordingBackend(models=["one", "two"]))

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["fake/one", "fake/two"]


def test_health_endpoint_does_not_query_backends(monkeypatch):
    manager = LocalLLMManager({"fake": RecordingBackend()})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Health check must not query backends")

    monkeypatch.setattr(manager, "list_models", fail_if_called)

    client = local_client(create_app(manager))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_respects_host_validation():
    manager = LocalLLMManager({"fake": RecordingBackend()})
    app = create_app(manager)

    evil_client = TestClient(app, base_url="http://attacker.example")
    assert evil_client.get("/health").status_code == 400

    good_client = TestClient(app, base_url="http://localhost")
    assert good_client.get("/health").status_code == 200
    assert good_client.get("/health").json() == {"status": "ok"}


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


def test_multimodal_chat_completions():
    backend = RecordingBackend(response=ChatResponse(content="I see a cat."))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/llava",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "I see a cat."
    assert len(backend.requests[0].messages) == 1
    assert isinstance(backend.requests[0].messages[0]["content"], list)


def test_server_never_reads_local_files_named_in_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "abcd").write_bytes(b"top secret")
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)

    rejected = client.post(
        "/v1/chat/completions",
        json={"model": "fake/m", "messages": [{"role": "user", "content": "x", "images": [str(tmp_path / "abcd")]}]},
    )
    accepted = client.post(
        "/v1/chat/completions",
        json={"model": "fake/m", "messages": [{"role": "user", "content": "x", "images": ["abcd"]}]},
    )
    remote = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/m",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://169.254.169.254/"}}]}
            ],
        },
    )

    assert rejected.status_code == 400
    assert remote.status_code == 400
    assert accepted.status_code == 200
    assert len(backend.requests) == 1
    assert backend.requests[0].messages[0]["images"] == ["data:image/jpeg;base64,abcd"]


def test_server_forwards_llamacpp_model_source_and_seed():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/owner/repo-GGUF",
            "messages": [{"role": "user", "content": "hi"}],
            "filename": "model-Q4_K_M.gguf",
            "mmproj_filename": "mmproj.gguf",
            "seed": 42,
            "max_completion_tokens": 9,
        },
    )

    assert response.status_code == 200
    request = backend.requests[0]
    assert request.model == "owner/repo-GGUF"
    assert request.model_source.filename == "model-Q4_K_M.gguf"
    assert request.model_source.mmproj_filename == "mmproj.gguf"
    assert request.seed == 42
    assert request.max_tokens == 9


def test_server_tool_choice_none_drops_tools_and_forced_choice_is_rejected():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    client = make_client(backend)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    base = {"model": "fake/m", "messages": [{"role": "user", "content": "hi"}], "tools": tools}

    none = client.post("/v1/chat/completions", json={**base, "tool_choice": "none"})
    auto = client.post("/v1/chat/completions", json={**base, "tool_choice": "auto"})
    required = client.post("/v1/chat/completions", json={**base, "tool_choice": "required"})

    assert none.status_code == 200 and backend.requests[0].tools is None
    assert auto.status_code == 200 and backend.requests[1].tools == tools
    assert required.status_code == 400
    assert "tool_choice" in required.json()["error"]["message"]


def test_streaming_tool_calls_have_indexes_finish_reason_and_optional_usage():
    call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    backend = RecordingBackend(
        chunks=(
            ChatChunk(tool_calls=(call,)),
            ChatChunk(finish_reason="stop", usage=Usage(5, 3)),
        )
    )
    client = make_client(backend)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_delta = events[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["index"] == 0
    assert tool_delta["id"] == "call_1"
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_server_routes_aliases_and_lists_them():
    backend = RecordingBackend(response=ChatResponse(content="ok"), models=["llama3"])
    manager = LocalLLMManager({"fake": backend}, default_backend="fake")
    manager.register_alias("coder", backend="fake", model="qwen-coder", temperature=0.2)
    client = local_client(create_app(manager))

    chat = client.post("/v1/chat/completions", json={"model": "coder", "messages": [{"role": "user", "content": "x"}]})
    namespaced = client.post(
        "/v1/chat/completions", json={"model": "huihui_ai/qwen3", "messages": [{"role": "user", "content": "x"}]}
    )
    models = client.get("/v1/models").json()["data"]

    assert chat.status_code == 200 and chat.json()["model"] == "coder"
    assert (backend.requests[0].model, backend.requests[0].temperature) == ("qwen-coder", 0.2)
    assert namespaced.status_code == 200 and backend.requests[1].model == "huihui_ai/qwen3"
    assert {"id": "coder", "object": "model", "created": 0, "owned_by": "alias:fake/qwen-coder"} in models
    assert any(m["id"] == "fake/llama3" for m in models)


def test_server_module_app_is_built_lazily(monkeypatch, tmp_path):
    import omnillm.server as server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OMNILLM_CONFIG", raising=False)
    monkeypatch.setattr(server, "_default_app", None)

    assert server.app is server.app


SECRET = "Traceback (most recent call last): File /srv/internal/secret.py"


@pytest.mark.parametrize(
    ("error", "status", "code", "fragment"),
    [
        (InvalidRequestError(SECRET), 400, "invalid_request", "backend rejected the request"),
        (ModelNotFoundError(SECRET), 404, "model_not_found", "not found on its backend"),
        (ModelNotDownloadedError(SECRET), 404, "model_not_downloaded", "--allow-downloads"),
        (UnsupportedFeatureError(SECRET, feature="tools"), 400, "unsupported_feature", "tool calling"),
        (UnsupportedFeatureError(SECRET, feature="new-thing"), 400, "unsupported_feature", "a feature used"),
        (BackendUnavailableError(SECRET), 503, "backend_unavailable", "currently unavailable"),
        (ValueError(SECRET), 400, "invalid_request", "Invalid request parameters"),
        (RuntimeError(SECRET), 500, "internal_error", "internal error"),
    ],
)
def test_errors_never_echo_exception_text(error, status, code, fragment, caplog):
    client = make_client(RecordingBackend(error=error))

    with caplog.at_level("WARNING", logger="omnillm.server"):
        response = client.post(
            "/v1/chat/completions", json={"model": "fake/m", "messages": [{"role": "user", "content": "x"}]}
        )

    body = response.json()["error"]
    request_id = response.headers["x-request-id"]
    assert response.status_code == status
    assert body["code"] == code and fragment in body["message"]
    assert "secret" not in response.text and "Traceback" not in response.text
    assert request_id in body["message"]
    assert request_id in caplog.text and "secret.py" in caplog.text  # details stay in the server log


def test_mid_stream_errors_never_echo_exception_text():
    class FailingStream(RecordingBackend):
        async def achat(self, request):
            async def stream():
                yield ChatChunk(content="partial")
                raise InvalidRequestError(SECRET)

            return stream()

    response = make_client(FailingStream()).post(
        "/v1/chat/completions", json={"model": "fake/m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    )

    assert "partial" in response.text and "backend rejected the request" in response.text
    assert "secret" not in response.text and response.text.rstrip().endswith("data: [DONE]")


def test_server_written_validation_messages_stay_specific():
    client = make_client(RecordingBackend())
    remote_image = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/m",
            "messages": [{"role": "user", "content": "x", "images": ["https://example.com/a.png"]}],
        },
    )

    assert remote_image.status_code == 400
    assert remote_image.json()["error"]["code"] == "invalid_image"
    assert "remote URLs are not accepted" in remote_image.json()["error"]["message"]


def test_host_header_is_checked_to_block_dns_rebinding(monkeypatch):
    monkeypatch.delenv("OMNILLM_ALLOWED_HOSTS", raising=False)
    manager = LocalLLMManager({"fake": RecordingBackend(models=["m"])})

    default = create_app(manager)
    assert TestClient(default, base_url="http://attacker.example").get("/v1/models").status_code == 400
    local = TestClient(default, base_url="http://localhost")
    for host in ("localhost", "localhost:8000", "127.0.0.1:8000", "[::1]:8000", "[::1]", "LOCALHOST"):
        assert local.get("/v1/models", headers={"host": host}).status_code == 200, host
    for host in ("attacker.example:8000", "localhost.attacker.example", "", "127.0.0.1.nip.io"):
        assert local.get("/v1/models", headers={"host": host}).status_code == 400, host

    custom = create_app(manager, allowed_hosts=["llm.internal", "*.corp.example"])
    assert TestClient(custom, base_url="http://llm.internal").get("/v1/models").status_code == 200
    assert TestClient(custom, base_url="http://gpu.corp.example").get("/v1/models").status_code == 200
    assert TestClient(custom, base_url="http://localhost").get("/v1/models").status_code == 400

    anything = create_app(manager, allowed_hosts=["*"])
    assert TestClient(anything, base_url="http://attacker.example").get("/v1/models").status_code == 200

    monkeypatch.setenv("OMNILLM_ALLOWED_HOSTS", "gpu-box, 10.0.0.5")
    from_env = create_app(manager)
    assert TestClient(from_env, base_url="http://10.0.0.5").get("/v1/models").status_code == 200
    assert TestClient(from_env, base_url="http://localhost").get("/v1/models").status_code == 400


def test_logged_error_text_cannot_forge_log_lines(caplog):
    client = make_client(RecordingBackend(error=InvalidRequestError("bad\nINFO forged entry\r\nmore")))

    with caplog.at_level("WARNING", logger="omnillm.server"):
        client.post("/v1/chat/completions", json={"model": "fake/m", "messages": [{"role": "user", "content": "x"}]})

    record = next(r for r in caplog.records if r.name == "omnillm.server")
    assert "\n" not in record.getMessage() and "\r" not in record.getMessage()
    assert "forged entry" in record.getMessage()


CHAT = {"model": "fake/m", "messages": [{"role": "user", "content": "x"}]}


def test_api_key_is_required_on_every_endpoint_except_health(monkeypatch):
    monkeypatch.delenv("OMNILLM_API_KEY", raising=False)
    client = local_client(create_app(LocalLLMManager({"fake": RecordingBackend(models=["m"])}), api_key="s3cret"))

    for headers in (
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "s3cret"},
        {"Authorization": "Basic s3cret"},
    ):
        for response in (
            client.get("/v1/models", headers=headers),
            client.post("/v1/chat/completions", json=CHAT, headers=headers),
            client.post("/v1/embeddings", json={"model": "fake/m", "input": "x"}, headers=headers),
            client.get("/openapi.json", headers=headers),
        ):
            assert response.status_code == 401, (headers, response.request.url)
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.json()["error"]["code"] == "invalid_api_key"

    authorized = {"Authorization": "Bearer s3cret"}
    assert client.get("/v1/models", headers=authorized).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "bearer s3cret"}).status_code == 200
    assert client.post("/v1/chat/completions", json=CHAT, headers=authorized).status_code == 200
    assert client.get("/health").status_code == 200  # container health checks need no key


def test_api_key_works_with_the_openai_sdk_header_and_env(monkeypatch):
    manager = LocalLLMManager({"fake": RecordingBackend(models=["m"])})
    monkeypatch.setenv("OMNILLM_API_KEY", " from-env ")

    client = local_client(create_app(manager))
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer from-env"}).status_code == 200

    assert local_client(create_app(manager, api_key="")).get("/v1/models").status_code == 200  # explicit opt-out


def test_host_check_runs_before_the_api_key_check():
    app = create_app(LocalLLMManager({"fake": RecordingBackend()}), api_key="k")

    response = TestClient(app, base_url="http://attacker.example").get(
        "/v1/models", headers={"Authorization": "Bearer k"}
    )

    assert response.status_code == 400 and response.text == "Invalid host header"


def test_api_requests_never_download_models_by_default(monkeypatch):
    monkeypatch.delenv("OMNILLM_ALLOW_DOWNLOADS", raising=False)
    backend = RecordingBackend()
    manager = LocalLLMManager({"fake": backend})
    manager.register_alias("coder", backend="fake", model="m", filename="m.gguf", local_files_only=False)
    client = local_client(create_app(manager))

    client.post("/v1/chat/completions", json={**CHAT, "model": "coder"})
    client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    client.post("/v1/embeddings", json={"model": "coder", "input": "x"})

    sources = [r.model_source for r in backend.requests] + [r.model_source for r in backend.embedding_requests]
    assert len(sources) == 3
    assert all(source.local_files_only for source in sources)  # even over an alias that allows downloads


@pytest.mark.parametrize(
    ("allow_downloads", "env", "expected"),
    [(True, None, False), (None, "1", False), (None, "true", False), (None, "0", True), (False, "1", True)],
)
def test_downloads_can_be_allowed_explicitly(monkeypatch, allow_downloads, env, expected):
    monkeypatch.delenv("OMNILLM_ALLOW_DOWNLOADS", raising=False)
    if env is not None:
        monkeypatch.setenv("OMNILLM_ALLOW_DOWNLOADS", env)
    backend = RecordingBackend()
    client = local_client(create_app(LocalLLMManager({"fake": backend}), allow_downloads=allow_downloads))

    client.post("/v1/chat/completions", json=CHAT)

    assert backend.requests[0].model_source.local_files_only is expected


def test_missing_ollama_model_is_reported_instead_of_pulled():
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    class Client:
        class ResponseError(Exception):
            status_code = 404

        pulled: list[str] = []

        def show(self, model):
            raise self.ResponseError()

        def pull(self, model):
            self.pulled.append(model)

    client = Client()
    app = create_app(LocalLLMManager({"ollama": OllamaAdapter(client=client, async_client_factory=lambda: client)}))

    response = local_client(app).post("/v1/chat/completions", json={**CHAT, "model": "ollama/huge:405b"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_downloaded"
    assert client.pulled == []
