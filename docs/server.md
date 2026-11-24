# API Server

Launch an OpenAI-compatible local API server. It loads `omnillm.toml` like the CLI, so clients can request any alias (`"model": "coder"`), any `backend/model` reference, or a bare model name for the default backend. One endpoint fronts all your local servers, with fallbacks:

```bash
omnillm serve
# Or: python -m omnillm.server
# Server runs on http://localhost:8000
```

For containerized deployment with Docker or Docker Compose, see the [Docker deployment guide](docker.md).

## Use with the OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Chat completions
response = client.chat.completions.create(
    model="ollama/llama3",          # prefix with backend name
    messages=[{"role": "user", "content": "Explain relativity in one sentence."}]
)
print(response.choices[0].message.content)

# Structured Outputs (JSON Schema)
structured_res = client.chat.completions.create(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Extract: Alice, age 29"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "User",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    },
)
print(structured_res.choices[0].message.content)

# Embeddings
embedding_res = client.embeddings.create(
    model="ollama/nomic-embed-text",
    input=["Embed this chunk for vector search", "Another document"]
)
print(embedding_res.data[0].embedding)
```

## Server notes

- **llama.cpp models** need a GGUF filename, passed as a non-standard field (`extra_body` in the OpenAI SDK):

  ```python
  client.chat.completions.create(
      model="llama.cpp/unsloth/llama-3-8b-Instruct-GGUF",
      messages=[{"role": "user", "content": "Hi"}],
      extra_body={"filename": "llama-3-8b-Instruct-Q4_K_M.gguf"},
  )
  ```

  An in-process llama.cpp model handles one request at a time; concurrent requests are queued.
- **Images** must be sent inline as base64 or `data:image/...;base64,...` URIs. Local file paths and remote URLs are rejected so API clients can never make the server read its own files or fetch arbitrary URLs.
- **`tool_choice`**: `"auto"` and `"none"` are supported; forcing a specific tool is not possible with local backends and returns a 400.
- `seed`, `max_completion_tokens`, and `stream_options: {"include_usage": true}` are supported.
- **Errors** follow the OpenAI shape (`{"error": {"message", "type", "code"}}`) with a fixed, server-written message and a request id (also in the `x-request-id` header). Exception details, which can include upstream backend output, are logged server-side under that id and never sent to clients.
- **Host check (DNS-rebinding protection)**: only requests whose `Host` header is a loopback name are accepted by default, so a malicious web page can't reach your local server through DNS rebinding. When serving other machines, list the names clients use: `omnillm serve --host 0.0.0.0 --allowed-host gpu-box --allowed-host 192.168.1.20` (or set `OMNILLM_ALLOWED_HOSTS`; `'*'` disables the check, e.g. behind a reverse proxy).
- **Health check**: `GET /health` returns `{"status": "ok"}` without querying model backends, making it suitable for container orchestration and uptime monitors.
- **Docker**: For running inside a container, see [Docker deployment](docker.md).
- There is no authentication yet: don't expose the server to untrusted networks.

## Use with curl

```bash
# Chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Structured Outputs with JSON Schema
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3",
    "messages": [{"role": "user", "content": "Alice is 29 years old."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "User",
        "schema": {
          "type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"]
        }
      }
    }
  }'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/nomic-embed-text",
    "input": ["Local vector embeddings with Omni-Local-LLM"]
  }'
```

---

[← Back to README](../README.md)
