# Backends, Configuration & Model Aliases

## Any OpenAI-compatible server

Most local inference servers speak the OpenAI API. Point Omni-Local-LLM at one and use it like any other backend:

| Server | Default `base_url` |
|---|---|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1` |
| [llama.cpp `llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | `http://localhost:8080/v1` |
| [vLLM](https://docs.vllm.ai/) | `http://localhost:8000/v1` |
| [SGLang](https://docs.sglang.ai/) | `http://localhost:30000/v1` |
| [KoboldCpp](https://github.com/LostRuins/koboldcpp) | `http://localhost:5001/v1` |
| [Ollama](https://ollama.com/) (OpenAI endpoint) | `http://localhost:11434/v1` |

`omnillm serve` also defaults to port 8000, so pass `--port` if you run it next to vLLM.

```bash
# No config needed for a quick try
omnillm chat --base-url http://localhost:1234/v1 --model qwen3-8b
```

```python
from omnillm import LocalLLMManager, OpenAICompatibleAdapter

manager = LocalLLMManager()
manager.register_backend("lmstudio", lambda: OpenAICompatibleAdapter("http://localhost:1234/v1", name="lmstudio"))
response = manager.chat(model="lmstudio/qwen3-8b", messages=[{"role": "user", "content": "Hi!"}])
```

Streaming (with token usage), tool calling, structured outputs, vision, and embeddings all work through it. Unreachable servers, timeouts, auth failures and 5xx responses raise `BackendUnavailableError`; bad requests (unknown model, context too long) raise `InvalidRequestError` with the server's message.

## `omnillm.toml`: named backends and model aliases

Describe your servers and favourite models once. The CLI and `omnillm serve` load the first file found among `--config`, `$OMNILLM_CONFIG`, `./omnillm.toml`, and `~/.config/omnillm/config.toml`; in Python use `LocalLLMManager.from_config()`.

```toml
default_backend = "ollama"          # where bare model names go

[backends.lmstudio]
type = "openai"                     # any OpenAI-compatible server
base_url = "http://localhost:1234/v1"
# api_key_env = "LMSTUDIO_API_KEY"  # or api_key = "..."; also: headers, timeout, capabilities

[backends.gpu-box]
type = "ollama"                     # a second Ollama server
host = "http://gpu-box:11434"

[models.coder]
backend = "llama.cpp"
model = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
n_ctx = 8192
temperature = 0.2
fallbacks = ["gpu-box/qwen2.5-coder:7b", "fast"]   # tried in order if a backend is unavailable

[models.fast]
model = "lmstudio/qwen3-4b"         # 'backend/model' works too
```

```python
manager = LocalLLMManager.from_config()
manager.chat(model="coder", messages=[...])                  # alias: backend, GGUF file, n_ctx, temperature preset
manager.chat(model="coder", messages=[...], temperature=1)   # explicit arguments override alias presets
manager.chat(model="gpu-box/llama3.2:3b", messages=[...])    # 'backend/model' reference
session = manager.create_session(model="coder")              # aliases resolve on every turn, so fallbacks apply
```

`n_ctx` and `n_gpu_layers` work for Ollama models too: they're sent as Ollama's `num_ctx` and `num_gpu`, so an alias can run a model with the context and GPU layer count that `omnillm fit` recommends (`n_gpu_layers = -1` means all layers, as in llama.cpp; leave it out to let Ollama decide):

```toml
[models.qwen]
model = "ollama/qwen2.5:7b"
n_ctx = 8192
n_gpu_layers = 16
```

Model references resolve as: **alias** → **`backend/model`** (if the prefix is a known backend) → **bare name on `default_backend`**. So namespaced Ollama models such as `huihui_ai/qwen3` keep working. The config is validated strictly on load, so a typo like `temprature` fails with the file, table, and allowed keys instead of being silently ignored.

Fallbacks trigger only when a backend is unavailable or lacks a needed capability (e.g. vision), never on a bad request. If every target fails, the primary's error is raised with a note per attempted target. Aliases work everywhere: SDK, sessions, CLI (`--model coder`, `bench --targets coder,fast`), and the API server (`"model": "coder"`; aliases are listed in `/v1/models`).

---

[← Back to README](../README.md)
