<h1 align="center">omni-local-llm</h1>

<p align="center">
  <b>Which local model can your machine actually run?</b><br>
  Find out in one command, download the right file, and serve every local LLM behind one OpenAI-compatible API.
</p>

<p align="center">
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/v/omni-local-llm?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/pyversions/omni-local-llm" alt="Python"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/matteoespo/omni-local-llm?color=green" alt="License"></a>
</p>

<p align="center">
  <img src="docs/assets/demo.svg" width="100%" alt="Terminal demo: omnillm fit shows which quantization of Qwen3-8B fits a 4 GB GPU (UD-Q4_K_XL with 17 of 36 layers on the GPU), then downloads a small model with --pull --save-alias and runs it">
</p>

## Try it (no install)

```bash
# Which quant of this model fits my GPU, and how many layers go on it?
uvx omni-local-llm fit unsloth/Qwen3-8B-GGUF

# Download the recommendation and save it as an alias
uvx omni-local-llm fit unsloth/Qwen3-8B-GGUF --pull --save-alias qwen3
```

Then run it with the llama.cpp backend (`llama-cpp-python` compiles on first install):

```bash
pip install "omni-local-llm[llama-cpp]"
omnillm chat --model qwen3
```

`uvx` comes with [uv](https://docs.astral.sh/uv/). `pipx run omni-local-llm …` works too.

## What it does

### 1. `fit`: the right file for your hardware

Pass a Hugging Face repo, an installed Ollama model, or a `.gguf` file. `fit` reads the model's real architecture and file sizes (GGUF headers via HTTP range requests, so no download) and checks every quantization against your RAM and GPU:

- **KV cache from the real shape**, so grouped-query attention, MLA and `--kv-cache q8_0` are accounted for
- **Where it runs**: fully on GPU, Apple Silicon unified memory, **split GPU + CPU with a concrete `n_gpu_layers`**, CPU only, or not at all
- **MoE aware**: total vs. active parameters, with tips for keeping experts on CPU
- **`--pull` and `--save-alias`**: download the recommended quant and get a ready-to-run alias

Tested on a 4 GB GTX 1650: predicted VRAM was within 0.09–0.37 GB of measured use (always on the safe side), and every recommended layer count loaded. [See the measurements](docs/fit.md#accuracy).

→ [Hardware fit docs](docs/fit.md)

### 2. One API for every local server

Ollama, in-process llama.cpp, and **any OpenAI-compatible server** (vLLM, llama.cpp's `llama-server`, LM Studio, SGLang, KoboldCpp, MLX-LM, LocalAI) behind one interface. Name models once in `omnillm.toml`, and fall back automatically when a server is down:

```toml
[backends.lmstudio]
type = "openai"
base_url = "http://localhost:1234/v1"

[models.coder]
backend = "llama.cpp"
model = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
n_ctx = 8192
fallbacks = ["lmstudio/qwen2.5-coder-7b-instruct", "ollama/llama3.2:3b"]
```

`omnillm serve` then gives every tool you use one OpenAI-compatible endpoint, where `"model": "coder"` just works. Set `OMNILLM_API_KEY` to require a key, and API clients can only use models that are already downloaded.

→ [Backends & config](docs/backends.md) · [API server](docs/server.md)

### 3. A Python SDK for the fiddly parts

```python
from pydantic import BaseModel
from omnillm import LocalLLMManager, tool

manager = LocalLLMManager.from_config()          # reads omnillm.toml

class City(BaseModel):
    name: str
    country: str

city = manager.chat(model="coder", messages=[{"role": "user", "content": "Lyon is in France."}],
                    response_model=City).parsed   # schema enforced at the sampler

@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 21}

session = manager.create_session(model="coder")
print(session.act("What's the weather in Oslo?", tools=[get_weather]).content)  # runs the tool loop
```

Also streaming (sync and async), chat sessions with context-window pruning, vision inputs, and embeddings.

→ [SDK docs](docs/sdk.md)

### 4. Benchmarks and evals on your machine

```bash
omnillm bench --targets coder,ollama/llama3.2:3b     # time to first token, tokens/sec
omnillm eval --model coder --suite my_tests.jsonl    # regex, JSON-schema and tool-call checks
omnillm eval --model coder --needle                  # needle-in-a-haystack retrieval
```

→ [Harness docs](docs/harness.md)

## How is this different from…?

- **Ollama / LM Studio / llama.cpp**: they run models. omnillm works *with* them, choosing the right file, putting several servers behind one API with fallbacks, and adding SDK features on top. It doesn't replace them.
- **LiteLLM**: a gateway for 100+ (mostly cloud) providers. omnillm is local-first: it knows your hardware and GGUF files, can download and run models in-process, and enforces structured output with grammars. If you mostly route between cloud APIs, use LiteLLM.
- **"Can I run it?" calculators**: they estimate from parameter counts. `fit` reads the actual files and headers of the repo you name, accounts for your free VRAM, and downloads the file it recommends.

## Install

```bash
pip install "omni-local-llm[ollama,server]"   # or: uv pip install ...
```

| Extra | Adds |
|---|---|
| `ollama` | Ollama backend |
| `llama-cpp` | In-process llama.cpp backend and `fit --pull` (Hugging Face downloads) |
| `openai` | OpenAI-compatible servers (just `httpx`) |
| `server` | `omnillm serve` (FastAPI) |
| `all` | Everything |

The core package depends only on `httpx`, and `fit` works without any extras. Requires Python 3.12+.

## Documentation

| | |
|---|---|
| [Hardware fit](docs/fit.md) | Sources, KV cache, placement, `--pull`, `--save-alias`, SDK |
| [Backends & config](docs/backends.md) | OpenAI-compatible servers, `omnillm.toml`, aliases, fallbacks |
| [Python SDK](docs/sdk.md) | Chat, sessions, streaming, tools, structured outputs, embeddings, vision |
| [CLI](docs/cli.md) | `chat`, `run`, `models`, `bench`, `eval`, `serve`, `fit` |
| [API server](docs/server.md) | OpenAI SDK and curl usage, errors, host check |
| [Docker deployment](docs/docker.md) | Dockerfile, docker-compose, config mounting, host Ollama, security |
| [Benchmarks & evals](docs/harness.md) | `BenchmarkHarness`, `EvalHarness`, `NeedleHarness` |
| [Architecture](docs/architecture.md) | Module map, running tests |

## Roadmap

The live roadmap is in the pinned [Roadmap & feedback](https://github.com/matteoespo/omni-local-llm/issues/31) issue.

- [ ] `omnillm pick`: rank the models that fit your machine on *your own* prompts (quality vs. speed vs. memory)
- [ ] Server API keys, and turning off model downloads for HTTP requests
- [ ] AMD ROCm / Intel GPU detection in `fit` ([#26](https://github.com/matteoespo/omni-local-llm/issues/26), help wanted)
- [x] Docker image for the API server ([#32](https://github.com/matteoespo/omni-local-llm/issues/32)) ([docs](docs/docker.md))
- [ ] A web version of `fit`

Ideas and bug reports are very welcome: [open an issue](https://github.com/matteoespo/omni-local-llm/issues/new/choose), ask in [Discussions](https://github.com/matteoespo/omni-local-llm/discussions), or pick up a [good first issue](https://github.com/matteoespo/omni-local-llm/labels/good%20first%20issue).

## Contributing

See the [Contributing Guide](CONTRIBUTING.md) for setup, adding backend adapters, and submitting pull requests.

## License

MIT. See [LICENSE](LICENSE).
