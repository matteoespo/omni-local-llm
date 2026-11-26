# Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your Application                   │
├─────────────┬─────────────────────┬─────────────────────┤
│   Python    │   Interactive CLI   │  OpenAI-Compatible  │
│     SDK     │  (python -m omnillm)│   FastAPI Server    │
├─────────────┴─────────────────────┴─────────────────────┤
│                    ChatSession                          │
│              (History · Streaming · Tools)              │
├─────────────────────────────────────────────────────────┤
│                  LocalLLMManager                        │
│     (Backend Registry · Aliases · Router · Fallbacks)   │
├──────────────────┬──────────────────┬───────────────────┤
│  OllamaAdapter   │ LlamaCPPAdapter  │ OpenAICompatible- │
│ (ollama client,  │ (llama-cpp-python│ Adapter (httpx:   │
│  any host)       │  + HF Hub)       │ vLLM, LM Studio…) │
└──────────────────┴──────────────────┴───────────────────┘
```

## Module Map

| Module | Purpose |
|---|---|
| [`omnillm/core/base.py`](../omnillm/core/base.py) | `LLMBackend` — abstract base class for all adapters |
| [`omnillm/core/manager.py`](../omnillm/core/manager.py) | `LocalLLMManager` — backend registry, model aliases, request routing & fallbacks |
| [`omnillm/core/config.py`](../omnillm/core/config.py) | `omnillm.toml` loading and validation (backends, aliases), and appending aliases safely |
| [`omnillm/core/session.py`](../omnillm/core/session.py) | `ChatSession` — conversation state, streaming, tool calls |
| [`omnillm/adapters/ollama_adapter.py`](../omnillm/adapters/ollama_adapter.py) | Ollama integration via official Python client |
| [`omnillm/adapters/llamacpp_adapter.py`](../omnillm/adapters/llamacpp_adapter.py) | llama.cpp integration + automatic GGUF model caching |
| [`omnillm/adapters/openai_adapter.py`](../omnillm/adapters/openai_adapter.py) | Any OpenAI-compatible HTTP server (vLLM, llama-server, LM Studio, …) |
| [`omnillm/core/hardware.py`](../omnillm/core/hardware.py) | Hardware detection, KV-cache/memory estimation, GPU/CPU placement, and quant recommendation |
| [`omnillm/core/model_info.py`](../omnillm/core/model_info.py) · [`fit.py`](../omnillm/core/fit.py) · [`gguf.py`](../omnillm/core/gguf.py) | Model metadata from Hugging Face / Ollama / GGUF headers, `fit` orchestration and `--pull` |
| [`omnillm/harness/`](../omnillm/harness/) | `BenchmarkHarness`, `EvalHarness`, `NeedleHarness`, and `PickHarness` (`pick`: fit + eval + bench → Pareto ranking) |
| [`omnillm/server.py`](../omnillm/server.py) | OpenAI-compatible FastAPI HTTP server |
| [`omnillm/__main__.py`](../omnillm/__main__.py) | The `omnillm` CLI: chat, run, models, bench, eval, serve, fit, pick |

## Running Tests

```bash
# With uv (recommended)
uv run pytest

# Or activate the venv first
source .venv/bin/activate
pytest

# With coverage
uv run pytest --cov=omnillm --cov-report=term-missing

# Type checking
uv run mypy omnillm
```

---

[← Back to README](../README.md)
