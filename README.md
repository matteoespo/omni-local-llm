<p align="center">
  <img src="https://img.shields.io/badge/🧠-Omni--Local--LLM-blueviolet?style=for-the-badge&logoColor=white" alt="Omni-Local-LLM" height="50"/>
</p>

<h3 align="center">One interface. Every local LLM.</h3>

<p align="center">
  <em>A unified Python SDK and OpenAI-compatible API server for local large language models.</em>
</p>

<p align="center">
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml"><img src="https://github.com/matteoespo/omni-local-llm/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/v/omni-local-llm?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/omni-local-llm/"><img src="https://img.shields.io/pypi/pyversions/omni-local-llm" alt="Python"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/matteoespo/omni-local-llm?color=green" alt="License"></a>
  <a href="https://github.com/matteoespo/omni-local-llm/stargazers"><img src="https://img.shields.io/github/stars/matteoespo/omni-local-llm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-interactive-cli">CLI</a> •
  <a href="#-evaluation--benchmarking-harness">Harness</a> •
  <a href="#-api-server">API Server</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Features

| Feature | Description |
|---|---|
| **Multi-Backend** | Swap between [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp) with a single parameter change |
| **Vision & Multimodal** | Native image understanding across Ollama and llama.cpp (file paths, bytes, data URIs, and OpenAI format) |
| **Evaluation & Benchmarking** | Measure TTFT, tokens/sec, schema compliance, tool-calling precision, and needle-in-haystack context retrieval |
| **Unified CLI** | Dedicated binary `omnillm` with `chat`, `run`, `models`, `bench`, `eval`, `fit`, and `serve` commands |
| **Hardware Auto-Fit** | Auto-detect host RAM, CUDA VRAM, and Apple Silicon Metal memory to calculate exact model fit and recommend optimal quantization (`omnillm fit`) |
| **Chat Sessions & Pruning** | Resilient turn commits, sliding-window turn pruning, token budget enforcement, and thread-safe resets |
| **Streaming** | Real-time token-by-token streaming (sync & async) |
| **Tool Calling & Agents** | Autonomous function execution loop (`@tool` decorator & `session.act`) |
| **Structured Outputs** | Guaranteed JSON schema adherence and Pydantic validation via GBNF or Ollama schemas |
| **Vector Embeddings** | Generate vector embeddings for RAG and semantic search via SDK or API |
| **OpenAI-Compatible API** | Drop-in FastAPI server compatible with the OpenAI SDK |
| **Async-First** | Full `async`/`await` support for high-concurrency workloads |
| **Auto Model Pull** | Automatically downloads models from Ollama or Hugging Face Hub |

---

## Quick Start

```bash
# Install
pip install "omni-local-llm[ollama,server]"

# Or with uv (recommended)
uv pip install "omni-local-llm[ollama,server]"
```

```python
from omnillm import LocalLLMManager

manager = LocalLLMManager()
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)
```

**That's it.** Switch to llama.cpp by changing one word:

```python
response = manager.chat(
    backend="llama.cpp",
    model="unsloth/llama-3-8b-Instruct-GGUF",
    messages=[{"role": "user", "content": "Hello!"}],
    filename="llama-3-8b-Instruct-Q4_K_M.gguf"
)
```

---

## Architecture

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
│              (Backend Registry & Router)                │
├────────────────────────┬────────────────────────────────┤
│    OllamaAdapter       │       LlamaCPPAdapter          │
│  (ollama Python SDK)   │  (llama-cpp-python + HF Hub)  │
└────────────────────────┴────────────────────────────────┘
```

### Module Map

| Module | Purpose |
|---|---|
| [`omnillm/core/base.py`](omnillm/core/base.py) | `LLMBackend` — abstract base class for all adapters |
| [`omnillm/core/manager.py`](omnillm/core/manager.py) | `LocalLLMManager` — backend registry & request router |
| [`omnillm/core/session.py`](omnillm/core/session.py) | `ChatSession` — conversation state, streaming, tool calls |
| [`omnillm/adapters/ollama_adapter.py`](omnillm/adapters/ollama_adapter.py) | Ollama integration via official Python client |
| [`omnillm/adapters/llamacpp_adapter.py`](omnillm/adapters/llamacpp_adapter.py) | llama.cpp integration + automatic GGUF model caching |
| [`omnillm/core/hardware.py`](omnillm/core/hardware.py) | Hardware auto-detection, memory estimation, and quantization recommender |
| [`omnillm/server.py`](omnillm/server.py) | OpenAI-compatible FastAPI HTTP server |
| [`omnillm/__main__.py`](omnillm/__main__.py) | Interactive CLI chat, benchmark, and evaluation interface |

---

## Installation

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip
- At least one backend:
  - [Ollama](https://ollama.com/) installed and running, **or**
  - A GGUF model on [Hugging Face Hub](https://huggingface.co/) (auto-downloaded)

### Install from source

```bash
git clone https://github.com/matteoespo/omni-local-llm.git
cd omni-local-llm
uv pip install -e ".[ollama,server]"
```

### Install from PyPI

```bash
pip install "omni-local-llm[ollama,server]"
# or
uv pip install "omni-local-llm[ollama,server]"
```

---

## Usage

### Python SDK

#### Basic Chat

```python
from omnillm import LocalLLMManager

manager = LocalLLMManager()

response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)
```

#### Chat Session (with Memory & Context Window Pruning)

`ChatSession` maintains conversation history, commits turns only on success, and supports proactive sliding-window context pruning to keep local models within their context limits:

```python
# Create a session with turn and token budget constraints
session = manager.create_session(
    backend="ollama",
    model="llama3",
    system_prompt="You are a helpful coding assistant.",
    max_turns=10,             # Keep the most recent 10 conversational turns
    max_tokens_budget=4096,   # Automatically evict oldest turns before exceeding context budget
    strategy="sliding_window",
)

# System prompt is ALWAYS preserved during pruning
response1 = session.send("Hello, I am Bob.")
response2 = session.send("What is my name?")  # Remembers "Bob"

# Inspection properties
print(f"Stored turns: {session.turn_count}")
print(f"Estimated token footprint: {session.estimated_tokens}")

# Reset / clear controls
session.clear()                # Clears history, keeping original system_prompt
session.reset("New prompt")    # Clears history and updates system prompt
```

#### Async & Streaming

```python
import asyncio
from omnillm import LocalLLMManager

async def main():
    manager = LocalLLMManager()
    session = manager.create_session(backend="ollama", model="llama3")

    stream = await session.asend("Tell me a short story.", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

#### Tool Calling & Autonomous Agent Loop (`@tool` & `session.act`)

Omni-Local-LLM provides an **autonomous tool-calling loop**. Decorate any Python function with `@tool` (or pass plain functions), and `session.act()` will automatically inspect type hints, invoke the function, feed results back to the model, and return the final answer:

```python
from omnillm import tool

# 1. Define tools with standard Python type hints and docstrings
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather conditions for a city."""
    return f"24 degrees {unit}, sunny"

@tool
def calculate_travel_time(distance_km: float, speed_kmh: float = 80.0) -> float:
    """Calculate the estimated travel time in hours."""
    return round(distance_km / speed_kmh, 2)

# 2. Run the autonomous agent loop
session = manager.create_session(backend="ollama", model="llama3")
response = session.act(
    "What's the weather in Rome, and how long does it take to drive 320 km?",
    tools=[get_weather, calculate_travel_time]
)

print(response.content)
# The model invokes get_weather("Rome"), then calculate_travel_time(320.0),
# and automatically returns the final answer with observations included!

# Asynchronous agent loop
response = await session.aact(
    "Check weather in Tokyo",
    tools=[get_weather]
)
```

Manual low-level OpenAI-format tool calling is also supported via `manager.chat(..., tools=tools)`.

#### Structured Outputs & Pydantic Validation

Guarantee JSON schema adherence at the sampler level (using Ollama schemas or llama.cpp GBNF grammars) and receive typed Pydantic models automatically:

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    skills: list[str]

response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Extract: Alice is a 28-year-old engineer skilled in Python and Rust."}],
    response_model=UserProfile,
)

# Access validated Pydantic object directly
user: UserProfile = response.parsed
print(user.name)    # Alice
print(user.skills)  # ['Python', 'Rust']
```

You can also pass raw JSON schemas or request basic JSON mode:

```python
# Raw JSON Schema
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Generate a city record"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "City",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
                "required": ["city", "population"],
            },
        },
    },
)

# Basic JSON Mode
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    json_mode=True,
)
```

#### Embeddings (RAG & Semantic Search)

Generate dense vector embeddings for documents or search queries using either backend:

```python
# Synchronous embeddings with Ollama
response = manager.embed(
    backend="ollama",
    model="nomic-embed-text",
    input=["Retrieval-augmented generation with local LLMs", "Semantic similarity search"]
)
print(response.embeddings)  # [[0.021, -0.043, ...], [0.112, 0.009, ...]]
print(response.usage.prompt_tokens)

# Embeddings with llama.cpp (auto-downloads GGUF from Hugging Face Hub)
response = manager.embed(
    backend="llama.cpp",
    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    input="Single string input is also supported"
)

# Asynchronous embeddings
response = await manager.aembed(
    backend="ollama",
    model="nomic-embed-text",
    input=["Async batch embedding"]
)
```

#### Vision & Multimodal (Image Understanding)

Run local multimodal models (such as `llava`, `llama3.2-vision`, or `qwen2-vl`) using local file paths, raw bytes, base64 data URIs, or OpenAI-standard image parts:

```python
# Direct image input via file path or bytes (Ollama)
response = manager.chat(
    backend="ollama",
    model="llava",
    messages=[{
        "role": "user",
        "content": "What is depicted in this photo?",
        "images": ["photo.jpg"],  # accepts file paths, Path objects, base64, or bytes
    }]
)
print(response.content)

# Using llama.cpp with Hugging Face GGUF + multimodal projector (mmproj)
response = manager.chat(
    backend="llama.cpp",
    model="myself/llava-1.5-7b-GGUF",
    filename="llava-1.5-7b-Q4_K.gguf",
    mmproj_filename="mmproj-model-f16.gguf",  # automatically downloaded & loaded
    messages=[{
        "role": "user",
        "content": "Analyze this chart.",
        "images": ["chart.png"],
    }]
)

# In ChatSession with conversational memory
session = manager.create_session(backend="ollama", model="llava")
response = session.send("What is in this image?", images=["invoice.png"])
response2 = session.send("Extract the total amount.")  # Remembers the image context!

# OpenAI-compatible multi-part format (also supported via FastAPI server)
response = manager.chat(
    backend="ollama",
    model="llava",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
)
```

---

### Interactive CLI

Omni-Local-LLM provides a first-class command-line binary `omnillm` for interactive chat, one-shot prompt execution, model inspection, benchmarking, and serving:

```bash
# 1. Interactive multi-turn chat (with /clear, /reset, /tokens, /system, /exit)
omnillm chat --backend ollama --model llama3

# Chat with context window pruning constraints
omnillm chat --model llama3 --max-turns 10 --max-tokens 2048

# 2. One-shot prompt execution (ideal for bash scripts and UNIX pipes)
omnillm run --model llama3 "Explain quantum computing in one sentence."
cat document.txt | omnillm run --model llama3 "Summarize the key points:"

# 3. List installed / detected local models
omnillm models

# 4. Run performance benchmark (TTFT and throughput)
omnillm bench --backend ollama --model llama3 --runs 3

# 5. Run model evaluations or Needle-In-A-Haystack retrieval test
omnillm eval --backend ollama --model llama3
omnillm eval --backend ollama --model llama3 --needle

# 6. Start the OpenAI-compatible FastAPI server
omnillm serve --host 127.0.0.1 --port 8000

# 7. Hardware auto-detection and smart fit quantization recommender
omnillm fit
omnillm fit llama-3.1-8b --context 8192
omnillm fit mistral:7b --json
```

---

### Hardware Auto-Detection & Smart Fit Quantization

Never guess whether a model will fit into your VRAM or RAM again. Omni-Local-LLM inspects host RAM, Apple Silicon Metal unified memory, and NVIDIA CUDA VRAM, computing exact model weights, KV cache overhead, and runtime buffers to recommend the highest quality quantization (`FP16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`, `Q2_K`).

#### 1. CLI Hardware Profiling & Recommendation

Run `omnillm fit` without arguments to see your machine's hardware profile and a model compatibility matrix:

```bash
omnillm fit
```

Output:
```
🖥️  Detected Hardware Profile
------------------------------------------------------------------------------
OS:           Linux 6.8.0-142-generic (x86_64, 12 CPUs)
System RAM:   14.96 GB Total (3.26 GB Available)
GPU:          NVIDIA GeForce GTX 1650 (4.00 GB VRAM, 3.62 GB Free)
------------------------------------------------------------------------------

🚀 Model Compatibility Matrix for Your Machine:
MODEL                SIZE     FIT TARGET                 QUANT      EST. RAM
------------------------------------------------------------------------------
smollm:135m          0.135B   ✅ GPU (Full Offload)       Q8_0       0.69 GB
llama-3.2:1b         1.0B     ✅ GPU (Full Offload)       Q8_0       1.60 GB
llama-3.2:3b         3.0B     ✅ GPU (Full Offload)       Q8_0       3.74 GB
phi-3-mini:3.8b      3.8B     ✅ GPU (Full Offload)       Q5_K_M     3.30 GB
mistral:7b           7.0B     ✅ GPU (Full Offload)       Q2_K       3.50 GB
llama-3.1:8b         8.0B     ⚠️ CPU / System RAM        Q4_K_M     5.54 GB
deepseek-r1:14b      14.0B    ⚠️ CPU / System RAM        Q4_K_M     9.39 GB
qwen2.5:32b          32.0B    ❌ Insufficient Memory      N/A        0.00 GB
llama-3.3:70b        70.0B    ❌ Insufficient Memory      N/A        0.00 GB
```

Analyze a specific model and context window:

```bash
omnillm fit llama-3.1-8b --context 8192
```

```
🧠 Model Fit Analysis: llama-3.1-8b (~8.0B params, 8,192 context)
------------------------------------------------------------------------------
QUANT     SIZE (GB)   FIT STATUS               QUALITY                SPEED
------------------------------------------------------------------------------
FP16      17.40 GB    ❌ Insufficient Memory    Maximum (Lossless)     OOM
Q8_0       9.80 GB    ⚠️ CPU / System RAM      Near Lossless (~99.5%) Moderate / Slow (CPU Compute)
Q6_K       7.96 GB    ⚠️ CPU / System RAM      Extremely High (>99%)  Moderate / Slow (CPU Compute)
Q5_K_M     7.00 GB    ⚠️ CPU / System RAM      High Sweet-Spot (~98%) Moderate / Slow (CPU Compute)
Q4_K_M     6.04 GB    ⚠️ CPU / System RAM      Recommended Standard   Moderate / Slow (CPU Compute)
Q3_K_M     5.24 GB    ⚠️ CPU / System RAM      Noticeable Degradation Moderate / Slow (CPU Compute)
Q2_K       4.44 GB    ⚠️ CPU / System RAM      High Perplexity Loss   Moderate / Slow (CPU Compute)
------------------------------------------------------------------------------

💡 Recommendation:
   Recommended: Q4_K_M on CPU / System RAM (~6.04 GB). Exceeds GPU VRAM; will run using CPU RAM.
   💡 Tip: For full GPU acceleration, try a smaller model (e.g., 3B or 1B) to fit within your 4.0 GB VRAM.
```

#### 2. Python SDK

```python
from omnillm import detect_hardware, recommend_model_fit, estimate_model_memory

# Inspect host hardware
hw = detect_hardware()
print(hw.summary())
# "OS: Linux (x86_64, 12 CPUs) | RAM: 15.0 GB Total | GPU: NVIDIA GeForce GTX 1650 (4.0 GB VRAM)"

# Evaluate model fit
fit = recommend_model_fit("llama-3.2-3b", context_length=4096)
if fit.can_fit:
    print(f"Optimal Quantization: {fit.recommended_quant}")
    print(f"Target: {fit.best_target}")

# Estimate memory for any model size & quant
total_gb, weights_gb, kv_cache_gb = estimate_model_memory(param_count_b=70.0, quant="Q4_K_M", context_length=16384)
print(f"70B Q4_K_M @ 16K context requires ~{total_gb:.1f} GB memory")
```

---

### Evaluation & Benchmarking Harness

Omni-Local-LLM includes an evaluation and benchmarking harness to measure local model performance, schema reliability, and context retrieval:

#### 1. Performance Benchmarking (`BenchmarkHarness`)

Benchmark Time to First Token (TTFT), generation throughput (Tokens/sec), and latency across multiple models and backends:

```python
from omnillm.harness import BenchmarkHarness

harness = BenchmarkHarness()
suite = harness.run(
    targets=[
        ("ollama", "llama3"),
        ("llama.cpp", "unsloth/llama-3-8b-Instruct-GGUF", {"filename": "llama-3-8b-Instruct-Q4_K_M.gguf"}),
    ],
    prompt="Explain quantum physics in three sentences.",
    max_tokens=128,
    runs=3,
)

# Print markdown comparison table directly to terminal
suite.print_table()
```

Output:
```
| Backend   | Model                                | TTFT (ms) | Tokens/sec | Total (s) | Runs |
|-----------|--------------------------------------|-----------|------------|-----------|------|
| ollama    | llama3                               | 312.4     | 45.2       | 2.85      | 3    |
| llama.cpp | unsloth/llama-3-8b-Instruct-GGUF     | 240.1     | 48.6       | 2.63      | 3    |
```

#### 2. Capability & Schema Compliance (`EvalHarness`)

Systematically evaluate reasoning, JSON Schema adherence, and tool-calling precision:

```python
from pydantic import BaseModel
from omnillm.harness import EvalHarness, EvalTestCase

class Person(BaseModel):
    name: str
    age: int

harness = EvalHarness()
results = harness.run_suite(
    backend="ollama",
    model="llama3",
    test_cases=[
        EvalTestCase(name="Regex Check", prompt="Capital of France?", expected_pattern=r"Paris"),
        EvalTestCase(name="Schema Test", prompt="Extract: Bob, 34", expected_schema=Person),
    ],
)
results.print_table()
# Outputs pass rate %, individual test latency, and errors
```

#### 3. Context Retrieval ("Needle In A Haystack")

Stress-test context length retrieval by hiding a secret code at 0%, 25%, 50%, 75%, and 100% depths:

```python
from omnillm.harness import NeedleHarness

needle_harness = NeedleHarness()
result = needle_harness.run(
    backend="ollama",
    model="llama3",
    secret_code="ALPHA-777",
    depths=[0, 25, 50, 75, 100],
    target_word_count=1500,
)
result.print_table()
# Displays retrieval success and extraction latency per depth
```

---

### API Server

Launch an OpenAI-compatible local API server:

```bash
omnillm serve
# Or: python -m omnillm.server
# Server runs on http://localhost:8000
```

#### Use with the OpenAI SDK

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

#### Use with curl

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

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up your development environment
- Adding new LLM backend adapters
- Submitting pull requests

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Roadmap

- [ ] vLLM backend adapter
- [ ] KoboldCpp backend adapter
- [ ] Token usage tracking and reporting
- [ ] Vision/multimodal support
- [ ] Model hot-swapping in sessions
- [ ] Prompt template management
- [ ] Docker image for the API server

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/matteoespo">matteoespo</a>
</p>
