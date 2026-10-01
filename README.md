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
  <a href="#-api-server">API Server</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## Features

| Feature | Description |
|---|---|
| **Multi-Backend** | Swap between [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp) with a single parameter change |
| **Chat Sessions & Pruning** | Resilient turn commits, sliding-window turn pruning, token budget enforcement, and thread-safe resets |
| **Streaming** | Real-time token-by-token streaming (sync & async) |
| **Tool Calling** | Native function/tool calling support across backends |
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
| [`omnillm/server.py`](omnillm/server.py) | OpenAI-compatible FastAPI HTTP server |
| [`omnillm/__main__.py`](omnillm/__main__.py) | Interactive CLI chat interface |

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

#### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
# response.tool_calls contains the function call
```

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

---

### Interactive CLI

Start an interactive chat session directly in your terminal:

```bash
# Using Ollama
python -m omnillm --backend ollama --model llama3

# Using llama.cpp (auto-downloads from Hugging Face)
python -m omnillm --backend llama.cpp \
    --model unsloth/llama-3-8b-Instruct-GGUF \
    --filename llama-3-8b-Instruct-Q4_K_M.gguf

# With a system prompt
python -m omnillm --backend ollama --model llama3 \
    --system "You are a pirate. Respond only in pirate speak."
```

---

### API Server

Launch an OpenAI-compatible local API server:

```bash
python -m omnillm.server
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
