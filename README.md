# Omni-LLM

**Omni-LLM** is a unified Python interface and API server for running local Large Language Models (LLMs). It provides a single wrapper over multiple local inference backends (currently supporting [Ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggerganov/llama.cpp)) with built-in support for chat history, streaming responses, JSON mode, and tool calling (function calling).

## Table of Contents
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Interactive CLI](#interactive-cli)
  - [OpenAI-Compatible FastAPI Server](#openai-compatible-fastapi-server)
- [Running Tests](#running-tests)
- [License](#license)

---

## Architecture

The project has a modular design separating inference engines (adapters) from the session and server layers:

- **[omnillm/core/base.py](omnillm/core/base.py)**: Defines `LLMBackend`, the abstract base class for all LLM adapters.
- **[omnillm/core/manager.py](omnillm/core/manager.py)**: Implements `LocalLLMManager` which acts as the registry for loading backend adapters and routing chat requests.
- **[omnillm/core/session.py](omnillm/core/session.py)**: Implements `ChatSession` for managing conversational state/history, streaming, and tool calls.
- **[omnillm/adapters/](omnillm/adapters/)**:
  - `ollama_adapter.py`: Integrates with Ollama's Python client.
  - `llamacpp_adapter.py`: Integrates with `llama-cpp-python` and automatically pulls/caches GGUF models from Hugging Face Hub.
- **[omnillm/server.py](omnillm/server.py)**: Implements an OpenAI-compatible FastAPI HTTP server.
- **[omnillm/__main__.py](omnillm/__main__.py)**: Interactive CLI interface.

---

## Installation

Ensure you have Python 3.12+ installed. The project uses [uv](https://github.com/astral-sh/uv) for package and dependency management.

To install dependencies and package in editable mode:
```bash
uv pip install -e .
```

---

## Quick Start

### Python API

#### Basic Chat
```python
from omnillm import LocalLLMManager

manager = LocalLLMManager()

# Synchronous chat
response = manager.chat(
    backend="ollama",
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```

#### Chat Session (with Memory / History)
```python
session = manager.create_session(
    backend="ollama", 
    model="llama3", 
    system_prompt="You are a helpful assistant."
)

# First send
response1 = session.send("Hello, I am Bob.")
# Second send (remembers user name)
response2 = session.send("What is my name?")
```

#### Asynchronous Chat & Streaming
```python
import asyncio
from omnillm import LocalLLMManager

async def main():
    manager = LocalLLMManager()
    session = manager.create_session(backend="ollama", model="llama3")
    
    # Asynchronous streaming
    stream = await session.asend("Tell me a short story.", stream=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

## Interactive CLI

Start an interactive chat session in the terminal:
```bash
# Using Ollama
python -m omnillm --backend ollama --model llama3

# Using llama.cpp (downloads model from HuggingFace Hub)
python -m omnillm --backend llama.cpp --model unsloth/llama-3-8b-Instruct-GGUF --filename llama-3-8b-Instruct-Q4_K_M.gguf
```

---

## OpenAI-Compatible FastAPI Server

Launch the local FastAPI server simulating OpenAI's `/v1/chat/completions` endpoints:

```bash
python -m omnillm.server
```
The server runs on `http://localhost:8000`.

### Client Example (OpenAI SDK)
You can target the server using the standard `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ignored"
)

# Prefix the model parameter with "backend/" to route to a specific adapter.
# Defaults to ollama if no prefix is supplied.
response = client.chat.completions.create(
    model="ollama/llama3",
    messages=[{"role": "user", "content": "Explain relativity in one sentence."}]
)
print(response.choices[0].message.content)
```

---

## Running Tests

To run the test suite, make sure you run it within the project's virtual environment. You can run it in one of the following ways:

**Using the virtual environment's executable directly:**
```bash
.venv/bin/pytest
```

**Using `uv` (recommended if installed):**
```bash
uv run pytest
```

**Activating the virtual environment first:**
```bash
source .venv/bin/activate
pytest
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

