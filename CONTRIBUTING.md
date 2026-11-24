# Contributing to Omni-Local-LLM

Thank you for your interest in contributing to **Omni-Local-LLM**! 🎉

We welcome contributions of all kinds — new backend adapters, bug fixes, documentation improvements, test coverage, and feature ideas.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Adding a New Backend Adapter](#adding-a-new-backend-adapter)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct, version 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold it. Please report unacceptable behavior privately to the maintainer ([@matteoespo](https://github.com/matteoespo)) rather than in a public issue.

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager

### Setup

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/omni-local-llm.git
   cd omni-local-llm
   ```
3. **Install the package and the dev tools** (pytest, ruff, mypy):
   ```bash
   uv sync
   ```
   This creates `.venv/` from the lockfile with the `server`, `ollama` and `openai` extras; `uv run <command>` uses it automatically. The llama.cpp backend isn't installed by default, because `llama-cpp-python` compiles on install (it needs a C/C++ compiler and CMake, and takes a few minutes). The test suite doesn't need it. To work on the llama.cpp adapter, run `uv sync --extra llama-cpp`.
4. **Verify everything works:**
   ```bash
   uv run pytest
   ```

For a map of the code, see [`docs/architecture.md`](docs/architecture.md).

---

## Development Workflow

0. **Claim an issue first**: comment on it so the maintainer can assign it to you (GitHub only allows assigning people who have commented on the issue). For anything not covered by an issue, open one to discuss it before writing code.
1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Make your changes** — keep commits focused and descriptive.
3. **Write or update tests** for any new functionality.
4. **Run the same checks as CI** before pushing:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy omnillm
   uv run pytest
   ```
5. **Push and open a Pull Request** against `main`. Opening it early as a **draft** is welcome: you get feedback before polishing. On your first PR, a maintainer has to approve the CI run before checks start.

---

## Adding a New Backend Adapter

**First check whether you need one.** Servers that speak the OpenAI API (vLLM, llama.cpp's `llama-server`, LM Studio, SGLang, KoboldCpp, …) already work through `OpenAICompatibleAdapter`: add them in `omnillm.toml` with `type = "openai"` (see [`docs/backends.md`](docs/backends.md)). A new adapter is for backends with their own API.

To add support for a new inference backend:

1. **Create a new adapter** under `omnillm/adapters/`:
   ```
   omnillm/adapters/your_adapter.py
   ```
2. **Implement the `LLMBackend` interface** defined in [`omnillm/core/base.py`](omnillm/core/base.py):
   - `pull_model()` — download/verify the model
   - `chat()` — synchronous inference
   - `achat()` — asynchronous inference
   - `embed()` — synchronous vector embeddings (optional, if backend supports embeddings)
   - `aembed()` — asynchronous vector embeddings
   - `list_models()` — list available local models
3. **Register the adapter**: users can call `manager.register_backend(name, factory)`. To ship it built in, add it to the defaults in the `LocalLLMManager` constructor in [`omnillm/core/manager.py`](omnillm/core/manager.py), and add a `type` for it in [`omnillm/core/config.py`](omnillm/core/config.py) so it can be configured in `omnillm.toml`.
4. **Add tests** under `tests/` using mocks (see existing tests for patterns).
5. **Update the docs**: [`docs/backends.md`](docs/backends.md) and the README feature list.

---

## Running Tests

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov=omnillm --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_adapters.py -v
```

---

## Code Style

This project uses **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting:

```bash
# Check for lint issues
uv run ruff check .

# Auto-fix lint issues
uv run ruff check . --fix

# Check formatting
uv run ruff format --check .

# Auto-format
uv run ruff format .
```

The configuration lives in [`pyproject.toml`](pyproject.toml).

### Guidelines

- Keep code simple, clean, and self-documenting.
- Preserve existing docstrings and comments.
- Use type hints where practical.
- Maintain test coverage for new backends or features.

---

## Submitting a Pull Request

1. Ensure all checks pass (lint, format, mypy, tests).
2. Fill out the PR template completely.
3. Reference any related issues (e.g., `Closes #42`).
4. Keep PRs focused — one feature or fix per PR.
5. Be responsive to review feedback.

**First-time contributor?** Look for issues labeled [`good first issue`](https://github.com/matteoespo/omni-local-llm/labels/good%20first%20issue) 🏷️

---

Thank you for helping make Omni-Local-LLM better! 🚀
