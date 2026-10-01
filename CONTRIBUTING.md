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

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior via GitHub Issues.

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
3. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate    # Linux/macOS
   # .venv\Scripts\activate     # Windows
   uv pip install -e ".[dev]"
   ```
4. **Verify everything works:**
   ```bash
   uv run pytest
   ```

---

## Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Make your changes** — keep commits focused and descriptive.
3. **Write or update tests** for any new functionality.
4. **Run the full check suite** before pushing:
   ```bash
   # Lint
   uv run ruff check .
   uv run ruff format --check .

   # Tests
   uv run pytest --cov=omnillm
   ```
5. **Push and open a Pull Request** against `main`.

---

## Adding a New Backend Adapter

To add support for a new inference backend (e.g., `vllm`, `llamafile`, `koboldcpp`):

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
3. **Register the adapter** in the `LocalLLMManager` constructor in [`omnillm/core/manager.py`](omnillm/core/manager.py).
4. **Add tests** under `tests/` using mocks (see existing tests for patterns).
5. **Update the README** to mention the new backend.

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

1. Ensure all checks pass (lint + tests).
2. Fill out the PR template completely.
3. Reference any related issues (e.g., `Closes #42`).
4. Keep PRs focused — one feature or fix per PR.
5. Be responsive to review feedback.

**First-time contributor?** Look for issues labeled [`good first issue`](https://github.com/matteoespo/omni-local-llm/labels/good%20first%20issue) 🏷️

---

Thank you for helping make Omni-Local-LLM better! 🚀
