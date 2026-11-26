# Evaluation & Benchmarking Harness

Omni-Local-LLM includes an evaluation and benchmarking harness to measure local model performance, schema reliability, and context retrieval:

## 1. Performance Benchmarking (`BenchmarkHarness`)

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

## 2. Capability & Schema Compliance (`EvalHarness`)

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

### JSONL suites

`omnillm eval --model coder --suite my_tasks.jsonl` (or `EvalHarness.load_jsonl_suite(path)`) reads one test case per line. Every check that's set must pass:

| Key | Check |
|---|---|
| `prompt` (required), `name`, `system_prompt` | The request |
| `expected_pattern` | Regex the output must contain (case-insensitive) |
| `expected_json_schema` | JSON Schema the output must satisfy. It's sent to the backend as a structured-output format (enforced at the sampler where supported), then validated. Needs `pip install "omni-local-llm[eval]"` |
| `expected_json` | Values the output's JSON must contain. Objects may have extra keys; lists and scalars must match exactly (`true` never equals `1`). Turns on JSON mode unless a schema is set |
| `expected_tool_call` + `tools` | Name of a tool the model must call, and the OpenAI-format tool definitions to offer |
| `options` | Per-case request options, e.g. `{"temperature": 0, "max_tokens": 200}` |

```jsonl
{"name": "capital", "prompt": "Capital of France? One word.", "expected_pattern": "^\\W*paris\\W*$"}
{"name": "extract-person", "prompt": "Extract: Bob is 34.", "expected_json_schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}, "expected_json": {"name": "Bob", "age": 34}}
{"name": "weather-tool", "prompt": "Weather in Oslo?", "expected_tool_call": "get_weather", "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}
```

A schema on its own mostly shows that the backend enforced the format. Add `expected_json` to check the model actually extracted the right values. Suites are validated when loaded, so a malformed line or an invalid schema fails before any model runs, naming the file and line.

## 3. Context Retrieval ("Needle In A Haystack")

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

[← Back to README](../README.md)
