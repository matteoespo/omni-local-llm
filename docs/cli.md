# Interactive CLI

Omni-Local-LLM provides a first-class command-line binary `omnillm` for interactive chat, one-shot prompt execution, model inspection, benchmarking, model selection, and serving:

```bash
# Try without installing (uv): uvx omni-local-llm <command> ...

# 1. Interactive multi-turn chat (with /clear, /reset, /tokens, /system, /exit)
omnillm chat --backend ollama --model llama3

# Chat with context window pruning constraints
omnillm chat --model llama3 --max-turns 10 --max-tokens 2048

# 2. One-shot prompt execution (ideal for bash scripts and UNIX pipes)
omnillm run --model llama3 "Explain quantum computing in one sentence."
cat document.txt | omnillm run --model llama3 "Summarize the key points:"

# 3. List aliases and the models every backend reports
omnillm models

# Use an alias from omnillm.toml, or any OpenAI-compatible server without a config
omnillm run --model coder "Write a Python function that reverses a string."
omnillm chat --base-url http://localhost:8080/v1 --model any
omnillm --config ./team.toml chat --model fast

# 4. Run performance benchmark (TTFT and throughput)
omnillm bench --backend ollama --model llama3 --runs 3
omnillm bench --targets coder,fast,ollama/llama3.2:3b   # compare aliases and refs side by side

# 5. Run model evaluations or Needle-In-A-Haystack retrieval test
omnillm eval --backend ollama --model llama3
omnillm eval --backend ollama --model llama3 --needle

# 6. Start the OpenAI-compatible FastAPI server
omnillm serve --host 127.0.0.1 --port 8000
OMNILLM_API_KEY=... omnillm serve --host 0.0.0.0 --allowed-host gpu-box   # require a bearer token
omnillm serve --api-key-file ./key --allow-downloads   # key from a file; let requests download models

# 7. Hardware auto-detection and smart fit quantization recommender
omnillm fit
omnillm fit unsloth/Qwen3-8B-GGUF --context 8192          # exact: real files + GGUF header
omnillm fit unsloth/Qwen3-8B-GGUF --pull                  # ...and download the recommended quant
omnillm fit llama3.2:3b --json                            # installed Ollama model

# 8. Pick the best installed model for your own tasks (fit + eval suite + benchmark)
omnillm pick --suite docs/examples/tasks.jsonl --dry-run  # which candidates fit, measure nothing
omnillm pick --suite my_tasks.jsonl --prefer quality --save-alias best
omnillm pick --suite my_tasks.jsonl --models coder,ollama/qwen2.5:7b --context 8192 --json
```

`pick` is documented in [pick.md](pick.md).

---

[← Back to README](../README.md)
