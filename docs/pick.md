# Pick: the best model for your own tasks

`omnillm fit` answers "**can** this model run here?". `omnillm pick` answers "**which** of the models I have is best for **my** tasks?". It runs your own prompts on every candidate and measures quality, speed and memory on your hardware.

```bash
omnillm pick --suite docs/examples/tasks.jsonl
```

```
Candidates (GTX 1650 4 GB, 16 GB RAM, context 4,096), suite: tasks.jsonl (6 cases)

  MODEL                QUANT  PLACEMENT        MEMORY PASS   TTFT TOK/S
  ollama/qwen2.5:7b    Q4_K_M GPU+CPU (16/28) ~5.4 GB  6/6 0.9 s  12.0 ★ best quality
→ ollama/llama3.2:3b   Q4_K_M GPU             ~3.3 GB  5/6 0.2 s  55.9 ★ fastest
  ollama/llama3.1:8b   Q4_K_M GPU+CPU (16/32) ~6.2 GB  5/6 1.0 s  11.4 (dominated)

Not measured:
  ollama/llama3.3:70b: doesn't fit at context 4,096: Q2_K needs ~27.9 GB

Recommended (balanced): ollama/llama3.2:3b. 5/6 is within the tie margin (±1) of the best (ollama/qwen2.5:7b, 6/6), and it is 4.7× faster.
Pass counts within 1 of each other are within the noise of a 6-case suite. Memory is the fit estimate (~ when the file size is estimated).
```
*(Illustrative output.)*

Nothing is downloaded: every request is sent with `local_files_only`, so a model that isn't installed shows up as "not downloaded".

## How it works

1. **Candidates.** By default, your [aliases](backends.md#omnillmtoml-named-backends-and-model-aliases) (in config order), then every model installed on your Ollama backends, without duplicates. Or name them: `--models coder,ollama/qwen2.5:7b`. Models on other OpenAI-compatible servers are only included through aliases or `--models`, since such a server may list hundreds of hosted models. Embedding models are skipped.
2. **Fit.** Each candidate goes through [`fit`](fit.md) at `--context` (default 4,096). Models that don't fit are skipped. The rest run with the GPU layer count `fit` computed: Ollama `num_gpu`, llama.cpp `n_gpu_layers`. For an installed Ollama model, the installed quant is the one analysed; for a llama.cpp alias, it's the alias's `filename`. Models served by another server (OpenAI-compatible backends) can't be judged by this machine's memory, so they aren't filtered and their memory shows as `?`.
3. **Load check.** One short request per candidate, which also warms the model up. A candidate that fails here (not downloaded, out of memory, server down) is listed under "Not measured" with the reason, and the run continues.
4. **Suite.** Your [JSONL suite](harness.md#jsonl-suites) runs at temperature 0 with a fixed seed. Alias presets (such as `top_p` or `max_tokens`) still apply, and per-case `options` override everything. An alias's fallbacks are never used, so a fallback never stands in for the model being measured.
5. **Benchmark.** `--bench-runs` (default 3) streamed generations of `--bench-tokens` (default 128). TTFT and decode tokens/sec are medians.
6. **Rank.** Candidates no other candidate beats on all three of passes, speed and memory form the Pareto front. Dominated ones are marked `(dominated)`. Then one model is recommended according to `--prefer`.

## Choosing: `--prefer`, the tie margin and `--min-pass`

Small suites are noisy. With 6 cases, 6/6 against 5/6 is not good evidence that one model is better. `pick` treats pass counts within **about one standard error** of each other as tied: `floor(sqrt(cases / 2))` cases. That's 1 case for 6, 2 for 12, 5 for 50 and 7 for 100. The output always says what margin it used.

| `--prefer` | Recommends |
|---|---|
| `balanced` (default) | The fastest model whose pass count is within the tie margin of the best, so you don't pay in speed for a difference the suite can't measure |
| `quality` | The model with the most passes (exact ties go to the faster one). The explanation names any much faster model within the margin |
| `speed` | The fastest model that clears `--min-pass` |

`--min-pass` (default `0.5`, also accepts `50%`): a model that passes less than this share of your suite is never recommended. If no model reaches it, `pick` says so, names the best one, and recommends nothing.

More cases give a smaller margin relative to the suite size and a more reliable ranking. 20–50 cases that look like your real work beat a handful of trivia questions.

## Saving the winner

```bash
omnillm pick --suite tasks.jsonl --save-alias best
omnillm chat --model best
```

`--save-alias NAME` appends the recommended model to your `omnillm.toml` (the config in effect, else `./omnillm.toml`). It records the context and GPU layer count it was measured with, and the alias presets it ran with. Without `--save-alias`, the output prints the TOML to paste. Names that already exist are refused, and the file is restored if the result wouldn't load.

## Options

| Flag | Default | |
|---|---|---|
| `--suite PATH` | (required) | JSONL test cases ([format](harness.md#jsonl-suites)). [`docs/examples/tasks.jsonl`](examples/tasks.jsonl) is a small mixed example |
| `--models A,B,...` | aliases + installed Ollama models | Candidates: aliases or `backend/model` refs |
| `--prefer` | `balanced` | `quality`, `balanced` or `speed` |
| `--context N` | 4096 | Context every candidate runs with (and `fit` checks) |
| `--kv-cache` | `f16` | KV cache type, for `fit`'s memory estimate |
| `--min-pass SHARE` | 0.5 | Minimum share of passes for a recommendation |
| `--max-candidates N` | all | Measure only the first N candidates that fit |
| `--bench-tokens N` / `--bench-runs N` | 128 / 3 | Benchmark length and repetitions |
| `--dry-run` | | Show the candidates and `fit`'s verdicts; measure nothing |
| `--save-alias NAME` | | Save the recommendation as an alias |
| `--json` | | The full report: every candidate's numbers, failed case names, the margin and the reason |

Each candidate takes roughly its suite time plus a few seconds of benchmark. Progress and an estimate of the time left (scaled by model size) go to stderr. Use `--dry-run` first to see what would run, and `--max-candidates` to cap it.

## From Python

```python
from omnillm import LocalLLMManager, PickHarness
from omnillm.harness.eval import EvalHarness
from omnillm.harness.pick import discover_candidates

manager = LocalLLMManager.from_config()
harness = PickHarness(manager)
candidates = harness.plan(discover_candidates(manager), context_length=8192)
report = harness.run(
    candidates, EvalHarness.load_jsonl_suite("tasks.jsonl"), context_length=8192, prefer="quality"
)
print(report.format_table())
print(report.recommended.ref if report.recommended else report.reason)
```

## Limits

- **Memory is `fit`'s estimate, not a measurement.** It's exact for GGUF files whose sizes are known and estimated (`~`) otherwise.
- **GPU layer counts come from the free VRAM when `pick` starts.** Close other GPU apps first. Ollama keeps recently used models loaded for a few minutes and normally unloads them when a new model needs the room.
- **Pass/fail only.** Open-ended tasks (summaries, rewriting) need a check you can express as a regex, JSON Schema, JSON values or a tool call. An LLM judge is planned after v1.
- **Installed models only, for now.** Suggesting and downloading new models that fit (within a disk budget) is the next step ([#36](https://github.com/matteoespo/omni-local-llm/issues/36)).

---

[← Back to README](../README.md)
