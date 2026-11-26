# Hardware Fit: Which Quant Should I Run? (`omnillm fit`)

`omnillm fit` answers "which file of this model should I download for *my* machine, and how do I run it?" It detects RAM, NVIDIA VRAM, and Apple Silicon unified memory, reads the model's **real architecture**, and evaluates every available quantization:

| You pass | Where the numbers come from |
|---|---|
| A Hugging Face GGUF repo (`unsloth/Qwen3-8B-GGUF`) | Real per-quant file sizes (incl. split files) + architecture from the GGUF header, read with HTTP range requests (a few hundred KB, not the model) |
| A Hugging Face model repo (`Qwen/Qwen3-30B-A3B`) | `config.json` + exact parameter count; quant sizes estimated |
| An installed Ollama model (`llama3.2:3b`) | GGUF metadata via `ollama show` + the installed size |
| A config alias or `backend/model` (`coder`) | Whatever the alias points to, using its `n_ctx` and noting the alias's own file |
| A `.gguf` file | The file's header and size |
| Just a name (`llama-3.1-8b`) or `--offline` | Estimated from the name (`8b`, MoE `30b-a3b`) |

What it computes:

- **KV cache from the real shape**: layers × KV heads × head dim × context, so grouped-query attention (and DeepSeek's MLA) are accounted for. `--kv-cache q8_0|q4_0` shows the effect of a quantized KV cache.
- **Placement**: fully on GPU, Metal unified memory, **split GPU + CPU with a concrete `n_gpu_layers`**, CPU-only, or doesn't fit. It warns when the RAM a plan needs is not free right now.
- **MoE awareness**: total vs. active parameters, plus tips for running experts on CPU.
- **A recommendation**: the best quant (up to Q8_0) that runs fully on the GPU. Otherwise, the best quant up to Q4_K_M split across GPU and CPU, with the faster GPU-only alternative noted.

```bash
omnillm fit unsloth/Qwen3-30B-A3B-GGUF --context 8192     # on a 4 GB GTX 1650 + 16 GB RAM laptop
```

```
🧠 Model Fit Analysis: unsloth/Qwen3-30B-A3B-GGUF
   30.5B params, MoE ~3.3B active, 48 layers, GQA 32/4 heads, trained ctx 40,960 | context 8,192 | KV cache f16
   Source: Hugging Face GGUF repo (exact file sizes + GGUF header)
----------------------------------------------------------------------------------------------------
   QUANT            BPW   WEIGHTS      KV     TOTAL  FIT                                QUALITY            SPEED
----------------------------------------------------------------------------------------------------
   Q8_0            8.50     30.25    0.75     31.53  ❌ Insufficient Memory             Near lossless      -
   ...
   IQ4_XS          4.25     15.25    0.75     16.53  ❌ Insufficient Memory             Good               -
→  Q3_K_M          3.91     13.70    0.75     14.98  ⚠️  GPU+CPU (9 layers on GPU)       Good               Good (MoE)
   ...
💡 Recommendation:
   Recommended: Q3_K_M → GPU + CPU with n_gpu_layers=9 of 48 (~14.98 GB total).
   ℹ️  MoE model: only a fraction of the weights is used per token ...
```

No install needed to try it: `uvx omni-local-llm fit <model>`.

`--pull` downloads the recommended quant (or `--quant Q5_K_M`) into the Hugging Face cache that the llama.cpp backend uses. `--save-alias NAME` writes it to your `omnillm.toml` (the config in effect, else `./omnillm.toml`), so `omnillm run --model NAME` works right away. Without `--save-alias`, it prints a ready-to-paste alias:

```bash
omnillm fit bartowski/SmolLM2-135M-Instruct-GGUF --context 2048 --pull --quant Q4_K_M
```

```toml
[models.smollm2-135m-instruct]
backend = "llama.cpp"
model = "bartowski/SmolLM2-135M-Instruct-GGUF"
filename = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
n_ctx = 2048
n_gpu_layers = -1
```

For an installed Ollama model, apply the plan with an alias: `n_ctx` and `n_gpu_layers` are passed to Ollama as `num_ctx` and `num_gpu` (see [Backends & config](backends.md#omnillmtoml-named-backends-and-model-aliases)).

Run `omnillm fit` without arguments for a compatibility matrix of popular models on your machine. Set `HF_TOKEN` for gated repos and `HF_ENDPOINT` for a mirror.

## Accuracy

Measured on a GTX 1650 (4 GB VRAM, 3.62 GB free) with 16 GB RAM, loading each installed Ollama model with the
GPU layer count `fit` recommends and reading the VRAM actually used (`nvidia-smi`, per process). Speeds are
medians of 3 runs, compared with Ollama's own automatic placement. Ollama 0.21.0, default f16 KV cache.

| Model (Ollama) | Quant | Context | fit says | Predicted VRAM | Measured VRAM | tok/s at fit's layers | tok/s Ollama auto |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | Q4_K_M | 4,096 | all on GPU | 2.83 GB | 2.68 GB | 56.1 | 56.2 |
| llama3.2:3b | Q4_K_M | 8,192 | all on GPU | 3.33 GB | 3.12 GB | 55.9 | 43.4 |
| phi3:latest | Q4_0 | 4,096 | 26/32 layers on GPU | 3.38 GB | 3.03 GB | 32.9 | 34.3 |
| phi3:latest | Q4_0 | 8,192 | 18/32 layers on GPU | 3.40 GB | 3.03 GB | 23.5 | 21.1 |
| qwen2.5:7b | Q4_K_M | 4,096 | 17/28 layers on GPU | 3.30 GB | 3.21 GB | 13.0 | 12.6 |
| qwen2.5:7b | Q4_K_M | 8,192 | 16/28 layers on GPU | 3.33 GB | 3.19 GB | 12.0 | 11.8 |
| llama3.1:latest | Q4_K_M | 4,096 | 18/32 layers on GPU | 3.39 GB | 3.26 GB | 11.3 | 11.1 |
| llama3.1:latest | Q4_K_M | 8,192 | 16/32 layers on GPU | 3.40 GB | 3.25 GB | 11.4 | 10.8 |

- **VRAM predictions were 0.09–0.37 GB above the measured value in all 8 cases** (3–12%), so `fit` errs on the
  safe side, and every recommended configuration loaded without running out of memory.
- At `fit`'s layer counts, decode speed was usually within a few percent of Ollama's automatic choice, slower in
  one case (phi3 at 4K, −4%) and **29% faster** in another: for llama3.2:3b at 8K context, `fit` correctly
  says the whole model fits on the GPU (3.12 GB measured), while Ollama keeps part of it on the CPU.
- Not measured yet: system-RAM usage of the CPU part, Apple Silicon, AMD, and other runtimes.

**Help make it more accurate:** with an NVIDIA GPU and Ollama, run
`uv run python scripts/measure_fit_accuracy.py <model> [...]` and share the table in a
[fit accuracy report](https://github.com/matteoespo/omni-local-llm/issues/new?template=fit_accuracy.yml).
On other hardware, the report form asks for `fit`'s output and what you observed.

## Python SDK

```python
from omnillm.core.fit import analyze_model_fit, pull_quant
from omnillm import detect_hardware, estimate_model_memory

rec = analyze_model_fit("unsloth/Qwen3-8B-GGUF", context_length=8192, kv_cache_type="q8_0")
print(rec.recommended_quant, rec.best_target, rec.recommended.n_gpu_layers)
print(rec.format_table())
path, chosen = pull_quant(rec)            # download the recommendation

print(detect_hardware().summary())
total_gb, weights_gb, kv_gb = estimate_model_memory(70.0, quant="Q4_K_M", context_length=16384)
```

---

[← Back to README](../README.md)
