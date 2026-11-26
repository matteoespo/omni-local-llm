"""Resolve any model reference to fit metadata, and download the recommended GGUF.

`omnillm fit` accepts a Hugging Face repo (GGUF or safetensors), an installed Ollama model, a config alias or
``backend/model`` reference, a local ``.gguf`` file, or just a name (estimated).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omnillm.core.errors import BackendUnavailableError, InvalidRequestError
from omnillm.core.hardware import FitRecommendation, HardwareProfile, QuantFitResult, recommend_model_fit
from omnillm.core.model_info import (
    ModelInfo,
    download_hf_files,
    estimated_model_info,
    gguf_file_info,
    hf_model_info,
    make_hf_client,
    ollama_model_info,
)

if TYPE_CHECKING:
    from omnillm.core.manager import LocalLLMManager

_HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


@dataclass(slots=True)
class FitTarget:
    info: ModelInfo
    preferred_quant_file: str | None = None
    context_length: int | None = None


def _default_ollama_client(manager: "LocalLLMManager | None") -> Any:
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    if manager is not None and manager.default_backend in manager.backend_names:
        try:
            adapter = manager._get_backend(manager.default_backend)
        except BackendUnavailableError:
            return None
        return adapter._client if isinstance(adapter, OllamaAdapter) else None
    try:
        import ollama
    except ImportError:
        return None
    return ollama


def _from_ollama(name: str, client: Any, offline: bool, param_count_b: float | None) -> ModelInfo:
    if not offline and client is not None:
        try:
            info = ollama_model_info(name, client)
        except Exception:
            info = None
        if info is not None:
            return info
    return estimated_model_info(name, param_count_b)


def _from_hf(repo: str, offline: bool, param_count_b: float | None, hf_client: Any | None) -> ModelInfo:
    if offline:
        return estimated_model_info(repo, param_count_b)
    try:
        return hf_model_info(repo, client=hf_client)
    except BackendUnavailableError as error:
        info = estimated_model_info(repo, param_count_b)
        info.notes.insert(0, f"{str(error).rstrip('.')}. Falling back to estimates.")
        return info


def resolve_fit_target(
    ref: str,
    *,
    manager: "LocalLLMManager | None" = None,
    offline: bool = False,
    param_count_b: float | None = None,
    hf_client: Any | None = None,
    ollama_client: Any | None = None,
) -> FitTarget:
    """Finds the most exact metadata source for a model reference."""
    path = Path(ref).expanduser()
    if path.suffix.lower() == ".gguf" and path.is_file():
        return FitTarget(gguf_file_info(path))

    if manager is not None:
        prefix = ref.partition("/")[0]
        if ref in manager.alias_names or ("/" in ref and prefix in manager.backend_names):
            return _from_manager_target(ref, manager, offline, param_count_b, hf_client, ollama_client)

    if _HF_REPO.match(ref):
        return FitTarget(_from_hf(ref, offline, param_count_b, hf_client))

    client = ollama_client if ollama_client is not None else (None if offline else _default_ollama_client(manager))
    return FitTarget(_from_ollama(ref, client, offline, param_count_b))


def _from_manager_target(
    ref: str,
    manager: "LocalLLMManager",
    offline: bool,
    param_count_b: float | None,
    hf_client: Any | None,
    ollama_client: Any | None,
) -> FitTarget:
    from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    target = manager.resolve(ref)
    n_ctx = target.options.get("n_ctx")
    try:
        adapter = manager._get_backend(target.backend)
    except BackendUnavailableError:
        adapter = None

    if isinstance(adapter, LlamaCPPAdapter):
        info = _from_hf(target.model, offline, param_count_b, hf_client)
        return FitTarget(info, preferred_quant_file=target.options.get("filename"), context_length=n_ctx)
    if isinstance(adapter, OllamaAdapter):
        client = ollama_client if ollama_client is not None else adapter._client
        return FitTarget(_from_ollama(target.model, client, offline, param_count_b), context_length=n_ctx)

    info = estimated_model_info(target.model, param_count_b)
    info.notes.insert(
        0,
        f"'{ref}' is served by backend '{target.backend}'; its memory use there depends on how that server "
        "loads it. Estimating from the model name.",
    )
    return FitTarget(info, context_length=n_ctx)


def analyze_model_fit(
    ref: str,
    *,
    context_length: int | None = None,
    manager: "LocalLLMManager | None" = None,
    hardware: HardwareProfile | None = None,
    kv_cache_type: str = "f16",
    offline: bool = False,
    param_count_b: float | None = None,
    hf_client: Any | None = None,
    ollama_client: Any | None = None,
) -> FitRecommendation:
    """One call from a model reference to a recommendation (see the module docstring for accepted references)."""
    owns_client = hf_client is None and not offline
    client = hf_client
    if owns_client:
        try:
            client = make_hf_client()
        except BackendUnavailableError:
            client, owns_client = None, False
    try:
        target = resolve_fit_target(
            ref,
            manager=manager,
            offline=offline,
            param_count_b=param_count_b,
            hf_client=client,
            ollama_client=ollama_client,
        )
    finally:
        if owns_client and client is not None:
            client.close()

    recommendation = recommend_model_fit(
        ref,
        context_length=context_length or target.context_length or 4096,
        param_count_b=param_count_b,
        hardware=hardware,
        info=target.info,
        kv_cache_type=kv_cache_type,
    )
    if target.preferred_quant_file:
        chosen = next((r for r in recommendation.results if target.preferred_quant_file in r.files), None)
        if chosen is not None:
            recommendation.notes.insert(
                0, f"Alias '{ref}' uses {chosen.quant}: {chosen.status.value} (~{chosen.required_memory_gb:.2f} GB)."
            )
    return recommendation


def select_result(recommendation: FitRecommendation, quant: str | None = None) -> QuantFitResult:
    if quant is None:
        if recommendation.recommended is None:
            raise InvalidRequestError("Nothing fits this machine, so there is nothing to download.")
        return recommendation.recommended
    for result in recommendation.results:
        if result.quant.upper() == quant.upper():
            return result
    available = ", ".join(r.quant for r in recommendation.results)
    raise InvalidRequestError(f"Quant '{quant}' is not available. Choose from: {available}")


def pull_quant(
    recommendation: FitRecommendation,
    quant: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, QuantFitResult]:
    """Downloads the recommended (or chosen) GGUF quant into the Hugging Face cache used by the llama.cpp backend."""
    if recommendation.source != "huggingface-gguf" or not recommendation.repo:
        raise InvalidRequestError(
            "--pull needs a Hugging Face GGUF repo (e.g. 'unsloth/Qwen3-8B-GGUF'). "
            "For Ollama models use `ollama pull <model>`."
        )
    result = select_result(recommendation, quant)
    if not result.files:
        raise InvalidRequestError(f"No files recorded for {result.quant}.")
    path = download_hf_files(recommendation.repo, result.files, revision=recommendation.revision, progress=progress)
    return path, result


def default_alias_name(recommendation: FitRecommendation) -> str:
    repo = recommendation.repo or recommendation.model_name
    return re.sub(r"(?i)[-_.]?gguf$", "", repo.split("/")[-1]).lower()


def _alias_options(recommendation: FitRecommendation, result: QuantFitResult) -> dict[str, Any]:
    return {
        "filename": result.files[0],
        "n_ctx": recommendation.context_length,
        "n_gpu_layers": result.n_gpu_layers,
    }


def alias_snippet(recommendation: FitRecommendation, result: QuantFitResult, name: str | None = None) -> str:
    """An omnillm.toml alias for a downloaded quant, with the context and GPU layers the analysis assumed."""
    from omnillm.core.config import alias_table

    return alias_table(
        name or default_alias_name(recommendation),
        recommendation.repo or recommendation.model_name,
        backend="llama.cpp",
        options=_alias_options(recommendation, result),
    )


def save_alias(recommendation: FitRecommendation, result: QuantFitResult, name: str, config_path: str | Path) -> Path:
    """Appends an alias for `result` to a config file (created if missing), validating before and after."""
    from omnillm.core.config import append_alias

    if not name or "/" in name:
        raise InvalidRequestError("Alias names must be non-empty and must not contain '/'.")
    if recommendation.source != "huggingface-gguf" or not result.files:
        raise InvalidRequestError("--save-alias needs a Hugging Face GGUF repo (e.g. 'unsloth/Qwen3-8B-GGUF').")
    return append_alias(config_path, alias_snippet(recommendation, result, name), name)
