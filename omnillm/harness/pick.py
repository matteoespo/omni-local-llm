"""`omnillm pick`: which of the models that run on this machine is best for *your* tasks?

Pipeline: candidates (configured aliases and installed Ollama models, or an explicit list) → `fit` filter
(placement, memory and GPU layers at one context length) → load check → your eval suite → a short decode
benchmark → Pareto front over quality × speed × memory → one recommendation, chosen by `prefer`.

Nothing is downloaded: every request is sent with ``local_files_only``.
"""

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from omnillm.core.errors import BackendUnavailableError, ModelNotDownloadedError, OmniLLMError
from omnillm.core.fit import analyze_model_fit
from omnillm.core.hardware import FitRecommendation, HardwareProfile, QuantFitResult, QuantFitStatus, detect_hardware
from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ModelTarget
from omnillm.harness.benchmark import BenchmarkHarness, BenchmarkResult, BenchmarkTarget
from omnillm.harness.eval import EvalHarness, EvalSuiteResult, EvalTestCase

Prefer = Literal["quality", "balanced", "speed"]
PREFER_CHOICES: tuple[Prefer, ...] = ("quality", "balanced", "speed")
Status = Literal["pending", "ok", "skipped", "failed"]

DEFAULT_CONTEXT = 4096
DEFAULT_MIN_PASS = 0.5
DEFAULT_BENCH_PROMPT = "Explain the theory of general relativity in three concise sentences."

#: Options pick sets on every request; never copied into a saved alias.
_MEASUREMENT_ONLY = frozenset({"local_files_only"})


def tie_margin(n_cases: int) -> int:
    """How many passes two candidates may differ by and still count as tied: about one standard error.

    The standard error of the difference between two pass rates is at most ``sqrt(0.5 / n)``, which is
    ``floor(sqrt(n / 2))`` cases: 2 for a 12-case suite, 5 for 50 cases, 7 for 100.
    """
    return int(math.sqrt(max(n_cases, 0) / 2))


@dataclass(slots=True)
class PickCandidate:
    """One model under test, with its fit analysis and measurements."""

    ref: str
    target: ModelTarget
    fit: FitRecommendation | None = None
    quant: QuantFitResult | None = None
    status: Status = "pending"
    reason: str = ""
    eval: EvalSuiteResult | None = None
    bench: BenchmarkResult | None = None
    pareto: bool = False
    elapsed_s: float = 0.0
    local: bool = True  # runs on this machine (Ollama, llama.cpp), so fit and runtime options apply

    @property
    def memory_gb(self) -> float | None:
        return self.quant.required_memory_gb if self.quant else None

    @property
    def memory_exact(self) -> bool:
        return bool(self.quant and self.quant.size_exact)

    @property
    def passed(self) -> int:
        return self.eval.passed_count if self.eval else 0

    @property
    def total(self) -> int:
        return self.eval.total_count if self.eval else 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def tokens_per_second(self) -> float:
        """Median decode speed over the benchmark runs (0 when not measured)."""
        if not self.bench or not self.bench.runs:
            return 0.0
        return statistics.median(run.tokens_per_second for run in self.bench.runs)

    @property
    def ttft_ms(self) -> float | None:
        if not self.bench or not self.bench.runs:
            return None
        return statistics.median(run.ttft_ms for run in self.bench.runs)

    @property
    def placement(self) -> str:
        if not self.local:
            return "server"
        if self.quant is None:
            return "?"
        if self.quant.status == QuantFitStatus.PARTIAL:
            layers = self.fit.architecture.n_layers if self.fit and self.fit.architecture else None
            return f"GPU+CPU ({self.quant.n_gpu_layers}/{layers})" if layers else "GPU+CPU"
        return {
            QuantFitStatus.GPU_FULL: "GPU",
            QuantFitStatus.METAL_UNIFIED: "Metal",
            QuantFitStatus.CPU_RAM: "CPU",
            QuantFitStatus.OOM: "doesn't fit",
        }[self.quant.status]

    def runtime_options(self, context_length: int) -> dict[str, Any]:
        """The context and GPU layers the fit analysis assumed (no layer count when fit was unavailable).

        Empty for models on another server, which sets its own context and placement.
        """
        if not self.local:
            return {}
        options: dict[str, Any] = {"n_ctx": context_length}
        if self.quant is not None and self.quant.n_gpu_layers is not None:
            options["n_gpu_layers"] = self.quant.n_gpu_layers
        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "backend": self.target.backend,
            "model": self.target.model,
            "alias": self.target.alias,
            "status": self.status,
            "reason": self.reason or None,
            "quant": self.quant.quant if self.quant else None,
            "placement": self.placement,
            "n_gpu_layers": self.quant.n_gpu_layers if self.quant else None,
            "memory_gb": self.memory_gb,
            "memory_exact": self.memory_exact,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "pareto": self.pareto,
            "elapsed_s": round(self.elapsed_s, 1),
            "failures": [
                {"name": r.test_case.name, "error": r.error}
                for r in (self.eval.results if self.eval else ())
                if not r.passed
            ],
        }


def _memory_or_inf(candidate: PickCandidate) -> float:
    return candidate.memory_gb if candidate.memory_gb is not None else math.inf


def _dominates(a: PickCandidate, b: PickCandidate) -> bool:
    """True when `a` is at least as good as `b` on quality, speed and memory, and better on one."""
    a_mem, b_mem = _memory_or_inf(a), _memory_or_inf(b)
    at_least = a.passed >= b.passed and a.tokens_per_second >= b.tokens_per_second and a_mem <= b_mem
    better = a.passed > b.passed or a.tokens_per_second > b.tokens_per_second or a_mem < b_mem
    return at_least and better


def mark_pareto(candidates: Sequence[PickCandidate]) -> None:
    """Sets `pareto` on each candidate that no other candidate dominates."""
    for candidate in candidates:
        candidate.pareto = not any(_dominates(other, candidate) for other in candidates if other is not candidate)


def _fastest_first(candidate: PickCandidate) -> tuple[float, float, int]:
    return (-candidate.tokens_per_second, _memory_or_inf(candidate), -candidate.passed)


def _speedup(fast: PickCandidate, slow: PickCandidate) -> str:
    if slow.tokens_per_second <= 0:
        return "faster"
    return f"{fast.tokens_per_second / slow.tokens_per_second:.1f}× faster"


def _passes(count: int) -> str:
    return f"{count} fewer pass" + ("" if count == 1 else "es")


def choose(
    candidates: Sequence[PickCandidate],
    prefer: Prefer = "balanced",
    *,
    margin: int = 0,
    min_pass: float = DEFAULT_MIN_PASS,
) -> tuple[PickCandidate | None, str]:
    """Picks one measured candidate and explains why.

    * ``quality``: the most passes; exact ties go to the faster model.
    * ``balanced``: the fastest model whose passes are within `margin` of the best (differences the suite
      is too small to tell apart don't cost speed).
    * ``speed``: the fastest model.

    Candidates passing less than `min_pass` of the suite are never recommended.
    """
    measured = [c for c in candidates if c.status == "ok"]
    if not measured:
        return None, "No candidate finished the suite."
    n = measured[0].total
    eligible = [c for c in measured if c.pass_rate >= min_pass]
    if not eligible:
        best = max(measured, key=lambda c: (c.passed, c.tokens_per_second))
        return None, (
            f"No candidate passes at least {min_pass:.0%} of the suite (best: {best.ref}, {best.passed}/{n}). "
            "Lower --min-pass to accept it."
        )
    best_passes = max(c.passed for c in eligible)
    top = [c for c in eligible if c.passed == best_passes]

    if prefer == "speed":
        winner = min(eligible, key=_fastest_first)
        floor = f" passing at least {min_pass:.0%}" if min_pass > 0 else ""
        reason = f"Fastest model ({winner.tokens_per_second:.1f} tok/s){floor} ({winner.passed}/{n})."
        if winner.passed < best_passes:
            leader = min(top, key=_fastest_first)
            reason += f" {leader.ref} passes {best_passes}/{n}."
        return winner, reason

    if prefer == "quality":
        winner = min(top, key=_fastest_first)
        reason = f"Highest pass rate ({best_passes}/{n})" + (
            ", and the fastest of those tied." if len(top) > 1 else "."
        )
        rival = min(
            (
                c
                for c in eligible
                if c.passed >= best_passes - margin and c.tokens_per_second > winner.tokens_per_second
            ),
            key=_fastest_first,
            default=None,
        )
        if rival is not None:
            reason += (
                f" {rival.ref} is {_speedup(rival, winner)} with {_passes(best_passes - rival.passed)}, "
                f"within the noise of a {n}-case suite."
            )
        return winner, reason

    band = [c for c in eligible if c.passed >= best_passes - margin]
    winner = min(band, key=_fastest_first)
    if winner.passed == best_passes:
        others = len(band) > 1
        reason = f"Highest pass rate ({best_passes}/{n})" + (", and the fastest of the close ones." if others else ".")
        return winner, reason
    leader = min(top, key=_fastest_first)
    return winner, (
        f"{winner.passed}/{n} is within the tie margin (±{margin}) of the best ({leader.ref}, {best_passes}/{n}), "
        f"and it is {_speedup(winner, leader)}."
    )


def _hardware_label(hardware: HardwareProfile) -> str:
    if hardware.has_cuda and hardware.gpu_name:
        return f"{hardware.gpu_name} {hardware.total_vram_gb:.0f} GB, {hardware.total_ram_gb:.0f} GB RAM"
    if hardware.is_apple_silicon:
        return f"{hardware.gpu_name or 'Apple Silicon'}, {hardware.total_ram_gb:.0f} GB unified memory"
    return f"CPU only, {hardware.total_ram_gb:.0f} GB RAM"


def _format_ttft(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


@dataclass(slots=True)
class PickReport:
    candidates: list[PickCandidate]
    prefer: Prefer
    context_length: int
    hardware: HardwareProfile
    suite_size: int
    margin: int
    min_pass: float
    recommended: PickCandidate | None
    reason: str
    suite_name: str | None = None

    @property
    def measured(self) -> list[PickCandidate]:
        """Candidates that finished, most passes first, then fastest."""
        done = [c for c in self.candidates if c.status == "ok"]
        return sorted(done, key=lambda c: (-c.passed, -c.tokens_per_second, _memory_or_inf(c)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.to_dict(),
            "context_length": self.context_length,
            "suite": self.suite_name,
            "suite_size": self.suite_size,
            "prefer": self.prefer,
            "tie_margin": self.margin,
            "min_pass": self.min_pass,
            "recommended": self.recommended.ref if self.recommended else None,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def format_table(self) -> str:
        suite = f"suite: {self.suite_name} ({self.suite_size} cases)" if self.suite_name else f"{self.suite_size} cases"
        lines = [
            f"Candidates ({_hardware_label(self.hardware)}, context {self.context_length:,}), {suite}",
            "",
        ]
        measured = self.measured
        if measured:
            best_passes = max(c.passed for c in measured)
            top_speed = max(c.tokens_per_second for c in measured)
            rows = []
            for c in measured:
                notes = []
                if c.passed == best_passes:
                    notes.append("★ best quality")
                if c.tokens_per_second == top_speed and top_speed > 0:
                    notes.append("★ fastest")
                if not c.pareto:
                    notes.append("(dominated)")
                memory = "?" if c.memory_gb is None else f"{'' if c.memory_exact else '~'}{c.memory_gb:.1f} GB"
                rows.append(
                    [
                        "→" if c is self.recommended else " ",
                        c.ref,
                        c.quant.quant if c.quant else "?",
                        c.placement,
                        memory,
                        f"{c.passed}/{c.total}",
                        _format_ttft(c.ttft_ms),
                        f"{c.tokens_per_second:.1f}",
                        "  ".join(notes),
                    ]
                )
            headers = ["", "MODEL", "QUANT", "PLACEMENT", "MEMORY", "PASS", "TTFT", "TOK/S", ""]
            widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
            right = {4, 5, 6, 7}

            def line(values: list[str]) -> str:
                cells = [v.rjust(widths[i]) if i in right else v.ljust(widths[i]) for i, v in enumerate(values)]
                return " ".join(cells).rstrip()

            lines.append(line(headers))
            lines.extend(line(row) for row in rows)
        else:
            lines.append("No candidate finished the suite.")

        not_measured = [c for c in self.candidates if c.status in ("skipped", "failed")]
        if not_measured:
            lines.append("")
            lines.append("Not measured:")
            lines.extend(f"  {c.ref}: {c.reason}" for c in not_measured)

        lines.append("")
        if self.recommended is not None:
            lines.append(f"Recommended ({self.prefer}): {self.recommended.ref}. {self.reason}")
        else:
            lines.append(f"No recommendation: {self.reason}")
        if measured:
            lines.append(
                f"Pass counts within {self.margin} of each other are within the noise of a {self.suite_size}-case "
                "suite. Memory is the fit estimate (~ when the file size is estimated)."
            )
        return "\n".join(lines)


def format_plan(candidates: Sequence[PickCandidate], hardware: HardwareProfile, context_length: int) -> str:
    """The candidates and what `fit` decided for each, before anything is measured (``--dry-run``)."""
    lines = [f"Candidates ({_hardware_label(hardware)}, context {context_length:,})", ""]
    rows = []
    for c in candidates:
        memory = "?" if c.memory_gb is None else f"{'' if c.memory_exact else '~'}{c.memory_gb:.1f} GB"
        verdict = "will run" if c.status == "pending" else c.status
        rows.append([c.ref, c.quant.quant if c.quant else "?", c.placement, memory, verdict, c.reason])
    headers = ["MODEL", "QUANT", "PLACEMENT", "MEMORY", "STATUS", ""]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        lines.append(" ".join(v.rjust(widths[i]) if i == 3 else v.ljust(widths[i]) for i, v in enumerate(row)).rstrip())
    return "\n".join(lines)


def _ollama_key(model: str) -> str:
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def discover_candidates(manager: LocalLLMManager) -> list[str]:
    """Configured aliases (in config order), then models installed on Ollama backends, without duplicates.

    Other backends aren't listed (an OpenAI-compatible server may serve hundreds of hosted models): reach
    them through aliases or an explicit list.
    """
    from omnillm.adapters.ollama_adapter import OllamaAdapter

    refs: list[str] = []
    seen: set[tuple[str, str, str | None]] = set()
    for name, target in manager.aliases.items():
        key = (target.backend, _ollama_key(target.model), target.options.get("filename"))
        if key not in seen:
            seen.add(key)
            refs.append(name)
    for backend in manager.backend_names:
        try:
            adapter = manager._get_backend(backend)
        except (BackendUnavailableError, ImportError):
            continue
        if not isinstance(adapter, OllamaAdapter):
            continue
        try:
            models = adapter.list_models()
        except BackendUnavailableError:
            continue
        for model in models:
            key = (backend, _ollama_key(model), None)
            if key not in seen:
                seen.add(key)
                refs.append(f"{backend}/{model}")
    return refs


def _quant_in_use(recommendation: FitRecommendation, target: ModelTarget) -> QuantFitResult | None:
    """The quant this target actually loads: the installed one, the alias's file, else fit's recommendation."""
    installed = next((r for r in recommendation.results if r.installed), None)
    if installed is not None:
        return installed
    filename = target.options.get("filename")
    if filename:
        chosen = next((r for r in recommendation.results if filename in r.files), None)
        if chosen is not None:
            return chosen
    return recommendation.recommended


def _failure_reason(error: Exception) -> str:
    if isinstance(error, ModelNotDownloadedError):
        return "not downloaded (pick never downloads models; pull it first)"
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    if not isinstance(error, OmniLLMError):
        text = f"{type(error).__name__}: {text}"
    return text if len(text) <= 160 else text[:157] + "..."


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


Analyze = Callable[..., FitRecommendation]


class PickHarness:
    """Runs the pick pipeline. Use `plan` to filter candidates with `fit`, then `run` to measure them."""

    def __init__(
        self,
        manager: LocalLLMManager | None = None,
        *,
        hardware: HardwareProfile | None = None,
        analyze: Analyze | None = None,
    ):
        self.manager = manager or LocalLLMManager()
        self.hardware = hardware
        self._analyze = analyze  # None: omnillm.core.fit.analyze_model_fit

    def plan(
        self,
        refs: Sequence[str],
        *,
        context_length: int = DEFAULT_CONTEXT,
        kv_cache_type: str = "f16",
        max_candidates: int | None = None,
    ) -> list[PickCandidate]:
        """Resolves each reference and keeps the ones `fit` says run here; the rest are marked skipped."""
        if self.hardware is None:
            self.hardware = detect_hardware()
        candidates: list[PickCandidate] = []
        for ref in dict.fromkeys(refs):
            candidate = PickCandidate(ref=ref, target=self.manager.resolve(ref))
            candidates.append(candidate)
            if "embed" in candidate.target.model.lower():
                candidate.status, candidate.reason = "skipped", "embedding model (can't chat)"
                continue
            if not self._runs_locally(candidate.target.backend):
                candidate.local = False
                candidate.reason = "served by another server: memory and placement unknown, not filtered by fit"
                continue
            try:
                candidate.fit = (self._analyze or analyze_model_fit)(
                    ref,
                    context_length=context_length,
                    manager=self.manager,
                    hardware=self.hardware,
                    kv_cache_type=kv_cache_type,
                )
            except (OmniLLMError, ValueError) as error:
                candidate.reason = f"fit unavailable ({_failure_reason(error)}); runs without a GPU layer count"
                continue
            candidate.quant = _quant_in_use(candidate.fit, candidate.target)
            if candidate.quant is None or not candidate.quant.fits:
                # Nothing fits: report the smallest quant's needs, the closest this model gets to fitting.
                smallest = candidate.quant or min(
                    candidate.fit.results, key=lambda r: r.required_memory_gb, default=None
                )
                needed = f": {smallest.quant} needs ~{smallest.required_memory_gb:.1f} GB" if smallest else ""
                candidate.status, candidate.reason = "skipped", f"doesn't fit at context {context_length:,}{needed}"
        if max_candidates is not None:
            runnable = [c for c in candidates if c.status == "pending"]
            for candidate in runnable[max_candidates:]:
                candidate.status, candidate.reason = "skipped", f"left out by --max-candidates {max_candidates}"
        return candidates

    def _runs_locally(self, backend: str) -> bool:
        """Whether `fit`'s analysis of this machine applies: Ollama and llama.cpp backends, not other servers."""
        from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter
        from omnillm.adapters.ollama_adapter import OllamaAdapter

        try:
            adapter = self.manager._get_backend(backend)
        except (BackendUnavailableError, ImportError):
            return True  # let fit (and then the load check) report the problem
        return isinstance(adapter, OllamaAdapter | LlamaCPPAdapter)

    def run(
        self,
        candidates: list[PickCandidate],
        test_cases: Sequence[EvalTestCase],
        *,
        context_length: int = DEFAULT_CONTEXT,
        prefer: Prefer = "balanced",
        min_pass: float = DEFAULT_MIN_PASS,
        bench_prompt: str = DEFAULT_BENCH_PROMPT,
        bench_tokens: int = 128,
        bench_runs: int = 3,
        seed: int = 0,
        suite_name: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> PickReport:
        """Measures every pending candidate (load check, suite, benchmark), then ranks them."""
        if not test_cases:
            raise ValueError("pick needs at least one test case.")
        if prefer not in PREFER_CHOICES:
            raise ValueError(f"prefer must be one of {PREFER_CHOICES}, not {prefer!r}.")
        if self.hardware is None:
            self.hardware = detect_hardware()
        say = progress or (lambda _message: None)
        runnable = [c for c in candidates if c.status == "pending"]
        for index, candidate in enumerate(runnable, start=1):
            size = f", ~{candidate.memory_gb:.1f} GB" if candidate.memory_gb is not None else ""
            say(f"[{index}/{len(runnable)}] {candidate.ref} ({candidate.placement}{size})".replace(" (?)", ""))
            self._measure(candidate, test_cases, context_length, bench_prompt, bench_tokens, bench_runs, seed)
            if candidate.status == "ok":
                result = f"{candidate.passed}/{candidate.total} passed, {candidate.tokens_per_second:.1f} tok/s"
            else:
                result = f"failed: {candidate.reason}"
            say(f"      {result} in {_duration(candidate.elapsed_s)}{self._eta(runnable, index)}")

        margin = tie_margin(len(test_cases))
        measured = [c for c in candidates if c.status == "ok"]
        mark_pareto(measured)
        winner, reason = choose(measured, prefer, margin=margin, min_pass=min_pass)
        return PickReport(
            candidates=candidates,
            prefer=prefer,
            context_length=context_length,
            hardware=self.hardware,
            suite_size=len(test_cases),
            margin=margin,
            min_pass=min_pass,
            recommended=winner,
            reason=reason,
            suite_name=suite_name,
        )

    @staticmethod
    def _eta(runnable: Sequence[PickCandidate], done: int) -> str:
        """Time left, scaling the time spent so far by model size (bigger models run slower)."""
        remaining = runnable[done:]
        if not remaining:
            return ""
        finished = runnable[:done]
        spent = sum(c.elapsed_s for c in finished)
        sized = all(c.memory_gb for c in runnable)
        if sized:
            per_gb = spent / sum(c.memory_gb or 0 for c in finished)
            left = per_gb * sum(c.memory_gb or 0 for c in remaining)
        else:
            left = spent / len(finished) * len(remaining)
        return f"; about {_duration(left)} left"

    def _measure(
        self,
        candidate: PickCandidate,
        test_cases: Sequence[EvalTestCase],
        context_length: int,
        bench_prompt: str,
        bench_tokens: int,
        bench_runs: int,
        seed: int,
    ) -> None:
        target = candidate.target
        # Alias presets, then pick's measurement settings, then the runtime fit assumed. Per-case options still
        # override these inside the eval harness. The primary target is called directly, so an alias's
        # fallbacks never stand in for the model being measured.
        options: dict[str, Any] = {
            **target.options,
            "temperature": 0.0,
            "seed": seed,
            "local_files_only": True,
            **candidate.runtime_options(context_length),
        }
        start = time.perf_counter()
        try:
            self.manager.chat(
                backend=target.backend,
                model=target.model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                **{**options, "max_tokens": 8},
            )
            candidate.eval = EvalHarness(self.manager).run_suite(target.backend, target.model, test_cases, **options)
            bench_options = {key: value for key, value in options.items() if key != "max_tokens"}
            suite = BenchmarkHarness(self.manager).run(
                [BenchmarkTarget(target.backend, target.model, bench_options)],
                prompt=bench_prompt,
                max_tokens=bench_tokens,
                runs=bench_runs,
                warmup=False,  # the load check above already warmed the model up
            )
        except Exception as error:  # any failure means this candidate can't be measured; keep going
            candidate.status, candidate.reason = "failed", _failure_reason(error)
        else:
            candidate.bench = suite.results[0]
            candidate.status = "ok"
        candidate.elapsed_s = time.perf_counter() - start


def pick_alias_table(candidate: PickCandidate, context_length: int, name: str) -> str:
    """The omnillm.toml alias for a candidate: its alias presets plus the runtime it was measured at."""
    from omnillm.core.config import alias_table

    options = {key: value for key, value in candidate.target.options.items() if key not in _MEASUREMENT_ONLY}
    options.update(candidate.runtime_options(context_length))
    return alias_table(name, candidate.target.model, backend=candidate.target.backend, options=options)


def save_pick_alias(candidate: PickCandidate, context_length: int, name: str, config_path: str | Path) -> Path:
    """Saves a measured candidate as an alias with the runtime it was measured at (context, GPU layers)."""
    from omnillm.core.config import append_alias

    return append_alias(config_path, pick_alias_table(candidate, context_length, name), name)
