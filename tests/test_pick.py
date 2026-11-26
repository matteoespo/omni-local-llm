import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from omnillm import LocalLLMManager, OpenAICompatibleAdapter
from omnillm.adapters.ollama_adapter import ALL_GPU_LAYERS, OllamaAdapter
from omnillm.core.config import load_config
from omnillm.core.errors import InvalidRequestError
from omnillm.core.hardware import HardwareProfile, recommend_model_fit
from omnillm.core.types import ModelTarget
from omnillm.harness import benchmark as benchmark_module
from omnillm.harness.benchmark import BenchmarkMetrics, BenchmarkResult, BenchmarkTarget
from omnillm.harness.eval import EvalSuiteResult, EvalTestCase, EvalTestResult
from omnillm.harness.pick import (
    PickCandidate,
    PickHarness,
    choose,
    discover_candidates,
    format_plan,
    mark_pareto,
    save_pick_alias,
    tie_margin,
)

HW = HardwareProfile(
    "Linux",
    "6.8",
    "x86_64",
    8,
    16.0,
    14.0,
    gpu_name="GTX 1650",
    gpu_count=1,
    total_vram_gb=4.0,
    free_vram_gb=3.8,
    has_cuda=True,
)

# ---------------------------------------------------------------- ranking


def measured(ref: str, passed: int, total: int = 12, tps: float = 10.0, memory: float | None = 4.0) -> PickCandidate:
    """A candidate that finished: `passed` of `total` cases, `tps` decode speed, `memory` GB."""
    case = EvalTestCase(name="c", prompt="p")
    candidate = PickCandidate(ref=ref, target=ModelTarget("ollama", ref), status="ok")
    candidate.eval = EvalSuiteResult(
        "ollama", ref, [EvalTestResult(case, passed=i < passed, latency_ms=1.0) for i in range(total)]
    )
    run = BenchmarkMetrics(
        ttft_ms=200.0, total_time_ms=1000.0, prompt_tokens=8, completion_tokens=64, tokens_per_second=tps
    )
    candidate.bench = BenchmarkResult(BenchmarkTarget("ollama", ref), [run])
    if memory is not None:
        fit = recommend_model_fit(ref, param_count_b=memory / 0.7, hardware=HW)
        candidate.quant = fit.results[0]
        candidate.quant.required_memory_gb = memory
    return candidate


@pytest.mark.parametrize(("cases", "margin"), [(0, 0), (1, 0), (3, 1), (12, 2), (50, 5), (100, 7)])
def test_tie_margin_is_about_one_standard_error(cases: int, margin: int):
    assert tie_margin(cases) == margin


def test_pareto_front_flags_dominated_candidates():
    best = measured("qwen", 11, tps=12.0, memory=5.4)
    fast = measured("llama", 9, tps=55.9, memory=3.3)
    dominated = measured("llama31", 10, tps=11.4, memory=6.2)  # worse than qwen on every axis
    mark_pareto([best, fast, dominated])
    assert (best.pareto, fast.pareto, dominated.pareto) == (True, True, False)


def test_balanced_takes_the_fastest_model_tied_on_quality():
    best = measured("qwen", 11, tps=12.0)
    fast = measured("llama", 9, tps=55.9)  # 2 fewer passes: within the margin of a 12-case suite
    winner, reason = choose([best, fast], "balanced", margin=2)
    assert winner is fast
    assert "within the tie margin (±2) of the best (qwen, 11/12)" in reason and "4.7× faster" in reason

    winner, _ = choose([best, fast], "balanced", margin=1)
    assert winner is best  # outside a narrower margin, quality wins


def test_quality_takes_most_passes_and_names_a_faster_rival_within_noise():
    best = measured("qwen", 11, tps=12.0)
    fast = measured("llama", 9, tps=55.9)
    winner, reason = choose([best, fast], "quality", margin=2)
    assert winner is best
    assert "Highest pass rate (11/12)." in reason
    assert "llama is 4.7× faster with 2 fewer passes, within the noise of a 12-case suite" in reason


def test_exact_quality_ties_go_to_the_faster_model():
    slow, quick = measured("slow", 10, tps=5.0), measured("quick", 10, tps=20.0)
    for prefer in ("quality", "balanced"):
        winner, reason = choose([slow, quick], prefer, margin=0)  # type: ignore[arg-type]
        assert winner is quick and "fastest" in reason


def test_speed_takes_the_fastest_model_above_min_pass():
    best = measured("qwen", 11, tps=12.0)
    fast = measured("llama", 7, tps=55.9)
    useless = measured("tiny", 2, tps=300.0)  # fastest, but below --min-pass
    winner, reason = choose([best, fast, useless], "speed", margin=2, min_pass=0.5)
    assert winner is fast
    assert "Fastest model (55.9 tok/s) passing at least 50% (7/12). qwen passes 11/12." in reason


def test_no_recommendation_below_min_pass():
    winner, reason = choose([measured("a", 3), measured("b", 5)], "quality", min_pass=0.5)
    assert winner is None
    assert "No candidate passes at least 50% of the suite (best: b, 5/12). Lower --min-pass" in reason
    assert choose([measured("b", 5)], min_pass=0.4)[0] is not None


def test_no_recommendation_without_measured_candidates():
    skipped = PickCandidate(ref="x", target=ModelTarget("ollama", "x"), status="skipped")
    assert choose([skipped]) == (None, "No candidate finished the suite.")


# ------------------------------------------------------- pipeline (fake Ollama)


class Clock:
    """A fake perf_counter, advanced by the fake server as it generates tokens."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeOllamaServer:
    """An Ollama client serving a few models with fixed answers and speeds."""

    class ResponseError(Exception):
        def __init__(self, status_code: int, error: str = ""):
            super().__init__(error)
            self.status_code = status_code
            self.error = error

    def __init__(self, clock: Clock, models: dict[str, dict[str, Any]]):
        self.clock = clock
        self.models = models  # name -> {"answers": {prompt: reply}, "tps": float, "fail": str?}
        self.calls: list[dict[str, Any]] = []

    def show(self, model: str) -> dict[str, Any]:
        if model not in self.models:
            raise self.ResponseError(404, f"model '{model}' not found")
        return {}

    def list(self) -> dict[str, Any]:
        return {"models": [{"model": name} for name in self.models]}

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        spec = self.models[kwargs["model"]]
        if spec.get("fail"):
            raise self.ResponseError(500, spec["fail"])
        if not kwargs["stream"]:
            prompt = kwargs["messages"][-1]["content"]
            return {"message": {"content": spec["answers"].get(prompt, "OK")}, "done": True}
        return self._stream(spec["tps"], kwargs.get("options", {}).get("num_predict", 16))

    def _stream(self, tps: float, tokens: int) -> Iterator[dict[str, Any]]:
        self.clock.now += 0.25  # time to first token
        for _ in range(tokens):
            yield {"message": {"content": "tok "}}
            self.clock.now += 1 / tps
        yield {"message": {"content": ""}, "done": True, "eval_count": tokens + 1, "prompt_eval_count": 9}


SUITE = [
    EvalTestCase(name="capital", prompt="Capital of France?", expected_pattern="Paris"),
    EvalTestCase(name="sum", prompt="45 + 55?", expected_pattern="100"),
    EvalTestCase(name="color", prompt="Sky color?", expected_pattern="blue"),
]
ALL_RIGHT = {"Capital of France?": "Paris", "45 + 55?": "100", "Sky color?": "blue"}
ONE_WRONG = {**ALL_RIGHT, "Sky color?": "green"}
SIZES_B = {"big:7b": 7.0, "small:1b": 1.0, "huge:70b": 70.0, "broken:3b": 3.0}


def fake_fit(ref: str, *, context_length: int, manager: Any, hardware: HardwareProfile, kv_cache_type: str):
    model = manager.resolve(ref).model
    return recommend_model_fit(model, context_length, param_count_b=SIZES_B.get(model, 3.0), hardware=hardware)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(benchmark_module.time, "perf_counter", fake)
    return fake


def fake_manager(clock: Clock, **extra_models: dict[str, Any]) -> tuple[LocalLLMManager, FakeOllamaServer]:
    server = FakeOllamaServer(
        clock,
        {
            "big:7b": {"answers": ALL_RIGHT, "tps": 12.0},
            "small:1b": {"answers": ONE_WRONG, "tps": 60.0},
            **extra_models,
        },
    )
    return LocalLLMManager({"ollama": OllamaAdapter(client=server)}), server


def test_discovery_lists_aliases_then_installed_ollama_models_without_duplicates(clock: Clock):
    manager, _ = fake_manager(clock)
    manager.register_backend("llama.cpp", lambda: OpenAICompatibleAdapter("http://127.0.0.1:1"))
    manager.register_alias("fast", model="ollama/small:1b", n_ctx=2048)
    manager.register_alias("remote", backend="llama.cpp", model="whatever")

    # small:1b is covered by the alias; the OpenAI-compatible backend is never listed.
    assert discover_candidates(manager) == ["fast", "remote", "ollama/big:7b"]


def test_plan_skips_what_does_not_fit_and_respects_max_candidates(clock: Clock):
    manager, _ = fake_manager(clock)
    harness = PickHarness(manager, hardware=HW, analyze=fake_fit)

    plan = harness.plan(["ollama/big:7b", "ollama/huge:70b", "ollama/nomic-embed-text", "ollama/small:1b"])
    status = {c.ref: (c.status, c.reason) for c in plan}
    assert status["ollama/big:7b"] == ("pending", "")
    assert status["ollama/huge:70b"][0] == "skipped" and "doesn't fit at context 4,096" in status["ollama/huge:70b"][1]
    assert status["ollama/nomic-embed-text"] == ("skipped", "embedding model (can't chat)")

    plan = harness.plan(["ollama/big:7b", "ollama/huge:70b", "ollama/small:1b"], max_candidates=1)
    assert [c.status for c in plan] == ["pending", "skipped", "skipped"]
    assert plan[2].reason == "left out by --max-candidates 1"
    assert "will run" in format_plan(plan, HW, 4096)


def test_models_on_other_servers_are_not_filtered_by_local_fit(clock: Clock):
    manager, _ = fake_manager(clock)
    manager.register_backend("lmstudio", lambda: OpenAICompatibleAdapter("http://127.0.0.1:1"))
    harness = PickHarness(manager, hardware=HW, analyze=fake_fit)
    [candidate] = harness.plan(["lmstudio/huge:70b"])
    assert candidate.status == "pending" and candidate.quant is None
    assert candidate.placement == "server" and candidate.runtime_options(4096) == {}


def test_run_measures_ranks_and_recommends(clock: Clock):
    manager, server = fake_manager(clock, **{"broken:3b": {"fail": "out of memory", "tps": 1.0}})
    progress: list[str] = []
    harness = PickHarness(manager, hardware=HW, analyze=fake_fit)
    plan = harness.plan(
        ["ollama/big:7b", "ollama/small:1b", "ollama/broken:3b", "ollama/missing:7b"], context_length=2048
    )
    report = harness.run(plan, SUITE, context_length=2048, prefer="balanced", bench_runs=2, progress=progress.append)

    big, small, broken, missing = report.candidates
    assert (big.passed, small.passed) == (3, 2)
    assert big.tokens_per_second == pytest.approx(12.0) and small.tokens_per_second == pytest.approx(60.0)
    assert big.ttft_ms == pytest.approx(250.0)
    assert broken.status == "failed" and "out of memory" in broken.reason
    assert missing.status == "failed" and missing.reason.startswith("not downloaded")

    # 3 cases → tie margin 1, so small (2/3, 5× faster) is tied with big (3/3) on quality.
    assert report.margin == 1 and report.recommended is small
    assert big.pareto and small.pareto

    # Every request ran at fit's placement, at temperature 0, and never pulled anything.
    measured_calls = [c for c in server.calls if c["model"] in ("big:7b", "small:1b")]
    assert all(c["options"]["num_ctx"] == 2048 and c["options"]["temperature"] == 0.0 for c in measured_calls)
    assert {c["options"]["num_gpu"] for c in server.calls if c["model"] == "small:1b"} == {ALL_GPU_LAYERS}
    big_layers = {c["options"]["num_gpu"] for c in server.calls if c["model"] == "big:7b"}
    assert big_layers == {big.quant.n_gpu_layers} and 0 < big.quant.n_gpu_layers < ALL_GPU_LAYERS  # type: ignore[union-attr]

    assert progress[0] == "[1/4] ollama/big:7b (GPU+CPU (" + progress[0].split("GPU+CPU (")[1]
    assert any("left" in line for line in progress)

    table = report.format_table()
    assert "→ ollama/small:1b" in table and "★ best quality" in table and "★ fastest" in table
    assert "Not measured:" in table and "ollama/broken:3b: Ollama error: out of memory" in table
    assert "Recommended (balanced): ollama/small:1b." in table

    data = json.loads(json.dumps(report.to_dict()))
    assert data["recommended"] == "ollama/small:1b" and data["tie_margin"] == 1
    small_row = next(row for row in data["candidates"] if row["ref"] == "ollama/small:1b")
    assert small_row["failures"] == [{"name": "color", "error": "Output did not match pattern 'blue'"}]


def test_alias_presets_are_kept_but_fallbacks_never_stand_in(clock: Clock):
    manager, server = fake_manager(clock)
    manager.register_alias("coder", model="ollama/big:7b", fallbacks=["ollama/small:1b"], top_p=0.9, max_tokens=64)
    harness = PickHarness(manager, hardware=HW, analyze=fake_fit)
    report = harness.run(harness.plan(["coder"]), SUITE, bench_runs=1)

    assert report.candidates[0].status == "ok"
    assert {c["model"] for c in server.calls} == {"big:7b"}
    eval_calls = [c for c in server.calls if not c["stream"] and c["options"].get("num_predict") != 8]
    assert all(c["options"]["top_p"] == 0.9 and c["options"]["num_predict"] == 64 for c in eval_calls)
    bench_calls = [c for c in server.calls if c["stream"]]
    assert bench_calls and all(c["options"]["num_predict"] == 128 for c in bench_calls)


def test_save_pick_alias_writes_the_measured_runtime(clock: Clock, tmp_path: Path):
    manager, _ = fake_manager(clock)
    manager.register_alias("coder", model="ollama/big:7b", temperature=0.2)
    harness = PickHarness(manager, hardware=HW, analyze=fake_fit)
    [candidate] = harness.plan(["coder"], context_length=8192)

    config = tmp_path / "omnillm.toml"
    save_pick_alias(candidate, 8192, "picked", config)
    [alias] = load_config(config).aliases
    assert (alias.backend, alias.model) == ("ollama", "big:7b")
    assert dict(alias.options) == {
        "temperature": 0.2,
        "n_ctx": 8192,
        "n_gpu_layers": candidate.quant.n_gpu_layers,  # type: ignore[union-attr]
    }
    with pytest.raises(InvalidRequestError, match="already exists"):
        save_pick_alias(candidate, 8192, "picked", config)


def test_run_rejects_an_empty_suite_and_unknown_preference(clock: Clock):
    harness = PickHarness(fake_manager(clock)[0], hardware=HW, analyze=fake_fit)
    with pytest.raises(ValueError, match="at least one test case"):
        harness.run([], [])
    with pytest.raises(ValueError, match="prefer must be one of"):
        harness.run([], SUITE, prefer="cheap")  # type: ignore[arg-type]


# --------------------------------------------------------------------- CLI


def run_cli(monkeypatch: pytest.MonkeyPatch, manager: LocalLLMManager, *argv: str) -> None:
    from omnillm import __main__ as cli
    from omnillm.harness import pick as pick_module

    monkeypatch.setattr(sys, "argv", ["omnillm", *argv])
    monkeypatch.setattr(cli, "build_manager", lambda _args: manager)
    monkeypatch.setattr(pick_module, "detect_hardware", lambda: HW)
    monkeypatch.setattr(pick_module, "analyze_model_fit", fake_fit)
    cli.main()


def write_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "tasks.jsonl"
    suite.write_text(
        "\n".join(
            json.dumps({"name": c.name, "prompt": c.prompt, "expected_pattern": c.expected_pattern}) for c in SUITE
        )
    )
    return suite


def test_cli_pick_prints_the_table_and_saves_the_winner(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    manager, _ = fake_manager(clock)
    config = tmp_path / "omnillm.toml"
    config.write_text('default_backend = "ollama"\n')
    suite = write_suite(tmp_path)
    run_cli(
        monkeypatch, manager, "pick", "--config", str(config), "--suite", str(suite), "--prefer", "quality",
        "--bench-runs", "1", "--save-alias", "best",
    )  # fmt: skip
    out, err = capsys.readouterr()
    assert "Measuring 2 candidate(s)" in err and "[2/2] ollama/small:1b" in err
    assert "Recommended (quality): ollama/big:7b." in out
    assert "✅ Saved alias 'best'" in out
    [alias] = load_config(config).aliases
    assert (alias.name, alias.backend, alias.model) == ("best", "ollama", "big:7b")


def test_cli_pick_json_and_dry_run(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    manager, server = fake_manager(clock)
    suite = write_suite(tmp_path)

    run_cli(
        monkeypatch, manager, "pick", "--suite", str(suite), "--dry-run", "--models", "ollama/big:7b,ollama/huge:70b"
    )
    out, _ = capsys.readouterr()
    assert "will run" in out and "doesn't fit" in out and "Would measure 1 candidate(s) on 3 case(s)" in out
    assert server.calls == []  # a dry run measures nothing

    run_cli(monkeypatch, manager, "pick", "--suite", str(suite), "--json", "--bench-runs", "1", "--min-pass", "100%")
    data = json.loads(capsys.readouterr().out)
    assert data["recommended"] == "ollama/big:7b" and data["min_pass"] == 1.0 and data["saved_alias"] is None


def test_cli_pick_errors(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    manager, _ = fake_manager(clock)
    suite = write_suite(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, manager, "pick", "--suite", str(tmp_path / "nope.jsonl"))
    assert exit_info.value.code == 2 and "Cannot load suite" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, manager, "pick", "--suite", str(suite), "--models", "ollama/huge:70b")
    assert exit_info.value.code == 1 and "Nothing to measure" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, manager, "pick", "--suite", str(suite), "--min-pass", "2")
    assert exit_info.value.code == 2 and "expected a number between 0 and 1" in capsys.readouterr().err

    empty = LocalLLMManager()
    empty.register_backend("ollama", lambda: OllamaAdapter(client=FakeOllamaServer(clock, {})))
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, empty, "pick", "--suite", str(suite))
    assert exit_info.value.code == 1 and "No candidates" in capsys.readouterr().err
