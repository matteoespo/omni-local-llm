from pathlib import Path

import pytest
from pydantic import BaseModel

from omnillm import ChatChunk, ChatResponse, LocalLLMManager
from omnillm.harness.benchmark import BenchmarkHarness, BenchmarkTarget
from omnillm.harness.eval import EvalHarness, EvalTestCase
from omnillm.harness.needle import NeedleHarness
from tests.fakes import RecordingBackend


def test_benchmark_target_spec():
    t1 = BenchmarkTarget.from_spec("ollama/llama3")
    assert t1.backend == "ollama"
    assert t1.model == "llama3"

    t2 = BenchmarkTarget.from_spec("mistral")
    assert t2.backend == "ollama"
    assert t2.model == "mistral"

    t3 = BenchmarkTarget.from_spec(("llama.cpp", "model", {"filename": "m.gguf"}))
    assert t3.backend == "llama.cpp"
    assert t3.model == "model"
    assert t3.options == {"filename": "m.gguf"}


def test_benchmark_harness_sync_and_async():
    backend = RecordingBackend(
        chunks=(ChatChunk(content="Hello "), ChatChunk(content="world!"), ChatChunk(content=" Finished."))
    )
    manager = LocalLLMManager({"fake": backend})
    harness = BenchmarkHarness(manager)

    suite = harness.run(
        targets=[("fake", "test-model")],
        prompt="Hi",
        runs=2,
        warmup=False,
    )

    assert len(suite.results) == 1
    res = suite.results[0]
    assert res.target.model == "test-model"
    assert len(res.runs) == 2
    assert res.mean_ttft_ms >= 0.0
    assert res.mean_tokens_per_second >= 0.0

    table = suite.format_table()
    assert "fake" in table
    assert "test-model" in table

    d = suite.to_dict()
    assert "benchmarks" in d
    assert d["benchmarks"][0]["model"] == "test-model"


@pytest.mark.asyncio
async def test_benchmark_harness_async():
    backend = RecordingBackend(chunks=(ChatChunk(content="Async "), ChatChunk(content="benchmark")))
    manager = LocalLLMManager({"fake": backend})
    harness = BenchmarkHarness(manager)

    suite = await harness.arun(
        targets=[("fake", "async-model")],
        runs=1,
        warmup=False,
    )

    assert len(suite.results) == 1
    assert len(suite.results[0].runs) == 1


def test_eval_harness_regex_and_schema():
    class UserInfo(BaseModel):
        name: str
        age: int

    backend = RecordingBackend(
        response=ChatResponse(
            content='{"name": "Alice", "age": 30}',
            parsed=UserInfo(name="Alice", age=30),
        )
    )
    manager = LocalLLMManager({"fake": backend})
    harness = EvalHarness(manager)

    test_cases = [
        EvalTestCase(
            name="Name pattern test",
            prompt="Who are you?",
            expected_pattern=r"Alice",
        ),
        EvalTestCase(
            name="Schema test",
            prompt="Extract user",
            expected_schema=UserInfo,
        ),
        EvalTestCase(
            name="Failing regex test",
            prompt="Who are you?",
            expected_pattern=r"Bob",
        ),
    ]

    suite_result = harness.run_suite("fake", "test-model", test_cases)
    assert suite_result.total_count == 3
    assert suite_result.passed_count == 2
    assert suite_result.failed_count == 1
    assert round(suite_result.pass_rate, 1) == 66.7

    table = suite_result.format_table()
    assert "PASS" in table
    assert "FAIL" in table

    d = suite_result.to_dict()
    assert d["total_count"] == 3


def test_eval_harness_tool_calls():
    backend = RecordingBackend(
        response=ChatResponse(
            content="Calling tool",
            tool_calls=({"id": "1", "function": {"name": "calculator"}},),
        )
    )
    manager = LocalLLMManager({"fake": backend})
    harness = EvalHarness(manager)

    tc = [
        EvalTestCase(
            name="Tool test pass",
            prompt="Calculate",
            expected_tool_call="calculator",
        ),
        EvalTestCase(
            name="Tool test fail",
            prompt="Calculate",
            expected_tool_call="search",
        ),
    ]

    res = harness.run_suite("fake", "model", tc)
    assert res.results[0].passed is True
    assert res.results[1].passed is False
    assert "Expected tool call 'search'" in (res.results[1].error or "")


def test_eval_harness_load_jsonl(tmp_path: Path):
    jsonl_file = tmp_path / "suite.jsonl"
    jsonl_file.write_text(
        '{"name": "T1", "prompt": "Hello", "expected_pattern": "Hi"}\n'
        '{"name": "T2", "prompt": "Calc", "expected_tool_call": "add"}\n'
    )

    cases = EvalHarness.load_jsonl_suite(jsonl_file)
    assert len(cases) == 2
    assert cases[0].name == "T1"
    assert cases[0].expected_pattern == "Hi"
    assert cases[1].expected_tool_call == "add"


def test_needle_harness():
    secret = "GOLDEN-KEY-999"
    backend = RecordingBackend(response=ChatResponse(content=f"The secret code is {secret}."))
    manager = LocalLLMManager({"fake": backend})
    harness = NeedleHarness(manager)

    result = harness.run(
        backend="fake",
        model="model",
        secret_code=secret,
        depths=(0, 50, 100),
        target_word_count=200,
    )

    assert len(result.trials) == 3
    assert result.score == 100.0
    for trial in result.trials:
        assert trial.retrieved is True

    table = result.format_table()
    assert "FOUND" in table
    assert "100.0%" in table
