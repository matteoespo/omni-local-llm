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


def test_benchmark_prefers_backend_reported_token_counts():
    from omnillm import Usage

    exact = RecordingBackend(
        chunks=(ChatChunk(content="Hello"), ChatChunk(content=" world", finish_reason="stop", usage=Usage(11, 40)))
    )
    estimated = RecordingBackend(chunks=(ChatChunk(content="a longer chunk of text"), ChatChunk(content="b")))
    harness = BenchmarkHarness(LocalLLMManager({"exact": exact, "est": estimated}))

    suite = harness.run(targets=[("exact", "m"), ("est", "m")], prompt="Hi", runs=1, warmup=False)

    exact_run, estimated_run = suite.results[0].runs[0], suite.results[1].runs[0]
    assert (exact_run.prompt_tokens, exact_run.completion_tokens, exact_run.token_counts_exact) == (11, 40, True)
    assert (estimated_run.completion_tokens, estimated_run.token_counts_exact) == (2, False)


class ScriptedBackend(RecordingBackend):
    """Answers each prompt with a fixed reply, so one suite can mix passing and failing cases."""

    def __init__(self, replies: dict[str, str]):
        super().__init__()
        self.replies = replies

    def chat(self, request):
        self.requests.append(request)
        return ChatResponse(content=self.replies[request.messages[-1]["content"]])

    async def achat(self, request):
        return self.chat(request)


PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"],
}


async def test_expected_json_schema_is_enforced_and_validated():
    backend = ScriptedBackend(
        {
            "ok": '{"name": "Bob", "age": 34}',
            "fenced": '```json\n{"name": "Bob", "age": 34}\n```',
            "negative": '{"name": "Bob", "age": -1}',
            "missing": '{"name": "Bob"}',
            "prose": "Bob is 34.",
        }
    )
    harness = EvalHarness(LocalLLMManager({"fake": backend}))
    cases = [EvalTestCase(name=p, prompt=p, expected_json_schema=PERSON_SCHEMA) for p in backend.replies]

    for result in (harness.run_suite("fake", "m", cases), await harness.arun_suite("fake", "m", cases)):
        outcome = {r.test_case.name: (r.passed, r.error or "") for r in result.results}
        assert outcome["ok"] == (True, "") and outcome["fenced"] == (True, "")
        assert not outcome["negative"][0] and "at $.age" in outcome["negative"][1]
        assert not outcome["missing"][0] and "'age' is a required property" in outcome["missing"][1]
        assert outcome["prose"] == (False, "Output is not valid JSON")

    request = backend.requests[0]
    assert request.json_mode is True and request.json_schema == PERSON_SCHEMA  # enforced at the sampler


CITY = {"city": "Lyon", "country": "France"}


@pytest.mark.parametrize(
    ("expected", "reply", "error"),
    [
        (CITY, '{"city": "Lyon", "country": "France", "extra": 1}', None),  # extra keys are fine
        (CITY, '{"city": "Lyon", "country": "Italy"}', '$.country: expected "France", got "Italy"'),
        (CITY, '{"city": "Lyon"}', "$.country is missing"),
        ({"tags": ["a"]}, '{"tags": ["a"]}', None),
        ({"tags": ["a"]}, '{"tags": ["a", "b"]}', '$.tags: expected ["a"], got ["a", "b"]'),  # lists match exactly
        ({"ok": True}, '{"ok": 1}', "$.ok: expected true, got 1"),  # booleans never equal numbers
        ({"n": 2}, '{"n": 2.0}', None),
    ],
)
def test_expected_json_checks_values_and_requests_json_mode(expected, reply, error):
    backend = ScriptedBackend({"p": reply})

    result = EvalHarness(LocalLLMManager({"fake": backend})).run_suite(
        "fake", "m", [EvalTestCase(name="t", prompt="p", expected_json=expected)]
    )

    assert backend.requests[0].json_mode is True
    assert result.results[0].passed is (error is None), result.results[0].error
    if error is not None:
        assert error in (result.results[0].error or "")


def test_pydantic_and_json_schema_checks_are_exclusive():
    class Person(BaseModel):
        name: str

    with pytest.raises(ValueError, match="not both"):
        EvalTestCase(name="t", prompt="p", expected_schema=Person, expected_json_schema=PERSON_SCHEMA)


def test_jsonl_suite_loads_json_checks_and_reports_bad_lines(tmp_path: Path):
    import json

    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"name": "extract", "prompt": "Bob, 34", "expected_json_schema": PERSON_SCHEMA})
        + "\n"
        + json.dumps({"name": "values", "prompt": "Lyon", "expected_json": {"country": "France"}})
        + "\n"
    )
    cases = EvalHarness.load_jsonl_suite(good)
    assert cases[0].expected_json_schema == PERSON_SCHEMA
    assert cases[1].expected_json == {"country": "France"}

    for line, message in [
        (json.dumps({"prompt": "x", "expected_json_schema": {"type": "objekt"}}), "invalid expected_json_schema"),
        (json.dumps({"prompt": "x", "expected_json_schema": "object"}), "must be a JSON object"),
        (json.dumps({"name": "no prompt"}), "with a 'prompt'"),
        ("{not json", "Expecting property name"),
    ]:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"prompt": "fine"}\n\n' + line + "\n")
        with pytest.raises(ValueError, match=message) as excinfo:
            EvalHarness.load_jsonl_suite(bad)
        assert f"{bad}:3:" in str(excinfo.value)  # points at the offending line


def test_json_schema_without_the_eval_extra_gives_an_install_hint(tmp_path: Path, monkeypatch):
    import json
    import sys

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    suite = tmp_path / "s.jsonl"
    suite.write_text(json.dumps({"prompt": "x", "expected_json_schema": PERSON_SCHEMA}) + "\n")

    with pytest.raises(ImportError, match=r"omni-local-llm\[eval\]"):
        EvalHarness.load_jsonl_suite(suite)

    backend = ScriptedBackend({"x": '{"name": "Bob", "age": 1}'})
    result = EvalHarness(LocalLLMManager({"fake": backend})).run_suite(
        "fake", "m", [EvalTestCase(name="t", prompt="x", expected_json_schema=PERSON_SCHEMA)]
    )
    assert "omni-local-llm[eval]" in (result.results[0].error or "")


def test_per_case_options_override_the_checks_request_options():
    backend = ScriptedBackend({"p": '{"a": 1}'})
    case = EvalTestCase(name="t", prompt="p", expected_json={"a": 1}, options={"json_mode": False, "tools": []})

    result = EvalHarness(LocalLLMManager({"fake": backend})).run_suite("fake", "m", [case])

    assert result.results[0].passed, result.results[0].error  # no "multiple values for keyword argument"
    assert backend.requests[0].json_mode is False
