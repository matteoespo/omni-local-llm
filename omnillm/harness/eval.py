import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ChatResponse

if TYPE_CHECKING:
    from pydantic import BaseModel  # only for type hints: pydantic is not a core dependency


@dataclass(frozen=True, slots=True)
class EvalTestCase:
    """A test case for evaluating LLM capabilities.

    Checks (all that are set must pass):

    * `expected_pattern`: a regex the output must contain (case-insensitive).
    * `expected_schema`: a Pydantic model the output must parse into (enforced at the sampler).
    * `expected_json_schema`: a JSON Schema dict the output must satisfy (enforced at the sampler, then
      validated; needs ``omni-local-llm[eval]``).
    * `expected_json`: values the output's JSON must contain. Objects may have extra keys; lists and scalars
      must match exactly. Requests JSON mode unless a schema is set.
    * `expected_tool_call`: the name of a tool the model must call.
    """

    name: str
    prompt: str
    system_prompt: str | None = None
    expected_pattern: str | None = None
    expected_schema: "type[BaseModel] | None" = None
    expected_tool_call: str | None = None
    tools: Sequence[dict[str, Any]] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    expected_json_schema: dict[str, Any] | None = None
    expected_json: Any = None

    def __post_init__(self) -> None:
        if self.expected_schema is not None and self.expected_json_schema is not None:
            raise ValueError(f"Test '{self.name}': set expected_schema or expected_json_schema, not both.")


JSONSCHEMA_MISSING = "expected_json_schema needs the jsonschema package: pip install 'omni-local-llm[eval]'"


def _json_schema_validator(schema: dict[str, Any]) -> Any:
    """A validator for `schema` (draft taken from its ``$schema``, default 2020-12); raises ValueError if invalid."""
    try:
        import jsonschema
    except ImportError as error:
        raise ImportError(JSONSCHEMA_MISSING) from error
    cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
    try:
        cls.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise ValueError(f"invalid expected_json_schema: {error.message}") from error
    return cls(schema)


def _json_schema_problem(schema: dict[str, Any], value: Any) -> str | None:
    """The most relevant schema violation in `value` as "<path>: <message>", or None if it is valid."""
    validator = _json_schema_validator(schema)  # first, so a missing package gets the install hint
    from jsonschema.exceptions import best_match

    best = best_match(validator.iter_errors(value))
    if best is None:
        return None
    where = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in best.absolute_path)
    return f"{where}: {best.message}"


def _parse_json_output(output: str) -> Any:
    """Parses model output as JSON, tolerating a surrounding Markdown code fence."""
    text = output.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_mismatch(expected: Any, actual: Any, path: str = "$") -> str | None:
    """Describes where `actual` doesn't contain `expected` (None if it does).

    Objects match when every expected key matches (extra keys are fine); lists must have the same length
    and match item by item; scalars must be equal, and booleans never equal numbers.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected an object, got {_short(actual)}"
        for key, value in expected.items():
            if key not in actual:
                return f"{path}.{key} is missing"
            if (problem := _json_mismatch(value, actual[key], f"{path}.{key}")) is not None:
                return problem
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path}: expected {_short(expected)}, got {_short(actual)}"
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            if (problem := _json_mismatch(want, got, f"{path}[{index}]")) is not None:
                return problem
        return None
    if isinstance(expected, bool) != isinstance(actual, bool) or expected != actual:
        return f"{path}: expected {_short(expected)}, got {_short(actual)}"
    return None


def _request_options(tc: EvalTestCase) -> dict[str, Any]:
    """Chat arguments that make the backend produce the output format a test case checks."""
    options: dict[str, Any] = {"response_model": tc.expected_schema, "tools": tc.tools}
    if tc.expected_json_schema is not None:
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "expected_output", "schema": tc.expected_json_schema},
        }
    elif tc.expected_json is not None and tc.expected_schema is None:
        options["json_mode"] = True
    return options


def _messages(tc: EvalTestCase) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if tc.system_prompt:
        messages.append({"role": "system", "content": tc.system_prompt})
    messages.append({"role": "user", "content": tc.prompt})
    return messages


def _check(tc: EvalTestCase, response: ChatResponse) -> str | None:
    """Runs a test case's checks against a response; returns why it failed, or None if it passed."""
    output = response.content
    if tc.expected_pattern and not re.search(tc.expected_pattern, output, flags=re.IGNORECASE):
        return f"Output did not match pattern '{tc.expected_pattern}'"

    if tc.expected_schema is not None and (
        response.parsed is None or not isinstance(response.parsed, tc.expected_schema)
    ):
        return f"Response did not match expected schema {tc.expected_schema.__name__}"

    if tc.expected_json_schema is not None or tc.expected_json is not None:
        try:
            value = _parse_json_output(output)
        except ValueError:
            return "Output is not valid JSON"
        if tc.expected_json_schema is not None and (problem := _json_schema_problem(tc.expected_json_schema, value)):
            return f"Output does not match expected_json_schema at {problem}"
        if tc.expected_json is not None and (problem := _json_mismatch(tc.expected_json, value)) is not None:
            return f"Output JSON differs from expected_json: {problem}"

    if tc.expected_tool_call:
        called_names = [call.get("function", {}).get("name") for call in response.tool_calls]
        if tc.expected_tool_call not in called_names:
            return f"Expected tool call '{tc.expected_tool_call}', but model called: {called_names}"
    return None


@dataclass(slots=True)
class EvalTestResult:
    """The result of executing one evaluation test case."""

    test_case: EvalTestCase
    passed: bool
    latency_ms: float
    output: str = ""
    error: str | None = None
    parsed: Any | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvalSuiteResult:
    """The outcome of running an entire evaluation suite against a target model."""

    backend: str
    model: str
    results: list[EvalTestResult] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return (self.passed_count / self.total_count) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "total_count": self.total_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": round(self.pass_rate, 2),
            "results": [
                {
                    "name": r.test_case.name,
                    "passed": r.passed,
                    "latency_ms": round(r.latency_ms, 2),
                    "output": r.output,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def format_table(self) -> str:
        """Formats the evaluation results into a clear terminal markdown table."""
        if not self.results:
            return "No evaluation test results."

        headers = ["Status", "Test Case", "Latency (ms)", "Details"]
        rows = []
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            details = r.error if r.error else (r.output[:40] + "..." if len(r.output) > 40 else r.output)
            rows.append([status, r.test_case.name, f"{r.latency_ms:.1f}", details.replace("\n", " ")])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def format_row(values: list[str]) -> str:
            return "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values)) + " |"

        separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
        header_summary = f"### Model Evaluation: {self.backend}/{self.model} ({self.passed_count}/{self.total_count} Passed - {self.pass_rate:.1f}%)\n"
        lines = [header_summary, format_row(headers), separator]
        for row in rows:
            lines.append(format_row(row))
        return "\n".join(lines)

    def print_table(self) -> None:
        print(self.format_table())


class EvalHarness:
    """Harness for testing schema compliance, tool-calling adherence, and reasoning benchmarks."""

    def __init__(self, manager: LocalLLMManager | None = None):
        self.manager = manager or LocalLLMManager()

    def run_suite(
        self,
        backend: str | None,
        model: str,
        test_cases: Sequence[EvalTestCase],
        **common_options: Any,
    ) -> EvalSuiteResult:
        """Runs the test suite synchronously against a model."""
        suite_result = EvalSuiteResult(backend=backend or self.manager.resolve(model).backend, model=model)

        for tc in test_cases:
            opts = {**common_options, **tc.options}

            start = time.perf_counter()
            passed = True
            error: str | None = None
            output = ""
            tool_calls: list[dict[str, Any]] = []
            parsed: Any = None

            try:
                response = self.manager.chat(
                    backend=backend, model=model, messages=_messages(tc), **{**_request_options(tc), **opts}
                )
                if not isinstance(response, ChatResponse):
                    raise RuntimeError("Expected ChatResponse, got stream chunk or invalid type")
                output = response.content
                tool_calls = list(response.tool_calls)
                parsed = response.parsed
                error = _check(tc, response)
                passed = error is None
            except Exception as ex:
                passed = False
                error = str(ex)

            duration_ms = (time.perf_counter() - start) * 1000
            suite_result.results.append(
                EvalTestResult(
                    test_case=tc,
                    passed=passed,
                    latency_ms=duration_ms,
                    output=output,
                    error=error,
                    parsed=parsed,
                    tool_calls=tool_calls,
                )
            )

        return suite_result

    async def arun_suite(
        self,
        backend: str | None,
        model: str,
        test_cases: Sequence[EvalTestCase],
        **common_options: Any,
    ) -> EvalSuiteResult:
        """Runs the test suite asynchronously against a model."""
        suite_result = EvalSuiteResult(backend=backend or self.manager.resolve(model).backend, model=model)

        for tc in test_cases:
            opts = {**common_options, **tc.options}

            start = time.perf_counter()
            passed = True
            error: str | None = None
            output = ""
            tool_calls: list[dict[str, Any]] = []
            parsed: Any = None

            try:
                response = await self.manager.achat(
                    backend=backend, model=model, messages=_messages(tc), **{**_request_options(tc), **opts}
                )
                if not isinstance(response, ChatResponse):
                    raise RuntimeError("Expected ChatResponse, got stream chunk or invalid type")
                output = response.content
                tool_calls = list(response.tool_calls)
                parsed = response.parsed
                error = _check(tc, response)
                passed = error is None
            except Exception as ex:
                passed = False
                error = str(ex)

            duration_ms = (time.perf_counter() - start) * 1000
            suite_result.results.append(
                EvalTestResult(
                    test_case=tc,
                    passed=passed,
                    latency_ms=duration_ms,
                    output=output,
                    error=error,
                    parsed=parsed,
                    tool_calls=tool_calls,
                )
            )

        return suite_result

    @staticmethod
    def load_jsonl_suite(file_path: str | Path) -> list[EvalTestCase]:
        """Loads evaluation test cases from a JSONL file."""
        cases: list[EvalTestCase] = []
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict) or "prompt" not in data:
                        raise ValueError("each line must be a JSON object with a 'prompt'")
                    schema = data.get("expected_json_schema")
                    if schema is not None:
                        if not isinstance(schema, dict):
                            raise ValueError("expected_json_schema must be a JSON object")
                        _json_schema_validator(schema)  # fail at load time, not halfway through a run
                    cases.append(
                        EvalTestCase(
                            name=data.get("name", "Untitled Test"),
                            prompt=data["prompt"],
                            system_prompt=data.get("system_prompt"),
                            expected_pattern=data.get("expected_pattern"),
                            expected_tool_call=data.get("expected_tool_call"),
                            tools=data.get("tools"),
                            options=data.get("options", {}),
                            expected_json_schema=schema,
                            expected_json=data.get("expected_json"),
                        )
                    )
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
        return cases
