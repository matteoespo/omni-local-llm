import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ChatResponse


@dataclass(frozen=True, slots=True)
class EvalTestCase:
    """A test case for evaluating LLM capabilities."""

    name: str
    prompt: str
    system_prompt: str | None = None
    expected_pattern: str | None = None
    expected_schema: type[BaseModel] | None = None
    expected_tool_call: str | None = None
    tools: Sequence[dict[str, Any]] | None = None
    options: dict[str, Any] = field(default_factory=dict)


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
        backend: str,
        model: str,
        test_cases: Sequence[EvalTestCase],
        **common_options: Any,
    ) -> EvalSuiteResult:
        """Runs the test suite synchronously against a model."""
        suite_result = EvalSuiteResult(backend=backend, model=model)

        for tc in test_cases:
            opts = {**common_options, **tc.options}
            messages: list[dict[str, Any]] = []
            if tc.system_prompt:
                messages.append({"role": "system", "content": tc.system_prompt})
            messages.append({"role": "user", "content": tc.prompt})

            start = time.perf_counter()
            passed = True
            error: str | None = None
            output = ""
            tool_calls: list[dict[str, Any]] = []
            parsed: Any = None

            try:
                response = self.manager.chat(
                    backend=backend,
                    model=model,
                    messages=messages,
                    response_model=tc.expected_schema,
                    tools=tc.tools,
                    **opts,
                )

                if not isinstance(response, ChatResponse):
                    raise RuntimeError("Expected ChatResponse, got stream chunk or invalid type")

                output = response.content
                tool_calls = list(response.tool_calls)
                parsed = response.parsed

                # Assertion 1: Regex pattern match
                if tc.expected_pattern and not re.search(tc.expected_pattern, output, flags=re.IGNORECASE):
                    passed = False
                    error = f"Output did not match pattern '{tc.expected_pattern}'"

                # Assertion 2: Schema validation
                if (
                    tc.expected_schema is not None
                    and passed
                    and (parsed is None or not isinstance(parsed, tc.expected_schema))
                ):
                    passed = False
                    error = f"Response did not match expected schema {tc.expected_schema.__name__}"

                # Assertion 3: Tool call assertion
                if tc.expected_tool_call and passed:
                    called_names = [call.get("function", {}).get("name") for call in tool_calls]
                    if tc.expected_tool_call not in called_names:
                        passed = False
                        error = f"Expected tool call '{tc.expected_tool_call}', but model called: {called_names}"

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
        backend: str,
        model: str,
        test_cases: Sequence[EvalTestCase],
        **common_options: Any,
    ) -> EvalSuiteResult:
        """Runs the test suite asynchronously against a model."""
        suite_result = EvalSuiteResult(backend=backend, model=model)

        for tc in test_cases:
            opts = {**common_options, **tc.options}
            messages: list[dict[str, Any]] = []
            if tc.system_prompt:
                messages.append({"role": "system", "content": tc.system_prompt})
            messages.append({"role": "user", "content": tc.prompt})

            start = time.perf_counter()
            passed = True
            error: str | None = None
            output = ""
            tool_calls: list[dict[str, Any]] = []
            parsed: Any = None

            try:
                response = await self.manager.achat(
                    backend=backend,
                    model=model,
                    messages=messages,
                    response_model=tc.expected_schema,
                    tools=tc.tools,
                    **opts,
                )

                if not isinstance(response, ChatResponse):
                    raise RuntimeError("Expected ChatResponse, got stream chunk or invalid type")

                output = response.content
                tool_calls = list(response.tool_calls)
                parsed = response.parsed

                if tc.expected_pattern and not re.search(tc.expected_pattern, output, flags=re.IGNORECASE):
                    passed = False
                    error = f"Output did not match pattern '{tc.expected_pattern}'"

                if (
                    tc.expected_schema is not None
                    and passed
                    and (parsed is None or not isinstance(parsed, tc.expected_schema))
                ):
                    passed = False
                    error = f"Response did not match expected schema {tc.expected_schema.__name__}"

                if tc.expected_tool_call and passed:
                    called_names = [call.get("function", {}).get("name") for call in tool_calls]
                    if tc.expected_tool_call not in called_names:
                        passed = False
                        error = f"Expected tool call '{tc.expected_tool_call}', but model called: {called_names}"

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
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                cases.append(
                    EvalTestCase(
                        name=data.get("name", "Untitled Test"),
                        prompt=data["prompt"],
                        system_prompt=data.get("system_prompt"),
                        expected_pattern=data.get("expected_pattern"),
                        expected_tool_call=data.get("expected_tool_call"),
                        tools=data.get("tools"),
                        options=data.get("options", {}),
                    )
                )
        return cases
