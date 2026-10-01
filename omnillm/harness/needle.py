import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ChatResponse

DEFAULT_HAYSTACK_TEXT = (
    "The atmospheric composition of Neptune consists primarily of hydrogen and helium, with trace amounts of methane. "
    "Methane in the upper atmosphere absorbs red light, giving Neptune its vivid azure hue. "
    "Planetary scientists believe that deep within the mantle, extreme pressure breaks methane molecules into diamond crystals. "
    "These diamonds sink like hail toward the planet core in a phenomenon commonly known as diamond rain. "
    "Observations from the Voyager 2 spacecraft confirmed violent supersonic winds reaching speeds over two thousand kilometers per hour. "
)


@dataclass(slots=True)
class NeedleTrial:
    """One needle retrieval trial at a specific depth in the haystack."""

    depth_percent: int
    context_words: int
    retrieved: bool
    needle_secret: str
    output: str
    latency_ms: float


@dataclass(slots=True)
class NeedleResult:
    """Result of running a Needle in a Haystack evaluation across context depths."""

    backend: str
    model: str
    trials: list[NeedleTrial] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.trials:
            return 0.0
        return (sum(1 for t in self.trials if t.retrieved) / len(self.trials)) * 100.0

    def format_table(self) -> str:
        """Formats the needle retrieval results into a table."""
        if not self.trials:
            return "No needle trials executed."

        headers = ["Depth (%)", "Words", "Result", "Latency (ms)", "Extracted Secret"]
        rows = []
        for t in self.trials:
            status = "🎯 FOUND" if t.retrieved else "❌ MISSED"
            excerpt = t.output[:35] + "..." if len(t.output) > 35 else t.output
            rows.append(
                [f"{t.depth_percent}%", str(t.context_words), status, f"{t.latency_ms:.1f}", excerpt.replace("\n", " ")]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def format_row(values: list[str]) -> str:
            return "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values)) + " |"

        separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
        title = f"### Needle-In-A-Haystack: {self.backend}/{self.model} (Score: {self.score:.1f}%)\n"
        lines = [title, format_row(headers), separator]
        for row in rows:
            lines.append(format_row(row))
        return "\n".join(lines)

    def print_table(self) -> None:
        print(self.format_table())


class NeedleHarness:
    """Harness to evaluate context retrieval across varying depths and context lengths."""

    def __init__(self, manager: LocalLLMManager | None = None):
        self.manager = manager or LocalLLMManager()

    def run(
        self,
        backend: str,
        model: str,
        *,
        secret_code: str = "BLUE-TITAN-42",
        depths: Sequence[int] = (0, 25, 50, 75, 100),
        target_word_count: int = 600,
        **options: Any,
    ) -> NeedleResult:
        """Executes needle retrieval across the specified context depths."""
        result = NeedleResult(backend=backend, model=model)
        needle_sentence = f" IMPORTANT NOTE: The secret access code is {secret_code}. Remember this code exactly. "

        # Build haystack up to target_word_count
        words_per_block = len(DEFAULT_HAYSTACK_TEXT.split())
        repeats = max(1, target_word_count // words_per_block)
        base_blocks = [DEFAULT_HAYSTACK_TEXT] * repeats

        for depth in depths:
            # Insert needle at appropriate depth
            split_idx = int((depth / 100.0) * len(base_blocks))
            haystack = "".join(base_blocks[:split_idx]) + needle_sentence + "".join(base_blocks[split_idx:])
            total_words = len(haystack.split())

            prompt = (
                f"{haystack}\n\n"
                f"Question: What is the secret access code mentioned in the text? "
                f"State only the exact secret code."
            )

            start = time.perf_counter()
            retrieved = False
            output = ""

            try:
                response = self.manager.chat(
                    backend=backend,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32,
                    temperature=0.0,
                    **options,
                )
                if isinstance(response, ChatResponse):
                    output = response.content
                    if secret_code.lower() in output.lower():
                        retrieved = True
            except Exception as ex:
                output = f"Error: {ex}"

            latency_ms = (time.perf_counter() - start) * 1000
            result.trials.append(
                NeedleTrial(
                    depth_percent=depth,
                    context_words=total_words,
                    retrieved=retrieved,
                    needle_secret=secret_code,
                    output=output.strip(),
                    latency_ms=latency_ms,
                )
            )

        return result
