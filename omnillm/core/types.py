from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

T = TypeVar("T")

type ChatMessage = dict[str, Any]
type ToolCall = dict[str, Any]
type FinishReason = Literal["stop", "tool_calls", "length"]


@dataclass(frozen=True, slots=True)
class ModelSource:
    """Location options needed to resolve a model artifact."""

    filename: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    mmproj_filename: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Runtime options used only by backends that load a model in-process."""

    n_gpu_layers: int | None = None
    n_ctx: int | None = None
    clip_model_path: str | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One backend-neutral chat request."""

    model: str
    messages: Sequence[ChatMessage]
    stream: bool = False
    json_mode: bool = False
    response_format: dict[str, Any] | None = None
    tools: Sequence[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | Sequence[str] | None = None
    model_source: ModelSource = field(default_factory=ModelSource)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)

    @property
    def is_json(self) -> bool:
        return self.json_mode or (
            self.response_format is not None and self.response_format.get("type") in {"json_object", "json_schema"}
        )

    @property
    def json_schema(self) -> dict[str, Any] | None:
        """Returns the JSON schema dict if structured output is requested."""
        if self.response_format is None:
            return None
        fmt_type = self.response_format.get("type")
        if fmt_type == "json_schema":
            js = self.response_format.get("json_schema")
            if isinstance(js, dict):
                inner_schema = js.get("schema")
                if isinstance(inner_schema, dict):
                    return inner_schema
                return js
        if "schema" in self.response_format:
            schema = self.response_format["schema"]
            if isinstance(schema, dict):
                return schema
        if "properties" in self.response_format:
            return self.response_format
        return None


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def as_openai(self) -> dict[str, int] | None:
        prompt_tokens = self.prompt_tokens
        completion_tokens = self.completion_tokens
        if prompt_tokens is None or completion_tokens is None:
            return None
        total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def as_openai_embedding(self) -> dict[str, int] | None:
        if self.prompt_tokens is None:
            return None
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.prompt_tokens,
        }


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str = ""
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    finish_reason: FinishReason = "stop"
    usage: Usage | None = None
    parsed: Any | None = None

    def parse_as(self, model: type[T]) -> T:
        """Parses the response content into a Pydantic model or callable schema."""
        model_validate = getattr(model, "model_validate_json", None)
        if callable(model_validate):
            result: T = model_validate(self.content)
            return result
        import json

        data = json.loads(self.content)
        if callable(model):
            res: T = model(data)  # type: ignore[call-arg]
            return res
        return data


@dataclass(frozen=True, slots=True)
class ChatChunk:
    content: str = ""
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    finish_reason: FinishReason | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """One backend-neutral embedding request."""

    model: str
    input: Sequence[str]
    model_source: ModelSource = field(default_factory=ModelSource)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)


@dataclass(frozen=True, slots=True)
class EmbeddingData:
    index: int
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    data: Sequence[EmbeddingData]
    model: str = ""
    usage: Usage | None = None

    @property
    def embeddings(self) -> list[list[float]]:
        return [list(item.embedding) for item in self.data]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    streaming: bool = True
    json_mode: bool = True
    tools: bool = True
    embeddings: bool = True
    structured_outputs: bool = True
    vision: bool = True


type ChatResult = ChatResponse | Iterator[ChatChunk]
type AsyncChatResult = ChatResponse | AsyncIterator[ChatChunk]
