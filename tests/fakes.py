from collections.abc import AsyncIterator, Iterator, Sequence

from omnillm.core.base import LLMBackend
from omnillm.core.types import ChatChunk, ChatRequest, ChatResponse, ModelSource


class RecordingBackend(LLMBackend):
    def __init__(
        self,
        *,
        response: ChatResponse | None = None,
        chunks: Sequence[ChatChunk] = (),
        models: Sequence[str] = (),
        error: Exception | None = None,
    ):
        self.response = response or ChatResponse()
        self.chunks = tuple(chunks)
        self.models = list(models)
        self.error = error
        self.requests: list[ChatRequest] = []
        self.pulled: list[tuple[str, ModelSource]] = []

    def pull_model(self, model_name: str, source: ModelSource) -> None:
        self.pulled.append((model_name, source))

    def chat(self, request: ChatRequest) -> ChatResponse | Iterator[ChatChunk]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if request.stream:
            return iter(self.chunks)
        return self.response

    async def achat(self, request: ChatRequest) -> ChatResponse | AsyncIterator[ChatChunk]:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if not request.stream:
            return self.response

        async def stream() -> AsyncIterator[ChatChunk]:
            for chunk in self.chunks:
                yield chunk

        return stream()

    def list_models(self) -> list[str]:
        return self.models
