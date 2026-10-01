from collections.abc import AsyncIterator, Iterator, Sequence

from omnillm.core.base import LLMBackend
from omnillm.core.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelSource,
)


class RecordingBackend(LLMBackend):
    def __init__(
        self,
        *,
        response: ChatResponse | None = None,
        chunks: Sequence[ChatChunk] = (),
        embedding_response: EmbeddingResponse | None = None,
        models: Sequence[str] = (),
        error: Exception | None = None,
    ):
        self.response = response or ChatResponse()
        self.chunks = tuple(chunks)
        self.embedding_response = embedding_response or EmbeddingResponse(
            data=(EmbeddingData(index=0, embedding=(0.1, 0.2, 0.3)),)
        )
        self.models = list(models)
        self.error = error
        self.requests: list[ChatRequest] = []
        self.embedding_requests: list[EmbeddingRequest] = []
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

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_requests.append(request)
        if self.error is not None:
            raise self.error
        return self.embedding_response

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_requests.append(request)
        if self.error is not None:
            raise self.error
        return self.embedding_response

    def list_models(self) -> list[str]:
        return self.models
