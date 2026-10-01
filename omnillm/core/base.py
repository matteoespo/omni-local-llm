from abc import ABC, abstractmethod

from omnillm.core.errors import UnsupportedFeatureError
from omnillm.core.types import (
    AsyncChatResult,
    BackendCapabilities,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelSource,
)


class LLMBackend(ABC):
    capabilities = BackendCapabilities()

    @abstractmethod
    def pull_model(self, model_name: str, source: ModelSource) -> str | None:
        """Downloads the model if it doesn't exist locally."""
        raise NotImplementedError

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResult:
        """Sends a backend-neutral request and returns a response or stream of chunks."""
        raise NotImplementedError

    @abstractmethod
    async def achat(self, request: ChatRequest) -> AsyncChatResult:
        """Asynchronous version of chat."""
        raise NotImplementedError

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generates embeddings for the provided input texts."""
        raise UnsupportedFeatureError("Embeddings are not supported by this backend.")

    async def aembed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Asynchronously generates embeddings."""
        raise UnsupportedFeatureError("Embeddings are not supported by this backend.")

    def list_models(self) -> list[str]:
        """Returns models known to the backend without requiring callers to know its SDK."""
        return []
