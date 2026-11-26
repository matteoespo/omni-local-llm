from typing import Any


class OmniLLMError(Exception):
    """Base class for errors that callers can handle without backend knowledge."""


class BackendNotFoundError(OmniLLMError):
    """Raised when a requested backend has not been registered."""


class BackendUnavailableError(OmniLLMError):
    """Raised when an optional backend dependency or service is unavailable."""


class UnsupportedFeatureError(OmniLLMError):
    """Raised when a backend cannot satisfy a requested capability.

    `feature` names the capability (e.g. "tools", "vision") so callers such as the API server can report it
    without echoing the exception text.
    """

    def __init__(self, message: str, *, feature: str = ""):
        super().__init__(message)
        self.feature = feature


class InvalidRequestError(OmniLLMError):
    """Raised when a request cannot be processed as supplied."""


class ModelNotFoundError(InvalidRequestError):
    """Raised when a backend reports that the requested model does not exist."""


class ModelNotDownloadedError(ModelNotFoundError):
    """Raised when a model isn't available locally and downloads are disabled (`local_files_only`)."""


class InvalidImageError(InvalidRequestError):
    """Raised when an image from an untrusted source is not inline base64 data."""


class ToolLoopLimitError(OmniLLMError):
    """Raised when an autonomous tool loop exhausts its step budget while the model still requests tools.

    The session history is left unchanged. `response` is the last model response (with its pending tool
    calls) and `messages` holds the assistant/tool messages produced during the aborted turn.
    """

    def __init__(self, message: str, *, response: Any, messages: list[dict[str, Any]]):
        super().__init__(message)
        self.response = response
        self.messages = messages
