class OmniLLMError(Exception):
    """Base class for errors that callers can handle without backend knowledge."""


class BackendNotFoundError(OmniLLMError):
    """Raised when a requested backend has not been registered."""


class BackendUnavailableError(OmniLLMError):
    """Raised when an optional backend dependency or service is unavailable."""


class UnsupportedFeatureError(OmniLLMError):
    """Raised when a backend cannot satisfy a requested capability."""


class InvalidRequestError(OmniLLMError):
    """Raised when a request cannot be processed as supplied."""
