from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .adapters.openai_adapter import OpenAICompatibleAdapter
from .core.config import ConfigError
from .core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidImageError,
    InvalidRequestError,
    ModelNotDownloadedError,
    ModelNotFoundError,
    OmniLLMError,
    ToolLoopLimitError,
    UnsupportedFeatureError,
)
from .core.hardware import (
    FitRecommendation,
    HardwareProfile,
    QuantFitResult,
    QuantFitStatus,
    detect_hardware,
    estimate_model_memory,
    recommend_model_fit,
    suggest_models_for_hardware,
)
from .core.manager import LocalLLMManager
from .core.session import ChatSession
from .core.tools import ToolRegistry, function_to_tool_schema, tool
from .core.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelTarget,
    Usage,
)
from .harness import BenchmarkHarness, EvalHarness, NeedleHarness

try:
    __version__ = _package_version("omni-local-llm")
except PackageNotFoundError:  # running from a source tree without installed metadata
    __version__ = "0.0.0+unknown"
__all__ = [
    "BackendNotFoundError",
    "BackendUnavailableError",
    "BenchmarkHarness",
    "ChatChunk",
    "ChatRequest",
    "ChatResponse",
    "ChatSession",
    "ConfigError",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EvalHarness",
    "FitRecommendation",
    "HardwareProfile",
    "InvalidImageError",
    "InvalidRequestError",
    "LocalLLMManager",
    "ModelNotDownloadedError",
    "ModelNotFoundError",
    "ModelTarget",
    "NeedleHarness",
    "OmniLLMError",
    "OpenAICompatibleAdapter",
    "QuantFitResult",
    "QuantFitStatus",
    "ToolLoopLimitError",
    "ToolRegistry",
    "UnsupportedFeatureError",
    "Usage",
    "detect_hardware",
    "estimate_model_memory",
    "function_to_tool_schema",
    "recommend_model_fit",
    "suggest_models_for_hardware",
    "tool",
]
