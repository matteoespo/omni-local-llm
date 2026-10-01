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
    Usage,
)
from .harness import BenchmarkHarness, EvalHarness, NeedleHarness

__version__ = "0.3.0"
__all__ = [
    "BenchmarkHarness",
    "ChatChunk",
    "ChatRequest",
    "ChatResponse",
    "ChatSession",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EvalHarness",
    "FitRecommendation",
    "HardwareProfile",
    "LocalLLMManager",
    "NeedleHarness",
    "QuantFitResult",
    "QuantFitStatus",
    "ToolRegistry",
    "Usage",
    "detect_hardware",
    "estimate_model_memory",
    "function_to_tool_schema",
    "recommend_model_fit",
    "suggest_models_for_hardware",
    "tool",
]
