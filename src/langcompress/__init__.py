"""langcompress — pluggable, layered token-compression middleware for LangGraph/LangChain agents.

The package is split into a small core (depending only on ``langchain-core`` +
``pydantic``) and the LangChain-bound adapters (:mod:`langcompress.middleware`
for central L3 compression, :mod:`langcompress.toolcall_middleware` for
source-side L2 externalization). Core symbols are imported eagerly so a plain
``import langcompress`` works without the ``[middleware]`` extra; the
LangChain-bound symbols (``CompressionMiddleware`` / ``CompressionAgentState``
/ ``ToolCallExternalizerMiddleware`` / ``aggregate_external_refs``) are imported
lazily via ``__getattr__`` and surface a friendly error when the ``langchain``
dependency is missing.

Import order respects intra-package dependencies:
``externalizer`` ← ``retention`` / ``degradation`` ← ``summarizer`` ←
``config``, so each module's top-level imports resolve cleanly during package
init.
"""
from __future__ import annotations

from langcompress.config import CompressionConfig
from langcompress.degradation import (
    DefaultDegradationStrategy,
    DegradationContext,
    DegradationPatch,
    DegradationStrategy,
)

# --- Core symbols: eager import (langchain-core + pydantic only) ----------- #
from langcompress.externalizer import (
    Externalizer,
    ExternalRefRecord,
    FilesystemExternalizer,
    PurgeReport,
)
from langcompress.pipeline import CompressionStage, L0Filter
from langcompress.retention import (
    NullPolicy,
    RetentionManager,
    RetentionPolicy,
    TTLPolicy,
    collect_live_refs,
)
from langcompress.state import CompressionState
from langcompress.summarizer import (
    DEFAULT_SUMMARY_PROMPT,
    EIGHT_SEGMENT_TEMPLATE,
    FALLBACK_SUMMARY_PROMPT,
    HeuristicQualityValidator,
    LLMSummarizer,
    QualityValidator,
    Summarizer,
    ValidationResult,
)
from langcompress.token_counter import (
    ApproximateTokenCounter,
    TiktokenCounter,
    TokenCounter,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SUMMARY_PROMPT",
    "EIGHT_SEGMENT_TEMPLATE",
    "FALLBACK_SUMMARY_PROMPT",
    "ApproximateTokenCounter",
    "CompressionAgentState",
    "CompressionConfig",
    "CompressionMiddleware",
    "CompressionStage",
    "CompressionState",
    "DefaultDegradationStrategy",
    "DegradationContext",
    "DegradationPatch",
    "DegradationStrategy",
    "ExternalRefRecord",
    "Externalizer",
    "FilesystemExternalizer",
    "HeuristicQualityValidator",
    "L0Filter",
    "LLMSummarizer",
    "NullPolicy",
    "PurgeReport",
    "QualityValidator",
    "RetentionManager",
    "RetentionPolicy",
    "Summarizer",
    "TTLPolicy",
    "TiktokenCounter",
    "TokenCounter",
    "ToolCallExternalizerMiddleware",
    "ValidationResult",
    "__version__",
    "aggregate_external_refs",
    "collect_live_refs",
]


def __getattr__(name: str):
    """Lazily import the LangChain-bound symbols so core stays dependency-light.

    ``CompressionMiddleware`` / ``CompressionAgentState`` live in
    :mod:`langcompress.middleware`, and ``ToolCallExternalizerMiddleware`` /
    ``aggregate_external_refs`` live in :mod:`langcompress.toolcall_middleware`;
    both modules import ``langchain``. Requesting them without the
    ``[middleware]`` extra raises an actionable ``ImportError``.
    """
    if name in {
        "CompressionMiddleware",
        "CompressionAgentState",
        "ToolCallExternalizerMiddleware",
        "aggregate_external_refs",
    }:
        # Resolve the submodule name up front and import it once via
        # ``importlib.import_module`` so there is a single ``_m`` binding
        # (mypy flags the two-branch ``from ... import ... as _m`` form as a
        # redefinition). The ImportError surface is preserved so a missing
        # ``langchain`` still raises the actionable [middleware] extra hint.
        import importlib

        module_name = (
            "middleware"
            if name in {"CompressionMiddleware", "CompressionAgentState"}
            else "toolcall_middleware"
        )
        try:
            _m = importlib.import_module(f"langcompress.{module_name}")
        except ImportError as e:  # langchain missing
            raise ImportError(
                f"{name} requires the [middleware] extra: "
                "pip install langcompress[middleware]"
            ) from e
        return getattr(_m, name)
    raise AttributeError(f"module 'langcompress' has no attribute {name!r}")
