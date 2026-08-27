"""CompressionConfig — bridges design knobs to SummarizationMiddleware kwargs.

Holds the four extension-point hooks plus the optional content classifier, and
adapts the design's ``token_threshold`` / ``keep_recent`` / ``summary_template``
to the parent ``SummarizationMiddleware`` ``trigger`` / ``keep`` /
``summary_prompt`` tuple API.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langcompress.degradation import DegradationStrategy
from langcompress.externalizer import Externalizer
from langcompress.summarizer.quality import QualityValidator

# Context size tuples accepted by SummarizationMiddleware (used for type hints).
# ("fraction", 0.8) | ("tokens", 3000) | ("messages", 50)
ContextSize = tuple  # heterogeneous 2-tuple; validated by the parent at runtime

_SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"


# --------------------------------------------------------------------------- #
# Default hook implementations (mirror parent SummarizationMiddleware behaviour)
# --------------------------------------------------------------------------- #


def _default_build_summary_message(summary: str) -> BaseMessage:
    """Hook 1 default.

    Same ``HumanMessage`` the parent ``_build_new_messages`` produces, plus an
    ``__summarization__`` flag so host frontends can identify summary messages.
    """
    return HumanMessage(
        content=f"{_SUMMARY_PREFIX}{summary}",
        additional_kwargs={"lc_source": "summarization", "__summarization__": True},
    )


def _default_get_summary_llm_config() -> RunnableConfig:
    """Hook 2 default: empty config — merged with the parent's ``lc_source`` metadata."""
    return {}


def _default_post_compress(state: dict, result: dict) -> dict:
    """Hook 3 default: identity."""
    return result


def _default_should_summarize(messages: list, total_tokens: int, base_decision: bool) -> bool:
    """Hook 4 default: honour the parent's multi-signal decision."""
    return base_decision


def _default_content_classifier(message: BaseMessage) -> str:
    """Hook 5 default: classify by message type + size only (no business rules).

    Business scenarios (e.g. "feishu doc") are injected by the host via this hook.
    """
    if isinstance(message, SystemMessage):
        label = "system"
    elif isinstance(message, HumanMessage):
        label = "user"
    elif isinstance(message, AIMessage):
        label = "assistant"
    elif isinstance(message, ToolMessage):
        label = "tool_result"
    else:
        label = "other"
    if len(str(getattr(message, "content", ""))) > 8192:
        label = f"{label}:large"
    return label


class CompressionConfig(BaseModel):
    """Configuration for :class:`langcompress.CompressionMiddleware`.

    Bridges the design's ``token_threshold`` / ``keep_recent`` / ``summary_template``
    knobs to the parent ``SummarizationMiddleware`` ``trigger`` / ``keep`` /
    ``summary_prompt`` tuple API, and carries the four extension-point hooks plus
    the optional ``content_classifier``.

    With all hooks left at their defaults, behaviour is identical to the parent
    middleware (plus an ``__summarization__`` flag on summary messages).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # --- knobs bridged to the parent SummarizationMiddleware ---
    # Required — no default, no env fallback, and never an implicit reuse of
    # the agent's main model: the host passes a dedicated summary model
    # (selection principle: capability-floored / cost-floored — summarization
    # is a pure-overhead path, input ≈ trim_tokens_to_summarize tokens with
    # 0-2 calls per compression, and the eight-segment template's quality bar
    # is clearable by mini/flash/haiku-class models). str | BaseChatModel;
    # a str identifier is resolved via init_chat_model by the parent.
    summary_model: Any
    token_threshold: float | int | list | tuple | None = None
    keep_recent: int = 20
    keep: ContextSize | None = None  # advanced: explicit ContextSize, overrides keep_recent
    token_counter: Callable[[Iterable[BaseMessage]], int] | None = None
    summary_template: str | None = None  # None → EIGHT_SEGMENT_TEMPLATE (set by the middleware)
    trim_tokens_to_summarize: int | None = 4000

    # --- four extension-point hooks + optional content classifier ---
    summary_message_builder: Callable[[str], BaseMessage] = Field(
        default=_default_build_summary_message
    )
    summary_llm_config_provider: Callable[[], RunnableConfig] = Field(
        default=_default_get_summary_llm_config
    )
    post_compress_hook: Callable[[dict, dict], dict] = Field(default=_default_post_compress)
    should_summarize_hook: Callable[[list, int, bool], bool] = Field(
        default=_default_should_summarize
    )
    content_classifier: Callable[[BaseMessage], str] = Field(default=_default_content_classifier)

    # --- M3: summary quality validation + graceful degradation (design §8.2) ---
    # ABC fields: None → the middleware resolves a default instance
    # (HeuristicQualityValidator / DefaultDegradationStrategy), mirroring the
    # `token_counter` / `summary_template` None-resolution pattern. ABCs are
    # code, never sourced from the environment.
    quality_validator: QualityValidator | None = None
    degradation_strategy: DegradationStrategy | None = None
    degradation_externalizer: Externalizer | None = None
    # Scalar knobs (env-readable; see ``_read_env``). Defaults keep the gate a
    # no-op for well-formed summaries → v0.2 scenarios do not regress.
    quality_min_length: int = 5
    quality_min_reduction_ratio: float = 0.0
    quality_require_segments: bool = False
    degradation_widen_recent: int = 10
    degradation_min_keep: int = 3

    # --- M6: L0 content filter (design §4.1/§12.3) ---
    # An L0Filter (CompressionStage) instance for always-on pre-compression
    # cleanup; None → the middleware creates a default L0Filter() when
    # ``l0_enabled`` is True. Set ``l0_enabled=False`` (or
    # ``LANGCOMPRESS_L0_ENABLED=false``) to disable L0 entirely.
    # Like the other ABC fields this is code, not env-sourced; only the
    # ``l0_enabled`` scalar is env-readable.
    l0_filter: Any = None  # CompressionStage | None
    l0_enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _read_env(cls, values: Any) -> Any:
        """Read ``LANGCOMPRESS_*`` env vars only for fields not explicitly supplied.

        Hooks are code, not configuration, and are never sourced from the environment.
        """
        if not isinstance(values, dict):
            return values
        env = os.environ

        def _present(key: str) -> bool:
            return key in values and values[key] is not None

        if not _present("token_threshold") and env.get("LANGCOMPRESS_TOKEN_THRESHOLD"):
            raw = env["LANGCOMPRESS_TOKEN_THRESHOLD"]
            try:
                values["token_threshold"] = int(raw)
            except ValueError:
                values["token_threshold"] = float(raw)
        if not _present("keep_recent") and env.get("LANGCOMPRESS_KEEP_RECENT"):
            values["keep_recent"] = int(env["LANGCOMPRESS_KEEP_RECENT"])
        if not _present("trim_tokens_to_summarize") and env.get(
            "LANGCOMPRESS_TRIM_TOKENS_TO_SUMMARIZE"
        ):
            values["trim_tokens_to_summarize"] = int(
                env["LANGCOMPRESS_TRIM_TOKENS_TO_SUMMARIZE"]
            )
        # M3 quality / degradation scalar knobs (ABC instances are never env-sourced).
        if not _present("quality_min_length") and env.get("LANGCOMPRESS_QUALITY_MIN_LENGTH"):
            values["quality_min_length"] = int(env["LANGCOMPRESS_QUALITY_MIN_LENGTH"])
        if not _present("quality_min_reduction_ratio") and env.get(
            "LANGCOMPRESS_QUALITY_MIN_REDUCTION_RATIO"
        ):
            values["quality_min_reduction_ratio"] = float(
                env["LANGCOMPRESS_QUALITY_MIN_REDUCTION_RATIO"]
            )
        if not _present("quality_require_segments") and env.get(
            "LANGCOMPRESS_QUALITY_REQUIRE_SEGMENTS"
        ):
            values["quality_require_segments"] = env[
                "LANGCOMPRESS_QUALITY_REQUIRE_SEGMENTS"
            ].lower() in ("1", "true", "yes")
        if not _present("degradation_widen_recent") and env.get(
            "LANGCOMPRESS_DEGRADATION_WIDEN_RECENT"
        ):
            values["degradation_widen_recent"] = int(
                env["LANGCOMPRESS_DEGRADATION_WIDEN_RECENT"]
            )
        if not _present("degradation_min_keep") and env.get(
            "LANGCOMPRESS_DEGRADATION_MIN_KEEP"
        ):
            values["degradation_min_keep"] = int(env["LANGCOMPRESS_DEGRADATION_MIN_KEEP"])
        # M6: L0 enable/disable scalar (the L0Filter instance is code, not env-sourced).
        if not _present("l0_enabled") and env.get("LANGCOMPRESS_L0_ENABLED"):
            values["l0_enabled"] = env["LANGCOMPRESS_L0_ENABLED"].lower() in (
                "1",
                "true",
                "yes",
            )
        return values

    def _as_parent_trigger(self) -> ContextSize | list | None:
        """Translate ``token_threshold`` into the parent ``trigger`` ContextSize API.

        - ``None``            → ``None`` (no trigger; parent disables summarization)
        - ``float``           → ``[("fraction", v)]``
        - ``int`` (non-bool)  → ``[("tokens", v)]``
        - ``tuple``           → a single ContextSize, wrapped: ``[v]``
          (e.g. ``("messages", 10)`` → ``[("messages", 10)]``)
        - ``list``           → a list of ContextSize (multiple OR-conditions),
          passed through verbatim (e.g. ``[("tokens", 100), ("messages", 50)]``)
        """
        v = self.token_threshold
        if v is None:
            return None
        if isinstance(v, bool):  # bool is a subclass of int — reject explicitly
            raise TypeError("token_threshold must not be a bool")
        if isinstance(v, list):
            return list(v)  # multiple OR-conditions (list of ContextSize)
        if isinstance(v, tuple):
            return [v]  # a single ContextSize
        if isinstance(v, float):
            return [("fraction", v)]
        if isinstance(v, int):
            return [("tokens", v)]
        raise TypeError(f"Unsupported token_threshold type: {type(v).__name__}")
