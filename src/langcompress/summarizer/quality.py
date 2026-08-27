"""Summary quality validation (design §8.2).

A pluggable abstraction (mirroring :class:`langcompress.Externalizer` /
:class:`langcompress.Summarizer`) that judges whether a generated summary
string is safe to inject into the conversation context. When validation fails,
:class:`langcompress.CompressionMiddleware` runs the Plan A-D degradation chain
(see :mod:`langcompress.degradation`).

The default :class:`HeuristicQualityValidator` is conservative on purpose: with
its default thresholds it only flags clear failures (empty, the
``"Error generating summary:"`` prefix produced on LLM exception, below-min
length), so adopting v0.3 never regresses v0.2's passing scenarios — the
quality gate is a no-op for well-formed stub summaries. Opt-in knobs
(``min_reduction_ratio``, ``require_segments``) let stricter hosts turn the
screws without forking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

__all__ = ["HeuristicQualityValidator", "QualityValidator", "ValidationResult"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Quality validation outcome for a summary string.

    Attributes:
        passed: Whether the summary is acceptable.
        reason: Empty string when ``passed``; otherwise a human-readable cause.
        suggested_plan: Degradation plan the validator suggests on failure —
            one of ``"A"`` (retry summarization), ``"B"`` (widen the kept
            recent window), ``"D"`` (externalize the head), ``"C"`` (truncate),
            or ``None`` (no suggestion). The :class:`DegradationStrategy` is
            free to honour or ignore this hint.
    """

    passed: bool
    reason: str = ""
    suggested_plan: str | None = None


class QualityValidator(ABC):
    """Validate a generated summary against the messages it summarized."""

    @abstractmethod
    def validate(
        self,
        summary: str,
        summarized_messages: Sequence[BaseMessage],
    ) -> ValidationResult: ...

    async def avalidate(
        self,
        summary: str,
        summarized_messages: Sequence[BaseMessage],
    ) -> ValidationResult:
        """Default async delegates to the sync implementation (consistent with
        :class:`Externalizer` / :class:`Summarizer`). Subclasses that need real
        async (e.g. an LLM-as-judge validator) override this."""
        return self.validate(summary, summarized_messages)


class HeuristicQualityValidator(QualityValidator):
    """Conservative heuristic validator — flags only clear failures.

    Checks run short-circuit, in order:

    1. empty / whitespace-only → ``suggested_plan="C"`` (nothing useful was
       produced; just truncate).
    2. ``"Error generating summary:"`` prefix → ``"A"`` (LLM raised; a retry may
       succeed). This is the v0.2 latent-bug fix point: the middleware used to
       ship this error string as the summary.
    3. ``len(summary) < min_length`` → ``"A"`` (retry with a simpler prompt).
    4. reduction ratio below ``min_reduction_ratio`` (only when > 0) → ``"B"``
       (summary did not compress; keep more recent verbatim).
    5. missing eight-segment markers (only when ``require_segments=True``) →
       ``"A"``.

    With all defaults (``min_length=5``, ``min_reduction_ratio=0.0``,
    ``require_segments=False``) only checks 1-3 run, and every v0.2 stub
    summary ("SUMMARY", "STUB SUMMARY", "MY SUMMARY", "COMPRESSED SUMMARY")
    passes — so the quality gate is a no-op by default.
    """

    _ERROR_PREFIX = "Error generating summary:"

    def __init__(
        self,
        *,
        min_length: int = 5,
        min_reduction_ratio: float = 0.0,
        require_segments: bool = False,
        segment_markers: tuple[str, ...] = (
            "## 1. Primary Request and Intent",
            "## 8. Entity State",
        ),
        token_counter: Callable[[Iterable[BaseMessage]], int] | None = None,
    ) -> None:
        self.min_length = min_length
        self.min_reduction_ratio = min_reduction_ratio
        self.require_segments = require_segments
        self.segment_markers = segment_markers
        self._token_counter: Callable[[Iterable[BaseMessage]], int] = (
            token_counter or count_tokens_approximately
        )

    def validate(
        self,
        summary: str,
        summarized_messages: Sequence[BaseMessage],
    ) -> ValidationResult:
        # 1. empty
        if not summary or not summary.strip():
            return ValidationResult(False, "empty summary", "C")
        # 2. LLM error string (the v0.2 latent-bug fix point)
        if summary.startswith(self._ERROR_PREFIX):
            return ValidationResult(False, "llm error string", "A")
        # 3. too short
        if len(summary) < self.min_length:
            return ValidationResult(False, "summary too short", "A")
        # 4. reduction ratio (opt-in)
        if self.min_reduction_ratio > 0:
            src_tokens = self._token_counter(summarized_messages)
            ratio = 1 - len(summary) / max(src_tokens, 1)
            if ratio < self.min_reduction_ratio:
                return ValidationResult(False, "insufficient reduction", "B")
        # 5. eight-segment coverage (opt-in). Combined into one `if` (the
        # `and` short-circuits so `segment_markers` is only scanned when
        # require_segments is set — same semantics as the nested form).
        if self.require_segments and not all(
            marker in summary for marker in self.segment_markers
        ):
            return ValidationResult(False, "missing segments", "A")
        return ValidationResult(True)

    async def avalidate(
        self,
        summary: str,
        summarized_messages: Sequence[BaseMessage],
    ) -> ValidationResult:
        # Heuristics are pure CPU; no I/O. Delegate to sync.
        return self.validate(summary, summarized_messages)

    # Allow ``HeuristicQualityValidator(**kw)`` to be constructed with the
    # middleware's resolved token_counter (already a partial / callable).
    def __call__(self, *args: Any, **kwargs: Any) -> ValidationResult:  # pragma: no cover
        return self.validate(*args, **kwargs)
