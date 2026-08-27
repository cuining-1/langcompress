"""Graceful degradation for failed summaries (design §8.2, Plans B/C/D).

A pluggable abstraction (mirroring :class:`langcompress.Externalizer`) that
produces a safe replacement for the compression *result* when the summary fails
quality validation. Plan A (retry summarization with a fallback prompt) lives
in :class:`langcompress.CompressionMiddleware._create_summary`; this module
handles the result-level Plans B/C/D that the middleware applies via
:meth:`CompressionMiddleware._maybe_degrade`.

Execution chain in :class:`DefaultDegradationStrategy`: ``D → B → C``.

- **Plan D** — externalize the would-be-summarized head to an
  :class:`Externalizer`, keep a lightweight reference + recent window (needs an
  externalizer; preserves the head retrievably). Strictly more faithful than B,
  so it is tried first whenever an externalizer is configured.
- **Plan B** — drop the failed summary, widen the kept recent window (no I/O,
  preserves more context than C). The no-I/O fallback when there is no
  externalizer or Plan D's externalizer call raised.
- **Plan C** — conservative truncation to ``min_keep`` recent messages, no
  summary, no I/O (the absolute last resort; never fails).

Plan D runs only when ``ctx.externalizer`` is configured; otherwise the chain
starts at Plan B. Every step is wrapped so degradation never propagates an
exception — a broken degradation must not break the agent, so Plan C (no I/O,
no externalizer) is always reachable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import get_buffer_string

from langcompress.externalizer import Externalizer

__all__ = [
    "DefaultDegradationStrategy",
    "DegradationContext",
    "DegradationPatch",
    "DegradationStrategy",
]


@dataclass(frozen=True, slots=True)
class DegradationPatch:
    """Output of a degradation decision: a replacement ``messages`` list.

    Attributes:
        messages: New ``messages`` to substitute into ``result["messages"]``;
            ``None`` means "leave the result unchanged".
        plan: Which plan produced this patch — ``"B"`` / ``"D"`` / ``"C"`` /
            ``""`` (identity).
        reason: Human-readable note for observability.
        external_ref: The reference returned by the externalizer for Plan D
            (host projects aggregate these via
            :func:`langcompress.aggregate_external_refs`).
    """

    messages: list[BaseMessage] | None = None
    plan: str = ""
    reason: str = ""
    external_ref: str | None = None


@dataclass(slots=True)
class DegradationContext:
    """Inputs to :meth:`DegradationStrategy.degrade`.

    Pure data + callbacks; deliberately carries no reference to
    :class:`CompressionMiddleware` so a strategy is independently testable and
    usable outside the LangChain-bound adapter.
    """

    state: dict[str, Any]
    result: dict[str, Any]
    summary: str
    summary_message: BaseMessage
    messages_to_summarize: list[BaseMessage]
    preserved_recent: list[BaseMessage]
    remove_all_sentinel: BaseMessage
    externalizer: Externalizer | None
    summary_message_builder: Callable[[str], BaseMessage]
    widen_recent: int
    min_keep: int


class DegradationStrategy(ABC):
    """Result-level degradation strategy for failed summaries (Plans B/C/D)."""

    @abstractmethod
    def degrade(self, ctx: DegradationContext) -> DegradationPatch: ...

    async def adegrade(self, ctx: DegradationContext) -> DegradationPatch:
        """Default async delegates to the sync implementation.
        :class:`DefaultDegradationStrategy` overrides this to use
        :meth:`Externalizer.aexternalize` for Plan D."""
        return self.degrade(ctx)


class DefaultDegradationStrategy(DegradationStrategy):
    """Strictly-safe default chain: ``D → B → C``.

    Plan D is tried first whenever an externalizer is configured (it preserves
    the head retrievably — strictly more faithful than B). Plan B is the no-I/O
    fallback (no externalizer, or D's externalizer call raised); Plan C is the
    never-fails last resort. Never propagates an exception — a broken
    degradation must not break the agent.
    """

    def degrade(self, ctx: DegradationContext) -> DegradationPatch:
        if ctx.externalizer is not None:
            try:
                return self._plan_d(ctx)
            except Exception:  # noqa: BLE001, S110  # D's I/O failed → fall to B
                pass
        try:
            return self._plan_b(ctx)
        except Exception:  # noqa: BLE001, S110  # B must not break the agent
            pass
        return self._plan_c(ctx)

    async def adegrade(self, ctx: DegradationContext) -> DegradationPatch:
        if ctx.externalizer is not None:
            try:
                return await self._aplan_d(ctx)
            except Exception:  # noqa: BLE001, S110
                pass
        try:
            return self._plan_b(ctx)  # no I/O; sync is fine
        except Exception:  # noqa: BLE001, S110
            pass
        return self._plan_c(ctx)

    # -- Plan B: widen the kept recent window, drop the failed summary -------- #
    def _plan_b(self, ctx: DegradationContext) -> DegradationPatch:
        state_msgs = ctx.state.get("messages", [])
        window = min(len(ctx.preserved_recent) + max(ctx.widen_recent, 0), len(state_msgs))
        recent_tail = list(state_msgs[-window:]) if window > 0 else []
        return DegradationPatch(
            messages=[ctx.remove_all_sentinel, *recent_tail],
            plan="B",
            reason="dropped failed summary, widened recent window",
        )

    # -- Plan D: externalize the head, keep a reference + recent --------------- #
    def _plan_d(self, ctx: DegradationContext) -> DegradationPatch:
        if not ctx.messages_to_summarize:
            raise RuntimeError("nothing to externalize")  # → fall through to C
        blob = get_buffer_string(ctx.messages_to_summarize)
        ref = ctx.externalizer.externalize(blob)  # type: ignore[union-attr]
        return self._build_d_patch(ctx, ref)

    async def _aplan_d(self, ctx: DegradationContext) -> DegradationPatch:
        if not ctx.messages_to_summarize:
            raise RuntimeError("nothing to externalize")
        blob = get_buffer_string(ctx.messages_to_summarize)
        ref = await ctx.externalizer.aexternalize(blob)  # type: ignore[union-attr]
        return self._build_d_patch(ctx, ref)

    def _build_d_patch(self, ctx: DegradationContext, ref: str) -> DegradationPatch:
        ref_text = (
            f"[Conversation history externalized ({len(ctx.messages_to_summarize)} "
            f"messages). Ref: {ref}. Reload via externalizer.retrieve({ref!r}).]"
        )
        ref_msg = ctx.summary_message_builder(ref_text)
        try:
            ref_msg = ref_msg.model_copy(
                update={
                    "additional_kwargs": {
                        **ref_msg.additional_kwargs,
                        "external_ref": ref,
                    }
                }
            )
        except Exception:  # noqa: BLE001, S110  # non-pydantic message → keep ref_msg as-is
            pass
        return DegradationPatch(
            messages=[ctx.remove_all_sentinel, ref_msg, *ctx.preserved_recent],
            plan="D",
            reason="externalized head to externalizer",
            external_ref=ref,
        )

    # -- Plan C: conservative truncation to min_keep recent ------------------- #
    def _plan_c(self, ctx: DegradationContext) -> DegradationPatch:
        state_msgs = ctx.state.get("messages", [])
        keep = max(ctx.min_keep, 0)
        tail = list(state_msgs[-keep:]) if keep > 0 else []
        return DegradationPatch(
            messages=[ctx.remove_all_sentinel, *tail],
            plan="C",
            reason="conservative truncation to minimum recent window",
        )
