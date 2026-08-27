"""Zero-intrusion compression telemetry — the post_compress_hook collector.

This module is the concrete embodiment of the benchmark's consumer stance:
every compression fact the evaluator needs is captured inside a
``post_compress_hook`` closure the bench injects via ``CompressionConfig`` —
the package is never modified, extended, or subclassed for measurement.

What the hook sees per invocation (the package's own contract):

- ``state``  — the agent state dict; ``state["messages"]`` is the message
  list *before* this compression (pre-L0 originals).
- ``result`` — ``{"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), ...new]}``
  where the tail is either ``[summary_message, *preserved_recent]`` (clean
  L3), a degradation patch (first non-sentinel message stamped with
  ``additional_kwargs["degradation"] = {plan, reason, external_ref?}``), or
  the bare post-L0 list (L0-only cleanup turn, no summary involved).

Stage attribution (L0 vs L3) within one event recomputes the pure-rule
``L0Filter`` over ``state["messages"]`` — deterministic and side-effect-free,
so the split is exact: ``l0_delta = pre_tokens - post_l0_tokens`` and
``l3_delta = post_l0_tokens - post_tokens``. L0-only events carry only an
``l0_delta``.

The same recorder instance also serves the *baseline* arms (bare
SummarizationMiddleware, trim, full-context) through :meth:`record` — bench-
side middleware subclasses call it directly, so all four arms emit identically
shaped events and every metric downstream is arm-symmetric.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from benchmarks.llm import CountingChatModel, estimate_tokens_messages
from langcompress.pipeline.l0_filter import L0Filter

# Consumer-side copy of the default Hook-1 builder's prefix (config.py keeps
# the original private; a consumer must rely on public behaviour only).
_SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"

_TOKENS = estimate_tokens_messages


def message_keys(messages: list[BaseMessage]) -> set[str]:
    """Identity keys for a message list: ``id`` plus ``tool_call_id`` (ToolNode
    assigns fresh ids to tool results; the call id is the stable handle)."""
    keys: set[str] = set()
    for m in messages:
        mid = getattr(m, "id", None)
        if mid:
            keys.add(mid)
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            keys.add(tcid)
    return keys


def strip_summary_prefix(text: str) -> str:
    """Remove the default summary-message preamble the bench cannot suppress."""
    if text.startswith(_SUMMARY_PREFIX):
        return text[len(_SUMMARY_PREFIX) :]
    return text


def extract_summary_and_degradation(
    result_messages: list[BaseMessage],
) -> tuple[str | None, dict[str, Any] | None]:
    """Read summary text + degradation stamp out of a compression result.

    Detection rules (all public, observable behaviour):

    - a message with ``additional_kwargs["__summarization__"]`` is the summary
      message produced by the default Hook-1 builder (also used for Plan D's
      externalized-reference message);
    - otherwise, a message with ``additional_kwargs["lc_source"] ==
      "summarization"`` or the fixed ``"Here is a summary..."`` preamble is
      the *parent* SummarizationMiddleware's summary (the bare baseline arm
      needs this fallback to be judgeable at all);
    - a message with ``additional_kwargs["degradation"]`` carries the Plan
      B/D/C observability stamp the middleware attaches on substitution.
    """
    summary: str | None = None
    degradation: dict[str, Any] | None = None
    for m in result_messages:
        if isinstance(m, RemoveMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        ak = getattr(m, "additional_kwargs", {}) or {}
        if summary is None:
            tagged = isinstance(ak, dict) and (
                ak.get("__summarization__") or ak.get("lc_source") == "summarization"
            )
            if tagged or content.startswith(_SUMMARY_PREFIX):
                summary = strip_summary_prefix(content)
        if isinstance(ak, dict) and isinstance(ak.get("degradation"), dict):
            degradation = dict(ak["degradation"])
    return summary, degradation


@dataclass
class CompressEvent:
    """One compression occurrence observed through the hook."""

    seq: int
    turn: int
    kind: str  # "l3" | "l0_only" | "degraded"
    pre_tokens: int
    post_tokens: int
    pre_l0_tokens: int | None  # post-L0, pre-L3 estimate (L3 events only)
    l0_delta: int
    l3_delta: int
    summary_text: str | None
    summary_tokens: int
    plan: str | None  # "A" is implicit (retry inside _create_summary); "B"/"D"/"C" here
    degradation_reason: str | None
    external_ref: str | None
    preserved_keys: list[str]
    summarized_keys: list[str]
    summary_calls: int  # summary-LLM calls attributable to this event
    summary_seconds: float
    ts: float

    @property
    def reduction_ratio(self) -> float:
        """1 - post/pre over this event (negative if the result grew)."""
        return 1 - self.post_tokens / self.pre_tokens if self.pre_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "turn": self.turn,
            "kind": self.kind,
            "pre_tokens": self.pre_tokens,
            "post_tokens": self.post_tokens,
            "pre_l0_tokens": self.pre_l0_tokens,
            "l0_delta": self.l0_delta,
            "l3_delta": self.l3_delta,
            "reduction_ratio": round(self.reduction_ratio, 4),
            "summary_tokens": self.summary_tokens,
            "plan": self.plan,
            "degradation_reason": self.degradation_reason,
            "external_ref": self.external_ref,
            "n_preserved": len(self.preserved_keys),
            "n_summarized": len(self.summarized_keys),
            "summary_calls": self.summary_calls,
            "summary_seconds": round(self.summary_seconds, 4),
            "ts": self.ts,
        }


@dataclass
class CompressionRecorder:
    """Collects compression events; doubles as the injected post_compress_hook.

    Usage (langcompress arm)::

        recorder = CompressionRecorder(counter=..., l0_enabled=True, meter=summary_wrapper)
        cfg = CompressionConfig(..., post_compress_hook=recorder.hook)

    The hook returns ``result`` untouched — measurement must never perturb the
    system under test (the middleware itself also swallows hook exceptions,
    but the bench guarantees there are none to swallow).

    ``event_sink`` (optional) receives ``(event, before_messages,
    after_messages)`` after each recorded event — the dump writer hooks in
    here to persist full-text before/after records without telemetry itself
    knowing anything about dumping.
    """

    counter: Callable[[list[BaseMessage]], int] = _TOKENS
    l0_enabled: bool = True
    meter: CountingChatModel | None = None  # summary-model wrapper, for call/latency deltas
    events: list[CompressEvent] = field(default_factory=list)
    turn: int = 0  # advanced by the replay harness before each user turn
    hook_calls: int = 0
    event_sink: Callable[[CompressEvent, list[BaseMessage], list[BaseMessage]], None] | None = None

    # internal bookkeeping for per-event meter deltas
    _last_meter_calls: int = 0
    _last_meter_seconds: float = 0.0

    def hook(self, state: dict, result: dict) -> dict:
        """The ``post_compress_hook`` closure — identity return, record-only."""
        self.record(state, result)
        return result

    # ------------------------------------------------------------------ #
    # Core recording
    # ------------------------------------------------------------------ #

    def _meter_delta(self) -> tuple[int, float]:
        if self.meter is None:
            return 0, 0.0
        snap = self.meter.snapshot()
        calls = snap.calls - self._last_meter_calls
        seconds = snap.total_seconds - self._last_meter_seconds
        self._last_meter_calls = snap.calls
        self._last_meter_seconds = snap.total_seconds
        return max(calls, 0), max(seconds, 0.0)

    def record(self, state: dict, result: dict) -> None:
        """Shared entry: the hook path and the baseline-arm path converge here."""
        self.hook_calls += 1
        state_msgs = list(state.get("messages", []))
        result_msgs = [m for m in result.get("messages", []) if not (
            isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES
        )]
        summary, degradation = extract_summary_and_degradation(result_msgs)

        pre_tokens = self.counter(state_msgs)
        post_tokens = self.counter(result_msgs)
        preserved_keys = sorted(message_keys(result_msgs))
        state_keys = message_keys(state_msgs)
        summarized_keys = sorted(state_keys - set(preserved_keys))

        summary_calls, summary_seconds = self._meter_delta()

        if summary is None and degradation is None:
            # L0-only cleanup turn (no summarization fired).
            event = CompressEvent(
                seq=len(self.events),
                turn=self.turn,
                kind="l0_only",
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                pre_l0_tokens=post_tokens,
                l0_delta=max(pre_tokens - post_tokens, 0),
                l3_delta=0,
                summary_text=None,
                summary_tokens=0,
                plan=None,
                degradation_reason=None,
                external_ref=None,
                preserved_keys=preserved_keys,
                summarized_keys=[],
                summary_calls=0,
                summary_seconds=0.0,
                ts=time.time(),
            )
            self.events.append(event)
            self._emit(event, state_msgs, result_msgs)
            return

        # L3 event (clean or degraded). Attribute L0 vs L3 by recomputing the
        # pure-rule filter over the pre-state — exact, side-effect-free.
        if self.l0_enabled:
            post_l0 = L0Filter().run(state_msgs)
            pre_l0_tokens = self.counter(post_l0)
        else:
            pre_l0_tokens = pre_tokens

        plan = (degradation or {}).get("plan")
        event = CompressEvent(
            seq=len(self.events),
            turn=self.turn,
            kind="degraded" if degradation else "l3",
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            pre_l0_tokens=pre_l0_tokens,
            l0_delta=max(pre_tokens - pre_l0_tokens, 0),
            l3_delta=max(pre_l0_tokens - post_tokens, 0),
            summary_text=summary,
            summary_tokens=_TOKENS(summary) if summary else 0,
            plan=plan,
            degradation_reason=(degradation or {}).get("reason"),
            external_ref=(degradation or {}).get("external_ref"),
            preserved_keys=preserved_keys,
            summarized_keys=summarized_keys,
            summary_calls=summary_calls,
            summary_seconds=summary_seconds,
            ts=time.time(),
        )
        self.events.append(event)
        self._emit(event, state_msgs, result_msgs)

    def _emit(
        self,
        event: CompressEvent,
        state_msgs: list[BaseMessage],
        result_msgs: list[BaseMessage],
    ) -> None:
        if self.event_sink is not None:
            self.event_sink(event, state_msgs, result_msgs)

    # ------------------------------------------------------------------ #
    # Derived views for evaluation
    # ------------------------------------------------------------------ #

    @property
    def l3_events(self) -> list[CompressEvent]:
        return [e for e in self.events if e.kind in ("l3", "degraded")]

    @property
    def final_summary(self) -> str | None:
        """Latest surviving summary text (the only one present in final state)."""
        for event in reversed(self.events):
            if event.kind == "l3" and event.summary_text:
                return event.summary_text
            if event.kind == "degraded":
                # Degraded turns keep either a reference message (D) or no
                # summary at all (B/C) — not a scorable summary.
                return None
        return None

    def summarized_keys_all(self) -> set[str]:
        """Union of head-message keys removed by any L3 event (loss scope)."""
        keys: set[str] = set()
        for event in self.l3_events:
            keys.update(event.summarized_keys)
        return keys


__all__ = [
    "CompressEvent",
    "CompressionRecorder",
    "extract_summary_and_degradation",
    "message_keys",
    "strip_summary_prefix",
]
