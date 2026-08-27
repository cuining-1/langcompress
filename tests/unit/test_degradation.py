"""Unit tests for :class:`langcompress.DefaultDegradationStrategy` and the
:class:`langcompress.DegradationStrategy` ABC (design §8.2, Plans B/C/D).

Calls ``degrade`` / ``adegrade`` directly with a hand-built
:class:`DegradationContext`, asserting each plan's message shape and the
``D → B → C`` fallback chain. The middleware integration
(:meth:`CompressionMiddleware._maybe_degrade` wiring) is covered by
``tests/scenarios/test_degradation_resilience.py``.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langcompress import (
    DefaultDegradationStrategy,
    DegradationContext,
    DegradationPatch,
    DegradationStrategy,
    Externalizer,
)

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class _StubExternalizer(Externalizer):
    """Records calls; returns a fixed reference."""

    def __init__(self, ref: str = "stub-ref") -> None:
        self.ref = ref
        self.externalize_calls: list[str] = []
        self.aexternalize_calls: list[str] = []

    def externalize(self, blob: str, *, key: str | None = None) -> str:
        self.externalize_calls.append(blob)
        return self.ref

    async def aexternalize(self, blob: str, *, key: str | None = None) -> str:
        self.aexternalize_calls.append(blob)
        return self.ref

    def retrieve(self, ref: str) -> str:  # pragma: no cover  # not used by degradation
        return f"<blob for {ref}>"

    async def aretrieve(self, ref: str) -> str:  # pragma: no cover
        return self.retrieve(ref)


class _RaisingExternalizer(Externalizer):
    def externalize(self, blob: str, *, key: str | None = None) -> str:
        raise RuntimeError("disk full")

    async def aexternalize(self, blob: str, *, key: str | None = None) -> str:
        raise RuntimeError("disk full")

    def retrieve(self, ref: str) -> str:  # pragma: no cover
        raise RuntimeError("disk full")

    async def aretrieve(self, ref: str) -> str:  # pragma: no cover
        raise RuntimeError("disk full")


def _sentinel() -> RemoveMessage:
    return RemoveMessage(id=REMOVE_ALL_MESSAGES)


def _ctx(
    *,
    state_msgs: list[BaseMessage] | None = None,
    messages_to_summarize: list[BaseMessage] | None = None,
    preserved_recent: list[BaseMessage] | None = None,
    externalizer: Externalizer | None = None,
    widen_recent: int = 10,
    min_keep: int = 3,
    summary: str = "bad summary",
) -> DegradationContext:
    state_msgs = list(state_msgs or [])
    messages_to_summarize = list(messages_to_summarize or [])
    preserved_recent = list(preserved_recent or [])
    summary_message = HumanMessage(content=f"Here is a summary:\n\n{summary}")
    sentinel = _sentinel()
    return DegradationContext(
        state={"messages": state_msgs},
        result={"messages": [sentinel, summary_message, *preserved_recent]},
        summary=summary,
        summary_message=summary_message,
        messages_to_summarize=messages_to_summarize,
        preserved_recent=preserved_recent,
        remove_all_sentinel=sentinel,
        externalizer=externalizer,
        summary_message_builder=lambda s: HumanMessage(content=s),
        widen_recent=widen_recent,
        min_keep=min_keep,
    )


# --------------------------------------------------------------------------- #
# Dataclass / ABC contracts
# --------------------------------------------------------------------------- #


def test_degradation_patch_defaults() -> None:
    p = DegradationPatch()
    assert p.messages is None
    assert p.plan == ""
    assert p.reason == ""
    assert p.external_ref is None


def test_degradation_strategy_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        DegradationStrategy()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# Plan B — widen the kept recent window, drop the failed summary (no I/O)
# --------------------------------------------------------------------------- #


def test_plan_b_keeps_widened_recent_tail_and_drops_summary() -> None:
    msgs = [HumanMessage(content=f"m{i}") for i in range(6)]
    ctx = _ctx(
        state_msgs=msgs,
        messages_to_summarize=msgs[:5],
        preserved_recent=msgs[5:],  # last 1
        externalizer=None,
        widen_recent=3,
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "B"
    assert patch.external_ref is None
    # Window = min(len(preserved_recent) + widen_recent, len(state)) = min(1+3, 6) = 4
    # → keeps the last 4 messages (m2..m5), drops the summary and m0..m1.
    assert patch.messages[0] is ctx.remove_all_sentinel
    kept = patch.messages[1:]
    assert [m.content for m in kept] == ["m2", "m3", "m4", "m5"]
    assert not any(getattr(m, "additional_kwargs", {}).get("__summarization__") for m in kept)


def test_plan_b_window_capped_at_state_length() -> None:
    msgs = [HumanMessage(content=f"m{i}") for i in range(3)]
    ctx = _ctx(
        state_msgs=msgs,
        messages_to_summarize=msgs[:2],
        preserved_recent=msgs[2:],
        externalizer=None,
        widen_recent=99,  # would exceed state length
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "B"
    assert [m.content for m in patch.messages[1:]] == ["m0", "m1", "m2"]


# --------------------------------------------------------------------------- #
# Plan D — externalize the head, keep a reference + recent window
# --------------------------------------------------------------------------- #


def test_plan_d_externalizes_head_and_tags_ref_message() -> None:
    ext = _StubExternalizer(ref="ref-007")
    msgs = [HumanMessage(content=f"m{i}") for i in range(3)]
    ctx = _ctx(
        state_msgs=msgs,
        messages_to_summarize=msgs[:2],
        preserved_recent=msgs[2:],
        externalizer=ext,
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "D"
    assert patch.external_ref == "ref-007"
    assert ext.externalize_calls and ext.aexternalize_calls == []
    # [sentinel, ref_msg, *preserved_recent]
    assert patch.messages[0] is ctx.remove_all_sentinel
    ref_msg = patch.messages[1]
    assert ref_msg.additional_kwargs.get("external_ref") == "ref-007"
    assert "ref-007" in ref_msg.content
    assert [m.content for m in patch.messages[2:]] == ["m2"]


def test_plan_d_chosen_over_plan_b_when_externalizer_configured() -> None:
    # With an externalizer, the chain prefers D (strictly more faithful than B).
    ext = _StubExternalizer()
    ctx = _ctx(
        state_msgs=[HumanMessage(content="m0"), HumanMessage(content="m1")],
        messages_to_summarize=[HumanMessage(content="m0")],
        preserved_recent=[HumanMessage(content="m1")],
        externalizer=ext,
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "D"


# --------------------------------------------------------------------------- #
# Chain fallback: D → B → C
# --------------------------------------------------------------------------- #


def test_plan_d_failure_falls_back_to_plan_b() -> None:
    ctx = _ctx(
        state_msgs=[HumanMessage(content="m0"), HumanMessage(content="m1")],
        messages_to_summarize=[HumanMessage(content="m0")],
        preserved_recent=[HumanMessage(content="m1")],
        externalizer=_RaisingExternalizer(),  # D raises → fall to B
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "B"


def test_no_externalizer_skips_straight_to_plan_b() -> None:
    ctx = _ctx(
        state_msgs=[HumanMessage(content="m0")],
        messages_to_summarize=[HumanMessage(content="m0")],
        preserved_recent=[],
        externalizer=None,
    )
    patch = DefaultDegradationStrategy().degrade(ctx)
    assert patch.plan == "B"


def test_plan_c_runs_when_b_also_fails() -> None:
    # Force Plan B to raise via a subclass; the chain must reach Plan C.
    class _BBreaks(DefaultDegradationStrategy):
        def _plan_b(self, ctx):  # type: ignore[override]
            raise RuntimeError("b broken")

    msgs = [HumanMessage(content=f"m{i}") for i in range(5)]
    ctx = _ctx(
        state_msgs=msgs,
        messages_to_summarize=msgs[:4],
        preserved_recent=msgs[4:],
        externalizer=None,
        min_keep=2,
    )
    patch = _BBreaks().degrade(ctx)
    assert patch.plan == "C"
    # Plan C keeps the last min_keep (2) messages.
    assert [m.content for m in patch.messages[1:]] == ["m3", "m4"]


def test_plan_c_direct_shape() -> None:
    msgs = [HumanMessage(content=f"m{i}") for i in range(5)]
    ctx = _ctx(state_msgs=msgs, messages_to_summarize=msgs[:4], preserved_recent=msgs[4:], min_keep=3)
    patch = DefaultDegradationStrategy()._plan_c(ctx)  # direct unit test of the last resort
    assert patch.plan == "C"
    assert patch.messages[0] is ctx.remove_all_sentinel
    assert [m.content for m in patch.messages[1:]] == ["m2", "m3", "m4"]


def test_degrade_never_propagates_when_everything_but_c_raises() -> None:
    class _AllBreakButC(DefaultDegradationStrategy):
        def _plan_d(self, ctx):  # type: ignore[override]
            raise RuntimeError("d broken")

        def _plan_b(self, ctx):  # type: ignore[override]
            raise RuntimeError("b broken")

    ctx = _ctx(
        state_msgs=[HumanMessage(content="m0")],
        messages_to_summarize=[HumanMessage(content="m0")],
        preserved_recent=[],
        externalizer=_StubExternalizer(),  # D path tried, raises; B raises; C saves
        min_keep=1,
    )
    patch = _AllBreakButC().degrade(ctx)
    assert patch.plan == "C"


# --------------------------------------------------------------------------- #
# Async path — adegrade uses aexternalize for Plan D
# --------------------------------------------------------------------------- #


async def test_adegrade_uses_aexternalize_for_plan_d() -> None:
    ext = _StubExternalizer(ref="async-ref")
    msgs = [HumanMessage(content=f"m{i}") for i in range(3)]
    ctx = _ctx(
        state_msgs=msgs,
        messages_to_summarize=msgs[:2],
        preserved_recent=msgs[2:],
        externalizer=ext,
    )
    patch = await DefaultDegradationStrategy().adegrade(ctx)
    assert patch.plan == "D"
    assert ext.aexternalize_calls and ext.externalize_calls == []


async def test_adegrade_plan_d_failure_falls_back_to_b() -> None:
    ctx = _ctx(
        state_msgs=[HumanMessage(content="m0"), HumanMessage(content="m1")],
        messages_to_summarize=[HumanMessage(content="m0")],
        preserved_recent=[HumanMessage(content="m1")],
        externalizer=_RaisingExternalizer(),
    )
    patch = await DefaultDegradationStrategy().adegrade(ctx)
    assert patch.plan == "B"


# --------------------------------------------------------------------------- #
# Custom strategy is honoured by the strategy contract
# --------------------------------------------------------------------------- #


class _IdentityStrategy(DegradationStrategy):
    """Returns an identity patch (messages=None → leave the result unchanged)."""

    def degrade(self, ctx: DegradationContext) -> DegradationPatch:  # type: ignore[override]
        self.last_summary = ctx.summary
        return DegradationPatch(messages=None, plan="", reason="identity")

    async def adegrade(self, ctx: DegradationContext) -> DegradationPatch:  # type: ignore[override]
        return self.degrade(ctx)


def test_custom_identity_strategy_records_context_and_returns_identity() -> None:
    spy = _IdentityStrategy()
    ctx = _ctx(summary="failed summary")
    patch = spy.degrade(ctx)
    assert patch.messages is None  # identity → caller keeps the original result
    assert spy.last_summary == "failed summary"
