"""Scenario tests for summary quality validation + graceful degradation
(design §8.2, v0.3 M3) driven through the real
:class:`langcompress.CompressionMiddleware`.

These are the "consumer scenario" tests for M3: they stand up a
``CompressionMiddleware`` whose summary model produces a *bad* summary and
assert that the Plan A-D chain keeps the agent running with a safe context:

- Plan A (retry with the fallback prompt) recovers a bad primary summary.
- Plans B / D substitute the failed-summary result when retry is exhausted /
  not applicable (B with no externalizer, D with one).
- A broken externalizer falls back from D to B.
- A custom strategy is honoured; Hook 3 (``post_compress_hook``) still runs on
  the degraded result.

Calls ``before_model`` / ``abefore_model`` directly with a ``MagicMock``
runtime (the parent never touches the runtime), mirroring the v0.2 hook tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langcompress import (
    CompressionConfig,
    CompressionMiddleware,
    DegradationPatch,
    DegradationStrategy,
    Externalizer,
)

# --------------------------------------------------------------------------- #
# Summary-model fixtures (no API key needed)
# --------------------------------------------------------------------------- #


class _PlanAFixture:
    """First invoke → an LLM-error string (Plan A); second → a good summary."""

    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        if self.invoke_count == 1:
            return AIMessage(content="Error generating summary: simulated LLM failure")
        return AIMessage(content="RECOVERED SUMMARY")

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


class _AlwaysBad:
    """Always returns an empty summary → validator flags Plan C (no retry)."""

    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        return AIMessage(content="")

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


class _AlwaysTooShort:
    """Always returns a too-short summary → Plan A retry, which also fails."""

    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        return AIMessage(content="ab")  # len 2 < min_length 5 → Plan A

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


class _GoodModel:
    """Always returns a well-formed summary (v0.2 no-regression baseline)."""

    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        return AIMessage(content="GOOD SUMMARY")

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


class _StubExternalizer(Externalizer):
    def __init__(self, ref: str = "stub-ref") -> None:
        self.ref = ref

    def externalize(self, blob: str, *, key: str | None = None) -> str:
        return self.ref

    async def aexternalize(self, blob: str, *, key: str | None = None) -> str:
        return self.ref

    def retrieve(self, ref: str) -> str:  # pragma: no cover
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


class _IdentityStrategy(DegradationStrategy):
    """Spy that records the context and returns an identity patch."""

    def __init__(self) -> None:
        self.last_summary: str | None = None
        self.called = False

    def degrade(self, ctx):  # type: ignore[override]
        self.called = True
        self.last_summary = ctx.summary
        return DegradationPatch(messages=None, plan="", reason="identity")

    async def adegrade(self, ctx):  # type: ignore[override]
        return self.degrade(ctx)


def _simple_token_counter(_messages) -> int:
    return 999  # deterministic; bypasses parent model-introspection branches


def _make_mw(model, **overrides) -> CompressionMiddleware:
    base = {
        "summary_model": model,
        "token_threshold": [("messages", 2)],
        "keep_recent": 1,
        "token_counter": _simple_token_counter,
        "l0_enabled": False,  # L3/degradation tests focus on L3, not L0 cleanup
    }
    base.update(overrides)
    return CompressionMiddleware(CompressionConfig(**base))


def _state(*messages):
    return {"messages": list(messages)}


def _three() -> list[HumanMessage]:
    return [HumanMessage(content="m0"), HumanMessage(content="m1"), HumanMessage(content="m2")]


def _summary_messages(result) -> list:
    return [m for m in result["messages"] if m.additional_kwargs.get("__summarization__")]


# --------------------------------------------------------------------------- #
# Plan A — retry with the fallback prompt recovers a bad primary summary
# --------------------------------------------------------------------------- #


def test_plan_a_retry_recovers_bad_primary_summary() -> None:
    model = _PlanAFixture()
    mw = _make_mw(model)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    # Primary (bad) → Plan A retry → good summary → happy path, no degradation.
    assert model.invoke_count == 2
    summ = _summary_messages(result)
    assert len(summ) == 1
    assert "RECOVERED SUMMARY" in summ[0].content


def test_plan_a_retry_exhausted_then_degrades() -> None:
    # "ab" is too short → Plan A retry, which also returns "ab" → degrade.
    model = _AlwaysTooShort()
    mw = _make_mw(model)  # no externalizer → Plan B
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert model.invoke_count == 2  # primary + retry
    # Plan B drops the failed summary entirely.
    assert _summary_messages(result) == []
    # The widened recent window keeps the original messages (state is small).
    contents = [m.content for m in result["messages"]]
    assert "m0" in contents and "m1" in contents and "m2" in contents


# --------------------------------------------------------------------------- #
# Plans B / D — result-level substitution
# --------------------------------------------------------------------------- #


def test_plan_b_when_no_externalizer_and_unrecoverable() -> None:
    # Empty summary → Plan C hint (no retry); no externalizer → degrade to B.
    model = _AlwaysBad()
    mw = _make_mw(model)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert model.invoke_count == 1  # empty → plan C → no Plan-A retry
    assert _summary_messages(result) == []  # Plan B drops the summary
    # Plan B result = [REMOVE_ALL sentinel, *widened recent window] (keeps all 3
    # here because the state is smaller than the widened window).
    assert isinstance(result["messages"][0], RemoveMessage)
    assert result["messages"][0].id == REMOVE_ALL_MESSAGES
    assert [m.content for m in result["messages"][1:]] == ["m0", "m1", "m2"]


def test_plan_d_when_externalizer_configured() -> None:
    model = _AlwaysBad()
    ext = _StubExternalizer(ref="ref-007")
    mw = _make_mw(model, degradation_externalizer=ext)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    # Plan D: [sentinel, ref_msg, *preserved_recent]; m0/m1 externalized away.
    ref_msgs = [
        m for m in result["messages"] if m.additional_kwargs.get("external_ref") == "ref-007"
    ]
    assert len(ref_msgs) == 1
    assert "ref-007" in ref_msgs[0].content
    contents = [m.content for m in result["messages"]]
    assert "m2" in contents  # preserved recent kept
    assert "m0" not in contents and "m1" not in contents  # head externalized


def test_plan_d_falls_back_to_plan_b_when_externalizer_raises() -> None:
    model = _AlwaysBad()
    mw = _make_mw(model, degradation_externalizer=_RaisingExternalizer())
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    # D's externalize raised → fall back to B (no external_ref anywhere).
    assert not any(m.additional_kwargs.get("external_ref") for m in result["messages"])
    assert _summary_messages(result) == []  # Plan B drops the summary


# --------------------------------------------------------------------------- #
# Custom strategy + Hook 3 still honoured
# --------------------------------------------------------------------------- #


def test_custom_identity_strategy_keeps_failed_summary_and_records_context() -> None:
    model = _AlwaysBad()
    spy = _IdentityStrategy()
    mw = _make_mw(model, degradation_strategy=spy)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert spy.called
    assert spy.last_summary == ""  # the failed (empty) summary reached the strategy
    # Identity patch → the happy result (with the failed summary) is kept.
    assert len(_summary_messages(result)) == 1


def test_post_compress_hook_runs_on_degraded_result() -> None:
    def post(state, result):
        out = dict(result)
        out["degraded"] = True
        return out

    model = _AlwaysBad()
    mw = _make_mw(model, post_compress_hook=post)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert result.get("degraded") is True  # Hook 3 saw the degraded result
    assert "messages" in result


# --------------------------------------------------------------------------- #
# Async path + v0.2 no-regression baseline
# --------------------------------------------------------------------------- #


async def test_async_plan_d_degradation() -> None:
    model = _AlwaysBad()
    ext = _StubExternalizer(ref="async-ref")
    mw = _make_mw(model, degradation_externalizer=ext)
    result = await mw.abefore_model(_state(*_three()), MagicMock())
    assert result is not None
    ref_msgs = [
        m for m in result["messages"] if m.additional_kwargs.get("external_ref") == "async-ref"
    ]
    assert len(ref_msgs) == 1


async def test_async_plan_a_retry_recovers() -> None:
    model = _PlanAFixture()
    mw = _make_mw(model)
    result = await mw.abefore_model(_state(*_three()), MagicMock())
    assert result is not None
    assert model.invoke_count == 2
    assert "RECOVERED SUMMARY" in _summary_messages(result)[0].content


def test_well_formed_summary_skips_degradation() -> None:
    # v0.2 no-regression: a good summary → happy path, single invoke, no degrade.
    model = _GoodModel()
    mw = _make_mw(model)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert model.invoke_count == 1
    summ = _summary_messages(result)
    assert len(summ) == 1
    assert "GOOD SUMMARY" in summ[0].content
    assert not any(m.additional_kwargs.get("external_ref") for m in result["messages"])


# --------------------------------------------------------------------------- #
# v0.4: degradation stamp (additional_kwargs["degradation"]) + logging
# --------------------------------------------------------------------------- #
#
# Item 4 attaches ``additional_kwargs["degradation"] = {"plan", "reason"[,
# "external_ref"]}`` to the first non-sentinel message of a degraded result and
# emits an INFO log line per degradation. These tests pin both surfaces.


def _first_non_sentinel(result) -> object | None:
    """The first message in a ``before_model`` result that is not the
    ``REMOVE_ALL_MESSAGES`` sentinel — the one ``_stamp_degradation`` stamps."""
    for m in result["messages"]:
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            continue
        return m
    return None


def test_plan_b_stamps_degradation_metadata() -> None:
    model = _AlwaysBad()
    mw = _make_mw(model)  # no externalizer → Plan B
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    first = _first_non_sentinel(result)
    assert first is not None
    meta = first.additional_kwargs.get("degradation")  # type: ignore[attr-defined]
    assert meta is not None
    assert meta["plan"] == "B"
    assert "reason" in meta
    assert "widened" in meta["reason"]  # the Plan-B reason text


def test_plan_d_stamps_degradation_with_external_ref() -> None:
    model = _AlwaysBad()
    ext = _StubExternalizer(ref="ref-007")
    mw = _make_mw(model, degradation_externalizer=ext)
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    first = _first_non_sentinel(result)
    assert first is not None
    meta = first.additional_kwargs.get("degradation")  # type: ignore[attr-defined]
    assert meta is not None
    assert meta["plan"] == "D"
    # Plan D's stamp carries the external_ref alongside plan/reason.
    assert meta["external_ref"] == "ref-007"
    # The message itself also retains the external_ref kwarg (from _plan_d).
    assert first.additional_kwargs.get("external_ref") == "ref-007"  # type: ignore[attr-defined]


def test_plan_c_stamp_via_custom_strategy() -> None:
    """Plan C is unreachable through the default chain (Plan B never raises,
    so C is the never-fails last resort that is shadowed by B); this pins the
    stamp contract for a Plan-C-shaped patch via a custom strategy."""

    class _PlanCStrategy(DegradationStrategy):
        def degrade(self, ctx):  # type: ignore[override]
            return DegradationPatch(
                messages=[ctx.remove_all_sentinel, *ctx.preserved_recent],
                plan="C",
                reason="conservative truncation to minimum recent window",
            )

        async def adegrade(self, ctx):  # type: ignore[override]
            return self.degrade(ctx)

    model = _AlwaysBad()
    mw = _make_mw(model, degradation_strategy=_PlanCStrategy())
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    first = _first_non_sentinel(result)
    assert first is not None
    meta = first.additional_kwargs.get("degradation")  # type: ignore[attr-defined]
    assert meta is not None
    assert meta["plan"] == "C"
    assert "truncation" in meta["reason"]


def test_identity_patch_not_stamped() -> None:
    """An identity patch (``plan=""``, ``messages=None``) keeps the failed-
    summary result as-is — no ``degradation`` kwarg is attached anywhere."""
    model = _AlwaysBad()
    mw = _make_mw(model, degradation_strategy=_IdentityStrategy())
    result = mw.before_model(_state(*_three()), MagicMock())
    assert result is not None
    assert not any(
        m.additional_kwargs.get("degradation") for m in result["messages"]
    )


# --------------------------------------------------------------------------- #
# caplog — INFO logging for Plan A retry, Plan B/D degradation
# --------------------------------------------------------------------------- #


def test_logging_plan_a_retry_recovers(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _PlanAFixture()
        mw = _make_mw(model)
        mw.before_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    # Plan A retry triggered (primary failed) and recovered (retry passed).
    assert any("plan A" in m and "retrying" in m for m in msgs), msgs
    assert any("recovered" in m for m in msgs), msgs


def test_logging_plan_a_retry_exhausted(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _AlwaysTooShort()  # primary too short → retry → still too short
        mw = _make_mw(model)
        mw.before_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    # Plan A retry fired, exhausted, then deferred to result-level degradation.
    assert any("retrying" in m for m in msgs), msgs
    assert any("exhausted" in m for m in msgs), msgs
    # And the result-level Plan-B degradation line fired afterwards.
    assert any("plan=B" in m for m in msgs), msgs


def test_logging_plan_b_degradation(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _AlwaysBad()
        mw = _make_mw(model)
        mw.before_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    degraded = [m for m in msgs if "degraded" in m]
    assert len(degraded) == 1, msgs
    assert "plan=B" in degraded[0]
    assert "reason=" in degraded[0]


def test_logging_plan_d_degradation_includes_external_ref(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _AlwaysBad()
        ext = _StubExternalizer(ref="ref-007")
        mw = _make_mw(model, degradation_externalizer=ext)
        mw.before_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    degraded = [m for m in msgs if "degraded" in m]
    assert len(degraded) == 1, msgs
    assert "plan=D" in degraded[0]
    assert "ref-007" in degraded[0]  # external_ref surfaces in the log line


async def test_async_logging_plan_d_degradation(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _AlwaysBad()
        ext = _StubExternalizer(ref="async-ref")
        mw = _make_mw(model, degradation_externalizer=ext)
        await mw.abefore_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("plan=D" in m and "async-ref" in m for m in msgs), msgs


def test_good_summary_emits_no_degradation_log(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="langcompress.middleware"):
        model = _GoodModel()
        mw = _make_mw(model)
        mw.before_model(_state(*_three()), MagicMock())
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("degraded" in m for m in msgs), msgs
    assert not any("plan A" in m for m in msgs), msgs
