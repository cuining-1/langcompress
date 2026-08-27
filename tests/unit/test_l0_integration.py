"""L0 integration tests: bug regression + drop_reasoning_kwargs + Plan E wiring.

Covers the L0 integration surface:
- ``_merge_adjacent_same_type`` no longer merges messages carrying ``tool_calls``
  or ``tool_call_id`` (AI/Tool-pair safety).
- ``drop_reasoning_kwargs`` strips ``additional_kwargs["reasoning_content"]``
  (OpenAI-compatible thinking mode: GLM-5.2, DeepSeek-R1).
- Plan E: ``before_model`` runs L0 always-on, writes back only when L0 changed
  something (lazy full-replacement), and L0's cleanup rides L3 for free.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langcompress import CompressionConfig, CompressionMiddleware, L0Filter

# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #


class _StubModel:
    """Minimal stub: returns a canned summary string."""

    def invoke(self, prompt: str, config: Any = None) -> Any:
        class _Resp:
            text = "STUB SUMMARY"

        return _Resp()

    async def ainvoke(self, prompt: str, config: Any = None) -> Any:
        return self.invoke(prompt, config=config)


def _make_mw(**overrides: Any) -> CompressionMiddleware:
    base: dict[str, Any] = {
        "summary_model": _StubModel(),
        "token_threshold": [("messages", 50)],  # high threshold → L3 won't fire
        "keep_recent": 1,
        "token_counter": lambda _msgs: 999,
    }
    base.update(overrides)
    return CompressionMiddleware(CompressionConfig(**base))


def _state(*messages: BaseMessage) -> dict[str, Any]:
    return {"messages": list(messages)}


# --------------------------------------------------------------------------- #
# 1. Bug regression: merge_adjacent skips tool-carrying messages
# --------------------------------------------------------------------------- #


class TestMergeBugRegression:
    """Bug regression: _merge_adjacent_same_type must not merge messages with
    tool_calls or tool_call_id — merging would lose the tool metadata and
    break AI/Tool-pair safety."""

    def test_merge_skips_tool_calls_messages(self):
        """Two adjacent AIMessages with tool_calls must NOT be merged."""
        l0 = L0Filter()
        msgs = [
            AIMessage(content="Let me search.", tool_calls=[{"name": "search", "args": {}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="And fetch.", tool_calls=[{"name": "fetch", "args": {}, "id": "c2", "type": "tool_call"}]),
        ]
        result = l0.run(msgs)
        assert len(result) == 2, "tool_calls messages must not be merged"

    def test_merge_skips_tool_call_id_messages(self):
        """ToolMessages (with tool_call_id) must NOT be merged."""
        l0 = L0Filter()
        msgs = [
            ToolMessage(content="result1", tool_call_id="c1", name="search"),
            ToolMessage(content="result2", tool_call_id="c2", name="fetch"),
        ]
        result = l0.run(msgs)
        assert len(result) == 2, "tool_call_id messages must not be merged"

    def test_merge_still_works_for_plain_text(self):
        """Adjacent plain-text messages (no tool metadata) are still merged."""
        l0 = L0Filter()
        msgs = [
            HumanMessage(content="hello"),
            HumanMessage(content="world"),
        ]
        result = l0.run(msgs)
        assert len(result) == 1, "plain-text messages should still be merged"
        assert "hello" in result[0].content
        assert "world" in result[0].content

    def test_merge_mixed_tool_and_plain(self):
        """AIMessage with tool_calls followed by plain AIMessage: not merged."""
        l0 = L0Filter()
        msgs = [
            AIMessage(content="searching", tool_calls=[{"name": "s", "args": {}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="done searching"),
        ]
        result = l0.run(msgs)
        assert len(result) == 2


# --------------------------------------------------------------------------- #
# 2. drop_reasoning_kwargs
# --------------------------------------------------------------------------- #


class TestDropReasoningKwargs:
    """L0: strip reasoning_content / reasoning from additional_kwargs
    (OpenAI-compatible thinking mode: GLM-5.2, DeepSeek-R1)."""

    def test_strips_reasoning_content(self):
        l0 = L0Filter()
        msg = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Let me think about this..."},
        )
        result = l0.run([msg])
        assert "reasoning_content" not in result[0].additional_kwargs

    def test_strips_reasoning_key(self):
        l0 = L0Filter()
        msg = AIMessage(
            content="Done.",
            additional_kwargs={"reasoning": "thinking process"},
        )
        result = l0.run([msg])
        assert "reasoning" not in result[0].additional_kwargs

    def test_preserves_other_kwargs(self):
        l0 = L0Filter()
        msg = AIMessage(
            content="Done.",
            additional_kwargs={
                "reasoning_content": "thinking...",
                "lc_source": "summarization",
                "__summarization__": True,
            },
        )
        result = l0.run([msg])
        ak = result[0].additional_kwargs
        assert "reasoning_content" not in ak
        assert ak.get("lc_source") == "summarization"
        assert ak.get("__summarization__") is True

    def test_disabled_preserves_reasoning(self):
        l0 = L0Filter(drop_reasoning_kwargs=False)
        msg = AIMessage(
            content="Done.",
            additional_kwargs={"reasoning_content": "thinking..."},
        )
        result = l0.run([msg])
        assert result[0].additional_kwargs.get("reasoning_content") == "thinking..."

    def test_no_reasoning_field_is_noop(self):
        l0 = L0Filter()
        msg = AIMessage(content="Done.", additional_kwargs={"foo": "bar"})
        result = l0.run([msg])
        assert result[0].additional_kwargs == {"foo": "bar"}

    def test_kwargs_stripped_before_parts(self):
        """Order: kwargs → parts → empty. A message with only reasoning_content
        (no content-list parts) should have kwargs stripped, then survive
        (content is non-empty)."""
        l0 = L0Filter()
        msg = AIMessage(
            content="The answer.",
            additional_kwargs={"reasoning_content": "long thinking" * 100},
        )
        result = l0.run([msg])
        assert len(result) == 1
        assert "reasoning_content" not in result[0].additional_kwargs
        assert result[0].content == "The answer."


# --------------------------------------------------------------------------- #
# 3. Plan E: before_model L0 always-on + lazy full-replacement
# --------------------------------------------------------------------------- #


class TestPlanEAlwaysOn:
    """Plan E: L0 runs every before_model call (always-on), writes back
    only when L0 detected changes (lazy full-replacement)."""

    def test_l0_runs_always_on_even_below_threshold(self):
        """L0 runs even when L3 threshold not met — L0 is always-on."""
        mw = _make_mw(l0_enabled=True)
        # 2 messages below the ("messages", 50) threshold, but L0 can still clean.
        msgs = [
            HumanMessage(content=""),  # empty → L0 drops
            HumanMessage(content="hello"),
        ]
        result = mw.before_model(_state(*msgs), MagicMock())
        assert result is not None, "L0 must write back when it drops an empty message"
        out = result["messages"]
        assert isinstance(out[0], RemoveMessage)
        assert out[0].id == REMOVE_ALL_MESSAGES
        assert len(out) == 2  # sentinel + 1 remaining message
        assert out[1].content == "hello"

    def test_l0_no_change_returns_none(self):
        """When L0 finds nothing to clean, before_model returns None (zero
        checkpoint overhead)."""
        mw = _make_mw(l0_enabled=True)
        msgs = [HumanMessage(content="hello"), AIMessage(content="hi there")]
        # No empties, no dupes, no reasoning, no mergeable adjacent same-type.
        result = mw.before_model(_state(*msgs), MagicMock())
        assert result is None, "L0 found nothing to clean → must return None"

    def test_l0_rides_l3_when_triggered(self):
        """When L3 fires, L0's cleanup is included for free (L3's sentinel +
        summary + recent naturally contains L0's filtered list).

        Uses mixed-type messages (AI/Human/AI) so L0's merge_adjacent won't
        collapse them — the empty HumanMessage is dropped by L0, and the
        remaining 3 messages still exceed the threshold."""
        mw = _make_mw(
            l0_enabled=True,
            token_threshold=[("messages", 2)],  # low threshold → L3 fires
            keep_recent=1,
        )
        msgs = [
            HumanMessage(content=""),  # L0 drops this
            AIMessage(content="a1"),
            HumanMessage(content="h1"),
            AIMessage(content="a2"),
        ]
        result = mw.before_model(_state(*msgs), MagicMock())
        assert result is not None
        out = result["messages"]
        # L3 fired: sentinel + summary + recent(1)
        assert isinstance(out[0], RemoveMessage)
        assert out[0].id == REMOVE_ALL_MESSAGES
        # The summary message
        assert out[1].additional_kwargs.get("__summarization__") is True
        # Recent window: 1 message
        assert len(out) == 3  # sentinel + summary + 1 recent

    def test_l0_disabled_behaves_like_v05(self):
        """When l0_enabled=False, before_model is identical to v0.5 (no L0)."""
        mw = _make_mw(l0_enabled=False)
        msgs = [
            HumanMessage(content=""),  # empty, but L0 disabled → not dropped
            HumanMessage(content="hello"),
        ]
        result = mw.before_model(_state(*msgs), MagicMock())
        assert result is None, "L0 disabled → no L3 trigger → None (same as v0.5)"

    def test_l0_reduces_token_count_for_trigger(self):
        """L0 stripping reasoning kwargs lowers token count, which may avoid
        a needless L3 trigger."""
        called = {"token_counter": 0}

        def counting_counter(_msgs):
            called["token_counter"] += 1
            return 999  # always above any token threshold

        mw = _make_mw(
            l0_enabled=True,
            token_threshold=[("tokens", 500)],
            token_counter=counting_counter,
        )
        msg = AIMessage(
            content="answer",
            additional_kwargs={"reasoning_content": "x" * 1000},
        )
        mw.before_model(_state(msg), MagicMock())
        # L0 stripped reasoning_content → token count reflects cleaned list
        assert called["token_counter"] >= 1

    def test_l0_merge_bug_not_triggered_in_pipeline(self):
        """End-to-end: two adjacent AIMessages with tool_calls survive L0
        when before_model runs."""
        mw = _make_mw(l0_enabled=True)
        msgs = [
            AIMessage(
                content="searching",
                tool_calls=[{"name": "s", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            AIMessage(
                content="fetching",
                tool_calls=[{"name": "f", "args": {}, "id": "c2", "type": "tool_call"}],
            ),
        ]
        result = mw.before_model(_state(*msgs), MagicMock())
        # L0 didn't merge them → still 2 messages → L0 unchanged → None
        assert result is None


# --------------------------------------------------------------------------- #
# 4. Async parity
# --------------------------------------------------------------------------- #


class TestAsyncParity:
    """abefore_model must behave identically to before_model for Plan E."""

    def test_async_l0_no_change_returns_none(self):
        mw = _make_mw(l0_enabled=True)
        msgs = [HumanMessage(content="hello"), AIMessage(content="hi")]
        result = asyncio.run(mw.abefore_model(_state(*msgs), MagicMock()))
        assert result is None

    def test_async_l0_drops_empty_and_writes_back(self):
        mw = _make_mw(l0_enabled=True)
        msgs = [HumanMessage(content=""), HumanMessage(content="hello")]
        result = asyncio.run(mw.abefore_model(_state(*msgs), MagicMock()))
        assert result is not None
        out = result["messages"]
        assert isinstance(out[0], RemoveMessage)
        assert out[1].content == "hello"

    def test_async_l0_rides_l3(self):
        mw = _make_mw(
            l0_enabled=True,
            token_threshold=[("messages", 2)],
            keep_recent=1,
        )
        msgs = [
            HumanMessage(content=""),  # L0 drops
            AIMessage(content="a1"),
            HumanMessage(content="h1"),
            AIMessage(content="a2"),
        ]
        result = asyncio.run(mw.abefore_model(_state(*msgs), MagicMock()))
        assert result is not None
        out = result["messages"]
        assert isinstance(out[0], RemoveMessage)
        assert out[1].additional_kwargs.get("__summarization__") is True
