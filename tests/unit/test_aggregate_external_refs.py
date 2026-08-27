"""Tests for :func:`langcompress.aggregate_external_refs` after the v0.4 Item 3
fix — it now scans **every** message for ``external_ref``, not just
:class:`ToolMessage`.

Pre-v0.4 the scan was ``isinstance(m, ToolMessage)``-filtered, which silently
dropped the L3 Plan-D reference (stamped by
:meth:`DefaultDegradationStrategy._plan_d` on the summary-shaped
:class:`HumanMessage` built by ``summary_message_builder``). v0.4 lifts the
filter so both L2 (source-side ``wrap_tool_call``) and L3 (Plan-D) refs are
collected.

Coverage:

1. **Plan-D shape** — a ``HumanMessage`` carrying ``external_ref`` is now
   collected (was dropped pre-v0.4); its value is ``""`` (HumanMessage has no
   ``name``).
2. **L2 shape** — a ``ToolMessage`` carrying ``external_ref`` is still
   collected with the tool name (no regression).
3. **Mixed** — L2 + L3 refs in the same result dict both come back.
4. **Sentinel-safe** — a ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` in the
   list (which the real ``before_model`` result always starts with) does not
   perturb the scan.
5. **Reserved-key guard** — documented in the docstring as a
   langcompress-reserved key; the test asserts the deterministic
   ``{ref: name-or-""}`` contract the host ``post_compress_hook`` relies on.
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langcompress import aggregate_external_refs

# --------------------------------------------------------------------------- #
# Plan-D shape — HumanMessage with external_ref (the v0.4 fix's whole point)
# --------------------------------------------------------------------------- #


def test_collects_human_message_external_ref_plan_d_shape() -> None:
    """Plan D stamps ``external_ref`` on the summary-shaped message built by
    ``summary_message_builder`` — a :class:`HumanMessage` by default. Before
    v0.4 the ``isinstance(m, ToolMessage)`` filter dropped it; now it is
    collected. The value is ``""`` because ``HumanMessage`` carries no
    ``name`` attribute (the ref itself is the meaningful key)."""
    ref = "file:///degraded/head.md"
    msgs = [
        HumanMessage(content="[Conversation history externalized. Ref: file:///...]"),
    ]
    # Plan D's actual stamp is via additional_kwargs — simulate it directly.
    msgs[0] = msgs[0].model_copy(
        update={"additional_kwargs": {"external_ref": ref}}
    )
    out = aggregate_external_refs({"messages": msgs})
    assert out == {ref: ""}  # collected; name "" because HumanMessage has none


def test_plan_d_ref_collected_alongside_l2_tool_refs() -> None:
    """A degraded result that contains both the L3 Plan-D reference message
    and a surviving L2 ``ToolMessage`` must surface both refs."""
    plan_d_ref = "file:///degraded/head.md"
    l2_ref = "file:///tools/big_tool.md"
    ref_msg = HumanMessage(content="degraded head ref").model_copy(
        update={"additional_kwargs": {"external_ref": plan_d_ref}}
    )
    tool_msg = ToolMessage(
        content="ref text",
        tool_call_id="c1",
        name="big_tool",
        additional_kwargs={"external_ref": l2_ref},
    )
    out = aggregate_external_refs({"messages": [ref_msg, tool_msg]})
    assert out == {plan_d_ref: "", l2_ref: "big_tool"}


# --------------------------------------------------------------------------- #
# L2 shape — no regression on the existing ToolMessage path
# --------------------------------------------------------------------------- #


def test_collects_tool_message_external_ref_l2_shape() -> None:
    """L2 source compression stamps ``external_ref`` on a ``ToolMessage``;
    the value is the tool name. Lifting the ToolMessage filter does not
    change the L2 path's output."""
    l2_ref = "file:///tools/big.md"
    msgs = [
        ToolMessage(
            content="ref text",
            tool_call_id="c1",
            name="big_tool",
            additional_kwargs={"external_ref": l2_ref},
        )
    ]
    out = aggregate_external_refs({"messages": msgs})
    assert out == {l2_ref: "big_tool"}


def test_tool_message_without_external_ref_is_skipped() -> None:
    """A plain ``ToolMessage`` (no external_ref) must not produce a spurious
    entry — the scan reads ``additional_kwargs.get("external_ref")`` and
    skips when it is absent."""
    msgs = [
        ToolMessage(content="ok", tool_call_id="c1", name="small"),
        AIMessage(content="bye"),
    ]
    assert aggregate_external_refs({"messages": msgs}) == {}


# --------------------------------------------------------------------------- #
# Sentinel-safe — the real before_model result starts with a RemoveMessage
# --------------------------------------------------------------------------- #


def test_remove_all_sentinel_does_not_perturb_scan() -> None:
    """The real ``before_model`` result is
    ``[RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages, *preserved_recent]``.
    The sentinel has ``content == ""`` and no ``external_ref``; it must be
    skipped without raising."""
    ref = "file:///degraded/head.md"
    ref_msg = HumanMessage(content="degraded").model_copy(
        update={"additional_kwargs": {"external_ref": ref}}
    )
    msgs = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        ref_msg,
        HumanMessage(content="preserved recent"),
    ]
    out = aggregate_external_refs({"messages": msgs})
    assert out == {ref: ""}


# --------------------------------------------------------------------------- #
# Mixed / edge shapes — non-ToolMessage message types and empty/missing inputs
# --------------------------------------------------------------------------- #


def test_collects_ref_from_any_message_type() -> None:
    """The scan is type-agnostic: any ``BaseMessage`` subclass carrying
    ``external_ref`` is collected. Guards against a future regression to a
    type-filtered scan. ``SystemMessage`` is the most "surprising" carrier —
    it would only get ``external_ref`` via a custom builder, but if it does,
    the contract still holds (value ``""`` — no ``name`` on SystemMessage)."""
    ref = "file:///weird/system.md"
    sys_msg = SystemMessage(content="...").model_copy(
        update={"additional_kwargs": {"external_ref": ref}}
    )
    out = aggregate_external_refs({"messages": [sys_msg, AIMessage(content="x")]})
    assert out == {ref: ""}


def test_aggregate_external_refs_empty_inputs() -> None:
    assert aggregate_external_refs({"messages": []}) == {}
    assert aggregate_external_refs({}) == {}
    assert aggregate_external_refs({"messages": [HumanMessage(content="plain")]}) == {}


# --------------------------------------------------------------------------- #
# Reserved-key contract — {ref: name-or-""}, deterministic for the host hook
# --------------------------------------------------------------------------- #


def test_reserved_key_contract_value_is_name_or_empty() -> None:
    """The host ``post_compress_hook`` builds
    ``return {**result, "external_refs": aggregate_external_refs(result)}``
    and the dict-merge reducer accumulates that into state. The hook relies
    on a deterministic value shape: the tool name for L2 refs, ``""`` for any
    message without a ``name`` (Plan-D HumanMessage, SystemMessage, ...)."""
    tool_ref = "file:///t.md"
    plan_d_ref = "file:///d.md"
    msgs = [
        ToolMessage(
            content="r",
            tool_call_id="c",
            name="pdf_tool",
            additional_kwargs={"external_ref": tool_ref},
        ),
        HumanMessage(content="ref").model_copy(
            update={"additional_kwargs": {"external_ref": plan_d_ref}}
        ),
    ]
    out = aggregate_external_refs({"messages": msgs})
    assert out == {tool_ref: "pdf_tool", plan_d_ref: ""}
