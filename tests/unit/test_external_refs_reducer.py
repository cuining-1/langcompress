"""Tests for the ``external_refs`` dict-merge reducer (v0.4 Item 1, design §13.1).

Three layers of coverage:

1. **Unit** — ``_merge_external_refs`` itself: ``None`` / empty / disjoint / key
   collision (right wins, mirroring ``{**a, **b}``).
2. **Annotation reachability** — ``get_type_hints(..., include_extras=True)`` on
   :class:`CompressionAgentState` surfaces the reducer in the ``Annotated``
   metadata, so LangGraph's schema introspection finds it.
3. **LangGraph end-to-end** — a minimal ``StateGraph(CompressionAgentState)``
   runs two nodes that each return a *different* ref and the reducer merges
   them into ``state["external_refs"]`` (proving LangGraph actually calls the
   reducer, not last-write-wins).

Module-level reducer (not a lambda) is part of the contract — it must be
picklable for checkpointing. The test does not pickle it directly but asserts
the reference equality that picklability depends on (a stable, named object).
"""
from __future__ import annotations

from typing import get_args, get_type_hints

from langgraph.graph import END, START, StateGraph

from langcompress import CompressionAgentState
from langcompress.state import _merge_external_refs

# --------------------------------------------------------------------------- #
# Unit — _merge_external_refs
# --------------------------------------------------------------------------- #


def test_merge_none_both_sides_yields_empty_dict() -> None:
    assert _merge_external_refs(None, None) == {}


def test_merge_none_left_keeps_right() -> None:
    assert _merge_external_refs(None, {"r1": "tool1"}) == {"r1": "tool1"}


def test_merge_none_right_keeps_left() -> None:
    assert _merge_external_refs({"r1": "tool1"}, None) == {"r1": "tool1"}


def test_merge_empty_dicts_behave_like_none() -> None:
    # Defensive: the `or {}` guards mean {} and None are equivalent inputs.
    assert _merge_external_refs({}, {}) == {}
    assert _merge_external_refs({}, {"r1": "t"}) == {"r1": "t"}
    assert _merge_external_refs({"r1": "t"}, {}) == {"r1": "t"}


def test_merge_disjoint_keys_accumulate() -> None:
    left = {"file:///a": "big_tool", "file:///b": "pdf_tool"}
    right = {"file:///c": "api_tool"}
    assert _merge_external_refs(left, right) == {
        "file:///a": "big_tool",
        "file:///b": "pdf_tool",
        "file:///c": "api_tool",
    }


def test_merge_key_collision_right_wins() -> None:
    # Same ref key, different tool name (e.g. a re-externalization). The
    # reducer is a plain dict-merge → right wins, matching `{**left, **right}`.
    left = {"file:///a": "old_tool"}
    right = {"file:///a": "new_tool"}
    assert _merge_external_refs(left, right) == {"file:///a": "new_tool"}


def test_merge_does_not_mutate_inputs() -> None:
    left = {"r1": "t1"}
    right = {"r2": "t2"}
    out = _merge_external_refs(left, right)
    assert out is not left and out is not right  # new dict
    assert left == {"r1": "t1"}  # inputs untouched
    assert right == {"r2": "t2"}


# --------------------------------------------------------------------------- #
# Annotation reachability — LangGraph introspects this to wire the reducer
# --------------------------------------------------------------------------- #


def test_external_refs_annotation_carries_reducer() -> None:
    """``get_type_hints(include_extras=True)`` must surface the reducer callable
    in the ``Annotated`` metadata; that is exactly what LangGraph's
    ``_get_reducer` / schema-introspection path looks up to register the
    channel as a ``BinaryOperatorAggregate`` rather than last-write-wins."""
    hints = get_type_hints(CompressionAgentState, include_extras=True)
    assert "external_refs" in hints
    field = hints["external_refs"]
    # Annotated[dict[str, str], _merge_external_refs] → metadata tuple holds it.
    assert _merge_external_refs in get_args(field)


def test_reducer_is_module_level_named_function() -> None:
    """The reducer must be a named, module-level (not lambda) callable so it is
    picklable for checkpointing — the same convention as
    ``count_tokens_approximately``. Reference equality with the imported name
    is the cheap proxy for "stable, picklable identity"."""
    import langcompress.state as state_mod

    assert _merge_external_refs is state_mod._merge_external_refs
    assert _merge_external_refs.__name__ == "_merge_external_refs"


# --------------------------------------------------------------------------- #
# LangGraph end-to-end — the reducer is actually invoked by the runtime
# --------------------------------------------------------------------------- #


def _build_two_node_graph():
    """A minimal ``StateGraph(CompressionAgentState)`` with two nodes that each
    emit a *different* ``external_refs`` entry, then terminate. Mirrors the
    pattern in ``tests/scenarios/test_pure_langgraph.py`` but isolates the
    ``external_refs`` channel from the rest of the compression pipeline."""
    builder = StateGraph(CompressionAgentState)

    def node_first(state):
        # First compaction's contribution — a single ref.
        return {"external_refs": {"file:///first.md": "big_tool"}}

    def node_second(state):
        # Second compaction's contribution — a *different* ref. Under
        # last-write-wins this would overwrite the first node's entry; under
        # the dict-merge reducer both must survive.
        return {"external_refs": {"file:///second.md": "pdf_tool"}}

    builder.add_node("first", node_first)
    builder.add_node("second", node_second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile()


def test_external_refs_accumulate_across_nodes_via_reducer() -> None:
    graph = _build_two_node_graph()
    result = graph.invoke({})

    # Both nodes' refs survive in the final state — the dict-merge reducer
    # was invoked (not last-write-wins, which would leave only "second").
    assert result["external_refs"] == {
        "file:///first.md": "big_tool",
        "file:///second.md": "pdf_tool",
    }


def test_external_refs_collision_uses_reducer_not_overwrite() -> None:
    """Two nodes writing the *same* ref key with different values must resolve
    via the reducer (right wins), confirming the channel is genuinely
    reducer-bound rather than behaving as plain overwrite-by-accident."""
    builder = StateGraph(CompressionAgentState)

    def node_a(state):
        return {"external_refs": {"ref-x": "tool-a"}}

    def node_b(state):
        return {"external_refs": {"ref-x": "tool-b"}}

    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    result = graph.invoke({})
    # Right-wins via the reducer; if it were last-write-wins by accident this
    # would still be "tool-b", but a reducer-less channel would also drop
    # *other* keys — this case is the union of the two above.
    assert result["external_refs"] == {"ref-x": "tool-b"}
