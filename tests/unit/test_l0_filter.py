"""Unit tests for :class:`langcompress.L0Filter` (design §4.1).

L0 is a pure-Python heuristic stage: drop empty messages, strip thinking/
reasoning parts, drop adjacent duplicates, merge adjacent same-type string
content. It must return a *new* list and never mutate its input.
"""
from langchain_core.messages import AIMessage, HumanMessage

from langcompress.pipeline.l0_filter import L0Filter


def _human(content):
    return HumanMessage(content=content)


def _ai(content):
    return AIMessage(content=content)


def test_drops_empty_string_messages():
    f = L0Filter()
    out = f.run([_human(""), _human("hi"), _ai(""), _ai("yo")])
    # "hi"(human) and "yo"(ai) are different types → not merged
    assert [m.content for m in out] == ["hi", "yo"]


def test_drops_empty_list_content():
    f = L0Filter()
    out = f.run([_human([]), _human("keep")])
    assert [m.content for m in out] == ["keep"]


def test_strips_reasoning_parts():
    f = L0Filter()
    msg = _ai(
        [
            {"type": "thinking", "thinking": "secret reasoning"},
            {"type": "reasoning", "reasoning": "more secret"},
            {"type": "text", "text": "final answer"},
        ]
    )
    out = f.run([msg])
    assert len(out) == 1
    content = out[0].content
    assert isinstance(content, list)
    assert all(p.get("type") not in {"thinking", "reasoning"} for p in content)
    assert any(p.get("type") == "text" for p in content)


def test_drops_adjacent_duplicates():
    # merge_adjacent disabled to isolate dedup behaviour.
    f = L0Filter(merge_adjacent=False)
    out = f.run([_human("hi"), _human("hi"), _human("bye")])
    assert [m.content for m in out] == ["hi", "bye"]


def test_merges_adjacent_same_type_string_content():
    f = L0Filter()
    out = f.run([_ai("a"), _ai("b"), _human("h"), _ai("c")])
    # "a","b" (both ai, str) merge into "a\nb"; human breaks the run; "c" stays.
    assert out[0].content == "a\nb"
    assert out[0].type == "ai"
    assert out[1].content == "h"
    assert out[2].content == "c"


def test_does_not_mutate_input():
    f = L0Filter()
    msgs = [_human("hi"), _human(""), _ai("yo")]
    original = [m.content for m in msgs]
    _ = f.run(msgs)
    assert [m.content for m in msgs] == original


def test_disable_flags_passthrough():
    f = L0Filter(drop_duplicates=False, drop_empty=False, merge_adjacent=False)
    out = f.run([_human("x"), _human("x"), _human("")])
    assert [m.content for m in out] == ["x", "x", ""]


def test_arun_delegates_to_run():
    import asyncio

    f = L0Filter()
    out = asyncio.run(f.arun([_human("a"), _human("")]))
    assert [m.content for m in out] == ["a"]
