"""Design-intent compliance suite — verifies each compression level against the
*stated intent* of ``docs/design.md``.

The existing unit/scenario tests pin mechanics; this suite pins the **contract
the design document claims**, per level:

- §4.1 L0 — pure-rule filtering (no LLM), 5-15% typical reduction, never grows.
- §4.2 L1 — deliberately absent from core (MVP defers it / delegates to
  ContextZip; core must stay free of small-model deps).
- §4.3 L2 — big tool output → lightweight reference, >= 60% single-item
  reduction, AI/Tool pairing preserved, content recoverable via the ref.
- §4.4/§7/§9 L3 — eight-segment template (incl. the §7 "Entity State"
  enhancement), context reconstruction = [summary] + [recent N verbatim],
  >= 70% reduction.
- §4.5 L4 — externalized content leaves only a reference, >= 90% reduction.
- §5.2 anti-retrigger — when the head message is already a summary, only the
  message-count dimension may trigger (token/fraction re-firing on a large
  preserved ToolMessage buys nothing).
- §3.3 lossy-but-recoverable — end-to-end: L2 externalizes at the source, L3
  compacts centrally, refs accumulate in state (dict-merge reducer), and every
  ref retrieves the original content back.

Reduction assertions pin the **design floor** (the low end of each claimed
band), never the ceiling — compressing harder than the design band is never a
violation, compressing softer than the floor is.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.tool_node import ToolCallRequest

from langcompress import (
    CompressionConfig,
    CompressionMiddleware,
    FilesystemExternalizer,
    L0Filter,
    ToolCallExternalizerMiddleware,
    aggregate_external_refs,
)
from langcompress.summarizer.templates import (
    EIGHT_SEGMENT_TEMPLATE,
    FALLBACK_SUMMARY_PROMPT,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _chars(messages) -> int:
    """Total str(content) length — the cheap proxy for token volume."""
    return sum(len(str(getattr(m, "content", ""))) for m in messages)


def _reduction(before: int, after: int) -> float:
    return 1 - after / max(before, 1)


class _StubSummaryModel:
    """Deterministic summary model — returns a fixed well-formed summary."""

    def __init__(self, summary: str = "GOOD SUMMARY") -> None:
        self.summary = summary
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        return AIMessage(content=self.summary)

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


def _make_mw(model, **overrides) -> CompressionMiddleware:
    base = {
        "summary_model": model,
        "token_threshold": [("messages", 3)],
        "keep_recent": 1,
        "token_counter": lambda _msgs: 999,
        "l0_enabled": False,  # design-intent tests verify L3, not L0 cleanup
    }
    base.update(overrides)
    return CompressionMiddleware(CompressionConfig(**base))


def _request(tool_call=None):
    return ToolCallRequest(
        tool_call=tool_call or {"name": "big", "args": {}, "id": "c1", "type": "tool_call"},
        tool=None,
        state={"messages": []},
        runtime=None,
    )


# --------------------------------------------------------------------------- #
# §4.1 L0 — content filter
# --------------------------------------------------------------------------- #


def test_l0_pure_rules_no_llm_dependency():
    """L0 must be pure-rule (design: "无需 LLM 调用，纯规则过滤"): constructible
    and runnable with zero model/service objects, and the module must not
    import langchain (the heavy adapter) — only langchain_core."""
    filt = L0Filter()  # no model, no client, no network — pure rules
    out = filt.run([HumanMessage(content="hello"), AIMessage(content="hi")])
    assert len(out) == 2

    import langcompress.pipeline.l0_filter as l0_mod

    source = open(l0_mod.__file__, encoding="utf-8").read()  # noqa: SIM115
    assert "import langchain\n" not in source
    assert "from langchain import" not in source
    assert "from langchain." not in source  # only langchain_core.* allowed


def test_l0_typical_reduction_in_design_band():
    """Design claims L0 typically reduces 5-15%. A clean conversation must be
    a no-op (0% — L0 never invents content); a conversation carrying the junk
    L0 targets (empty messages, thinking-only messages) must land inside the
    design band."""
    filt = L0Filter()
    clean = [
        HumanMessage(content=f"question {i} " + "q" * 90) if i % 2 == 0 else AIMessage(content=f"answer {i} " + "a" * 90)
        for i in range(10)
    ]
    assert filt.run(clean) == clean  # clean input: untouched, 0% reduction

    # 10 clean messages (1000 chars) + one thinking-only AIMessage (100 chars
    # of reasoning part, no text part) + one empty message.
    thinking_only = AIMessage(
        content=[{"type": "thinking", "thinking": "t" * 100}]
    )
    dirty = [*clean, thinking_only, AIMessage(content="")]
    before, after = _chars(dirty), _chars(filt.run(dirty))
    reduction = _reduction(before, after)
    # thinking-only drops to empty → removed; empty message removed.
    # 100 chars of junk in 1100 total ≈ 9% — inside the 5-15% band.
    assert 0.05 <= reduction <= 0.15, f"L0 reduction {reduction:.0%} outside design band"


def test_l0_never_grows_context():
    """L0 is a filter — it never invents content and never balloons the
    context. Adjacent same-type merging (design §4.1 "合并相邻的同类内容块")
    is a structural join that keeps all original text and adds only the
    join separator; an alternating conversation (nothing adjacent to merge)
    passes through completely unchanged."""
    filt = L0Filter()

    # Adjacent same-type → merged: 3 messages → 2, full text preserved, only
    # the "\n" join separator added (never a multiplication).
    msgs = [
        HumanMessage(content="a" * 500),
        HumanMessage(content="b" * 500),
        AIMessage(content="c" * 500),
    ]
    out = filt.run(msgs)
    assert len(out) == 2  # structural join: one fewer message
    assert _chars(out) <= _chars(msgs) + 1  # +1 = the join separator only

    # Nothing adjacent to merge → byte-identical passthrough.
    alternating = [
        HumanMessage(content="x" * 300),
        AIMessage(content="y" * 300),
        HumanMessage(content="z" * 300),
    ]
    assert filt.run(alternating) == alternating


# --------------------------------------------------------------------------- #
# §4.2 L1 — token pruning is deliberately absent (deferred / delegated)
# --------------------------------------------------------------------------- #


def test_l1_deliberately_absent_from_core():
    """Design §4.2/§17.2: MVP contains NO L1 (needs a small model; delegated
    to ContextZip). Core must not gain an L1 stage — an accidental
    ``pipeline/l1_prune`` module would smuggle a small-model dependency into
    the zero-dependency core. This pins the negative contract."""
    try:
        importlib.import_module("langcompress.pipeline.l1_prune")
    except ImportError:
        pass
    else:  # pragma: no cover
        raise AssertionError(
            "langcompress.pipeline.l1_prune exists — design §4.2 defers L1 "
            "(delegate to ContextZip); core must stay free of small-model deps"
        )


# --------------------------------------------------------------------------- #
# §4.3 L2 — reference substitution at the source
# --------------------------------------------------------------------------- #


def test_l2_single_item_reduction_meets_design_floor(tmp_path):
    """Design claims L2 compresses a single large item 60-90%. A 3000-char
    tool output replaced by a reference message must reduce at least 60%,
    while preserving the pairing contract (tool_call_id + name) so the
    AI(tool_calls)/Tool pair survives."""
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    original = ToolMessage(content="D" * 3000, tool_call_id="c1", name="web_fetch")

    out = mw.wrap_tool_call(_request(), lambda _r: original)

    assert isinstance(out, ToolMessage)
    assert out.tool_call_id == "c1"  # pairing preserved
    assert out.name == "web_fetch"  # pairing preserved
    reduction = _reduction(3000, len(str(out.content)))
    assert reduction >= 0.60, f"L2 reduction {reduction:.0%} below design floor 60%"


def test_l2_reference_is_recoverable(tmp_path):
    """§3.3 "引用优于保留": the reference message must carry a retrievable ref
    and the externalizer must hand back the exact original content
    (just-in-time retrieval)."""
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    payload = "API-RESPONSE-" + "P" * 2500

    out = mw.wrap_tool_call(_request(), lambda _r: ToolMessage(content=payload, tool_call_id="c1", name="api"))

    ref = out.additional_kwargs.get("external_ref")
    assert ref, "reference message must carry external_ref"
    assert "Ref:" in str(out.content)  # the ref is visible in-context
    assert ext.retrieve(ref) == payload  # original fully recoverable


def test_l2_small_results_untouched(tmp_path):
    """L2 only substitutes *large* outputs — small tool results pass through
    verbatim (no pointless indirection below the threshold)."""
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    small = ToolMessage(content="ok", tool_call_id="c2", name="calc")

    out = mw.wrap_tool_call(_request(), lambda _r: small)

    assert out is small


# --------------------------------------------------------------------------- #
# §4.4/§7/§9 L3 — semantic summary
# --------------------------------------------------------------------------- #


def test_eight_segment_template_complete():
    """§7: the default template must contain all eight section headings,
    including section 8 'Entity State' — this project's enhancement over
    Claude Code's format — and section 6 'All User Messages' (user input is
    sacred). Plus the ``{messages}`` placeholder the parent consumes."""
    for heading in (
        "## 1. Primary Request and Intent",
        "## 2. Key Technical Concepts",
        "## 3. Files and Code Sections",
        "## 4. Errors and Fixes",
        "## 5. Problem Solving",
        "## 6. All User Messages",
        "## 7. Pending Tasks",
        "## 8. Entity State",
    ):
        assert heading in EIGHT_SEGMENT_TEMPLATE, f"missing section: {heading}"
    assert "{messages}" in EIGHT_SEGMENT_TEMPLATE


def test_fallback_prompt_preserves_core_intents():
    """§8.2 Plan-A fallback drops the eight-segment *hard constraint* but must
    still steer the LLM toward the design's never-lose items: goal, errors,
    pending tasks, entity state."""
    lowered = FALLBACK_SUMMARY_PROMPT.lower()
    for keyword in (
        "primary goal",  # §7.1 intent
        "errors",  # §7.4 error history
        "pending tasks",  # §7.7
        "entity state",  # §7.8
    ):
        assert keyword in lowered, f"fallback prompt lost core intent: {keyword}"


def test_l3_context_reconstruction_formula():
    """§9: reconstructed context = [structured summary] + [recent N verbatim].
    A ``before_model`` result must be exactly the REMOVE_ALL sentinel, one
    summary message (flagged), then the preserved recent messages verbatim."""
    model = _StubSummaryModel()
    mw = _make_mw(model, token_threshold=[("messages", 3)], keep_recent=2)
    history = [HumanMessage(content=f"m{i}") for i in range(5)]

    result = mw.before_model({"messages": history}, MagicMock())
    assert result is not None
    msgs = result["messages"]

    assert isinstance(msgs[0], RemoveMessage) and msgs[0].id == REMOVE_ALL_MESSAGES
    summary = msgs[1]
    assert summary.additional_kwargs.get("__summarization__"), "head must be the summary"
    # Recent window preserved verbatim — the exact tail objects of the input.
    assert msgs[2:] == history[-2:]


def test_l3_reduction_meets_design_floor():
    """Design claims L3 compresses 70-95%. Summarized head (20 × 100 chars)
    replaced by a short summary with a 5-message verbatim recent window must
    reduce total volume by at least 70%."""
    model = _StubSummaryModel(summary="SUMMARY")
    mw = _make_mw(model, token_threshold=[("messages", 3)], keep_recent=5)
    history = [
        HumanMessage(content="q" * 100) if i % 2 == 0 else AIMessage(content="a" * 100)
        for i in range(20)
    ]

    result = mw.before_model({"messages": history}, MagicMock())
    assert result is not None
    body = [m for m in result["messages"] if not (isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES)]
    reduction = _reduction(_chars(history), _chars(body))
    assert reduction >= 0.70, f"L3 reduction {reduction:.0%} below design floor 70%"


# --------------------------------------------------------------------------- #
# §4.5 L4 — externalized storage
# --------------------------------------------------------------------------- #


def test_l4_reduction_meets_design_floor(tmp_path):
    """Design claims L4 compresses 90-99%: externalized content leaves only a
    lightweight reference in context. A 10k-char blob must reduce to a
    reference at least 90% smaller."""
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    blob = "R" * 10_000

    ref = ext.externalize(blob)

    assert isinstance(ref, str) and len(ref) > 0
    reduction = _reduction(len(blob), len(ref))
    assert reduction >= 0.90, f"L4 reduction {reduction:.0%} below design floor 90%"


def test_l4_round_trip_recovers_original(tmp_path):
    """§4.5 just-in-time retrieval: ``retrieve(externalize(blob)) == blob``."""
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    blob = "check me verbatim " * 200

    ref = ext.externalize(blob)
    assert ext.retrieve(ref) == blob


# --------------------------------------------------------------------------- #
# §5.2 anti-retrigger — head-is-summary → messages-count dimension only
# --------------------------------------------------------------------------- #


def _summary_head() -> HumanMessage:
    return HumanMessage(content="S", additional_kwargs={"__summarization__": True})


def test_no_retrigger_on_token_dimension_when_head_is_summary():
    """§5.2: a large preserved ToolMessage keeps total tokens over the
    threshold, so the token dimension would re-fire every turn and re-summarize
    the same content. With the head already a summary, it must NOT trigger."""
    model = _StubSummaryModel()
    mw = _make_mw(model, token_threshold=[("tokens", 100)])
    messages = [_summary_head(), HumanMessage(content="m1"), AIMessage(content="m2")]

    # total_tokens 9999 >= 100 → parent would fire; §5.2 must veto it.
    assert mw._should_summarize(messages, 9999) is False


def test_retrigger_by_message_count_when_head_is_summary():
    """§5.2: with the head already a summary, the message-count dimension is
    the one that may still trigger."""
    model = _StubSummaryModel()
    mw = _make_mw(model, token_threshold=[("tokens", 100), ("messages", 3)])
    messages = [_summary_head(), HumanMessage(content="m1"), AIMessage(content="m2")]

    assert mw._should_summarize(messages, 9999) is True  # len == 3 >= 3


def test_head_summary_falls_back_to_50_messages_default():
    """§5.2: "等消息条数累积到阈值（默认 50 条）再压缩" — with only a
    token/fraction trigger configured, the head-is-summary branch falls back
    to the design's default 50-message threshold."""
    model = _StubSummaryModel()
    mw = _make_mw(model, token_threshold=[("tokens", 100)])

    def _msgs(n: int):
        return [_summary_head(), *[HumanMessage(content="m") for _ in range(n - 1)]]

    assert mw._should_summarize(_msgs(49), 9999) is False  # below default 50
    assert mw._should_summarize(_msgs(50), 9999) is True  # reaches default 50


def test_plain_head_unaffected_by_antiretrigger():
    """The §5.2 veto must only apply when the head IS a summary — a normal
    conversation over the token threshold still triggers as before (no
    regression of the base trigger semantics)."""
    model = _StubSummaryModel()
    mw = _make_mw(model, token_threshold=[("tokens", 100)])
    messages = [HumanMessage(content="m0"), HumanMessage(content="m1")]

    assert mw._should_summarize(messages, 9999) is True


def test_custom_builder_without_flag_opts_out_of_antiretrigger():
    """A host ``summary_message_builder`` that does not stamp the
    ``__summarization__`` flag deliberately opts out of the §5.2 policy —
    the middleware must not misclassify a plain message as a summary."""
    model = _StubSummaryModel()
    mw = _make_mw(
        model,
        token_threshold=[("tokens", 100)],
        summary_message_builder=lambda s: HumanMessage(content=f"[summary] {s}"),
    )
    # A message without the flag is never treated as a summary head.
    messages = [HumanMessage(content="plain head"), HumanMessage(content="m1")]
    assert mw._should_summarize(messages, 9999) is True


def test_fraction_dimension_also_vetoed_when_head_is_summary():
    """§5.2 says "跳过 token/fraction 维度触发" — both dimensions. With the
    head a summary, an over-fraction token count must not trigger; only the
    message count (here: the default 50) may."""
    # fraction trigger requires a profile on the *summary* model (parent reads
    # self.model.profile) — FakeMessagesListChatModel inherits the field.
    summary_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="SUMMARY")], profile={"max_input_tokens": 50}
    )
    mw = _make_mw(
        summary_model,
        token_threshold=0.8,  # → ("fraction", 0.8) → threshold = int(50*0.8) = 40
        token_counter=lambda _msgs: 100,  # fixed 100 >= 40 → parent fires
    )
    messages = [_summary_head(), HumanMessage(content="m1")]

    assert mw._should_summarize(messages, 100) is False  # fraction vetoed
    # ...and the default-50 fallback still opens the door when reached:
    long_enough = [_summary_head(), *[HumanMessage(content="m") for _ in range(49)]]
    assert mw._should_summarize(long_enough, 100) is True


# --------------------------------------------------------------------------- #
# §3.3 — lossy but recoverable, end to end through a real agent
# --------------------------------------------------------------------------- #


async def test_lossy_but_recoverable_end_to_end(tmp_path):
    """The design's core promise (§3.3): compression may be lossy, but nothing
    is lost *irrecoverably*. A real agent run exercising the full chain:

    1. L2 externalizes a large tool output at the source (wrap_tool_call).
    2. L3 compacts centrally once the message-count trigger fires.
    3. The host ``post_compress_hook`` aggregates refs into state, where the
       dict-merge reducer accumulates them.
    4. Every accumulated ref retrieves the original content back.
    """
    payload = "PAYLOAD-" + "D" * 3000

    @tool
    def big_tool() -> str:
        """Large tool output that should be externalized at the source (L2)."""
        return payload

    class FakeWithTools(FakeMessagesListChatModel):
        # Fake models don't implement bind_tools; override so create_agent works.
        def bind_tools(self, tools, **kwargs):
            return self

    agent_model = FakeWithTools(
        responses=[
            AIMessage(
                content="calling tool",
                tool_calls=[{"name": "big_tool", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            AIMessage(content="first answer"),
            AIMessage(content="second answer"),
        ]
    )
    # Separate summary model so summarization does not eat the agent's
    # scripted responses.
    summary_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="COMPRESSED HISTORY"), AIMessage(content="COMPRESSED HISTORY")]
    )
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")

    def post_compress(state, result):
        return {**result, "external_refs": aggregate_external_refs(result)}

    cmw = CompressionMiddleware(
        CompressionConfig(
            summary_model=summary_model,
            token_threshold=[("messages", 5)],
            keep_recent=4,
            post_compress_hook=post_compress,
        )
    )
    tmw = ToolCallExternalizerMiddleware(ext)
    # state_schema comes from the middlewares' own state_schema
    # (CompressionAgentState, with the external_refs dict-merge reducer) —
    # create_agent merges them automatically.
    agent = create_agent(
        model=agent_model,
        tools=[big_tool],
        middleware=[cmw, tmw],
        checkpointer=InMemorySaver(),
    )
    thread = {"configurable": {"thread_id": "intent"}}

    # Turn 1: user → tool call → externalized tool result → first answer.
    await agent.ainvoke({"messages": [HumanMessage(content="call big_tool")]}, config=thread)
    state1 = (await agent.aget_state(thread)).values
    tool_msgs = [m for m in state1["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    externalized_ref = tool_msgs[0].additional_kwargs.get("external_ref")
    assert externalized_ref, "L2 should have externalized the large tool output"

    # Turn 2: 5 messages ≥ trigger → L3 compacts centrally.
    await agent.ainvoke({"messages": [HumanMessage(content="continue")]}, config=thread)
    state2 = (await agent.aget_state(thread)).values
    msgs = state2["messages"]
    assert msgs[0].additional_kwargs.get("__summarization__"), "head must be the summary after compaction"

    # Refs accumulated into state via the dict-merge reducer (Hook 3 + §13.1).
    refs = state2.get("external_refs") or {}
    assert externalized_ref in refs, "ref should have been aggregated into state"
    assert refs[externalized_ref] == "big_tool"

    # §3.3: every reference recovers the original content — lossy, not lost.
    assert ext.retrieve(externalized_ref) == payload
