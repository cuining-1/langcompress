"""Scenario test: pure LangChain LCEL composition (no LangGraph), proving the
core abstractions (:class:`langcompress.L0Filter`,
:class:`langcompress.LLMSummarizer`) are decoupled from LangGraph and usable
with only the ``langchain-core`` dependency (design §13.2 — "假想消费者场景:
纯 LangChain LCEL").

This file imports only from ``langchain_core`` and ``langcompress`` core symbols
— no ``langchain`` and no ``langgraph`` import — demonstrating that a host
project can build a compression pipeline with plain LCEL ``RunnableLambda``
composition, without adopting the LangGraph middleware adapter. The final test
statically asserts the core modules' source contains no ``langgraph`` import,
guarding against accidental coupling regressions.
"""
import inspect

from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from langcompress import EIGHT_SEGMENT_TEMPLATE, L0Filter, LLMSummarizer


def _summarize_step(model, *, keep_recent: int = 2):
    """Build an LCEL-compatible step that summarizes everything but the tail.

    Mirrors the middleware's ``summary + preserved-recent`` shape: when there
    are more than ``keep_recent`` messages, the head is summarized into a
    ``SystemMessage`` and the tail is retained verbatim; otherwise the input is
    returned untouched.
    """
    summarizer = LLMSummarizer(model=model, template=EIGHT_SEGMENT_TEMPLATE)

    def _step(msgs):
        if len(msgs) <= keep_recent:
            return list(msgs)
        recent = list(msgs[-keep_recent:])
        summary = summarizer.summarize(list(msgs[:-keep_recent]))
        return [SystemMessage(content=f"Summary: {summary}"), *recent]

    return _step


def test_l0_filter_dedup_and_drop_empty_in_lcel():
    # L0 alone, wrapped as a Runnable: empties dropped, distinct content kept.
    l0 = RunnableLambda(L0Filter().run)
    out = l0.invoke(
        [
            HumanMessage(content=""),
            HumanMessage(content="hi"),
            AIMessage(content=""),
            AIMessage(content="yo"),
        ]
    )
    assert [m.content for m in out] == ["hi", "yo"]


def test_lcel_pipeline_runs_summarization():
    model = FakeMessagesListChatModel(responses=[AIMessage(content="MY SUMMARY")])
    pipeline = RunnableLambda(L0Filter().run) | RunnableLambda(_summarize_step(model))
    # Alternating human/ai so L0's adjacent-merge does not collapse the run.
    initial = [
        msg
        for i in range(5)
        for msg in (HumanMessage(content=f"turn {i}"), AIMessage(content=f"reply {i}"))
    ]

    out = pipeline.invoke(initial)
    contents = [str(m.content) for m in out]

    # 1. A summary message replaced the older history.
    assert any("Summary:" in c and "MY SUMMARY" in c for c in contents), contents
    # 2. The most recent turn survives.
    assert any("turn 4" in c for c in contents), contents
    # 3. Early turns were compressed away.
    assert not any("turn 0" in c for c in contents), contents
    # 4. The pipeline output stayed small.
    assert len(out) <= 4, f"expected a small output, got {len(out)}: {contents}"


def test_core_modules_do_not_import_langgraph():
    # Static decoupling guard: the core compression modules must not pull in
    # langgraph, so a langchain-core-only host can use them.
    from langcompress.pipeline import base as pipeline_base
    from langcompress.pipeline import l0_filter
    from langcompress.summarizer import base as summarizer_base
    from langcompress.summarizer import llm_summarizer

    for mod in (pipeline_base, l0_filter, summarizer_base, llm_summarizer):
        src = inspect.getsource(mod)
        assert "import langgraph" not in src, f"{mod.__name__} must not import langgraph"
        assert "from langgraph" not in src, f"{mod.__name__} must not import from langgraph"
