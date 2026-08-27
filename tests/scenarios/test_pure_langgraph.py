"""Scenario test: a pure-LangGraph agent (no CopilotKit) wired with
:class:`langcompress.CompressionMiddleware` end-to-end (design §13.1).

Uses ``FakeMessagesListChatModel`` so no API key is needed. Sends six user
turns through an agent with ``trigger=("messages", 5)`` and ``keep_recent=2``,
then asserts summarization fired, the latest turn survived, early turns were
compressed away, and the state stayed small — proving the
``REMOVE_ALL_MESSAGES`` sentinel + ``add_messages`` reducer + summary-chaining
pipeline works through ``create_agent``.
"""
from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langcompress import CompressionConfig, CompressionMiddleware


def _has_summary(messages):
    return any(m.additional_kwargs.get("__summarization__") for m in messages)


async def test_compression_runs_in_pure_langgraph_agent():
    # One AI reply per turn → messages grow by 2/turn. trigger=(messages,5) +
    # keep_recent=2 → summarization fires mid-session and stays small.
    agent_model = FakeMessagesListChatModel(
        responses=[AIMessage(content=f"reply {i}") for i in range(50)]
    )
    summary_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="COMPRESSED SUMMARY")]
    )
    cfg = CompressionConfig(
        summary_model=summary_model,
        token_threshold=[("messages", 5)],
        keep_recent=2,
    )
    agent = create_agent(
        model=agent_model,
        tools=[],
        middleware=[CompressionMiddleware(cfg)],
        checkpointer=InMemorySaver(),
    )

    thread = {"configurable": {"thread_id": "t1"}}
    for i in range(6):
        await agent.ainvoke(
            {"messages": [HumanMessage(content=f"hi {i}")]}, config=thread
        )

    state = await agent.aget_state(thread)
    msgs = state.values.get("messages", [])
    contents = [str(m.content) for m in msgs]

    # 1. A summarization message replaced the older history.
    assert _has_summary(msgs), "expected a __summarization__ summary in final state"
    # 2. The most recent user turn survives.
    assert any("hi 5" in c for c in contents), "recent hi 5 should be retained"
    # 3. Early turns were compressed away.
    assert not any("hi 0" in c for c in contents), "hi 0 should have been compressed away"
    assert not any("hi 1" in c for c in contents), "hi 1 should have been compressed away"
    # 4. The state stayed small — compression is actually trimming.
    assert len(msgs) <= 4, f"state should be small after compression, got {len(msgs)}"
