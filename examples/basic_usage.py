"""Minimal runnable langcompress example — no API key required.

Wires :class:`langcompress.CompressionMiddleware` into a LangGraph agent built
with ``langchain.agents.create_agent`` and drives six user turns through it.
With ``trigger=("messages", 5)`` and ``keep_recent=2``, summarization fires
mid-session: the early turns are replaced by a single summary message while
the most recent turns survive untouched.

Run:
    pip install langcompress[middleware]   # also installs langchain
    python examples/basic_usage.py
"""
from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from langcompress import CompressionConfig, CompressionMiddleware


def _is_summary(msg) -> bool:
    return bool(msg.additional_kwargs.get("__summarization__"))


async def main() -> None:
    # A fake "agent" model: each invoke pops the next canned reply. No API key.
    agent_model = FakeMessagesListChatModel(
        responses=[AIMessage(content=f"reply {i}") for i in range(50)]
    )
    # A fake "summary" model: any summarization call returns this canned text.
    summary_model = FakeMessagesListChatModel(
        responses=[AIMessage(content="<eight-segment summary of turns 0..3>")]
    )

    # --- Hook 1 (optional): build the summary message yourself ---------------
    # Here we render the summary as a SystemMessage so the host frontend can
    # treat it distinctly. We also keep the ``__summarization__`` flag so the
    # rest of the stack can still recognise it. Omit this hook entirely to get
    # the default HumanMessage (already carries the flag).
    def build_summary_message(summary: str):
        return SystemMessage(
            content=f"[context summary]\n{summary}",
            additional_kwargs={"lc_source": "summarization", "__summarization__": True},
        )

    cfg = CompressionConfig(
        summary_model=summary_model,
        # Compress once the conversation reaches 5 messages, keep the latest 2.
        token_threshold=[("messages", 5)],
        keep_recent=2,
        summary_message_builder=build_summary_message,
    )
    middleware = [CompressionMiddleware(cfg)]

    agent = create_agent(
        model=agent_model,
        tools=[],
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )

    thread = {"configurable": {"thread_id": "demo"}}
    print("Sending 6 user turns (trigger=5 messages, keep_recent=2)...\n")
    for i in range(6):
        await agent.ainvoke(
            {"messages": [HumanMessage(content=f"hi {i}")]}, config=thread
        )

    state = await agent.aget_state(thread)
    msgs = state.values.get("messages", [])
    print(f"Final state: {len(msgs)} message(s) after compaction:\n")
    for m in msgs:
        kind = m.__class__.__name__
        tag = "  <<SUMMARY>>" if _is_summary(m) else ""
        body = str(m.content).replace("\n", " ")[:70]
        print(f"  {kind}{tag}: {body}")

    # Proof compaction ran: a summary replaced the early turns, recent turn kept.
    assert any(_is_summary(m) for m in msgs), "no summary message found"
    contents = [str(m.content) for m in msgs]
    assert any("hi 5" in c for c in contents), "recent turn should survive"
    assert not any("hi 0" in c for c in contents), "early turn should be compressed"
    print("\n✓ Compaction verified: early turns compressed, recent turn retained.")


if __name__ == "__main__":
    asyncio.run(main())
