"""Scenario test: ``CompressionMiddleware`` compatibility with multiple LLM
shapes (design §13.3 — "假想消费者场景: 多种 LLM 形状"), the third of the M2
"three-scenario" suite.

Parametrised over agent-model shapes a host project might plug in:

* ``FakeMessagesListChatModel`` with a ``("messages", N)`` trigger (no profile).
* The same model carrying a ``profile`` so the parent's ``("fraction", v)``
  trigger resolves a token threshold (``int(max_input_tokens * v)``).
* A subclass overriding ``bind_tools`` (Fake models otherwise raise
  ``NotImplementedError``) so ``create_agent(tools=[...])`` can bind a tool —
  the canned first response emits a tool_call, exercising the parent's
  AI/Tool-pair-safe cutoff.
* ``GenericFakeChatModel`` — a different model implementation (iterator-based),
  proving the middleware does not favour one model class.

A real-provider case (skipped without ``OPENAI_API_KEY``) notes coverage of the
parent's ``_should_summarize_based_on_reported_tokens`` branch (needs
``usage_metadata`` + matching ``model_provider``), a different path from the
Fake-model token-counter estimate.
"""
import os

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langcompress import CompressionConfig, CompressionMiddleware


def _has_summary(messages):
    return any(m.additional_kwargs.get("__summarization__") for m in messages)


def _responses(text: str, n: int = 30) -> list[AIMessage]:
    return [AIMessage(content=f"{text} {i}") for i in range(n)]


class _FakeWithTools(FakeMessagesListChatModel):
    # Fake models inherit the base ``bind_tools`` which raises NotImplementedError;
    # override so ``create_agent(tools=[...])`` can bind tools without a real model.
    def bind_tools(self, tools, **kwargs):
        return self


def _counter_returning(value: int):
    # Deterministic counter: forces ``total_tokens`` above a fraction threshold so
    # the fraction math (int(max_input_tokens * fraction)) is exercised without
    # depending on the approximate char/token ratio.
    def _c(messages):
        return value

    return _c


@tool
def stub_tool() -> str:
    """A stub tool returning a small string; proves tools bind and run."""
    return "stub-ack"


async def _run_session(
    *,
    agent_model,
    summary_model,
    trigger,
    keep_recent: int,
    tools: list,
    token_counter,
    num_turns: int,
) -> list:
    cfg = CompressionConfig(
        summary_model=summary_model,
        token_threshold=trigger,
        keep_recent=keep_recent,
        token_counter=token_counter,
    )
    agent = create_agent(
        model=agent_model,
        tools=tools,
        middleware=[CompressionMiddleware(cfg)],
        checkpointer=InMemorySaver(),
    )
    thread = {"configurable": {"thread_id": "t1"}}
    for i in range(num_turns):
        await agent.ainvoke(
            {"messages": [HumanMessage(content=f"turn {i}")]}, config=thread
        )
    state = await agent.aget_state(thread)
    return state.values.get("messages", [])


def _case_fake_messages_trigger():
    return {
        "agent_model": FakeMessagesListChatModel(responses=_responses("reply")),
        "summary_model": FakeMessagesListChatModel(responses=[AIMessage(content="SUMMARY")]),
        "trigger": ("messages", 4),
        "keep_recent": 2,
        "tools": [],
        "token_counter": None,
        "num_turns": 5,
    }


def _case_fake_fraction_trigger():
    # The profile lives on the *summary* model — the parent reads it via
    # ``_get_profile_limits()`` (``self.model.profile``) at construction and at
    # runtime. A deterministic counter forces total_tokens=100 >= int(50*0.8)=40.
    return {
        "agent_model": FakeMessagesListChatModel(responses=_responses("reply")),
        "summary_model": FakeMessagesListChatModel(
            profile={"max_input_tokens": 50},
            responses=[AIMessage(content="SUMMARY")],
        ),
        "trigger": ("fraction", 0.8),
        "keep_recent": 2,
        "tools": [],
        "token_counter": _counter_returning(100),
        "num_turns": 3,
    }


def _case_fake_with_tools():
    # First canned response emits a tool_call so the agent runs the ToolNode and
    # the parent's AI/Tool-pair-safe cutoff is exercised (not just bind_tools).
    responses = [
        AIMessage(
            content="calling stub",
            tool_calls=[
                {"name": "stub_tool", "args": {}, "id": "c1", "type": "tool_call"}
            ],
        ),
        *_responses("reply"),
    ]
    return {
        "agent_model": _FakeWithTools(
            profile={"max_input_tokens": 100}, responses=responses
        ),
        "summary_model": FakeMessagesListChatModel(responses=[AIMessage(content="SUMMARY")]),
        "trigger": ("messages", 4),
        "keep_recent": 2,
        "tools": [stub_tool],
        "token_counter": None,
        "num_turns": 5,
    }


def _case_generic_fake_model():
    return {
        "agent_model": GenericFakeChatModel(messages=iter(_responses("reply"))),
        "summary_model": FakeMessagesListChatModel(responses=[AIMessage(content="SUMMARY")]),
        "trigger": ("messages", 4),
        "keep_recent": 2,
        "tools": [],
        "token_counter": None,
        "num_turns": 5,
    }


_CASES = [
    pytest.param(_case_fake_messages_trigger, id="fake-messages-trigger"),
    pytest.param(_case_fake_fraction_trigger, id="fake-fraction-trigger"),
    pytest.param(_case_fake_with_tools, id="fake-with-tools"),
    pytest.param(_case_generic_fake_model, id="generic-fake-model"),
]


@pytest.mark.parametrize("build", _CASES)
async def test_compression_runs_with_various_llm_shapes(build):
    case = build()
    msgs = await _run_session(**case)
    assert _has_summary(msgs), (
        f"expected summarization to fire for {build.__name__}, "
        f"final state has {len(msgs)} messages"
    )
    # State stayed small after compression.
    assert len(msgs) <= 6, f"state should stay small, got {len(msgs)} for {build.__name__}"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_real_provider_reported_tokens_branch():
    # Real providers populate ``usage_metadata`` + ``model_provider``, exercising
    # the parent's ``_should_summarize_based_on_reported_tokens`` branch — a
    # different path from the Fake-model token-counter estimate. Skipped in CI.
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        pytest.skip("langchain-openai not installed")
    agent_model = ChatOpenAI(model="gpt-4o-mini")
    summary_model = ChatOpenAI(model="gpt-4o-mini")
    msgs = await _run_session(
        agent_model=agent_model,
        summary_model=summary_model,
        trigger=("tokens", 1),
        keep_recent=2,
        tools=[],
        token_counter=None,
        num_turns=2,
    )
    assert _has_summary(msgs)
