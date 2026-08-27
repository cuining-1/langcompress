"""Conversation replayer — drives a real LangGraph agent through a scenario.

The agent under replay is *real* (``create_agent`` + middleware + checkpointer
+ tool node); only its two model roles are bench-controlled:

- the **agent model** is a scripted fake returning the scenario's assistant
  entries in order (deterministic, keyless — the transcript is the script);
- the **summary model** is the model under test (real via ``init_chat_model``,
  or a stub in keyless mode), metered by :class:`CountingChatModel`.

Four comparison arms share one replay loop and one telemetry shape:

``langcompress``       the full package — eight-segment template + quality
                       validation + A/B/D/C degradation chain + L0 filter.
``bare_summarization`` the parent ``SummarizationMiddleware`` with its default
                       prompt, no validation, no degradation — isolates the
                       package's *incremental* value.
``trim``               the "dumbest reference": keep the last ``keep_recent``
                       messages, drop the rest, no summary at all.
``full_context``       no middleware — the fidelity ceiling / cost floor.

Baseline arms record through the same :class:`CompressionRecorder` the
langcompress arm uses (bench-side middleware subclasses call ``record``
directly), so every downstream metric is arm-symmetric.
"""
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, ConfigDict

from benchmarks.config import BenchSettings
from benchmarks.dump import DumpWriter
from benchmarks.llm import CountingChatModel, estimate_tokens_messages
from benchmarks.scenario import Scenario
from benchmarks.telemetry import CompressionRecorder
from langcompress import CompressionConfig
from langcompress.summarizer.templates import DEFAULT_SUMMARY_PROMPT

ARMS = ("langcompress", "bare_summarization", "trim", "full_context")


# --------------------------------------------------------------------------- #
# Scripted agent plumbing
# --------------------------------------------------------------------------- #


class _ScriptedAgentModel(FakeMessagesListChatModel):
    """Fake chat model returning the transcript's assistant entries in order.

    ``bind_tools`` is a passthrough — ``create_agent`` requires the call, but
    the script is fixed so there is nothing to bind.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedAgentModel:
        return self


class _AnyArgs(BaseModel):
    """Permissive tool args schema — the bench scripts tool results, so any
    argument shape the transcript scripted must validate."""

    model_config = ConfigDict(extra="allow")


def build_agent_responses(scenario: Scenario) -> list[AIMessage]:
    """Materialize the scripted model responses from the transcript."""
    responses: list[AIMessage] = []
    for i, entry in scenario.assistant_entries():
        tool_calls = [
            {**call, "type": "tool_call"} for call in (entry.tool_calls or [])
        ]
        # langchain 1.x AIMessage rejects tool_calls=None outright — the
        # field must simply be absent for plain-text responses.
        kwargs: dict[str, Any] = {}
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        responses.append(
            AIMessage(
                content=entry.content if entry.content is not None else "",
                id=f"t{i}",
                additional_kwargs=entry.additional_kwargs or {},
                **kwargs,
            )
        )
    return responses


def build_scripted_tools(scenario: Scenario) -> list[StructuredTool]:
    """One tool per declared name, popping scripted results FIFO.

    Tool results are keyed per tool name in transcript order — the fake model
    emits tool calls in exactly that order, so FIFO per tool is exact.
    """
    queues: dict[str, deque[str]] = {spec.name: deque() for spec in scenario.tools}
    for _index, entry in scenario.tool_entries():
        name = entry.name or ""
        content = entry.content if isinstance(entry.content, str) else str(entry.content or "")
        if name in queues:
            queues[name].append(content)

    def _make(queue: deque[str]) -> Callable[..., str]:
        def _run(*args: Any, **kwargs: Any) -> str:
            return queue.popleft() if queue else "{}"

        return _run

    return [
        StructuredTool(
            name=spec.name,
            description=spec.description or f"Scripted tool {spec.name}.",
            func=_make(queues[spec.name]),
            args_schema=_AnyArgs,
        )
        for spec in scenario.tools
    ]


# --------------------------------------------------------------------------- #
# Baseline arms (bench-side middleware — the package itself is untouched)
# --------------------------------------------------------------------------- #


class _RecordingSummarizationMiddleware(SummarizationMiddleware):
    """Bare parent middleware + telemetry: isolates langcompress's increment.

    Subclassing here is bench-side consumer code (the same extension the
    package itself used); the recorder sees identically-shaped events, except
    summaries use the parent's default prompt and no quality gate fires.
    """

    def __init__(self, *, recorder: CompressionRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        result = super().before_model(state, runtime)
        if result:
            self._recorder.record(dict(state), result)
        return result

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        result = await super().abefore_model(state, runtime)
        if result:
            self._recorder.record(dict(state), result)
        return result


class _TrimMessagesMiddleware(AgentMiddleware):
    """The dumbest viable reference: truncate to the last N messages.

    No summary, no LLM call — whatever falls out of the window is gone. The
    only sophistication (matching the window semantics the compression arms
    use) is the AI/Tool pair guard: a leading orphaned ToolMessage whose
    AIMessage fell out of the window is dropped too, because an orphaned
    tool result is an invalid message list for most providers.
    """

    def __init__(self, *, keep_recent: int, recorder: CompressionRecorder) -> None:
        self._keep_recent = max(keep_recent, 0)
        self._recorder = recorder

    def _trim(self, state: dict[str, Any]) -> dict[str, Any] | None:
        messages: list[BaseMessage] = list(state.get("messages", []))
        if len(messages) <= self._keep_recent:
            return None
        tail = list(messages[-self._keep_recent :]) if self._keep_recent else []
        while tail and getattr(tail[0], "tool_call_id", None):
            tail = tail[1:]  # drop orphaned tool results at the cut
        if len(tail) == len(messages):
            return None
        result: dict[str, Any] = {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *tail]}
        self._recorder.record(state, result)
        return result

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._trim(dict(state))

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._trim(dict(state))


# --------------------------------------------------------------------------- #
# Replay outcome
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    """Everything one (scenario × arm) replay produced."""

    arm: str
    scenario_id: str
    turns: int = 0
    events: list[Any] = field(default_factory=list)  # list[CompressEvent]
    final_messages: list[BaseMessage] = field(default_factory=list)
    summarized_keys: set[str] = field(default_factory=set)
    final_summary: str | None = None
    agent_stats: dict[str, float | int] = field(default_factory=dict)
    summary_stats: dict[str, float | int] = field(default_factory=dict)
    transcript_tokens: int = 0
    final_state_tokens: int = 0
    wall_seconds: float = 0.0
    error: str | None = None
    dump_path: Path | None = None  # full-text before/after dump, for review

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


class ReplayHarness:
    """Builds and replays one agent per (scenario, arm); collects telemetry.

    ``summary_model_factory`` is called once per arm-run and must return a
    fresh model instance (stub or real) — per-run isolation keeps stub scripts
    aligned with expected call counts and real runs independent.
    """

    def __init__(
        self,
        settings: BenchSettings,
        *,
        summary_model_factory: Callable[[], Any],
        run_tag: str = "bench",
        config_overrides: dict[str, Any] | None = None,
        dump_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.summary_model_factory = summary_model_factory
        self.run_tag = run_tag
        # Extra CompressionConfig kwargs for the langcompress arm — the
        # fault-injection suite uses this to swap the degradation
        # externalizer; the package API itself is untouched.
        self.config_overrides = dict(config_overrides or {})
        # When set, every (scenario × arm) replay also writes a full-text
        # before/after dump file under this directory for human/AI review.
        self.dump_dir = dump_dir

    # -- middleware construction per arm ---------------------------------- #

    def _build_middleware(
        self, arm: str, summary_model: Any, recorder: CompressionRecorder
    ) -> list[Any]:
        s = self.settings
        if arm == "langcompress":
            from langcompress import CompressionMiddleware

            cfg = CompressionConfig(
                summary_model=summary_model,
                token_threshold=[("messages", s.trigger_messages)],
                keep_recent=s.keep_recent,
                trim_tokens_to_summarize=s.trim_tokens_to_summarize,
                l0_enabled=s.l0_enabled,
                post_compress_hook=recorder.hook,
                # Hook-2 usage: tag every summary LLM call with the run id so
                # provider-side traces map back to this benchmark run.
                summary_llm_config_provider=lambda: {"metadata": {"bench_run": self.run_tag}},
                **self.config_overrides,
            )
            return [CompressionMiddleware(cfg)]
        if arm == "bare_summarization":
            return [
                _RecordingSummarizationMiddleware(
                    recorder=recorder,
                    model=summary_model,
                    trigger=[("messages", s.trigger_messages)],
                    keep=("messages", s.keep_recent),
                    token_counter=count_tokens_approximately,
                    summary_prompt=DEFAULT_SUMMARY_PROMPT,
                    trim_tokens_to_summarize=s.trim_tokens_to_summarize,
                )
            ]
        if arm == "trim":
            return [_TrimMessagesMiddleware(keep_recent=s.keep_recent, recorder=recorder)]
        if arm == "full_context":
            return []
        raise ValueError(f"unknown arm {arm!r} (expected one of {ARMS})")

    # -- the replay loop --------------------------------------------------- #

    async def run(self, scenario: Scenario, arm: str) -> ArmResult:
        result = ArmResult(arm=arm, scenario_id=scenario.id)
        result.transcript_tokens = self._transcript_tokens(scenario)
        start = time.perf_counter()
        try:
            await self._replay(scenario, arm, result)
        except Exception as exc:  # noqa: BLE001  # robustness: record, never crash the bench
            result.error = f"{type(exc).__name__}: {exc}"
        result.wall_seconds = time.perf_counter() - start
        return result

    def _transcript_tokens(self, scenario: Scenario) -> int:
        total = 0
        for entry in scenario.transcript:
            content = entry.content
            if isinstance(content, str):
                total += max(len(content) // 4, 1)
            elif isinstance(content, list):
                total += max(len(json.dumps(content)) // 4, 1)
        return total

    async def _replay(self, scenario: Scenario, arm: str, result: ArmResult) -> None:
        # Fresh per-run objects: recorder, models, agent, thread.
        recorder = CompressionRecorder(
            counter=estimate_tokens_messages,
            l0_enabled=self.settings.l0_enabled and arm == "langcompress",
        )
        dump_writer: DumpWriter | None = None
        if self.dump_dir is not None:
            dump_writer = DumpWriter(
                self.dump_dir / f"{scenario.id}__{arm}.md",
                scenario_id=scenario.id,
                arm=arm,
            )
            # Self-contained review material: original conversation and the
            # ground-truth checklist come first, so a third-party judge can
            # score the summary without opening any other file.
            dump_writer.write_original(scenario)
            dump_writer.write_ground_truth(scenario)
            recorder.event_sink = dump_writer.write_event
            result.dump_path = dump_writer.path
        summary_wrapper: CountingChatModel | None = None
        if arm in ("langcompress", "bare_summarization"):
            summary_wrapper = CountingChatModel(
                inner=self.summary_model_factory(), role=f"summary:{arm}"
            )
            recorder.meter = summary_wrapper

        agent_wrapper = CountingChatModel(
            inner=_ScriptedAgentModel(responses=build_agent_responses(scenario)),
            role="agent",
        )
        middleware = self._build_middleware(
            arm, summary_wrapper if summary_wrapper else None, recorder
        )
        agent = create_agent(
            model=agent_wrapper,
            tools=build_scripted_tools(scenario),
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )

        thread = {"configurable": {"thread_id": f"{self.run_tag}-{scenario.id}-{arm}"}}
        try:
            for _i, entry in scenario.user_entries():
                recorder.turn += 1
                await agent.ainvoke(
                    {"messages": [HumanMessage(content=entry.content or "", id=f"t{_i}")]},
                    config=thread,
                )
                result.turns += 1

            state = await agent.aget_state(thread)
            final_messages = list(state.values.get("messages", []))
            result.final_messages = final_messages
            result.final_state_tokens = estimate_tokens_messages(final_messages)
            result.events = list(recorder.events)
            result.summarized_keys = recorder.summarized_keys_all()
            result.final_summary = recorder.final_summary
            result.agent_stats = agent_wrapper.snapshot().as_dict()
            if summary_wrapper is not None:
                result.summary_stats = summary_wrapper.snapshot().as_dict()
            if dump_writer is not None:
                dump_writer.write_final(final_messages)
        finally:
            # Close the dump even on failure — a partial before/after record
            # of a crashed replay is exactly what a reviewer wants to see.
            if dump_writer is not None:
                dump_writer.close()


__all__ = [
    "ARMS",
    "ArmResult",
    "ReplayHarness",
    "build_agent_responses",
    "build_scripted_tools",
]
