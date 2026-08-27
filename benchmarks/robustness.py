"""Fault-injection suite — the robustness dimension.

The functional tests in ``tests/`` prove the degradation chain works when
each link is exercised in isolation with synthetic states; this suite
proves it works *end to end inside a live agent replay*, which is a
different claim: the middleware must recover while the agent keeps
running, the checkpointer must survive the patch, and nothing from the
failure may leak into the conversation.

Five cases, each a full ``create_agent`` replay with one fault planted:

=============================  ==============================================
case                           what it demonstrates
=============================  ==============================================
``summary_llm_flaky_once``     **Plan A**: first summary attempt raises,
                               fallback-prompt retry succeeds → a *clean*
                               L3 event (no degradation stamp), while the
                               planted model's internal counter shows the
                               retry actually happened.
``summary_llm_always_raises``  **Plan B/C**: every summary attempt raises,
                               default config (no externalizer) → the
                               widen-window / truncate patches carry the
                               conversation onward.
``summary_llm_raises_plan_d``  **Plan D**: same failure, but a
                               ``FilesystemExternalizer`` is injected via
                               ``config_overrides`` → the head is
                               externalized, ``external_ref`` recorded.
``short_conversation``         far below the trigger threshold → zero
                               compression events, message list untouched
                               (no eager compression).
``oversized_single_message``   one 30k-char tool result inside a long
                               conversation → compression still succeeds
                               (``trim_tokens_to_summarize`` bounds the
                               summarizer input), invariants hold.
=============================  ==============================================

Invariants asserted after *every* case (the "error strings never leak"
guarantee, design §8.3):

1. the final message list is non-empty;
2. no orphaned ``ToolMessage`` (every ``tool_call_id`` has its issuing
   AIMessage — a broken patch would orphan tool results immediately);
3. none of the injected failure strings (exception text, "Traceback")
   appears anywhere in the final conversation.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from benchmarks.config import BenchSettings
from benchmarks.llm import STUB_SUMMARY, FlakyModel, RaisingModel, make_stub_summary_model
from benchmarks.probes import message_text, normalize
from benchmarks.replayer import ArmResult, ReplayHarness
from benchmarks.scenario import Scenario, ToolSpec, TranscriptEntry

_LEAK_PROBES = ("traceback", "runtimeerror")


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def check_invariants(
    final_messages: list[BaseMessage], forbidden_texts: list[str]
) -> list[str]:
    """Structural + leakage invariants over a final message list.

    Returns human-readable problems (empty list = all invariants hold)."""
    problems: list[str] = []
    if not final_messages:
        return ["final message list is empty"]

    call_ids: set[str] = set()
    for m in final_messages:
        for call in getattr(m, "tool_calls", None) or []:
            cid = call.get("id") if isinstance(call, dict) else None
            if cid:
                call_ids.add(cid)
    for m in final_messages:
        tcid = getattr(m, "tool_call_id", None)
        if tcid and tcid not in call_ids:
            problems.append(f"orphaned tool result (tool_call_id={tcid})")

    blob = normalize("\n".join(message_text(m) for m in final_messages))
    for text in forbidden_texts:
        if text and normalize(text) in blob:
            problems.append(f"failure string leaked into conversation: {text!r}")
    for probe in _LEAK_PROBES:
        if probe in blob:
            problems.append(f"generic failure marker leaked: {probe!r}")
    return problems


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #


@dataclass
class FaultOutcome:
    """One fault case's result."""

    case: str
    description: str
    ok: bool = False
    plans: dict[str, int] = field(default_factory=dict)  # plan letter → count
    summary_calls: int = 0  # total summary-LLM attempts (failures included)
    summary_errors: int = 0  # attempts that raised
    final_messages: int = 0
    problems: list[str] = field(default_factory=list)  # invariant violations
    error: str | None = None  # replay-level crash (middleware re-raised)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "description": self.description,
            "ok": self.ok,
            "plans": self.plans,
            "summary_calls": self.summary_calls,
            "summary_errors": self.summary_errors,
            "final_messages": self.final_messages,
            "problems": self.problems,
            "error": self.error,
        }


@dataclass
class RobustnessReport:
    """Aggregated fault-suite result."""

    outcomes: list[FaultOutcome] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.outcomes) and all(o.ok for o in self.outcomes)

    @property
    def plan_distribution(self) -> dict[str, int]:
        """Plan letters observed across all cases (A excluded — Plan A is
        internal to ``_create_summary`` and evidenced per-case by the retry
        counter, not by an event stamp)."""
        dist: dict[str, int] = {}
        for o in self.outcomes:
            for plan, count in o.plans.items():
                dist[plan] = dist.get(plan, 0) + count
        return dist

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_ok": self.all_ok,
            "plan_distribution": self.plan_distribution,
            "cases": [o.as_dict() for o in self.outcomes],
        }


# --------------------------------------------------------------------------- #
# Inline mini-scenarios (fault suite-local, not part of the scored corpus)
# --------------------------------------------------------------------------- #


def _short_scenario() -> Scenario:
    """8 messages — far below the default 24-message trigger."""
    entries = [
        TranscriptEntry(role="user", content="Check the deployment status."),
        TranscriptEntry(role="assistant", content="Everything is green: all services report healthy."),
        TranscriptEntry(role="user", content="Good. What about the queue depth?"),
        TranscriptEntry(role="assistant", content="Queue depth is within normal bounds."),
        TranscriptEntry(role="user", content="Fine, then we are done for today."),
        TranscriptEntry(role="assistant", content="Agreed — nothing outstanding on my side."),
        TranscriptEntry(role="user", content="Thanks!"),
        TranscriptEntry(role="assistant", content="Anytime. Have a good one!"),
    ]
    return Scenario(id="rb_short_conversation", category="robustness", transcript=entries)


def _oversized_scenario() -> Scenario:
    """26 messages where one tool result is ~30k chars.

    The oversized payload must flow through ``trim_tokens_to_summarize``
    truncation without breaking compression or the replay loop."""
    big = "order item " + ("x" * 30000)  # single oversized tool payload
    entries: list[TranscriptEntry] = []
    for i in range(12):
        entries.append(TranscriptEntry(role="user", content=f"Step {i + 1}: verify batch."))
        if i == 5:
            entries.append(
                TranscriptEntry(
                    role="assistant",
                    content="Fetching the full batch report now.",
                    tool_calls=[{"name": "read_report", "args": {"batch": i}, "id": f"call_{i}"}],
                )
            )
            entries.append(
                TranscriptEntry(role="tool", name="read_report", content=big, tool_call_id=f"call_{i}")
            )
            entries.append(TranscriptEntry(role="assistant", content="Report fetched; batch looks consistent."))
        else:
            entries.append(TranscriptEntry(role="assistant", content=f"Step {i + 1} verified; moving on."))
    return Scenario(
        id="rb_oversized_message",
        category="robustness",
        tools=[ToolSpec(name="read_report", description="Reads a batch report.")],
        transcript=entries,
    )


def _plans_from(result: ArmResult) -> dict[str, int]:
    plans: dict[str, int] = {}
    for event in result.events:
        if event.plan:
            plans[event.plan] = plans.get(event.plan, 0) + 1
    return plans


def _stats(result: ArmResult) -> tuple[int, int]:
    calls = int(result.summary_stats.get("calls", 0))
    errors = int(result.summary_stats.get("errors", 0))
    return calls, errors


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #


async def run_fault_suite(settings: BenchSettings, scenario: Scenario) -> RobustnessReport:
    """Run all five fault cases against the langcompress arm.

    ``scenario`` is the corpus scenario used by the LLM-failure cases (it
    must exceed the trigger threshold so compression actually fires);
    the short/oversized cases build their own inline scenarios."""
    report = RobustnessReport()

    # -- Case 1: Plan A (flaky once, fallback prompt succeeds) ------------ #
    flaky = FlakyModel(STUB_SUMMARY, fail_calls=1)
    result = await ReplayHarness(
        settings, summary_model_factory=lambda: flaky, run_tag="fault-plan-a"
    ).run(scenario, "langcompress")
    plans = _plans_from(result)
    clean_l3 = any(e.kind == "l3" and e.plan is None for e in result.events)
    problems = check_invariants(result.final_messages, [flaky.message])
    if result.error:
        problems.append(f"replay crashed: {result.error}")
    if not clean_l3:
        problems.append("no clean L3 event — Plan A retry did not produce a summary")
    if flaky.calls < 2:
        problems.append(f"retry not observed (model calls={flaky.calls}, expected >= 2)")
    if plans:
        problems.append(f"unexpected degradation plans during Plan-A recovery: {plans}")
    calls, errors = _stats(result)
    report.outcomes.append(
        FaultOutcome(
            case="summary_llm_flaky_once",
            description="First summary attempt raises; fallback-prompt retry must succeed without degradation.",
            ok=not problems,
            plans=plans,
            summary_calls=calls,
            summary_errors=errors,
            final_messages=len(result.final_messages),
            problems=problems,
            error=result.error,
        )
    )

    # -- Case 2: Plans B/C (always raises, no externalizer) ---------------- #
    raising = RaisingModel()
    result = await ReplayHarness(
        settings, summary_model_factory=lambda: RaisingModel(raising.message), run_tag="fault-bc"
    ).run(scenario, "langcompress")
    plans = _plans_from(result)
    problems = check_invariants(result.final_messages, [raising.message])
    if result.error:
        problems.append(f"replay crashed: {result.error}")
    degraded = [e for e in result.events if e.kind == "degraded"]
    if not degraded:
        problems.append("no degraded event — every summary failure must produce a patch")
    bad_plans = {p for p in plans if p not in ("B", "C")}
    if bad_plans:
        problems.append(f"unexpected plans without externalizer: {sorted(bad_plans)}")
    calls, errors = _stats(result)
    report.outcomes.append(
        FaultOutcome(
            case="summary_llm_always_raises",
            description="Every summary attempt raises; widen-window/truncate patches (B/C) must keep the agent alive.",
            ok=not problems,
            plans=plans,
            summary_calls=calls,
            summary_errors=errors,
            final_messages=len(result.final_messages),
            problems=problems,
            error=result.error,
        )
    )

    # -- Case 3: Plan D (always raises, externalizer injected) ------------- #
    with tempfile.TemporaryDirectory(prefix="langcompress_bench_d_") as tmp:
        from langcompress.externalizer import FilesystemExternalizer

        raising_d = RaisingModel()
        result = await ReplayHarness(
            settings,
            summary_model_factory=lambda: RaisingModel(raising_d.message),
            run_tag="fault-d",
            config_overrides={"degradation_externalizer": FilesystemExternalizer(tmp)},
        ).run(scenario, "langcompress")
        plans = _plans_from(result)
        problems = check_invariants(result.final_messages, [raising_d.message])
        if result.error:
            problems.append(f"replay crashed: {result.error}")
        plan_d_events = [e for e in result.events if e.plan == "D"]
        if not plan_d_events:
            problems.append("no Plan-D event despite injected externalizer")
        elif not any(e.external_ref for e in plan_d_events):
            problems.append("Plan-D events carry no external_ref")
        calls, errors = _stats(result)
        report.outcomes.append(
            FaultOutcome(
                case="summary_llm_raises_plan_d",
                description="Every summary attempt raises with a filesystem externalizer; head must be externalized (D).",
                ok=not problems,
                plans=plans,
                summary_calls=calls,
                summary_errors=errors,
                final_messages=len(result.final_messages),
                problems=problems,
                error=result.error,
            )
        )

    # -- Case 4: short conversation (below trigger) ------------------------ #
    result = await ReplayHarness(
        settings, summary_model_factory=make_stub_summary_model, run_tag="fault-short"
    ).run(_short_scenario(), "langcompress")
    problems = check_invariants(result.final_messages, [])
    if result.error:
        problems.append(f"replay crashed: {result.error}")
    if result.events:
        problems.append(f"compression fired below threshold: {len(result.events)} events")
    if len(result.final_messages) != len(_short_scenario().transcript):
        problems.append(
            f"message count changed without compression: {len(result.final_messages)} != 8"
        )
    calls, errors = _stats(result)
    report.outcomes.append(
        FaultOutcome(
            case="short_conversation",
            description="8-message conversation must stay untouched below the 24-message trigger.",
            ok=not problems,
            plans=_plans_from(result),
            summary_calls=calls,
            summary_errors=errors,
            final_messages=len(result.final_messages),
            problems=problems,
            error=result.error,
        )
    )

    # -- Case 5: oversized single message ----------------------------------- #
    oversized = _oversized_scenario()
    result = await ReplayHarness(
        settings, summary_model_factory=make_stub_summary_model, run_tag="fault-oversized"
    ).run(oversized, "langcompress")
    problems = check_invariants(result.final_messages, [])
    if result.error:
        problems.append(f"replay crashed: {result.error}")
    if not any(e.kind in ("l3", "degraded") for e in result.events):
        problems.append("no compression event fired for the oversized conversation")
    calls, errors = _stats(result)
    report.outcomes.append(
        FaultOutcome(
            case="oversized_single_message",
            description="A 30k-char tool result inside a long conversation must compress without crashing.",
            ok=not problems,
            plans=_plans_from(result),
            summary_calls=calls,
            summary_errors=errors,
            final_messages=len(result.final_messages),
            problems=problems,
            error=result.error,
        )
    )

    return report


__all__ = [
    "FaultOutcome",
    "RobustnessReport",
    "check_invariants",
    "run_fault_suite",
]
