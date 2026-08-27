"""Scenario corpus data model — ground-truth-annotated replay transcripts.

A scenario is the benchmark's unit of work: a scripted conversation (the
replay transcript), plus the ground truth needed to *score* a compression of
it (fact checklist, entity list, QA probes with reference answers). Scenario
files are plain JSON under ``benchmarks/scenarios/`` and are git-versioned —
the corpus is data, not code, so it can be regenerated/augmented by an LLM
batch tool (``tools/gen_scenarios.py`` pattern) and human-reviewed via diff.

Categories deliberately target the eight-segment summary's segments one by
one (design §6 content taxonomy → §7 template sections):

============================  ==========================  ====================
category id                  targets (segments)          stress point
============================  ==========================  ====================
``long_multi_topic_drift``   1, 2, 6, 8                  intent survives topic
                                                          drift; all user msgs
``tool_json_heavy``          2, 3, 8                     big tool payloads;
                                                          numbers survive
``error_fix_loop``           4, 5                        error history; no
                                                          repeat mistakes
``entity_tracking``          8, 1, 6                     latest-vs-stale entity
                                                          state across renames
``cross_compression``        1, 7, 2, 6                  task continuity across
                                                          2+ compaction points
``thinking_heavy``           5, 2, 8                     conclusions survive
                                                          reasoning-stripping
============================  ==========================  ====================

Transcript roles map onto the replay harness exactly:

- ``user``      → a message the harness sends into the agent (drives a turn).
- ``assistant`` → a scripted model response (returned in order by the fake
  agent model; may carry ``tool_calls``).
- ``tool``      → a scripted tool result (consumed FIFO per tool name; carries
  ``tool_call_id`` matching the issuing assistant entry).

``message_ids`` on facts/entities/probes are 0-based transcript indices —
the same indices the harness stamps onto replayed messages as ids
(``t{i}``) so loss can be attributed back to concrete dropped messages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_VALID_ROLES = {"user", "assistant", "tool"}


class ToolSpec(BaseModel):
    """A tool the scripted agent may call (name + description for the schema)."""

    name: str
    description: str = ""


class TranscriptEntry(BaseModel):
    """One entry of the replay script.

    ``content`` may be a string or a content-parts list (the thinking-heavy
    category uses ``[{"type": "reasoning", ...}, {"type": "text", ...}]``
    Gemini-style parts, which L0 strips); assistant entries may carry
    ``tool_calls`` (``{name, args, id}``); tool entries carry ``tool_call_id``
    plus the tool ``name``.
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    additional_kwargs: dict[str, Any] | None = None


class Fact(BaseModel):
    """One ground-truth fact the summary should preserve.

    ``message_ids`` localize the fact to transcript entries — used both to
    scope the judge (facts whose sources were compressed away are the ones the
    *summary* must carry) and to explain probe losses.
    """

    id: str
    text: str
    message_ids: list[int] = Field(default_factory=list)
    segment: int | None = None


class EntitySpec(BaseModel):
    """A key entity whose *latest* value must survive in the summary (§7.8).

    ``aliases`` allow objective recall matching when the summary may
    legitimately use a different surface form (e.g. with/without a prefix).
    """

    name: str
    value: str
    message_ids: list[int] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    note: str = ""


class QAProbe(BaseModel):
    """An extrinsic probe: answerable only by reading the (compressed) context.

    Grading defaults to objective keyword containment — every keyword in
    ``answer_keywords`` (case-insensitive, whitespace-normalized) must appear
    in the answerer's response — so phase-1 consistency scoring carries zero
    judge subjectivity. An LLM grader is available opt-in per run.
    """

    id: str
    question: str
    answer: str
    answer_keywords: list[str] = Field(default_factory=list)
    source_message_ids: list[int] = Field(default_factory=list)
    segment: int | None = None


class Scenario(BaseModel):
    """A ground-truth-annotated conversation replay (see module docstring)."""

    id: str
    category: str
    description: str = ""
    target_segments: list[int] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    transcript: list[TranscriptEntry]
    facts: list[Fact] = Field(default_factory=list)
    entities: list[EntitySpec] = Field(default_factory=list)
    qa_probes: list[QAProbe] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived views over the transcript
    # ------------------------------------------------------------------ #

    def user_entries(self) -> list[tuple[int, TranscriptEntry]]:
        return [(i, e) for i, e in enumerate(self.transcript) if e.role == "user"]

    def assistant_entries(self) -> list[tuple[int, TranscriptEntry]]:
        return [(i, e) for i, e in enumerate(self.transcript) if e.role == "assistant"]

    def tool_entries(self) -> list[tuple[int, TranscriptEntry]]:
        return [(i, e) for i, e in enumerate(self.transcript) if e.role == "tool"]

    def message_key(self, index: int) -> str:
        """Stable key for transcript entry ``index`` (id used at replay time).

        Tool results are keyed by ``tool_call_id`` (ToolNode assigns fresh
        message ids we cannot predict); every other entry is keyed by the
        ``t{index}`` id the harness stamps on the message it constructs.
        """
        entry = self.transcript[index]
        if entry.role == "tool" and entry.tool_call_id:
            return entry.tool_call_id
        return f"t{index}"

    def content_hash(self) -> str:
        """Stable content digest — recorded in reports so a report always
        states exactly which corpus version produced it."""
        payload = {
            "transcript": [e.model_dump(exclude_none=True) for e in self.transcript],
            "facts": [f.model_dump() for f in self.facts],
            "entities": [e.model_dump() for e in self.entities],
            "qa_probes": [p.model_dump() for p in self.qa_probes],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def validate_structure(self) -> list[str]:
        """Cheap structural lint (called by the corpus loader and the test
        suite): role spelling, tool-call pairing, ground-truth references.
        Returns a list of human-readable problems (empty = clean)."""
        problems: list[str] = []
        seen_call_ids: set[str] = set()
        tool_names = {t.name for t in self.tools}
        pending_calls: dict[str, str] = {}  # call_id -> tool name
        for i, entry in enumerate(self.transcript):
            if entry.role not in _VALID_ROLES:
                problems.append(f"[{i}] invalid role {entry.role!r}")
                continue
            if entry.role == "assistant" and entry.tool_calls:
                # L0 drops empty-content messages; an AIMessage whose entire
                # payload is tool_calls would be dropped and orphan its
                # ToolMessage. Transcripts must give such entries text.
                if not entry.content or entry.content == "":
                    problems.append(f"[{i}] assistant tool_call entry needs non-empty text content")
                for call in entry.tool_calls:
                    cid = str(call.get("id", ""))
                    cname = str(call.get("name", ""))
                    if not cid or not cname:
                        problems.append(f"[{i}] tool_call missing id/name")
                    if cname not in tool_names:
                        problems.append(f"[{i}] tool_call references undeclared tool {cname!r}")
                    pending_calls[cid] = cname
            if entry.role == "tool":
                cid = entry.tool_call_id or ""
                if cid not in pending_calls:
                    problems.append(f"[{i}] tool result without matching tool_call")
                else:
                    seen_call_ids.add(cid)
                    del pending_calls[cid]
                if entry.name and entry.name not in tool_names:
                    problems.append(f"[{i}] tool result references undeclared tool {entry.name!r}")
        if pending_calls:
            problems.append(f"unanswered tool calls: {sorted(pending_calls)}")
        n = len(self.transcript)
        for fact in self.facts:
            if any(not (0 <= m < n) for m in fact.message_ids):
                problems.append(f"fact {fact.id} references out-of-range message ids")
        for ent in self.entities:
            if any(not (0 <= m < n) for m in ent.message_ids):
                problems.append(f"entity {ent.name} references out-of-range message ids")
        for probe in self.qa_probes:
            if any(not (0 <= m < n) for m in probe.source_message_ids):
                problems.append(f"probe {probe.id} references out-of-range message ids")
            if not probe.answer_keywords:
                problems.append(f"probe {probe.id} has no answer_keywords (objective grading impossible)")
        return problems


def load_scenario(path: Path | str) -> Scenario:
    """Load one scenario JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Scenario.model_validate(data)


def load_corpus(scenario_dir: Path | str | None = None) -> list[Scenario]:
    """Load every ``*.json`` under the corpus dir, validating structure.

    Raises ``ValueError`` naming the offending file(s) when any scenario fails
    structural lint — a broken corpus must never silently produce numbers.
    """
    if scenario_dir is not None:
        directory = Path(scenario_dir)
    else:
        from benchmarks.config import BENCH_ROOT

        directory = BENCH_ROOT / "scenarios"
    scenarios: list[Scenario] = []
    failures: list[str] = []
    for path in sorted(directory.glob("*.json")):
        scenario = load_scenario(path)
        problems = scenario.validate_structure()
        if problems:
            failures.append(f"{path.name}: {'; '.join(problems)}")
        scenarios.append(scenario)
    if failures:
        raise ValueError("corpus structural lint failed: " + " | ".join(failures))
    if not scenarios:
        raise ValueError(f"no scenario files found under {directory}")
    return scenarios
