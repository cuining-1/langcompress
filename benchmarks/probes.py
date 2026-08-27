"""Extrinsic QA probes — the consistency dimension's hardest metric.

Method (the "外在评估" half of the dual evaluation): the reconstructed
context after compression (summary + preserved recent window) must still
support answering questions that *only* someone who read the original
conversation can answer. Score = fraction of probes answered correctly =
information retention as seen from downstream.

The probes themselves live in the scenario corpus (question / reference
answer / answer_keywords / source_message_ids, LLM-batch-generated then
human-spot-checked) — that is the offline, versioned form of "generate
questions from the content about to be compressed". Bench-side we only
*answer* and *grade*:

====================  =======================================================
stage                 mechanism
====================  =======================================================
answering             an :class:`Answerer` renders the arm's final message
                      list into text and answers each question from it.
                      ``LLMAnswerer`` (real mode) calls the probe model
                      through the content-hash cache; ``ContextEchoAnswerer``
                      (stub mode) "answers" with the rendered context
                      itself — an ideal parrot, turning the probe into a
                      pure context-containment check so the whole pipeline
                      produces non-trivial, deterministic numbers keyless.
keyword grading       every ``answer_keywords`` entry must appear in the
                      answer (case-insensitive, whitespace-normalized;
                      word-boundary anchored when the keyword is a bare
                      word, so ``500`` does not match ``1500``). Zero judge
                      cost, zero subjectivity — phase-1 default.
LLM grading (opt-in)  an independent judge model compares answer vs
                      reference (numbers/names/identifiers must match,
                      wording may differ). Reported alongside the keyword
                      score; ``passed = keyword OR llm`` so synonyms never
                      count as losses while keyword hits stay a strict
                      lower bound.
loss attribution      each failed probe is bucketed by whether its
                      ``source_message_ids`` survived in the final context:
                      ``all_compressed`` (the compression genuinely paid
                      for this loss), ``partially_compressed``, or
                      ``all_retained`` (sources survived — probe too hard
                      or grading too strict, i.e. *not* a compression cost).
====================  =======================================================

The loss scope is computed as *all transcript keys minus final-message
keys* — arm-agnostic by construction (an L3-summarized message, a trimmed
message and a never-replayed one are all simply "absent from the context
the answerer saw"). On the ``full_context`` arm the scope is empty, so its
probe score doubles as a corpus self-check: probes that fail with every
message present are probes (or keyword sets) that need fixing, not
compression findings.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from benchmarks.llm import LLMCache, cached_call, cached_json_call, estimate_tokens_text
from benchmarks.scenario import QAProbe, Scenario
from benchmarks.telemetry import message_keys

# --------------------------------------------------------------------------- #
# Context rendering
# --------------------------------------------------------------------------- #


def message_text(message: BaseMessage) -> str:
    """Flatten one message to plain text the way a model would see it.

    Content-parts lists keep only ``text`` parts — reasoning parts are
    stripped by L0 before the model ever sees the message, and the echo
    answerer must grade against exactly that post-strip view.
    ``tool_calls`` are rendered as ``name(args)`` so tool *invocations*
    (as opposed to results) are answerable too.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content or "")
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        rendered = "; ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
        text = f"{text}\n[tool_calls] {rendered}" if text else f"[tool_calls] {rendered}"
    return text.strip()


def render_context(messages: list[BaseMessage]) -> str:
    """Render the final message list as the text given to the answerer.

    Summary messages (``additional_kwargs["__summarization__"]``) are
    labelled ``[SUMMARY]`` — everything else by its message type — so the
    answerer can tell compressed memory from live messages.
    """
    blocks: list[str] = []
    for m in messages:
        ak = getattr(m, "additional_kwargs", None)
        is_summary = isinstance(ak, dict) and (
            ak.get("__summarization__") or ak.get("lc_source") == "summarization"
        )
        label = "SUMMARY" if is_summary else getattr(m, "type", "message")
        body = message_text(m)
        if body:
            blocks.append(f"[{label}]\n{body}")
    return "\n\n".join(blocks)


def compute_loss_scope(scenario: Scenario, final_messages: list[BaseMessage]) -> set[str]:
    """Keys of transcript entries absent from the final context.

    Union over the whole replay: anything not in ``final_messages`` was
    removed somewhere along the way (L3 summarization, trim window, or an
    earlier compression's own drop) — for probe attribution the mechanism
    is irrelevant, only presence is.
    """
    present = message_keys(final_messages)
    return {
        scenario.message_key(i) for i in range(len(scenario.transcript))
    } - present


# --------------------------------------------------------------------------- #
# Keyword grading
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    """Casefold + collapse all whitespace runs to single spaces.

    Shared normalization contract for every text-matching grader in the
    bench (keyword hits, fact recall, segment coverage) — one rule, so
    scores from different modules stay comparable.
    """
    return " ".join(text.split()).lower()


_BARE_WORD = re.compile(r"^\w+$", re.UNICODE)
# CJK ideographs (incl. extension A) — scripts without spaces/punctuation
# between words, where regex \b anchoring is meaningless.
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def keyword_hit(answer: str, keyword: str) -> bool:
    """Case-insensitive containment with word-boundary anchoring.

    A keyword consisting solely of word characters (``500``, ``p95``,
    ``invoice_counters``) must match on token boundaries — ``500`` inside
    ``1500`` is a miss. Keywords with punctuation (``3.11``, ``SELECT MAX``)
    fall back to plain normalized substring containment (regex word
    boundaries around punctuation are unreliable). Keywords containing CJK
    characters are always plain substrings: Chinese has no word boundaries
    for ``\\b`` to anchor on, so ``网关`` inside ``支付网关超时`` is a hit.
    """
    kw = normalize(keyword)
    if not kw:
        return True  # degenerate empty keyword: nothing to demand
    haystack = normalize(answer)
    if not haystack:
        return False
    if _CJK_CHAR.search(kw):
        return kw in haystack
    if _BARE_WORD.fullmatch(kw):
        return re.search(rf"\b{re.escape(kw)}\b", haystack) is not None
    return kw in haystack


def grade_keywords(answer: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """Split keywords into (matched, missing)."""
    matched, missing = [], []
    for kw in keywords:
        (matched if keyword_hit(answer, kw) else missing).append(kw)
    return matched, missing


# --------------------------------------------------------------------------- #
# Answerers
# --------------------------------------------------------------------------- #

_ANSWER_TEMPLATE = """Answer the question using ONLY the conversation context below. \
The context may contain a [SUMMARY] block compressing earlier turns plus recent messages. \
If the context does not contain the information, answer exactly: insufficient context. \
Otherwise answer concisely (one sentence or a short list); preserve exact numbers, \
names, identifiers and version strings.

<context>
{context}
</context>

Question: {question}
Answer:"""


class Answerer(Protocol):
    """Anything that can answer a question from a rendered context."""

    async def answer(self, context: str, question: str) -> str: ...


@dataclass
class LLMAnswerer:
    """Real-mode answerer: probe model + content-hash cache, temp pinned 0."""

    model: Any
    model_id: str
    cache: LLMCache
    temperature: float = 0.0

    async def answer(self, context: str, question: str) -> str:
        prompt = _ANSWER_TEMPLATE.format(context=context, question=question)
        return await cached_call(
            self.model,
            self.model_id,
            prompt,
            purpose="probe_answer",
            cache=self.cache,
            temperature=self.temperature,
        )


@dataclass
class ContextEchoAnswerer:
    """Keyless stand-in: "answers" with the rendered context verbatim.

    The answer text *is* the context, so keyword grading measures whether
    the answer-bearing tokens survived compression at all — a deterministic
    lower bound on information retention that makes the whole probe
    pipeline runnable (and smoke-testable) without any API key.
    """

    async def answer(self, context: str, question: str) -> str:
        return context


# --------------------------------------------------------------------------- #
# LLM grader (opt-in second grading mode)
# --------------------------------------------------------------------------- #

_GRADE_TEMPLATE = """You are grading a short answer against a reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Does the graded answer convey the key information of the reference? \
Exact numbers, names, identifiers and version strings must match; wording may differ. \
An "insufficient context" answer is always a fail.

Return ONLY a JSON object: {{"pass": true, "reason": "..."}} or {{"pass": false, "reason": "..."}}"""


class Grader(Protocol):
    """Optional second grading mode (independent judge model)."""

    async def grade(self, probe: QAProbe, answer: str) -> bool: ...


@dataclass
class LLMGrader:
    """Judge-model grader: answer vs reference, JSON-verdict, cached."""

    model: Any
    model_id: str
    cache: LLMCache
    temperature: float = 0.0

    async def grade(self, probe: QAProbe, answer: str) -> bool:
        prompt = _GRADE_TEMPLATE.format(
            question=probe.question, reference=probe.answer, answer=answer
        )
        verdict = await cached_json_call(
            self.model,
            self.model_id,
            prompt,
            purpose="probe_grade",
            cache=self.cache,
            temperature=self.temperature,
        )
        return bool(verdict.get("pass") is True)


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #


@dataclass
class ProbeOutcome:
    """One probe: its answer, both grading modes, and loss attribution."""

    probe_id: str
    question: str
    expected_answer: str
    answer: str
    keyword_passed: bool
    llm_passed: bool | None  # None → LLM grading not requested this run
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    attribution: str = ""  # all_compressed | partially_compressed | all_retained | unattributed
    dropped_source_keys: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Keyword OR LLM — keyword hits are a strict lower bound, the LLM
        grader only rescues genuine synonym answers, never overrides a
        keyword fail with another fail."""
        if self.llm_passed is None:
            return self.keyword_passed
        return self.keyword_passed or self.llm_passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "passed": self.passed,
            "keyword_passed": self.keyword_passed,
            "llm_passed": self.llm_passed,
            "missing_keywords": self.missing_keywords,
            "attribution": self.attribution,
            "dropped_source_keys": self.dropped_source_keys,
            "answer": self.answer[:400],
        }


@dataclass
class ProbeReport:
    """All probe outcomes for one (scenario × arm)."""

    arm: str
    scenario_id: str
    outcomes: list[ProbeOutcome] = field(default_factory=list)
    context_tokens: int = 0
    answer_seconds: float = 0.0
    grade_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def score(self) -> float:
        """Information retention rate: passed / total (0.0 when no probes)."""
        return self.passed_count / self.total if self.total else 0.0

    @property
    def keyword_score(self) -> float:
        kw = sum(1 for o in self.outcomes if o.keyword_passed)
        return kw / self.total if self.total else 0.0

    @property
    def failure_attribution(self) -> dict[str, int]:
        """Failed probes bucketed by loss attribution — the "where did the
        information go" breakdown that turns a score into a diagnosis."""
        buckets: dict[str, int] = {}
        for o in self.outcomes:
            if not o.passed:
                buckets[o.attribution or "unattributed"] = buckets.get(o.attribution or "unattributed", 0) + 1
        return buckets

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "scenario_id": self.scenario_id,
            "total": self.total,
            "passed": self.passed_count,
            "score": round(self.score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "failure_attribution": self.failure_attribution,
            "context_tokens": self.context_tokens,
            "answer_seconds": round(self.answer_seconds, 4),
            "grade_seconds": round(self.grade_seconds, 4),
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #


def attribute_probe(
    probe: QAProbe, source_keys: set[str], loss_scope: set[str]
) -> tuple[str, list[str]]:
    """Bucket a probe's provenance against what compression removed.

    Returns ``(attribution, dropped_keys)``:

    - ``all_compressed``    every source message is absent from the final
                            context — a failure here is a real compression
                            cost (the summary failed to carry it);
    - ``partially_compressed`` some sources absent, some present;
    - ``all_retained``      every source survived — failure is not a
                            compression cost (probe difficulty / grading
                            strictness; on ``full_context`` every failure
                            lands here by construction);
    - ``unattributed``      corpus probe without ``source_message_ids``.
    """
    if not source_keys:
        return "unattributed", []
    dropped = sorted(source_keys & loss_scope)
    if not dropped:
        return "all_retained", []
    if len(dropped) == len(source_keys):
        return "all_compressed", dropped
    return "partially_compressed", dropped


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #


async def evaluate_probes(
    scenario: Scenario,
    final_messages: list[BaseMessage],
    *,
    arm: str,
    answerer: Answerer,
    grader: Grader | None = None,
) -> ProbeReport:
    """Answer and grade every corpus probe against one arm's final context.

    Sequential per-probe execution on purpose: deterministic ordering,
    reproducible reports, and no rate-limit bursts — reruns are paid for
    once by the content-hash cache, not by concurrency.
    """
    report = ProbeReport(arm=arm, scenario_id=scenario.id)
    context = render_context(final_messages)
    report.context_tokens = estimate_tokens_text(context) if context else 0
    loss_scope = compute_loss_scope(scenario, final_messages)

    for probe in scenario.qa_probes:
        source_keys = {scenario.message_key(i) for i in probe.source_message_ids}
        attribution, dropped = attribute_probe(probe, source_keys, loss_scope)

        started = time.perf_counter()
        answer = await answerer.answer(context, probe.question)
        report.answer_seconds += time.perf_counter() - started

        matched, missing = grade_keywords(answer, probe.answer_keywords)
        keyword_passed = not missing  # every keyword must hit

        llm_passed: bool | None = None
        if grader is not None and not keyword_passed:
            # Grade only keyword misses — hits are already passes; this
            # halves judge cost while never changing a pass outcome.
            started = time.perf_counter()
            llm_passed = await grader.grade(probe, answer)
            report.grade_seconds += time.perf_counter() - started

        report.outcomes.append(
            ProbeOutcome(
                probe_id=probe.id,
                question=probe.question,
                expected_answer=probe.answer,
                answer=answer,
                keyword_passed=keyword_passed,
                llm_passed=llm_passed,
                matched_keywords=matched,
                missing_keywords=missing,
                attribution=attribution,
                dropped_source_keys=dropped,
            )
        )
    return report


__all__ = [
    "Answerer",
    "ContextEchoAnswerer",
    "Grader",
    "LLMAnswerer",
    "LLMGrader",
    "ProbeOutcome",
    "ProbeReport",
    "attribute_probe",
    "compute_loss_scope",
    "evaluate_probes",
    "grade_keywords",
    "keyword_hit",
    "message_text",
    "normalize",
    "render_context",
]
