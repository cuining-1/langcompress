"""Intrinsic AI reviewer — rubric scoring grounded in the fact checklist.

The "内在评估" half of the dual evaluation: a judge inspects the summary
*itself* (not downstream answerability) against the scenario's ground
truth. Output is a per-dimension rubric, never a single number:

=======================  ====================================================
dimension                what it measures
=======================  ====================================================
``fact_recall``          fraction of ground-truth facts the summary carries
                         (checklist-grounded: every fact gets a verdict)
``segment_coverage``     how many of the eight template segments carry real
                         content (design §7 section list)
``structure_compliance`` whether the eight-section header structure is used
``conciseness``          information density vs. summary bloat
=======================  ====================================================

Anti-drift constraints (all four from the methodology, enforced here):

1. **rubric + checklist grounding** — the judge returns structured
   per-fact verdicts, not vibes; subjective judgment is converted into
   a verifiable checklist audit.
2. **independence** — the judge model is a separate role
   (``BENCH_JUDGE_MODEL``); in pairwise mode it never grades a summary
   produced by itself.
3. **position-bias control** — pairwise comparisons run twice with
   A/B order swapped; the reported win rate is the average of both
   orders.
4. **calibration** — :func:`calibrate_judge` scores the golden set and
   reports judge-vs-human agreement; the reviewer itself gets reviewed.

Two judge implementations share one verdict shape:

- :class:`HeuristicJudge` — fully objective, keyless, deterministic
  (fact recall via content-word hits with the probes' normalizer,
  header presence for segments, token-ratio curve for conciseness,
  numeric fabrication check against the source text). This is the
  phase-1 default and the CI smoke path.
- :class:`LLMJudge` — judge model via ``cached_json_call``
  (temperature pinned 0, content-hash cached); its numeric fabrication
  check still runs the objective pass, so hallucination detection never
  depends on a possibly-hallucinating judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from benchmarks.llm import LLMCache, cached_json_call, estimate_tokens_text
from benchmarks.probes import normalize
from benchmarks.scenario import EntitySpec, Fact

# Eight-segment titles exactly as the package template emits them
# (src/langcompress/summarizer/templates.py) — bench-side copy: a consumer
# grades against the documented §7 contract, not private internals.
SEGMENTS: tuple[str, ...] = (
    "Primary Request and Intent",
    "Key Technical Concepts",
    "Files and Code Sections",
    "Errors and Fixes",
    "Problem Solving",
    "All User Messages",
    "Pending Tasks",
    "Entity State",
)

# Rubric weights for the scalar ``overall`` (fact recall dominates — it is
# the dimension with checklist grounding; structure is worth least because
# it is cheap to satisfy). Exported so reports can restate the weighting.
RUBRIC_WEIGHTS: dict[str, float] = {
    "fact_recall": 0.5,
    "segment_coverage": 0.2,
    "structure_compliance": 0.1,
    "conciseness": 0.2,
}

_STOPWORDS = frozenset(
    ["the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "been", "have", "has", "had", "not", "but", "its", "into", "onto", "when", "then", "they", "them", "their", "there", "here", "which", "while", "after", "before", "because", "about", "would", "should", "could", "will", "shall", "must", "also", "more", "most", "some", "such", "only", "than", "very", "just", "over", "under", "again", "both", "each", "other", "using", "used", "use", "into", "within", "without", "across", "during", "between", "against", "above", "below", "does", "didn't", "don't", "isn't", "wasn't", "weren't", "cannot", "can't", "won't", "our", "your", "their", "we", "you", "he", "she", "it", "as", "at", "by", "of", "on", "or", "if", "so", "no", "nor", "too", "own", "same", "s", "t", "d", "ll", "re", "ve", "m", "y", "o", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"]
)

_WORD_RE = re.compile(r"[a-z][a-z0-9_\-\.]{2,}")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# CJK ideograph runs (incl. extension A) — the unit of Chinese fact text.
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def _cjk_bigrams(text: str) -> list[str]:
    """Character bigrams of every CJK run (runs of length 1 yield none).

    Chinese facts must stay gradeable without shipping a word segmenter
    (the bench adds no dependencies), so CJK text is compared at
    two-character granularity: coarse enough that scattered single
    characters don't trivially match everywhere, fine enough that
    paraphrase keeps most bigrams."""
    grams: list[str] = []
    for run in _CJK_RUN.findall(text):
        grams.extend(run[i : i + 2] for i in range(len(run) - 1))
    return grams


def content_words(text: str) -> list[str]:
    """Normalized content tokens for fact grading.

    ASCII: alphabetic identifiers ≥3 chars minus stopwords, plus every
    numeric token (numbers carry facts — versions, ports, thresholds —
    and are far too load-bearing to filter). CJK: character bigrams per
    run (see :func:`_cjk_bigrams`) — ``in`` containment against the
    normalized summary then works for Chinese facts the same way word
    containment works for English ones."""
    norm = normalize(text)
    words = [w for w in _WORD_RE.findall(norm) if w not in _STOPWORDS]
    words.extend(_NUM_RE.findall(norm))
    words.extend(_cjk_bigrams(norm))
    return words


def numeric_tokens(text: str) -> set[str]:
    return set(_NUM_RE.findall(normalize(text)))


# --------------------------------------------------------------------------- #
# Verdict shapes
# --------------------------------------------------------------------------- #


@dataclass
class Rubric:
    """Per-dimension scores, each in [0, 1]."""

    fact_recall: float = 0.0
    segment_coverage: float = 0.0
    structure_compliance: float = 0.0
    conciseness: float = 0.0

    @property
    def overall(self) -> float:
        return sum(getattr(self, k) * w for k, w in RUBRIC_WEIGHTS.items())

    def as_dict(self) -> dict[str, float]:
        return {
            "fact_recall": round(self.fact_recall, 4),
            "segment_coverage": round(self.segment_coverage, 4),
            "structure_compliance": round(self.structure_compliance, 4),
            "conciseness": round(self.conciseness, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class JudgeVerdict:
    """One judged summary: rubric + checklist audit + fabrications."""

    summary_tokens: int = 0
    source_tokens: int = 0
    rubric: Rubric = field(default_factory=Rubric)
    fact_verdicts: dict[str, str] = field(default_factory=dict)  # id → preserved|partial|lost
    entity_verdicts: dict[str, str] = field(default_factory=dict)  # name → present|absent
    fabrications: list[str] = field(default_factory=list)  # unsupported numeric claims
    fabrication_rate: float = 0.0  # fabricated numbers / all numbers in summary
    judge_kind: str = "heuristic"  # "heuristic" | "llm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge_kind,
            "summary_tokens": self.summary_tokens,
            "source_tokens": self.source_tokens,
            "rubric": self.rubric.as_dict(),
            "fact_verdicts": self.fact_verdicts,
            "entity_verdicts": self.entity_verdicts,
            "fabrications": self.fabrications,
            "numeric_fabrication_rate": round(self.fabrication_rate, 4),
        }


# --------------------------------------------------------------------------- #
# Objective (heuristic) judge — keyless, deterministic
# --------------------------------------------------------------------------- #


def _fact_score(fact_text: str, summary_norm: str) -> str:
    """Verdict for one fact via content-word containment.

    All content words hit → ``preserved``; ≥40% hit → ``partial``; below →
    ``lost``. The 0.4 threshold tolerates legitimate paraphrase (articles,
    reordering, synonym function words) while refusing to credit a summary
    that kept only a stray token or two.
    """
    words = content_words(fact_text)
    if not words:
        return "preserved"  # nothing gradeable in the fact itself
    hits = sum(1 for w in words if w in summary_norm)
    ratio = hits / len(words)
    if ratio >= 1.0:
        return "preserved"
    if ratio >= 0.4:
        return "partial"
    return "lost"


def _conciseness_score(summary_tokens: int, source_tokens: int) -> float:
    """Piecewise-linear density curve over the compression ratio.

    ratio ≤ 0.25 → 1.0 (dense); 0.25–0.5 → 1.0→0.5 linear; 0.5–1.0 →
    0.5→0.0 linear; ratio ≥ 1.0 (summary longer than source) → 0.0."""
    if source_tokens <= 0 or summary_tokens <= 0:
        return 0.0
    ratio = summary_tokens / source_tokens
    if ratio <= 0.25:
        return 1.0
    if ratio <= 0.5:
        return 1.0 - (ratio - 0.25) / 0.25 * 0.5
    if ratio < 1.0:
        return 0.5 * (1.0 - (ratio - 0.5) / 0.5)
    return 0.0


def numeric_fabrications(summary: str, source_text: str) -> tuple[list[str], float]:
    """Numbers asserted in the summary but absent from the source text.

    Objective hallucination detector: a number the compressed-away
    conversation never contained cannot be a compression artifact — it is
    a fabrication. Returns ``(fabricated_numbers, fabrication_rate)``.

    Single-digit tokens are excluded on both sides: template section
    numbering ("## 1. Primary Request...") makes them structurally
    ubiquitous while carrying no fact — including them would bury the
    signal in false positives."""
    summary_nums = {n for n in numeric_tokens(summary) if len(n.replace(".", "")) >= 2}
    if not summary_nums:
        return [], 0.0
    source_nums = numeric_tokens(source_text)
    fabricated = sorted(n for n in summary_nums if n not in source_nums)
    return fabricated, len(fabricated) / len(summary_nums)


class HeuristicJudge:
    """Fully objective judge — the default in stub mode and phase 1."""

    async def judge(
        self,
        summary: str,
        *,
        facts: list[Fact],
        entities: list[EntitySpec],
        source_text: str,
    ) -> JudgeVerdict:
        summary_norm = normalize(summary)
        source_tokens = estimate_tokens_text(source_text) if source_text else 0
        summary_tokens = estimate_tokens_text(summary) if summary else 0

        verdicts = {f.id: _fact_score(f.text, summary_norm) for f in facts}
        # Score: preserved=1.0, partial=0.5, lost=0.0
        fact_recall = (
            sum({"preserved": 1.0, "partial": 0.5, "lost": 0.0}[v] for v in verdicts.values())
            / len(verdicts)
            if verdicts
            else 0.0
        )

        covered = sum(1 for seg in SEGMENTS if normalize(seg) in summary_norm)
        segment_coverage = covered / len(SEGMENTS) if SEGMENTS else 0.0

        # Structure: segment titles that appear as markdown headings
        # (``## N. Title``) rather than buried in prose.
        heading_lines = [
            normalize(line) for line in summary.splitlines() if line.strip().startswith("#")
        ]
        structured = sum(
            1 for seg in SEGMENTS if any(normalize(seg) in line for line in heading_lines)
        )
        structure_compliance = structured / len(SEGMENTS) if SEGMENTS else 0.0

        entity_verdicts = {
            e.name: ("present" if any(normalize(a) in summary_norm for a in [e.value, *e.aliases]) else "absent")
            for e in entities
        }

        fabricated, fab_rate = numeric_fabrications(summary, source_text)
        verdict = JudgeVerdict(
            summary_tokens=summary_tokens,
            source_tokens=source_tokens,
            rubric=Rubric(
                fact_recall=fact_recall,
                segment_coverage=segment_coverage,
                structure_compliance=structure_compliance,
                conciseness=_conciseness_score(summary_tokens, source_tokens),
            ),
            fact_verdicts=verdicts,
            entity_verdicts=entity_verdicts,
            fabrications=fabricated,
            fabrication_rate=fab_rate,
            judge_kind="heuristic",
        )
        return verdict


# --------------------------------------------------------------------------- #
# LLM judge — rubric + checklist, cached, temperature 0
# --------------------------------------------------------------------------- #

_JUDGE_TEMPLATE = """You are an impartial judge evaluating a conversation summary against ground truth.

The summary is expected to follow an eight-segment template with these sections:
{segments}

Grade four dimensions, each 0.0-1.0:
- fact_recall: fraction of ground-truth facts the summary preserves (audit the checklist below, fact by fact)
- segment_coverage: how many of the eight segments carry real content
- structure_compliance: whether the eight-section header structure is actually used
- conciseness: information density (penalize redundancy and bloat)

Fact checklist (id: text):
{facts}

Entity checklist (name: latest value):
{entities}

Also flag fabrications: specific numbers in the summary that appear in NEITHER the checklists above NOR this list of numbers from the source conversation: {source_numbers}.
Numbers only — do not flag paraphrase or omitted information as fabrication.

Return ONLY a JSON object:
{{"rubric": {{"fact_recall": 0.0, "segment_coverage": 0.0, "structure_compliance": 0.0, "conciseness": 0.0}},
 "fact_verdicts": {{"<fact_id>": "preserved"|"partial"|"lost"}},
 "entity_verdicts": {{"<name>": "present"|"absent"}},
 "fabrications": ["<unsupported number>"]}}"""


class LLMJudge:
    """Judge-model reviewer: one cached JSON call per summary.

    The objective numeric-fabrication pass always runs alongside — a judge
    that hallucinates must never be the only hallucination detector.
    """

    def __init__(self, model: Any, model_id: str, cache: LLMCache, temperature: float = 0.0) -> None:
        self.model = model
        self.model_id = model_id
        self.cache = cache
        self.temperature = temperature

    async def judge(
        self,
        summary: str,
        *,
        facts: list[Fact],
        entities: list[EntitySpec],
        source_text: str,
    ) -> JudgeVerdict:
        heuristic = await HeuristicJudge().judge(
            summary, facts=facts, entities=entities, source_text=source_text
        )
        facts_block = "\n".join(f"- {f.id}: {f.text}" for f in facts) or "- (none)"
        entities_block = (
            "\n".join(
                f"- {e.name}: {e.value}" + (f" (aliases: {', '.join(e.aliases)})" if e.aliases else "")
                for e in entities
            )
            or "- (none)"
        )
        prompt = _JUDGE_TEMPLATE.format(
            segments="\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(SEGMENTS)),
            facts=facts_block,
            entities=entities_block,
            source_numbers=", ".join(sorted(numeric_tokens(source_text))) or "(none)",
        )
        raw = await cached_json_call(
            self.model,
            self.model_id,
            prompt,
            purpose="judge_rubric",
            cache=self.cache,
            temperature=self.temperature,
        )
        rubric_raw = raw.get("rubric", {})

        def _dim(key: str, fallback: float) -> float:
            try:
                val = float(rubric_raw.get(key, fallback))
            except (TypeError, ValueError):
                return fallback
            return min(max(val, 0.0), 1.0)

        # Fact recall from the judge's own verdicts, not its self-reported
        # score — the checklist audit is the grounded number.
        fact_verdicts = {
            str(k): ("preserved" if v == "preserved" else "partial" if v == "partial" else "lost")
            for k, v in (raw.get("fact_verdicts") or {}).items()
        } or heuristic.fact_verdicts
        fact_recall = (
            sum({"preserved": 1.0, "partial": 0.5, "lost": 0.0}[v] for v in fact_verdicts.values())
            / len(fact_verdicts)
            if fact_verdicts
            else 0.0
        )

        entity_verdicts = {
            str(k): ("present" if v == "present" else "absent")
            for k, v in (raw.get("entity_verdicts") or {}).items()
        } or heuristic.entity_verdicts

        # Fabrications: objective numbers are ground truth and always kept;
        # LLM findings are listed when they look numeric, but the *rate*
        # stays on the objective numerator/denominator so it can never be
        # inflated by a judge hallucination of its own.
        llm_fabs = [str(x) for x in (raw.get("fabrications") or []) if str(x)]
        objective_fabs = set(heuristic.fabrications)
        llm_extra = {f for f in llm_fabs if _NUM_RE.search(f)}
        all_numbers = numeric_tokens(summary)
        verdict = JudgeVerdict(
            summary_tokens=heuristic.summary_tokens,
            source_tokens=heuristic.source_tokens,
            rubric=Rubric(
                fact_recall=fact_recall,
                segment_coverage=_dim("segment_coverage", heuristic.rubric.segment_coverage),
                structure_compliance=_dim("structure_compliance", heuristic.rubric.structure_compliance),
                conciseness=_dim("conciseness", heuristic.rubric.conciseness),
            ),
            fact_verdicts=fact_verdicts,
            entity_verdicts=entity_verdicts,
            fabrications=sorted(objective_fabs | llm_extra),
            fabrication_rate=len(objective_fabs) / len(all_numbers) if all_numbers else 0.0,
            judge_kind="llm",
        )
        return verdict


class SummaryJudge(Protocol):
    """Common interface: heuristic and LLM judges are interchangeable."""

    async def judge(
        self, summary: str, *, facts: list[Fact], entities: list[EntitySpec], source_text: str
    ) -> JudgeVerdict: ...


# --------------------------------------------------------------------------- #
# Pairwise comparison (position-bias controlled)
# --------------------------------------------------------------------------- #

_PAIRWISE_TEMPLATE = """Compare two summaries (A and B) of the same conversation.

Fact checklist (id: text):
{facts}

Summary A:
{summary_a}

Summary B:
{summary_b}

Which summary better preserves the ground-truth facts, covers the eight template sections (Primary Request and Intent, Key Technical Concepts, Files and Code Sections, Errors and Fixes, Problem Solving, All User Messages, Pending Tasks, Entity State) and stays concise?
Judge content quality only, not length for its own sake.

Return ONLY a JSON object: {{"better": "A"|"B"|"tie", "reason": "..."}}"""


@dataclass
class PairwiseResult:
    """Aggregated pairwise outcome over both presentation orders."""

    a_label: str
    b_label: str
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.ties

    @property
    def a_win_rate(self) -> float:
        """(wins + ties/2) / comparisons — ties count half for either side."""
        return (self.a_wins + 0.5 * self.ties) / self.n if self.n else 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a_label,
            "b": self.b_label,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "a_win_rate": round(self.a_win_rate, 4),
        }


async def judge_pairwise(
    model: Any,
    model_id: str,
    cache: LLMCache,
    *,
    facts: list[Fact],
    summary_a: str,
    summary_b: str,
    a_label: str,
    b_label: str,
    temperature: float = 0.0,
) -> PairwiseResult:
    """LLM pairwise with order swap: the average of (A,B) and (B,A) runs.

    Position bias is the classic pairwise failure mode (judges favour
    whichever summary is presented first); asking both orders and averaging
    cancels first/second-position preference symmetrically.
    """
    result = PairwiseResult(a_label=a_label, b_label=b_label)
    facts_block = "\n".join(f"- {f.id}: {f.text}" for f in facts) or "- (none)"

    async def _one_order(first: str, second: str, first_is_a: bool) -> None:
        prompt = _PAIRWISE_TEMPLATE.format(
            facts=facts_block,
            summary_a=first,
            summary_b=second,
        )
        raw = await cached_json_call(
            model, model_id, prompt, purpose="judge_pairwise", cache=cache, temperature=temperature
        )
        better = str(raw.get("better", "tie")).strip().upper()
        if better == "TIE" or better not in ("A", "B"):
            result.ties += 1
        elif (better == "A") == first_is_a:
            result.a_wins += 1
        else:
            result.b_wins += 1

    await _one_order(summary_a, summary_b, first_is_a=True)
    await _one_order(summary_b, summary_a, first_is_a=False)
    return result


def objective_pairwise(
    verdict_a: JudgeVerdict, verdict_b: JudgeVerdict, *, tolerance: float = 0.02
) -> str:
    """Keyless pairwise: compare scalar overall scores with a dead zone.

    Used directly in stub mode, and as a sanity cross-check against LLM
    pairwise in real mode (large divergence between objective and LLM
    pairwise is itself a judge-drift signal)."""
    diff = verdict_a.rubric.overall - verdict_b.rubric.overall
    if diff > tolerance:
        return "a"
    if diff < -tolerance:
        return "b"
    return "tie"


# --------------------------------------------------------------------------- #
# Golden-set calibration — the judge is itself reviewed
# --------------------------------------------------------------------------- #


class GoldenCase(BaseModel):
    """One human-annotated judgement the judge must reproduce.

    Human annotations come from corpus review; a judge that disagrees with
    these too often is drifting, not the corpus."""

    id: str
    summary: str
    source_text: str = ""
    facts: list[Fact] = Field(default_factory=list)
    entities: list[EntitySpec] = Field(default_factory=list)
    human_rubric: dict[str, float] = Field(default_factory=dict)
    human_fact_verdicts: dict[str, str] = Field(default_factory=dict)


@dataclass
class CalibrationReport:
    """Judge-vs-human agreement over the golden set."""

    n_cases: int = 0
    rubric_mae: dict[str, float] = field(default_factory=dict)  # per-dimension mean absolute error
    fact_agreement: float = 0.0  # exact-verdict agreement rate
    overall_mae: float = 0.0
    passed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "rubric_mae": {k: round(v, 4) for k, v in self.rubric_mae.items()},
            "overall_mae": round(self.overall_mae, 4),
            "fact_agreement": round(self.fact_agreement, 4),
            "passed": self.passed,
        }


async def calibrate_judge(
    judge: SummaryJudge,
    cases: list[GoldenCase],
    *,
    max_fact_mae: float = 0.15,
    min_fact_agreement: float = 0.75,
) -> CalibrationReport:
    """Run the judge over golden cases and measure human agreement.

    Pass thresholds (defaults): every rubric dimension MAE ≤ 0.15 and
    fact-verdict agreement ≥ 0.75. A judge that fails calibration gets a
    loud ``passed=False`` in the report — its numbers are then advisory,
    not evidence."""
    report = CalibrationReport(n_cases=len(cases))
    if not cases:
        return report

    dims = ("fact_recall", "segment_coverage", "structure_compliance", "conciseness")
    abs_err: dict[str, list[float]] = {d: [] for d in dims}
    overall_errs: list[float] = []
    agreements: list[float] = []

    for case in cases:
        verdict = await judge.judge(
            case.summary, facts=case.facts, entities=case.entities, source_text=case.source_text
        )
        for d in dims:
            human = case.human_rubric.get(d)
            if human is not None:
                abs_err[d].append(abs(verdict.rubric.as_dict()[d] - float(human)))
            overall_errs.append(
                abs(
                    verdict.rubric.overall
                    - sum(float(case.human_rubric.get(k, 0.0)) * w for k, w in RUBRIC_WEIGHTS.items())
                )
            )
        if case.human_fact_verdicts:
            agree = sum(
                1
                for fid, expected in case.human_fact_verdicts.items()
                if verdict.fact_verdicts.get(fid) == expected
            )
            agreements.append(agree / len(case.human_fact_verdicts))

    report.rubric_mae = {d: sum(v) / len(v) for d, v in abs_err.items() if v}
    report.overall_mae = sum(overall_errs) / len(overall_errs) if overall_errs else 0.0
    report.fact_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    report.passed = bool(
        report.rubric_mae
        and all(mae <= max_fact_mae for mae in report.rubric_mae.values())
        and report.fact_agreement >= min_fact_agreement
    )
    return report


def load_golden(path) -> list[GoldenCase]:
    """Load ``golden_set.json`` (``{"cases": [...]}``)."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(c) for c in data.get("cases", [])]


__all__ = [
    "RUBRIC_WEIGHTS",
    "SEGMENTS",
    "CalibrationReport",
    "GoldenCase",
    "HeuristicJudge",
    "JudgeVerdict",
    "LLMJudge",
    "PairwiseResult",
    "Rubric",
    "calibrate_judge",
    "content_words",
    "judge_pairwise",
    "load_golden",
    "numeric_fabrications",
    "objective_pairwise",
]
