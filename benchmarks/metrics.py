"""Four-dimension metric aggregation — raw run artifacts become numbers.

The evaluation model is a vector, not a scalar. Every metric below is
attributable to exactly one dimension and one measurement channel:

==================  ============================  ==========================
dimension           metrics                      source
==================  ============================  ==========================
efficiency          compression_ratio,           ArmResult + CompressEvent
(省了多少/多快/多贵)  avg_event_reduction,         telemetry (hook)
                    l0/l3 contribution shares,
                    summary calls/seconds/tokens
fidelity            rubric (fact recall,          JudgeVerdict (intrinsic,
(丢了多少/有没有捏造)  segment coverage, ...)       AI reviewer + heuristics)
                    entity recall,
                    fabrication rate
consistency         probe score (retention),      ProbeReport (extrinsic,
(还能做对事吗)        failure attribution          QA probes)
robust              fault cases ok-rate,          RobustnessReport
(兜底可靠吗)          plan distribution            (fault injection)
==================  ============================  ==========================

Token accounting caveat (documented, not hidden): ``transcript_tokens``
comes from the harness's char/4 estimate over raw scenario content, while
``final_state_tokens`` estimates over materialized messages — both are the
same estimator family, so *ratios* are comparable across arms even though
absolute counts are approximations. The ``full_context`` arm doubles as
the sanity anchor: its ratio must sit at ~1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.judge import JudgeVerdict
from benchmarks.probes import ProbeReport
from benchmarks.replayer import ArmResult
from benchmarks.robustness import RobustnessReport


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


# --------------------------------------------------------------------------- #
# Per (scenario × arm)
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioMetrics:
    """All dimension metrics for one scenario under one arm."""

    scenario_id: str
    arm: str

    # -- efficiency -------------------------------------------------------- #
    transcript_tokens: int = 0
    final_state_tokens: int = 0
    compression_ratio: float | None = None  # final / transcript (lower = better)
    event_counts: dict[str, int] = field(default_factory=dict)  # kind → count
    avg_event_reduction: float | None = None
    l0_share: float | None = None  # L0's fraction of total token reduction
    l3_share: float | None = None
    summary_calls: int = 0
    summary_errors: int = 0
    summary_seconds: float = 0.0
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    wall_seconds: float = 0.0

    # -- fidelity (intrinsic) ---------------------------------------------- #
    judge_kind: str | None = None
    rubric: dict[str, float] | None = None
    entity_recall: float | None = None
    fabrication_rate: float | None = None

    # -- consistency (extrinsic) -------------------------------------------- #
    probe_total: int | None = None
    probe_score: float | None = None
    probe_keyword_score: float | None = None
    probe_failures_compressed: int | None = None  # failures whose sources were removed

    # -- run health --------------------------------------------------------- #
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "ok": self.ok,
            "error": self.error,
            "efficiency": {
                "transcript_tokens": self.transcript_tokens,
                "final_state_tokens": self.final_state_tokens,
                "compression_ratio": _rounded(self.compression_ratio),
                "event_counts": self.event_counts,
                "avg_event_reduction": _rounded(self.avg_event_reduction),
                "l0_share": _rounded(self.l0_share),
                "l3_share": _rounded(self.l3_share),
                "summary_calls": self.summary_calls,
                "summary_errors": self.summary_errors,
                "summary_seconds": round(self.summary_seconds, 4),
                "summary_input_tokens": self.summary_input_tokens,
                "summary_output_tokens": self.summary_output_tokens,
                "wall_seconds": round(self.wall_seconds, 4),
            },
            "fidelity": {
                "judge_kind": self.judge_kind,
                "rubric": self.rubric,
                "entity_recall": _rounded(self.entity_recall),
                "fabrication_rate": _rounded(self.fabrication_rate),
            },
            "consistency": {
                "probe_total": self.probe_total,
                "probe_score": _rounded(self.probe_score),
                "probe_keyword_score": _rounded(self.probe_keyword_score),
                "probe_failures_compressed": self.probe_failures_compressed,
            },
        }


def scenario_metrics(
    arm_result: ArmResult,
    probe_report: ProbeReport | None = None,
    judge_verdict: JudgeVerdict | None = None,
) -> ScenarioMetrics:
    """Fold one arm run's artifacts into the metric vector."""
    m = ScenarioMetrics(scenario_id=arm_result.scenario_id, arm=arm_result.arm)
    m.transcript_tokens = arm_result.transcript_tokens
    m.final_state_tokens = arm_result.final_state_tokens
    m.error = arm_result.error
    m.wall_seconds = arm_result.wall_seconds

    if arm_result.transcript_tokens > 0:
        m.compression_ratio = arm_result.final_state_tokens / arm_result.transcript_tokens

    counts: dict[str, int] = {}
    reductions: list[float] = []
    l0_total = 0
    l3_total = 0
    for event in arm_result.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
        reductions.append(event.reduction_ratio)
        l0_total += event.l0_delta
        l3_total += event.l3_delta
    m.event_counts = counts
    m.avg_event_reduction = _mean(reductions)
    total_cut = l0_total + l3_total
    if total_cut > 0:
        m.l0_share = l0_total / total_cut
        m.l3_share = l3_total / total_cut

    m.summary_calls = int(arm_result.summary_stats.get("calls", 0))
    m.summary_errors = int(arm_result.summary_stats.get("errors", 0))
    m.summary_seconds = float(arm_result.summary_stats.get("total_seconds", 0.0))
    m.summary_input_tokens = int(arm_result.summary_stats.get("input_tokens", 0))
    m.summary_output_tokens = int(arm_result.summary_stats.get("output_tokens", 0))

    if judge_verdict is not None:
        m.judge_kind = judge_verdict.judge_kind
        m.rubric = judge_verdict.rubric.as_dict()
        if judge_verdict.entity_verdicts:
            present = sum(1 for v in judge_verdict.entity_verdicts.values() if v == "present")
            m.entity_recall = present / len(judge_verdict.entity_verdicts)
        m.fabrication_rate = judge_verdict.fabrication_rate

    if probe_report is not None:
        m.probe_total = probe_report.total
        m.probe_score = probe_report.score
        m.probe_keyword_score = probe_report.keyword_score
        attribution = probe_report.failure_attribution
        compressed_failures = attribution.get("all_compressed", 0) + attribution.get(
            "partially_compressed", 0
        )
        m.probe_failures_compressed = compressed_failures
    return m


# --------------------------------------------------------------------------- #
# Per arm (across scenarios)
# --------------------------------------------------------------------------- #


@dataclass
class ArmMetrics:
    """Scenario-averaged metrics for one arm — the comparison-table row."""

    arm: str
    n_scenarios: int = 0
    n_errors: int = 0

    # efficiency averages
    avg_compression_ratio: float | None = None
    avg_event_reduction: float | None = None
    avg_l0_share: float | None = None
    total_summary_calls: int = 0
    total_summary_errors: int = 0
    total_summary_seconds: float = 0.0
    total_summary_input_tokens: int = 0
    total_summary_output_tokens: int = 0

    # fidelity averages
    avg_rubric: dict[str, float] = field(default_factory=dict)
    avg_entity_recall: float | None = None
    avg_fabrication_rate: float | None = None

    # consistency averages
    avg_probe_score: float | None = None
    avg_probe_keyword_score: float | None = None
    total_probe_failures_compressed: int = 0
    total_probes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_scenarios": self.n_scenarios,
            "n_errors": self.n_errors,
            "avg_compression_ratio": _rounded(self.avg_compression_ratio),
            "avg_event_reduction": _rounded(self.avg_event_reduction),
            "avg_l0_share": _rounded(self.avg_l0_share),
            "total_summary_calls": self.total_summary_calls,
            "total_summary_errors": self.total_summary_errors,
            "total_summary_seconds": round(self.total_summary_seconds, 4),
            "total_summary_input_tokens": self.total_summary_input_tokens,
            "total_summary_output_tokens": self.total_summary_output_tokens,
            "avg_rubric": {k: _rounded(v) for k, v in self.avg_rubric.items()},
            "avg_entity_recall": _rounded(self.avg_entity_recall),
            "avg_fabrication_rate": _rounded(self.avg_fabrication_rate),
            "avg_probe_score": _rounded(self.avg_probe_score),
            "avg_probe_keyword_score": _rounded(self.avg_probe_keyword_score),
            "total_probe_failures_compressed": self.total_probe_failures_compressed,
            "total_probes": self.total_probes,
        }


def aggregate_arm(arm: str, metrics: list[ScenarioMetrics]) -> ArmMetrics:
    """Average one arm's scenario metrics (failed runs count as errors and
    drop out of dimension averages — a crashed run has no fidelity)."""
    agg = ArmMetrics(arm=arm, n_scenarios=len(metrics))
    agg.n_errors = sum(1 for m in metrics if not m.ok)
    ok_metrics = [m for m in metrics if m.ok]

    agg.avg_compression_ratio = _mean([m.compression_ratio for m in ok_metrics if m.compression_ratio is not None])
    agg.avg_event_reduction = _mean([m.avg_event_reduction for m in ok_metrics if m.avg_event_reduction is not None])
    agg.avg_l0_share = _mean([m.l0_share for m in ok_metrics if m.l0_share is not None])
    agg.total_summary_calls = sum(m.summary_calls for m in ok_metrics)
    agg.total_summary_errors = sum(m.summary_errors for m in ok_metrics)
    agg.total_summary_seconds = sum(m.summary_seconds for m in ok_metrics)
    agg.total_summary_input_tokens = sum(m.summary_input_tokens for m in ok_metrics)
    agg.total_summary_output_tokens = sum(m.summary_output_tokens for m in ok_metrics)

    rubric_keys = ("fact_recall", "segment_coverage", "structure_compliance", "conciseness", "overall")
    agg.avg_rubric = {
        key: _mean([m.rubric[key] for m in ok_metrics if m.rubric and key in m.rubric])  # type: ignore[index]
        for key in rubric_keys
    }
    agg.avg_rubric = {k: v for k, v in agg.avg_rubric.items() if v is not None}
    agg.avg_entity_recall = _mean([m.entity_recall for m in ok_metrics if m.entity_recall is not None])
    agg.avg_fabrication_rate = _mean([m.fabrication_rate for m in ok_metrics if m.fabrication_rate is not None])

    agg.avg_probe_score = _mean([m.probe_score for m in ok_metrics if m.probe_score is not None])
    agg.avg_probe_keyword_score = _mean([m.probe_keyword_score for m in ok_metrics if m.probe_keyword_score is not None])
    agg.total_probe_failures_compressed = sum(m.probe_failures_compressed or 0 for m in ok_metrics)
    agg.total_probes = sum(m.probe_total or 0 for m in ok_metrics)
    return agg


# --------------------------------------------------------------------------- #
# Whole run
# --------------------------------------------------------------------------- #


@dataclass
class BenchMetrics:
    """Everything the report layer needs."""

    scenarios: list[ScenarioMetrics] = field(default_factory=list)
    arms: dict[str, ArmMetrics] = field(default_factory=dict)
    robustness: RobustnessReport | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": {arm: agg.as_dict() for arm, agg in self.arms.items()},
            "scenarios": [m.as_dict() for m in self.scenarios],
            "robustness": self.robustness.as_dict() if self.robustness else None,
        }


def build_metrics(
    scenario_rows: list[ScenarioMetrics],
    *,
    robustness: RobustnessReport | None = None,
) -> BenchMetrics:
    """Assemble per-arm aggregates + the whole-run metrics object."""
    bench = BenchMetrics(scenarios=list(scenario_rows), robustness=robustness)
    arms: dict[str, list[ScenarioMetrics]] = {}
    for row in scenario_rows:
        arms.setdefault(row.arm, []).append(row)
    bench.arms = {arm: aggregate_arm(arm, rows) for arm, rows in arms.items()}
    return bench


__all__ = [
    "ArmMetrics",
    "BenchMetrics",
    "ScenarioMetrics",
    "aggregate_arm",
    "build_metrics",
    "scenario_metrics",
]
