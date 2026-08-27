"""Report rendering — reproducible envelopes, JSON + Markdown output.

A report is only evidence when it carries its own reproduction recipe.
Every report therefore embeds an *envelope*: run id, wall-clock timestamp,
git commit, Python + package versions, the full ``BenchSettings`` snapshot
and the content hash of every scenario that produced the numbers. Two
reports with identical envelopes and identical scenario hashes differ
only by non-determinism — which, given temperature=0 and seeded stubs,
should be nothing.

Two formats, one truth (both serialize the same ``BenchMetrics`` object):

- **JSON** (``bench_<run_id>.json``) — machine-diffable; regression jobs
  and CI gates consume this one.
- **Markdown** (``bench_<run_id>.md``) — human-readable; four-dimension
  tables with the comparison arms as rows.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.config import BenchSettings
from benchmarks.judge import CalibrationReport
from benchmarks.metrics import BenchMetrics
from benchmarks.scenario import Scenario


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    """Short commit hash, or ``unknown`` outside a git worktree."""
    try:
        # fixed argv, no user input; check=False — a non-git worktree is
        # handled via the returncode below, not an exception
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


@dataclass
class ReportEnvelope:
    """Reproduction recipe for one benchmark run."""

    run_id: str
    created_at: str
    mode: str  # "stub" | "real"
    commit: str
    python: str
    versions: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    scenario_hashes: dict[str, str] = field(default_factory=dict)
    cache_stats: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "commit": self.commit,
            "python": self.python,
            "versions": self.versions,
            "settings": self.settings,
            "scenario_hashes": self.scenario_hashes,
            "cache_stats": self.cache_stats,
        }


def build_envelope(
    settings: BenchSettings,
    scenarios: list[Scenario],
    *,
    mode: str,
    cache_stats: dict[str, int] | None = None,
) -> ReportEnvelope:
    now = datetime.now(timezone.utc)
    commit = _git_commit()
    return ReportEnvelope(
        run_id=f"{now.strftime('%Y%m%d_%H%M%S')}_{commit}",
        created_at=now.isoformat(timespec="seconds"),
        mode=mode,
        commit=commit,
        python=platform.python_version(),
        versions={
            "langcompress": _version("langcompress"),
            "langchain": _version("langchain"),
            "langchain-core": _version("langchain-core"),
            "langgraph": _version("langgraph"),
            "python": sys.version.split()[0],
        },
        settings=settings.as_dict(),
        scenario_hashes={s.id: s.content_hash() for s in scenarios},
        cache_stats=dict(cache_stats or {}),
    )


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _f(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_markdown(
    envelope: ReportEnvelope,
    metrics: BenchMetrics,
    *,
    calibration: CalibrationReport | None = None,
    pairwise: list[dict[str, Any]] | None = None,
    dump_files: dict[str, str] | None = None,
) -> str:
    """Render the full Markdown report body."""
    lines: list[str] = []
    add = lines.append

    add("# langcompress compression benchmark report")
    add("")
    add(f"- **run**: `{envelope.run_id}` ({envelope.mode} mode)")
    add(f"- **created**: {envelope.created_at}")
    add(f"- **commit**: `{envelope.commit}`  ·  python {envelope.python}")
    versions = " · ".join(f"{k} {v}" for k, v in envelope.versions.items() if k != "python")
    add(f"- **stack**: {versions}")
    if envelope.cache_stats:
        cache = envelope.cache_stats
        add(
            f"- **llm cache**: {cache.get('hits', 0)} hits / {cache.get('misses', 0)} misses"
            f" ({cache.get('entries', 0)} entries)"
        )
    add("")

    arm_order = [a for a in ("langcompress", "bare_summarization", "trim", "full_context") if a in metrics.arms]
    if not arm_order:
        arm_order = list(metrics.arms)

    # -- headline table ---------------------------------------------------- #
    add("## Headline (four dimensions, one row per arm)")
    add("")
    add(
        "| arm | compression ratio ↓ | probes ↑ | judge overall ↑ | "
        "fact recall ↑ | fabrication ↓ | summary calls | wall s |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for arm in arm_order:
        agg = metrics.arms[arm]
        add(
            f"| `{arm}` | {_f(agg.avg_compression_ratio)} | {_pct(agg.avg_probe_score)} | "
            f"{_pct(agg.avg_rubric.get('overall'))} | {_pct(agg.avg_rubric.get('fact_recall'))} | "
            f"{_pct(agg.avg_fabrication_rate)} | {agg.total_summary_calls} | "
            f"{_f(agg.total_summary_seconds, 1)} |"
        )
    add("")

    # -- efficiency -------------------------------------------------------- #
    add("## Efficiency — how much, how fast, at what cost")
    add("")
    add(
        "| arm | ratio ↓ | avg event reduction ↑ | L0 share of cut | "
        "events (l3/l0/degraded) | summary s | summary tokens in/out | errors |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for arm in arm_order:
        agg = metrics.arms[arm]
        by_arm = [m for m in metrics.scenarios if m.arm == arm]
        kinds = {"l3": 0, "l0_only": 0, "degraded": 0}
        for m in by_arm:
            for kind, n in m.event_counts.items():
                kinds[kind] = kinds.get(kind, 0) + n
        events = f"{kinds.get('l3', 0)}/{kinds.get('l0_only', 0)}/{kinds.get('degraded', 0)}"
        add(
            f"| `{arm}` | {_f(agg.avg_compression_ratio)} | {_pct(agg.avg_event_reduction)} | "
            f"{_pct(agg.avg_l0_share)} | {events} | {_f(agg.total_summary_seconds, 1)} | "
            f"{agg.total_summary_input_tokens}/{agg.total_summary_output_tokens} | {agg.n_errors} |"
        )
    add("")
    add(
        "> L0/L3 attribution: `l0_share` is the rule-based filter's fraction of the total "
        "token cut (sum of `l0_delta` / sum of `l0_delta + l3_delta`); the rest is the "
        "summarizer's. `full_context` has no cut by construction."
    )
    add("")

    # -- fidelity ---------------------------------------------------------- #
    add("## Fidelity — intrinsic review of the summary itself")
    add("")
    add(
        "| arm | judge | overall ↑ | fact recall ↑ | segment ↑ | structure ↑ | "
        "concise ↑ | entity recall ↑ | fabricated numbers |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for arm in arm_order:
        agg = metrics.arms[arm]
        r = agg.avg_rubric
        judge = next((m.judge_kind for m in metrics.scenarios if m.arm == arm and m.judge_kind), "—")
        add(
            f"| `{arm}` | {judge} | {_pct(r.get('overall'))} | {_pct(r.get('fact_recall'))} | "
            f"{_pct(r.get('segment_coverage'))} | {_pct(r.get('structure_compliance'))} | "
            f"{_pct(r.get('conciseness'))} | {_pct(agg.avg_entity_recall)} | "
            f"{_pct(agg.avg_fabrication_rate)} |"
        )
    add("")

    # -- consistency ------------------------------------------------------- #
    add("## Consistency — extrinsic QA probes (the hardest metric)")
    add("")
    add("| arm | probes | score ↑ | keyword score ↑ | failures (compressed-away) |")
    add("|---|---|---|---|---|")
    for arm in arm_order:
        agg = metrics.arms[arm]
        add(
            f"| `{arm}` | {agg.total_probes} | {_pct(agg.avg_probe_score)} | "
            f"{_pct(agg.avg_probe_keyword_score)} | {agg.total_probe_failures_compressed} |"
        )
    add("")
    add(
        "> Failures bucketed as `all_compressed`/`partially_compressed` are real compression "
        "costs (the source messages were removed); `all_retained` failures are probe/grading "
        "artifacts — on `full_context` every failure lands there by construction, so a low "
        "`full_context` score flags a corpus problem, not a compression one."
    )
    add("")

    # -- per-scenario detail ----------------------------------------------- #
    add("## Per-scenario detail")
    add("")
    add("| scenario | arm | ratio ↓ | probes ↑ | judge ↑ | facts ↑ | fabrication ↓ |")
    add("|---|---|---|---|---|---|---|")
    for m in metrics.scenarios:
        add(
            f"| {m.scenario_id} | `{m.arm}` | {_f(m.compression_ratio)} | "
            f"{_pct(m.probe_score)} | {_pct(m.rubric.get('overall') if m.rubric else None)} | "
            f"{_pct(m.rubric.get('fact_recall') if m.rubric else None)} | "
            f"{_pct(m.fabrication_rate)} |"
        )
    add("")

    # -- robustness -------------------------------------------------------- #
    if metrics.robustness is not None:
        add("## Robustness — fault injection")
        add("")
        add(f"**all cases ok: {metrics.robustness.all_ok}**  ·  plan distribution: "
            f"{metrics.robustness.plan_distribution or '—'}")
        add("")
        add("| case | ok | plans | summary calls | errors | final msgs | problems |")
        add("|---|---|---|---|---|---|---|")
        for outcome in metrics.robustness.outcomes:
            problems = "; ".join(outcome.problems[:3]) or ("—" if outcome.ok else "")
            add(
                f"| {outcome.case} | {outcome.ok} | {outcome.plans or '—'} | "
                f"{outcome.summary_calls} | {outcome.summary_errors} | "
                f"{outcome.final_messages} | {problems} |"
            )
        add("")

    # -- judge calibration -------------------------------------------------- #
    if calibration is not None:
        add("## Judge calibration (golden set)")
        add("")
        add(f"- cases: {calibration.n_cases}")
        add(f"- rubric MAE per dimension: "
            f"{ {k: round(v, 3) for k, v in calibration.rubric_mae.items()} }")
        add(f"- fact-verdict agreement: {_pct(calibration.fact_agreement)}")
        add(f"- **calibration passed: {calibration.passed}**")
        add("")

    # -- pairwise ----------------------------------------------------------- #
    if pairwise:
        add("## Pairwise (langcompress vs bare, position-bias controlled)")
        add("")
        add("| pair | A wins | B wins | ties | A win rate |")
        add("|---|---|---|---|---|")
        for p in pairwise:
            add(
                f"| `{p.get('a')}` vs `{p.get('b')}` | {p.get('a_wins')} | {p.get('b_wins')} | "
                f"{p.get('ties')} | {_pct(p.get('a_win_rate'))} |"
            )
        add("")

    # -- dumps (human/AI review workflow) ----------------------------------- #
    if dump_files:
        add("## Dumps (full-text before/after, for human or AI review)")
        add("")
        add(
            "Each file records every compression event's complete message "
            "lists (before/after) plus the final context. Open one when a "
            "number above looks suspicious and judge the compression yourself."
        )
        add("")
        for label in sorted(dump_files):
            add(f"- `{label}` → {dump_files[label]}")
        add("")

    # -- reproduce ----------------------------------------------------------- #
    add("## Reproduce")
    add("")
    add(f"```bash\npython -m benchmarks run --mode {envelope.mode}\n```")
    add("")
    add("Scenario corpus hashes (report is only comparable when these match):")
    add("")
    add("```json")
    add(json.dumps(envelope.scenario_hashes, indent=2, sort_keys=True))
    add("```")
    add("")
    add("Settings snapshot:")
    add("")
    add("```json")
    add(json.dumps(envelope.settings, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_reports(
    directory: Path,
    envelope: ReportEnvelope,
    metrics: BenchMetrics,
    *,
    calibration: CalibrationReport | None = None,
    pairwise: list[dict[str, Any]] | None = None,
    dump_files: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write JSON + Markdown side by side; returns (json_path, md_path)."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "envelope": envelope.as_dict(),
        "metrics": metrics.as_dict(),
        "calibration": calibration.as_dict() if calibration else None,
        "pairwise": pairwise or [],
        "dump_files": dump_files or {},
    }
    json_path = directory / f"bench_{envelope.run_id}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    md_path = directory / f"bench_{envelope.run_id}.md"
    md_path.write_text(
        render_markdown(
            envelope,
            metrics,
            calibration=calibration,
            pairwise=pairwise,
            dump_files=dump_files,
        ),
        encoding="utf-8",
    )
    return json_path, md_path


__all__ = [
    "ReportEnvelope",
    "build_envelope",
    "render_markdown",
    "write_reports",
]
