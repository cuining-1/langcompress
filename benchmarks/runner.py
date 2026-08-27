"""Benchmark orchestration — one command from corpus to signed report.

Pipeline (per scenario × arm, sequential for determinism and rate-limit
friendliness — the content-hash cache pays for reruns, not concurrency)::

    load corpus → replay arm → probe evaluation (extrinsic)
                            → judge evaluation (intrinsic, arms with a summary)
    → fault-injection suite (longest scenario)
    → pairwise langcompress vs bare (order-swapped in LLM mode)
    → golden-set judge calibration
    → four-dimension metrics → JSON + Markdown reports

Mode resolution: with no model configured anywhere the run degrades to
fully deterministic stub mode (echo answerer + heuristic judge + objective
pairwise) — the same code path CI smokes. With models configured the roles
split per ``BenchSettings``; ``--mode stub`` forces the keyless path even
then, which is how a real-model regression A/B keeps its control arm.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.config import BenchSettings, load_settings
from benchmarks.judge import (
    HeuristicJudge,
    JudgeVerdict,
    LLMJudge,
    PairwiseResult,
    calibrate_judge,
    judge_pairwise,
    load_golden,
    objective_pairwise,
)
from benchmarks.llm import LLMCache, make_real_model, make_stub_summary_model
from benchmarks.metrics import BenchMetrics, ScenarioMetrics, build_metrics, scenario_metrics
from benchmarks.probes import ContextEchoAnswerer, LLMAnswerer, LLMGrader, evaluate_probes
from benchmarks.replayer import ARMS, ReplayHarness
from benchmarks.report import ReportEnvelope, build_envelope, write_reports
from benchmarks.robustness import RobustnessReport, run_fault_suite
from benchmarks.scenario import Scenario, load_corpus


def render_scenario_text(scenario: Scenario) -> str:
    """Full original conversation as flat text — the ground-truth source
    for fabrication checks (a number is fabricated iff the *original*
    conversation never contained it)."""
    parts: list[str] = []
    for entry in scenario.transcript:
        content = entry.content
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        label = entry.name or entry.role
        parts.append(f"[{label}] {content}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Role assembly
# --------------------------------------------------------------------------- #


@dataclass
class Roles:
    """The three LLM roles of a run, resolved once up front."""

    summary_factory: Any  # () -> model (fresh instance per arm-run)
    answerer: Any
    judge: Any
    grader: Any | None
    pairwise_model: Any | None  # (model, model_id) | None → objective pairwise
    pairwise_model_id: str = ""


def _build_roles(settings: BenchSettings, *, use_llm_grading: bool = False) -> Roles:
    if settings.stub:
        return Roles(
            summary_factory=make_stub_summary_model,
            answerer=ContextEchoAnswerer(),
            judge=HeuristicJudge(),
            grader=None,
            pairwise_model=None,
        )
    cache = LLMCache(settings.cache_path, settings.cache_enabled)

    def _summary_factory() -> Any:
        if settings.summary_model:
            return make_real_model(settings.summary_model, settings.temperature)
        return make_stub_summary_model()

    answerer: Any = ContextEchoAnswerer()
    if settings.probe_model:
        probe_model = make_real_model(settings.probe_model, settings.temperature)
        answerer = LLMAnswerer(
            model=probe_model,
            model_id=settings.probe_model,
            cache=cache,
            temperature=settings.temperature,
        )

    judge: Any = HeuristicJudge()
    grader = None
    pairwise_model = None
    pairwise_model_id = ""
    if settings.judge_model:
        judge_model = make_real_model(settings.judge_model, settings.temperature)
        judge = LLMJudge(
            model=judge_model,
            model_id=settings.judge_model,
            cache=cache,
            temperature=settings.temperature,
        )
        pairwise_model = judge_model
        pairwise_model_id = settings.judge_model
        if use_llm_grading:
            grader = LLMGrader(
                model=judge_model,
                model_id=settings.judge_model,
                cache=cache,
                temperature=settings.temperature,
            )
    return Roles(
        summary_factory=_summary_factory,
        answerer=answerer,
        judge=judge,
        grader=grader,
        pairwise_model=pairwise_model,
        pairwise_model_id=pairwise_model_id,
    )


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclass
class RunArtifacts:
    """What a completed run produced (paths + headline numbers)."""

    settings: BenchSettings
    envelope: ReportEnvelope
    metrics: BenchMetrics
    json_path: Path | None = None
    md_path: Path | None = None
    dump_dir: Path | None = None  # full-text dumps root for this run


async def run_benchmark(
    settings: BenchSettings,
    *,
    arms: list[str] | None = None,
    scenario_filter: list[str] | None = None,
    with_robustness: bool = True,
    with_pairwise: bool = True,
    with_calibration: bool = True,
    use_llm_grading: bool = False,
    run_tag: str = "bench",
) -> RunArtifacts:
    """Execute the full pipeline and write both report formats."""
    arm_list = list(arms or ARMS)
    unknown = [a for a in arm_list if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown} (expected subsets of {ARMS})")

    scenarios = load_corpus(settings.scenario_dir)
    if scenario_filter:
        scenarios = [s for s in scenarios if s.id in scenario_filter or s.category in scenario_filter]
        if not scenarios:
            raise ValueError(f"no scenarios match filter {scenario_filter}")

    roles = _build_roles(settings, use_llm_grading=use_llm_grading)
    cache = LLMCache(settings.cache_path, settings.cache_enabled)

    # Full-text dumps: one directory per run, one file per (scenario × arm).
    # Disabled via BENCH_DUMP=0 / --no-dump (e.g. to keep CI artifacts small).
    dump_dir: Path | None = None
    if settings.dump_enabled:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dump_dir = settings.dump_dir / f"bench_{stamp}_{run_tag}"
    harness = ReplayHarness(
        settings,
        summary_model_factory=roles.summary_factory,
        run_tag=run_tag,
        dump_dir=dump_dir,
    )

    rows: list[ScenarioMetrics] = []
    summaries: dict[tuple[str, str], str] = {}
    verdicts: dict[tuple[str, str], JudgeVerdict] = {}
    dump_files: dict[str, str] = {}

    for scenario in scenarios:
        source_text = render_scenario_text(scenario)
        for arm in arm_list:
            arm_result = await harness.run(scenario, arm)
            probe_report = None
            verdict = None
            if arm_result.ok:
                probe_report = await evaluate_probes(
                    scenario,
                    arm_result.final_messages,
                    arm=arm,
                    answerer=roles.answerer,
                    grader=roles.grader,
                )
                if arm_result.final_summary:
                    verdict = await roles.judge.judge(
                        arm_result.final_summary,
                        facts=scenario.facts,
                        entities=scenario.entities,
                        source_text=source_text,
                    )
                    summaries[(scenario.id, arm)] = arm_result.final_summary
                    verdicts[(scenario.id, arm)] = verdict
            rows.append(scenario_metrics(arm_result, probe_report, verdict))
            if arm_result.dump_path is not None:
                dump_files[f"{scenario.id} × {arm}"] = str(arm_result.dump_path)

    robustness: RobustnessReport | None = None
    if with_robustness and scenarios:
        # The fault suite needs a conversation long enough to actually
        # trigger compression — pick the longest transcript.
        robustness = await run_fault_suite(settings, max(scenarios, key=lambda s: len(s.transcript)))

    metrics = build_metrics(rows, robustness=robustness)

    pairwise: list[dict[str, Any]] = []
    if with_pairwise and "langcompress" in arm_list and "bare_summarization" in arm_list:
        for scenario in scenarios:
            a = summaries.get((scenario.id, "langcompress"))
            b = summaries.get((scenario.id, "bare_summarization"))
            if not (a and b):
                continue
            if roles.pairwise_model is not None:
                result = await judge_pairwise(
                    roles.pairwise_model,
                    roles.pairwise_model_id,
                    cache,
                    facts=scenario.facts,
                    summary_a=a,
                    summary_b=b,
                    a_label="langcompress",
                    b_label="bare_summarization",
                    temperature=settings.temperature,
                )
            else:
                va = verdicts.get((scenario.id, "langcompress"))
                vb = verdicts.get((scenario.id, "bare_summarization"))
                if not (va and vb):
                    continue
                winner = objective_pairwise(va, vb)
                result = PairwiseResult(
                    a_label="langcompress",
                    b_label="bare_summarization",
                    a_wins=1 if winner == "a" else 0,
                    b_wins=1 if winner == "b" else 0,
                    ties=1 if winner == "tie" else 0,
                )
            pairwise.append(result.as_dict())

    calibration = None
    if with_calibration and settings.golden_path.is_file():
        calibration = await calibrate_judge(roles.judge, load_golden(settings.golden_path))

    mode = "stub" if settings.stub else "real"
    envelope = build_envelope(settings, scenarios, mode=mode, cache_stats=cache.stats())
    json_path, md_path = write_reports(
        settings.report_dir,
        envelope,
        metrics,
        calibration=calibration,
        pairwise=pairwise,
        dump_files=dump_files,
    )
    return RunArtifacts(
        settings=settings,
        envelope=envelope,
        metrics=metrics,
        json_path=json_path,
        md_path=md_path,
        dump_dir=dump_dir,
    )


async def run_calibration_only(settings: BenchSettings) -> tuple[Any, Any]:
    """``calibrate`` subcommand: judge-vs-golden agreement, nothing else."""
    roles = _build_roles(settings)
    cases = load_golden(settings.golden_path)
    return roles.judge, await calibrate_judge(roles.judge, cases)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_summary(artifacts: RunArtifacts) -> None:
    m = artifacts.metrics
    print(f"\n=== benchmark {artifacts.envelope.run_id} ({artifacts.envelope.mode}) ===")
    header = f"{'arm':<20} {'ratio':>7} {'probes':>8} {'judge':>7} {'facts':>7} {'fabr':>7}"
    print(header)
    print("-" * len(header))
    for arm, agg in m.arms.items():
        r = agg.avg_rubric

        def _p(v: float | None) -> str:
            return "-" if v is None else f"{v * 100:.1f}%"

        print(
            f"{arm:<20} "
            f"{(agg.avg_compression_ratio or 0):>7.3f} "
            f"{_p(agg.avg_probe_score):>8} "
            f"{_p(r.get('overall')):>7} "
            f"{_p(r.get('fact_recall')):>7} "
            f"{_p(agg.avg_fabrication_rate):>7}"
        )
    if m.robustness:
        print(f"\nrobustness: all_ok={m.robustness.all_ok} plans={m.robustness.plan_distribution}")
    if artifacts.dump_dir is not None:
        print(f"\ndumps: {artifacts.dump_dir}")
    if artifacts.json_path:
        print(f"\nreport: {artifacts.json_path}")
        print(f"        {artifacts.md_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="langcompress compression-effect benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="full pipeline: replay → evaluate → report")
    run_p.add_argument("--mode", choices=("stub", "real"), default=None,
                       help="force a mode (default: auto — stub when no model configured)")
    run_p.add_argument("--arms", nargs="*", choices=ARMS, default=None,
                       help="subset of arms to run (default: all four)")
    run_p.add_argument("--scenarios", nargs="*", default=None,
                       help="scenario ids or categories to include (default: all)")
    run_p.add_argument("--no-robustness", action="store_true", help="skip fault injection")
    run_p.add_argument("--no-pairwise", action="store_true", help="skip pairwise comparison")
    run_p.add_argument("--no-calibration", action="store_true", help="skip golden-set calibration")
    run_p.add_argument("--llm-grade", action="store_true",
                       help="enable LLM grading of probe answers (needs judge model)")
    run_p.add_argument("--summary-model", default=None, help="override BENCH_SUMMARY_MODEL")
    run_p.add_argument("--probe-model", default=None, help="override BENCH_PROBE_MODEL")
    run_p.add_argument("--judge-model", default=None, help="override BENCH_JUDGE_MODEL")
    run_p.add_argument("--trigger-messages", type=int, default=None)
    run_p.add_argument("--keep-recent", type=int, default=None)
    run_p.add_argument("--no-cache", action="store_true", help="disable the LLM response cache")
    run_p.add_argument("--no-dump", action="store_true",
                       help="disable full-text before/after dumps (default: on)")
    run_p.add_argument("--tag", default="bench", help="run tag stamped into summary-LLM metadata")

    cal_p = sub.add_parser("calibrate", help="golden-set judge calibration only")
    cal_p.add_argument("--judge-model", default=None, help="override BENCH_JUDGE_MODEL")

    sub.add_parser("list", help="list the scenario corpus")

    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "list":
        settings = load_settings()
        for s in load_corpus(settings.scenario_dir):
            print(
                f"{s.id:<28} {s.category:<24} entries={len(s.transcript):>3} "
                f"facts={len(s.facts):>2} entities={len(s.entities):>2} "
                f"probes={len(s.qa_probes):>2} hash={s.content_hash()}"
            )
        return 0

    overrides: dict[str, Any] = {}
    if command == "run":
        for key in ("summary_model", "probe_model", "judge_model"):
            value = getattr(args, key)
            if value:
                overrides[key] = value
        if args.trigger_messages is not None:
            overrides["trigger_messages"] = args.trigger_messages
        if args.keep_recent is not None:
            overrides["keep_recent"] = args.keep_recent
        if args.no_cache:
            overrides["cache_enabled"] = False
        if args.no_dump:
            overrides["dump_enabled"] = False
    elif command == "calibrate":
        if args.judge_model:
            overrides["judge_model"] = args.judge_model

    settings = load_settings(**overrides)
    if command == "run" and args.mode == "stub":
        settings = replace(settings, summary_model=None, probe_model=None, judge_model=None)
    if command == "run" and args.mode == "real" and settings.stub:
        print("error: --mode real requires at least one model "
              "(BENCH_SUMMARY_MODEL / BENCH_PROBE_MODEL / BENCH_JUDGE_MODEL)")
        return 2

    if command == "calibrate":
        if not settings.golden_path.is_file():
            print(f"error: golden set not found at {settings.golden_path}")
            return 2
        _judge, calibration = asyncio.run(run_calibration_only(settings))
        print(f"golden-set calibration ({calibration.n_cases} cases):")
        print(f"  rubric MAE: { {k: round(v, 3) for k, v in calibration.rubric_mae.items()} }")
        print(f"  fact agreement: {calibration.fact_agreement:.3f}")
        print(f"  passed: {calibration.passed}")
        return 0 if calibration.passed else 1

    artifacts = asyncio.run(
        run_benchmark(
            settings,
            arms=args.arms,
            scenario_filter=args.scenarios,
            with_robustness=not args.no_robustness,
            with_pairwise=not args.no_pairwise,
            with_calibration=not args.no_calibration,
            use_llm_grading=args.llm_grade,
            run_tag=args.tag,
        )
    )
    _print_summary(artifacts)
    return 0 if all(agg.n_errors == 0 for agg in artifacts.metrics.arms.values()) else 1


__all__ = [
    "RunArtifacts",
    "main",
    "render_scenario_text",
    "run_benchmark",
    "run_calibration_only",
]
