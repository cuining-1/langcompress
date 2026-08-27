"""Stub-mode end-to-end smoke test of the benchmark pipeline.

Drives the real ``run_benchmark`` orchestration (replay → probes → judge →
fault suite → pairwise → calibration → dual-format reports) in forced stub
mode over one corpus scenario, with reports and cache redirected to a
temporary directory. This is the CI counterpart of
``python -m benchmarks run --mode stub``: keyless, deterministic, seconds.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.config import BenchSettings, load_settings
from benchmarks.runner import RunArtifacts, main, run_benchmark
from benchmarks.scenario import load_corpus

SCENARIO_ID = "cross_compression_001"
ALL_ARMS = {"langcompress", "bare_summarization", "trim", "full_context"}

CN_SCENARIO_IDS = (
    "long_multi_topic_drift_cn_001",
    "tool_json_heavy_cn_001",
    "error_fix_loop_cn_001",
    "entity_tracking_cn_001",
    "cross_compression_cn_001",
    "thinking_heavy_cn_001",
)

MD_CORE_SECTIONS = (
    "# langcompress compression benchmark report",
    "## Headline",
    "## Efficiency",
    "## Fidelity",
    "## Consistency",
    "## Robustness",
    "## Judge calibration",
    "## Pairwise",
    "## Reproduce",
)


def _stub_settings(report_dir: Path, cache_dir: Path) -> BenchSettings:
    """Force keyless stub mode regardless of any ``.env`` / env model config."""
    settings = load_settings()
    return replace(
        settings,
        summary_model=None,
        probe_model=None,
        judge_model=None,
        report_dir=report_dir,
        cache_path=cache_dir / "llm_cache.jsonl",
        dump_enabled=False,  # per-test dumps opt back in explicitly
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> RunArtifacts:
    tmp = tmp_path_factory.mktemp("bench_smoke")
    settings = _stub_settings(tmp / "reports", tmp / "cache")
    return asyncio.run(run_benchmark(settings, scenario_filter=[SCENARIO_ID]))


def test_reports_written_both_formats(artifacts):
    assert artifacts.json_path is not None and artifacts.json_path.is_file()
    assert artifacts.md_path is not None and artifacts.md_path.is_file()


def test_json_envelope_is_reproducibility_recipe(artifacts):
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    envelope = payload["envelope"]
    assert envelope["mode"] == "stub"
    assert SCENARIO_ID in envelope["scenario_hashes"]
    assert envelope["settings"]["summary_model"] is None  # forced-stub snapshot
    assert envelope["settings"]["temperature"] == 0.0
    # calibration + pairwise sections present and healthy in stub mode
    assert payload["calibration"] and payload["calibration"]["passed"]
    assert payload["pairwise"], "objective pairwise should run in stub mode"


def test_all_four_arms_scored_without_errors(artifacts):
    assert set(artifacts.metrics.arms) == ALL_ARMS
    for arm, agg in artifacts.metrics.arms.items():
        assert agg.n_errors == 0, f"arm {arm} reported replay errors"


def test_full_context_is_the_sanity_anchor(artifacts):
    agg = artifacts.metrics.arms["full_context"]
    # Any probe miss on full_context is a corpus bug, not a compression cost.
    assert agg.avg_probe_score == 1.0
    assert agg.avg_compression_ratio == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("scenario_id", CN_SCENARIO_IDS)
async def test_chinese_scenario_full_context_anchor(scenario_id, tmp_path):
    """Corpus self-check for the Chinese half: every scenario must score
    100% probes on full_context — keyword-verbatim mistakes surface here
    (as all_retained failures) instead of polluting compression findings."""
    settings = _stub_settings(tmp_path / "reports", tmp_path / "cache")
    artifacts = await run_benchmark(
        settings,
        arms=["full_context"],
        scenario_filter=[scenario_id],
        with_robustness=False,
        with_pairwise=False,
        with_calibration=False,
    )
    agg = artifacts.metrics.arms["full_context"]
    assert agg.n_errors == 0
    assert agg.avg_probe_score == 1.0


def test_langcompress_compresses_and_gets_judged(artifacts):
    agg = artifacts.metrics.arms["langcompress"]
    assert agg.avg_compression_ratio < 1.0
    assert agg.total_summary_calls > 0
    assert "overall" in agg.avg_rubric  # heuristic judge produced a rubric


def test_robustness_fault_suite_passes(artifacts):
    robustness = artifacts.metrics.robustness
    assert robustness is not None
    assert robustness.all_ok, [o.problems for o in robustness.outcomes if o.problems]
    assert robustness.plan_distribution, "no degradation plan fired at all"


def test_markdown_renders_core_sections(artifacts):
    md = artifacts.md_path.read_text(encoding="utf-8")
    for section in MD_CORE_SECTIONS:
        assert section in md, f"missing report section {section!r}"


def test_stub_mode_is_deterministic(tmp_path):
    """Two runs with identical settings must produce identical numbers.

    Only timing fields are excluded — everything the report is *about*
    (ratios, rubrics, probe scores, event counts, token metering, plan
    distribution) must repeat exactly in stub mode."""

    def signature(run: RunArtifacts):
        return [
            (
                m.scenario_id,
                m.arm,
                m.compression_ratio,
                m.event_counts,
                m.rubric,
                m.probe_score,
                m.fabrication_rate,
                m.summary_calls,
                m.summary_input_tokens,
                m.summary_output_tokens,
            )
            for m in run.metrics.scenarios
        ]

    first = asyncio.run(
        run_benchmark(
            _stub_settings(tmp_path / "r1", tmp_path / "c1"), scenario_filter=[SCENARIO_ID]
        )
    )
    second = asyncio.run(
        run_benchmark(
            _stub_settings(tmp_path / "r2", tmp_path / "c2"), scenario_filter=[SCENARIO_ID]
        )
    )
    assert signature(first) == signature(second)
    assert (
        first.metrics.robustness.plan_distribution
        == second.metrics.robustness.plan_distribution
    )


def test_cli_run_stub_writes_reports(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BENCH_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("BENCH_CACHE_DIR", str(tmp_path / "cache" / "llm_cache.jsonl"))
    monkeypatch.setenv("BENCH_DUMP_DIR", str(tmp_path / "dumps"))
    assert main(["run", "--mode", "stub", "--scenarios", SCENARIO_ID]) == 0
    out = capsys.readouterr().out
    assert "langcompress" in out
    assert "robustness: all_ok=True" in out
    assert list((tmp_path / "reports").glob("bench_*.json"))


DUMP_SECTIONS = (
    "# Compression dump",
    "## Original conversation (",
    "## Ground truth — facts to preserve",
    "## Ground truth — entities",
    "## Event #",
    "### BEFORE (what the agent had)",
    "### AFTER (what replaced it)",
    "### Summary text",
    "## Final context (as the probe answerer sees it)",
)


async def test_dump_written_per_scenario_and_arm(tmp_path):
    """Full-text before/after dumps: one file per (scenario × arm), carrying
    every compression event's complete message lists plus the final context —
    the artifacts a human/AI reviewer reads to judge the compression."""
    settings = replace(
        _stub_settings(tmp_path / "reports", tmp_path / "cache"),
        dump_enabled=True,
        dump_dir=tmp_path / "dumps",
    )
    artifacts = await run_benchmark(
        settings,
        arms=["langcompress", "full_context"],
        scenario_filter=[SCENARIO_ID],
        with_robustness=False,
        with_pairwise=False,
        with_calibration=False,
    )
    assert artifacts.dump_dir is not None and artifacts.dump_dir.is_dir()
    # exactly one file per (scenario × arm), including the no-compression arm
    names = {p.name for p in artifacts.dump_dir.iterdir()}
    assert names == {f"{SCENARIO_ID}__langcompress.md", f"{SCENARIO_ID}__full_context.md"}

    compressed = artifacts.dump_dir / f"{SCENARIO_ID}__langcompress.md"
    text = compressed.read_text(encoding="utf-8")
    for section in DUMP_SECTIONS[:-1]:  # events + summary always present here
        assert section in text, f"missing dump section {section!r}"
    # every scenario fact appears verbatim in the ground-truth checklist
    for fact in next(s for s in load_corpus() if s.id == SCENARIO_ID).facts:
        assert fact.text in text, f"fact {fact.id} missing from dump checklist"
    # the full_context arm compresses nothing — only the final context exists
    full = artifacts.dump_dir / f"{SCENARIO_ID}__full_context.md"
    full_text = full.read_text(encoding="utf-8")
    assert DUMP_SECTIONS[-1] in full_text
    assert "## Event #" not in full_text


def test_dump_files_listed_in_report(tmp_path):
    """The report's Dumps section points reviewers at the dump files, so the
    number → full-text review path is discoverable from the report alone."""
    settings = replace(
        _stub_settings(tmp_path / "reports", tmp_path / "cache"),
        dump_enabled=True,
        dump_dir=tmp_path / "dumps",
    )
    artifacts = asyncio.run(
        run_benchmark(
            settings,
            arms=["langcompress"],
            scenario_filter=[SCENARIO_ID],
            with_robustness=False,
            with_pairwise=False,
            with_calibration=False,
        )
    )
    md = artifacts.md_path.read_text(encoding="utf-8")
    assert "## Dumps (full-text before/after" in md
    assert f"{SCENARIO_ID}__langcompress.md" in md
