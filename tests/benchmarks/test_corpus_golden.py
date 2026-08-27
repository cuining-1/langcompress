"""Benchmark data-layer smoke tests: scenario corpus + golden set.

The effect-numbers are only as good as the data they are computed from.
These tests fail loudly when a scenario file breaks structural lint, when
a content hash stops covering the ground truth, or when the heuristic
judge can no longer reproduce the human-annotated golden set — each of
which would otherwise silently corrupt every downstream metric.
"""
from __future__ import annotations

import pytest

from benchmarks.config import BENCH_ROOT
from benchmarks.judge import HeuristicJudge, calibrate_judge, load_golden
from benchmarks.scenario import Scenario, load_corpus, load_scenario

EXPECTED_CATEGORIES = {
    "long_multi_topic_drift",
    "tool_json_heavy",
    "error_fix_loop",
    "entity_tracking",
    "cross_compression",
    "thinking_heavy",
}

# Every corpus scenario must exceed the default 24-message trigger, or the
# benchmark would score arms that never actually compress.
MIN_TRANSCRIPT_ENTRIES = 24


@pytest.fixture(scope="module")
def corpus() -> list[Scenario]:
    # load_corpus raises ValueError when any scenario fails structural lint
    return load_corpus()


class TestCorpus:
    def test_loads_clean_with_all_six_categories(self, corpus):
        assert len(corpus) >= 6
        assert {s.category for s in corpus} == EXPECTED_CATEGORIES

    def test_scenario_ids_unique(self, corpus):
        ids = [s.id for s in corpus]
        assert len(ids) == len(set(ids))

    def test_chinese_corpus_covers_all_categories(self, corpus):
        """The corpus is bilingual: every category must carry at least one
        Chinese scenario (ids tagged ``_cn_``) so template/param evolutions
        are validated against CJK grading too."""
        cn_categories = {s.category for s in corpus if "_cn_" in s.id}
        assert cn_categories == EXPECTED_CATEGORIES

    def test_long_enough_to_trigger_compression(self, corpus):
        for s in corpus:
            assert len(s.transcript) >= MIN_TRANSCRIPT_ENTRIES, s.id

    def test_ground_truth_present(self, corpus):
        for s in corpus:
            assert s.facts, f"{s.id}: no facts — fidelity would be unmeasurable"
            assert s.qa_probes, f"{s.id}: no probes — consistency would be unmeasurable"

    def test_content_hash_stable_across_loads(self, corpus):
        by_id = {s.id: s.content_hash() for s in corpus}
        for s in corpus:
            path = BENCH_ROOT / "scenarios" / f"{s.id}.json"
            assert load_scenario(path).content_hash() == by_id[s.id]

    def test_content_hash_covers_ground_truth(self, corpus):
        """The hash must digest facts/entities/probes, not just the transcript
        — otherwise two corpora with identical transcripts but different
        ground truth would silently compare as "the same corpus"."""
        s = next((x for x in corpus if x.facts and x.entities and x.qa_probes), None)
        assert s is not None, "no scenario carries facts+entities+probes"
        assert s.model_copy(update={"facts": []}).content_hash() != s.content_hash()
        assert s.model_copy(update={"entities": []}).content_hash() != s.content_hash()
        assert s.model_copy(update={"qa_probes": []}).content_hash() != s.content_hash()


@pytest.fixture(scope="module")
def golden():
    path = BENCH_ROOT / "golden" / "golden_set.json"
    assert path.is_file()
    return load_golden(path)


class TestGoldenSet:
    async def test_heuristic_judge_passes_calibration(self, golden):
        report = await calibrate_judge(HeuristicJudge(), golden)
        assert report.n_cases >= 4
        assert report.passed, report.as_dict()

    async def test_fabricated_number_is_caught(self, golden):
        case = next(c for c in golden if c.id == "gold-fabricated-number")
        verdict = await HeuristicJudge().judge(
            case.summary,
            facts=case.facts,
            entities=case.entities,
            source_text=case.source_text,
        )
        assert "42" in verdict.fabrications
        assert verdict.fabrication_rate > 0.0


def test_cli_lists_corpus(capsys):
    from benchmarks.runner import main

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for s in load_corpus():
        assert s.id in out
