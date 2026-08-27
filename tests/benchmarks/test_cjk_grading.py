"""CJK (Chinese) grading support — keyword hits and fact-recall bigrams.

Half the corpus is Chinese; these tests pin the two grading primitives
that make that possible (plain-substring keywords for CJK, character
bigrams for fact containment), so an ASCII-only regression cannot
silently turn every Chinese probe into a false miss.
"""
from __future__ import annotations

from benchmarks.judge import HeuristicJudge, content_words
from benchmarks.probes import keyword_hit
from benchmarks.scenario import Fact

FACT = Fact(id="F1", text="限流阈值最终定为每分钟 200 次")


class TestKeywordHitCJK:
    def test_cjk_keyword_matches_inside_compound(self):
        # No \b boundary exists between Chinese chars — must be a hit.
        assert keyword_hit("支付网关超时是根因", "网关")

    def test_cjk_keyword_miss_when_absent(self):
        assert not keyword_hit("今天天气不错", "网关")

    def test_cjk_with_attached_digits(self):
        # "300秒" is all \w chars, yet must not require a \b boundary.
        assert keyword_hit("容差扩大到 300秒 了", "300秒")

    def test_ascii_bare_word_still_boundary_anchored(self):
        # English/numeric behavior unchanged by the CJK path.
        assert keyword_hit("value is 500 now", "500")
        assert not keyword_hit("value is 1500 now", "500")

    def test_ascii_word_inside_cjk_text(self):
        # ASCII keyword against Chinese haystack keeps \b semantics.
        assert keyword_hit("选用 W-TinyLFU 策略", "tinylfu")


class TestContentWordsCJK:
    def test_extracts_bigrams_and_numbers(self):
        words = content_words("限流阈值 200 次")
        assert "200" in words
        assert "限流" in words and "阈值" in words

    def test_single_cjk_char_run_yields_nothing(self):
        # "次" alone is too weak to grade on — must not enter the list.
        assert content_words("阈值 次") == content_words("阈值")

    def test_english_unchanged(self):
        words = content_words("the timeout is 30 seconds")
        assert words == ["timeout", "seconds", "30"]  # words first, numbers appended


class TestHeuristicJudgeCJK:
    async def test_fully_carried_chinese_fact_is_preserved(self):
        verdict = await HeuristicJudge().judge(
            "限流阈值最终定为每分钟 200 次，已生效。",
            facts=[FACT],
            entities=[],
            source_text="原文",
        )
        assert verdict.fact_verdicts["F1"] == "preserved"

    async def test_unrelated_summary_marks_fact_lost(self):
        verdict = await HeuristicJudge().judge(
            "今天讨论了缓存分片方案。",
            facts=[FACT],
            entities=[],
            source_text="原文",
        )
        assert verdict.fact_verdicts["F1"] == "lost"
