"""Unit tests for :class:`langcompress.HeuristicQualityValidator` and the
:class:`langcompress.QualityValidator` ABC (design §8.2).

The default validator is conservative on purpose — with its default knobs it
only flags clear failures (empty, the ``"Error generating summary:"`` prefix
produced on LLM exception, below-min length), so the quality gate is a no-op
for every well-formed stub summary the v0.2 scenarios produce. These tests pin
that no-regression contract and the opt-in stricter knobs
(``min_reduction_ratio`` / ``require_segments``).
"""
from __future__ import annotations

import inspect

import pytest
from langchain_core.messages import HumanMessage

from langcompress import (
    HeuristicQualityValidator,
    QualityValidator,
    ValidationResult,
)


def _msgs(*contents: str) -> list[HumanMessage]:
    return [HumanMessage(content=c) for c in contents]


# --------------------------------------------------------------------------- #
# No-regression: every well-formed stub summary the v0.2 scenarios produce
# passes the default validator.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("summary", ["SUMMARY", "MY SUMMARY", "COMPRESSED SUMMARY", "STUB SUMMARY"])
def test_well_formed_stub_summaries_pass_with_defaults(summary: str) -> None:
    result = HeuristicQualityValidator().validate(summary, _msgs("a", "b"))
    assert result.passed is True
    assert result.reason == ""
    assert result.suggested_plan is None


def test_validation_result_defaults() -> None:
    r = ValidationResult(passed=True)
    assert r.passed is True
    assert r.reason == ""
    assert r.suggested_plan is None
    # frozen dataclass
    with pytest.raises((AttributeError, Exception)):
        r.passed = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Clear failures (run with all defaults)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "   ", "\n\t  "])
def test_empty_summary_fails_and_suggests_truncation(bad: str) -> None:
    result = HeuristicQualityValidator().validate(bad, _msgs("a", "b"))
    assert result.passed is False
    assert result.reason == "empty summary"
    assert result.suggested_plan == "C"


def test_llm_error_prefix_fails_and_suggests_retry() -> None:
    # The v0.2 latent-bug fix point: the middleware used to ship this string as
    # the summary. The validator now flags it for Plan-A retry.
    result = HeuristicQualityValidator().validate(
        "Error generating summary: simulated LLM failure", _msgs("a", "b")
    )
    assert result.passed is False
    assert result.reason == "llm error string"
    assert result.suggested_plan == "A"


def test_below_min_length_fails_and_suggests_retry() -> None:
    result = HeuristicQualityValidator().validate("ab", _msgs("a", "b"))  # len 2 < 5
    assert result.passed is False
    assert result.reason == "summary too short"
    assert result.suggested_plan == "A"


def test_custom_min_length_boundary() -> None:
    v = HeuristicQualityValidator(min_length=10)
    assert v.validate("short", _msgs("a")).passed is False  # len 5 < 10
    assert v.validate("long enough", _msgs("a")).passed is True  # len 11 >= 10


# --------------------------------------------------------------------------- #
# Opt-in knob: min_reduction_ratio (Plan B hint)
# --------------------------------------------------------------------------- #


def _fixed_counter(n: int):
    def _count(_messages) -> int:
        return n

    return _count


def test_reduction_ratio_passes_when_summary_compresses() -> None:
    # src tokens = 1000, summary len 10 → ratio = 1 - 10/1000 = 0.99 >= 0.5.
    v = HeuristicQualityValidator(min_reduction_ratio=0.5, token_counter=_fixed_counter(1000))
    result = v.validate("x" * 10, _msgs("a", "b"))
    assert result.passed is True


def test_reduction_ratio_fails_when_summary_barely_shrinks() -> None:
    # src tokens = 1000, summary len 990 → ratio = 1 - 990/1000 = 0.01 < 0.5.
    v = HeuristicQualityValidator(min_reduction_ratio=0.5, token_counter=_fixed_counter(1000))
    result = v.validate("x" * 990, _msgs("a", "b"))
    assert result.passed is False
    assert result.reason == "insufficient reduction"
    assert result.suggested_plan == "B"


def test_reduction_ratio_off_by_default() -> None:
    # A 990-char "summary" of a 1000-token source passes when the knob is off.
    v = HeuristicQualityValidator(token_counter=_fixed_counter(1000))
    assert v.validate("x" * 990, _msgs("a", "b")).passed is True


# --------------------------------------------------------------------------- #
# Opt-in knob: require_segments (Plan A hint)
# --------------------------------------------------------------------------- #


_FULL_SUMMARY = (
    "## 1. Primary Request and Intent\nthe goal\n"
    "## 8. Entity State\nstate here"
)


def test_require_segments_passes_with_both_markers() -> None:
    v = HeuristicQualityValidator(require_segments=True)
    assert v.validate(_FULL_SUMMARY, _msgs("a", "b")).passed is True


def test_require_segments_fails_when_marker_missing() -> None:
    v = HeuristicQualityValidator(require_segments=True)
    result = v.validate("## 1. Primary Request and Intent\nonly first", _msgs("a", "b"))
    assert result.passed is False
    assert result.reason == "missing segments"
    assert result.suggested_plan == "A"


def test_require_segments_off_by_default() -> None:
    assert HeuristicQualityValidator().validate("no markers at all", _msgs("a")).passed is True


# --------------------------------------------------------------------------- #
# ABC contract + async delegation
# --------------------------------------------------------------------------- #


def test_quality_validator_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        QualityValidator()  # type: ignore[abstract]


def test_avalidate_is_a_coroutine_function() -> None:
    # HeuristicQualityValidator.avalidate is a real coroutine fn (so an
    # LLM-as-judge subclass can override it with awaitable I/O).
    v = HeuristicQualityValidator()
    assert inspect.iscoroutinefunction(v.avalidate)


async def test_avalidate_returns_same_result_as_sync() -> None:
    v = HeuristicQualityValidator()
    expected = v.validate("Error generating summary: x", _msgs("a"))
    got = await v.avalidate("Error generating summary: x", _msgs("a"))
    assert got == expected


# --------------------------------------------------------------------------- #
# Custom validator subclass is usable by the middleware
# --------------------------------------------------------------------------- #


class _AlwaysFailValidator(QualityValidator):
    """A host-supplied validator that always demands Plan B."""

    def validate(self, summary: str, summarized_messages) -> ValidationResult:
        return ValidationResult(False, "host policy: always degrade", "B")


def test_custom_validator_subclass_overrides_default_behaviour() -> None:
    v = _AlwaysFailValidator()
    result = v.validate("perfectly fine summary", _msgs("a", "b"))
    assert result.passed is False
    assert result.suggested_plan == "B"
