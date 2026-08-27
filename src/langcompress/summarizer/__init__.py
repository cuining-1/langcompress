"""Summarizer abstractions, quality validation, and the summary templates
(design §7 / §8.2)."""
from langcompress.summarizer.base import Summarizer
from langcompress.summarizer.llm_summarizer import LLMSummarizer
from langcompress.summarizer.quality import (
    HeuristicQualityValidator,
    QualityValidator,
    ValidationResult,
)
from langcompress.summarizer.templates import (
    DEFAULT_SUMMARY_PROMPT,
    EIGHT_SEGMENT_TEMPLATE,
    FALLBACK_SUMMARY_PROMPT,
)

__all__ = [
    "DEFAULT_SUMMARY_PROMPT",
    "EIGHT_SEGMENT_TEMPLATE",
    "FALLBACK_SUMMARY_PROMPT",
    "HeuristicQualityValidator",
    "LLMSummarizer",
    "QualityValidator",
    "Summarizer",
    "ValidationResult",
]
