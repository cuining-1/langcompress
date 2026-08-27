"""Unit tests for :class:`langcompress.CompressionConfig` (design §6 / §11.2).

Verifies the ``token_threshold`` → parent ``trigger`` translation, the
``LANGCOMPRESS_*`` env-var reading (which must not clobber explicit values),
and that all five default hooks are callable and behave as documented.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langcompress import CompressionConfig
from langcompress.config import (
    _default_build_summary_message,
    _default_content_classifier,
    _default_get_summary_llm_config,
    _default_post_compress,
    _default_should_summarize,
)


class _Stub:
    """Minimal stand-in for a chat model (only used as a non-str placeholder)."""


# --------------------------------------------------------------------------- #
# _as_parent_trigger type mapping (design §6)
# --------------------------------------------------------------------------- #


def test_trigger_none_disables_summarization():
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=None)
    assert cfg._as_parent_trigger() is None


def test_trigger_float_maps_to_fraction():
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=0.8)
    assert cfg._as_parent_trigger() == [("fraction", 0.8)]


def test_trigger_int_maps_to_tokens():
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=3000)
    assert cfg._as_parent_trigger() == [("tokens", 3000)]


def test_trigger_list_passthrough():
    trig = [("tokens", 100), ("messages", 50)]
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=trig)
    assert cfg._as_parent_trigger() == [("tokens", 100), ("messages", 50)]


def test_trigger_tuple_passthrough():
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=("messages", 10))
    assert cfg._as_parent_trigger() == [("messages", 10)]


def test_trigger_bool_rejected():
    # Bypass field validation so the defensive isinstance(v, bool) branch fires.
    cfg = CompressionConfig(summary_model=_Stub())
    cfg.token_threshold = True
    with pytest.raises(TypeError):
        cfg._as_parent_trigger()


# --------------------------------------------------------------------------- #
# env-var reading (design §6.4) — hooks are never sourced from env
# --------------------------------------------------------------------------- #


def test_env_token_threshold(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_TOKEN_THRESHOLD", "5000")
    cfg = CompressionConfig(summary_model=_Stub())
    assert cfg.token_threshold == 5000
    assert cfg._as_parent_trigger() == [("tokens", 5000)]


def test_env_token_threshold_float(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_TOKEN_THRESHOLD", "0.75")
    cfg = CompressionConfig(summary_model=_Stub())
    assert cfg.token_threshold == 0.75
    assert cfg._as_parent_trigger() == [("fraction", 0.75)]


def test_env_keep_recent(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_KEEP_RECENT", "7")
    cfg = CompressionConfig(summary_model=_Stub())
    assert cfg.keep_recent == 7


def test_env_trim_tokens_to_summarize(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_TRIM_TOKENS_TO_SUMMARIZE", "8000")
    cfg = CompressionConfig(summary_model=_Stub())
    assert cfg.trim_tokens_to_summarize == 8000


def test_explicit_value_not_overridden_by_env(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_TOKEN_THRESHOLD", "5000")
    monkeypatch.setenv("LANGCOMPRESS_KEEP_RECENT", "99")
    cfg = CompressionConfig(summary_model=_Stub(), token_threshold=0.5, keep_recent=3)
    assert cfg.token_threshold == 0.5
    assert cfg._as_parent_trigger() == [("fraction", 0.5)]
    assert cfg.keep_recent == 3


# --------------------------------------------------------------------------- #
# default hooks (design §11.2 — defaults equal parent behaviour + flag)
# --------------------------------------------------------------------------- #


def test_default_hooks_callable_and_behave():
    # Hook 1 — summary message carries the prefix + __summarization__ flag
    msg = _default_build_summary_message("hello world")
    assert msg.content.startswith("Here is a summary of the conversation to date:")
    assert msg.content.endswith("hello world")
    assert msg.additional_kwargs.get("__summarization__") is True
    assert msg.additional_kwargs.get("lc_source") == "summarization"

    # Hook 2 — empty config (merged with parent's lc_source metadata at call site)
    assert _default_get_summary_llm_config() == {}

    # Hook 3 — identity
    assert _default_post_compress({"x": 1}, {"y": 2}) == {"y": 2}

    # Hook 4 — honours the parent's multi-signal decision
    assert _default_should_summarize([], 0, True) is True
    assert _default_should_summarize([], 0, False) is False


def test_default_content_classifier_labels():
    assert _default_content_classifier(SystemMessage(content="s")) == "system"
    assert _default_content_classifier(HumanMessage(content="h")) == "user"
    assert _default_content_classifier(AIMessage(content="a")) == "assistant"
    assert (
        _default_content_classifier(ToolMessage(content="t", tool_call_id="x"))
        == "tool_result"
    )


def test_default_content_classifier_large_suffix():
    big = HumanMessage(content="x" * 9000)
    assert _default_content_classifier(big) == "user:large"
    small = HumanMessage(content="x")
    assert _default_content_classifier(small) == "user"


def test_config_accepts_custom_hooks():
    def builder(s):
        return SystemMessage(content=s)

    cfg = CompressionConfig(summary_model=_Stub(), summary_message_builder=builder)
    assert cfg.summary_message_builder("z").content == "z"
