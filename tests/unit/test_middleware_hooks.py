"""Unit tests for :class:`langcompress.CompressionMiddleware`'s four
extension-point hooks (design §11.1) — the deterministic core evidence.

Calls ``before_model`` directly with a recording stub model and a ``MagicMock``
runtime (the parent never touches the runtime), asserting each hook is wired
into the compression pipeline.
"""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langcompress import CompressionConfig
from langcompress.middleware import CompressionMiddleware, _merge_runnable_config


class _RecordingStubModel:
    """Minimal chat-model stand-in.

    Records the ``config`` passed to ``invoke``/``ainvoke`` and returns an
    :class:`AIMessage` whose ``.text`` is the stub summary. Carries no
    ``profile``/``_llm_type`` — fine because a custom token counter bypasses
    the parent's model-introspection branches.
    """

    def __init__(self, summary_text="STUB SUMMARY"):
        self.summary_text = summary_text
        self.last_config = None
        self.invoke_count = 0

    def invoke(self, prompt, config=None):
        self.invoke_count += 1
        self.last_config = config
        return AIMessage(content=self.summary_text)

    async def ainvoke(self, prompt, config=None):
        return self.invoke(prompt, config=config)


def _simple_token_counter(messages):
    """Deterministic counter; enough for the messages-trigger path."""
    return 999


def _make_mw(**overrides):
    base = {
        "summary_model": _RecordingStubModel(),
        "token_threshold": [("messages", 2)],
        "keep_recent": 1,
        "token_counter": _simple_token_counter,
        "l0_enabled": False,  # hook tests focus on L3, not L0 cleanup
    }
    base.update(overrides)
    return CompressionMiddleware(CompressionConfig(**base))


def _state(*messages):
    return {"messages": list(messages)}


# --------------------------------------------------------------------------- #
# Hook 1 — summary message construction
# --------------------------------------------------------------------------- #


def test_hook1_custom_summary_message_builder():
    def builder(summary):
        return SystemMessage(content=f"[SUM] {summary}")

    mw = _make_mw(summary_message_builder=builder)
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is not None
    sys_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(sys_msgs) == 1
    assert sys_msgs[0].content == "[SUM] STUB SUMMARY"


# --------------------------------------------------------------------------- #
# Hook 2 — summary LLM config merge
# --------------------------------------------------------------------------- #


def test_hook2_summary_llm_config_merged():
    model = _RecordingStubModel()
    mw = _make_mw(summary_model=model)
    mw.config.summary_llm_config_provider = lambda: {
        "metadata": {"request_id": "abc"},
        "tags": ["compress"],
    }
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is not None
    assert model.invoke_count == 1
    cfg = model.last_config
    # base metadata preserved (deep merge)
    assert cfg["metadata"]["lc_source"] == "summarization"
    # provider metadata merged in (provider wins on collision)
    assert cfg["metadata"]["request_id"] == "abc"
    # non-metadata keys shallow-merged
    assert cfg["tags"] == ["compress"]


def test_merge_runnable_config_deep_merges_metadata():
    merged = _merge_runnable_config(
        {"metadata": {"lc_source": "summarization", "a": 1}},
        {"metadata": {"a": 2, "b": 3}, "tags": ["x"]},
    )
    assert merged["metadata"] == {"lc_source": "summarization", "a": 2, "b": 3}
    assert merged["tags"] == ["x"]


def test_merge_runnable_config_empty_override_keeps_base():
    merged = _merge_runnable_config({"metadata": {"lc_source": "s"}}, {})
    assert merged == {"metadata": {"lc_source": "s"}}


# --------------------------------------------------------------------------- #
# Hook 3 — post-compression result transform
# --------------------------------------------------------------------------- #


def test_hook3_post_compress_mutates_result():
    def post(state, result):
        out = dict(result)
        out["compression_count"] = 1
        return out

    mw = _make_mw(post_compress_hook=post)
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is not None
    assert result.get("compression_count") == 1
    assert "messages" in result  # original payload preserved


def test_hook3_degrades_gracefully_on_exception():
    def bad(state, result):
        raise RuntimeError("boom")

    mw = _make_mw(post_compress_hook=bad)
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    # A broken post-hook must not break compression: return the original result.
    assert result is not None
    assert "messages" in result
    assert any(m.additional_kwargs.get("__summarization__") for m in result["messages"])


# --------------------------------------------------------------------------- #
# Hook 4 — trigger decision
# --------------------------------------------------------------------------- #


def test_hook4_force_trigger_compresses_below_threshold():
    # Base trigger needs 10 messages; we pass 3, so base=False. Hook forces True.
    mw = CompressionMiddleware(
        CompressionConfig(
            summary_model=_RecordingStubModel(),
            token_threshold=[("messages", 10)],
            keep_recent=1,
            token_counter=_simple_token_counter,
            should_summarize_hook=lambda msgs, toks, base: True,
            l0_enabled=False,  # hook test, not L0
        )
    )
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is not None
    assert any(m.additional_kwargs.get("__summarization__") for m in result["messages"])


def test_hook4_force_false_blocks_compression():
    # Base trigger would fire (3 >= 2); hook forces False.
    mw = CompressionMiddleware(
        CompressionConfig(
            summary_model=_RecordingStubModel(),
            token_threshold=[("messages", 2)],
            keep_recent=1,
            token_counter=_simple_token_counter,
            should_summarize_hook=lambda msgs, toks, base: False,
            l0_enabled=False,  # hook test, not L0
        )
    )
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is None


# --------------------------------------------------------------------------- #
# Default behaviour — summary message carries the flag + prefix
# --------------------------------------------------------------------------- #


def test_default_summary_message_has_flag_and_prefix():
    mw = _make_mw()  # all hooks at defaults
    result = mw.before_model(
        _state(HumanMessage("m0"), HumanMessage("m1"), HumanMessage("m2")), MagicMock()
    )
    assert result is not None
    summary_msgs = [
        m for m in result["messages"] if m.additional_kwargs.get("__summarization__")
    ]
    assert len(summary_msgs) == 1
    assert summary_msgs[0].content.startswith("Here is a summary of the conversation to date:")
    assert summary_msgs[0].additional_kwargs.get("lc_source") == "summarization"


def test_no_compression_when_below_threshold():
    mw = _make_mw()  # trigger needs 2 messages
    result = mw.before_model(_state(HumanMessage("only-one")), MagicMock())
    assert result is None  # 1 < 2 → no summarization
