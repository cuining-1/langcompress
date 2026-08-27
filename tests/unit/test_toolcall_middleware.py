"""Unit tests for :class:`langcompress.ToolCallExternalizerMiddleware` (L2 source
compression skeleton, design §4.3/§12.3) plus an end-to-end composition case.

Direct-call cases construct a ``ToolCallRequest`` + stub handler (bypassing the
LangGraph ToolNode runtime injection) to assert the skeleton's externalize /
pass-through / preserve-identity behaviour. The final case wires the middleware
into ``create_agent`` alongside ``CompressionMiddleware`` to prove central L3 +
source L2 composition.
"""
from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest

from langcompress import (
    CompressionConfig,
    CompressionMiddleware,
    FilesystemExternalizer,
    ToolCallExternalizerMiddleware,
    aggregate_external_refs,
)


def _request(tool_call=None):
    return ToolCallRequest(
        tool_call=tool_call or {"name": "big", "args": {}, "id": "c1", "type": "tool_call"},
        tool=None,
        state={"messages": []},
        runtime=None,
    )


def _stub_handler(result):
    def _h(_request):
        return result

    return _h


def _big_result(content="X" * 3000, name="big", tool_call_id="c1"):
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)


def _small_result(content="ok", name="small", tool_call_id="c2"):
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)


# --------------------------------------------------------------------------- #
# Direct-call cases
# --------------------------------------------------------------------------- #


def test_externalizes_large_result(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    original = _big_result(content="Y" * 3000, name="big", tool_call_id="c1")

    out = mw.wrap_tool_call(_request(), _stub_handler(original))

    assert isinstance(out, ToolMessage)
    assert out.tool_call_id == "c1"  # preserved
    assert out.name == "big"  # preserved
    ref = out.additional_kwargs.get("external_ref")
    assert ref is not None
    assert "Ref:" in str(out.content)
    # The full content is recoverable from the externalizer.
    assert ext.retrieve(ref) == "Y" * 3000


def test_passes_through_small_result(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    small = _small_result()

    out = mw.wrap_tool_call(_request(), _stub_handler(small))

    assert out is small  # untouched, same object
    assert "external_ref" not in (out.additional_kwargs or {})


def test_passes_through_non_toolmessage(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    other = {"not": "a tool message"}

    out = mw.wrap_tool_call(_request(), _stub_handler(other))

    assert out is other  # non-ToolMessage results pass through verbatim


def test_custom_should_externalize_overrides_default(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    # Force externalization even for the small result.
    mw = ToolCallExternalizerMiddleware(ext, should_externalize=lambda tc, r: True)
    small = _small_result(content="tiny")

    out = mw.wrap_tool_call(_request(), _stub_handler(small))

    assert out.additional_kwargs.get("external_ref") is not None
    assert ext.retrieve(out.additional_kwargs["external_ref"]) == "tiny"


def test_custom_build_reference_returns_str_wrapped_in_toolmessage(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(
        ext,
        build_reference=lambda tc, r, ref: f"[[ref={ref}]]",
    )

    out = mw.wrap_tool_call(_request(), _stub_handler(_big_result()))

    assert isinstance(out, ToolMessage)
    ref = out.additional_kwargs["external_ref"]
    assert out.content == f"[[ref={ref}]]"  # custom str wrapped verbatim


def test_threshold_boundary(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext, threshold=10)
    # 11 chars > 10 → externalized
    out = mw.wrap_tool_call(_request(), _stub_handler(ToolMessage(content="x" * 11, tool_call_id="c", name="t")))
    assert out.additional_kwargs.get("external_ref") is not None
    # 10 chars not > 10 → pass-through
    out2 = mw.wrap_tool_call(_request(), _stub_handler(ToolMessage(content="x" * 10, tool_call_id="c", name="t")))
    assert out2.additional_kwargs.get("external_ref") is None


async def test_async_path_externalizes(tmp_path):
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    mw = ToolCallExternalizerMiddleware(ext)
    original = _big_result(content="Z" * 2500, name="big", tool_call_id="c1")

    async def _ah(_request):
        return original

    out = await mw.awrap_tool_call(_request(), _ah)

    assert isinstance(out, ToolMessage)
    ref = out.additional_kwargs.get("external_ref")
    assert ref is not None
    assert ext.retrieve(ref) == "Z" * 2500


# --------------------------------------------------------------------------- #
# aggregate_external_refs helper
# --------------------------------------------------------------------------- #


def test_aggregate_external_refs_collects_refs():
    ext_ref = "file:///tmp/a.md"
    msgs = [
        HumanMessage(content="hi"),
        ToolMessage(content="ref text", tool_call_id="c1", name="big", additional_kwargs={"external_ref": ext_ref}),
        ToolMessage(content="no ref", tool_call_id="c2", name="small"),
        AIMessage(content="bye"),
    ]
    out = aggregate_external_refs({"messages": msgs})
    assert out == {ext_ref: "big"}


def test_aggregate_external_refs_empty():
    assert aggregate_external_refs({"messages": []}) == {}
    assert aggregate_external_refs({}) == {}


# --------------------------------------------------------------------------- #
# End-to-end composition: create_agent + CompressionMiddleware + L2 middleware
# --------------------------------------------------------------------------- #


async def test_integration_create_agent_source_composition(tmp_path):
    @tool
    def big_tool() -> str:
        """Returns a large string that should be externalized at the source."""
        return "B" * 3000

    class FakeWithTools(FakeMessagesListChatModel):
        # Fake models don't implement bind_tools; override so create_agent works.
        def bind_tools(self, tools, **kwargs):
            return self

    model = FakeWithTools(
        responses=[
            AIMessage(
                content="calling tool",
                tool_calls=[
                    {"name": "big_tool", "args": {}, "id": "c1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    # CompressionMiddleware with no trigger → a no-op that proves the two
    # middlewares compose without conflict.
    cmw = CompressionMiddleware(
        CompressionConfig(summary_model=model, token_threshold=None)
    )
    tmw = ToolCallExternalizerMiddleware(ext)
    agent = create_agent(
        model=model,
        tools=[big_tool],
        middleware=[cmw, tmw],
        checkpointer=InMemorySaver(),
    )

    thread = {"configurable": {"thread_id": "t1"}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="call the tool")]}, config=thread
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    tm = tool_msgs[0]
    ref = tm.additional_kwargs.get("external_ref")
    assert ref is not None, "tool result should have been externalized"
    assert "Ref:" in str(tm.content)
    assert tm.name == "big_tool"  # name preserved through externalization
    assert ext.retrieve(ref) == "B" * 3000  # original content recoverable
