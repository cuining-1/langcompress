"""L2 source compression skeleton — ``wrap_tool_call`` middleware + Externalizer.

This module is the *source-side* counterpart to :mod:`langcompress.middleware`
(the central L3 adapter). Per design §12.3, L2 (reference substitution) and L4
(externalization) are handled at the **source** — the tool boundary — because
the tool best understands its own return structure and recoverability, while
the central middleware owns L0/L1/L3 (global view).

Following design §12.3 verbatim ("the package only provides the
``wrap_tool_call`` middleware skeleton and the ``Externalizer`` abstraction; it
does **not** ship tool-specific compression logic"),
:class:`ToolCallExternalizerMiddleware` is a **skeleton**: it externalizes
tool outputs that a host-supplied predicate flags as "large", replacing the
full content with a lightweight reference message. Hosts configure the
predicate (``should_externalize``) and reference text (``build_reference``) for
their own tools.

Importing this module requires the ``[middleware]`` extra
(``pip install langcompress[middleware]``); it is lazily re-exported from
:mod:`langcompress` via ``__getattr__``.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolCall, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from langcompress.externalizer import Externalizer
from langcompress.middleware import CompressionAgentState

__all__ = ["ToolCallExternalizerMiddleware", "aggregate_external_refs"]

ToolCallHandler = Callable[[ToolCallRequest], "ToolMessage | Any"]
AsyncToolCallHandler = Callable[[ToolCallRequest], Awaitable["ToolMessage | Any"]]


class ToolCallExternalizerMiddleware(AgentMiddleware):
    """Source-side (``wrap_tool_call``) compression skeleton (design §4.3/§12.3).

    Wraps tool execution: when a tool returns a large ``ToolMessage``, the full
    content is offloaded to an :class:`Externalizer` and the in-context message
    is replaced with a lightweight reference (the ``ref`` string is also stored
    in ``additional_kwargs["external_ref"]``). Small results pass through
    untouched. Hosts customise the behaviour via ``should_externalize`` and
    ``build_reference``; the package ships no tool-specific logic.

    State is **never mutated** here. Multiple tool calls run concurrently under
    ``asyncio.gather`` (see ``langgraph.prebuilt.tool_node``), so writing the
    ``external_refs`` channel from ``wrap_tool_call`` would race
    (last-write-wins under the default overwrite reducer). Aggregation of refs
    into ``state["external_refs"]`` is the host's responsibility, performed in
    the single, sequential ``post_compress_hook`` (Hook 3) via
    :func:`aggregate_external_refs`.
    """

    # Reuses the compression-aware state schema (declares the
    # ``external_refs`` channel alongside ``messages``).
    state_schema = CompressionAgentState

    def __init__(
        self,
        externalizer: Externalizer,
        *,
        threshold: int = 2000,
        should_externalize: Callable[[ToolCall, ToolMessage], bool] | None = None,
        build_reference: Callable[[ToolCall, ToolMessage, str], str] | None = None,
    ) -> None:
        self.externalizer = externalizer
        self.threshold = threshold
        # NOTE: ``ToolMessage.content`` may be a list of content blocks or a
        # dict rather than a plain string, so the default ``len(str(...))``
        # predicate is a rough heuristic. Production users should supply a
        # custom ``should_externalize`` (or a token-based one) tuned to their
        # tools' return shapes.
        self.should_externalize = (
            should_externalize
            if should_externalize is not None
            else lambda tool_call, result: len(str(result.content)) > threshold
        )
        self.build_reference = (
            build_reference
            if build_reference is not None
            else lambda tool_call, result, ref: (
                f"[Content externalized ({len(str(result.content))} chars). "
                f"Ref: {ref}. Reload via externalizer.retrieve({ref!r}).]"
            )
        )

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: ToolCallHandler
    ) -> ToolMessage | Any:
        """Intercept a sync tool call; externalize large results (Hook: source)."""
        result = handler(request)
        return self._maybe_externalize(request, result, sync=True)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: AsyncToolCallHandler
    ) -> ToolMessage | Any:
        """Intercept an async tool call; externalize large results (Hook: source).

        ``Externalizer.aexternalize`` defaults to offloading the sync
        ``externalize`` to a worker thread via :func:`asyncio.to_thread`, so even
        a filesystem-backed externalizer does not block the event loop. Override
        the externalizer's async methods only when a subclass has a genuinely
        async-native backend.
        """
        result = await handler(request)
        return await self._amaybe_externalize(request, result)

    # -- internals ---------------------------------------------------------- #

    def _maybe_externalize(
        self, request: ToolCallRequest, result: Any, *, sync: bool
    ) -> ToolMessage | Any:
        if not isinstance(result, ToolMessage):
            return result
        if not self.should_externalize(request.tool_call, result):
            return result
        ref = self.externalizer.externalize(str(result.content))
        return self._build_externalized(request, result, ref)

    async def _amaybe_externalize(
        self, request: ToolCallRequest, result: Any
    ) -> ToolMessage | Any:
        if not isinstance(result, ToolMessage):
            return result
        if not self.should_externalize(request.tool_call, result):
            return result
        ref = await self.externalizer.aexternalize(str(result.content))
        return self._build_externalized(request, result, ref)

    def _build_externalized(
        self, request: ToolCallRequest, result: ToolMessage, ref: str
    ) -> ToolMessage:
        content = self.build_reference(request.tool_call, result, ref)
        return ToolMessage(
            content=content,
            tool_call_id=result.tool_call_id,
            # ToolNode / the model expect ``name`` to match the tool; preserve it.
            name=result.name or request.tool_call.get("name"),
            additional_kwargs={**result.additional_kwargs, "external_ref": ref},
        )


def aggregate_external_refs(result: dict) -> dict[str, str]:
    """Collect ``external_ref`` entries from a compression-result dict.

    Scans **every** message in ``result["messages"]`` (not just
    :class:`ToolMessage`) for an ``external_ref`` in ``additional_kwargs`` and
    returns ``{ref: name}``:

    - **L2 source compression** (``wrap_tool_call``) stamps ``external_ref`` on
      a :class:`ToolMessage`, so ``name`` is the tool name.
    - **L3 Plan-D degradation** (``DefaultDegradationStrategy._plan_d``) stamps
      it on the summary-shaped reference message built by
      ``summary_message_builder`` (a :class:`HumanMessage` by default), which
      has no ``name`` → the value is ``""``. The ref is the key, so it is still
      collected (previously dropped because the scan was ToolMessage-only).

    ``additional_kwargs["external_ref"]`` is a **langcompress-reserved key**:
    hosts must not set it on their own messages, or those messages will be
    picked up here as false positives.

    Intended to be called from a ``post_compress_hook`` (Hook 3), which runs in
    a single, sequential node. The ``external_refs`` state channel carries a
    dict-merge reducer (v0.4), so the hook only needs to return *this*
    compaction's new refs and the reducer merges them into the already-
    accumulated ``state["external_refs"]``::

        def post_compress(state, result):
            return {**result, "external_refs": aggregate_external_refs(result)}
    """
    refs: dict[str, str] = {}
    for m in result.get("messages", []):
        ref = getattr(m, "additional_kwargs", {}).get("external_ref")
        if ref:
            refs[ref] = getattr(m, "name", None) or ""
    return refs
