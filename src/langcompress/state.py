"""Compression state contract (core-only, no langchain/langgraph binding).

The host project wires the ``messages`` channel with
``Annotated[list[AnyMessage], add_messages]`` in its own LangGraph graph
(design §13.1). This module only declares the compression-metadata contract;
the langchain-bound graph state (``CompressionAgentState``) lives in
:mod:`langcompress.middleware`.
"""
from __future__ import annotations

from typing import Annotated, Any

# ``TypedDict`` is imported from ``typing_extensions`` (a transitive dependency
# of langchain-core / pydantic, present on every supported Python version) so
# that ``CompressionState`` shares the same ``_TypedDictMeta`` metaclass as
# LangChain's ``AgentState``. Mixing ``typing.TypedDict`` with
# ``typing_extensions.TypedDict`` triggers a metaclass conflict when
# ``CompressionAgentState`` multiply inherits them in ``middleware.py``.
from typing_extensions import TypedDict


def _merge_external_refs(
    left: dict[str, str] | None, right: dict[str, str] | None
) -> dict[str, str]:
    """Dict-merge reducer for the ``external_refs`` state channel.

    Refs accumulate across compactions instead of last-write-wins: a
    ``post_compress_hook`` only needs to return *this* compaction's new refs
    and the reducer merges them into the already-accumulated
    ``state["external_refs"]``. LangGraph calls it as ``(current, update)``;
    on the first update ``current`` is ``{}`` (the ``BinaryOperatorAggregate``
    channel inits to ``typ()``), so the ``or {}`` guards are defensive.

    Module-level (not a lambda) so it is picklable for checkpointing and
    referenceable as ``langcompress.state._merge_external_refs`` — the same
    convention as ``count_tokens_approximately``. Pure Python, no langgraph
    import, keeping this a core-only module.
    """
    return {**(left or {}), **(right or {})}


class CompressionState(TypedDict, total=False):
    """Compression metadata fields attached to an agent's state.

    Deliberately declares **only** compression metadata (no ``messages`` key) so
    that :class:`langcompress.middleware.CompressionAgentState` can multiply
    inherit ``AgentState`` (which owns ``messages``) without a TypedDict key
    overlap.

    ``external_refs`` carries a dict-merge reducer so refs survive
    ``REMOVE_ALL_MESSAGES`` replacements of the ``messages`` channel and
    accumulate across compactions; ``compression_count`` and
    ``compression_history`` stay last-value (host-managed — the host reads and
    returns the full new value/list from ``post_compress_hook``).
    """

    compression_count: int
    external_refs: Annotated[dict[str, str], _merge_external_refs]
    compression_history: list[dict[str, Any]]
