"""L0 content filter (design §4.1) — pure-Python heuristics, zero LLM, zero extra deps.

Operations: drop empty messages, strip reasoning content (both list-of-parts and
additional_kwargs forms), drop duplicates, merge adjacent same-type content blocks.
Returns a new list; never mutates input.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any

from langchain_core.messages import BaseMessage

from langcompress.pipeline.base import CompressionStage

# content-list part types (Gemini CLI style: content is a list of dicts with "type")
_REASONING_PART_TYPES = {"thinking", "reasoning"}

# additional_kwargs keys (OpenAI-compatible thinking mode: GLM-5.2 / DeepSeek-R1)
_REASONING_KWARGS_KEYS = ("reasoning_content", "reasoning")


def _is_empty(message: BaseMessage) -> bool:
    content = getattr(message, "content", None)
    return content is None or content == "" or content == []


def _clone_with(message: BaseMessage, **updates: Any) -> BaseMessage:
    """Return a copy of ``message`` with fields replaced, without mutation.

    Uses pydantic's ``model_copy(update=...)`` (no re-validation); falls back to a
    shallow copy + attribute assignment for non-pydantic message objects.
    """
    try:
        return message.model_copy(update=updates)
    except Exception:  # noqa: BLE001  # non-pydantic message → fallback below
        new = copy.copy(message)
        for k, v in updates.items():
            try:
                setattr(new, k, v)
            except Exception:  # noqa: BLE001  # unsettable attr → give up, return original
                return message
        return new


def _strip_reasoning_kwargs(message: BaseMessage) -> BaseMessage:
    """Remove reasoning_content / reasoning from additional_kwargs.

    Targets the OpenAI-compatible thinking mode (GLM-5.2, DeepSeek-R1, etc.) where
    ``reasoning_content`` is stored as a top-level key in ``additional_kwargs``
    rather than as a content-list part. This is distinct from
    :func:`_strip_reasoning_parts` which handles the Gemini CLI style
    (content is a list of ``{"type": "thinking"/"reasoning"}`` parts).
    """
    ak = getattr(message, "additional_kwargs", None)
    if not isinstance(ak, dict) or not any(k in ak for k in _REASONING_KWARGS_KEYS):
        return message
    new_ak = {k: v for k, v in ak.items() if k not in _REASONING_KWARGS_KEYS}
    return _clone_with(message, additional_kwargs=new_ak)


def _strip_reasoning_parts(message: BaseMessage) -> BaseMessage:
    """Remove thinking/reasoning content parts (only when content is a list of parts)."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return message
    filtered = [
        p
        for p in content
        if not (isinstance(p, dict) and p.get("type") in _REASONING_PART_TYPES)
    ]
    if filtered == content:
        return message
    return _clone_with(message, content=filtered)


def _has_tool_metadata(message: BaseMessage) -> bool:
    """True when a message carries tool_calls or tool_call_id.

    Merging such messages would lose the tool metadata (the second message's
    ``tool_calls`` / ``tool_call_id`` silently dropped) and break AI/Tool-pair
    safety — the orphaned ToolMessage would have no matching AIMessage.
    """
    return bool(
        getattr(message, "tool_calls", None)
        or getattr(message, "tool_call_id", None)
    )


def _is_dupe(message: BaseMessage, out: list[BaseMessage]) -> bool:
    if not out:
        return False
    last = out[-1]
    return (
        getattr(last, "type", None) == getattr(message, "type", None)
        and getattr(last, "content", None) == getattr(message, "content", None)
    )


def _merge_adjacent_same_type(messages: list[BaseMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        if (
            out
            and getattr(out[-1], "type", None) == getattr(m, "type", None)
            and isinstance(getattr(out[-1], "content", None), str)
            and isinstance(getattr(m, "content", None), str)
            and not _has_tool_metadata(out[-1])
            and not _has_tool_metadata(m)
        ):
            merged = _clone_with(
                out[-1],
                content=f"{out[-1].content}\n{m.content}",
                id=str(uuid.uuid4()),
            )
            out[-1] = merged
        else:
            out.append(m)
    return out


class L0Filter(CompressionStage):
    """L0: drop empty/duplicate messages, strip reasoning (parts + kwargs), merge adjacent.

    Two forms of reasoning content are handled:
    - ``drop_reasoning_parts``: strips ``{"type": "thinking"/"reasoning"}`` entries from
      content-list-of-parts messages (Gemini CLI style).
    - ``drop_reasoning_kwargs``: removes ``reasoning_content`` / ``reasoning`` keys from
      ``additional_kwargs`` (OpenAI-compatible thinking mode: GLM-5.2, DeepSeek-R1, etc.).

    Both default to ``True``; set either to ``False`` to preserve that form.
    """

    name = "l0_filter"

    def __init__(
        self,
        *,
        drop_empty: bool = True,
        drop_duplicates: bool = True,
        drop_reasoning_parts: bool = True,
        drop_reasoning_kwargs: bool = True,
        merge_adjacent: bool = True,
    ) -> None:
        self.drop_empty = drop_empty
        self.drop_duplicates = drop_duplicates
        self.drop_reasoning_parts = drop_reasoning_parts
        self.drop_reasoning_kwargs = drop_reasoning_kwargs
        self.merge_adjacent = merge_adjacent

    def run(self, messages: list[BaseMessage], **kwargs: Any) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in messages:
            if self.drop_reasoning_kwargs:
                m = _strip_reasoning_kwargs(m)
            if self.drop_reasoning_parts:
                m = _strip_reasoning_parts(m)
            if self.drop_empty and _is_empty(m):
                continue
            if self.drop_duplicates and _is_dupe(m, out):
                continue
            out.append(m)
        if self.merge_adjacent:
            out = _merge_adjacent_same_type(out)
        return out
