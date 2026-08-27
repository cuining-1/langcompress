"""Precise tiktoken-backed token counter (lazy import; requires [tiktoken] extra)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.messages import BaseMessage

from langcompress.token_counter.base import TokenCounter


class TiktokenCounter(TokenCounter):
    """Precise counter using OpenAI's tiktoken BPE tokenizer.

    Requires the ``[tiktoken]`` extra: ``pip install langcompress[tiktoken]``.
    The tokenizer is imported lazily on first use.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._encoding_name = encoding
        # ``tiktoken.Encoding`` is a lazy import (the ``[tiktoken]`` extra may
        # be absent), so type the slot as ``Any`` — the value is either
        # ``None`` (pre-first-use) or a ``tiktoken.Encoding`` instance.
        self._enc: Any = None

    def _get_encoding(self) -> Any:
        if self._enc is None:
            try:
                import tiktoken
            except ImportError as e:  # pragma: no cover - absence path
                raise ImportError(
                    "tiktoken is required for precise token counting. "
                    "Install with: pip install langcompress[tiktoken]"
                ) from e
            self._enc = tiktoken.get_encoding(self._encoding_name)
        return self._enc

    def __call__(self, messages: Iterable[BaseMessage]) -> int:
        enc = self._get_encoding()
        total = 0
        for m in messages:
            text = getattr(m, "text", None)
            if text is None:
                text = str(getattr(m, "content", ""))
            total += len(enc.encode(text))
        return total
