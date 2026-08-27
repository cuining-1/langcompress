"""TokenCounter abstraction — a callable protocol: ``int counter(messages)``.

This shape is directly compatible with ``SummarizationMiddleware(token_counter=...)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from langchain_core.messages import BaseMessage


class TokenCounter(ABC):
    """Count tokens across a sequence of messages."""

    @abstractmethod
    def __call__(self, messages: Iterable[BaseMessage]) -> int: ...

    def count(self, messages: Iterable[BaseMessage]) -> int:
        """Convenience alias for :meth:`__call__`."""
        return self(messages)
