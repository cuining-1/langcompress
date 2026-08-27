"""Summarizer abstraction — usable independently of the middleware."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage


class Summarizer(ABC):
    """Abstract summarizer producing a structured summary string from messages."""

    @abstractmethod
    def summarize(self, messages: list[BaseMessage], **kwargs: Any) -> str: ...

    async def asummarize(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        """Default async delegates to the sync implementation."""
        return self.summarize(messages, **kwargs)
