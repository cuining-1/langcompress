"""CompressionStage abstraction — one level of the five-level pipeline (design §4).

Each stage is independently replaceable, skippable, or customisable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage


class CompressionStage(ABC):
    """A pluggable compression stage operating on a message list.

    Stages return a *new* message list and must not mutate their input.
    """

    name: str = "stage"

    @abstractmethod
    def run(self, messages: list[BaseMessage], **kwargs: Any) -> list[BaseMessage]: ...

    async def arun(self, messages: list[BaseMessage], **kwargs: Any) -> list[BaseMessage]:
        """Default async delegates to the sync implementation."""
        return self.run(messages, **kwargs)
