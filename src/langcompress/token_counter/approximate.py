"""Approximate token counter wrapping langchain-core's heuristic counter (zero deps)."""
from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately

from langcompress.token_counter.base import TokenCounter


class ApproximateTokenCounter(TokenCounter):
    """Default heuristic counter — no model-specific tokenizer required.

    Wraps :func:`langchain_core.messages.utils.count_tokens_approximately`.
    """

    def __init__(
        self,
        *,
        chars_per_token: float = 4.0,
        use_usage_metadata_scaling: bool = True,
    ) -> None:
        self._chars_per_token = chars_per_token
        self._use_usage_metadata_scaling = use_usage_metadata_scaling

    def __call__(self, messages: Iterable[BaseMessage]) -> int:
        return count_tokens_approximately(
            messages,
            chars_per_token=self._chars_per_token,
            use_usage_metadata_scaling=self._use_usage_metadata_scaling,
        )
