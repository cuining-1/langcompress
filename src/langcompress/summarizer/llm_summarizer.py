"""LLM-backed summarizer using the eight-segment template (usable standalone)."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    get_buffer_string,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig

from langcompress.summarizer.base import Summarizer
from langcompress.summarizer.templates import EIGHT_SEGMENT_TEMPLATE


class LLMSummarizer(Summarizer):
    """Summarize messages via a chat model and a structured template.

    Can be used standalone (independent of ``CompressionMiddleware``) or as the
    L3 engine. Mirrors the parent ``SummarizationMiddleware._create_summary`` flow:
    trim → buffer-string format → ``model.invoke`` with ``lc_source`` metadata.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        template: str = EIGHT_SEGMENT_TEMPLATE,
        trim_tokens: int | None = 4000,
        config: RunnableConfig | None = None,
    ) -> None:
        self.model = model
        self.template = template
        self.trim_tokens = trim_tokens
        self.config: RunnableConfig = config or {}

    def _prepare(self, messages: list[BaseMessage]) -> tuple[str, RunnableConfig]:
        if self.trim_tokens:
            try:
                trimmed: list[BaseMessage] = trim_messages(
                    messages,
                    max_tokens=self.trim_tokens,
                    token_counter=count_tokens_approximately,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                )
            except Exception:  # noqa: BLE001  # trim failure → fall back to untrimmed messages
                trimmed = messages
        else:
            trimmed = messages
        formatted = get_buffer_string(trimmed)
        cfg: RunnableConfig = {
            **self.config,
            "metadata": {
                **(self.config.get("metadata") or {}),
                "lc_source": "summarization",
            },
        }
        return self.template.format(messages=formatted).rstrip(), cfg

    def summarize(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        if not messages:
            return "No previous conversation history."
        prompt, cfg = self._prepare(messages)
        try:
            resp = self.model.invoke(prompt, config=cfg)
            return resp.text.strip()
        except Exception as e:  # noqa: BLE001
            return f"Error generating summary: {e!s}"

    async def asummarize(self, messages: list[BaseMessage], **kwargs: Any) -> str:
        if not messages:
            return "No previous conversation history."
        prompt, cfg = self._prepare(messages)
        try:
            resp = await self.model.ainvoke(prompt, config=cfg)
            return resp.text.strip()
        except Exception as e:  # noqa: BLE001
            return f"Error generating summary: {e!s}"
