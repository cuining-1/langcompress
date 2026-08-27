"""LangChain/LangGraph adapter — the only module that depends on ``langchain``.

:class:`langcompress.CompressionMiddleware` subclasses the real
:class:`langchain.agents.middleware.SummarizationMiddleware` and overrides its
private extension points to expose them as the design's hooks
(``summary_message_builder`` / ``summary_llm_config_provider`` /
``post_compress_hook`` / ``should_summarize_hook``) plus, as of v0.3, summary
quality validation and graceful degradation (design §8.2). With all hooks left
at their defaults the middleware behaves identically to the parent (plus an
``__summarization__`` flag on summary messages), so adopting the package never
regresses existing behaviour.

Quality + degradation wiring (design §8.2):

- **Plan A** (retry summarization with the simpler ``FALLBACK_SUMMARY_PROMPT``)
  lives in :meth:`_summarize_validated` / :meth:`_asummarize_validated`, driven
  by :attr:`ValidationResult.suggested_plan == "A"`.
- **Plans B / D / C** (result-level message substitution) live in
  :meth:`_maybe_degrade` / :meth:`_amaybe_degrade`, invoked from
  :meth:`before_model` / :meth:`abefore_model` whenever the surviving summary
  still fails validation. The :class:`DegradationStrategy` is free to honour or
  ignore the validator's ``suggested_plan`` hint; the default chain runs
  ``D → B → C`` (Plan D first when an externalizer is configured, Plan B the
  no-I/O fallback, Plan C the never-fails last resort) and never propagates an
  exception.

``before_model`` / ``abefore_model`` deliberately orchestrate the parent's
building blocks (``_should_summarize`` / ``_determine_cutoff_index`` /
``_partition_messages`` / ``_create_summary`` / ``_build_new_messages``) rather
than calling ``super().before_model`` so that validation + degradation can run
inline with the partitioned ``messages_to_summarize`` in scope — no re-derival,
no per-call state on ``self`` (which would break concurrent agents).

Importing this module requires the ``[middleware]`` extra
(``pip install langcompress[middleware]``); it is therefore lazily re-exported
from :mod:`langcompress` via ``__getattr__`` so a plain ``import langcompress``
works with only the core ``langchain-core`` dependency.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, cast

from langchain.agents.middleware import AgentState, SummarizationMiddleware
from langchain.chat_models import BaseChatModel  # noqa: F401  (re-exported convenience)
from langchain_core.messages import AnyMessage, BaseMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import override

from langcompress.config import CompressionConfig
from langcompress.degradation import (
    DefaultDegradationStrategy,
    DegradationContext,
    DegradationPatch,
    DegradationStrategy,
)
from langcompress.pipeline.l0_filter import L0Filter
from langcompress.state import CompressionState
from langcompress.summarizer.quality import (
    HeuristicQualityValidator,
    QualityValidator,
    ValidationResult,
)
from langcompress.summarizer.templates import (
    EIGHT_SEGMENT_TEMPLATE,
    FALLBACK_SUMMARY_PROMPT,
)

__all__ = ["CompressionAgentState", "CompressionMiddleware"]

# Hierarchical logger so hosts can dial per-submodule
# (``logging.getLogger("langcompress")`` catches all of langcompress; raise the
# level on ``langcompress.middleware`` alone to silence just the adapter).
logger = logging.getLogger("langcompress.middleware")


# --------------------------------------------------------------------------- #
# Graph state: AgentState (owns `messages`) + compression metadata channels
# --------------------------------------------------------------------------- #


class CompressionAgentState(AgentState[Any], CompressionState, total=False):
    """Agent graph state combining LangChain's ``AgentState`` with the
    compression metadata channels declared by :class:`CompressionState`.

    ``messages`` stays ``Required`` (carried from ``AgentState``); the
    compression fields stay ``NotRequired`` (carried from ``CompressionState``).
    The host project wires the ``messages`` channel with ``add_messages`` via
    ``AgentState``; of the compression channels, ``external_refs`` carries a
    dict-merge reducer (v0.4) so refs accumulate across compactions, while
    ``compression_count`` / ``compression_history`` stay last-value (the host
    reads and returns the full new value/list from ``post_compress_hook``).
    """


# --------------------------------------------------------------------------- #
# Config merge helper (Hook 2)
# --------------------------------------------------------------------------- #


def _merge_runnable_config(base: RunnableConfig, override: RunnableConfig) -> RunnableConfig:
    """Merge two ``RunnableConfig`` dicts.

    ``metadata`` is deep-merged (override wins on key collision); every other
    key is shallow-merged with override winning. A ``None`` override is treated
    as empty.
    """
    merged: RunnableConfig = {**base, **override}
    base_meta = base.get("metadata") or {}
    override_meta = override.get("metadata") or {}
    if base_meta or override_meta:
        merged["metadata"] = {**base_meta, **override_meta}
    return merged


# --------------------------------------------------------------------------- #
# CompressionMiddleware
# --------------------------------------------------------------------------- #


class CompressionMiddleware(SummarizationMiddleware):
    """Pluggable token-compression middleware for LangGraph/LangChain agents.

    A thin adapter over :class:`SummarizationMiddleware`: it inherits the
    parent's triggering, AI/Tool-pair-safe cutoff, summary chaining
    (old summary + new messages → new summary), and ``REMOVE_ALL_MESSAGES``
    replacement, and redirects four private extension points to the hooks
    declared on :class:`CompressionConfig`, then adds summary quality
    validation + graceful degradation (design §8.2).

    The four hooks (all optional, all defaulting to parent behaviour):

    1. ``summary_message_builder(summary)`` — builds the message that replaces
       the summarized history (overrides ``_build_new_messages``).
    2. ``summary_llm_config_provider()`` — returns a ``RunnableConfig`` merged
       into the summary LLM call (overrides ``_create_summary`` /
       ``_acreate_summary``).
    3. ``post_compress_hook(state, result)`` — post-processes the compression
       result dict (overrides ``before_model`` / ``abefore_model``).
    4. ``should_summarize_hook(messages, total_tokens, base)`` — decides whether
       to compress, given the parent's multi-signal decision
       (overrides ``_should_summarize``).

    v0.3 additions (design §8.2, all default to a no-op for well-formed
    summaries so v0.2 scenarios do not regress):

    5. ``quality_validator`` — judges the generated summary string.
    6. ``degradation_strategy`` — produces a safe replacement ``messages`` list
       when the summary fails validation (Plans B/D/C).
    7. ``degradation_externalizer`` — optional :class:`Externalizer` enabling
       Plan D (externalize the would-be-summarized head).
    """

    # Declares the compression metadata channels on the host agent's graph.
    state_schema = CompressionAgentState

    def __init__(self, config: CompressionConfig) -> None:
        self.config = config
        trigger = config._as_parent_trigger()
        keep = config.keep if config.keep is not None else ("messages", config.keep_recent)
        # Pass the count_tokens_approximately *function object* itself (not a
        # partial / wrapper) when the user supplies no counter, so the parent's
        # `token_counter is count_tokens_approximately` branch is hit and the
        # per-model tuning (e.g. Anthropic 3.3 chars/token) is applied.
        token_counter = (
            config.token_counter if config.token_counter is not None else count_tokens_approximately
        )
        summary_prompt = (
            config.summary_template if config.summary_template is not None else EIGHT_SEGMENT_TEMPLATE
        )
        super().__init__(
            model=config.summary_model,
            trigger=trigger,
            keep=keep,
            # The parent's token_counter parameter expects a wider signature
            # (str | tuple[str, str] | list[str] | dict[..., Any] inputs), but
            # count_tokens_approximately only handles BaseMessage — at runtime
            # the middleware only ever feeds it messages, so the narrower
            # signature is sound.
            token_counter=token_counter,  # type: ignore[arg-type]
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=config.trim_tokens_to_summarize,
        )

        # -- M3: quality validation + graceful degradation (design §8.2) ------ #
        # ABC instances are code, never env-sourced; None → resolve a default
        # instance (mirroring the `token_counter` / `summary_template`
        # None-resolution pattern above). The resolved validator shares the
        # parent's tuned token counter so reduction-ratio checks are consistent
        # with the triggering counter.
        self._quality_validator: QualityValidator = config.quality_validator or HeuristicQualityValidator(
            min_length=config.quality_min_length,
            min_reduction_ratio=config.quality_min_reduction_ratio,
            require_segments=config.quality_require_segments,
            token_counter=self.token_counter,
        )
        self._degradation_strategy: DegradationStrategy = (
            config.degradation_strategy or DefaultDegradationStrategy()
        )
        self._degradation_externalizer = config.degradation_externalizer
        self._degradation_widen_recent = config.degradation_widen_recent
        self._degradation_min_keep = config.degradation_min_keep

        # -- M6: L0 content filter (design §4.1/§12.3) --------------------------- #
        # None + l0_enabled=True → default L0Filter(); an explicit instance is
        # used as-is; l0_enabled=False → None (L0 disabled, v0.5 behaviour).
        # L0 runs always-on in before_model (Plan E: lazy full-replacement).
        if config.l0_filter is not None:
            self._l0_filter: L0Filter | None = config.l0_filter
        elif config.l0_enabled:
            self._l0_filter = L0Filter()
        else:
            self._l0_filter = None

    # -- Hook 1: summary message construction ------------------------------- #
    # The parent's `_build_new_messages` is a @staticmethod; overriding it as
    # an instance method is safe — `self._build_new_messages(summary)` in the
    # parent binds to this override via MRO and passes (self, summary).
    def _build_new_messages(self, summary: str) -> list[BaseMessage]:  # type: ignore[override]  # parent narrows to list[HumanMessage]; host builders may return any BaseMessage
        return [self.config.summary_message_builder(summary)]

    # -- Hook 2: summary LLM call with merged config ------------------------ #
    def _summary_config(self) -> RunnableConfig:
        base: RunnableConfig = {"metadata": {"lc_source": "summarization"}}
        override = self.config.summary_llm_config_provider() or {}
        return _merge_runnable_config(base, override)

    def _summarize_with_prompt(
        self, messages_to_summarize: list[AnyMessage], prompt: str
    ) -> str:
        """Run the summary model against ``prompt``; return summary or error string.

        Mirrors the parent's trimming, formatting, and error-handling; only the
        ``config`` passed to ``model.invoke`` differs (Hook 2 merge). Used by
        both the primary (eight-segment) and Plan-A fallback prompts.
        """
        if not messages_to_summarize:
            return "No previous conversation history."
        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."
        formatted_messages = get_buffer_string(trimmed_messages)
        try:
            response = self.model.invoke(
                prompt.format(messages=formatted_messages).rstrip(),
                config=self._summary_config(),
            )
            return response.text.strip()
        except Exception as e:  # noqa: BLE001  # mirror parent: LLM failure → error-string summary, never crash the agent
            return f"Error generating summary: {e}"

    async def _asummarize_with_prompt(
        self, messages_to_summarize: list[AnyMessage], prompt: str
    ) -> str:
        """Async variant of :meth:`_summarize_with_prompt`."""
        if not messages_to_summarize:
            return "No previous conversation history."
        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."
        formatted_messages = get_buffer_string(trimmed_messages)
        try:
            response = await self.model.ainvoke(
                prompt.format(messages=formatted_messages).rstrip(),
                config=self._summary_config(),
            )
            return response.text.strip()
        except Exception as e:  # noqa: BLE001  # mirror parent: LLM failure → error-string summary, never crash the agent
            return f"Error generating summary: {e}"

    # -- Plan A: primary summary + retry on quality failure (design §8.2) --- #
    def _summarize_validated(
        self, messages_to_summarize: list[AnyMessage]
    ) -> tuple[str, ValidationResult]:
        """Generate the primary summary and retry with the fallback prompt when
        the validator hints Plan A (retry may succeed).

        Returns the surviving summary plus its validation result. Only Plan A
        (retry summarization) lives here; result-level Plans B/D/C run later in
        :meth:`before_model` via :meth:`_maybe_degrade`.
        """
        summary = self._summarize_with_prompt(messages_to_summarize, self.summary_prompt)
        validation = self._quality_validator.validate(summary, messages_to_summarize)
        if validation.passed:
            return summary, validation
        # Retry only when the validator suggests a retry may help (empty, too
        # short, the LLM-error prefix, missing eight-segment markers). For B
        # (insufficient reduction) or C (unrecoverable) the retry won't help and
        # before_model's _maybe_degrade handles it.
        if validation.suggested_plan == "A":
            logger.info(
                "summary failed validation (plan A): %s — retrying with fallback prompt",
                validation.reason,
            )
            retry = self._summarize_with_prompt(messages_to_summarize, FALLBACK_SUMMARY_PROMPT)
            retry_validation = self._quality_validator.validate(retry, messages_to_summarize)
            if retry_validation.passed:
                logger.info("plan A retry recovered summary")
                return retry, retry_validation
            # Retry also failed; keep the primary (canonical) attempt — the
            # result-level Plans B/D/C in _maybe_degrade will discard either.
            logger.info(
                "plan A retry exhausted (%s) — deferring to result-level degradation",
                retry_validation.reason,
            )
        return summary, validation

    async def _asummarize_validated(
        self, messages_to_summarize: list[AnyMessage]
    ) -> tuple[str, ValidationResult]:
        """Async variant of :meth:`_summarize_validated`."""
        summary = await self._asummarize_with_prompt(messages_to_summarize, self.summary_prompt)
        validation = self._quality_validator.validate(summary, messages_to_summarize)
        if validation.passed:
            return summary, validation
        if validation.suggested_plan == "A":
            logger.info(
                "summary failed validation (plan A): %s — retrying with fallback prompt",
                validation.reason,
            )
            retry = await self._asummarize_with_prompt(
                messages_to_summarize, FALLBACK_SUMMARY_PROMPT
            )
            retry_validation = self._quality_validator.validate(retry, messages_to_summarize)
            if retry_validation.passed:
                logger.info("plan A retry recovered summary")
                return retry, retry_validation
            logger.info(
                "plan A retry exhausted (%s) — deferring to result-level degradation",
                retry_validation.reason,
            )
        return summary, validation

    # Keep the parent's str-returning contract for direct callers / subclasses.
    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        summary, _validation = self._summarize_validated(messages_to_summarize)
        return summary

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        summary, _validation = await self._asummarize_validated(messages_to_summarize)
        return summary

    # -- Plans B/D/C: result-level degradation (design §8.2) ---------------- #
    def _build_degradation_context(
        self,
        state: dict[str, Any],
        summary: str,
        summary_message: BaseMessage,
        messages_to_summarize: list[AnyMessage],
        preserved_recent: list[AnyMessage],
        remove_all_sentinel: BaseMessage,
    ) -> DegradationContext:
        return DegradationContext(
            state=state,
            result={
                "messages": [remove_all_sentinel, summary_message, *preserved_recent]
            },
            summary=summary,
            summary_message=summary_message,
            messages_to_summarize=list(messages_to_summarize),
            preserved_recent=list(preserved_recent),
            remove_all_sentinel=remove_all_sentinel,
            externalizer=self._degradation_externalizer,
            summary_message_builder=self.config.summary_message_builder,
            widen_recent=self._degradation_widen_recent,
            min_keep=self._degradation_min_keep,
        )

    def _stamp_degradation(self, patch: DegradationPatch) -> DegradationPatch:
        """Attach ``additional_kwargs["degradation"]`` to the first non-sentinel
        message of ``patch.messages`` for in-state observability.

        Non-mutating (``model_copy``); skips silently for non-pydantic messages
        (logging still captured the event). ``additional_kwargs`` is not rendered
        into the LLM prompt — verified: ``get_buffer_string`` only reads
        ``function_call`` and ``convert_to_openai_messages`` only reads specific
        keys, so a custom ``degradation`` key is invisible to the model. The
        metadata round-trips through checkpoints/state, which is the point.
        """
        if not patch.plan or patch.messages is None:
            return patch
        meta: dict[str, Any] = {"plan": patch.plan, "reason": patch.reason}
        if patch.external_ref is not None:
            meta["external_ref"] = patch.external_ref
        msgs = list(patch.messages)
        for i, m in enumerate(msgs):
            if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
                continue
            try:
                msgs[i] = m.model_copy(
                    update={"additional_kwargs": {**m.additional_kwargs, "degradation": meta}}
                )
            except Exception:  # noqa: BLE001, S110  # non-pydantic message → skip
                pass
            break
        return replace(patch, messages=msgs)

    def _maybe_degrade(
        self,
        state: dict[str, Any],
        summary: str,
        summary_message: BaseMessage,
        messages_to_summarize: list[AnyMessage],
        preserved_recent: list[AnyMessage],
        remove_all_sentinel: BaseMessage,
    ) -> dict[str, Any]:
        """Substitute the failed-summary result with a degradation patch.

        Never raises — a broken degradation strategy falls back to the original
        summary-bearing result so the agent keeps running. On a successful
        substitution the patch is stamped with observability metadata
        (:meth:`_stamp_degradation`) and logged at INFO.
        """
        ctx = self._build_degradation_context(
            state,
            summary,
            summary_message,
            messages_to_summarize,
            preserved_recent,
            remove_all_sentinel,
        )
        try:
            patch = self._degradation_strategy.degrade(ctx)
        except Exception:  # noqa: BLE001  # degradation must not break the agent
            logger.info("degradation strategy raised — keeping failed summary result")
            return ctx.result
        if patch.messages is None:
            return ctx.result  # identity: keep the (failed) summary result
        patch = self._stamp_degradation(patch)
        logger.info(
            "summary degraded: plan=%s reason=%s%s",
            patch.plan,
            patch.reason,
            f" external_ref={patch.external_ref}" if patch.external_ref else "",
        )
        return {"messages": patch.messages}

    async def _amaybe_degrade(
        self,
        state: dict[str, Any],
        summary: str,
        summary_message: BaseMessage,
        messages_to_summarize: list[AnyMessage],
        preserved_recent: list[AnyMessage],
        remove_all_sentinel: BaseMessage,
    ) -> dict[str, Any]:
        """Async variant of :meth:`_maybe_degrade` (uses ``adegrade`` so Plan D
        can run a true-async externalizer)."""
        ctx = self._build_degradation_context(
            state,
            summary,
            summary_message,
            messages_to_summarize,
            preserved_recent,
            remove_all_sentinel,
        )
        try:
            patch = await self._degradation_strategy.adegrade(ctx)
        except Exception:  # noqa: BLE001
            logger.info("degradation strategy raised — keeping failed summary result")
            return ctx.result
        if patch.messages is None:
            return ctx.result
        patch = self._stamp_degradation(patch)
        logger.info(
            "summary degraded: plan=%s reason=%s%s",
            patch.plan,
            patch.reason,
            f" external_ref={patch.external_ref}" if patch.external_ref else "",
        )
        return {"messages": patch.messages}

    # -- Hook 3 + the orchestration that wires in validation/degradation ---- #
    def _probe(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], list[AnyMessage]] | None:
        """Run the parent's trigger + cutoff + partition on ``messages``,
        returning ``(messages_to_summarize, preserved_recent)`` or ``None`` when
        no compression should run this turn. Accepts the (possibly L0-filtered)
        messages list from the caller — no longer reads ``state["messages"]``
        so L0's cleanup is reflected in the trigger decision (design §12.3:
        middleware owns L0/L1/L3)."""
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None
        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None
        return self._partition_messages(messages, cutoff_index)

    def _assemble_result(
        self,
        state: dict[str, Any],
        summary: str,
        validation: ValidationResult,
        messages_to_summarize: list[AnyMessage],
        preserved_recent: list[AnyMessage],
    ) -> dict[str, Any]:
        """Build the ``before_model`` result, degrading on validation failure."""
        new_messages = self._build_new_messages(summary)
        remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
        if validation.passed:
            return {"messages": [remove_all, *new_messages, *preserved_recent]}
        return self._maybe_degrade(
            state,
            summary,
            new_messages[0],
            messages_to_summarize,
            preserved_recent,
            remove_all,
        )

    async def _aassemble_result(
        self,
        state: dict[str, Any],
        summary: str,
        validation: ValidationResult,
        messages_to_summarize: list[AnyMessage],
        preserved_recent: list[AnyMessage],
    ) -> dict[str, Any]:
        new_messages = self._build_new_messages(summary)
        remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
        if validation.passed:
            return {"messages": [remove_all, *new_messages, *preserved_recent]}
        return await self._amaybe_degrade(
            state,
            summary,
            new_messages[0],
            messages_to_summarize,
            preserved_recent,
            remove_all,
        )

    def _apply_post_hook(
        self, state: dict[str, Any], result: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not result:
            return None
        try:
            return self.config.post_compress_hook(state, result)
        except Exception:  # noqa: BLE001  # a broken post-hook must not break compression
            return result

    def _l0_unchanged(
        self, original: list[AnyMessage], filtered: list[AnyMessage]
    ) -> bool:
        """True when L0 produced no substantive changes (quick shallow compare).

        Compares length + content + additional_kwargs per message — not object
        identity, because L0's strip functions return the same object when
        there's nothing to strip. When L0 merged or stripped something, the
        lengths differ or the content/additional_kwargs of at least one
        message changed, so this returns ``False`` and ``before_model``
        writes back the full replacement.
        """
        if len(original) != len(filtered):
            return False
        for a, b in zip(original, filtered):
            if getattr(a, "content", None) != getattr(b, "content", None):
                return False
            if getattr(a, "additional_kwargs", None) != getattr(
                b, "additional_kwargs", None
            ):
                return False
        return True

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        # ``AgentState`` is a TypedDict; LangGraph passes a plain dict at
        # runtime, so cast to the ``dict[str, Any]`` the orchestration helpers
        # are typed against (no per-call state on ``self``).
        state_dict = cast("dict[str, Any]", state)
        messages = state_dict["messages"]

        # ① L0 always-on (design §4.1/§12.3): run in-memory, don't write back yet.
        # ``L0Filter.run`` is typed to return ``list[BaseMessage]``; cast back to
        # the ``list[AnyMessage]`` the orchestration helpers expect — at runtime
        # L0 only emits the same ``AnyMessage`` subclasses it read in.
        filtered = cast(
            "list[AnyMessage]",
            self._l0_filter.run(messages)
            if self._l0_filter is not None
            else messages,
        )

        # ② L3 trigger check on the L0-cleaned list — stripping reasoning kwargs
        # may lower the token count enough to avoid a needless summarization.
        probe = self._probe(filtered)
        if probe is not None:
            messages_to_summarize, preserved_recent = probe
            summary, validation = self._summarize_validated(messages_to_summarize)
            result = self._assemble_result(
                state_dict, summary, validation, messages_to_summarize, preserved_recent
            )
            return self._apply_post_hook(state_dict, result)

        # ③ L3 not triggered — L0's cleanup rides the L3 replacement for free
        # when L3 fires; when it doesn't, write back only if L0 changed
        # something. No change → None (zero checkpoint overhead). Return the
        # replacement inline (no intermediate ``result`` binding) so the name
        # isn't redefined across the L3 branch above.
        if self._l0_filter is None or self._l0_unchanged(messages, filtered):
            return None
        remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
        return self._apply_post_hook(
            state_dict, {"messages": [remove_all, *filtered]}
        )

    @override
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        # See ``before_model`` for the cast rationale and Plan-E flow.
        state_dict = cast("dict[str, Any]", state)
        messages = state_dict["messages"]

        filtered = cast(
            "list[AnyMessage]",
            self._l0_filter.run(messages)
            if self._l0_filter is not None
            else messages,
        )

        probe = self._probe(filtered)
        if probe is not None:
            messages_to_summarize, preserved_recent = probe
            summary, validation = await self._asummarize_validated(messages_to_summarize)
            result = await self._aassemble_result(
                state_dict, summary, validation, messages_to_summarize, preserved_recent
            )
            return self._apply_post_hook(state_dict, result)

        if self._l0_filter is None or self._l0_unchanged(messages, filtered):
            return None
        remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
        return self._apply_post_hook(
            state_dict, {"messages": [remove_all, *filtered]}
        )

    # -- Hook 4: trigger decision ------------------------------------------- #
    # §5.2 anti-retrigger: after a compaction the kept recent window may still
    # hold large ToolMessages, so the token/fraction dimensions would re-fire
    # on every turn and re-summarize the same content — which buys nothing.
    # When the first message is already a summary, only the message-count
    # dimension may trigger (design default: 50 messages).
    _RETRIGGER_MESSAGES_DEFAULT = 50

    def _is_own_summary(self, message: AnyMessage) -> bool:
        """True when ``message`` was produced by a summarization pass.

        Detected via the ``__summarization__`` flag the default Hook-1 builder
        stamps on summary messages. A host ``summary_message_builder`` that
        does not set the flag opts out of the §5.2 anti-retrigger policy (and
        owns retrigger prevention itself) — deliberately conservative so a
        plain first message is never misclassified as a summary.
        """
        return bool(getattr(message, "additional_kwargs", {}).get("__summarization__"))

    def _messages_only_should_summarize(self, messages: list[AnyMessage]) -> bool:
        """§5.2: head-is-summary → only the ``("messages", N)`` dimension fires.

        Keeps the parent's OR semantics over the message-count conditions; when
        no messages condition is configured, falls back to the design's default
        threshold of 50 messages.
        """
        counts = [
            value for kind, value in self._trigger_conditions if kind == "messages"
        ]
        if counts:
            return any(len(messages) >= value for value in counts)
        return len(messages) >= self._RETRIGGER_MESSAGES_DEFAULT

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        base = super()._should_summarize(messages, total_tokens)
        if base and messages and self._is_own_summary(messages[0]):
            base = self._messages_only_should_summarize(messages)
        return self.config.should_summarize_hook(messages, total_tokens, base)
