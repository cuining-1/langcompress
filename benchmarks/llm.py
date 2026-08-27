"""LLM plumbing for the benchmark: factories, metering wrapper, response cache.

Three concerns, one module:

1. **Factories** — real models via ``init_chat_model`` (``"provider:model"``
   ids, ``temperature`` pinned by the settings snapshot) or deterministic
   ``FakeMessagesListChatModel`` stubs, so the whole harness runs keyless
   (CI smoke / fault injection never need an API key).
2. **Metering** — :class:`CountingChatModel` wraps any model and records per
   call: wall duration, input/output token estimates, call count. It is
   injected wherever the benchmark supplies a model (agent model, summary
   model), which is how *cost* and *latency* are measured with zero intrusion
   into the package — the wrapper is just a model the bench happens to pass in.
3. **Cache** — :class:`LLMCache`, an append-only JSONL store keyed by
   ``sha256(model + temperature + purpose + prompt)``. Judge and probe calls
   are the expensive part of a full run; caching by *content* hash (not
   sequence position) makes reruns and A/B config comparisons cheap while
   staying reproducible: same inputs → same cached output, different inputs →
   different key.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.language_models.chat_models import BaseChatModel, ChatResult
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from pydantic import ConfigDict, Field

# --------------------------------------------------------------------------- #
# Token estimation (approximate, dependency-free)
# --------------------------------------------------------------------------- #

_CHARS_PER_TOKEN = 4  # the same heuristic family as count_tokens_approximately


def estimate_tokens_text(text: str) -> int:
    """Cheap char/4 estimate for a plain string (never zero for non-empty)."""
    return max((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN, 1 if text else 0)


def estimate_tokens_messages(messages: Any) -> int:
    """Estimate tokens for a message list (or a bare prompt string)."""
    if isinstance(messages, str):
        return estimate_tokens_text(messages)
    total = 0
    for m in messages if isinstance(messages, list) else [messages]:
        content = getattr(m, "content", None)
        if isinstance(content, str):
            total += estimate_tokens_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens_text(str(part.get("text") or part.get("reasoning") or ""))
                else:
                    total += estimate_tokens_text(str(part))
        else:
            total += estimate_tokens_text(str(content or ""))
    return total


# --------------------------------------------------------------------------- #
# Metering wrapper
# --------------------------------------------------------------------------- #


@dataclass
class MeterStats:
    """Cumulative meter snapshot (compare two snapshots for a delta)."""

    calls: int = 0
    total_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "total_seconds": round(self.total_seconds, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "errors": self.errors,
        }


class CountingChatModel(BaseChatModel):
    """Metering wrapper that stays a first-class ``BaseChatModel``.

    Subclassing matters: ``create_agent`` and the middlewares probe the
    model for the full chat-model surface (``_llm_type``, ``bind``,
    ``bind_tools``, structured-output helpers), so a plain duck-typed
    wrapper breaks deep inside langchain. This class implements the
    minimal generating surface and delegates everything to ``inner`` —
    which may be a real ``init_chat_model`` result, a
    ``FakeMessagesListChatModel`` stub, or a fault-injection stub.

    Per call it records wall duration and token estimates; a *raising*
    call still counts (attempts burn input tokens and time — the fault
    suite reads ``errors`` to prove Plan-A retries happened).
    """

    inner: Any
    role: str = "model"
    stats: MeterStats = Field(default_factory=MeterStats)
    trace: bool = False
    call_log: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return f"counting:{self.role}"

    # -- metering plumbing ------------------------------------------------ #

    def _record(self, messages: Any, response: AIMessage, duration: float) -> AIMessage:
        out_text = response.text if hasattr(response, "text") else str(response.content)
        self.stats.calls += 1
        self.stats.total_seconds += duration
        self.stats.input_tokens += estimate_tokens_messages(messages)
        self.stats.output_tokens += estimate_tokens_text(str(out_text))
        if self.trace:
            self.call_log.append(
                {
                    "role": self.role,
                    "ts": time.time(),
                    "seconds": round(duration, 4),
                    "input_tokens": self.stats.input_tokens,
                    "output_tokens": self.stats.output_tokens,
                }
            )
        return response

    def _record_failure(self, messages: Any, duration: float) -> None:
        """Count a raising call: a failed attempt still burns input tokens
        and wall time — the fault-injection suite reads ``errors`` to verify
        Plan A retries actually paid for a second attempt."""
        self.stats.calls += 1
        self.stats.total_seconds += duration
        self.stats.input_tokens += estimate_tokens_messages(messages)
        self.stats.errors += 1

    # -- BaseChatModel surface --------------------------------------------- #

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        start = time.perf_counter()
        try:
            response = self.inner.invoke(messages)
        except Exception:
            self._record_failure(messages, time.perf_counter() - start)
            raise
        message = self._record(messages, response, time.perf_counter() - start)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        start = time.perf_counter()
        try:
            response = await self.inner.ainvoke(messages)
        except Exception:
            self._record_failure(messages, time.perf_counter() - start)
            raise
        message = self._record(messages, response, time.perf_counter() - start)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> CountingChatModel:
        """Passthrough for ``create_agent`` (which binds tools onto the model).

        The bound model is re-wrapped *sharing the same stats accumulator*
        so per-run metering stays exact no matter which object the framework
        ends up invoking. ``bind_tools`` returning ``self`` (stub models) is
        handled by identity check.
        """
        bound = self.inner.bind_tools(tools, **kwargs)
        if bound is self.inner:
            return self
        return CountingChatModel(
            inner=bound, role=self.role, stats=self.stats, trace=self.trace
        )

    def snapshot(self) -> MeterStats:
        return MeterStats(**self.stats.as_dict())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #


def make_real_model(model_id: str, temperature: float = 0.0) -> Any:
    """Instantiate a real chat model via ``init_chat_model``.

    Accepts ``"provider:model"`` (e.g. ``openai:gpt-4o-mini``) or a bare model
    name. Auth comes from the provider's standard env vars. ``temperature``
    is pinned here (not per call) so the reproducibility contract lives in
    one place.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(model_id, temperature=temperature)


def make_stub_model(responses: list[str | AIMessage]) -> FakeMessagesListChatModel:
    """Deterministic scripted model — the keyless backbone of stub mode."""
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=str(r)) for r in responses]
    return FakeMessagesListChatModel(responses=msgs)


# A generic eight-segment summary for stub mode: header structure matches the
# package template, content is deliberately number-free so the heuristic
# fabrication detector never flags the stub itself. Section *numbers* are
# omitted on purpose (a bare "## Primary Request and Intent" heading still
# passes segment/structure checks without injecting stray digits).
STUB_SUMMARY = """## Primary Request and Intent
The user is driving a multi-step engineering task through an agent session,
requesting implementation work and reviewing intermediate results.

## Key Technical Concepts
Context-window compression, summarization middleware, scripted tool calls,
and deterministic conversation replay.

## Files and Code Sections
Source files were read and edited during the session; the latest reviewed
state is the current reference for further changes.

## Errors and Fixes
Earlier failures were diagnosed during the session and resolved; the current
state is stable and no error is outstanding.

## Problem Solving
The conversation progressed through diagnosis, implementation, and
verification stages, with the user confirming each step before the next.

## All User Messages
The user requested the implementation, provided constraints, asked
clarifying questions, and requested verification of the results.

## Pending Tasks
Remaining work: verify final behaviour and confirm the outcome summary.

## Entity State
Key entities retain their latest values as stated in the most recent turns."""


def make_stub_summary_model(repeats: int = 16) -> FakeMessagesListChatModel:
    """Stub summary model returning the generic eight-segment text.

    ``repeats`` covers multiple compression points in long scenarios
    (FakeMessagesListChatModel pops responses in order and raises when
    exhausted).
    """
    return make_stub_model([STUB_SUMMARY] * repeats)


class FlakyModel:
    """Raises on the first ``fail_calls`` invocations, then succeeds.

    Plan-A detection: when the primary eight-segment attempt fails and the
    fallback-prompt retry succeeds, the middleware emits a *clean* L3 event
    while this model's own call counter shows >= 2 attempts — the retry is
    invisible from the outside, so the bench counts it from the inside.
    """

    def __init__(self, response: str, fail_calls: int = 1, message: str = "injected transient summary failure") -> None:
        self.response = response
        self.fail_calls = fail_calls
        self.message = message
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        self.calls += 1
        if self.calls <= self.fail_calls:
            raise RuntimeError(self.message)
        return AIMessage(content=self.response)

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        return self.invoke(messages, config=config)


class RaisingModel:
    """Stub model whose every call raises — fault injection for Plan A/D/B."""

    def __init__(self, message: str = "injected summary-LLM failure") -> None:
        self.message = message
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None) -> AIMessage:
        self.calls += 1
        raise RuntimeError(self.message)

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        return self.invoke(messages, config=config)


# --------------------------------------------------------------------------- #
# Content-hash response cache
# --------------------------------------------------------------------------- #


def cache_key(model_id: str, purpose: str, prompt: str, temperature: float = 0.0) -> str:
    """Stable cache key: model + temperature + purpose + exact prompt content."""
    blob = json.dumps(
        {"model": model_id, "purpose": purpose, "temperature": temperature, "prompt": prompt},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    """Append-only JSONL response cache (``{"key": ..., "response": ...}``).

    Loaded lazily on first use; every store() appends one line and updates the
    in-memory map, so concurrent readers within one process stay coherent and
    a crashed run never corrupts earlier entries.
    """

    def __init__(self, path: Path | None, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._data: dict[str, str] = {}
        self._loaded = False
        self.hits = 0
        self.misses = 0

    def _ensure_loaded(self) -> None:
        if self._loaded or self.path is None or not self.enabled:
            self._loaded = True
            return
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated last line from a crashed run
        self._loaded = True

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        self._ensure_loaded()
        if key in self._data:
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, response: str) -> None:
        if not self.enabled or self.path is None:
            return
        self._ensure_loaded()
        if key in self._data:
            return
        self._data[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._data)}


class NullCache(LLMCache):
    """Disabled cache (``BENCH_CACHE=0``) — every call goes to the model."""

    def __init__(self) -> None:
        super().__init__(path=None, enabled=False)


# --------------------------------------------------------------------------- #
# Cached text/JSON call helpers
# --------------------------------------------------------------------------- #


async def cached_call(
    model: Any,
    model_id: str,
    prompt: str,
    *,
    purpose: str,
    cache: LLMCache,
    temperature: float = 0.0,
) -> str:
    """Invoke ``model`` with a string prompt, through the content-hash cache.

    The cache key covers model identity, purpose, temperature and the exact
    prompt — so identical evaluations are free and any input change (new
    summary, different rubric) correctly misses.
    """
    key = cache_key(model_id, purpose, prompt, temperature)
    hit = cache.get(key)
    if hit is not None:
        return hit
    response = await model.ainvoke(prompt)
    text = response.text if hasattr(response, "text") else str(response.content)
    cache.put(key, text.strip())
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced ``{...}`` object from LLM output.

    Tolerates prose before/after the JSON (models love to add "Here is the
    JSON:") and code fences. Raises ``ValueError`` when no object parses —
    callers treat that as a judge failure, not a crash.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break  # unbalanced inner object; try next "{"
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


async def cached_json_call(
    model: Any,
    model_id: str,
    prompt: str,
    *,
    purpose: str,
    cache: LLMCache,
    temperature: float = 0.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Cached call that must return a JSON object, with one repair retry.

    On a parse failure the retry appends an explicit "Return ONLY valid JSON"
    instruction — the retry prompt differs, so it also produces a different
    cache key (correct: it is a different input).
    """
    text = await cached_call(
        model, model_id, prompt, purpose=purpose, cache=cache, temperature=temperature
    )
    try:
        return extract_json_object(text)
    except ValueError:
        if max_attempts <= 1:
            raise
    retry_prompt = prompt + "\n\nReturn ONLY a valid JSON object. No prose, no code fences."
    text = await cached_call(
        model, model_id, retry_prompt, purpose=purpose + ":json_retry", cache=cache,
        temperature=temperature,
    )
    return extract_json_object(text)


__all__ = [
    "STUB_SUMMARY",
    "CountingChatModel",
    "FlakyModel",
    "LLMCache",
    "NullCache",
    "RaisingModel",
    "cache_key",
    "cached_call",
    "cached_json_call",
    "estimate_tokens_messages",
    "estimate_tokens_text",
    "extract_json_object",
    "make_real_model",
    "make_stub_model",
    "make_stub_summary_model",
]
