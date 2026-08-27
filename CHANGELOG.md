# Changelog

All notable changes to `langcompress` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (with the
caveat that the package is pre-1.0, so minor versions may carry breaking
changes until the v1.0 API freeze — see `CONTRIBUTING.md`).

## [0.1.0] — 2026-08-27

### Summary

Initial public release. `langcompress` is a production-grade, pluggable,
five-level layered token-compression middleware for LangGraph / LangChain
agents: L0 always-on content filtering, L2 source-side tool-output
externalization, L3 eight-segment semantic summarization, and L4 filesystem
external storage (L1 token pruning is deliberately delegated — design §4.2),
plus summary quality validation with a graceful degradation chain,
externalized-content lifecycle management, and a compression-effect benchmark
suite. See `docs/design.md` for the full design.

### Added

- **Package structure** — `src` layout, `hatchling` build. Core dependency is
  only `langchain-core` + `pydantic`; the LangChain-bound adapters
  (`CompressionMiddleware`, `ToolCallExternalizerMiddleware`) go through the
  `[middleware]` extra (`langchain`) and are lazily re-exported from
  `__init__` with an actionable ImportError when the extra is missing.
- **Core modules** — `config.CompressionConfig` (env-driven scalars, ABC
  instances never env-sourced), `state.CompressionState` (`external_refs`
  dict-merge reducer — refs accumulate across compactions instead of
  last-write-wins), `token_counter` (`base` / `approximate` / `tiktoken`),
  `summarizer` (eight-segment template / `llm_summarizer` / quality
  validation), `externalizer` (`base` / `filesystem`), `pipeline` (`base` /
  `L0Filter`), `degradation`, `retention`.
- **`CompressionMiddleware`** — subclasses
  `langchain.agents.middleware.SummarizationMiddleware` and overrides four
  private extension points (`summary_message_builder`,
  `summary_llm_config_provider`, `post_compress_hook`,
  `should_summarize_hook`) so hosts adapt framework-specific concerns without
  forking. `summary_model` is a required constructor param — the package
  ships zero LLM configuration (reads no model env vars, creates no model
  instances, binds no vendor; `langchain-openai` is an optional `[llm]` extra).
  A mini/flash/haiku-class dedicated summary model is the recommended default
  (capability-floored / cost-floored: summarization is a pure-overhead path on
  the synchronous `before_model` critical path).
- **L0 always-on content filtering** — `L0Filter` runs in-memory every
  `before_model` turn regardless of the L3 threshold: drop empty messages,
  drop back-to-back duplicates, strip reasoning content (`drop_reasoning_parts`
  for content-list thinking parts; `drop_reasoning_kwargs` for
  `reasoning_content` / `reasoning` in `additional_kwargs`, as emitted by
  GLM-4.6 / GLM-5.2 / DeepSeek-R1), merge adjacent same-type messages.
  State is written back only when L0 changed something; when L3 fires, L0's
  cleanup rides L3's full replacement. Messages carrying `tool_calls` /
  `tool_call_id` are excluded from merging (AI/Tool-pair invariant).
  `l0_filter` / `l0_enabled` config knobs + `LANGCOMPRESS_L0_ENABLED` env.
- **L2 source-side externalization** — `ToolCallExternalizerMiddleware`
  (`wrap_tool_call` / `awrap_tool_call`) swaps oversized tool outputs for a
  lightweight `file://` reference before they enter history (works with any
  `BaseTool`, including MCP-loaded tools); refs are stamped on
  `additional_kwargs["external_ref"]` (a langcompress-reserved key) and
  aggregated into state via the `aggregate_external_refs` helper.
- **Summary quality validation + graceful degradation** —
  `HeuristicQualityValidator` (5-level short-circuit; deliberately
  conservative defaults, stricter checks are opt-in knobs) plus a `D → B → C`
  `DegradationStrategy` chain: Plan A retries with the simpler fallback
  prompt, Plan D externalizes the would-be-summarized head (retrievable),
  Plan B widens the kept recent window (no I/O), Plan C truncates to
  `min_keep` (never fails). A broken strategy falls back to the original
  result so the agent always keeps running. Every retry / degradation emits
  an INFO log line and a `degradation` stamp observable in state/checkpoints.
- **§5.2 anti-retrigger** — when the head message is already a summary, only
  the message-count trigger dimension may fire; a large preserved
  `ToolMessage` no longer re-triggers compaction every turn.
- **Externalized-content lifecycle** — two-phase deletion
  (`Active → Stale (soft-deleted into .trash/, still retrievable) → Purged`)
  via `RetentionManager` / `TTLPolicy` / `NullPolicy` + `collect_live_refs`
  root-set aggregation. Root-set refs are never purged (stale ones
  auto-restored), a record evicted in a run is never purged in the same run,
  no environment variable alone can turn cleanup on, backends without
  lifecycle support silently degrade to no-op, and every run produces an
  auditable `PurgeReport` instead of raising.
- **Async parity throughout** — `aexternalize` / `aretrieve` default to
  `asyncio.to_thread(...)` so filesystem-backed externalizers never block
  the event loop; every sync surface has an async mirror.
- **Benchmark suite (`benchmarks/`)** — 12 scenarios (6 archetypes × EN/CN:
  cross-compression, entity tracking, error-fix loop, long multi-topic drift,
  thinking-heavy, tool-JSON-heavy) × 4 arms (full-context / trim /
  bare-summarization / langcompress), golden-set calibration, deterministic
  stub mode, dual JSON + Markdown reports, CJK-aware token grading, and a
  replay/dump toolchain for qualitative review.
- `examples/basic_usage.py` — runnable, no API key.

### Tests

236 passed, 1 skipped (1 needs `OPENAI_API_KEY`).

[0.1.0]: https://github.com/cuining-1/langcompress/releases/tag/v0.1.0
