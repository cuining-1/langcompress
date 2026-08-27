# langcompress

> Production-grade context compression middleware for LangGraph / LangChain agents. Open-source, pluggable, five-level layered token compression. Version: v0.1.

`langcompress` provides a `CompressionMiddleware` that progressively compresses conversation history (content filtering → token pruning → reference substitution → semantic summarization → external storage) without breaking agent decision-making, maximising token utilisation.

## Install

```bash
# Core (abstractions only, langchain-core + pydantic)
pip install langcompress

# With the LangGraph/LangChain middleware adapter (recommended)
pip install langcompress[middleware]

# Everything
pip install langcompress[all]
```

| Extra | Adds |
|---|---|
| `middleware` | `CompressionMiddleware` adapter (needs `langchain`) |
| `tiktoken` | Precise token counting |
| `llm` | LLM summarization via `langchain-openai` |
| `all` | All of the above |

## Minimal usage

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langcompress import CompressionConfig, CompressionMiddleware

mw = CompressionMiddleware(CompressionConfig(
    summary_model=ChatOpenAI(model="gpt-4o-mini"),
    token_threshold=0.8,   # 80% of model input window triggers compaction
    keep_recent=6,
))
agent = create_agent(model=..., tools=[...], middleware=[mw])
```

The four extension-point hooks (`summary_message_builder`, `summary_llm_config_provider`,
`post_compress_hook`, `should_summarize_hook`) plus the optional `content_classifier`
let host projects adapt framework-specific concerns (frontend sync, message tagging,
business-scenario routing) without forking the package.

## Summary model: dedicated, cheap, fast — never the agent's main model

`summary_model` is **required**: no default, no env fallback, and no implicit
reuse of the agent's main LLM — not configuring it is a hard error, so silent
main-model pass-through is impossible. The host passes a dedicated summary
model at construction time: either a configured `BaseChatModel` instance or a
model identifier string (resolved via `init_chat_model` by the parent
middleware). The package itself ships zero LLM configuration — it reads no
model env vars, creates no model instances, and binds no vendor
(`langchain-openai` etc. are optional `[llm]`/`[all]` extras). Following the
project convention, scalars come from env vars while types and callables come
from constructor params — a model instance is the latter and is never loaded
implicitly from the environment.

Selection principle: **capability-floored, cost-floored** — a mini/flash/haiku
class model (`gpt-4o-mini` in the example above is deliberate, not a
placeholder) is the recommended default:

- **Cost** — summarization is a pure-overhead path: input ≈
  `trim_tokens_to_summarize` (default 4000 tokens), 0-2 calls per compression,
  no user-visible output. The eight-segment template's quality bar is low
  enough that cheap models pass it.
- **Latency** — compression runs on the synchronous `before_model` critical
  path; a fast dedicated model cuts the extra pause on the triggering turn
  from seconds to sub-second.
- **Failure isolation** — the summary call draws on its own rate limit/quota
  instead of the main model's, and summary failure already has a full fallback
  chain (Plan A retry with the simpler prompt → Plans B/D/C result-level
  degradation); the architecture is designed to tolerate it.

Hosts previously writing `CompressionConfig(summary_model=agent_llm, ...)`
should construct a dedicated model instead, e.g.
`CompressionConfig(summary_model=ChatOpenAI(model="gpt-4o-mini"), ...)`.

## L2 source compression (`wrap_tool_call`)

Large tool outputs (web pages, PDFs, API responses) are externalized to a lightweight
reference at the source — by a separate `ToolCallExternalizerMiddleware` — so they
never bloat the message history. Central L3 summarization (`CompressionMiddleware`)
runs alongside it; the two middlewares compose without conflict (design §4.3 / §12.3).

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langcompress import (
    CompressionConfig,
    CompressionMiddleware,
    FilesystemExternalizer,
    ToolCallExternalizerMiddleware,
)


@tool
def fetch_page(url: str) -> str:
    """Returns a large page body that should not bloat the context."""
    ...


# L2 at the source: big tool results are swapped for a reference (file:// URI)
# before they enter the history. L3 still summarizes as usual.
agent = create_agent(
    model=...,
    tools=[fetch_page],
    middleware=[
        CompressionMiddleware(CompressionConfig(summary_model=..., token_threshold=0.8)),
        ToolCallExternalizerMiddleware(FilesystemExternalizer()),
    ],
)
```

`ToolCallExternalizerMiddleware` is a **skeleton** — it ships no tool-specific
compression logic (design §12.3). It replaces oversized `ToolMessage` content with a
reference string, preserving `tool_call_id` / `name`, and stores the reference in
`ToolMessage.additional_kwargs["external_ref"]`. Aggregate those refs in a
`post_compress_hook` via the `aggregate_external_refs` helper when you need them in
state; by default the middleware does not touch state. The
`external_refs` state channel carries a **dict-merge reducer**, so the hook only
needs to return *this* compaction's new refs and the reducer accumulates them
across compactions (no last-write-wins loss across `REMOVE_ALL_MESSAGES`):

```python
def post_compress(state, result):
    return {**result, "external_refs": aggregate_external_refs(result)}
```

## Summary quality validation + graceful degradation (design §8.2)

When the summary LLM misbehaves (empty output, an `Error generating summary:`
string, a too-short / malformed summary), `CompressionMiddleware` no longer
ships the bad summary into context — it validates, then degrades:

- **Plan A — retry** with the simpler `FALLBACK_SUMMARY_PROMPT` (driven by
  `QualityValidator.validate(...).suggested_plan == "A"`).
- **Plans B / D / C — result-level substitution** via a pluggable
  `DegradationStrategy` (`D → B → C` by default): externalize the
  would-be-summarized head when an `Externalizer` is configured (D, retrievable),
  else widen the kept recent window (B, no I/O), else truncate to `min_keep`
  recent messages (C, never fails). A broken strategy falls back to the original
  result so the agent always keeps running.

```python
from langcompress import (
    CompressionConfig, CompressionMiddleware,
    HeuristicQualityValidator, FilesystemExternalizer,
)

mw = CompressionMiddleware(CompressionConfig(
    summary_model=...,
    token_threshold=0.8,
    keep_recent=6,
    # opt-in stricter quality gate (defaults are a no-op)
    quality_min_reduction_ratio=0.5,
    # enable Plan D: externalize the head instead of truncating on failure
    degradation_externalizer=FilesystemExternalizer(),
    degradation_min_keep=3,
))
```

The default `HeuristicQualityValidator` is deliberately conservative (it only
flags clear failures), so well-formed summaries always pass; the
opt-in knobs (`quality_min_reduction_ratio`, `quality_require_segments`) let
stricter hosts turn the screws without forking. `QualityValidator` and
`DegradationStrategy` are ABCs — swap in an LLM-as-judge validator or a custom
degradation chain the same way you swap `Externalizer` / `Summarizer`.

## Pipeline robustness

Four non-breaking refinements of the compression pipeline:

1. **`external_refs` dict-merge reducer** (design §13.1) — refs now accumulate
   across compactions instead of last-write-wins, so a `REMOVE_ALL_MESSAGES`
   replacement or concurrent tool calls no longer drop earlier refs. The host's
   `post_compress_hook` only returns *this* compaction's new refs and the
   reducer merges them into `state["external_refs"]`.
2. **True non-blocking async externalizer** — `Externalizer.aexternalize` /
   `aretrieve` now default to `asyncio.to_thread(self.externalize, ...)`, so a
   filesystem-backed externalizer does not block the event loop. Zero new
   dependency; semantically equivalent to an `aiofiles` override. Subclasses
   with a genuinely async-native backend still override the async methods
   directly.
3. **`aggregate_external_refs` scans all messages** — the scan was previously
   `ToolMessage`-only, which silently dropped the L3 Plan-D reference (stamped
   on the summary-shaped `HumanMessage`). It now collects `external_ref` from
   any `BaseMessage`, so both L2 (source-side `wrap_tool_call`) and L3
   (Plan-D degradation) refs are aggregated. The L2 `ToolMessage` path
   (value = tool name) is unchanged; Plan-D `HumanMessage` refs surface with
   value `""` (no `name` attribute). `additional_kwargs["external_ref"]` is a
   **langcompress-reserved key**.
4. **Degradation / quality-validation observability** — every Plan-A retry,
   Plan-B/C/D degradation, and degradation-strategy failure now emits an INFO
   log line via the `langcompress.middleware` logger (hierarchical — silence
   just the adapter by raising its level). Degraded results also carry
   `additional_kwargs["degradation"] = {"plan", "reason"[, "external_ref"]}`
   on the first non-sentinel message, so the event is observable in
   state/checkpoints. `additional_kwargs` is never rendered into the LLM
   prompt (verified: `get_buffer_string` only reads `function_call`).

See [docs/design.md](docs/design.md) for the full design and
`examples/basic_usage.py` for a runnable example. See [CHANGELOG.md](CHANGELOG.md)
for the version history and [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, architecture constraints, and PR conventions.

## Externalized-content lifecycle (design §18)

Externalized blobs do not accumulate forever: a retention system built
around two-phase deletion ensures cleanup never silently breaks the
"lossy but recoverable" promise (§3.3):

```python
from datetime import timedelta
from langcompress import (
    FilesystemExternalizer, RetentionManager, TTLPolicy, NullPolicy,
    collect_live_refs,
)

ext = FilesystemExternalizer(base_dir="./cache")
# Env-driven scalars: LANGCOMPRESS_RETENTION_TTL_HOURS (unset → None → disabled).
policy = TTLPolicy.from_env() or NullPolicy()
manager = RetentionManager(ext, policy)

# The host decides when to run (conversation end, cron, post_compress_hook):
manager.run(keep_refs=collect_live_refs(state, messages))
```

State machine per ref: `Active → Stale (soft-deleted into .trash/, still
retrievable) → Purged (after the grace period, irreversible)`. Five
invariants hold (design §18.2): root-set refs are never purged (stale ones are
auto-restored), a record evicted in a run is never purged in the same run, no
environment variable alone can turn cleanup on, backends without lifecycle
support silently degrade to no-op, and every run produces an auditable
`PurgeReport` instead of raising.

## L0 always-on content filtering (design §4.1/§12.3)

The L0 content filter (`L0Filter`) is wired into `before_model` and runs
**every turn** — regardless of whether the L3 summarization threshold has been
reached — stripping purely mechanical noise before it ever reaches the model
or the checkpoint. L0 is pure-rule (no LLM): drop empty messages, drop
back-to-back duplicates, strip reasoning content, merge adjacent same-type
messages.

```python
from langcompress import CompressionConfig, CompressionMiddleware

# L0 is on by default. It runs in-memory each before_model call; state is
# written back only when L0 actually changed something (zero checkpoint
# overhead when nothing changed). L0's cleanup rides L3's full-replacement
# for free when L3 fires, so the two never double-write.
mw = CompressionMiddleware(CompressionConfig(
    summary_model=...,
    token_threshold=0.8,
    keep_recent=6,
    l0_enabled=True,            # default; set False to disable L0 entirely
))
```

**New `drop_reasoning_kwargs` operation.** Two forms of reasoning content now
have dedicated L0 operations:

- `drop_reasoning_parts` (existing) — strips `{"type":"thinking"/"reasoning"}`
  entries from content-list-of-parts messages (Gemini-CLI style).
- `drop_reasoning_kwargs` (new) — removes `additional_kwargs["reasoning_content"]`
  / `["reasoning"]`, the OpenAI-compatible thinking-mode payload emitted by
  GLM-4.6 / GLM-5.2 / DeepSeek-R1. Both default to `True`.

```python
from langcompress import L0Filter
# Tune the four L0 operations independently (all default True):
L0Filter(drop_reasoning_kwargs=True, drop_reasoning_parts=True,
         drop_empty=True, drop_duplicates=True, merge_adjacent=True)
```

`CompressionConfig.l0_filter` accepts a custom `L0Filter` (or any
`CompressionStage`) for host-side tuning; `l0_enabled=False` (or env
`LANGCOMPRESS_L0_ENABLED=0`) turns L0 off and reverts to L3-only behaviour.

**Bug fix:** `_merge_adjacent_same_type` previously cloned only the first
message's attributes when merging two adjacent same-type messages, silently
dropping the second message's `tool_calls` / `tool_call_id` — breaking the
AI/Tool-pair invariant. Messages carrying tool metadata are now excluded from
merging.

## Middleware ordering (design §12.3)

Compression middlewares compose with framework/handoff middlewares. The
load-order rule is **producers before compressors** — `before_model` runs in
load order, so anything that injects/rewrites messages should run before the
compressor, and `wrap_tool_call` source-externalization should run so its
large `ToolMessage` outputs are externalized before the next `before_model`:

```python
from copilotkit.langgraph import CopilotKitMiddleware
# from your_handoff_pkg import HandoffMiddleware     # if using handoffs

middleware = [
    # dynamic_prompt,           # 1. (host) prompt assembly — produces SystemMessage
    # HandoffMiddleware(...),   # 2. (framework) route/handoff — produces messages
    # CopilotKitMiddleware(...),# 3. (framework) frontend sync — produces messages
    CompressionMiddleware(CompressionConfig(...)),          # 4. L0 + L3
    ToolCallExternalizerMiddleware(FilesystemExternalizer(  # 5. L2 source
        base_dir="./cache",
    )),
]
agent = create_agent(model=..., tools=[...], middleware=middleware)
```

`wrap_tool_call` (L2) intercepts any `BaseTool`, including MCP-loaded tools
(`langchain-mcp-adapters.get_tools()` returns standard `BaseTool` instances),
so MCP tools are externalized the same as native ones — no special wiring.

Each agent loads its own middleware independently (per-agent config), and the
host decides the exact order; the ordering above is the recommended default.
