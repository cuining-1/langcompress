# Contributing to langcompress

Thanks for considering a contribution! `langcompress` is pre-1.0 software
(see `CHANGELOG.md`), so the bar is "keep the v0.x surface stable, fix bugs,
add non-breaking features, and tighten the test floor toward the v1.0 API
freeze (M4)".

## Project layout

```
src/langcompress/              # the package (src layout)
├── __init__.py                # eager core exports + lazy [middleware] re-exports
├── config.py                  # CompressionConfig (env-driven, ABC instances never env-sourced)
├── state.py                   # CompressionState TypedDict + external_refs dict-merge reducer
├── token_counter/             # TokenCounter ABC + approximate/tiktoken impls
├── summarizer/                # Summarizer ABC + templates + llm_summarizer + quality
├── externalizer/              # Externalizer ABC + filesystem impl
├── pipeline/                  # CompressionStage ABC + L0Filter (L1_prune reserved)
├── degradation.py             # DegradationStrategy ABC + DefaultDegradationStrategy (D→B→C)
├── middleware.py              # [middleware] extra: CompressionMiddleware (L3 adapter)
└── toolcall_middleware.py     # [middleware] extra: ToolCallExternalizerMiddleware (L2 source)
tests/
├── unit/                      # per-module unit tests
└── scenarios/                 # end-to-end consumer-scenario tests (M2 milestone gate)
examples/basic_usage.py        # runnable, no API key
benchmarks/                    # compression-effect benchmark suite (scenarios + golden)
docs/design.md                 # full design document (five-level pipeline, hooks, invariants)
```

### Dependency tiers

- **Core** (`langchain-core` + `pydantic` only) — `config`, `state`,
  `token_counter`, `summarizer`, `externalizer`, `pipeline`, `degradation`.
  Importing these must NOT pull in `langchain` or `langgraph`.
- **`[middleware]` extra** (`langchain`) — `middleware.py` and
  `toolcall_middleware.py`. These are lazily re-exported from `__init__` via
  `__getattr__` so a plain `import langcompress` works without the extra.

Keep new code in the right tier. If a core module starts needing `langchain`,
it belongs in the `[middleware]` extra.

## Setup

```bash
git clone <repo> langcompress && cd langcompress
python -m pip install -e ".[dev]"          # langchain + tiktoken + pytest + ruff + mypy
```

On Windows PowerShell, use `python -m pip` (no bare `pip`) and `;` to chain
commands (`&&` is not a separator in PowerShell).

## Development loop

```bash
# Lint (project convention: ruff check, lint-only — do not mass-reformat
# pre-existing files, it pollutes git blame; new code may be formatted).
python -m ruff check src/langcompress/ tests/

# Type-check (strict config lives in pyproject.toml [tool.mypy]).
python -m mypy src/langcompress/

# Test (asyncio_mode = auto, so async tests just work).
python -m pytest -q
```

All three must be green before a PR. CI runs the same trio.

## Test conventions

- **Unit tests** (`tests/unit/`) — per-module, fast, no network. Use
  `FakeMessagesListChatModel` (no API key). For models that need `bind_tools`,
  subclass and override `bind_tools` to return `self` (the fakes inherit a
  `NotImplementedError`).
- **Scenario tests** (`tests/scenarios/`) — end-to-end through `create_agent`
  with `InMemorySaver()`. These are the M2/M4 "API does not bias toward one
  consumer" gate; any new hook or public API change must pass all three.
- **No network in CI** — a test needing a real provider uses
  `pytest.skip` guarded by an env var (see `test_different_llms.py`).
- **`# noqa`** — only with a concrete rule list (`# noqa: BLE001, S110`),
  never bare. `RUF100` will flag stale ones. Broad `except: pass` in the
  degradation chain is intentional (a broken degradation must not break the
  agent) and carries `# noqa: BLE001, S110`.

## Architecture constraints (do not regress)

1. **No per-call state on `self` in `before_model`** — concurrent agents
   share a middleware instance; `messages_to_summarize` / `preserved_recent`
   stay in local scope. `before_model` orchestrates the parent's building
   blocks rather than calling `super().before_model`.
2. **ABC instances are code, never env-sourced** — `quality_validator`,
   `degradation_strategy`, `degradation_externalizer` resolve to a default
   when `None`, never from `LANGCOMPRESS_*`. Scalars (thresholds, knobs) are
   env-sourced; callables/types are constructor args.
3. **`additional_kwargs["external_ref"]` is reserved** — hosts must not set
   it on their own messages, or `aggregate_external_refs` will pick them up
   as false positives. `additional_kwargs["degradation"]` is similarly
   reserved.
4. **Default async externalizer uses `asyncio.to_thread`** — do not regress
   to inline-blocking defaults. Subclasses with an async-native backend
   override `aexternalize` / `aretrieve` directly.
5. **`HeuristicQualityValidator` defaults stay conservative** — the
   stub-summary fixtures (`"SUMMARY"`, `"STUB SUMMARY"`, …) must still pass. Stricter
   checks are opt-in knobs.

## Commit / PR conventions

- Conventional-style subjects (`feat:`, `fix:`, `test:`, `docs:`,
  `refactor:`) are appreciated but not enforced.
- One logical change per PR. A change to a public hook signature or state
  schema is breaking pre-1.0 — call it out in the PR body and bump the
  minor version in `__init__.py` / `pyproject.toml` / `README.md` + add a
  `CHANGELOG.md` entry.
- Non-breaking additions (new opt-in knobs, new tests, new ABC subclass
  defaults) do not need a version bump until release.

## Versioning

Pre-1.0: minor versions may carry breaking changes (the v1.0 API freeze is
the M4 milestone). Patch versions are bug fixes / non-breaking additions.
Bump `__version__` in three places, kept in sync:

- `src/langcompress/__init__.py`
- `pyproject.toml` (`[project] version`)
- `README.md` (header line)

and add a `CHANGELOG.md` entry under the new version.

## License

By contributing you agree your contributions are licensed under the project's
MIT license (see `LICENSE`).
