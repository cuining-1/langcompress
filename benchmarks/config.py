"""Bench-side configuration — environment / ``.env`` driven, nothing hardcoded.

Deliberately lives *outside* ``src/langcompress``: the package itself stays
zero-config (models injected by the host), while the benchmark — being a
consumer with its own LLM roles (model-under-test, probe answerer, judge) —
follows the host-project convention of configuring through env vars loaded
from ``.env`` at the repo root.

Roles and their env vars (all optional; any role left unset degrades to a
deterministic stub model so the harness always runs keyless):

==================  ==========================  ==================================
role                env var                     consumed by
==================  ==========================  ==================================
summary model       ``BENCH_SUMMARY_MODEL``     the langcompress / bare arms (the
                                                model under test)
probe answerer      ``BENCH_PROBE_MODEL``       QA-probe answering (defaults to
                                                the summary model when only that
                                                one is set)
judge               ``BENCH_JUDGE_MODEL``       AI reviewer + pairwise + golden
                                                calibration (independent of the
                                                summary model by design)
==================  ==========================  ==================================

Other knobs: ``BENCH_TEMPERATURE`` (default 0.0 — reproducibility contract),
``BENCH_SEED``, ``BENCH_CACHE`` (``0`` disables the content-hash LLM cache),
``BENCH_CACHE_DIR``, ``BENCH_REPORT_DIR``, ``BENCH_TRIGGER_MESSAGES``,
``BENCH_KEEP_RECENT``. Model ids use the ``init_chat_model`` form
``"provider:model"`` (e.g. ``openai:gpt-4o-mini``); provider auth keys are read
from their standard provider env vars (``OPENAI_API_KEY`` etc.).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent

_TRUTHY = ("1", "true", "yes", "on")


def load_dotenv(path: Path | None = None) -> int:
    """Minimal stdlib ``.env`` loader — ``KEY=VALUE`` lines, no override.

    Existing environment variables always win (CI injects secrets; a stale
    local ``.env`` must never shadow them). Returns the number of vars loaded.
    Comments (``#``) and blank lines are skipped; ``export `` prefixes and
    surrounding quotes are tolerated.
    """
    env_path = path if path is not None else REPO_ROOT / ".env"
    if not env_path.is_file():
        return 0
    loaded = 0
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


@dataclass(frozen=True)
class BenchSettings:
    """Immutable snapshot of every knob that influences a benchmark run.

    The whole object is serialized into the report envelope — a report is only
    "reproducible" when the exact config snapshot travels with it.
    """

    # LLM roles (None → deterministic stub for that role)
    summary_model: str | None = None
    probe_model: str | None = None
    judge_model: str | None = None

    # Reproducibility contract: temperature 0 everywhere, fixed seed recorded
    # in the report envelope. Seed is applied to ``random`` and embedded in
    # cache-key namespaces; stub models are fully deterministic regardless.
    temperature: float = 0.0
    seed: int = 20260826

    # Compression knobs under test (bridged into CompressionConfig per arm)
    trigger_messages: int = 24  # ("messages", N) trigger — deterministic offline
    keep_recent: int = 6
    trim_tokens_to_summarize: int | None = 4000
    l0_enabled: bool = True

    # Paths
    scenario_dir: Path = BENCH_ROOT / "scenarios"
    golden_path: Path = BENCH_ROOT / "golden" / "golden_set.json"
    cache_path: Path = BENCH_ROOT / "cache" / "llm_cache.jsonl"
    report_dir: Path = BENCH_ROOT / "reports"
    cache_enabled: bool = True

    # Full-text before/after dumps per (scenario × arm) — the human/AI review
    # workflow's entry point. A model judge is deliberately NOT configured
    # yet; the dump files are what a reviewer (human or AI) reads.
    dump_enabled: bool = True
    dump_dir: Path = BENCH_ROOT / "dumps"

    @property
    def stub(self) -> bool:
        """True when no real model is configured for any role (keyless mode)."""
        return not (self.summary_model or self.probe_model or self.judge_model)

    def as_dict(self) -> dict[str, object]:
        """Report-snapshot form: paths relativized, booleans plain."""
        return {
            "summary_model": self.summary_model,
            "probe_model": self.probe_model or self.summary_model,
            "judge_model": self.judge_model,
            "temperature": self.temperature,
            "seed": self.seed,
            "trigger_messages": self.trigger_messages,
            "keep_recent": self.keep_recent,
            "trim_tokens_to_summarize": self.trim_tokens_to_summarize,
            "l0_enabled": self.l0_enabled,
            "cache_enabled": self.cache_enabled,
            "dump_enabled": self.dump_enabled,
            "dump_dir": str(self.dump_dir),
            "scenario_dir": str(self.scenario_dir),
            "golden_path": str(self.golden_path),
            "cache_path": str(self.cache_path),
            "report_dir": str(self.report_dir),
        }


def load_settings(**overrides: object) -> BenchSettings:
    """Resolve settings: ``.env`` → env vars → explicit ``overrides`` (last wins).

    ``trigger_messages`` / ``keep_recent`` accept ``None``-safe int coercion;
    a malformed value falls back to the default rather than crashing the run
    (a benchmark harness should degrade loudly in the report, not silently
    refuse to start).
    """
    load_dotenv()
    env = os.environ

    def _int(key: str, default: int) -> int:
        raw = env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _float(key: str, default: float) -> float:
        raw = env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    summary_model = env.get("BENCH_SUMMARY_MODEL") or None
    settings = BenchSettings(
        summary_model=summary_model,
        # The probe answerer may reuse the summary model; the judge must be
        # chosen independently (anti-drift: never grade your own homework).
        probe_model=env.get("BENCH_PROBE_MODEL") or summary_model,
        judge_model=env.get("BENCH_JUDGE_MODEL") or None,
        temperature=_float("BENCH_TEMPERATURE", 0.0),
        seed=_int("BENCH_SEED", 20260826),
        trigger_messages=_int("BENCH_TRIGGER_MESSAGES", 24),
        keep_recent=_int("BENCH_KEEP_RECENT", 6),
        cache_enabled=env.get("BENCH_CACHE", "1").strip().lower() not in ("0", "false", "no"),
        cache_path=Path(env.get("BENCH_CACHE_DIR") or (BENCH_ROOT / "cache" / "llm_cache.jsonl")),
        report_dir=Path(env.get("BENCH_REPORT_DIR") or (BENCH_ROOT / "reports")),
        dump_enabled=env.get("BENCH_DUMP", "1").strip().lower() not in ("0", "false", "no"),
        dump_dir=Path(env.get("BENCH_DUMP_DIR") or (BENCH_ROOT / "dumps")),
    )
    # Explicit overrides (CLI flags) beat everything. Booleans must pass
    # through even when False (only None means "not overridden").
    clean = {k: v for k, v in overrides.items() if v is not None}
    from dataclasses import replace

    return replace(settings, **clean) if clean else settings
