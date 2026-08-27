"""Tests for the externalized-content lifecycle system (design §18, v0.5).

Covers the five design invariants (§18.2) as executable contracts:

1. reachable ⇒ recoverable — root-set refs are never purged; stale ones in
   the root set are auto-restored;
2. two-phase deletion — evict keeps content retrievable; physical delete only
   after the grace period, and never within the run that evicted;
3. default zero behaviour change — no wiring, no cleanup; a no-op retention
   run does not even create a ``.trash/`` directory;
4. optional backend support — ABC no-op defaults degrade silently;
5. observable & never raises — reports + logs; failures land in ``errors``.

Time is simulated with ``os.utime`` (mtime is the record's ``created_at``),
so the tests are deterministic and fast.
"""
from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from langcompress import (
    Externalizer,
    ExternalRefRecord,
    FilesystemExternalizer,
    NullPolicy,
    PurgeReport,
    RetentionManager,
    TTLPolicy,
    collect_live_refs,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _age_file(path: Path, hours: float) -> None:
    """Set a file's mtime ``hours`` into the past (mtime == created_at)."""
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def _mk_ext(tmp_path: Path) -> FilesystemExternalizer:
    return FilesystemExternalizer(base_dir=tmp_path / "ext")


def _record(ref: str, *, hours: float = 0.0, state: str = "active") -> ExternalRefRecord:
    from datetime import datetime, timezone

    return ExternalRefRecord(
        ref=ref,
        created_at=datetime.fromtimestamp(time.time() - hours * 3600, tz=timezone.utc),
        size=10,
        state=state,
    )


class _MinimalExternalizer(Externalizer):
    """Backend that implements only the two required methods — pins the ABC's
    no-op lifecycle defaults (invariant 4: silent degradation)."""

    def externalize(self, blob: str, *, key: str | None = None) -> str:
        return f"mem://{key}"

    def retrieve(self, ref: str) -> str:
        return "blob"


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #


def test_external_ref_record_defaults_to_active():
    r = _record("ref-1")
    assert r.state == "active"
    assert r.size == 10


def test_purge_report_totals():
    report = PurgeReport(
        staled=["a"], purged=["b", "c"], restored=[], kept=["d"], errors=["e"]
    )
    assert report.total_affected == 4  # staled + purged + restored + errors


# --------------------------------------------------------------------------- #
# Invariant 4 — ABC no-op defaults degrade silently
# --------------------------------------------------------------------------- #


def test_minimal_backend_lifecycle_defaults():
    ext = _MinimalExternalizer()
    assert ext.list_refs() == []
    report = ext.purge({"ref-1"}, evict_refs={"ref-1"})
    assert report == PurgeReport(kept=["ref-1"])
    assert ext.restore("ref-1") is False


async def test_minimal_backend_async_lifecycle_defaults():
    ext = _MinimalExternalizer()
    assert await ext.alist_refs() == []
    report = await ext.apurge({"ref-1"}, purge_refs={"ref-1"})
    assert report.purged == []
    assert await ext.arestore("ref-1") is False


# --------------------------------------------------------------------------- #
# Filesystem backend — list_refs / purge / restore / retrieve-on-stale
# --------------------------------------------------------------------------- #


def test_list_refs_reports_active_and_stale(tmp_path):
    ext = _mk_ext(tmp_path)
    live = ext.externalize("live blob", key="live")
    dead = ext.externalize("dead blob", key="dead")
    ext.purge(set(), evict_refs={dead})

    records = {r.ref: r for r in ext.list_refs()}
    assert set(records) == {live, dead}
    assert records[live].state == "active"
    assert records[dead].state == "stale"
    # A stale record's ref still points at the active location (ref stability).
    assert records[dead].ref == dead
    assert records[live].size == len("live blob")


def test_retrieve_still_works_on_stale_ref(tmp_path):
    """Invariant 2 (two-phase): evicted content stays retrievable."""
    ext = _mk_ext(tmp_path)
    ref = ext.externalize("payload", key="k")
    ext.purge(set(), evict_refs={ref})

    assert (ext.base_dir / ".trash" / "k.md").exists()
    assert not (ext.base_dir / "k.md").exists()
    assert ext.retrieve(ref) == "payload"  # still readable after soft delete


def test_purge_restore_moves_file_back(tmp_path):
    ext = _mk_ext(tmp_path)
    ref = ext.externalize("payload", key="k")
    ext.purge(set(), evict_refs={ref})

    assert ext.restore(ref) is True
    assert (ext.base_dir / "k.md").exists()
    assert ext.retrieve(ref) == "payload"
    assert ext.restore(ref) is False  # nothing stale left for this ref


def test_purge_physically_deletes_from_trash(tmp_path):
    ext = _mk_ext(tmp_path)
    ref = ext.externalize("payload", key="k")
    ext.purge(set(), evict_refs={ref})
    report = ext.purge(set(), purge_refs={ref})

    assert report.purged == [ref]
    assert not (ext.base_dir / ".trash" / "k.md").exists()
    with pytest.raises(FileNotFoundError):
        ext.retrieve(ref)


def test_purge_reports_missing_targets_as_errors(tmp_path):
    ext = _mk_ext(tmp_path)
    ghost = f"file://{(tmp_path / 'ext' / 'ghost.md').resolve()}"
    report = ext.purge(set(), evict_refs={ghost}, purge_refs={ghost}, restore_refs={ghost})
    # Each transition reports where it looked: evict → active storage,
    # purge/restore → trash.
    assert f"{ghost}: not found in active storage" in report.errors
    assert report.errors.count(f"{ghost}: not found in trash") == 2
    assert report.staled == [] and report.purged == [] and report.restored == []


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #


def test_null_policy_never_evicts():
    policy = NullPolicy()
    assert policy.should_evict(_record("r", hours=10_000)) is False
    assert policy.grace_period == timedelta.max


def test_ttl_policy_evicts_only_old_records():
    policy = TTLPolicy(ttl=timedelta(hours=24))
    assert policy.should_evict(_record("old", hours=25)) is True
    assert policy.should_evict(_record("young", hours=1)) is False


def test_ttl_policy_default_grace_is_24h():
    assert TTLPolicy(ttl=timedelta(hours=1)).grace_period == timedelta(hours=24)
    assert TTLPolicy(ttl=timedelta(hours=1), grace_period=timedelta(hours=2)).grace_period == (
        timedelta(hours=2)
    )


def test_ttl_policy_from_env_unset_is_none(monkeypatch):
    monkeypatch.delenv("LANGCOMPRESS_RETENTION_TTL_HOURS", raising=False)
    assert TTLPolicy.from_env() is None  # unconfigured → retention disabled


def test_ttl_policy_from_env_configures(monkeypatch):
    monkeypatch.setenv("LANGCOMPRESS_RETENTION_TTL_HOURS", "48")
    monkeypatch.delenv("LANGCOMPRESS_RETENTION_GRACE_HOURS", raising=False)
    policy = TTLPolicy.from_env()
    assert policy is not None
    assert policy.ttl == timedelta(hours=48)
    assert policy.grace_period == timedelta(hours=24)  # default

    monkeypatch.setenv("LANGCOMPRESS_RETENTION_GRACE_HOURS", "6")
    assert TTLPolicy.from_env().grace_period == timedelta(hours=6)


# --------------------------------------------------------------------------- #
# RetentionManager.run — the state machine
# --------------------------------------------------------------------------- #


def test_run_keeps_root_set_untouched(tmp_path):
    """Invariant 1: reachable refs are never evicted or purged."""
    ext = _mk_ext(tmp_path)
    live = ext.externalize("live", key="live")
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=1)))

    report = manager.run({live})

    assert report.kept == [live]
    assert report.staled == [] and report.purged == []
    assert ext.retrieve(live) == "live"


def test_run_evicts_unreachable_old_record(tmp_path):
    ext = _mk_ext(tmp_path)
    live = ext.externalize("live", key="live")
    dead = ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=48)
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=24)))

    report = manager.run({live})

    assert report.staled == [dead]
    assert (ext.base_dir / ".trash" / "dead.md").exists()
    assert ext.retrieve(dead) == "dead"  # still recoverable in grace window


def test_run_never_purges_within_same_run(tmp_path):
    """Invariant 2: a record evicted *this* run must not be purged this run,
    even when its created_at already exceeds the grace period (the grace
    window is real, not a formality)."""
    ext = _mk_ext(tmp_path)
    dead = ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=10_000)  # older than ttl + grace
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=1), grace_period=timedelta(hours=1)))

    report = manager.run(set())

    assert report.staled == [dead]
    assert report.purged == []  # evicted only — purge happens on a later run


def test_run_purges_stale_after_grace_expired(tmp_path):
    ext = _mk_ext(tmp_path)
    dead = ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=48)  # over TTL → evictable
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=1), grace_period=timedelta(hours=24)))

    manager.run(set())  # run 1: soft-delete
    _age_file(ext.base_dir / ".trash" / "dead.md", hours=48)  # grace long gone
    report = manager.run(set())  # run 2: physically delete

    assert report.purged == [dead]
    with pytest.raises(FileNotFoundError):
        ext.retrieve(dead)


def test_run_auto_restores_reachable_stale_record(tmp_path):
    """Invariant 1 (mis-delete self-healing): a ref that re-enters the root
    set while stale is revived, not purged."""
    ext = _mk_ext(tmp_path)
    ref = ext.externalize("payload", key="k")
    _age_file(ext.base_dir / "k.md", hours=2)  # over TTL → evictable
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=1)))

    manager.run(set())  # unreachable → soft-deleted
    _age_file(ext.base_dir / ".trash" / "k.md", hours=48)  # grace long gone

    # The host re-pins the ref (e.g. an archived conversation reloads it).
    report = manager.run({ref})

    assert report.restored == [ref]
    assert report.purged == []
    assert ext.retrieve(ref) == "payload"
    assert (ext.base_dir / "k.md").exists()


def test_run_with_null_policy_is_full_noop(tmp_path):
    """Invariant 3: default policy never touches anything — and a no-op run
    must not even create a ``.trash/`` directory."""
    ext = _mk_ext(tmp_path)
    ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=10_000)
    manager = RetentionManager(ext, NullPolicy())

    report = manager.run(set())

    assert report == PurgeReport(kept=[])
    assert (ext.base_dir / "dead.md").exists()
    assert not (ext.base_dir / ".trash").exists()  # no side effects at all


def test_run_survives_list_refs_failure():
    """Invariant 5: a broken backend lands in errors, never raises."""

    class _BrokenList(Externalizer):
        def externalize(self, blob, *, key=None):
            return "ref"

        def retrieve(self, ref):
            return "blob"

        def list_refs(self):
            raise RuntimeError("backend down")

    manager = RetentionManager(_BrokenList(), NullPolicy())
    report = manager.run(set())
    assert "list_refs failed" in report.errors[0]


def test_run_survives_purge_failure(tmp_path):
    class _BrokenPurge(_mk_ext(tmp_path).__class__):
        def purge(self, keep_refs, **kw):
            raise RuntimeError("purge exploded")

    ext = _BrokenPurge(base_dir=tmp_path / "ext2")
    dead = ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=48)
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=1)))

    report = manager.run(set())
    assert "purge failed" in report.errors[0]
    assert ext.retrieve(dead) == "dead"  # nothing was actually lost


async def test_arun_matches_run_semantics(tmp_path):
    ext = _mk_ext(tmp_path)
    live = ext.externalize("live", key="live")
    dead = ext.externalize("dead", key="dead")
    _age_file(ext.base_dir / "dead.md", hours=48)
    manager = RetentionManager(ext, TTLPolicy(ttl=timedelta(hours=24)))

    report = await manager.arun({live})

    assert report.staled == [dead]
    assert report.kept == [live]
    assert ext.retrieve(dead) == "dead"  # grace window intact


async def test_arun_survives_backend_failure(tmp_path):
    class _BrokenAList(_mk_ext(tmp_path).__class__):
        async def alist_refs(self):
            raise RuntimeError("async backend down")

    ext = _BrokenAList(base_dir=tmp_path / "ext3")
    manager = RetentionManager(ext, NullPolicy())
    report = await manager.arun(set())
    assert "alist_refs failed" in report.errors[0]


# --------------------------------------------------------------------------- #
# collect_live_refs — the root-set aggregation (design §18.4)
# --------------------------------------------------------------------------- #


def test_collect_live_refs_three_sources():
    state = {"external_refs": {"ref-state": "tool-a"}}
    messages = [
        SimpleNamespace(additional_kwargs={"external_ref": "ref-msg"}),
        SimpleNamespace(additional_kwargs={}),  # no ref → ignored
    ]
    extra = ["ref-pinned"]

    refs = collect_live_refs(state, messages, extra)

    assert refs == {"ref-state", "ref-msg", "ref-pinned"}


def test_collect_live_refs_empty_inputs():
    assert collect_live_refs() == set()
    assert collect_live_refs({}, [], []) == set()
    assert collect_live_refs(None, None, None) == set()


# --------------------------------------------------------------------------- #
# Invariant 3 — no wiring, no cleanup (pre-v0.5 behaviour byte-identical)
# --------------------------------------------------------------------------- #


def test_no_retention_wiring_leaves_storage_untouched(tmp_path):
    """Simply externalizing and retrieving — the whole v0.4 surface — must
    leave zero lifecycle artifacts behind (no .trash, no removed files)."""
    ext = _mk_ext(tmp_path)
    ref = ext.externalize("payload", key="k")

    assert ext.retrieve(ref) == "payload"
    assert sorted(p.name for p in ext.base_dir.iterdir()) == ["k.md"]
    assert ext.list_refs()[0].state == "active"
