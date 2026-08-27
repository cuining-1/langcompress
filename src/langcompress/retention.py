"""Externalized-content lifecycle management (design §18, v0.5).

The retention layer answers one question safely: *which externalized blobs
may be reclaimed without breaking the "lossy but recoverable" promise
(§3.3)?* It is deliberately split (same philosophy as §12.3):

- **mechanism here** — the state machine, the root-set model, the two-phase
  delete, audit reports;
- **policy/timing with the host** — when to run, what counts as reachable,
  how aggressive the TTL is. The package ships no scheduler; a host calls
  :meth:`RetentionManager.run` from wherever it wants (conversation-end hook,
  cron, ``post_compress_hook`` — see design §18.7).

State machine (design §18.3)::

    Active --(unreachable + policy evicts)--> Stale --(grace expired)--> Purged
       ^                                          |
       +---------------- restore() ---------------+

The manager never maps a ref Active → Purged within one run: physical deletion
only targets records that were *already* Stale before the run started, so the
grace period is a real window, not a formality.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from langcompress.externalizer.base import (
    Externalizer,
    ExternalRefRecord,
    PurgeReport,
)

__all__ = [
    "NullPolicy",
    "RetentionManager",
    "RetentionPolicy",
    "TTLPolicy",
    "collect_live_refs",
]

logger = logging.getLogger("langcompress.retention")


def collect_live_refs(
    state: Mapping[str, Any] | None = None,
    messages: Iterable[Any] | None = None,
    extra: Iterable[str] | None = None,
) -> set[str]:
    """Aggregate the reachability root set (design §18.4).

    A ref is *reachable* — and therefore must stay recoverable — when it
    appears in any of:

    1. ``state["external_refs"]`` keys (the v0.4 dict-merge channel — both an
       observation surface and the GC root set);
    2. any message's ``additional_kwargs["external_ref"]`` (the same scan
       :func:`langcompress.aggregate_external_refs` performs);
    3. ``extra`` — refs the host pins explicitly (e.g. archived conversations
       whose resources must survive).

    Everything is duck-typed (``Mapping`` / objects with ``additional_kwargs``)
    so the helper works with plain dicts and LangChain messages alike, with no
    langchain import in this core module.
    """
    refs: set[str] = set((state or {}).get("external_refs") or {})
    for m in messages or ():
        ref = getattr(m, "additional_kwargs", {}).get("external_ref")
        if ref:
            refs.add(ref)
    refs |= set(extra or ())
    return refs


class RetentionPolicy(ABC):
    """Decides *which* unreachable records may be soft-deleted (design §18.5.2).

    A policy is a pure predicate over :class:`ExternalRefRecord` plus a grace
    period; it never performs I/O and never sees the root set (reachability is
    the manager's job — a policy only grades *unreachable* records).
    """

    @abstractmethod
    def should_evict(self, record: ExternalRefRecord) -> bool: ...

    @property
    @abstractmethod
    def grace_period(self) -> timedelta:
        """How long a soft-deleted record stays recoverable before purging."""


class NullPolicy(RetentionPolicy):
    """Never evict anything — the default, i.e. pre-v0.5 behaviour.

    A manager running with this policy still auto-restores reachable stale
    records and purges nothing, so wiring it up is always safe.
    """

    def should_evict(self, record: ExternalRefRecord) -> bool:
        return False

    @property
    def grace_period(self) -> timedelta:
        return timedelta.max  # nothing is ever evicted, so nothing expires


class TTLPolicy(RetentionPolicy):
    """Evict records older than ``ttl`` (by ``created_at``/mtime).

    Args:
        ttl: Age threshold — an unreachable record at least this old is a
            soft-delete candidate.
        grace_period: Recoverable window between soft-delete and purge.
            Defaults to 24h (design §18.6).

    ``from_env`` reads ``LANGCOMPRESS_RETENTION_TTL_HOURS`` (unset → ``None``,
    i.e. retention disabled — the caller falls back to :class:`NullPolicy`)
    and ``LANGCOMPRESS_RETENTION_GRACE_HOURS`` (default 24). Scalar knobs come
    from the environment; the *decision to run at all* stays in host code.
    """

    def __init__(self, ttl: timedelta, grace_period: timedelta | None = None) -> None:
        self.ttl = ttl
        self._grace_period = grace_period if grace_period is not None else timedelta(hours=24)

    def should_evict(self, record: ExternalRefRecord) -> bool:
        now = datetime.now(tz=timezone.utc)
        return now - record.created_at >= self.ttl

    @property
    def grace_period(self) -> timedelta:
        return self._grace_period

    @classmethod
    def from_env(cls) -> TTLPolicy | None:
        """Build from ``LANGCOMPRESS_RETENTION_*`` env vars, or ``None``.

        ``None`` means "retention not configured" — the host then uses
        :class:`NullPolicy` (or skips wiring the manager entirely), which
        keeps the default zero-behaviour-change invariant (§18.2.3): nothing
        in the environment alone can turn cleanup on.
        """
        raw_ttl = os.environ.get("LANGCOMPRESS_RETENTION_TTL_HOURS")
        if not raw_ttl:
            return None
        ttl = timedelta(hours=float(raw_ttl))
        raw_grace = os.environ.get("LANGCOMPRESS_RETENTION_GRACE_HOURS", "24")
        return cls(ttl=ttl, grace_period=timedelta(hours=float(raw_grace)))


@dataclass(slots=True)
class RetentionManager:
    """Orchestrates one lifecycle pass over an externalizer (design §18.5.3).

    ``run`` implements the state machine:

    1. enumerate records via ``list_refs`` (a failure lands in ``errors``,
       never raises — §18.2.5);
    2. records **in the root set**: reported ``kept``; if one is found Stale
       it is auto-restored (§18.2.1: reachable ⇒ recoverable);
    3. unreachable **Active** records the policy grades evictable → soft
       delete (still retrievable during the grace window);
    4. unreachable **Stale** records older than ``grace_period`` → physical
       delete. Only records stale *before* this run are eligible, so one run
       never skips the grace window.
    """

    externalizer: Externalizer
    policy: RetentionPolicy

    def run(self, keep_refs: AbstractSet[str]) -> PurgeReport:
        try:
            records = self.externalizer.list_refs()
        except Exception as e:  # noqa: BLE001  # retention must not break the host
            logger.info("retention run aborted: list_refs failed: %s", e)
            return PurgeReport(errors=[f"list_refs failed: {e}"])

        now = datetime.now(tz=timezone.utc)
        # (4) first: only records already Stale *before* this run, unreachable,
        # past their grace period (created_at is a conservative lower bound of
        # the soft-delete instant — see filesystem.py docstring).
        to_purge = {
            r.ref
            for r in records
            if r.state == "stale"
            and r.ref not in keep_refs
            and now - r.created_at > self.policy.grace_period
        }
        # (3) unreachable Active records the policy grades evictable.
        to_evict = {
            r.ref
            for r in records
            if r.state == "active"
            and r.ref not in keep_refs
            and self.policy.should_evict(r)
        }
        # (2) reachable records found Stale → revive (mis-delete self-healing).
        to_restore = {r.ref for r in records if r.ref in keep_refs and r.state == "stale"}

        if not (to_purge or to_evict or to_restore):
            # Common fast path: nothing to do. Report the root set for audit
            # parity with real runs, but skip the backend round-trip entirely
            # (a no-op purge() call must not create e.g. a .trash directory).
            logger.info(
                "retention run: no-op (records=%d keep=%d)",
                len(records),
                len(keep_refs),
            )
            return PurgeReport(kept=sorted(keep_refs))

        try:
            report = self.externalizer.purge(
                keep_refs,
                evict_refs=to_evict,
                purge_refs=to_purge,
                restore_refs=to_restore,
            )
        except Exception as e:  # noqa: BLE001  # retention must not break the host
            logger.info("retention run aborted: purge failed: %s", e)
            return PurgeReport(errors=[f"purge failed: {e}"])

        logger.info(
            "retention run: staled=%d purged=%d restored=%d kept=%d errors=%d",
            len(report.staled),
            len(report.purged),
            len(report.restored),
            len(report.kept),
            len(report.errors),
        )
        return report

    async def arun(self, keep_refs: AbstractSet[str]) -> PurgeReport:
        """Async variant — same state machine, thread-offloaded I/O.

        Mirrors :meth:`run`; the decision logic itself is pure CPU and runs on
        the event loop, while ``alist_refs`` / ``apurge`` offload the blocking
        backend calls (``asyncio.to_thread`` defaults, v0.4 convention).
        """
        try:
            records = await self.externalizer.alist_refs()
        except Exception as e:  # noqa: BLE001  # retention must not break the host
            logger.info("retention run aborted: alist_refs failed: %s", e)
            return PurgeReport(errors=[f"alist_refs failed: {e}"])

        now = datetime.now(tz=timezone.utc)
        to_purge = {
            r.ref
            for r in records
            if r.state == "stale"
            and r.ref not in keep_refs
            and now - r.created_at > self.policy.grace_period
        }
        to_evict = {
            r.ref
            for r in records
            if r.state == "active"
            and r.ref not in keep_refs
            and self.policy.should_evict(r)
        }
        to_restore = {r.ref for r in records if r.ref in keep_refs and r.state == "stale"}

        if not (to_purge or to_evict or to_restore):
            logger.info(
                "retention run: no-op (records=%d keep=%d)",
                len(records),
                len(keep_refs),
            )
            return PurgeReport(kept=sorted(keep_refs))

        try:
            report = await self.externalizer.apurge(
                keep_refs,
                evict_refs=to_evict,
                purge_refs=to_purge,
                restore_refs=to_restore,
            )
        except Exception as e:  # noqa: BLE001  # retention must not break the host
            logger.info("retention run aborted: apurge failed: %s", e)
            return PurgeReport(errors=[f"apurge failed: {e}"])

        logger.info(
            "retention run: staled=%d purged=%d restored=%d kept=%d errors=%d",
            len(report.staled),
            len(report.purged),
            len(report.restored),
            len(report.kept),
            len(report.errors),
        )
        return report
