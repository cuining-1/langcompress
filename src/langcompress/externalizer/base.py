"""Externalizer abstraction (L4: offload content, keep a lightweight reference).

As of v0.5 the ABC also carries the **lifecycle** surface (design §18):
``list_refs`` / ``purge`` / ``restore`` with no-op defaults, so existing
subclasses keep working unchanged and backends without lifecycle support
silently degrade (retention management is simply a no-op for them).
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ExternalRefRecord:
    """One entry in an externalizer's storage, as returned by ``list_refs``.

    Attributes:
        ref: The reference string handed out by ``externalize`` (stable across
            the record's whole lifecycle — a soft-deleted record keeps its
            original ref; the physical location is an implementation detail).
        created_at: Creation time of the stored blob. For stale records the
            retention manager deliberately uses this as a *conservative*
            lower bound of the soft-delete time (rename does not change
            mtime, so the exact soft-delete instant is not tracked) — the
            grace period therefore never expires early.
        size: Blob size in bytes.
        state: ``"active"`` (live) or ``"stale"`` (soft-deleted, still
            retrievable, awaiting the grace-period deadline).
    """

    ref: str
    created_at: datetime
    size: int
    state: str = "active"


@dataclass(frozen=True, slots=True)
class PurgeReport:
    """Audit surface of one ``purge`` run (design §18.2 invariant 5).

    All five lists are plain refs; ``errors`` carries human-readable
    ``"ref: reason"`` strings. A report is always produced — a failing
    cleanup step lands in ``errors`` instead of raising.
    """

    staled: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_affected(self) -> int:
        """How many refs this run touched (everything except ``kept``)."""
        return len(self.staled) + len(self.purged) + len(self.restored) + len(self.errors)


class Externalizer(ABC):
    """Persist a blob and return a retrievable reference string.

    The reference is the only thing that needs to live in the conversation
    context; the full content can be pulled back on demand (just-in-time
    retrieval). This is the "structured notes" strategy (design §4.5).

    Subclasses implement the sync ``externalize`` / ``retrieve`` pair (the only
    required methods). The async defaults offload them to a thread via
    :func:`asyncio.to_thread` so the event loop is never blocked by sync I/O —
    zero new dependency, semantically equivalent to an ``aiofiles``-based
    override for file-backed subclasses. Override the async methods only when a
    subclass has a genuinely async-native backend.

    Lifecycle methods (v0.5, design §18) all have **no-op defaults**: a backend
    that does not implement them simply never has anything purged, and
    ``RetentionManager`` degrades to a no-op against it.
    """

    @abstractmethod
    def externalize(self, blob: str, *, key: str | None = None) -> str: ...

    @abstractmethod
    def retrieve(self, ref: str) -> str: ...

    # ------------------------------------------------------------------ #
    # Lifecycle surface (design §18.5.1) — optional for backends
    # ------------------------------------------------------------------ #

    def list_refs(self) -> list[ExternalRefRecord]:
        """Enumerate stored records (active and stale). Default: none.

        Backends supporting retention override this so
        :class:`langcompress.RetentionManager` can see what exists.
        """
        return []

    def purge(
        self,
        keep_refs: AbstractSet[str],
        *,
        evict_refs: AbstractSet[str] = frozenset(),
        purge_refs: AbstractSet[str] = frozenset(),
        restore_refs: AbstractSet[str] = frozenset(),
    ) -> PurgeReport:
        """Apply a batch of lifecycle transitions. Default: empty report.

        Args:
            keep_refs: The reachability root set (design §18.4). Active records
                here are reported as ``kept``; the backend must never evict or
                purge them.
            evict_refs: Soft-delete (Active → Stale). Content must remain
                retrievable afterwards (design §18.2 invariant 2).
            purge_refs: Physically delete (Stale → Purged, irreversible).
                The manager only ever puts refs here that were already stale
                *before* this run and whose grace period has expired.
            restore_refs: Recover from Stale back to Active (design §18.2
                invariant 1 — a reachable ref found stale is auto-restored).

        The split into three explicit sets keeps the state machine in
        :class:`~langcompress.retention.RetentionManager` (one place) while the
        backend only executes batch filesystem/storage operations.
        """
        return PurgeReport(kept=sorted(keep_refs))

    def restore(self, ref: str) -> bool:
        """Move a stale record back to active. Default: ``False`` (unsupported).

        Manual single-ref counterpart of ``purge(restore_refs=...)`` for hosts
        that want to revive one reference outside a retention run.
        """
        return False

    async def aexternalize(self, blob: str, *, key: str | None = None) -> str:
        """Default async: run the sync ``externalize`` on a worker thread.

        ``asyncio.to_thread`` schedules the (blocking) sync call on the default
        :class:`~concurrent.futures.ThreadPoolExecutor`, freeing the event loop
        — file / network I/O releases the GIL on the syscall, so other
        coroutines proceed. Hosts externalizing at very high concurrency may
        pass a dedicated executor via ``loop.run_in_executor`` in a subclass
        override.
        """
        return await asyncio.to_thread(self.externalize, blob, key=key)

    async def aretrieve(self, ref: str) -> str:
        """Default async: run the sync ``retrieve`` on a worker thread.

        See :meth:`aexternalize` for the threading rationale.
        """
        return await asyncio.to_thread(self.retrieve, ref)

    async def alist_refs(self) -> list[ExternalRefRecord]:
        """Default async: run the sync ``list_refs`` on a worker thread."""
        return await asyncio.to_thread(self.list_refs)

    async def apurge(
        self,
        keep_refs: AbstractSet[str],
        *,
        evict_refs: AbstractSet[str] = frozenset(),
        purge_refs: AbstractSet[str] = frozenset(),
        restore_refs: AbstractSet[str] = frozenset(),
    ) -> PurgeReport:
        """Default async: run the sync ``purge`` on a worker thread."""
        return await asyncio.to_thread(
            self.purge,
            keep_refs,
            evict_refs=evict_refs,
            purge_refs=purge_refs,
            restore_refs=restore_refs,
        )

    async def arestore(self, ref: str) -> bool:
        """Default async: run the sync ``restore`` on a worker thread."""
        return await asyncio.to_thread(self.restore, ref)
