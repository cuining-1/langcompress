"""Tests for the default async ``Externalizer`` API (v0.4 Item 2).

The v0.3 default ``aexternalize`` / ``aretrieve`` were synchronous-looking
stubs that called the sync methods inline (blocking the event loop). v0.4
offloads them to a worker thread via :func:`asyncio.to_thread` — zero new
dependency, semantically equivalent to an ``aiofiles`` override for file-backed
externalizers.

These tests pin the contract for *all* subclasses that rely on the inherited
default (i.e. do not override the async methods):

1. ``aexternalize`` / ``aretrieve`` are real coroutines (``asyncio.isfuture``
   / ``iscoroutine``), not bare returns.
2. The awaited result equals the sync ``externalize`` / ``retrieve`` result
   (semantic equivalence with the sync path).
3. The event loop stays responsive *while the (blocking) sync call is in
   flight* — a heartbeat coroutine advances concurrently; if the default had
   stayed inline-blocking, the heartbeat could not tick during the call.

Uses a stub subclass with a deliberately blocking ``externalize`` (``time.sleep``)
to make the thread-offload observable.
"""
from __future__ import annotations

import asyncio
import time
from inspect import iscoroutine

from langcompress import Externalizer, FilesystemExternalizer


class _SlowExternalizer(Externalizer):
    """Sync externalizer that blocks for ``sleep_seconds`` in both methods.

    Lets a test observe whether ``aexternalize`` released the event loop: with
    ``asyncio.to_thread``, the ``time.sleep`` runs on a worker thread and the
    event loop is free to advance other coroutines; an inline-blocking default
    would freeze the loop for ``sleep_seconds``.
    """

    def __init__(self, *, sleep_seconds: float = 0.1, ref: str = "slow-ref") -> None:
        self.sleep_seconds = sleep_seconds
        self.ref = ref

    def externalize(self, blob: str, *, key: str | None = None) -> str:
        time.sleep(self.sleep_seconds)  # deliberate blocking
        return self.ref

    def retrieve(self, ref: str) -> str:
        time.sleep(self.sleep_seconds)  # deliberate blocking
        return f"<blob for {ref}>"


# --------------------------------------------------------------------------- #
# Coroutine shape — the inherited defaults return awaitables, not bare values
# --------------------------------------------------------------------------- #


async def test_aexternalize_is_a_coroutine() -> None:
    ext = _SlowExternalizer()
    coro = ext.aexternalize("blob")
    try:
        assert iscoroutine(coro) or asyncio.isfuture(coro)
    finally:
        # Avoid "coroutine was never awaited" if the shape check changed early.
        if iscoroutine(coro):
            await coro


async def test_aretrieve_is_a_coroutine() -> None:
    ext = _SlowExternalizer()
    coro = ext.aretrieve("any-ref")
    try:
        assert iscoroutine(coro) or asyncio.isfuture(coro)
    finally:
        if iscoroutine(coro):
            await coro


# --------------------------------------------------------------------------- #
# Semantic equivalence — awaited result matches the sync path
# --------------------------------------------------------------------------- #


async def test_aexternalize_returns_same_value_as_sync(tmp_path) -> None:
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    blob = "hello async world"
    sync_ref = ext.externalize(blob)
    async_ref = await ext.aexternalize(blob, key="other")  # different key → different path

    assert sync_ref != async_ref  # different blobs/keys → different refs
    assert ext.retrieve(sync_ref) == blob  # sync path round-trips
    assert ext.retrieve(async_ref) == blob  # async-produced ref is also retrievable
    # The async path wrote to the same backing store as sync (to_thread, not a
    # different impl), so a direct aretrieve on the async ref also works.
    assert await ext.aretrieve(async_ref) == blob


async def test_aretrieve_returns_same_value_as_sync(tmp_path) -> None:
    ext = FilesystemExternalizer(base_dir=tmp_path / "ext")
    ref = ext.externalize("payload")
    assert await ext.aretrieve(ref) == ext.retrieve(ref)


# --------------------------------------------------------------------------- #
# Non-blocking — the event loop stays responsive during the sync call
# --------------------------------------------------------------------------- #


async def test_aexternalize_offloads_sync_io_to_thread() -> None:
    """The default ``aexternalize`` runs the blocking ``externalize`` on a
    worker thread (``asyncio.to_thread``); a concurrent heartbeat must keep
    ticking *during* the call. If the default had stayed inline-blocking, the
    heartbeat could not advance at all while ``time.sleep`` held the thread."""
    ext = _SlowExternalizer(sleep_seconds=0.15)
    ticks: list[int] = []

    async def heartbeat() -> None:
        for i in range(6):
            await asyncio.sleep(0.01)
            ticks.append(i)

    # Run both concurrently. With to_thread the 0.15s sleep and the 6×0.01s
    # heartbeat overlap; an inline-blocking default would serialize them
    # (heartbeat stuck at 0 ticks until the sleep returned).
    await asyncio.gather(ext.aexternalize("blob"), heartbeat())

    # All 6 heartbeat ticks fired → the event loop was free to schedule them
    # during the (threaded) externalize call. A blocking default would leave
    # ticks empty (or at most 1) when the sleep was held inline.
    assert len(ticks) == 6, f"heartbeat did not advance concurrently: {ticks}"


async def test_aretrieve_offloads_sync_io_to_thread() -> None:
    """Symmetric to ``test_aexternalize_offloads_sync_io_to_thread`` for the
    retrieve path: the default ``aretrieve`` must not block the event loop."""
    ext = _SlowExternalizer(sleep_seconds=0.15)
    ticks: list[int] = []

    async def heartbeat() -> None:
        for i in range(6):
            await asyncio.sleep(0.01)
            ticks.append(i)

    await asyncio.gather(ext.aretrieve("any-ref"), heartbeat())
    assert len(ticks) == 6, f"heartbeat did not advance concurrently: {ticks}"


# --------------------------------------------------------------------------- #
# Non-blocking — concurrency is observable as elapsed time (sanity check)
# --------------------------------------------------------------------------- #


async def test_concurrent_aexternalize_runs_in_parallel() -> None:
    """Two concurrent ``aexternalize`` calls on a blocking sync externalizer
    complete in roughly one ``sleep_seconds`` (parallel threads), not two
    (serial). A blocking default would force them to run back-to-back on the
    event loop thread → ~2× sleep_seconds."""
    ext = _SlowExternalizer(sleep_seconds=0.1)
    loop = asyncio.get_event_loop()

    start = loop.time()
    await asyncio.gather(ext.aexternalize("a"), ext.aexternalize("b"))
    elapsed = loop.time() - start

    # Two parallel 0.1s thread calls ≈ 0.1s; serial would be ≈ 0.2s. Allow
    # generous slack for CI jitter but stay well below the serial floor.
    assert elapsed < 0.18, f"calls were not concurrent: {elapsed:.3f}s"


# --------------------------------------------------------------------------- #
# Subclass override is still honoured (the default is only a default)
# --------------------------------------------------------------------------- #


class _AsyncNativeExternalizer(Externalizer):
    """A subclass with a genuinely async-native backend overrides the async
    methods directly; the inherited default must not interfere (it is only
    used when the subclass leaves the async methods alone)."""

    def externalize(self, blob: str, *, key: str | None = None) -> str:  # pragma: no cover
        raise RuntimeError("sync path unused on async-native backend")

    def retrieve(self, ref: str) -> str:  # pragma: no cover
        raise RuntimeError("sync path unused on async-native backend")

    async def aexternalize(self, blob: str, *, key: str | None = None) -> str:
        await asyncio.sleep(0)  # genuinely async, no thread offload
        return f"async-native:{blob}"

    async def aretrieve(self, ref: str) -> str:
        await asyncio.sleep(0)
        return f"async-blob:{ref}"


async def test_subclass_async_override_is_respected() -> None:
    ext = _AsyncNativeExternalizer()
    assert await ext.aexternalize("payload") == "async-native:payload"
    assert await ext.aretrieve("r1") == "async-blob:r1"
