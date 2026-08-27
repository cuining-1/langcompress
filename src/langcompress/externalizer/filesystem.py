"""Filesystem-backed externalizer (MVP L4 backend) with two-phase deletion (v0.5).

Lifecycle mapping (design §18.5.4):

- **Active** files live at ``base_dir/<key>.md`` (unchanged since v0.1).
- **Stale** (soft-deleted) files live at ``base_dir/.trash/<key>.md`` — moved
  there by a same-directory ``os.rename`` (zero-copy, atomic on POSIX and
  Windows). The **ref string never changes**: ``retrieve`` locates the blob in
  either place, so a stale record stays readable for the whole grace period.
- **Purged** files are unlinked from ``.trash/`` (irreversible).

Metadata needs no database: file ``mtime`` is ``created_at`` and file size is
``size``. ``rename`` does not touch ``mtime``, so the retention manager treats
``created_at`` as a conservative lower bound of the soft-delete instant —
the grace period can only expire late, never early.
"""
from __future__ import annotations

import os
import uuid
from collections.abc import Set as AbstractSet
from datetime import datetime, timezone
from pathlib import Path

from langcompress.externalizer.base import (
    Externalizer,
    ExternalRefRecord,
    PurgeReport,
)

_TRASH_DIRNAME = ".trash"


class FilesystemExternalizer(Externalizer):
    """Persist blobs to a local directory; return ``file://`` URIs as references.

    Directory resolution order: explicit ``base_dir`` arg >
    ``LANGCOMPRESS_EXTERNALIZER_DIR`` env > ``.langcompress_cache`` (cwd-relative).
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        env_dir = os.environ.get("LANGCOMPRESS_EXTERNALIZER_DIR")
        self.base_dir = Path(base_dir or env_dir or ".langcompress_cache")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core externalize/retrieve (unchanged public behaviour)
    # ------------------------------------------------------------------ #

    def externalize(self, blob: str, *, key: str | None = None) -> str:
        name = f"{key or uuid.uuid4().hex}.md"
        path = (self.base_dir / name).resolve()
        path.write_text(blob, encoding="utf-8")
        return f"file://{path}"

    def retrieve(self, ref: str) -> str:
        p = self._locate(ref)
        if p is None:
            # Preserve the pre-v0.5 error surface for a genuinely missing ref:
            # raise the same FileNotFoundError a direct read would have.
            p = Path(ref.replace("file://", "", 1))
        return p.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Lifecycle surface (design §18.5.4)
    # ------------------------------------------------------------------ #

    def _name_of(self, ref: str) -> str:
        """File name component of a ref (``file://`` URI or bare path)."""
        return Path(ref.replace("file://", "", 1)).name

    def _active_path(self, ref: str) -> Path:
        return self.base_dir / self._name_of(ref)

    def _trash_path(self, ref: str) -> Path:
        return self.base_dir / _TRASH_DIRNAME / self._name_of(ref)

    def _locate(self, ref: str) -> Path | None:
        """Where the blob for ``ref`` physically lives, or ``None``."""
        active = self._active_path(ref)
        if active.exists():
            return active
        trashed = self._trash_path(ref)
        if trashed.exists():
            return trashed
        return None

    def list_refs(self) -> list[ExternalRefRecord]:
        records: list[ExternalRefRecord] = []
        for path in sorted(self.base_dir.glob("*.md")):
            stat = path.stat()
            records.append(
                ExternalRefRecord(
                    ref=f"file://{path.resolve()}",
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    size=stat.st_size,
                    state="active",
                )
            )
        trash = self.base_dir / _TRASH_DIRNAME
        if trash.is_dir():
            for path in sorted(trash.glob("*.md")):
                stat = path.stat()
                records.append(
                    ExternalRefRecord(
                        ref=f"file://{(self.base_dir / path.name).resolve()}",
                        # The ref points at the *active* location — the record
                        # was soft-deleted, its ref string is unchanged.
                        created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        size=stat.st_size,
                        state="stale",
                    )
                )
        return records

    def purge(
        self,
        keep_refs: AbstractSet[str],
        *,
        evict_refs: AbstractSet[str] = frozenset(),
        purge_refs: AbstractSet[str] = frozenset(),
        restore_refs: AbstractSet[str] = frozenset(),
    ) -> PurgeReport:
        report = PurgeReport(kept=sorted(keep_refs))
        # Order matters: purge first (frees space), then evict (may collide
        # with a same-named trash slot vacated by the purge), then restore.
        for ref in sorted(purge_refs):
            try:
                p = self._trash_path(ref)
                if p.exists():
                    p.unlink()
                    report.purged.append(ref)
                else:
                    report.errors.append(f"{ref}: not found in trash")
            except OSError as e:
                report.errors.append(f"{ref}: {e}")
        for ref in sorted(evict_refs):
            try:
                src = self._active_path(ref)
                if not src.exists():
                    report.errors.append(f"{ref}: not found in active storage")
                    continue
                trash_dir = self.base_dir / _TRASH_DIRNAME
                trash_dir.mkdir(parents=True, exist_ok=True)
                dst = self._trash_path(ref)
                if dst.exists():
                    # Slot collision (a stale twin with the same key): the
                    # older trash entry is overwritten — it was itself already
                    # soft-deleted, so no live data is lost.
                    dst.unlink()
                os.rename(src, dst)
                report.staled.append(ref)
            except OSError as e:
                report.errors.append(f"{ref}: {e}")
        for ref in sorted(restore_refs):
            try:
                src = self._trash_path(ref)
                if not src.exists():
                    report.errors.append(f"{ref}: not found in trash")
                    continue
                os.rename(src, self._active_path(ref))
                report.restored.append(ref)
            except OSError as e:
                report.errors.append(f"{ref}: {e}")
        return report

    def restore(self, ref: str) -> bool:
        src = self._trash_path(ref)
        if not src.exists():
            return False
        os.rename(src, self._active_path(ref))
        return True
