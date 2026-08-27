"""Externalizer abstractions (L4: offload content, keep a lightweight reference)."""
from langcompress.externalizer.base import (
    Externalizer,
    ExternalRefRecord,
    PurgeReport,
)
from langcompress.externalizer.filesystem import FilesystemExternalizer

__all__ = [
    "ExternalRefRecord",
    "Externalizer",
    "FilesystemExternalizer",
    "PurgeReport",
]
