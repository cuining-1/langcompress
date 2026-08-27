"""Compression pipeline stages (design §4 — five levels L0..L4)."""
from langcompress.pipeline.base import CompressionStage
from langcompress.pipeline.l0_filter import L0Filter

__all__ = ["CompressionStage", "L0Filter"]
