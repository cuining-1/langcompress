"""langcompress effect-benchmarks — an independent consumer of the package.

This package answers a question the functional test-suite (``tests/``) cannot:
**is the compression actually good** — how many tokens are saved, how much
information survives, can a post-compression agent still answer questions,
and do the degradation fallbacks fire when they should.

Design stance (mirrors design-doc §16.1 "hypothetical second consumer"):

- **Zero intrusion** — every byte of data is captured through the existing
  ``CompressionConfig.post_compress_hook`` telemetry hook plus constructor
  knobs the package already exposes. No file under ``src/langcompress`` is
  touched, no new dependency is added, no package API is extended. If a
  benchmark ever *needed* a new hook to run, that would itself be evidence of
  an API design gap.
- **Four-dimension evaluation vector** — efficiency (objective counting),
  fidelity (AI judge + fact checklist + entity recall), consistency (QA-probe
  retention against the reconstructed context), robustness (fault injection
  over the A/B/D/C degradation chain).
- **Intrinsic + extrinsic dual evaluation** — the judge inspects summaries
  (intuitive but subjective); the QA probes measure downstream capability
  (objective, and the actual point of compression). Both are cross-checked,
  and token reduction is attributed per pipeline stage (L0 filter vs L3
  summary) so each level's contribution is stated separately.
- **Reproducible reports** — fixed seed, temperature=0, judge/probe responses
  cached by content hash, reports carry the commit hash + full config
  snapshot, and are emitted in both JSON (machine diffing) and Markdown
  (human review).

Run ``python -m benchmarks --help`` (repo root) for the CLI.
"""
from __future__ import annotations

import os
import sys

# Same bootstrap as the repo-root conftest.py: prefer an installed
# (editable) langcompress, fall back to importing from the source tree so
# the harness runs from a plain checkout without installation.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_ROOT), "src")
try:
    import langcompress  # noqa: F401
except ModuleNotFoundError:
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)

__version__ = "0.1.0"
