"""Pytest bootstrap: ensure ``src/`` is importable even without an editable install.

With an editable install (``pip install -e .``) this is a no-op; without one it
lets the test suite find ``langcompress`` from the source tree directly.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
