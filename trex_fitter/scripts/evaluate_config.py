#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in trex_fitter.evaluate."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trex_fitter import evaluate as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Preserve imports and module-level configuration used by older callers.
    sys.modules[__name__] = _implementation
