"""Incremental-conversion targeting engine.

Uplift modeling on the Criteo Uplift v2.1 randomized incrementality dataset.

Dataset: Diemert, Betlei, Renaudin, Amini. "A Large Scale Benchmark for Uplift
Modeling". AdKDD 2018. Released under CC BY-NC-SA 4.0.
"""

from __future__ import annotations

__version__ = "0.1.0"

ARTIFACT_SCHEMA_VERSION = 3

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]
