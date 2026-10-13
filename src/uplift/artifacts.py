"""Artifact serialization.

Every artifact is written with ``allow_nan=False`` so that a stray NaN or Inf
raises here instead of producing JSON that ``JSON.parse`` rejects in the browser.
NaN is mapped to ``null`` and numpy scalars to Python natives, recursively.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def sanitize(obj: Any) -> Any:
    """Convert to JSON-safe primitives: numpy -> python, non-finite -> None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy 0-d arrays and scalars
        return sanitize(obj.item())
    raise TypeError(f"cannot serialize {type(obj).__name__}: {obj!r}")


def write_json(path: Path, payload: Any) -> Path:
    """Write a sanitized payload, refusing to emit invalid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sanitize(payload), indent=2, allow_nan=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    log.info("wrote %s (%.1f KB)", path.name, len(text) / 1024)
    return path


def roundtrip_check(payload: Any) -> None:
    """Fail loudly if the payload would not survive a browser JSON.parse."""
    text = json.dumps(sanitize(payload), allow_nan=False)
    json.loads(text)


def array_payload(a: np.ndarray, decimals: int | None = None) -> list:
    """Compact float list, optionally rounded to shrink the artifact."""
    arr = np.asarray(a, dtype=np.float64)
    if decimals is not None:
        arr = np.round(arr, decimals)
    return sanitize(arr)


__all__ = ["array_payload", "roundtrip_check", "sanitize", "write_json"]
