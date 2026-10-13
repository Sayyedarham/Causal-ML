"""Data loading, materialization and splitting.

The archive holds 13.98M rows x 16 columns. It is parsed exactly once into
memory-mapped float32/uint8 arrays; every later stage reads through the memmap
so peak RAM stays bounded regardless of how large a sample a run uses.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CONVERSION_COL,
    DATASET_ROWS,
    DATASET_SHA256,
    EXPOSURE_COL,
    FEATURE_NAMES,
    TREATMENT_COL,
    VISIT_COL,
    Config,
    Paths,
)

log = logging.getLogger(__name__)

_BLOCK = 1 << 20  # rows per streaming block (~48 MB of float32 features)

_COLUMNS = [*FEATURE_NAMES, TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL]
_DTYPES = {**{f: np.float32 for f in FEATURE_NAMES},
           **{c: np.uint8 for c in (TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL)}}


# --------------------------------------------------------------------------- #
# materialization
# --------------------------------------------------------------------------- #

def materialize(paths: Paths, force: bool = False) -> dict:
    """Parse the gz archive once into memmap-friendly .npy files.

    Idempotent: returns the existing manifest unless ``force`` is set.
    """
    paths.ensure()
    manifest_path = paths.processed_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    archive = paths.raw_archive
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} not found. Download it with `uplift fetch` (see README)."
        )

    n_rows = DATASET_ROWS
    features = np.lib.format.open_memmap(
        paths.processed_dir / "features.npy", mode="w+", dtype=np.float32, shape=(n_rows, 12)
    )
    labels = {
        c: np.lib.format.open_memmap(
            paths.processed_dir / f"{c}.npy", mode="w+", dtype=np.uint8, shape=(n_rows,)
        )
        for c in (TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL)
    }
    split_key = np.lib.format.open_memmap(
        paths.processed_dir / "split_key.npy", mode="w+", dtype=np.float32, shape=(n_rows,)
    )

    rng = np.random.default_rng(Config().split.seed)
    written = 0
    t0 = time.time()

    with gzip.open(archive, "rt", encoding="utf-8") as fh:
        reader = pd.read_csv(
            fh,
            usecols=_COLUMNS,
            dtype=_DTYPES,
            chunksize=_BLOCK,
            engine="c",
        )
        for chunk in reader:
            n = len(chunk)
            sl = slice(written, written + n)
            features[sl] = chunk[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)
            for c in (TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL):
                labels[c][sl] = chunk[c].to_numpy(dtype=np.uint8, copy=False)
            # Drawn in file order from a seeded generator, so the assignment is
            # reproducible without ever storing a 14M-element index array.
            split_key[sl] = rng.random(n, dtype=np.float32)
            written += n
            if written % (4 * _BLOCK) == 0:
                log.info("materialized %s / %s rows (%.0fs)", f"{written:,}", f"{n_rows:,}",
                         time.time() - t0)

    if written != n_rows:
        raise RuntimeError(f"expected {n_rows:,} rows, parsed {written:,}")

    for arr in (features, *labels.values(), split_key):
        arr.flush()
    del features, labels, split_key

    manifest = _build_manifest(paths, written, time.time() - t0)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("materialization complete in %.1fs", manifest["elapsed_sec"])
    return manifest


def _build_manifest(paths: Paths, n_rows: int, elapsed: float) -> dict:
    treatment = np.asarray(_mmap(paths, TREATMENT_COL))
    conversion = np.asarray(_mmap(paths, CONVERSION_COL))
    exposure = np.asarray(_mmap(paths, EXPOSURE_COL))
    treated = treatment == 1
    return {
        "rows": n_rows,
        "columns": _COLUMNS,
        "sha256_raw": DATASET_SHA256,
        "treatment_ratio": float(treatment.mean()),
        "conversion_rate": float(conversion.mean()),
        "exposure_rate_treated": float(exposure[treated].mean()),
        "exposure_rate_control": float(exposure[~treated].mean()),
        "elapsed_sec": round(elapsed, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _mmap(paths: Paths, name: str) -> np.ndarray:
    return np.load(paths.processed_dir / f"{name}.npy", mmap_mode="r")


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SplitData:
    """A materialized, in-memory split."""

    name: str
    X: np.ndarray            # float32 (n, 12)
    treatment: np.ndarray    # uint8   (n,)
    conversion: np.ndarray   # uint8   (n,)
    visit: np.ndarray        # uint8   (n,)
    exposure: np.ndarray     # uint8   (n,)
    row_ids: np.ndarray      # int64   (n,) original positions in the archive

    def __len__(self) -> int:
        return len(self.row_ids)

    @property
    def n_treated(self) -> int:
        return int(self.treatment.sum())

    @property
    def n_control(self) -> int:
        return len(self) - self.n_treated

    def outcome(self, name: str) -> np.ndarray:
        if name == CONVERSION_COL:
            return self.conversion
        if name == VISIT_COL:
            return self.visit
        raise KeyError(f"unknown outcome {name!r}")

    def subset(self, mask: np.ndarray) -> SplitData:
        return SplitData(
            name=self.name,
            X=self.X[mask], treatment=self.treatment[mask],
            conversion=self.conversion[mask], visit=self.visit[mask],
            exposure=self.exposure[mask], row_ids=self.row_ids[mask],
        )

    def sample(self, n: int, seed: int) -> SplitData:
        """Uniform subsample without replacement (a no-op if n >= len(self))."""
        if n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=n, replace=False)
        idx.sort()  # keeps the downstream memory access pattern sequential
        return self.subset(_boolean_from_indices(idx, len(self)))

    def summary(self) -> dict:
        t = self.treatment == 1
        y = self.conversion
        return {
            "n": len(self),
            "n_treated": int(t.sum()),
            "n_control": int((~t).sum()),
            "treatment_ratio": float(t.mean()),
            "conversion_rate": float(y.mean()),
            "conversion_rate_treated": float(y[t].mean()) if t.any() else float("nan"),
            "conversion_rate_control": float(y[~t].mean()) if (~t).any() else float("nan"),
            "visit_rate": float(self.visit.mean()),
            "exposure_rate_treated": float(self.exposure[t].mean()) if t.any() else float("nan"),
            "exposure_rate_control": float(self.exposure[~t].mean()) if (~t).any() else float("nan"),
        }


def _boolean_from_indices(idx: np.ndarray, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def split_mask(paths: Paths, name: str, cfg: Config) -> np.ndarray:
    """Boolean membership mask for a split, derived from the stored split key."""
    key = _mmap(paths, "split_key")
    test_hi = cfg.split.test
    valid_hi = test_hi + cfg.split.valid
    if name == "test":
        return np.asarray(key) < test_hi
    if name == "valid":
        k = np.asarray(key)
        return (k >= test_hi) & (k < valid_hi)
    if name == "train":
        return np.asarray(key) >= valid_hi
    raise KeyError(f"unknown split {name!r}")


def load_split(paths: Paths, name: str, cfg: Config, sample_rows: int | None = None,
               seed: int | None = None) -> SplitData:
    """Materialize one split into RAM, optionally subsampled."""
    mask = split_mask(paths, name, cfg)
    n_total = int(mask.sum())

    if sample_rows is not None and sample_rows < n_total:
        rng = np.random.default_rng(seed if seed is not None else cfg.split.seed)
        keep = rng.random(n_total) < (sample_rows / n_total)
        mask = _scatter(mask, keep)
        log.info("split=%s sampled %s -> %s rows", name, f"{n_total:,}", f"{int(mask.sum()):,}")

    return _gather(paths, name, mask)


def _scatter(split_mask_full: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Expand a per-member keep vector back to full-length coordinates."""
    out = np.zeros_like(split_mask_full)
    out[split_mask_full] = keep
    return out


def _gather(paths: Paths, name: str, mask: np.ndarray) -> SplitData:
    """Copy masked rows out of the memmaps block by block to bound peak RAM."""
    n_out = int(mask.sum())
    X = np.empty((n_out, 12), dtype=np.float32)
    cols = {c: np.empty(n_out, dtype=np.uint8) for c in
            (TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL)}
    row_ids = np.empty(n_out, dtype=np.int64)

    features = _mmap_features(paths)
    written = 0
    n_rows = len(mask)
    for start in range(0, n_rows, _BLOCK):
        stop = min(start + _BLOCK, n_rows)
        block_mask = mask[start:stop]
        k = int(block_mask.sum())
        if k == 0:
            continue
        dest = slice(written, written + k)
        X[dest] = features[start:stop][block_mask]
        for c in cols:
            cols[c][dest] = _mmap(paths, c)[start:stop][block_mask]
        row_ids[dest] = np.nonzero(block_mask)[0] + start
        written += k

    if written != n_out:
        raise RuntimeError(f"gather mismatch: {written} != {n_out}")

    return SplitData(name=name, X=X, treatment=cols[TREATMENT_COL],
                     conversion=cols[CONVERSION_COL], visit=cols[VISIT_COL],
                     exposure=cols[EXPOSURE_COL], row_ids=row_ids)


def _mmap_features(paths: Paths) -> np.ndarray:
    return np.load(paths.processed_dir / "features.npy", mmap_mode="r")


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def covariate_balance(split: SplitData) -> pd.DataFrame:
    """Standardized mean differences between arms.

    Under correct randomization every |SMD| should be far below 0.1. This is the
    empirical justification for treating assignment as ignorable.
    """
    t = split.treatment == 1
    rows = []
    for j, fname in enumerate(FEATURE_NAMES):
        x = split.X[:, j].astype(np.float64)
        a, b = x[t], x[~t]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        rows.append({
            "feature": fname, "mean_treated": a.mean(), "mean_control": b.mean(),
            "sd_treated": a.std(ddof=1), "sd_control": b.std(ddof=1),
            "smd": smd, "abs_smd": abs(smd),
        })
    df = pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)
    return df


def verify_randomization(split: SplitData, sample_rows: int = 500_000,
                         seed: int = 4242) -> dict:
    """Test that treatment assignment is independent of the covariates.

    Two complementary checks:
      1. Standardized mean differences per feature (|SMD| < 0.1 is balanced).
      2. Predictability of assignment: a cross-validated propensity model should
         achieve AUC ~0.5. Any real separation would mean assignment is
         confounded and the whole ITT framing would be invalid.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import cross_val_predict

    sub = split.sample(min(sample_rows, len(split)), seed)
    balance = covariate_balance(sub)

    t = (sub.treatment == 1).astype(int)
    X = sub.X.astype(np.float64)

    # Standardized so the L2 penalty applies comparably across features.
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

    lr = LogisticRegression(max_iter=2000, C=1.0)
    p_logit = cross_val_predict(lr, Xs, t, cv=3, method="predict_proba")[:, 1]

    intercept_only = np.full(len(t), t.mean())
    return {
        "n_sample": len(sub),
        "max_abs_smd": float(balance["abs_smd"].max()),
        "mean_abs_smd": float(balance["abs_smd"].mean()),
        "n_features_over_0.1_smd": int((balance["abs_smd"] > 0.1).sum()),
        "propensity_auc": float(roc_auc_score(t, p_logit)),
        "propensity_log_loss": float(log_loss(t, p_logit)),
        "intercept_only_log_loss": float(log_loss(t, intercept_only)),
        "log_loss_gap": float(log_loss(t, p_logit) - log_loss(t, intercept_only)),
        "treatment_rate": float(t.mean()),
        "balance_table": balance.to_dict(orient="records"),
    }


def dataset_facts(paths: Paths) -> dict:
    """Archive-level facts used for the artifacts provenance block."""
    manifest = paths.processed_dir / "manifest.json"
    facts = {
        "rows": DATASET_ROWS,
        "sha256": DATASET_SHA256,
        "columns": _COLUMNS,
    }
    if manifest.exists():
        facts.update(json.loads(manifest.read_text()))
    if paths.raw_validation.exists():
        v = json.loads(paths.raw_validation.read_text())
        facts["validation"] = {
            k: v[k] for k in ("rows", "malformed_rows", "treatment_ratio", "conversion_rate",
                              "visit_rate", "exposure_rate", "naive_ate_conversion",
                              "naive_ate_visit", "conv_rate_treated", "conv_rate_control",
                              "exposure_rate_treated", "exposure_rate_control")
            if k in v
        }
    return facts


def ensure_materialized(cfg: Config, force: bool = False) -> dict:
    return materialize(cfg.paths, force=force)


__all__ = [
    "Paths",
    "SplitData",
    "covariate_balance",
    "dataset_facts",
    "ensure_materialized",
    "load_split",
    "materialize",
    "split_mask",
    "verify_randomization",
]
