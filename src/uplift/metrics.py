"""Uplift evaluation: Qini curves, uplift curves, ATE estimators, bootstrap CIs.

Definitions used throughout (population of size N, ranked by descending score):

    For the top fraction phi, let
        N_T(phi), N_C(phi)  treated / control users targeted
        Y_T(phi), Y_C(phi)  conversions among them

    Qini(phi)   = Y_T(phi) - Y_C(phi) * N_T(phi) / N_C(phi)
    Uplift(phi) = (Y_T/N_T - Y_C/N_C) * (N_T + N_C)

Both are "incremental conversions earned by targeting the top phi", differing
only in how they re-weight the arms. Under randomization with a ranking that is
independent of assignment the treated share is ~constant in phi, so
Qini(phi) ~= p_treat * Uplift(phi). We report both: Qini is the convention in
the uplift literature, Uplift is the quantity a campaign budget consumes.

Bootstrap resamples rows with replacement. Curves are recomputed on the
*original* score ordering, so a draw reduces to reweighting: if w_i is the
number of times row i appears, every cumulative sum becomes a weighted sum.
That makes a draw O(N) with no re-sorting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def rank_desc(score: np.ndarray, seed: int) -> np.ndarray:
    """Indices that sort ``score`` descending, ties broken uniformly at random.

    Random tie-breaking is not cosmetic: a constant-score baseline has every row
    tied, and a deterministic tiebreak would silently order rows by array
    position, which correlates with nothing but can still bias the curve.
    """
    jitter = np.random.default_rng(seed).random(score.shape[0])
    return np.lexsort((jitter, -score.astype(np.float64)))


def grid_indices(n: int, n_points: int) -> np.ndarray:
    """Row positions (0-based, into the ranked list) at each curve breakpoint."""
    phi = np.linspace(1.0 / n_points, 1.0, n_points)
    idx = np.ceil(phi * n).astype(np.int64) - 1
    return np.clip(idx, 0, n - 1)


# --------------------------------------------------------------------------- #
# cumulative tables
# --------------------------------------------------------------------------- #

def cumulative_counts(y: np.ndarray, t: np.ndarray, order: np.ndarray) -> dict[str, np.ndarray]:
    """Full-resolution cumulative arm sizes and conversions along ``order``."""
    ys = y[order].astype(np.float64)
    ts = t[order].astype(np.float64)
    cs = 1.0 - ts
    return {
        "n_t": np.cumsum(ts),
        "n_c": np.cumsum(cs),
        "y_t": np.cumsum(ts * ys),
        "y_c": np.cumsum(cs * ys),
    }


def _curve_values(y_t: np.ndarray, y_c: np.ndarray, n_t: np.ndarray,
                  n_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(uplift, qini) from cumulative arm counts, guarding empty arms."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(n_t > 0, y_t / np.maximum(n_t, 1e-12), np.nan)
        rate_c = np.where(n_c > 0, y_c / np.maximum(n_c, 1e-12), np.nan)
        ratio = np.where(n_c > 0, n_t / np.maximum(n_c, 1e-12), np.nan)
    uplift = (rate_t - rate_c) * (n_t + n_c)
    qini = y_t - y_c * ratio
    return uplift, qini


# --------------------------------------------------------------------------- #
# curve container
# --------------------------------------------------------------------------- #

@dataclass
class CurveEstimate:
    """Point estimate plus bootstrap percentile band for a ranked policy."""

    phi: np.ndarray
    n_targeted: np.ndarray
    uplift: np.ndarray
    qini: np.ndarray
    uplift_lo: np.ndarray | None = None
    uplift_hi: np.ndarray | None = None
    qini_lo: np.ndarray | None = None
    qini_hi: np.ndarray | None = None
    rate_treated: np.ndarray = field(default_factory=lambda: np.empty(0))
    rate_control: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_draws: int = 0
    confidence: float = 0.95

    def at(self, phi: float) -> dict[str, float]:
        """Nearest-breakpoint metrics at an arbitrary targeting fraction.

        Nearest-neighbour rather than ``searchsorted``: the grid comes from
        ``linspace``, so a requested phi such as 0.10 can land a float ulp above
        its own grid value and silently resolve to the next breakpoint.
        """
        j = int(np.argmin(np.abs(self.phi - phi)))

        def pick(a: np.ndarray | None) -> float:
            return float("nan") if a is None else float(a[j])

        return {
            "phi": float(self.phi[j]),
            "n_targeted": float(self.n_targeted[j]),
            "uplift": float(self.uplift[j]),
            "qini": float(self.qini[j]),
            "uplift_lo": pick(self.uplift_lo),
            "uplift_hi": pick(self.uplift_hi),
            "qini_lo": pick(self.qini_lo),
            "qini_hi": pick(self.qini_hi),
            "rate_treated": float(self.rate_treated[j]) if self.rate_treated.size else float("nan"),
            "rate_control": float(self.rate_control[j]) if self.rate_control.size else float("nan"),
        }


def evaluate_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_points: int = 200,
    n_draws: int = 0,
    seed: int = 0,
    confidence: float = 0.95,
) -> CurveEstimate:
    """Point estimate and (optionally) bootstrap band for one ranking."""
    n = len(y)
    order = rank_desc(score, seed)
    phi = np.linspace(1.0 / n_points, 1.0, n_points)
    gi = grid_indices(n, n_points)

    cum = cumulative_counts(y, t, order)
    n_t, n_c = cum["n_t"][gi], cum["n_c"][gi]
    y_t, y_c = cum["y_t"][gi], cum["y_c"][gi]
    uplift, qini = _curve_values(y_t, y_c, n_t, n_c)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(n_t > 0, y_t / np.maximum(n_t, 1e-12), np.nan)
        rate_c = np.where(n_c > 0, y_c / np.maximum(n_c, 1e-12), np.nan)

    est = CurveEstimate(
        phi=phi, n_targeted=n_t + n_c, uplift=uplift, qini=qini,
        rate_treated=rate_t, rate_control=rate_c, n_draws=0, confidence=confidence,
    )
    if n_draws > 0:
        _attach_bootstrap(est, y, t, order, gi, n_draws, seed, confidence)
    return est


def _attach_bootstrap(
    est: CurveEstimate, y: np.ndarray, t: np.ndarray, order: np.ndarray,
    gi: np.ndarray, n_draws: int, seed: int, confidence: float,
) -> None:
    n = len(y)
    ys = y[order].astype(np.float64)
    ts = t[order].astype(np.float64)
    cs = 1.0 - ts
    # Pre-multiply once; every draw then only needs a weight vector.
    a_t, a_c = ts, cs
    a_yt, a_yc = ts * ys, cs * ys

    # Segment id of each ranked row, so a draw accumulates straight into the
    # grid cells instead of materializing an n-length cumulative sum.
    seg = np.searchsorted(gi, np.arange(n), side="left")
    seg = np.minimum(seg, len(gi) - 1)

    P = len(gi)
    draws = np.empty((n_draws, P, 4), dtype=np.float64)
    rng = np.random.default_rng(seed ^ 0x5EED)
    alpha = (1.0 - confidence) / 2.0

    for b in range(n_draws):
        # Multinomial weights via bincount: drawing n indices and counting is
        # ~100x faster than rng.multinomial(n, 1/n), which is O(n log n) per draw.
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(np.float64)
        draws[b, :, 0] = np.bincount(seg, weights=w * a_t, minlength=P)
        draws[b, :, 1] = np.bincount(seg, weights=w * a_c, minlength=P)
        draws[b, :, 2] = np.bincount(seg, weights=w * a_yt, minlength=P)
        draws[b, :, 3] = np.bincount(seg, weights=w * a_yc, minlength=P)
        if (b + 1) % max(n_draws // 5, 1) == 0:
            log.debug("bootstrap curve draw %d/%d", b + 1, n_draws)

    cum = np.cumsum(draws, axis=1)
    n_t, n_c, y_t, y_c = cum[:, :, 0], cum[:, :, 1], cum[:, :, 2], cum[:, :, 3]
    up = np.empty((n_draws, P))
    qi = np.empty((n_draws, P))
    for b in range(n_draws):
        up[b], qi[b] = _curve_values(y_t[b], y_c[b], n_t[b], n_c[b])

    est.uplift_lo = np.nanpercentile(up, 100 * alpha, axis=0)
    est.uplift_hi = np.nanpercentile(up, 100 * (1 - alpha), axis=0)
    est.qini_lo = np.nanpercentile(qi, 100 * alpha, axis=0)
    est.qini_hi = np.nanpercentile(qi, 100 * (1 - alpha), axis=0)
    est.n_draws = n_draws


# --------------------------------------------------------------------------- #
# scalar summaries
# --------------------------------------------------------------------------- #

def qini_coefficient(curve: CurveEstimate, oracle: CurveEstimate | None = None) -> dict[str, float]:
    """Area between the model Qini curve and the random-targeting diagonal.

    ``oracle`` is optional. When supplied the coefficient is additionally
    normalized by the achievable maximum, which is only knowable when true
    individual effects exist (synthetic data and unit tests).
    """
    area = float(np.trapezoid(curve.qini, curve.phi))
    # Random targeting is the straight line from (0,0) to (1, Qini_total).
    random_area = float(curve.qini[-1] / 2.0)
    out = {"qini_area": area, "random_area": random_area,
           "qini_coefficient": area - random_area, "qini_total": float(curve.qini[-1])}
    if oracle is not None:
        oracle_area = float(np.trapezoid(oracle.qini, oracle.phi))
        denom = oracle_area - random_area
        out["oracle_area"] = oracle_area
        out["qini_coefficient_normalized"] = (
            float((area - random_area) / denom) if abs(denom) > 1e-12 else float("nan")
        )
    return out


def uplift_at_fraction(curve: CurveEstimate, phi: float, baseline_rate: float) -> dict[str, float]:
    """Headline targeting metric at one budget fraction, with its CI."""
    p = curve.at(phi)
    n = p["n_targeted"]
    rate = (p["rate_treated"] - p["rate_control"]) if n > 0 else float("nan")

    def scaled(lo_key: str, hi_key: str) -> tuple[float, float]:
        lo, hi = p[lo_key], p[hi_key]
        if np.isnan(lo) or np.isnan(hi) or n <= 0:
            return float("nan"), float("nan")
        # The band is on incremental conversions; convert it to the same units
        # as the point estimate by dividing by the targeted population.
        return lo / n, hi / n

    rate_lo, rate_hi = scaled("uplift_lo", "uplift_hi")
    return {
        "phi": p["phi"],
        "n_targeted": n,
        "incremental_conversions": p["uplift"],
        "incremental_lo": p["uplift_lo"],
        "incremental_hi": p["uplift_hi"],
        "uplift_rate": rate,
        "uplift_rate_lo": rate_lo,
        "uplift_rate_hi": rate_hi,
        "baseline_rate": baseline_rate,
        "relative_lift": (rate / baseline_rate) if baseline_rate > 0 else float("nan"),
        "qini": p["qini"],
        "qini_lo": p["qini_lo"],
        "qini_hi": p["qini_hi"],
    }


def bootstrap_means(
    y: np.ndarray, n_draws: int, rng: np.random.Generator,
    budget_bytes: int = 256 << 20,
) -> np.ndarray:
    """Bootstrap means of one array, resampling rows with replacement.

    Chunked so the index and gather buffers stay inside ``budget_bytes``: on the
    multi-million-row test split a whole ``(n_draws, n)`` index matrix would
    need tens of GB, while the result only needs ``n_draws`` floats.
    """
    n = len(y)
    chunk = max(1, int(budget_bytes // (16 * n)))
    out = np.empty(n_draws)
    for s in range(0, n_draws, chunk):
        k = min(chunk, n_draws - s)
        idx = rng.integers(0, n, (k, n))
        out[s:s + k] = y[idx].mean(axis=1)
    return out


def ate_difference_in_means(
    y: np.ndarray, t: np.ndarray, n_draws: int = 1000, seed: int = 0, confidence: float = 0.95,
) -> dict[str, float]:
    """Randomization (ITT) estimator with bootstrap and analytic standard errors."""
    t_b = t == 1
    y1, y0 = y[t_b].astype(np.float64), y[~t_b].astype(np.float64)
    m1, m0 = y1.mean(), y0.mean()
    ate = m1 - m0

    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se_analytic = float(np.sqrt(v1 + v0))
    alpha = (1.0 - confidence) / 2.0

    rng = np.random.default_rng(seed ^ 0xA7E)
    n1, n0 = len(y1), len(y0)
    # Resample each arm independently: equivalent to the full nonparametric
    # bootstrap for a difference of arm means, but touches only what it needs.
    bs = bootstrap_means(y1, n_draws, rng) - bootstrap_means(y0, n_draws, rng)
    lo, hi = np.percentile(bs, [100 * alpha, 100 * (1 - alpha)])

    return {
        "estimator": "difference_in_means",
        "ate": float(ate), "se": se_analytic, "se_bootstrap": float(bs.std(ddof=1)),
        "ci_lo": float(lo), "ci_hi": float(hi), "confidence": confidence,
        "n_treated": int(n1), "n_control": int(n0),
        "rate_treated": float(m1), "rate_control": float(m0),
        "relative_lift": float(ate / m0) if m0 > 0 else float("nan"),
        "n_draws": n_draws,
    }


def ate_aipw(
    y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray,
    e_clip: float = 0.01,
) -> dict[str, float]:
    """Augmented IPW (doubly robust) ATE.

    Consistent if either the outcome models or the propensity model is correct.
    Under a known randomized propensity this is the semiparametric-efficient
    estimator, so the gap against difference-in-means is itself diagnostic.
    """
    y = y.astype(np.float64)
    tb = t.astype(np.float64)
    ec = np.clip(e.astype(np.float64), e_clip, 1.0 - e_clip)
    psi = (mu1 - mu0) + tb * (y - mu1) / ec - (1.0 - tb) * (y - mu0) / (1.0 - ec)
    n = len(psi)
    se = float(psi.std(ddof=1) / np.sqrt(n))
    return {
        "estimator": "aipw",
        "ate": float(psi.mean()), "se": se,
        "ci_lo": float(psi.mean() - 1.959963985 * se),
        "ci_hi": float(psi.mean() + 1.959963985 * se),
        "n": n, "e_clip": e_clip,
        "propensity_min": float(e.min()), "propensity_max": float(e.max()),
    }


def calibration_table(
    y: np.ndarray, t: np.ndarray, predicted: np.ndarray, n_bins: int = 10,
) -> list[dict[str, float]]:
    """Realized vs predicted uplift within bins of predicted uplift.

    A well-calibrated uplift model has realized ~= predicted in every bin. This
    is the check that separates a model that merely *orders* users from one
    whose magnitudes can be plugged into a budget calculation.
    """
    # Equal-count bins; fall back to unique edges for degenerate predictions.
    qs = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return []
    edges = np.concatenate(([predicted.min() - 1e-9], qs[1:-1], [predicted.max() + 1e-9]))
    b = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        nk = int(m.sum())
        if nk == 0:
            continue
        tk = t[m] == 1
        yk = y[m].astype(np.float64)
        n_t, n_c = int(tk.sum()), int((~tk).sum())
        if n_t == 0 or n_c == 0:
            continue
        r_t, r_c = yk[tk].mean(), yk[~tk].mean()
        se = np.sqrt(
            yk[tk].var(ddof=1) / n_t + yk[~tk].var(ddof=1) / n_c
        )
        rows.append({
            "bin": k, "n": nk, "n_treated": n_t, "n_control": n_c,
            "predicted_uplift_mean": float(predicted[m].mean()),
            "realized_uplift": float(r_t - r_c),
            "realized_se": float(se),
            "realized_lo": float(r_t - r_c - 1.959963985 * se),
            "realized_hi": float(r_t - r_c + 1.959963985 * se),
            "rate_treated": float(r_t), "rate_control": float(r_c),
            "predicted_lo": float(predicted[m].min()),
            "predicted_hi": float(predicted[m].max()),
        })
    return rows


def incremental_conversions_at(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, phi: float, seed: int = 0,
) -> float:
    """Unbiased incremental conversions if the top ``phi`` fraction is targeted."""
    n = len(y)
    # Same rounding convention as grid_indices so this matches evaluate_curve.
    k = int(np.clip(np.ceil(phi * n), 1, n))
    order = rank_desc(score, seed)[:k]
    tk = t[order] == 1
    yk = y[order].astype(np.float64)
    if not tk.any() or tk.all():
        return float("nan")
    return float((yk[tk].mean() - yk[~tk].mean()) * k)


__all__ = [
    "CurveEstimate",
    "ate_aipw",
    "ate_difference_in_means",
    "bootstrap_means",
    "calibration_table",
    "cumulative_counts",
    "evaluate_curve",
    "grid_indices",
    "incremental_conversions_at",
    "qini_coefficient",
    "rank_desc",
    "uplift_at_fraction",
]
