"""The exposure trap: why the ITT contrast is the only clean one here.

``exposure`` records whether a user was *actually shown* the ad. It is measured
after assignment, and in this dataset ``exposure == 0`` for every single control
user -- controls were prevented from being targeted, so they could not be
exposed. Two consequences:

1. There is no exposed control group. Any contrast that conditions on exposure
   compares exposed treated users against a control group that was never
   eligible for exposure, so it is not a like-for-like comparison.
2. Exposure is downstream of user behaviour (you have to show up to be shown an
   ad), so conditioning on it also conditions on a collider.

Filtering to ``exposure == 1`` therefore mixes the causal effect of the ad with
the selection of who was reachable. This module measures how large that
contamination is, and shows the legitimate way to recover an effect-on-the-
exposed: treat randomization as an instrument and take the Wald ratio. With
one-sided noncompliance (no control unit is ever exposed) that ratio identifies
the complier average causal effect without conditioning on anything post-
treatment.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import SplitData
from .metrics import bootstrap_means

log = logging.getLogger(__name__)


def _rate(y: np.ndarray, mask: np.ndarray) -> float:
    return float(y[mask].mean()) if mask.any() else float("nan")


def _diff_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_draws: int,
             rng: np.random.Generator) -> tuple[float, float, float]:
    """Difference of subgroup means with a bootstrap percentile interval."""
    ya, yb = y[a].astype(np.float64), y[b].astype(np.float64)
    if len(ya) == 0 or len(yb) == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(ya.mean() - yb.mean())
    bs = bootstrap_means(ya, n_draws, rng) - bootstrap_means(yb, n_draws, rng)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def exposure_sensitivity(
    test: SplitData,
    outcome: str = "conversion",
    n_draws: int = 1000,
    seed: int = 0,
) -> dict:
    """Compare the ITT estimate against the exposure-conditioned alternatives."""
    y = test.outcome(outcome)
    t = test.treatment == 1
    e = test.exposure == 1
    rng = np.random.default_rng(seed)

    exposed_treated = t & e
    unexposed_treated = t & ~e
    control = ~t

    def contrast(name: str, a: np.ndarray, b: np.ndarray, note: str) -> dict:
        est, lo, hi = _diff_ci(y, a, b, n_draws, rng)
        return {"name": name, "estimate": est, "ci_lo": lo, "ci_hi": hi,
                "n_a": int(a.sum()), "n_b": int(b.sum()),
                "rate_a": _rate(y, a), "rate_b": _rate(y, b), "note": note}

    itt = contrast(
        "itt_randomized", t, control,
        "Intention-to-treat: contrast on randomized assignment. The primary "
        "estimate. Valid because assignment is random and nothing downstream of "
        "it is conditioned on.",
    )

    cace = _wald_cace(y, t, e, n_draws, rng)

    naive_vs_control = contrast(
        "naive_exposed_vs_all_control", exposed_treated, control,
        "Filters to users who were actually shown the ad and compares them with "
        "the whole control arm. Biased twice over: controls could never be "
        "exposed, and exposure selects on engagement.",
    )

    naive_within_treated = contrast(
        "naive_exposed_vs_unexposed_treated", exposed_treated, unexposed_treated,
        "As-treated comparison inside the treated arm. Exposure is a collider on "
        "the path from user activity to the outcome, so this picks up "
        "selection, not causation.",
    )

    # Why the naive contrasts are biased, made concrete: the exposed subgroup is
    # not comparable to the control arm on the covariates.
    imbalance = _subgroup_imbalance(test, exposed_treated, control)

    return {
        "outcome": outcome,
        "n_test": len(test),
        "exposure_rate_treated": float(e[t].mean()),
        "exposure_rate_control": float(e[control].mean()),
        "one_sided_noncompliance": bool(e[control].sum() == 0),
        "contrasts": [itt, cace, naive_vs_control, naive_within_treated],
        "exposed_vs_control_imbalance": imbalance,
        "inflation_exposed_vs_all_control": _ratio(naive_vs_control["estimate"], itt["estimate"]),
        "inflation_within_treated": _ratio(naive_within_treated["estimate"], itt["estimate"]),
        "n_draws": n_draws,
        "verdict": (
            "The primary analysis uses randomized assignment only. Recovering an "
            "effect-on-the-exposed is done with the Wald/IV ratio, which keeps "
            "randomization as the source of variation instead of conditioning on "
            "a post-treatment variable."
        ),
    }


def _ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _wald_cace(y: np.ndarray, t: np.ndarray, e: np.ndarray, n_draws: int,
               rng: np.random.Generator) -> dict:
    """Complier average causal effect via the Wald ratio ITT_Y / ITT_D.

    With one-sided noncompliance the compliers are exactly the exposed treated
    units, so this is the average causal effect of *being shown the ad* among
    users who would be shown it if targeted -- obtained without ever
    conditioning on exposure.
    """
    y64 = y.astype(np.float64)
    tb = t.astype(np.float64)
    db = e.astype(np.float64)

    def wald(yy: np.ndarray, tt: np.ndarray, dd: np.ndarray) -> float:
        num = yy[tt == 1].mean() - yy[tt == 0].mean()
        den = dd[tt == 1].mean() - dd[tt == 0].mean()
        return float(num / den) if abs(den) > 1e-12 else float("nan")

    est = wald(y64, tb, db)
    n = len(y64)
    bs = np.empty(n_draws)
    for b in range(n_draws):
        idx = rng.integers(0, n, n)
        bs[b] = wald(y64[idx], tb[idx], db[idx])
    finite = bs[np.isfinite(bs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "name": "cace_wald_iv",
        "estimate": est, "ci_lo": float(lo), "ci_hi": float(hi),
        "n_a": int(e.sum()), "n_b": 0,
        "rate_a": float("nan"), "rate_b": float("nan"),
        "note": (
            "Wald ratio of the ITT outcome effect to the ITT effect on exposure. "
            "Identifies the effect of actually being shown the ad among users who "
            "would be exposed if targeted, using randomization as the instrument. "
            "Valid under one-sided noncompliance, which holds exactly here."
        ),
    }


def _subgroup_imbalance(test: SplitData, a: np.ndarray, b: np.ndarray,
                        top_k: int = 12) -> dict:
    """Standardized mean differences between an exposed subgroup and controls."""
    from .config import FEATURE_NAMES

    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        x = test.X[:, j].astype(np.float64)
        xa, xb = x[a], x[b]
        pooled = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2.0)
        smd = float((xa.mean() - xb.mean()) / pooled) if pooled > 0 else 0.0
        rows.append({"feature": name, "mean_exposed": float(xa.mean()),
                     "mean_control": float(xb.mean()), "smd": smd})
    rows.sort(key=lambda r: -abs(r["smd"]))
    return {
        "rows": rows[:top_k],
        "max_abs_smd": float(max(abs(r["smd"]) for r in rows)),
        "mean_abs_smd": float(np.mean([abs(r["smd"]) for r in rows])),
        "interpretation": (
            "Treated vs control is balanced by randomization, but *exposed* "
            "treated vs control is not: exposure is behaviourally selected. Any "
            "|SMD| above ~0.1 is covariate imbalance the naive contrast would "
            "mistake for a treatment effect."
        ),
    }


__all__ = ["exposure_sensitivity"]
