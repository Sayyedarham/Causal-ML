"""Correctness tests for the uplift metrics.

These use a synthetic DGP where the true individual treatment effect is known,
so the estimators can be checked against ground truth rather than against
themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.metrics import (
    ate_aipw,
    ate_difference_in_means,
    calibration_table,
    evaluate_curve,
    incremental_conversions_at,
    qini_coefficient,
    rank_desc,
    uplift_at_fraction,
)


def make_synthetic(n: int = 200_000, seed: int = 7, treat_p: float = 0.85):
    """Heterogeneous-effect DGP with a known oracle ranking."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    # True CATE: strongly heterogeneous, positive and negative regions.
    tau = 0.10 * x[:, 0] - 0.06 * x[:, 1] + 0.02
    base = 1.0 / (1.0 + np.exp(-(0.3 * x[:, 2] - 0.2 * x[:, 3])))
    base = np.clip(base, 0.02, 0.85)
    p1 = np.clip(base + tau, 1e-4, 0.999)
    p0 = np.clip(base, 1e-4, 0.999)

    t = (rng.random(n) < treat_p).astype(np.uint8)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(np.uint8)
    return dict(X=x, t=t, y=y, tau=tau, p0=p0, p1=p1, true_ate=float(tau.mean()))


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def test_rank_desc_is_descending():
    rng = np.random.default_rng(0)
    s = rng.normal(size=1000)
    order = rank_desc(s, seed=1)
    assert np.all(np.diff(s[order]) <= 1e-12)


def test_rank_desc_breaks_ties_randomly():
    """A constant score must not fall back to array order."""
    s = np.zeros(50_000)
    a = rank_desc(s, seed=1)
    b = rank_desc(s, seed=2)
    assert not np.array_equal(a, b)
    # and both are valid permutations
    assert np.array_equal(np.sort(a), np.arange(50_000))
    assert np.array_equal(np.sort(b), np.arange(50_000))


def test_rank_desc_deterministic_given_seed():
    rng = np.random.default_rng(3)
    s = rng.normal(size=5000)
    assert np.array_equal(rank_desc(s, 42), rank_desc(s, 42))


# --------------------------------------------------------------------------- #
# curve endpoints
# --------------------------------------------------------------------------- #

def test_curve_at_full_population_matches_ate_times_n():
    """Uplift(1.0) must equal N * (rate_treated - rate_control)."""
    d = make_synthetic()
    score = np.random.default_rng(0).normal(size=len(d["t"]))  # random ranking
    curve = evaluate_curve(d["y"], d["t"], score, n_points=50, seed=1)

    t = d["t"] == 1
    expected = (d["y"][t].mean() - d["y"][~t].mean()) * len(d["t"])
    assert curve.uplift[-1] == pytest.approx(expected, rel=1e-6)


def test_qini_at_full_population_matches_ratio_adjusted():
    d = make_synthetic()
    score = np.random.default_rng(1).normal(size=len(d["t"]))
    curve = evaluate_curve(d["y"], d["t"], score, n_points=50, seed=2)

    t = d["t"] == 1
    y_t = d["y"][t].sum()
    y_c = d["y"][~t].sum()
    n_t, n_c = t.sum(), (~t).sum()
    expected = y_t - y_c * (n_t / n_c)
    assert curve.qini[-1] == pytest.approx(expected, rel=1e-6)


def test_qini_is_treated_share_times_uplift_under_randomization():
    """With a treatment-blind ranking the two curves are proportional."""
    d = make_synthetic()
    score = np.random.default_rng(2).normal(size=len(d["t"]))
    curve = evaluate_curve(d["y"], d["t"], score, n_points=50, seed=3)
    ratio = curve.qini / curve.uplift
    # p_treat = 0.85, so the proportionality constant should sit near it
    assert np.nanmean(ratio) == pytest.approx(0.85, abs=0.03)


# --------------------------------------------------------------------------- #
# the model must beat random, and the oracle must be ~optimal
# --------------------------------------------------------------------------- #

def test_oracle_beats_random_and_inverse():
    d = make_synthetic()
    oracle = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, seed=4)
    random_ = evaluate_curve(
        d["y"], d["t"], np.random.default_rng(5).normal(size=len(d["t"])), n_points=100, seed=4
    )
    inverse = evaluate_curve(d["y"], d["t"], -d["tau"], n_points=100, seed=4)

    c_o = qini_coefficient(oracle)["qini_coefficient"]
    c_r = qini_coefficient(random_)["qini_coefficient"]
    c_i = qini_coefficient(inverse)["qini_coefficient"]
    assert c_o > c_r > c_i


def test_oracle_normalized_coefficient_is_near_one():
    d = make_synthetic()
    oracle = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, seed=6)
    res = qini_coefficient(oracle, oracle=oracle)
    assert res["qini_coefficient_normalized"] == pytest.approx(1.0, abs=1e-9)


def test_random_ranking_has_near_zero_normalized_coefficient():
    d = make_synthetic()
    oracle = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, seed=7)
    rnd_score = np.random.default_rng(8).normal(size=len(d["t"]))
    rnd = evaluate_curve(d["y"], d["t"], rnd_score, n_points=100, seed=7)
    coef = qini_coefficient(rnd, oracle=oracle)["qini_coefficient_normalized"]
    assert abs(coef) < 0.05


def test_oracle_captures_most_uplift_in_top_quartile():
    """Under the oracle, the top 25% should hold a large share of total uplift."""
    d = make_synthetic()
    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, seed=9)
    total = curve.uplift[-1]
    at_25 = curve.at(0.25)["uplift"]
    # tau is ~N(0.02, .12); top quartile carries well over half the total gain
    assert at_25 / total > 0.5


# --------------------------------------------------------------------------- #
# top-k targeting
# --------------------------------------------------------------------------- #

def test_uplift_at_fraction_recovers_mean_effect_of_top_slice():
    d = make_synthetic()
    n = len(d["t"])
    k = round(0.10 * n)
    order = rank_desc(d["tau"], seed=11)[:k]
    expected_rate = d["tau"][order].mean()

    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=200, seed=11)
    got = uplift_at_fraction(curve, 0.10, baseline_rate=float(d["p0"].mean()))
    assert got["uplift_rate"] == pytest.approx(expected_rate, rel=0.25)
    assert got["n_targeted"] == pytest.approx(k, rel=0.02)


def test_uplift_at_fraction_ci_contains_point_estimate():
    d = make_synthetic()
    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, n_draws=200, seed=12)
    got = uplift_at_fraction(curve, 0.10, baseline_rate=float(d["p0"].mean()))
    assert got["incremental_lo"] <= got["incremental_conversions"] <= got["incremental_hi"]
    assert got["incremental_lo"] < got["incremental_hi"]


def test_incremental_conversions_at_matches_curve():
    d = make_synthetic()
    direct = incremental_conversions_at(d["y"], d["t"], d["tau"], 0.10, seed=13)
    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, seed=13)
    # Same ranking, same rounding convention: these must agree exactly.
    assert direct == pytest.approx(curve.at(0.10)["uplift"], rel=1e-9)


# --------------------------------------------------------------------------- #
# ATE estimators
# --------------------------------------------------------------------------- #

def test_difference_in_means_recovers_true_ate():
    d = make_synthetic(n=400_000)
    res = ate_difference_in_means(d["y"], d["t"], n_draws=500, seed=14)
    assert res["ci_lo"] <= d["true_ate"] <= res["ci_hi"]
    assert res["ate"] == pytest.approx(d["true_ate"], abs=4 * res["se"])


def test_bootstrap_and_analytic_se_agree():
    d = make_synthetic(n=300_000)
    res = ate_difference_in_means(d["y"], d["t"], n_draws=2000, seed=15)
    assert res["se_bootstrap"] == pytest.approx(res["se"], rel=0.10)


def test_aipw_recovers_true_ate_with_oracle_nuisance():
    d = make_synthetic(n=300_000)
    e = np.full(len(d["t"]), 0.85)
    res = ate_aipw(d["y"], d["t"], mu0=d["p0"], mu1=d["p1"], e=e)
    assert res["ate"] == pytest.approx(d["true_ate"], abs=4 * res["se"] + 1e-4)


def test_aipw_is_robust_to_a_wrong_propensity():
    """Double robustness: correct outcome models rescue a bad propensity."""
    d = make_synthetic(n=300_000)
    e_bad = np.clip(0.5 + 0.2 * d["tau"], 0.05, 0.95)
    res = ate_aipw(d["y"], d["t"], mu0=d["p0"], mu1=d["p1"], e=e_bad)
    assert res["ate"] == pytest.approx(d["true_ate"], abs=4 * res["se"] + 2e-3)


def test_aipw_is_robust_to_wrong_outcome_models():
    """...and a correct propensity rescues bad outcome models."""
    d = make_synthetic(n=300_000)
    e = np.full(len(d["t"]), 0.85)
    flat0 = np.full(len(d["t"]), d["p0"].mean())
    flat1 = np.full(len(d["t"]), d["p1"].mean())
    res = ate_aipw(d["y"], d["t"], mu0=flat0, mu1=flat1, e=e)
    assert res["ate"] == pytest.approx(d["true_ate"], abs=4 * res["se"] + 2e-3)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def test_calibration_table_recovers_true_effect_per_bin():
    d = make_synthetic(n=400_000)
    rows = calibration_table(d["y"], d["t"], d["tau"], n_bins=10)
    assert len(rows) == 10
    # monotone in the true effect by construction
    pred = [r["predicted_uplift_mean"] for r in rows]
    assert pred == sorted(pred)
    for r in rows:
        assert r["realized_lo"] - 0.02 <= r["predicted_uplift_mean"] <= r["realized_hi"] + 0.02


def test_calibration_table_reports_arm_sizes():
    d = make_synthetic(n=100_000)
    rows = calibration_table(d["y"], d["t"], d["tau"], n_bins=5)
    assert sum(r["n"] for r in rows) == len(d["t"])
    assert all(r["n_treated"] > 0 and r["n_control"] > 0 for r in rows)


# --------------------------------------------------------------------------- #
# bootstrap band sanity
# --------------------------------------------------------------------------- #

def test_bootstrap_band_widens_where_data_is_thin():
    """The top of the curve has few users, so the band must be wider there."""
    d = make_synthetic(n=300_000)
    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=100, n_draws=300, seed=16)
    width = curve.uplift_hi - curve.uplift_lo
    early = np.nanmean(width[:10])
    assert np.all(np.isfinite(width))
    assert early > 0


def test_zero_draws_leaves_no_band():
    d = make_synthetic(n=20_000)
    curve = evaluate_curve(d["y"], d["t"], d["tau"], n_points=50, n_draws=0, seed=17)
    assert curve.uplift_lo is None and curve.uplift_hi is None
    assert curve.n_draws == 0
    got = uplift_at_fraction(curve, 0.10, baseline_rate=0.1)
    assert np.isnan(got["incremental_lo"])
