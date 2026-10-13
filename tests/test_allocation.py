"""Tests for budget allocation.

The central check is analytical: with tau ~ Uniform(0, a) the profit-maximizing
rule "target while v*tau > c" gives phi* = 1 - (c/v)/a exactly. The measured
profit curve built from a held-out sample must peak near that value, which ties
the whole allocation layer back to a closed-form answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.allocation import (
    allocate,
    best_policy_at_budget,
    breakeven_cost_per_impression,
    optimal_point,
    policy_comparison,
    summarize,
    threshold_rule_phi,
)
from uplift.metrics import CurveEstimate, evaluate_curve


def uniform_tau_dgp(n: int = 200_000, seed: int = 5, a: float = 0.10,
                    base: float = 0.05, treat_p: float = 0.5):
    """Known CATE distribution so the optimal budget can be derived in closed form."""
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.0, a, size=n)
    t = (rng.random(n) < treat_p).astype(np.uint8)
    p0 = np.full(n, base)
    p1 = base + tau
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(np.uint8)
    return y, t, tau


def closed_form_phi_star(a: float, cost: float, value: float) -> float:
    """argmax_phi of  value * N * int_top-phi tau  -  cost * phi * N."""
    thr = cost / value
    return float(np.clip(1.0 - thr / a, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# the optimum must match theory
# --------------------------------------------------------------------------- #

def test_threshold_rule_matches_closed_form():
    tau = np.linspace(0.0, 0.10, 100_001)
    got = threshold_rule_phi(tau, cost_per_impression=0.05, value_per_conversion=1.0)
    assert got["uplift_threshold"] == pytest.approx(0.05)
    assert got["phi"] == pytest.approx(closed_form_phi_star(0.10, 0.05, 1.0), abs=1e-4)
    assert got["share_positive_effect"] == pytest.approx(1.0, abs=1e-4)


def analytic_curve(a: float, n_test: int = 100_000, n_points: int = 100) -> CurveEstimate:
    """Exact uplift curve for tau ~ Uniform(0, a), with no sampling noise.

    Ranking by the true tau, the top phi slice has tau falling linearly from a to
    a(1-phi), so its mean effect is a(1 - phi/2) and the incremental conversions
    in a sample of size n_test are a(1 - phi/2) * phi * n_test.

    n_points=100 puts breakpoints on exact hundredths, so closed-form optima
    like 0.50 and 0.75 land on the grid and can be compared to machine precision.
    """
    phi = np.linspace(1.0 / n_points, 1.0, n_points)
    n_targeted = phi * n_test
    uplift = a * (1.0 - phi / 2.0) * n_targeted
    return CurveEstimate(phi=phi, n_targeted=n_targeted, uplift=uplift, qini=0.5 * uplift)


def test_allocation_math_matches_closed_form_exactly():
    """Deterministic check of the optimization, free of sampling noise."""
    curve = analytic_curve(a=0.10)
    alloc = allocate(curve, population=1_000_000,
                     cost_per_impression=0.05, value_per_conversion=1.0)
    opt = optimal_point(alloc)
    assert opt["phi"] == pytest.approx(closed_form_phi_star(0.10, 0.05, 1.0), abs=1e-9)

    # Scaling the value rescales the optimum exactly as the rule v*tau > c implies.
    opt2 = optimal_point(allocate(curve, 1_000_000, 0.05, 2.0))
    assert opt2["phi"] == pytest.approx(closed_form_phi_star(0.10, 0.025, 1.0), abs=1e-9)

    # Cost above the maximum effect makes targeting unprofitable everywhere.
    opt3 = optimal_point(allocate(curve, 1_000_000, 0.50, 1.0))
    assert opt3["profit"] <= 0


def test_analytic_profit_curve_matches_hand_computed_value():
    curve = analytic_curve(a=0.10)
    v, c, N = 2.0, 0.01, 500_000
    alloc = allocate(curve, N, c, v)
    # profit(phi) = N*phi*(v*a*(1-phi/2) - c)
    expected = N * alloc.phi * (v * 0.10 * (1 - alloc.phi / 2) - c)
    assert np.allclose(alloc.profit, expected, rtol=1e-9)


def test_measured_profit_curve_peaks_near_closed_form():
    """Same optimum, but recovered from sampled data rather than an exact curve.

    The argmax is a weak statistic here -- the curve is flat at its peak, so
    sampling noise moves the location far more than the height. Assert on the
    profit *level* at phi* (sharp) and keep the location check loose enough to
    catch only gross errors like sign flips or unit mistakes.
    """
    y, t, tau = uniform_tau_dgp()
    curve = evaluate_curve(y, t, tau, n_points=100, seed=3)
    alloc = allocate(curve, population=1_000_000,
                     cost_per_impression=0.05, value_per_conversion=1.0)
    opt = optimal_point(alloc)
    expected = closed_form_phi_star(0.10, 0.05, 1.0)

    j_star = int(np.argmin(np.abs(alloc.phi - expected)))
    profit_at_star = float(alloc.profit[j_star])
    assert profit_at_star == pytest.approx(opt["profit"], rel=0.05)
    assert opt["phi"] == pytest.approx(expected, abs=0.10)


def test_optimum_reports_a_plateau_containing_the_argmax():
    y, t, tau = uniform_tau_dgp()
    curve = evaluate_curve(y, t, tau, n_points=100, seed=31)
    alloc = allocate(curve, 1_000_000, 0.05, 1.0)
    opt = optimal_point(alloc)
    assert opt["plateau_phi_lo"] <= opt["phi"] <= opt["plateau_phi_hi"]
    assert opt["plateau_phi_lo"] < opt["plateau_phi_hi"]
    # Everything in the plateau is within tolerance of the best profit.
    inside = (alloc.phi >= opt["plateau_phi_lo"]) & (alloc.phi <= opt["plateau_phi_hi"])
    assert np.all(alloc.profit[inside] >= opt["profit"] * (1 - opt["plateau_tolerance"]) - 1e-9)
    assert opt["plateau_population_lo"] <= opt["plateau_population_hi"]


def test_partial_targeting_beats_targeting_everyone():
    """Below the threshold the marginal user costs more than they return."""
    y, t, tau = uniform_tau_dgp()
    curve = evaluate_curve(y, t, tau, n_points=100, seed=4)
    alloc = allocate(curve, population=1_000_000,
                     cost_per_impression=0.05, value_per_conversion=1.0)
    opt = optimal_point(alloc)
    assert opt["profit"] > alloc.profit[-1]
    assert opt["phi"] < 1.0


def test_higher_cost_shrinks_the_optimal_budget():
    y, t, tau = uniform_tau_dgp()
    curve = evaluate_curve(y, t, tau, n_points=100, seed=5)
    cheap = optimal_point(allocate(curve, 1_000_000, 0.02, 1.0))
    dear = optimal_point(allocate(curve, 1_000_000, 0.08, 1.0))
    assert cheap["phi"] > dear["phi"]
    assert dear["phi"] == pytest.approx(closed_form_phi_star(0.10, 0.08, 1.0), abs=0.04)


def test_higher_conversion_value_expands_the_optimal_budget():
    y, t, tau = uniform_tau_dgp()
    curve = evaluate_curve(y, t, tau, n_points=100, seed=6)
    low = optimal_point(allocate(curve, 1_000_000, 0.05, 1.0))
    high = optimal_point(allocate(curve, 1_000_000, 0.05, 2.0))
    assert high["phi"] > low["phi"]


def test_zero_cost_targets_everyone():
    y, t, tau = uniform_tau_dgp(n=80_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=7)
    alloc = allocate(curve, 500_000, cost_per_impression=1e-9, value_per_conversion=1.0)
    assert optimal_point(alloc)["phi"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# mechanics
# --------------------------------------------------------------------------- #

def test_spend_is_linear_in_phi():
    y, t, tau = uniform_tau_dgp(n=50_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=8)
    alloc = allocate(curve, population=1_000_000, cost_per_impression=0.01,
                     value_per_conversion=1.0)
    expected = 0.01 * alloc.phi * 1_000_000
    assert np.allclose(alloc.spend, expected)


def test_population_scaling_is_linear():
    y, t, tau = uniform_tau_dgp(n=60_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=9)
    small = allocate(curve, 100_000, 0.01, 1.0)
    big = allocate(curve, 1_000_000, 0.01, 1.0)
    ratio = big.incremental_conversions / np.where(
        np.abs(small.incremental_conversions) > 1e-9, small.incremental_conversions, np.nan)
    assert np.nanmean(ratio) == pytest.approx(10.0, rel=1e-6)


def test_profit_equals_value_times_incremental_minus_spend():
    y, t, tau = uniform_tau_dgp(n=60_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=10)
    v, c = 3.0, 0.02
    alloc = allocate(curve, 250_000, c, v)
    assert np.allclose(alloc.profit, v * alloc.incremental_conversions - alloc.spend)


def test_roi_is_profit_over_spend():
    y, t, tau = uniform_tau_dgp(n=60_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=11)
    alloc = allocate(curve, 250_000, 0.02, 1.0)
    ok = alloc.spend > 0
    assert np.allclose(alloc.roi[ok], alloc.profit[ok] / alloc.spend[ok])


def test_bootstrap_band_propagates_into_profit():
    y, t, tau = uniform_tau_dgp(n=80_000)
    curve = evaluate_curve(y, t, tau, n_points=50, n_draws=150, seed=12)
    alloc = allocate(curve, 500_000, 0.05, 1.0)
    assert np.all(alloc.profit_lo <= alloc.profit + 1e-9)
    assert np.all(alloc.profit <= alloc.profit_hi + 1e-9)
    assert np.all(alloc.profit_lo < alloc.profit_hi)


def test_allocate_rejects_bad_inputs():
    y, t, tau = uniform_tau_dgp(n=5_000)
    curve = evaluate_curve(y, t, tau, n_points=20, seed=13)
    with pytest.raises(ValueError, match="positive"):
        allocate(curve, 1000, cost_per_impression=0.0, value_per_conversion=1.0)
    with pytest.raises(ValueError, match="positive"):
        allocate(curve, 1000, cost_per_impression=0.01, value_per_conversion=-1.0)
    with pytest.raises(ValueError, match="population"):
        allocate(curve, 0, cost_per_impression=0.01, value_per_conversion=1.0)


def test_breakeven_cost_equals_revenue_per_impression_at_optimum():
    y, t, tau = uniform_tau_dgp(n=80_000)
    curve = evaluate_curve(y, t, tau, n_points=50, seed=14)
    alloc = allocate(curve, 500_000, 0.02, 1.0)
    j = int(np.argmax(np.nan_to_num(alloc.profit, nan=-np.inf)))
    be = breakeven_cost_per_impression(alloc)
    expected = 1.0 * alloc.incremental_conversions[j] / alloc.population_targeted[j]
    assert be > 0
    assert be == pytest.approx(expected, rel=1e-9)
    # Break-even must exceed the assumed cost, otherwise the optimum would lose money.
    assert be > 0.02


# --------------------------------------------------------------------------- #
# policy comparison
# --------------------------------------------------------------------------- #

def test_perfect_policy_beats_random_at_matched_budget():
    y, t, tau = uniform_tau_dgp(n=150_000, seed=17)
    rng = np.random.default_rng(18)
    # n_points=100 puts the grid on 0.01 steps so the requested budgets are exact.
    curves = {
        "oracle": evaluate_curve(y, t, tau, n_points=100, seed=19),
        "random": evaluate_curve(y, t, rng.normal(size=len(tau)), n_points=100, seed=19),
        "perverse": evaluate_curve(y, t, -tau, n_points=100, seed=19),
    }
    rows = policy_comparison(curves, population=500_000, cost_per_impression=0.05,
                             value_per_conversion=1.0, budget_fractions=(0.10, 0.25, 0.50))
    by = {(r["policy"], round(r["phi"], 3)): r for r in rows}
    for phi in (0.10, 0.25, 0.50):
        o = by[("oracle", phi)]
        r = by[("random", phi)]
        p = by[("perverse", phi)]
        # Same spend, same reach: the gap is purely who got chosen.
        assert o["budget"] == pytest.approx(r["budget"], rel=1e-6)
        assert o["population_targeted"] == pytest.approx(r["population_targeted"], rel=0.05)
        assert o["incremental_conversions"] > r["incremental_conversions"] > p["incremental_conversions"]


def test_best_policy_at_budget_selects_the_winner():
    y, t, tau = uniform_tau_dgp(n=120_000, seed=20)
    rng = np.random.default_rng(21)
    curves = {
        "oracle": evaluate_curve(y, t, tau, n_points=50, seed=22),
        "random": evaluate_curve(y, t, rng.normal(size=len(tau)), n_points=50, seed=22),
    }
    rows = policy_comparison(curves, 500_000, 0.05, 1.0, budget_fractions=(0.10, 0.30))
    best = best_policy_at_budget(rows, 0.10)
    assert best is not None and best["policy"] == "oracle"
    assert best_policy_at_budget([], 0.1) is None


def test_summarize_has_everything_the_dashboard_needs():
    y, t, tau = uniform_tau_dgp(n=100_000, seed=23)
    curve = evaluate_curve(y, t, tau, n_points=50, n_draws=100, seed=24)
    out = summarize(curve, tau, population=1_000_000,
                    cost_per_impression=0.05, value_per_conversion=1.0)
    assert set(out) == {"allocation", "optimum", "threshold_rule",
                        "breakeven_cost_per_impression", "target_everyone"}
    assert out["optimum"]["phi"] == pytest.approx(closed_form_phi_star(0.10, 0.05, 1.0), abs=0.05)
    # Model-based rule and measured-optimum should agree, since the model here is
    # the oracle and its magnitudes are correct.
    assert out["threshold_rule"]["phi"] == pytest.approx(out["optimum"]["phi"], abs=0.05)
    for key in ("phi", "spend", "profit", "incremental_conversions", "roi"):
        assert key in out["allocation"]
        assert len(out["allocation"][key]) == 50


def test_allocate_without_bootstrap_yields_nan_bands():
    y, t, tau = uniform_tau_dgp(n=40_000, seed=25)
    curve = evaluate_curve(y, t, tau, n_points=40, n_draws=0, seed=26)
    alloc = allocate(curve, 200_000, 0.05, 1.0)
    assert np.all(np.isnan(alloc.incremental_lo))
    assert np.all(np.isnan(alloc.profit_lo))
    opt = optimal_point(alloc)
    assert np.isnan(opt["profit_lo"]) and np.isnan(opt["incremental_lo"])


def test_allocation_result_serializes_with_nulls_not_nans():
    """JSON has no NaN; non-finite values must become null."""
    import json

    y, t, tau = uniform_tau_dgp(n=40_000, seed=27)
    curve = evaluate_curve(y, t, tau, n_points=40, n_draws=0, seed=28)
    blob = json.dumps(allocate(curve, 200_000, 0.05, 1.0).to_dict())
    assert "NaN" not in blob and "Infinity" not in blob
    assert json.loads(blob)["incremental_lo"][0] is None


def test_curve_estimate_at_is_nearest_not_overshooting():
    """A requested phi must not silently resolve to the next grid point."""
    c = CurveEstimate(
        phi=np.linspace(0.01, 1.0, 100), n_targeted=np.arange(1, 101) * 1000.0,
        uplift=np.arange(1, 101) * 1.0, qini=np.arange(1, 101) * 0.85,
    )
    assert c.at(0.10)["phi"] == pytest.approx(0.10, abs=1e-9)
    assert c.at(0.50)["phi"] == pytest.approx(0.50, abs=1e-9)
    assert c.at(1.00)["phi"] == pytest.approx(1.00, abs=1e-9)
    assert c.at(0.0)["phi"] == pytest.approx(0.01, abs=1e-9)
