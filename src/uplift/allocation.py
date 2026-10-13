"""Budget allocation: turning a ranked uplift model into a spend decision.

Targeting the top phi fraction of a population of size N costs c * phi * N and
earns v * (incremental conversions among those users). Everything here is driven
by the *measured* uplift curve from the held-out test split, not by the model's
own predicted magnitudes, so the recommendation inherits the curve's bootstrap
interval instead of the model's optimism.

The theoretically optimal rule is worth stating plainly: target user i exactly
when the value of their incremental conversion exceeds the cost of reaching them,

    v * tau(x_i) > c     <=>     tau(x_i) > c / v

so the optimal targeting fraction is simply the share of users above that
threshold. ``threshold_rule_phi`` computes it from the model's own predictions
and ``empirical_optimal_phi`` finds the argmax of the measured profit curve.
When those two agree, the model's magnitudes are trustworthy enough to budget
against, not merely to rank with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .metrics import CurveEstimate

log = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Profit curve over the targeting fraction, with uncertainty."""

    phi: np.ndarray
    population_targeted: np.ndarray
    incremental_conversions: np.ndarray
    incremental_lo: np.ndarray
    incremental_hi: np.ndarray
    spend: np.ndarray
    profit: np.ndarray
    profit_lo: np.ndarray
    profit_hi: np.ndarray
    roi: np.ndarray
    cost_per_impression: float
    value_per_conversion: float
    population: int

    def to_dict(self) -> dict:
        def lst(a: np.ndarray) -> list[float]:
            return [None if not np.isfinite(v) else float(v) for v in a]

        return {
            "cost_per_impression": self.cost_per_impression,
            "value_per_conversion": self.value_per_conversion,
            "population": self.population,
            "phi": lst(self.phi),
            "population_targeted": lst(self.population_targeted),
            "incremental_conversions": lst(self.incremental_conversions),
            "incremental_lo": lst(self.incremental_lo),
            "incremental_hi": lst(self.incremental_hi),
            "spend": lst(self.spend),
            "profit": lst(self.profit),
            "profit_lo": lst(self.profit_lo),
            "profit_hi": lst(self.profit_hi),
            "roi": lst(self.roi),
        }


def allocate(
    curve: CurveEstimate,
    population: int,
    cost_per_impression: float,
    value_per_conversion: float,
) -> AllocationResult:
    """Build the profit curve for one policy at the given economics.

    ``curve`` is measured on the test split; per-capita quantities are rescaled
    to ``population`` so the result reads as a full-campaign projection.
    """
    if cost_per_impression <= 0 or value_per_conversion <= 0:
        raise ValueError("cost and value must both be positive")
    if population <= 0:
        raise ValueError("population must be positive")

    n_targeted_test = curve.n_targeted.astype(np.float64)
    # Incremental conversion *rate* among the targeted slice. Dividing out the
    # test-set headcount is what makes the curve portable to any population.
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n_targeted_test > 0, curve.uplift / np.maximum(n_targeted_test, 1.0), np.nan)
        rate_lo = np.where(n_targeted_test > 0,
                           curve.uplift_lo / np.maximum(n_targeted_test, 1.0), np.nan) \
            if curve.uplift_lo is not None else np.full_like(rate, np.nan)
        rate_hi = np.where(n_targeted_test > 0,
                           curve.uplift_hi / np.maximum(n_targeted_test, 1.0), np.nan) \
            if curve.uplift_hi is not None else np.full_like(rate, np.nan)

    reached = curve.phi * population
    inc = rate * reached
    inc_lo, inc_hi = rate_lo * reached, rate_hi * reached
    spend = cost_per_impression * reached
    profit = value_per_conversion * inc - spend
    profit_lo = value_per_conversion * inc_lo - spend
    profit_hi = value_per_conversion * inc_hi - spend

    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(spend > 0, profit / spend, np.nan)

    return AllocationResult(
        phi=curve.phi.copy(), population_targeted=reached,
        incremental_conversions=inc, incremental_lo=inc_lo, incremental_hi=inc_hi,
        spend=spend, profit=profit, profit_lo=profit_lo, profit_hi=profit_hi,
        roi=roi, cost_per_impression=cost_per_impression,
        value_per_conversion=value_per_conversion, population=population,
    )


def optimal_point(alloc: AllocationResult, plateau_tolerance: float = 0.05) -> dict:
    """Best targeting fraction on the measured profit curve, plus its interval.

    The profit curve is *flat* at its maximum -- the first-order condition sets
    the marginal profit to zero there -- so the argmax is a poorly identified
    point even when the profit level is well estimated. Reporting a single
    number overstates the precision of the recommendation, so this also returns
    the plateau: the range of targeting fractions whose profit stays within
    ``plateau_tolerance`` of the maximum. Any point in the plateau is as good a
    budget decision as the argmax.
    """
    profit = np.nan_to_num(alloc.profit, nan=-np.inf)
    j = int(np.argmax(profit))
    best = float(profit[j])

    lo_phi = hi_phi = float(alloc.phi[j])
    if best > 0:
        keep = profit >= best * (1.0 - plateau_tolerance)
        if keep.any():
            lo_phi = float(alloc.phi[np.nonzero(keep)[0][0]])
            hi_phi = float(alloc.phi[np.nonzero(keep)[0][-1]])

    def finite(a: np.ndarray) -> float:
        v = float(a[j])
        return v if np.isfinite(v) else float("nan")

    return {
        "phi": float(alloc.phi[j]),
        "population_targeted": float(alloc.population_targeted[j]),
        "spend": float(alloc.spend[j]),
        "incremental_conversions": float(alloc.incremental_conversions[j]),
        "incremental_lo": finite(alloc.incremental_lo),
        "incremental_hi": finite(alloc.incremental_hi),
        "profit": best,
        "profit_lo": finite(alloc.profit_lo),
        "profit_hi": finite(alloc.profit_hi),
        "roi": finite(alloc.roi),
        "plateau_tolerance": plateau_tolerance,
        "plateau_phi_lo": lo_phi,
        "plateau_phi_hi": hi_phi,
        "plateau_population_lo": float(alloc.population_targeted[np.argmin(np.abs(alloc.phi - lo_phi))]),
        "plateau_population_hi": float(alloc.population_targeted[np.argmin(np.abs(alloc.phi - hi_phi))]),
    }


def threshold_rule_phi(predicted_uplift: np.ndarray, cost_per_impression: float,
                       value_per_conversion: float) -> dict:
    """Optimal targeting share implied by tau(x) > c/v, from model predictions."""
    threshold = cost_per_impression / value_per_conversion
    above = predicted_uplift > threshold
    return {
        "uplift_threshold": float(threshold),
        "phi": float(above.mean()),
        "share_positive_effect": float((predicted_uplift > 0).mean()),
        "mean_predicted_uplift_targeted": float(predicted_uplift[above].mean()) if above.any() else float("nan"),
        "mean_predicted_uplift_untargeted": float(predicted_uplift[~above].mean()) if (~above).any() else float("nan"),
    }


def policy_comparison(
    curves: dict[str, CurveEstimate],
    population: int,
    cost_per_impression: float,
    value_per_conversion: float,
    budget_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
) -> list[dict]:
    """Incremental conversions and profit per policy at matched budgets.

    Comparing at a *fixed* phi is what makes the comparison fair: every policy
    reaches the same number of people and spends the same money, so any gap is
    purely about whom it chose.
    """
    rows: list[dict] = []
    allocs = {name: allocate(c, population, cost_per_impression, value_per_conversion)
              for name, c in curves.items()}
    for phi in budget_fractions:
        spend = cost_per_impression * phi * population
        for name, a in allocs.items():
            j = int(np.argmin(np.abs(a.phi - phi)))
            inc = float(a.incremental_conversions[j])
            lo = float(a.incremental_lo[j]) if np.isfinite(a.incremental_lo[j]) else float("nan")
            hi = float(a.incremental_hi[j]) if np.isfinite(a.incremental_hi[j]) else float("nan")
            rows.append({
                "policy": name,
                "phi": float(a.phi[j]),
                "budget": float(spend),
                "population_targeted": float(a.population_targeted[j]),
                "incremental_conversions": inc,
                "incremental_lo": lo,
                "incremental_hi": hi,
                "profit": float(a.profit[j]),
                "roi": float(a.roi[j]) if np.isfinite(a.roi[j]) else float("nan"),
            })
    return rows


def best_policy_at_budget(rows: list[dict], phi: float) -> dict | None:
    """Winning policy at the budget fraction nearest ``phi``."""
    if not rows:
        return None
    phis = np.array([r["phi"] for r in rows])
    target = phis[np.argmin(np.abs(phis - phi))]
    candidates = [r for r in rows if abs(r["phi"] - target) < 1e-9]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["incremental_conversions"]
                                          if np.isfinite(r["incremental_conversions"]) else -np.inf))


def breakeven_cost_per_impression(alloc: AllocationResult) -> float:
    """Highest impression cost at which the optimal policy still breaks even."""
    j = int(np.argmax(np.nan_to_num(alloc.profit, nan=-np.inf)))
    inc = float(alloc.incremental_conversions[j])
    reached = float(alloc.population_targeted[j])
    if reached <= 0 or not np.isfinite(inc) or inc <= 0:
        return 0.0
    return alloc.value_per_conversion * inc / reached


def summarize(
    curve: CurveEstimate,
    predicted_uplift: np.ndarray,
    population: int,
    cost_per_impression: float,
    value_per_conversion: float,
) -> dict:
    """Everything the budget section of the dashboard needs, in one call."""
    alloc = allocate(curve, population, cost_per_impression, value_per_conversion)
    opt = optimal_point(alloc)
    rule = threshold_rule_phi(predicted_uplift, cost_per_impression, value_per_conversion)
    return {
        "allocation": alloc.to_dict(),
        "optimum": opt,
        "threshold_rule": rule,
        "breakeven_cost_per_impression": breakeven_cost_per_impression(alloc),
        "target_everyone": {
            "phi": 1.0,
            "spend": cost_per_impression * population,
            "incremental_conversions": float(alloc.incremental_conversions[-1]),
            "profit": float(alloc.profit[-1]),
        },
    }


__all__ = [
    "AllocationResult",
    "allocate",
    "best_policy_at_budget",
    "breakeven_cost_per_impression",
    "optimal_point",
    "policy_comparison",
    "summarize",
    "threshold_rule_phi",
]
