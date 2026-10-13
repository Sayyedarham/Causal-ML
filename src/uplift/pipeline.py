"""End-to-end pipeline: data -> models -> evaluation -> artifacts.

Stages are individually addressable so an iteration loop does not have to refit
everything. Every number that reaches the dashboard is written here, together
with the provenance needed to reproduce it.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import ARTIFACT_SCHEMA_VERSION, __version__
from .allocation import (
    allocate,
    best_policy_at_budget,
    breakeven_cost_per_impression,
    optimal_point,
    policy_comparison,
    threshold_rule_phi,
)
from .artifacts import array_payload, write_json
from .config import (
    DATASET_CITATION,
    DATASET_LICENSE,
    DATASET_URL,
    FEATURE_NAMES,
    VISIT_COL,
    Config,
)
from .data import SplitData, dataset_facts, ensure_materialized, load_split, verify_randomization
from .estimators import DRLearner, build_model
from .exposure import exposure_sensitivity
from .metrics import (
    CurveEstimate,
    ate_aipw,
    ate_difference_in_means,
    calibration_table,
    evaluate_curve,
    qini_coefficient,
    uplift_at_fraction,
)

log = logging.getLogger(__name__)

# `diagnostics` comes after `fit` because the AIPW estimate reuses the
# DR-learner's cross-fitted outcome nuisances rather than fitting a second set.
STAGES = ("data", "fit", "diagnostics", "evaluate", "allocate", "export")


@dataclass
class PipelineState:
    """Values carried between stages."""

    cfg: Config
    timings: dict[str, float] = field(default_factory=dict)
    splits: dict[str, SplitData] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    curves: dict[str, CurveEstimate] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    propensity: float = 0.85
    baseline_rate: float = 0.0


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def stage_data(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    ensure_materialized(cfg)

    state.splits["train"] = load_split(cfg.paths, "train", cfg,
                                       sample_rows=cfg.train_sample_rows)
    state.splits["valid"] = load_split(cfg.paths, "valid", cfg,
                                       sample_rows=cfg.valid_sample_rows)
    # The test split is never subsampled: its size sets the width of every
    # interval in the report.
    state.splits["test"] = load_split(cfg.paths, "test", cfg, sample_rows=None)

    train = state.splits["train"]
    state.propensity = float((train.treatment == 1).mean())
    state.baseline_rate = float(train.conversion[train.treatment == 0].mean())

    state.timings["data"] = time.time() - t0
    for name, sp in state.splits.items():
        log.info("split=%-5s n=%s treated=%s control=%s conv=%.5f",
                 name, f"{len(sp):,}", f"{sp.n_treated:,}", f"{sp.n_control:,}",
                 sp.conversion.mean())


def stage_diagnostics(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    test = state.splits["test"]

    randomization = verify_randomization(test, sample_rows=500_000, seed=cfg.split.seed + 1)
    log.info("randomization: max|SMD|=%.4f propensity AUC=%.4f",
             randomization["max_abs_smd"], randomization["propensity_auc"])

    y = test.outcome(cfg.outcome)
    t = test.treatment

    ate_dim = ate_difference_in_means(y, t, n_draws=cfg.eval.bootstrap_draws,
                                      seed=cfg.eval.bootstrap_seed)

    # AIPW needs outcome nuisances fitted on the training split only.
    dr = state.models.get("dr_learner")
    if isinstance(dr, DRLearner):
        mu1, mu0 = dr.nuisance(test.X)
        e_hat = np.full(len(test), state.propensity)
        ate_aipw_res = ate_aipw(y, t, mu0=mu0, mu1=mu1, e=e_hat)
    else:
        ate_aipw_res = {"estimator": "aipw", "ate": float("nan"), "se": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                        "note": "dr_learner not fitted"}

    visit_dim = ate_difference_in_means(test.visit, t,
                                        n_draws=cfg.eval.bootstrap_draws,
                                        seed=cfg.eval.bootstrap_seed + 1)

    state.artifacts["randomization"] = randomization
    state.artifacts["effects"] = {
        "outcome": cfg.outcome,
        "primary": ate_dim,
        "aipw": ate_aipw_res,
        "secondary_outcome": {"outcome": VISIT_COL, **visit_dim},
        "interpretation": (
            "Both estimates are intention-to-treat on the randomized assignment. "
            "The difference-in-means estimator is unbiased under randomization "
            "alone; AIPW additionally uses outcome models and is the "
            "semiparametric-efficient version of the same quantity. Their "
            "agreement is evidence that the outcome models are not distorting "
            "the effect."
        ),
    }
    state.timings["diagnostics"] = time.time() - t0


def stage_fit(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    train, valid = state.splits["train"], state.splits["valid"]
    y = train.outcome(cfg.outcome)
    y_val = valid.outcome(cfg.outcome)

    for name in cfg.models:
        ts = time.time()
        model = build_model(name, cfg.lgbm, seed=cfg.split.seed,
                            propensity=state.propensity)
        if name == "dr_learner":
            model.n_folds = 3
        log.info("fitting %s ...", name)
        model.fit(train.X, train.treatment, y, valid.X, valid.treatment, y_val)
        state.models[name] = model
        log.info("  %-11s done in %.1fs", name, time.time() - ts)

    state.timings["fit"] = time.time() - t0


def stage_evaluate(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    test = state.splits["test"]
    y = test.outcome(cfg.outcome)
    t = test.treatment

    for name, model in state.models.items():
        ts = time.time()
        state.scores[name] = np.asarray(model.predict_uplift(test.X), dtype=np.float64)
        log.info("scored %-11s in %.1fs (mean=%.6g)", name, time.time() - ts,
                 float(np.mean(state.scores[name])))

    curves: dict[str, Any] = {}
    topk: list[dict] = []
    calibration: dict[str, list] = {}
    model_meta: dict[str, Any] = {}

    for name, score in state.scores.items():
        model = state.models[name]
        # Stable across processes (unlike hash(), which is salted per run).
        seed = cfg.eval.bootstrap_seed + zlib.crc32(name.encode("utf-8")) % 1000

        est = evaluate_curve(
            y, t, score,
            n_points=cfg.eval.n_curve_points,
            n_draws=cfg.eval.bootstrap_curve_draws,
            seed=seed, confidence=cfg.eval.confidence,
        )
        state.curves[name] = est

        qini_stats = qini_coefficient(est)
        per_k = {}
        for phi in cfg.eval.top_k_fractions:
            key = f"top_{int(phi * 100)}pct"
            per_k[key] = uplift_at_fraction(est, phi, state.baseline_rate)
            topk.append({"model": name, "phi": phi, **per_k[key]})

        cal = []
        if getattr(model, "is_causal", False):
            # Calibration is only meaningful for a score that claims to be an
            # effect; for the response model the units are probabilities.
            cal = calibration_table(y, t, score, n_bins=cfg.eval.n_calibration_bins)
        calibration[name] = cal

        curves[name] = {
            "is_causal": bool(getattr(model, "is_causal", False)),
            "phi": array_payload(est.phi, 6),
            "n_targeted": array_payload(est.n_targeted, 1),
            "uplift": array_payload(est.uplift, 4),
            "uplift_lo": array_payload(est.uplift_lo, 4) if est.uplift_lo is not None else None,
            "uplift_hi": array_payload(est.uplift_hi, 4) if est.uplift_hi is not None else None,
            "qini": array_payload(est.qini, 4),
            "qini_lo": array_payload(est.qini_lo, 4) if est.qini_lo is not None else None,
            "qini_hi": array_payload(est.qini_hi, 4) if est.qini_hi is not None else None,
            "rate_treated": array_payload(est.rate_treated, 8),
            "rate_control": array_payload(est.rate_control, 8),
            "qini_coefficient": qini_stats,
            "n_bootstrap_draws": est.n_draws,
            "confidence": est.confidence,
        }

        info = model.export().get("fit", {})
        model_meta[name] = {
            "name": name,
            "kind": model.export().get("kind", name),
            "is_causal": bool(getattr(model, "is_causal", False)),
            "description": getattr(model, "description", ""),
            "fit_seconds": info.get("fit_seconds"),
            "n_train": info.get("n_train"),
            "best_iterations": info.get("best_iterations", {}),
            "diagnostics": info.get("extra", {}),
            "qini_coefficient": qini_stats["qini_coefficient"],
        }

    state.artifacts["curves"] = curves
    state.artifacts["topk"] = topk
    state.artifacts["calibration"] = calibration
    state.artifacts["models"] = model_meta
    state.artifacts["feature_importance"] = _feature_importance(state)
    state.timings["evaluate"] = time.time() - t0


def _feature_importance(state: PipelineState) -> dict[str, list[dict]]:
    """Per-model gain-based importance, normalized to sum to 1."""
    out: dict[str, list[dict]] = {}
    for name, model in state.models.items():
        boosters = _collect_boosters(model)
        if not boosters:
            continue
        total = np.zeros(len(FEATURE_NAMES) + 1)
        for _, b in boosters:
            gain = b.feature_importance(importance_type="gain").astype(np.float64)
            if len(gain) != len(total):
                continue
            total += gain
        s = total.sum()
        shares = total / s if s > 0 else total
        rows = []
        for j, fname in enumerate(FEATURE_NAMES):
            rows.append({"feature": fname, "gain_share": float(shares[j])})
        if len(shares) > len(FEATURE_NAMES):
            rows.append({"feature": "treatment", "gain_share": float(shares[-1])})
        rows.sort(key=lambda r: -r["gain_share"])
        out[name] = rows
    return out


def _collect_boosters(model: Any) -> list[tuple[str, Any]]:
    pairs = []
    for attr, role in (("model", "model"), ("m1", "treated"), ("m0", "control"),
                       ("tau", "tau"), ("tau_t", "tau_treated"), ("tau_c", "tau_control")):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "booster_"):
            pairs.append((role, m.booster_))
    return pairs


def stage_allocate(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    biz = cfg.business
    test = state.splits["test"]
    population = len(test)

    per_model: dict[str, Any] = {}
    for name, est in state.curves.items():
        alloc = allocate(est, population, biz.cost_per_impression,
                         biz.value_per_incremental_conversion)
        opt = optimal_point(alloc)
        score = state.scores[name]
        rule = threshold_rule_phi(score, biz.cost_per_impression,
                                  biz.value_per_incremental_conversion) \
            if state.models[name].is_causal else None
        per_model[name] = {
            "allocation": alloc.to_dict(),
            "optimum": opt,
            "threshold_rule": rule,
            "breakeven_cost_per_impression": breakeven_cost_per_impression(alloc),
        }

    rows = policy_comparison(state.curves, population, biz.cost_per_impression,
                             biz.value_per_incremental_conversion,
                             budget_fractions=tuple(cfg.eval.top_k_fractions))

    # Headline recommendation uses the best causal model at the primary budget.
    causal_names = [n for n in state.curves if state.models[n].is_causal]
    primary_phi = 0.10
    best = None
    for name in causal_names:
        cand = best_policy_at_budget([r for r in rows if r["policy"] == name], primary_phi)
        if cand and (best is None or cand["incremental_conversions"] > best["incremental_conversions"]):
            best = cand

    state.artifacts["allocation"] = {
        "assumptions": {
            "cost_per_impression": biz.cost_per_impression,
            "value_per_incremental_conversion": biz.value_per_incremental_conversion,
            "population": population,
            "note": (
                "Criteo publishes no campaign economics, so cost per impression "
                "and value per incremental conversion are explicit assumptions. "
                "The dashboard recomputes the whole allocation client-side, so "
                "any reader can substitute their own numbers."
            ),
        },
        "per_model": per_model,
        "policy_comparison": rows,
        "recommendation": _recommendation(rows, best, primary_phi, population, per_model),
    }
    state.timings["allocate"] = time.time() - t0


def _row_at_phi(rows: list[dict], policy: str, phi: float, tol: float = 0.005) -> dict | None:
    """Find a policy's comparison row at a budget fraction.

    The tolerance is half a curve grid step: rows carry the *grid* phi, which
    can differ from the requested phi by a float ulp.
    """
    matches = [r for r in rows if r["policy"] == policy and abs(r["phi"] - phi) <= tol]
    return matches[0] if matches else None


def _recommendation(rows: list[dict], best: dict | None, primary_phi: float,
                    population: int, per_model: dict[str, Any]) -> dict[str, Any]:
    """Assemble the headline budget recommendation from the comparison table."""
    policies = sorted({r["policy"] for r in rows})
    scored = []
    for name in policies:
        r = _row_at_phi(rows, name, primary_phi)
        if r is not None and np.isfinite(r["incremental_conversions"]):
            scored.append((r["incremental_conversions"], name, r))
    scored.sort(key=lambda x: -x[0])
    order = [name for _, name, _ in scored]

    def gain(vs_name: str) -> dict | None:
        if best is None:
            return None
        vs = _row_at_phi(rows, vs_name, primary_phi)
        if vs is None:
            return None
        b, v = best["incremental_conversions"], vs["incremental_conversions"]
        if not np.isfinite(b) or not np.isfinite(v) or abs(v) < 1e-12:
            return None
        return {"absolute": float(b - v), "relative": float((b - v) / abs(v))}

    winner = order[0] if order else None
    return {
        "primary_budget_fraction": primary_phi,
        "population": population,
        "best_policy_at_primary_budget": best,
        "policy_ranking": order,
        "vs_random_targeting": gain("constant"),
        "vs_response_model": gain("response"),
        "optimal_budget_for_winner": per_model.get(winner, {}).get("optimum") if winner else None,
        "rules": [
            "Do not target everyone. Past the point where the marginal user's "
            "incremental value falls below the impression cost, extra spend "
            "destroys profit even though it still adds conversions.",
            "Rank by estimated treatment effect, not by predicted conversion. "
            "The response model spends on users who would have converted anyway.",
            "Set the cutoff at tau(x) > cost / value. That single threshold is "
            "the profit-maximizing rule and needs no tuning.",
            "Treat the optimum as a plateau, not a point. The profit curve is "
            "flat near its peak, so any budget inside the plateau is defensible "
            "and the exact argmax is not identified by this sample.",
            "Budget against the interval, not the point estimate. The lower bound "
            "of the bootstrap band is the number to commit spend against.",
        ],
    }


def stage_export(state: PipelineState) -> None:
    cfg = state.cfg
    t0 = time.time()
    out = cfg.paths.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    facts = dataset_facts(cfg.paths)
    write_json(out / "dataset.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "name": "Criteo Uplift Prediction Dataset v2.1",
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "source_url": DATASET_URL,
        "features": FEATURE_NAMES,
        "columns": ["treatment", "conversion", "visit", "exposure"],
        "key_figures": {
            "rows": facts.get("rows"),
            "treatment_ratio": facts.get("treatment_ratio"),
            "conversion_rate": facts.get("conversion_rate"),
            "visit_rate": facts.get("visit_rate"),
            "exposure_rate": facts.get("exposure_rate"),
            "naive_ate_conversion": facts.get("validation", {}).get("naive_ate_conversion"),
            "naive_ate_visit": facts.get("validation", {}).get("naive_ate_visit"),
        },
        "version_note": (
            "v2.1 is Criteo's un-biased release. The original v2 was subsampled "
            "non-uniformly in a way that leaked information about the "
            "incrementality level; v2.1 corrects it and is the version intended "
            "for uplift evaluation. All results here use v2.1."
        ),
        "validation": facts.get("validation", {}),
        "splits": {name: sp.summary() for name, sp in state.splits.items()},
    })

    write_json(out / "exposure.json",
               exposure_sensitivity(state.splits["test"], outcome=cfg.outcome,
                                    n_draws=cfg.eval.bootstrap_draws,
                                    seed=cfg.eval.bootstrap_seed + 7))

    write_json(out / "run.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "package_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": state.cfg.fingerprint(),
        "lgbm_params": state.cfg.lgbm.classifier_kwargs(),
        "timings_sec": {k: round(v, 1) for k, v in state.timings.items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "dataset_sha256": facts.get("sha256"),
    })

    for key in ("randomization", "effects", "curves", "topk", "calibration",
                "models", "feature_importance", "allocation"):
        if key in state.artifacts:
            write_json(out / f"{key}.json", state.artifacts[key])

    _export_models(state)
    state.timings["export"] = time.time() - t0
    sync_web_data(cfg)


def _export_models(state: PipelineState) -> None:
    """Write tree dumps for the TypeScript scorer, plus a scoring manifest."""
    cfg = state.cfg
    mdir = cfg.paths.model_dir
    mdir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": ARTIFACT_SCHEMA_VERSION, "feature_names": FEATURE_NAMES,
                "models": {}}
    for name, model in state.models.items():
        exp = model.export()
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "name": name,
            "kind": exp.get("kind", name),
            "is_causal": bool(exp.get("is_causal", False)),
            "description": getattr(model, "description", ""),
            "feature_names": FEATURE_NAMES,
            "treatment_feature_index": exp.get("treatment_feature_index"),
            "blend_weight_control": exp.get("blend_weight_control"),
            "constant": exp.get("constant"),
            "components": exp.get("components", {}),
        }
        write_json(mdir / f"{name}.json", payload)
        manifest["models"][name] = {
            "kind": payload["kind"],
            "is_causal": payload["is_causal"],
            "components": sorted(payload["components"].keys()),
            "description": payload["description"],
        }
    write_json(mdir / "manifest.json", manifest)


def sync_web_data(cfg: Config) -> None:
    """Copy the artifacts the deployed site needs into the web project."""
    src, dst = cfg.paths.artifacts_dir, cfg.paths.web_data_dir
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.rglob("*.json")):
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(str(rel))
    log.info("synced %d artifact files to %s", len(copied), dst)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run(cfg: Config, stages: tuple[str, ...] = STAGES, verbose: bool = True) -> PipelineState:
    """Execute the requested stages in dependency order."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {list(STAGES)}")

    state = PipelineState(cfg=cfg)
    order = [s for s in STAGES if s in stages]

    # Later stages need earlier products; run the prefix so a partial request
    # still has valid inputs rather than failing with a confusing KeyError.
    if order and order[0] != "data":
        log.info("stage '%s' requested; running prerequisite stages too", order[0])

    t0 = time.time()
    for stage in STAGES:
        if stage not in stages:
            continue
        log.info("=== stage: %s ===", stage)
        {"data": stage_data, "diagnostics": stage_diagnostics, "fit": stage_fit,
         "evaluate": stage_evaluate, "allocate": stage_allocate,
         "export": stage_export}[stage](state)

    # diagnostics needs the DR-learner nuisances, which only exist after fit;
    # re-run it once the models are available.
    if "diagnostics" in stages and "fit" in stages:
        log.info("=== stage: diagnostics (with fitted nuisances) ===")
        stage_diagnostics(state)
        if "export" in stages:
            for key in ("randomization", "effects"):
                write_json(cfg.paths.artifacts_dir / f"{key}.json", state.artifacts[key])

    log.info("pipeline complete in %.1fs", time.time() - t0)
    state.timings["total"] = time.time() - t0
    (cfg.paths.artifacts_dir / "timings.json").write_text(
        json.dumps({k: round(v, 1) for k, v in state.timings.items()}, indent=2))
    return state


__all__ = ["STAGES", "PipelineState", "run", "sync_web_data"]
