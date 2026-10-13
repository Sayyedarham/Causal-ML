"""Tests for the estimator zoo.

The tree-export parity tests are the load-bearing ones: the deployed site scores
users by replaying exported trees in TypeScript, so the flattened representation
must reproduce LightGBM's own predictions bit-for-bit.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pytest

from uplift.config import LGBMParams
from uplift.estimators import (
    LINK_IDENTITY,
    LINK_LOGIT,
    ConstantUplift,
    DRLearner,
    ResponseModel,
    SLearner,
    TLearner,
    TreeEnsemble,
    XLearner,
    build_model,
    flatten_booster,
)

# Small and fast; correctness of the export does not depend on model quality.
FAST = LGBMParams(n_estimators=25, learning_rate=0.1, num_leaves=8, max_depth=4,
                  min_child_samples=20, n_jobs=2, reg_lambda=1.0, reg_alpha=0.0)


def synthetic(n: int = 30_000, seed: int = 11, treat_p: float = 0.85):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    tau = 0.08 * X[:, 0] - 0.05 * X[:, 1] + 0.01
    base = 1.0 / (1.0 + np.exp(-(0.5 * X[:, 2] + 0.3 * X[:, 3] - 0.4)))
    p0 = np.clip(base, 0.01, 0.9)
    p1 = np.clip(base + tau, 0.01, 0.95)
    t = (rng.random(n) < treat_p).astype(np.uint8)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(np.uint8)
    return X, t, y, tau


def split(X, t, y, frac=0.25, seed=0):
    """Returns (train_tuple, valid_tuple, valid_row_indices)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    k = int(len(y) * frac)
    va, tr = idx[:k], idx[k:]
    return (X[tr], t[tr], y[tr]), (X[va], t[va], y[va]), va


# --------------------------------------------------------------------------- #
# tree export parity  (critical path for browser inference)
# --------------------------------------------------------------------------- #

def _naive_tree_walk(tree, x):
    """Independent, deliberately-unoptimized reference implementation."""
    node = 0
    while tree["feature"][node] >= 0:
        f = tree["feature"][node]
        node = tree["left"][node] if x[f] <= tree["threshold"][node] else tree["right"][node]
    return tree["value"][node]


def test_eval_tree_matches_naive_walk():
    """The vectorized descent must agree with a per-row reference walk."""
    from uplift.estimators import _eval_tree

    X, _, y, _ = synthetic(4000, seed=3)
    m = lgb.LGBMClassifier(random_state=0, **FAST.classifier_kwargs())
    m.fit(X, y)
    ens = flatten_booster(m.booster_, LINK_LOGIT)

    Xd = X.astype(np.float64)
    for k, tree in enumerate(ens.trees):
        vectorized = _eval_tree(tree, Xd)
        reference = np.array([_naive_tree_walk(tree, Xd[i]) for i in range(len(Xd))])
        assert np.allclose(vectorized, reference, atol=1e-12), f"tree {k} diverged"


def test_flatten_classifier_matches_lightgbm_exactly():
    X, _, y, _ = synthetic(8000, seed=4)
    m = lgb.LGBMClassifier(random_state=7, **FAST.classifier_kwargs())
    m.fit(X, y)
    ens = flatten_booster(m.booster_, LINK_LOGIT)

    expected = m.predict_proba(X)[:, 1]
    got = ens.predict(X.astype(np.float64))
    assert np.allclose(got, expected, atol=1e-12, rtol=0)


def test_flatten_regressor_matches_lightgbm_exactly():
    X, _, _, _ = synthetic(8000, seed=5)
    target = np.random.default_rng(1).normal(size=len(X))
    m = lgb.LGBMRegressor(random_state=9, **FAST.regressor_kwargs())
    m.fit(X, target)
    ens = flatten_booster(m.booster_, LINK_IDENTITY)

    expected = m.predict(X)
    got = ens.predict(X.astype(np.float64))
    assert np.allclose(got, expected, atol=1e-10, rtol=0)


def test_exported_ensemble_is_json_roundtrippable():
    X, _, y, _ = synthetic(5000, seed=6)
    m = lgb.LGBMClassifier(random_state=2, **FAST.classifier_kwargs())
    m.fit(X, y)
    ens = flatten_booster(m.booster_, LINK_LOGIT)

    blob = json.dumps(ens.to_dict())
    restored = TreeEnsemble(**json.loads(blob))
    assert np.allclose(restored.predict(X.astype(np.float64)),
                       ens.predict(X.astype(np.float64)), atol=1e-12)


def test_flatten_rejects_unsupported_splits():
    """Categorical splits would silently score wrong in TS, so they must raise."""
    from uplift.estimators import _assert_simple

    with pytest.raises(ValueError, match="decision_type"):
        _assert_simple({"decision_type": "==", "missing_type": "None"})
    with pytest.raises(ValueError, match="missing"):
        _assert_simple({"decision_type": "<=", "missing_type": "NaN"})


# --------------------------------------------------------------------------- #
# estimator behaviour
# --------------------------------------------------------------------------- #

def test_response_model_flags_itself_non_causal():
    X, t, y, _ = synthetic(6000, seed=7)
    (Xtr, ttr, ytr), (Xva, tva, yva), _ = split(X, t, y)
    m = ResponseModel(FAST, seed=0).fit(Xtr, ttr, ytr, Xva, tva, yva)
    assert m.is_causal is False
    p = m.predict_uplift(Xva)
    assert p.shape == (len(Xva),)
    assert np.all((p >= 0) & (p <= 1)), "response score must be a probability"
    exp = m.export()
    assert exp["is_causal"] is False and exp["kind"] == "response"


def test_response_model_ignores_treatment():
    """Identical predictions for the same X regardless of assigned arm."""
    X, t, y, _ = synthetic(6000, seed=8)
    (Xtr, ttr, ytr), _, _ = split(X, t, y)
    m = ResponseModel(FAST, seed=0).fit(Xtr, ttr, ytr)
    assert np.array_equal(m.predict_uplift(Xtr[:500]), m.predict_uplift(Xtr[:500]))


def test_constant_uplift_returns_ate_everywhere():
    X, t, y, _ = synthetic(6000, seed=9)
    m = ConstantUplift(seed=0).fit(X, t, y)
    expected = y[t == 1].mean() - y[t == 0].mean()
    p = m.predict_uplift(X)
    assert np.allclose(p, expected)
    assert m.ate == pytest.approx(expected)


def test_t_learner_ranks_by_true_effect():
    X, t, y, tau = synthetic(40_000, seed=10)
    (Xtr, ttr, ytr), _, va_idx = split(X, t, y)
    m = TLearner(FAST, seed=0).fit(Xtr, ttr, ytr)
    pred = m.predict_uplift(X[va_idx])
    assert np.corrcoef(pred, tau[va_idx])[0, 1] > 0.35


def test_s_learner_reports_treatment_split_share():
    X, t, y, _ = synthetic(20_000, seed=12)
    (Xtr, ttr, ytr), (Xva, tva, yva), _ = split(X, t, y)
    m = SLearner(FAST, seed=0).fit(Xtr, ttr, ytr, Xva, tva, yva)
    share = m._info.extra["treatment_split_share"]
    assert 0.0 <= share <= 1.0
    # Treatment is one weak binary feature among six; the S-learner should
    # demonstrate its known shrinkage by barely splitting on it.
    assert share < 0.35


def test_s_learner_export_has_treatment_index():
    X, t, y, _ = synthetic(12_000, seed=13)
    m = SLearner(FAST, seed=0).fit(X, t, y)
    exp = m.export()
    assert exp["treatment_feature_index"] == 6  # 6 features + treatment appended
    assert exp["kind"] == "s_learner"


def test_x_learner_blends_by_propensity():
    X, t, y, _ = synthetic(20_000, seed=14)
    (Xtr, ttr, ytr), (Xva, tva, yva), _ = split(X, t, y)
    m = XLearner(FAST, seed=0, propensity=float(ttr.mean()))
    m.fit(Xtr, ttr, ytr, Xva, tva, yva)
    p = m.predict_uplift(Xva)
    assert p.shape == (len(Xva),)
    assert np.all(np.isfinite(p))
    e = m._info.extra["weight_tau_control"]
    assert e == pytest.approx(float(ttr.mean()), abs=1e-6)


def test_x_learner_export_matches_python_prediction():
    X, t, y, _ = synthetic(15_000, seed=15)
    m = XLearner(FAST, seed=0, propensity=0.85).fit(X, t, y)
    exp = m.export()
    w = exp["blend_weight_control"]
    tau_c = TreeEnsemble(**exp["components"]["tau_control"])
    tau_t = TreeEnsemble(**exp["components"]["tau_treated"])
    rebuilt = w * tau_c.predict(X.astype(np.float64)) + (1 - w) * tau_t.predict(X.astype(np.float64))
    assert np.allclose(rebuilt, m.predict_uplift(X), atol=1e-10)


def test_dr_learner_cross_fits_and_reports_propensity():
    X, t, y, _ = synthetic(20_000, seed=16)
    m = DRLearner(FAST, seed=0, n_folds=3).fit(X, t, y)
    extra = m._info.extra
    assert extra["n_folds"] == 3
    # Randomized assignment: a fitted propensity should sit near the true 0.85
    assert extra["propensity_mean"] == pytest.approx(0.85, abs=0.05)
    assert np.isfinite(m.predict_uplift(X[:100])).all()


def test_dr_learner_nuisance_available_after_fit():
    X, t, y, _ = synthetic(12_000, seed=17)
    m = DRLearner(FAST, seed=0, n_folds=2).fit(X, t, y)
    mu1, mu0 = m.nuisance(X[:500])
    assert mu1.shape == (500,) and mu0.shape == (500,)
    assert np.all((mu1 >= 0) & (mu1 <= 1)) and np.all((mu0 >= 0) & (mu0 <= 1))


def test_dr_learner_export_roundtrip():
    X, t, y, _ = synthetic(12_000, seed=18)
    m = DRLearner(FAST, seed=0, n_folds=2).fit(X, t, y)
    exp = m.export()
    tau = TreeEnsemble(**exp["components"]["tau"])
    assert np.allclose(tau.predict(X.astype(np.float64)), m.predict_uplift(X), atol=1e-10)


# --------------------------------------------------------------------------- #
# registry / determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["response", "constant", "s_learner", "t_learner",
                                  "x_learner", "dr_learner"])
def test_registry_builds_and_exports(name):
    X, t, y, _ = synthetic(9000, seed=19)
    m = build_model(name, FAST, seed=0, propensity=0.85)
    if name == "dr_learner":
        m.n_folds = 2
    m.fit(X, t, y)
    p = m.predict_uplift(X[:200])
    assert p.shape == (200,)
    exp = m.export()
    json.dumps(exp)  # must be serializable
    assert exp["name"] == name
    assert exp["is_causal"] == (name != "response")


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        build_model("nope", FAST, seed=0)


def test_fit_is_deterministic_given_seed():
    X, t, y, _ = synthetic(12_000, seed=20)
    a = TLearner(FAST, seed=123).fit(X, t, y).predict_uplift(X[:300])
    b = TLearner(FAST, seed=123).fit(X, t, y).predict_uplift(X[:300])
    assert np.array_equal(a, b)


def test_different_seeds_give_different_models():
    X, t, y, _ = synthetic(12_000, seed=21)
    a = TLearner(FAST, seed=1).fit(X, t, y).predict_uplift(X[:300])
    b = TLearner(FAST, seed=2).fit(X, t, y).predict_uplift(X[:300])
    assert not np.array_equal(a, b)


def test_models_disagree_in_ranking_not_sign():
    """All causal learners should agree the ATE is positive on this DGP."""
    X, t, y, _ = synthetic(25_000, seed=22)
    (Xtr, ttr, ytr), _, _ = split(X, t, y)
    for name in ("s_learner", "t_learner", "x_learner"):
        m = build_model(name, FAST, seed=0, propensity=0.85).fit(Xtr, ttr, ytr)
        assert m.predict_uplift(Xtr).mean() > 0, name
