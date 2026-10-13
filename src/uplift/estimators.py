"""Uplift estimators sharing one interface.

Five approaches, spanning the range from "ignore causality entirely" to
"doubly robust":

    ResponseModel  treatment-blind P(Y=1|X). The conventional targeting policy.
                   Its score is NOT a treatment effect and must not be read as one.
    ConstantUplift the ATE applied to everyone. The randomization-only baseline;
                   any model worth its salt must beat this ranking.
    SLearner       one model, treatment as a feature.
    TLearner       one model per arm, difference of predictions.
    XLearner       Künzel et al. (2019). Built for lopsided arms, which is exactly
                   this dataset (85% treated / 15% control).
    DRLearner      cross-fitted AIPW pseudo-outcome regression. Doubly robust and
                   the most defensible under a known propensity.

All LightGBM-backed. Every fitted model can emit a compact JSON tree dump that
the TypeScript scorer in `web/` replays bit-for-bit, so the deployed site scores
real users without a Python runtime.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np

from .config import LGBMParams

log = logging.getLogger(__name__)

LINK_IDENTITY = "identity"
LINK_LOGIT = "logit"


# --------------------------------------------------------------------------- #
# tree export
# --------------------------------------------------------------------------- #

@dataclass
class TreeEnsemble:
    """A flattened, framework-independent representation of a boosted model."""

    trees: list[dict[str, list]] = field(default_factory=list)
    base_value: float = 0.0
    link: str = LINK_IDENTITY

    def apply_link(self, raw: np.ndarray) -> np.ndarray:
        if self.link == LINK_LOGIT:
            return 1.0 / (1.0 + np.exp(-raw))
        return raw

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Sum of leaf values across all trees, per row."""
        out = np.full(X.shape[0], self.base_value, dtype=np.float64)
        for tree in self.trees:
            out += _eval_tree(tree, X)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.apply_link(self.predict_raw(X))

    def to_dict(self) -> dict[str, Any]:
        return {"trees": self.trees, "base_value": self.base_value, "link": self.link}


def _eval_tree(tree: dict[str, list], X: np.ndarray) -> np.ndarray:
    """Vectorized descent of one flattened tree over all rows.

    Rows that reach a leaf have ``feature == -1`` and are skipped from then on,
    so the loop bound (total node count) is a safe upper bound on tree depth.
    """
    # The tree arrives as plain lists (it is JSON round-tripped), and lists
    # cannot be fancy-indexed by an ndarray, so densify once per tree.
    feature = np.asarray(tree["feature"], dtype=np.int64)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    left = np.asarray(tree["left"], dtype=np.int64)
    right = np.asarray(tree["right"], dtype=np.int64)
    value = np.asarray(tree["value"], dtype=np.float64)

    node = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(feature.size):
        # f[row] is the split feature of the node that row currently sits on.
        # Rows parked on a leaf have f == -1 and drop out of `rows` for good.
        f = feature[node]
        rows = np.nonzero(f >= 0)[0]
        if rows.size == 0:
            break
        nd = node[rows]
        goes_left = X[rows, f[rows]] <= threshold[nd]
        node[rows] = np.where(goes_left, left[nd], right[nd])
    return value[node]


def flatten_booster(booster: lgb.Booster, link: str) -> TreeEnsemble:
    """Convert LightGBM's nested dump into flat per-node arrays.

    LightGBM routes a row left when ``x[feature] <= threshold``. The intercept
    for the sklearn wrapper is folded into the trees, so ``base_value`` is 0; it
    stays in the format so a non-boosted model could be exported later.
    """
    dump = booster.dump_model()
    trees = [_flatten_node(info["tree_structure"]) for info in dump["tree_info"]]
    return TreeEnsemble(trees=trees, base_value=0.0, link=link)


def _flatten_node(node: dict[str, Any]) -> dict[str, list]:
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def walk(n: dict[str, Any]) -> int:
        i = len(feature)
        if "leaf_value" in n:
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(n["leaf_value"]))
            return i
        _assert_simple(n)
        feature.append(int(n["split_feature"]))
        threshold.append(float(n["threshold"]))
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        left[i] = walk(n["left_child"])
        right[i] = walk(n["right_child"])
        return i

    walk(node)
    return {"feature": feature, "threshold": threshold, "left": left,
            "right": right, "value": value}


def _assert_simple(node: dict[str, Any]) -> None:
    dt = node.get("decision_type", "<=")
    if dt != "<=":
        raise ValueError(
            f"unsupported decision_type {dt!r}; the TS scorer implements '<=' only"
        )
    if node.get("missing_type", "None") not in ("None", ""):
        raise ValueError("missing-value handling is not supported by the TS scorer")


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #

class UpliftModel(Protocol):
    name: str
    is_causal: bool
    description: str

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, t_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> UpliftModel: ...

    def predict_uplift(self, X: np.ndarray) -> np.ndarray: ...

    def export(self) -> dict[str, Any]: ...


@dataclass
class _Fitted:
    """Shared bookkeeping for every estimator."""

    fit_seconds: float = 0.0
    n_train: int = 0
    best_iterations: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _make_classifier(p: LGBMParams, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(random_state=seed, **p.classifier_kwargs())


def _make_regressor(p: LGBMParams, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(random_state=seed, **p.regressor_kwargs())


_LGBM_FIT_PARAMS = inspect.signature(lgb.LGBMClassifier.fit).parameters
# LightGBM >= 4.6 renamed eval_set to eval_X/eval_y and warns on the old name.
_USE_EVAL_XY = "eval_X" in _LGBM_FIT_PARAMS


def _fit_with_early_stopping(
    model: Any, X: np.ndarray, y: np.ndarray,
    X_val: np.ndarray | None, y_val: np.ndarray | None,
    rounds: int = 50,
) -> int:
    """Fit, using the validation split for early stopping when available."""
    callbacks = [lgb.log_evaluation(period=0)]
    kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None and len(X_val) > 0:
        if _USE_EVAL_XY:
            kwargs["eval_X"], kwargs["eval_y"] = X_val, y_val
        else:
            kwargs["eval_set"] = [(X_val, y_val)]
        callbacks.append(lgb.early_stopping(rounds, verbose=False))
    model.fit(X, y, callbacks=callbacks, **kwargs)
    return int(getattr(model, "best_iteration_", 0) or model.n_estimators)


def _with_treatment(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.hstack([X, t.reshape(-1, 1).astype(np.float32)])


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #

class ResponseModel:
    """Conventional conversion prediction. Ranks by P(Y=1|X), ignoring treatment.

    This is the policy most marketing models actually ship. It answers "who will
    convert?", not "who converts *because of the ad?*". Users with a high baseline
    propensity score at the top even when the ad changes nothing for them, and
    users the ad would flip are buried if their baseline rate is low.
    """

    name = "response"
    is_causal = False
    description = (
        "Treatment-blind gradient-boosted classifier ranking users by predicted "
        "conversion probability P(Y=1|X). The conventional targeting policy: it "
        "identifies who will convert, not who converts because of the ad."
    )

    def __init__(self, params: LGBMParams, seed: int = 0) -> None:
        self.params = params
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None
        self._info = _Fitted()

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        t0 = time.time()
        self.model = _make_classifier(self.params, self.seed)
        best = _fit_with_early_stopping(self.model, X, y, X_val, y_val)
        extra: dict[str, Any] = {"best_iteration": best}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            p = self.model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score

            extra["auc_valid"] = float(roc_auc_score(y_val, p))
            extra["logloss_valid"] = float(_logloss(y_val, p))
        self._info = _Fitted(fit_seconds=time.time() - t0, n_train=len(X),
                             best_iterations={"response": best}, extra=extra)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        # A probability, not an effect. Downstream code must not interpret it as
        # an incremental rate; `is_causal=False` guards that.
        return self.model.predict_proba(X)[:, 1]

    def export(self) -> dict[str, Any]:
        assert self.model is not None
        ens = flatten_booster(self.model.booster_, LINK_LOGIT)
        return {"name": self.name, "is_causal": False, "kind": "response",
                "components": {"response": ens.to_dict()},
                "fit": _asdict_info(self._info)}


class ConstantUplift:
    """Homogeneous-effect baseline: everyone receives the ATE.

    Randomized targeting. It cannot rank anyone, so it defines the floor every
    heterogeneous model must clear.
    """

    name = "constant"
    is_causal = True
    description = (
        "Applies the sample average treatment effect to every user. Equivalent to "
        "random targeting and to assuming the effect is homogeneous; it is the "
        "floor that any heterogeneous-effect model must beat."
    )

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.ate: float = 0.0
        self._info = _Fitted()

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        t0 = time.time()
        tb = t == 1
        self.ate = float(y[tb].mean() - y[~tb].mean())
        self._info = _Fitted(fit_seconds=time.time() - t0, n_train=len(X),
                             extra={"ate": self.ate})
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        # Identical scores for everyone. rank_desc breaks ties uniformly at
        # random, which is exactly the random-targeting behaviour we want.
        return np.full(X.shape[0], self.ate, dtype=np.float64)

    def export(self) -> dict[str, Any]:
        return {"name": self.name, "is_causal": True, "kind": "constant",
                "components": {}, "constant": self.ate, "fit": _asdict_info(self._info)}


class SLearner:
    """One model with treatment as an input feature.

    Cheap and shares statistical strength across arms, but biased toward zero
    effect: with treatment as 1 of 13 features and a small true effect, the tree
    learner rarely splits on it and shrinks heterogeneity toward the ATE.
    """

    name = "s_learner"
    is_causal = True
    description = (
        "Single gradient-boosted classifier with the treatment indicator appended "
        "as one more feature; uplift is f(x,1) - f(x,0). Shares statistical strength "
        "across arms but is biased toward zero effect, because the learner rarely "
        "splits on one weak binary feature among many stronger ones."
    )

    def __init__(self, params: LGBMParams, seed: int = 0) -> None:
        self.params = params
        self.seed = seed
        self.model: lgb.LGBMClassifier | None = None
        self.n_features: int = 0
        self._info = _Fitted()

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        t0 = time.time()
        self.n_features = int(X.shape[1])
        self.model = _make_classifier(self.params, self.seed)
        best = _fit_with_early_stopping(
            self.model, _with_treatment(X, t), y,
            None if X_val is None else _with_treatment(X_val, t_val), y_val,
        )
        # Diagnostic: how often did the learner actually use treatment?
        imp = self.model.booster_.feature_importance(importance_type="split")
        self._info = _Fitted(
            fit_seconds=time.time() - t0, n_train=len(X),
            best_iterations={"s_learner": best},
            extra={"treatment_split_share": float(imp[-1] / max(imp.sum(), 1)),
                   "treatment_splits": int(imp[-1])},
        )
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        n = X.shape[0]
        one = self.model.predict_proba(_with_treatment(X, np.ones(n, dtype=np.uint8)))[:, 1]
        zero = self.model.predict_proba(_with_treatment(X, np.zeros(n, dtype=np.uint8)))[:, 1]
        return one - zero

    def export(self) -> dict[str, Any]:
        assert self.model is not None
        ens = flatten_booster(self.model.booster_, LINK_LOGIT)
        return {"name": self.name, "is_causal": True, "kind": "s_learner",
                "components": {"s_learner": ens.to_dict()},
                "treatment_feature_index": self.n_features, "fit": _asdict_info(self._info)}


class TLearner:
    """One model per arm; uplift is the difference of predicted outcomes.

    Lets each arm fit its own surface, so heterogeneity is not shrunk by design.
    The cost here is real: the control arm holds only 15% of rows and ~1,200
    conversions in a 4M-row sample, so mu_0 is the noisier half of the difference.
    """

    name = "t_learner"
    is_causal = True
    description = (
        "Two independent gradient-boosted classifiers, one per treatment arm; "
        "uplift is mu_1(x) - mu_0(x). Captures heterogeneity without shrinking it, "
        "at the cost of a much noisier control-arm model under this dataset's "
        "85/15 split."
    )

    def __init__(self, params: LGBMParams, seed: int = 0) -> None:
        self.params = params
        self.seed = seed
        self.m1: lgb.LGBMClassifier | None = None
        self.m0: lgb.LGBMClassifier | None = None
        self._info = _Fitted()

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        t0 = time.time()
        tb = t == 1
        self.m1 = _make_classifier(self.params, self.seed + 1)
        self.m0 = _make_classifier(self.params, self.seed + 2)

        Xv1 = yv1 = Xv0 = yv0 = None
        if X_val is not None:
            tvb = t_val == 1
            Xv1, yv1 = X_val[tvb], y_val[tvb]
            Xv0, yv0 = X_val[~tvb], y_val[~tvb]

        b1 = _fit_with_early_stopping(self.m1, X[tb], y[tb], Xv1, yv1)
        b0 = _fit_with_early_stopping(self.m0, X[~tb], y[~tb], Xv0, yv0)
        self._info = _Fitted(
            fit_seconds=time.time() - t0, n_train=len(X),
            best_iterations={"treated": b1, "control": b0},
            extra={"n_treated": int(tb.sum()), "n_control": int((~tb).sum()),
                   "conv_treated": float(y[tb].mean()), "conv_control": float(y[~tb].mean())},
        )
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        assert self.m1 is not None and self.m0 is not None, "call fit() first"
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]

    def predict_mu(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.m1 is not None and self.m0 is not None
        return self.m1.predict_proba(X)[:, 1], self.m0.predict_proba(X)[:, 1]

    def export(self) -> dict[str, Any]:
        assert self.m1 is not None and self.m0 is not None
        return {
            "name": self.name, "is_causal": True, "kind": "t_learner",
            "components": {"treated": flatten_booster(self.m1.booster_, LINK_LOGIT).to_dict(),
                           "control": flatten_booster(self.m0.booster_, LINK_LOGIT).to_dict()},
            "fit": _asdict_info(self._info),
        }


class XLearner:
    """Künzel et al. (2019) X-learner.

    Two stages. First impute individual effects on *observed* units --
    D1 = Y - mu_0(X) for the treated, D0 = mu_1(X) - Y for the controls -- then
    regress each imputed effect on X and blend by propensity:
    tau = e(x) * tau_control(x) + (1 - e(x)) * tau_treated(x).

    The point of the blend is to lean on whichever imputation rests on the larger
    arm. With 85% treated, mu_1 is well estimated, so tau_control (whose target
    depends on mu_1) is the more trustworthy half and receives weight e = 0.85.
    """

    name = "x_learner"
    is_causal = True
    description = (
        "X-learner (Künzel et al. 2019): imputes individual effects on observed "
        "units, regresses them on X within each arm, then blends by the propensity "
        "score so the arm with more data dominates. Designed for exactly the "
        "lopsided 85/15 allocation in this dataset."
    )

    def __init__(self, params: LGBMParams, seed: int = 0, propensity: float = 0.85) -> None:
        self.params = params
        self.seed = seed
        self.propensity = propensity
        self.m1: lgb.LGBMClassifier | None = None
        self.m0: lgb.LGBMClassifier | None = None
        self.tau_t: lgb.LGBMRegressor | None = None
        self.tau_c: lgb.LGBMRegressor | None = None
        self._info = _Fitted()

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        t0 = time.time()
        tb = t == 1
        X1, y1, X0, y0 = X[tb], y[tb], X[~tb], y[~tb]

        self.m1 = _make_classifier(self.params, self.seed + 11)
        self.m0 = _make_classifier(self.params, self.seed + 12)
        Xv1 = Xv0 = yv1 = yv0 = None
        if X_val is not None:
            tvb = t_val == 1
            Xv1, yv1 = X_val[tvb], y_val[tvb]
            Xv0, yv0 = X_val[~tvb], y_val[~tvb]
        b1 = _fit_with_early_stopping(self.m1, X1, y1, Xv1, yv1)
        b0 = _fit_with_early_stopping(self.m0, X0, y0, Xv0, yv0)

        # Stage 2: imputed effects on observed units.
        d1 = y1.astype(np.float64) - self.m0.predict_proba(X1)[:, 1]
        d0 = self.m1.predict_proba(X0)[:, 1] - y0.astype(np.float64)

        # Mirror the imputation on the validation split so stage 2 can early-stop
        # against a target built the same way. mu_1/mu_0 were fitted on train only,
        # so these validation pseudo-outcomes are honest.
        dv1 = dv0 = Xv1e = Xv0e = None
        if X_val is not None:
            tvb = t_val == 1
            Xv1e, Xv0e = X_val[tvb], X_val[~tvb]
            dv1 = y_val[tvb].astype(np.float64) - self.m0.predict_proba(Xv1e)[:, 1]
            dv0 = self.m1.predict_proba(Xv0e)[:, 1] - y_val[~tvb].astype(np.float64)

        self.tau_t = _make_regressor(self.params, self.seed + 13)
        self.tau_c = _make_regressor(self.params, self.seed + 14)
        bt = _fit_with_early_stopping(self.tau_t, X1, d1, Xv1e, dv1)
        bc = _fit_with_early_stopping(self.tau_c, X0, d0, Xv0e, dv0)

        e = float(np.clip(self.propensity, 0.01, 0.99))
        self._info = _Fitted(
            fit_seconds=time.time() - t0, n_train=len(X),
            best_iterations={"mu_treated": b1, "mu_control": b0,
                             "tau_treated": bt, "tau_control": bc},
            extra={"propensity_used": e, "weight_tau_control": e,
                   "mean_d1": float(d1.mean()), "mean_d0": float(d0.mean())},
        )
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        assert self.tau_t is not None and self.tau_c is not None, "call fit() first"
        e = float(np.clip(self.propensity, 0.01, 0.99))
        return e * self.tau_c.predict(X) + (1.0 - e) * self.tau_t.predict(X)

    def export(self) -> dict[str, Any]:
        assert all(m is not None for m in (self.tau_t, self.tau_c))
        return {
            "name": self.name, "is_causal": True, "kind": "x_learner",
            "components": {
                "tau_treated": flatten_booster(self.tau_t.booster_, LINK_IDENTITY).to_dict(),
                "tau_control": flatten_booster(self.tau_c.booster_, LINK_IDENTITY).to_dict(),
            },
            "blend_weight_control": float(np.clip(self.propensity, 0.01, 0.99)),
            "fit": _asdict_info(self._info),
        }


class DRLearner:
    """Cross-fitted doubly-robust learner (AIPW pseudo-outcome regression).

    Pseudo-outcome per row:
        psi = mu_1 - mu_0 + T(Y - mu_1)/e - (1-T)(Y - mu_0)/(1-e)
    then regress psi on X. Cross-fitting keeps each row's nuisance prediction
    out-of-sample, which is what makes the estimator root-n consistent for the
    ATE rather than inheriting the nuisance models' regularization bias.

    With e = 0.85 the control correction carries a 1/0.15 = 6.7x multiplier, so
    psi is heavy-tailed. The second stage uses an L1 objective for that reason.
    """

    name = "dr_learner"
    is_causal = True
    description = (
        "Cross-fitted doubly-robust learner: builds AIPW pseudo-outcomes with "
        "out-of-fold nuisance predictions and regresses them on X with an L1 "
        "objective. Consistent if either the outcome or the propensity model is "
        "correct; the reference estimator here."
    )

    def __init__(self, params: LGBMParams, seed: int = 0, n_folds: int = 4,
                 e_clip: float = 0.02) -> None:
        self.params = params
        self.seed = seed
        self.n_folds = n_folds
        self.e_clip = e_clip
        self.tau: lgb.LGBMRegressor | None = None
        self._info = _Fitted()
        self._mu1_full: lgb.LGBMClassifier | None = None
        self._mu0_full: lgb.LGBMClassifier | None = None

    def fit(self, X, t, y, X_val=None, t_val=None, y_val=None):
        from sklearn.model_selection import StratifiedKFold

        t0 = time.time()
        n = len(X)
        y64 = y.astype(np.float64)
        tb = t == 1

        mu1 = np.zeros(n)
        mu0 = np.zeros(n)
        e_hat = np.zeros(n)

        # Stratify on treatment x outcome so rare control conversions appear in
        # every fold's training set.
        strata = (t.astype(np.int64) * 2 + y.astype(np.int64)).astype(np.int64)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for k, (tr, te) in enumerate(skf.split(X, strata)):
            t_tr = tb[tr]
            m1 = _make_classifier(self.params, self.seed + 100 + k)
            m0 = _make_classifier(self.params, self.seed + 200 + k)
            m1.fit(X[tr][t_tr], y64[tr][t_tr])
            m0.fit(X[tr][~t_tr], y64[tr][~t_tr])
            mu1[te] = m1.predict_proba(X[te])[:, 1]
            mu0[te] = m0.predict_proba(X[te])[:, 1]

            pe = _make_classifier(self.params, self.seed + 300 + k)
            pe.fit(X[tr], tb[tr].astype(np.uint8))
            e_hat[te] = pe.predict_proba(X[te])[:, 1]
            log.debug("dr_learner fold %d/%d done", k + 1, self.n_folds)
            del m1, m0, pe

        ec = np.clip(e_hat, self.e_clip, 1.0 - self.e_clip)
        psi = (mu1 - mu0
               + tb * (y64 - mu1) / ec
               - (~tb) * (y64 - mu0) / (1.0 - ec))

        # Assignment is randomized, so the propensity is a known constant equal
        # to the treated share. Estimated here rather than assumed, so the number
        # in the artifacts reflects the actual training split.
        e_const = float(tb.mean())

        # Nuisances for the validation split and for AIPW on the test split.
        # Fitted on the full training split, so nothing from valid/test leaks in.
        self._mu1_full = _make_classifier(self.params, self.seed + 500)
        self._mu0_full = _make_classifier(self.params, self.seed + 501)
        self._mu1_full.fit(X[tb], y64[tb])
        self._mu0_full.fit(X[~tb], y64[~tb])

        psi_val = None
        if X_val is not None and len(X_val) > 0:
            tvb = t_val == 1
            yv = y_val.astype(np.float64)
            mu1v = self._mu1_full.predict_proba(X_val)[:, 1]
            mu0v = self._mu0_full.predict_proba(X_val)[:, 1]
            evc = np.clip(e_const, self.e_clip, 1.0 - self.e_clip)
            psi_val = (mu1v - mu0v
                       + tvb * (yv - mu1v) / evc
                       - (~tvb) * (yv - mu0v) / (1.0 - evc))

        self.tau = _make_regressor(self.params, self.seed + 400)
        best = _fit_with_early_stopping(self.tau, X, psi, X_val, psi_val)

        self._info = _Fitted(
            fit_seconds=time.time() - t0, n_train=n,
            best_iterations={"tau": best},
            extra={
                "n_folds": self.n_folds,
                "e_clip": self.e_clip,
                "propensity_mean": float(e_hat.mean()),
                "propensity_std": float(e_hat.std()),
                "propensity_min": float(e_hat.min()),
                "propensity_max": float(e_hat.max()),
                "constant_propensity": e_const,
                "pseudo_outcome_mean": float(psi.mean()),
                "pseudo_outcome_std": float(psi.std()),
                "pseudo_outcome_p99": float(np.percentile(psi, 99)),
                "mu1_mean": float(mu1.mean()), "mu0_mean": float(mu0.mean()),
            },
        )
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        assert self.tau is not None, "call fit() first"
        return self.tau.predict(X)

    def nuisance(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu_1, mu_0) from models fitted on the full training split."""
        return self._mu1_full.predict_proba(X)[:, 1], self._mu0_full.predict_proba(X)[:, 1]

    def export(self) -> dict[str, Any]:
        assert self.tau is not None
        return {
            "name": self.name, "is_causal": True, "kind": "dr_learner",
            "components": {"tau": flatten_booster(self.tau.booster_, LINK_IDENTITY).to_dict()},
            "fit": _asdict_info(self._info),
        }


# --------------------------------------------------------------------------- #
# helpers / registry
# --------------------------------------------------------------------------- #

def _asdict_info(info: _Fitted) -> dict[str, Any]:
    return {"fit_seconds": round(info.fit_seconds, 2), "n_train": info.n_train,
            "best_iterations": info.best_iterations, "extra": info.extra}


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


REGISTRY: dict[str, type] = {
    "response": ResponseModel,
    "constant": ConstantUplift,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
}


def build_model(name: str, params: LGBMParams, seed: int = 0,
                propensity: float = 0.85) -> UpliftModel:
    """Instantiate an estimator by registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    if name == "constant":
        return ConstantUplift(seed=seed)  # type: ignore[return-value]
    if name == "x_learner":
        return XLearner(params, seed=seed, propensity=propensity)  # type: ignore[return-value]
    return REGISTRY[name](params, seed=seed)  # type: ignore[return-value]


__all__ = [
    "LINK_IDENTITY",
    "LINK_LOGIT",
    "REGISTRY",
    "ConstantUplift",
    "DRLearner",
    "ResponseModel",
    "SLearner",
    "TLearner",
    "TreeEnsemble",
    "UpliftModel",
    "XLearner",
    "build_model",
    "flatten_booster",
]
