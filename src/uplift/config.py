"""Central configuration.

Everything that affects a result lives here so that a run is fully described by
(a config file plus the dataset checksum). Nothing reads environment variables
or global state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_NAMES = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
CONVERSION_COL = "conversion"
VISIT_COL = "visit"
EXPOSURE_COL = "exposure"

DATASET_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
DATASET_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
DATASET_ROWS = 13_979_592
DATASET_LICENSE = "CC BY-NC-SA 4.0"
DATASET_CITATION = (
    "Diemert E., Betlei A., Renaudin C., Amini M-R. (2018). "
    "A Large Scale Benchmark for Uplift Modeling. "
    "Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom."
)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout, all relative to the repository root."""

    root: Path = PROJECT_ROOT

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def raw_archive(self) -> Path:
        return self.raw_dir / "criteo-research-uplift-v2.1.csv.gz"

    @property
    def raw_validation(self) -> Path:
        return self.raw_dir / "validation.json"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def web_data_dir(self) -> Path:
        """Directory the Next.js app reads at build time."""
        return self.root / "web" / "data"

    def ensure(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.artifacts_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitConfig:
    """Row-level assignment fractions.

    A single large held-out test set matters more than a large training set
    here: with a 0.29% conversion rate the width of every uncertainty interval
    is driven by how many control conversions land in the evaluation split.
    """

    test: float = 0.30
    valid: float = 0.10
    seed: int = 20240607

    @property
    def train(self) -> float:
        return 1.0 - self.test - self.valid


@dataclass(frozen=True)
class LGBMParams:
    """LightGBM hyperparameters.

    Deliberately conservative. The causal signal (ATE = 0.00115) is an order of
    magnitude smaller than the outcome noise, so aggressive trees fit the noise
    and produce uplift rankings that look great in-sample and invert out of
    sample. `min_child_samples` is large for the same reason: a leaf needs
    thousands of rows before its control conversion rate is measurable at all.
    """

    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_child_samples: int = 500
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 10.0
    reg_alpha: float = 1.0
    n_jobs: int = 8
    verbose: int = -1

    def classifier_kwargs(self) -> dict[str, Any]:
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "deterministic": True,
            "force_row_wise": True,
        }

    def regressor_kwargs(self) -> dict[str, Any]:
        kw = self.classifier_kwargs()
        kw["objective"] = "regression"
        # Pseudo-outcome regression targets are heavy-tailed (AIPW weights blow
        # up when e(x) is near 0 or 1); L1 is far more robust here than L2.
        kw["objective"] = "regression_l1"
        return kw


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and bootstrap settings."""

    n_curve_points: int = 200
    bootstrap_draws: int = 1000
    bootstrap_curve_draws: int = 200
    bootstrap_seed: int = 99001
    confidence: float = 0.95
    top_k_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
    n_calibration_bins: int = 10


@dataclass(frozen=True)
class BusinessConfig:
    """Economic assumptions for the budget-allocation recommendation.

    Criteo does not publish campaign economics, so these are explicit,
    documented assumptions rather than data. The web app re-runs the allocation
    client-side over the whole grid, so a reader can substitute their own.
    """

    cost_per_impression: float = 0.002
    value_per_incremental_conversion: float = 12.00
    budget_grid_size: int = 101
    max_target_fraction: float = 1.0


@dataclass(frozen=True)
class Config:
    """Top-level run configuration."""

    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    lgbm: LGBMParams = field(default_factory=LGBMParams)
    eval: EvalConfig = field(default_factory=EvalConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    # None => use every row of the train split. Set an int for a fast iteration
    # loop; the pipeline records whichever value was used.
    train_sample_rows: int | None = 4_000_000
    valid_sample_rows: int | None = 1_000_000

    # Primary outcome. `conversion` is the revenue-relevant label; `visit` is a
    # higher-frequency label used as a robustness check.
    outcome: str = CONVERSION_COL

    # Models to fit. Order is preserved in the artifacts and drives chart order.
    models: tuple[str, ...] = (
        "response",
        "constant",
        "s_learner",
        "t_learner",
        "x_learner",
        "dr_learner",
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = list(self.models)
        return d

    def fingerprint(self) -> dict[str, Any]:
        """A minimal, human-readable description of the run."""
        return {
            "seed": self.split.seed,
            "outcome": self.outcome,
            "train_sample_rows": self.train_sample_rows,
            "valid_sample_rows": self.valid_sample_rows,
            "split": {"train": self.split.train, "valid": self.split.valid, "test": self.split.test},
            "models": list(self.models),
            "bootstrap_draws": self.eval.bootstrap_draws,
        }


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a Config, optionally patched from a JSON file and keyword overrides."""
    cfg = Config()
    if path is not None:
        raw = json.loads(Path(path).read_text())
        cfg = _apply_overrides(cfg, raw)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: Config, raw: dict[str, Any]) -> Config:
    """Shallow-merge nested dataclass sections from a plain dict."""
    kwargs: dict[str, Any] = {}
    sections = {"paths": Paths, "split": SplitConfig, "lgbm": LGBMParams,
                "eval": EvalConfig, "business": BusinessConfig}
    current = asdict(cfg)
    for key, value in raw.items():
        if key in sections and isinstance(value, dict):
            merged = {**current[key], **value}
            if key == "split" and "seed" in value:
                merged["seed"] = int(value["seed"])
            if key == "eval":
                for tk in ("top_k_fractions",):
                    if tk in merged:
                        merged[tk] = tuple(merged[tk])
                merged["n_curve_points"] = int(merged["n_curve_points"])
                merged["bootstrap_draws"] = int(merged["bootstrap_draws"])
                merged["bootstrap_curve_draws"] = int(merged["bootstrap_curve_draws"])
            if key == "paths":
                merged = {"root": Path(merged["root"])}
            kwargs[key] = sections[key](**merged)
        else:
            kwargs[key] = value
    if "models" in kwargs:
        kwargs["models"] = tuple(kwargs["models"])
    from dataclasses import replace

    return replace(cfg, **kwargs)
