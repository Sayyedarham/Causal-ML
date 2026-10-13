# uplift — incremental-conversion targeting engine

Uplift modeling on the [Criteo Uplift Prediction Dataset v2.1](https://huggingface.co/datasets/criteo/criteo-uplift):
13,979,592 rows of randomized ad exposure (85% treated / 15% control, 0.29% conversion rate),
used to answer one question — *who converts because of the ad, not merely who converts?*

The pipeline fits six targeting policies, measures each on a large held-out randomized
test split, turns the winning curve into a budget recommendation, and exports everything
(dataset facts, effect estimates, uplift/Qini curves with bootstrap bands, calibration
tables, allocation grids, tree dumps) as JSON for the Next.js dashboard in `web/`.

## Dataset

| | |
|---|---|
| Source | Criteo Uplift v2.1 (unbiased release; v2 leaked incrementality through non-uniform subsampling) |
| Rows | 13,979,592 × 16 columns (12 anonymized features `f0`–`f11`, `treatment`, `conversion`, `visit`, `exposure`) |
| Treatment ratio | 0.85 |
| Conversion rate | 0.0029 (treated 0.0031, control 0.0019 → naive ATE ≈ 0.00115) |
| License | CC BY-NC-SA 4.0 |
| SHA-256 | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` |

Citation: Diemert E., Betlei A., Renaudin C., Amini M-R. (2018). *A Large Scale Benchmark
for Uplift Modeling.* AdKDD@KDD, London.

## Install

Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Workflow

```bash
uplift fetch        # download the 311 MB archive from HuggingFace, verify sha256
uplift validate     # re-check the checksum, print raw-data facts
uplift run          # full pipeline: data -> fit -> diagnostics -> evaluate -> allocate -> export
```

`uplift run` options:

```bash
--stages data,fit          # run a subset of stages (prerequisites run automatically)
--train-rows 500000        # override the train sample size (default 4,000,000)
--valid-rows 200000        # override the valid sample size (default 1,000,000)
--config my.json           # JSON patch over the default Config dataclasses
--quiet
```

Speed profile of a default run: the train/valid splits are subsampled (4M / 1M rows) for
iteration speed, but the **test split is never subsampled** — at a 0.29% conversion rate
the width of every reported interval is driven by how many control conversions land in
the evaluation split. A full run takes on the order of 1–2 hours on 8 threads; most of
it is the DR-learner's cross-fitting and the curve bootstraps over the 4.2M-row test split.

## What the pipeline computes

**Causal ground truth.** Treatment is randomized, so the intention-to-treat (ITT)
difference-in-means is unbiased by construction. It is cross-checked against an AIPW
(doubly robust) estimate that reuses the DR-learner's outcome nuisances; agreement
between the two is evidence the outcome models aren't distorting the effect. Randomization
itself is verified, not assumed: per-feature standardized mean differences (all |SMD| ≪ 0.1)
plus a cross-fitted propensity model that fails to predict assignment (AUC ≈ 0.5).

**The exposure trap.** `exposure == 0` for *every* control user (controls could never be
shown the ad), so filtering to exposed users compares against a group that was never
eligible — and exposure is a collider on the path from user activity to conversion.
`exposure.py` quantifies exactly how much the naive exposed-vs-control contrast inflates
the effect, and recovers a legitimate effect-on-the-exposed via the Wald IV ratio
(one-sided noncompliance ⇒ CACE), which never conditions on the post-treatment variable.

**Six estimators** (`estimators.py`), all LightGBM, sharing one interface:

| model | idea | role |
|---|---|---|
| `response` | P(Y=1\|X), treatment-blind | the policy most marketing teams ship — the strawman |
| `constant` | ATE for everyone | random targeting; the floor a real model must beat |
| `s_learner` | one model, treatment as feature | shrinks heterogeneity toward zero |
| `t_learner` | one model per arm | control arm holds 15% of rows → noisy mu_0 |
| `x_learner` | Künzel 2019 imputation + propensity blend | built for exactly this 85/15 lopsidedness |
| `dr_learner` | cross-fitted AIPW pseudo-outcome regression | the reference; doubly robust |

Tree ensembles are exported as flat, framework-independent JSON (feature / threshold /
left / right / value arrays) that a ~100-line TypeScript scorer in `web/` replays
bit-for-bit, so the deployed site scores users without a Python runtime.

**Evaluation** (`metrics.py`). Uplift and Qini curves at 200 breakpoints with bootstrap
percentile bands (draws reuse the original score ordering, so a draw is an O(N) weighted
cumsum, no re-sorting). Qini coefficient, uplift at top 1–50%, an uplift calibration table
(realized vs predicted effect per bin — the check that separates *ranking* quality from
*magnitude* trustworthiness), and bootstrap CIs throughout.

**Decision layer** (`allocation.py`). Target user i iff `v·tau(x_i) > c`, i.e. the cutoff
is `c/v` — no tuning. The measured profit curve gives the optimal targeting fraction, its
plateau (the curve is flat at the maximum, so a single argmax overstates precision), the
breakeven impression cost, and a policy comparison at *matched* budgets. Cost per
impression and value per conversion are explicit assumptions (Criteo publishes no
economics); the dashboard recomputes the whole allocation client-side so any reader can
substitute their own numbers.

## Artifacts

`uplift run` writes to `artifacts/` (schema-versioned, NaN-free JSON):

| file | contents |
|---|---|
| `dataset.json` | provenance, key figures, split summaries |
| `run.json` | config fingerprint, timings, environment, dataset sha256 |
| `randomization.json` | balance table, propensity AUC — proof the ITT framing holds |
| `effects.json` | ITT + AIPW ATEs with CIs, primary (conversion) and secondary (visit) |
| `curves.json` | per-model uplift & Qini curves with bootstrap bands |
| `topk.json` / `calibration.json` / `models.json` / `feature_importance.json` | leaderboard detail |
| `allocation.json` | profit curves, optima + plateaus, policy comparison, headline recommendation |
| `exposure.json` | ITT vs naive contrasts, Wald/CACE, subgroup imbalance |
| `models/*.json` | flattened tree dumps + scoring manifest for the TS scorer |

`stage_export` also syncs everything to `web/data/` for the dashboard build.

## Web dashboard

`web/` is a static-export Next.js app (TypeScript) that reads `web/data/*.json` at build
time and re-serves the full analysis — curves, calibration, allocation with live
cost/value sliders, exposure analysis — plus a scorer playground that runs the exported
tree ensembles in the browser. See `web/README.md`.

## Reproducibility

Every run is described by its config fingerprint plus the dataset checksum — no
environment variables, no global state. Split assignment, subsampling, tie-breaking,
bootstrap draws and LightGBM seeds are all derived from two recorded seeds. LightGBM runs
with `deterministic=True`.

## Repository layout

```
src/uplift/        data, estimators, metrics, allocation, exposure, pipeline, CLI
scripts/           one-off raw-archive validation
tests/             unit tests (metrics, estimators, allocation)
data/raw|processed raw gz + validation report; memmapped .npy materialization
artifacts/         generated JSON reports + tree dumps
web/               static Next.js dashboard (reads web/data at build time)
```

Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` (the env var avoids
unrelated third-party pytest plugins in the global environment).

## License

Code: MIT. Dataset: Criteo Uplift v2.1, CC BY-NC-SA 4.0 (non-commercial).
