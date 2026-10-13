# uplift dashboard

Static Next.js (App Router, TypeScript, Tailwind v4) front end for the uplift
pipeline. All data is read from `data/` (synced by the Python pipeline's export
stage) **at build time** — the deployed site has no runtime data dependency.

## Pages

| route | contents |
|---|---|
| `/` | headline effects, model leaderboard, budget recommendation, provenance |
| `/effects` | ITT vs AIPW estimates, randomization diagnostics, balance table |
| `/curves` | uplift + Qini curves with bootstrap bands, top-k table |
| `/calibration` | predicted vs realized uplift per bin, per causal model |
| `/allocation` | profit curves with **live cost/value sliders** (recomputed client-side from the measured grid) |
| `/exposure` | the exposure trap: naive contrasts vs ITT vs Wald-IV CACE |
| `/methods` | estimator descriptions, fit diagnostics, feature importance, config |
| `/scorer` | playground: the exported tree ensembles score a synthetic user in-browser |

## The TypeScript scorer

`lib/scorer.ts` replays the flattened tree dumps in `data/models/*.json`
bit-for-bit with LightGBM: route left when `x[feature] <= threshold`, sum leaf
values, apply the ensemble's link (logit for classifiers, identity for effect
regressors). Model-specific composition (S/T/X/DR) mirrors `estimators.py`.

## Build

```bash
npm install
npm run build     # prebuild copies data/ -> public/data, then next build (static export to out/)
npx serve out     # or any static file server
```

`npm run build` fails fast if `data/` is missing — run `uplift run` (in the
repo root) first. During development, `npm run dev` reads the same `data/`
directory.
