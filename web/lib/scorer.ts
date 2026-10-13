// TypeScript replay of the flattened LightGBM tree dumps exported by the
// Python pipeline. LightGBM routes a row left when x[feature] <= threshold;
// predictions are the sum of leaf values across trees, passed through the
// ensemble's link (identity for effect regressors, logit for classifiers).
import type { ModelDump, TreeEnsemble } from "./types";

export function scoreEnsemble(ens: TreeEnsemble, x: number[]): number {
  let raw = ens.base_value;
  for (const t of ens.trees) {
    let node = 0;
    while (t.feature[node] >= 0) {
      node = x[t.feature[node]] <= t.threshold[node] ? t.left[node] : t.right[node];
    }
    raw += t.value[node];
  }
  return ens.link === "logit" ? 1 / (1 + Math.exp(-raw)) : raw;
}

function get(ens: TreeEnsemble | undefined, x: number[]): number {
  return ens ? scoreEnsemble(ens, x) : NaN;
}

/** Uplift prediction for one user under an exported model dump. */
export function predictUplift(dump: ModelDump, x: number[]): number {
  switch (dump.kind) {
    case "response":
      return get(dump.components.response, x);
    case "constant":
      return dump.constant ?? NaN;
    case "s_learner": {
      const i = dump.treatment_feature_index ?? x.length;
      const treated = [...x.slice(0, i), 1, ...x.slice(i + 1)];
      const control = [...x.slice(0, i), 0, ...x.slice(i + 1)];
      return get(dump.components.s_learner, treated) - get(dump.components.s_learner, control);
    }
    case "t_learner":
      return get(dump.components.treated, x) - get(dump.components.control, x);
    case "x_learner": {
      const e = dump.blend_weight_control ?? 0.85;
      return e * get(dump.components.tau_control, x) + (1 - e) * get(dump.components.tau_treated, x);
    }
    case "dr_learner":
      return get(dump.components.tau, x);
    default:
      return NaN;
  }
}
