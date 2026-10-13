// Artifact schemas (schema_version 3), as written by the Python pipeline.

export type ModelName =
  | "response" | "constant" | "s_learner" | "t_learner" | "x_learner" | "dr_learner";

export interface Tree {
  feature: number[];
  threshold: number[];
  left: number[];
  right: number[];
  value: number[];
}

export interface TreeEnsemble {
  trees: Tree[];
  base_value: number;
  link: "identity" | "logit";
}

export interface ModelDump {
  schema_version: number;
  name: string;
  kind: string;
  is_causal: boolean;
  description: string;
  feature_names: string[];
  treatment_feature_index?: number;
  blend_weight_control?: number;
  constant?: number;
  components: Record<string, TreeEnsemble>;
}

export interface AteEstimate {
  estimator: string;
  ate: number;
  se: number;
  se_bootstrap?: number;
  ci_lo: number;
  ci_hi: number;
  confidence?: number;
  n_treated?: number;
  n_control?: number;
  rate_treated?: number;
  rate_control?: number;
  relative_lift?: number;
  n_draws?: number;
  n?: number;
  note?: string;
}

export interface Effects {
  outcome: string;
  primary: AteEstimate;
  aipw: AteEstimate;
  secondary_outcome: { outcome: string } & AteEstimate;
  interpretation: string;
}

export interface BalanceRow {
  feature: string;
  mean_treated: number;
  mean_control: number;
  sd_treated: number;
  sd_control: number;
  smd: number;
  abs_smd: number;
}

export interface Randomization {
  n_sample: number;
  max_abs_smd: number;
  mean_abs_smd: number;
  "n_features_over_0.1_smd": number;
  propensity_auc: number;
  propensity_log_loss: number;
  intercept_only_log_loss: number;
  log_loss_gap: number;
  treatment_rate: number;
  balance_table: BalanceRow[];
}

export interface Curve {
  is_causal: boolean;
  phi: number[];
  n_targeted: number[];
  uplift: number[];
  uplift_lo: number[] | null;
  uplift_hi: number[] | null;
  qini: number[];
  qini_lo: number[] | null;
  qini_hi: number[] | null;
  rate_treated: number[];
  rate_control: number[];
  qini_coefficient: {
    qini_area: number;
    random_area: number;
    qini_coefficient: number;
    qini_total: number;
  };
  n_bootstrap_draws: number;
  confidence: number;
}

export interface TopKRow {
  model: string;
  phi: number;
  n_targeted: number;
  incremental_conversions: number;
  incremental_lo: number;
  incremental_hi: number;
  uplift_rate: number;
  uplift_rate_lo: number;
  uplift_rate_hi: number;
  baseline_rate: number;
  relative_lift: number;
  qini: number;
  qini_lo: number;
  qini_hi: number;
}

export interface CalibrationBin {
  bin: number;
  n: number;
  n_treated: number;
  n_control: number;
  predicted_uplift_mean: number;
  realized_uplift: number;
  realized_se: number;
  realized_lo: number;
  realized_hi: number;
  rate_treated: number;
  rate_control: number;
  predicted_lo: number;
  predicted_hi: number;
}

export interface ModelMeta {
  name: string;
  kind: string;
  is_causal: boolean;
  description: string;
  fit_seconds: number | null;
  n_train: number | null;
  best_iterations: Record<string, number>;
  diagnostics: Record<string, unknown>;
  qini_coefficient: number;
}

export interface FeatureImportance {
  feature: string;
  gain_share: number;
}

export interface AllocationGrid {
  cost_per_impression: number;
  value_per_conversion: number;
  population: number;
  phi: number[];
  population_targeted: (number | null)[];
  incremental_conversions: (number | null)[];
  incremental_lo: (number | null)[];
  incremental_hi: (number | null)[];
  spend: (number | null)[];
  profit: (number | null)[];
  profit_lo: (number | null)[];
  profit_hi: (number | null)[];
  roi: (number | null)[];
}

export interface Optimum {
  phi: number;
  population_targeted: number;
  spend: number;
  incremental_conversions: number;
  incremental_lo: number;
  incremental_hi: number;
  profit: number;
  profit_lo: number;
  profit_hi: number;
  roi: number;
  plateau_tolerance: number;
  plateau_phi_lo: number;
  plateau_phi_hi: number;
  plateau_population_lo: number;
  plateau_population_hi: number;
}

export interface ThresholdRule {
  uplift_threshold: number;
  phi: number;
  share_positive_effect: number;
  mean_predicted_uplift_targeted: number;
  mean_predicted_uplift_untargeted: number;
}

export interface AllocationModel {
  allocation: AllocationGrid;
  optimum: Optimum;
  threshold_rule: ThresholdRule | null;
  breakeven_cost_per_impression: number;
}

export interface PolicyRow {
  policy: string;
  phi: number;
  budget: number;
  population_targeted: number;
  incremental_conversions: number;
  incremental_lo: number;
  incremental_hi: number;
  profit: number;
  roi: number;
}

export interface Recommendation {
  primary_budget_fraction: number;
  population: number;
  best_policy_at_primary_budget: PolicyRow | null;
  policy_ranking: string[];
  vs_random_targeting: { absolute: number; relative: number } | null;
  vs_response_model: { absolute: number; relative: number } | null;
  optimal_budget_for_winner: Optimum | null;
  rules: string[];
}

export interface Allocation {
  assumptions: {
    cost_per_impression: number;
    value_per_incremental_conversion: number;
    population: number;
    note: string;
  };
  per_model: Record<string, AllocationModel>;
  policy_comparison: PolicyRow[];
  recommendation: Recommendation;
}

export interface Contrast {
  name: string;
  estimate: number;
  ci_lo: number;
  ci_hi: number;
  n_a: number;
  n_b: number;
  rate_a: number;
  rate_b: number;
  note: string;
}

export interface Exposure {
  outcome: string;
  n_test: number;
  exposure_rate_treated: number;
  exposure_rate_control: number;
  one_sided_noncompliance: boolean;
  contrasts: Contrast[];
  exposed_vs_control_imbalance: {
    rows: { feature: string; mean_exposed: number; mean_control: number; smd: number }[];
    max_abs_smd: number;
    mean_abs_smd: number;
    interpretation: string;
  };
  inflation_exposed_vs_all_control: number;
  inflation_within_treated: number;
  n_draws: number;
  verdict: string;
}

export interface SplitSummary {
  n: number;
  n_treated: number;
  n_control: number;
  treatment_ratio: number;
  conversion_rate: number;
  conversion_rate_treated: number;
  conversion_rate_control: number;
  visit_rate: number;
  exposure_rate_treated: number;
  exposure_rate_control: number;
}

export interface Dataset {
  schema_version: number;
  name: string;
  license: string;
  citation: string;
  source_url: string;
  features: string[];
  columns: string[];
  key_figures: {
    rows: number;
    treatment_ratio: number;
    conversion_rate: number;
    visit_rate: number;
    exposure_rate: number;
    naive_ate_conversion: number;
    naive_ate_visit: number;
  };
  version_note: string;
  validation: Record<string, unknown>;
  splits: Record<string, SplitSummary>;
}

export interface RunInfo {
  schema_version: number;
  package_version: string;
  generated_utc: string;
  config: {
    seed: number;
    outcome: string;
    train_sample_rows: number | null;
    valid_sample_rows: number | null;
    split: { train: number; valid: number; test: number };
    models: string[];
    bootstrap_draws: number;
  };
  lgbm_params: Record<string, unknown>;
  timings_sec: Record<string, number>;
  environment: { python: string; platform: string; numpy: string };
  dataset_sha256: string;
}

export interface ModelManifest {
  schema_version: number;
  feature_names: string[];
  models: Record<string, { kind: string; is_causal: boolean; components: string[]; description: string }>;
}
