// Model identity used consistently across pages and charts.
export const MODEL_LABELS: Record<string, string> = {
  dr_learner: "DR-learner",
  x_learner: "X-learner",
  t_learner: "T-learner",
  s_learner: "S-learner",
  constant: "Constant (ATE)",
  response: "Response model",
};

export const MODEL_COLORS: Record<string, string> = {
  dr_learner: "#5eead4",
  x_learner: "#818cf8",
  t_learner: "#f472b6",
  s_learner: "#fbbf24",
  constant: "#8b94a7",
  response: "#f87171",
};

export const modelLabel = (m: string) => MODEL_LABELS[m] ?? m;
export const modelColor = (m: string) => MODEL_COLORS[m] ?? "#aab2c5";
