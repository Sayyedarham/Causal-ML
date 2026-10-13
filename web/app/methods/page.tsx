import BarList from "@/components/charts/BarList";
import { Badge, Card, Note, PageHeader } from "@/components/ui";
import { modelColor, modelLabel } from "@/lib/colors";
import { getDataset, getFeatureImportance, getModelMeta, getRun, ALL_ORDER } from "@/lib/data";
import { fmtInt, fmtNum, fmtPct } from "@/lib/format";

export default async function MethodsPage() {
  const [meta, imp, run, ds] = await Promise.all([
    getModelMeta(), getFeatureImportance(), getRun(), getDataset(),
  ]);

  const lgbm = run.lgbm_params;

  return (
    <>
      <PageHeader
        title="Methods"
        lead="Six estimators spanning the range from 'ignore causality entirely' to 'doubly robust', all LightGBM-backed, sharing one interface: fit, predict an uplift score, export a flat tree dump the browser can replay."
      />

      <div className="space-y-4">
        {ALL_ORDER.map((name) => {
          const m = meta[name];
          if (!m) return null;
          return (
            <Card key={name}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="flex items-center gap-2 text-[15px] font-semibold">
                  <span className="inline-block h-[3px] w-5 rounded" style={{ background: modelColor(name) }} />
                  {modelLabel(name)}
                  <Badge tone={m.is_causal ? "accent" : "muted"}>{m.is_causal ? "causal" : "not causal"}</Badge>
                  <Badge tone="muted">{m.kind}</Badge>
                </h2>
                <div className="tnum text-[12px] text-muted">
                  fit {m.fit_seconds?.toFixed(0) ?? "—"}s · n_train {fmtInt(m.n_train)} · Qini {fmtNum(m.qini_coefficient, 0)}
                </div>
              </div>
              <p className="mt-2 max-w-4xl text-[13px] leading-relaxed text-muted">{m.description}</p>
              {Object.keys(m.diagnostics).length > 0 && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-[12px] text-muted hover:text-fg">fit diagnostics</summary>
                  <dl className="mt-2 grid gap-x-8 gap-y-1 text-[12px] sm:grid-cols-2 lg:grid-cols-3">
                    {Object.entries(m.diagnostics).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-4 border-b border-line/40 py-0.5">
                        <dt className="text-muted">{k}</dt>
                        <dd className="tnum">{typeof v === "number" ? fmtNum(v, 4) : String(v)}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              )}
            </Card>
          );
        })}
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card title="Feature importance" subtitle="Gain share, aggregated across each model's boosters and normalized to 1. The treatment feature's share shows how much splitting on assignment itself drives predictions.">
          <div className="space-y-5">
            {ALL_ORDER.filter((n) => imp[n]).map((n) => (
              <div key={n}>
                <h3 className="mb-1.5 text-[13px] font-medium">{modelLabel(n)}</h3>
                <BarList rows={imp[n].map((r) => ({ label: r.feature, value: r.gain_share }))}
                         fmt={(v) => fmtPct(v, 1)} color={modelColor(n)} max={0.35} />
              </div>
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="LightGBM configuration" subtitle="Deliberately conservative: the causal signal (ATE ≈ 0.00115) is an order of magnitude smaller than the outcome noise, so aggressive trees fit noise and invert out of sample.">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[13px] sm:grid-cols-3">
              {Object.entries(lgbm).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3 border-b border-line/40 py-0.5">
                  <dt className="text-muted">{k}</dt>
                  <dd className="tnum">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </Card>

          <Card title="Dataset & splits">
            <dl className="grid gap-x-8 gap-y-1.5 text-[13px] sm:grid-cols-2">
              {Object.entries(ds.splits).map(([name, s]) => (
                <div key={name} className="rounded border border-line bg-panel2 p-3">
                  <dt className="font-medium capitalize">{name}</dt>
                  <dd className="tnum mt-1 text-[12px] leading-relaxed text-muted">
                    {fmtInt(s.n)} rows · {fmtPct(s.treatment_ratio, 1)} treated · conv {fmtNum(s.conversion_rate, 5)}
                  </dd>
                </div>
              ))}
            </dl>
            <Note>
              Train/valid are subsampled for iteration speed; the test split is never subsampled — at a 0.29%
              conversion rate its control conversions set the width of every interval on this site.
            </Note>
          </Card>
        </div>
      </div>
    </>
  );
}

