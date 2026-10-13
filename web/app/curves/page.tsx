import LineBandChart from "@/components/charts/LineBandChart";
import { Card, Note, PageHeader } from "@/components/ui";
import { modelColor, modelLabel } from "@/lib/colors";
import { ALL_ORDER, getCurves, getTopK } from "@/lib/data";
import type { TopKRow } from "@/lib/types";
import { fmtInt, fmtNum } from "@/lib/format";

export default async function CurvesPage() {
  const [curves, topk] = await Promise.all([getCurves(), getTopK()]);

  const causal = ALL_ORDER.filter((m) => curves[m]?.is_causal);
  const byPhi = new Map<number, Record<string, TopKRow>>();
  for (const r of topk) {
    const k = byPhi.get(r.phi) ?? {};
    k[r.model] = r;
    byPhi.set(r.phi, k);
  }

  return (
    <>
      <PageHeader
        title="Uplift & Qini curves"
        lead="Incremental conversions earned by targeting the top fraction of the population, ranked by each model's score. Bands are 95% bootstrap percentile intervals over the full held-out split — draws reweight the original ranking, so a draw is an O(N) weighted cumsum."
      />

      <Card
        title="Uplift curve — incremental conversions vs targeting fraction"
        subtitle="Uplift(φ) = (rate_treated − rate_control) × N_targeted. This is the quantity a campaign budget consumes: each point is the measured incremental conversions if the top φ of users are targeted."
      >
        <LineBandChart
          series={causal.map((m) => ({
            name: modelLabel(m),
            color: modelColor(m),
            xs: curves[m].phi,
            ys: curves[m].uplift,
            band: curves[m].uplift_lo ? { los: curves[m].uplift_lo!, his: curves[m].uplift_hi! } : null,
          }))}
          xLabel="targeting fraction φ"
          yLabel="incremental conversions"
          xFmt={{ kind: "num", digits: 2 }}
          yFmt={{ kind: "int" }}
        />
        <Note>
          A model earns its keep where its curve sits above the others at small φ — the left edge is where
          targeting decisions are actually made. Curves that only separate at large φ are ranking the crowd,
          not the persuadable.
        </Note>
      </Card>

      <Card
        title="Qini curve"
        subtitle="Qini(φ) = Y_T(φ) − Y_C(φ)·N_T(φ)/N_C(φ). The literature's convention: control conversions are re-weighted by the arm ratio instead of differencing rates. Under randomization both curves carry the same information; this is the shape reviewers expect."
        className="mt-6"
      >
        <LineBandChart
          series={causal.map((m) => ({
            name: modelLabel(m),
            color: modelColor(m),
            xs: curves[m].phi,
            ys: curves[m].qini,
            band: curves[m].qini_lo ? { los: curves[m].qini_lo!, his: curves[m].qini_hi! } : null,
          }))}
          xLabel="targeting fraction φ"
          yLabel="Qini (incremental conversions, re-weighted)"
          xFmt={{ kind: "num", digits: 2 }}
          yFmt={{ kind: "int" }}
        />
      </Card>

      <Card
        title="Top-k targeting table"
        subtitle="Rate-based uplift and incremental conversions at matched budgets, with 95% bootstrap intervals. Comparing policies at the same φ is what makes the comparison fair: everyone spends the same, so any gap is about whom each policy chose."
        className="mt-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="py-1.5 pr-2 font-medium">policy</th>
                <th className="py-1.5 pr-2 text-right font-medium">φ</th>
                <th className="py-1.5 pr-2 text-right font-medium">users</th>
                <th className="py-1.5 pr-2 text-right font-medium">incr. conversions</th>
                <th className="py-1.5 pr-2 font-medium">95% CI</th>
                <th className="py-1.5 pr-2 text-right font-medium">uplift rate</th>
                <th className="py-1.5 text-right font-medium">rel. lift</th>
              </tr>
            </thead>
            <tbody>
              {causal.flatMap((m) =>
                [...byPhi.entries()].map(([phi, rows]) => {
                  const r = rows[m];
                  if (!r) return null;
                  return (
                    <tr key={m + phi} className="border-b border-line/40 last:border-0">
                      <td className="py-2 pr-2">
                        <span className="mr-2 inline-block h-[3px] w-4 align-middle" style={{ background: modelColor(m) }} />
                        {modelLabel(m)}
                      </td>
                      <td className="tnum py-2 pr-2 text-right">{(100 * r.phi).toFixed(0)}%</td>
                      <td className="tnum py-2 pr-2 text-right text-muted">{fmtInt(r.n_targeted)}</td>
                      <td className="tnum py-2 pr-2 text-right">{fmtNum(r.incremental_conversions, 0)}</td>
                      <td className="py-2 pr-2 text-muted tnum text-right text-[12px]">
                        [{fmtNum(r.incremental_lo, 0)}, {fmtNum(r.incremental_hi, 0)}]
                      </td>
                      <td className="tnum py-2 pr-2 text-right">{fmtNum(r.uplift_rate * 100, 3)} pp</td>
                      <td className="tnum py-2 text-right">{fmtNum(r.relative_lift, 2)}×</td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}

