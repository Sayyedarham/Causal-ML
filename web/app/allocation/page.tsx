import AllocationExplorer, { type AllocSeries } from "@/components/AllocationExplorer";
import { Note, PageHeader } from "@/components/ui";
import { getAllocation } from "@/lib/data";

export default async function AllocationPage() {
  const alloc = await getAllocation();
  const a = alloc.assumptions;

  const series: AllocSeries[] = Object.entries(alloc.per_model).map(([model, pm]) => ({
    model,
    isCausal: model !== "response",
    phi: pm.allocation.phi,
    inc: pm.allocation.incremental_conversions.map((v) => v ?? NaN),
    lo: pm.allocation.incremental_lo.map((v) => v ?? NaN),
    hi: pm.allocation.incremental_hi.map((v) => v ?? NaN),
  }));
  const order = ["dr_learner", "x_learner", "t_learner", "s_learner", "constant", "response"];
  series.sort((x, y) => order.indexOf(x.model) - order.indexOf(y.model));

  return (
    <>
      <PageHeader
        title="Budget allocation"
        lead="Target user i exactly when the value of their incremental conversion exceeds the cost of reaching them: v·τ(xᵢ) > c, i.e. a cutoff at c/v — no tuning. The profit curve below is measured, not modeled: built from the held-out split's realized uplift, so the recommendation inherits its bootstrap interval."
      />
      <AllocationExplorer
        population={a.population}
        defaultCost={a.cost_per_impression}
        defaultValue={a.value_per_incremental_conversion}
        defaultModel={alloc.recommendation.best_policy_at_primary_budget?.policy}
        series={series}
        gridPhis={series[0]?.phi ?? []}
      />
      <div className="mt-6 rounded-lg border border-line bg-panel p-5">
        <h2 className="text-[15px] font-semibold">Operating rules</h2>
        <ul className="mt-3 space-y-2 text-[13px] leading-relaxed text-muted">
          {alloc.recommendation.rules.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span className="tnum shrink-0 text-accent">{i + 1}.</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
        <Note>
          Grid resolution: {series[0]?.phi.length ?? 0} points over φ ∈ (0, 1] on the{" "}
          {a.population.toLocaleString("en-US")}-row test population.
        </Note>
      </div>
    </>
  );
}
