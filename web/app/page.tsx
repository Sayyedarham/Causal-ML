import Link from "next/link";
import { getAllocation, getCurves, getDataset, getEffects, getModelMeta, getRun } from "@/lib/data";
import { modelColor, modelLabel } from "@/lib/colors";
import { Badge, Card, Note, PageHeader, Stat } from "@/components/ui";
import { fmtInt, fmtNum, fmtPct, fmtUsd } from "@/lib/format";

export default async function OverviewPage() {
  const [ds, run, fx, curves, meta, alloc] = await Promise.all([
    getDataset(), getRun(), getEffects(), getCurves(), getModelMeta(), getAllocation(),
  ]);

  const p = fx.primary;
  const ranking = alloc.recommendation.policy_ranking;
  const winner = ranking[0];
  const rec = alloc.recommendation.best_policy_at_primary_budget;
  const qiniSorted = Object.entries(curves)
    .filter(([, c]) => c.is_causal)
    .sort((a, b) => b[1].qini_coefficient.qini_coefficient - a[1].qini_coefficient.qini_coefficient);
  const leaderboard = [...Object.entries(curves)].sort(
    (a, b) => b[1].qini_coefficient.qini_coefficient - a[1].qini_coefficient.qini_coefficient,
  );

  return (
    <>
      <PageHeader
        title="Who converts because of the ad?"
        lead={
          <>
            Uplift modeling on the Criteo Uplift v2.1 randomized experiment — {fmtInt(ds.key_figures.rows)}{" "}
            impressions, 85/15 treatment split, {fmtPct(ds.key_figures.conversion_rate)} conversion rate. Six
            targeting policies are fitted, scored on a never-subsampled {fmtInt(ds.splits.test.n)}-row held-out
            split, and turned into a budget recommendation.
          </>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="True incremental effect (ITT)"
          value={<>{fmtNum(p.ate * 100, 3)} pp</>}
          sub={`${fmtNum(p.ate, 5)} conversions per impression · 95% CI [${fmtNum(p.ci_lo, 5)}, ${fmtNum(p.ci_hi, 5)}] · difference-in-means on randomized assignment`}
          tone="accent"
        />
        <Stat
          label="Conversion rate"
          value={fmtPct(ds.key_figures.conversion_rate)}
          sub={`treated ${fmtPct(ds.splits.test.conversion_rate_treated)} vs control ${fmtPct(ds.splits.test.conversion_rate_control)} — a ${(100 * (p.relative_lift ?? NaN)).toFixed(0)}% relative lift that only targeting can earn`}
        />
        <Stat
          label="Best ranking model"
          value={modelLabel(qiniSorted[0]?.[0] ?? "—")}
          sub={`Qini coefficient ${fmtNum(qiniSorted[0]?.[1].qini_coefficient.qini_coefficient, 0)} incremental conversions (area above random targeting)`}
        />
        <Stat
          label="Headline recommendation"
          value={rec ? modelLabel(rec.policy) : winner ? modelLabel(winner) : "—"}
          sub={
            rec
              ? `At ${fmtPct(alloc.recommendation.primary_budget_fraction, 0)} of budget: ${fmtInt(rec.incremental_conversions)} incremental conversions, 95% CI [${fmtInt(rec.incremental_lo)}, ${fmtInt(rec.incremental_hi)}]`
              : undefined
          }
        />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card
          title="Model leaderboard"
          subtitle="Qini coefficient on the held-out test split — area between the model's Qini curve and the random-targeting diagonal, in incremental conversions."
        >
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="py-1.5 pr-2 font-medium">policy</th>
                <th className="py-1.5 pr-2 text-right font-medium">Qini coef.</th>
                <th className="py-1.5 pr-2 text-right font-medium">uplift @10%</th>
                <th className="py-1.5 text-right font-medium">fit</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map(([name, c]) => {
                const top10 = c.phi.findIndex((v) => v >= 0.099);
                return (
                  <tr key={name} className="border-b border-line/50 last:border-0">
                    <td className="py-2 pr-2">
                      <span className="mr-2 inline-block h-[3px] w-4 align-middle" style={{ background: modelColor(name) }} />
                      {modelLabel(name)}
                      {!c.is_causal && <Badge tone="muted">not causal</Badge>}
                    </td>
                    <td className="tnum py-2 pr-2 text-right">{fmtNum(c.qini_coefficient.qini_coefficient, 0)}</td>
                    <td className="tnum py-2 pr-2 text-right">
                      {top10 >= 0 ? fmtNum(c.uplift[top10], 0) : "—"}
                    </td>
                    <td className="tnum py-2 text-right text-muted">
                      {meta[name]?.fit_seconds ? `${meta[name].fit_seconds?.toFixed(0)}s` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <Note>
            The response model ranks by P(Y=1|X) — who will convert, not who converts because of the ad.
            The constant policy applies the average effect to everyone; it cannot rank, and defines the floor.
            Both are shown to calibrate what the causal models add.
          </Note>
        </Card>

        <Card
          title="The decision"
          subtitle="Incremental conversions are measured, not modeled: the profit curve is built from the held-out test split's realized uplift, so the recommendation inherits its bootstrap interval."
        >
          {rec && (
            <div className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-3">
                <Stat label="budget" value={fmtUsd(rec.budget)} sub={`${fmtInt(rec.population_targeted)} users reached`} />
                <Stat label="incremental conversions" value={fmtInt(rec.incremental_conversions)}
                      sub={`CI [${fmtInt(rec.incremental_lo)}, ${fmtInt(rec.incremental_hi)}]`} tone="accent" />
                <Stat label="profit" value={fmtUsd(rec.profit)}
                      sub={`ROI ${(100 * rec.roi).toFixed(0)}% at ${fmtUsd(alloc.assumptions.cost_per_impression, 3)}/impression, ${fmtUsd(alloc.assumptions.value_per_incremental_conversion)}/conversion`} />
              </div>
              {alloc.recommendation.vs_random_targeting && (
                <p className="text-[13px] text-muted">
                  {(() => {
                    const signed = (d: number) => `${d >= 0 ? "+" : "−"}${fmtInt(Math.abs(d))} conversions`;
                    const verb = (d: number, pos: string, neg: string) => (d >= 0 ? pos : neg);
                    const vr = alloc.recommendation.vs_random_targeting;
                    const vf = alloc.recommendation.vs_response_model;
                    return (
                      <>
                        {verb(vr.absolute, "Beats", "Trails")} random targeting by{" "}
                        <span className="tnum text-fg">{signed(vr.absolute)}</span>{" "}
                        ({fmtPct(vr.relative, 0)}). Against the response baseline it{" "}
                        {vf ? verb(vf.absolute, "leads", "trails") : "compares to"} by{" "}
                        <span className="tnum text-fg">{vf ? signed(vf.absolute) : "—"}</span>{" "}
                        at the same spend.
                      </>
                    );
                  })()}{" "}
                  <Link href="/allocation" className="text-accent hover:underline">Full allocation →</Link>
                </p>
              )}
            </div>
          )}
          <Note>
            Cost per impression and value per conversion are explicit assumptions — Criteo publishes no campaign
            economics. The <Link href="/allocation" className="text-accent hover:underline">budget page</Link>{" "}
            recomputes the whole allocation client-side, so you can substitute your own numbers.
          </Note>
        </Card>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card title="Run provenance" className="lg:col-span-2">
          <dl className="grid gap-x-8 gap-y-2 text-[13px] sm:grid-cols-2">
            <Row k="generated">{new Date(run.generated_utc).toUTCString()}</Row>
            <Row k="pipeline">{`uplift v${run.package_version} · schema v${run.schema_version}`}</Row>
            <Row k="seeds">{`split ${run.config.seed} · bootstrap ${run.config.bootstrap_draws} draws`}</Row>
            <Row k="outcome">{run.config.outcome}</Row>
            <Row k="train split">{`${fmtInt(ds.splits.train.n)} rows sampled (cap ${fmtInt(run.config.train_sample_rows)})`}</Row>
            <Row k="valid split">{`${fmtInt(ds.splits.valid.n)} rows sampled (cap ${fmtInt(run.config.valid_sample_rows)})`}</Row>
            <Row k="test split">{`${fmtInt(ds.splits.test.n)} rows — never subsampled`}</Row>
            <Row k="python / numpy">{`${run.environment.python} / ${run.environment.numpy}`}</Row>
            <Row k="dataset sha256" mono>{run.dataset_sha256.slice(0, 16) + "…"}</Row>
            <Row k="total runtime">{`${fmtNum(Object.values(run.timings_sec).reduce((a, b) => a + b, 0), 0)}s`}</Row>
          </dl>
        </Card>
        <Card title="Why v2.1">
          <p className="text-[13px] leading-relaxed text-muted">{ds.version_note}</p>
        </Card>
      </div>
    </>
  );
}

function Row({ k, children, mono }: { k: string; children: React.ReactNode; mono?: boolean }) {
  return (
    <>
      <dt className="text-muted">{k}</dt>
      <dd className={mono ? "tnum truncate" : "tnum"} title={String(children)}>{children}</dd>
    </>
  );
}
