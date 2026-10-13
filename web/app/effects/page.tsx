import BarList from "@/components/charts/BarList";
import { Badge, Card, CiBar, Note, PageHeader, Stat } from "@/components/ui";
import { getEffects, getRandomization } from "@/lib/data";
import { fmtInt, fmtNum } from "@/lib/format";

export default async function EffectsPage() {
  const [fx, rnd] = await Promise.all([getEffects(), getRandomization()]);

  return (
    <>
      <PageHeader
        title="Effects & randomization"
        lead="Treatment was assigned at random, so the difference-in-means is unbiased by construction — but randomization is verified here, not assumed, and the headline effect is cross-checked with a doubly-robust estimator."
      />

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="difference-in-means (ITT)" value={fmtNum(fx.primary.ate, 5)}
              sub={`95% CI [${fmtNum(fx.primary.ci_lo, 5)}, ${fmtNum(fx.primary.ci_hi, 5)}] · SE ${fmtNum(fx.primary.se, 5)}`} tone="accent" />
        <Stat label="AIPW (doubly robust)" value={fmtNum(fx.aipw.ate, 5)}
              sub={`95% CI [${fmtNum(fx.aipw.ci_lo, 5)}, ${fmtNum(fx.aipw.ci_hi, 5)}] · SE ${fmtNum(fx.aipw.se, 5)}`} />
        <Stat label={`secondary outcome: ${fx.secondary_outcome.outcome}`} value={fmtNum(fx.secondary_outcome.ate, 5)}
              sub={`95% CI [${fmtNum(fx.secondary_outcome.ci_lo, 5)}, ${fmtNum(fx.secondary_outcome.ci_hi, 5)}]`} />
      </div>

      <Card title="Two estimators, one number" className="mt-6">
        <p className="text-[13px] leading-relaxed text-muted">{fx.interpretation}</p>
        <table className="mt-4 w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-1.5 pr-2 font-medium">estimator</th>
              <th className="py-1.5 pr-2 text-right font-medium">ATE</th>
              <th className="py-1.5 pr-2 text-right font-medium">SE</th>
              <th className="py-1.5 pr-2 font-medium">95% CI</th>
              <th className="py-1.5 text-right font-medium">n</th>
            </tr>
          </thead>
          <tbody>
            {[fx.primary, fx.aipw, fx.secondary_outcome].map((e, i) => (
              <tr key={e.estimator + i} className="border-b border-line/50 last:border-0">
                <td className="py-2 pr-2">
                  {e.estimator}
                  {i === 2 && <Badge tone="muted">{fx.secondary_outcome.outcome}</Badge>}
                </td>
                <td className="tnum py-2 pr-2 text-right">{fmtNum(e.ate, 5)}</td>
                <td className="tnum py-2 pr-2 text-right">{fmtNum(e.se, 5)}</td>
                <td className="py-2 pr-2">
                  <CiBar lo={e.ci_lo} hi={e.ci_hi} est={e.ate} fmt={(v) => fmtNum(v, 5)} />
                </td>
                <td className="tnum py-2 text-right text-muted">{fmtInt((e.n_treated ?? 0) + (e.n_control ?? e.n ?? 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card
          title="Randomization checks"
          subtitle="If assignment were confounded, everything downstream would be invalid. Two independent checks say it is not."
          className="lg:col-span-1"
        >
          <div className="space-y-3">
            <Stat label="max |SMD| across 12 features" value={fmtNum(rnd.max_abs_smd, 4)}
                  sub="balanced when ≪ 0.1; standardized mean difference between arms" />
            <Stat label="propensity AUC (3-fold CV)" value={fmtNum(rnd.propensity_auc, 4)}
                  sub={`a logistic model cannot predict assignment better than chance (0.5) — log-loss gap vs intercept-only: ${fmtNum(rnd.log_loss_gap, 5)}`} />
            <Stat label="features with |SMD| > 0.1" value={fmtInt(rnd["n_features_over_0.1_smd"])}
                  sub={`of ${rnd.balance_table.length} features · ${fmtInt(rnd.n_sample)}-row sample`} />
          </div>
        </Card>

        <Card
          title="Covariate balance table"
          subtitle="Standardized mean differences per feature between treated and control arms."
          className="lg:col-span-2"
        >
          <BarList
            rows={[...rnd.balance_table]
              .sort((a, b) => b.abs_smd - a.abs_smd)
              .map((r) => ({ label: r.feature, value: r.abs_smd }))}
            fmt={(v) => fmtNum(v, 4)}
            color="#818cf8"
          />
          <Note>
            Under correct randomization every |SMD| is far below 0.1. This is the empirical justification for
            treating assignment as ignorable — and for not reaching for propensity reweighting.
          </Note>
        </Card>
      </div>
    </>
  );
}
