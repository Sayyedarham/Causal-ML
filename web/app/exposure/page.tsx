import BarList from "@/components/charts/BarList";
import { Badge, Card, Note, PageHeader, Stat } from "@/components/ui";
import { getExposure } from "@/lib/data";
import { fmtInt, fmtNum, fmtPct } from "@/lib/format";

const CONTRAST_LABELS: Record<string, { label: string; tone: "accent" | "warn" | "bad" }> = {
  itt_randomized: { label: "ITT — randomized assignment", tone: "accent" },
  cace_wald_iv: { label: "CACE — Wald IV ratio", tone: "accent" },
  naive_exposed_vs_all_control: { label: "naive — exposed treated vs all control", tone: "bad" },
  naive_exposed_vs_unexposed_treated: { label: "naive — exposed vs unexposed (treated arm)", tone: "warn" },
};

export default async function ExposurePage() {
  const ex = await getExposure();

  return (
    <>
      <PageHeader
        title="The exposure trap"
        lead={`exposure records whether the ad was actually shown — and it is zero for every control user. Conditioning on it compares exposed users against a group that was never eligible, and controls on a collider of user activity. This page quantifies the damage and shows the legitimate alternative.`}
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="exposure rate, treated" value={fmtPct(ex.exposure_rate_treated)} sub={`${fmtInt(ex.n_test)}-row test split`} />
        <Stat label="exposure rate, control" value={fmtPct(ex.exposure_rate_control)}
              sub={ex.one_sided_noncompliance ? "one-sided noncompliance, exactly" : undefined} tone={ex.one_sided_noncompliance ? "warn" : "default"} />
        <Stat label="inflation: exposed vs all control" value={`${fmtNum(ex.inflation_exposed_vs_all_control, 2)}×`}
              sub="naive estimate ÷ ITT estimate" tone="bad" />
        <Stat label="inflation: within treated arm" value={`${fmtNum(ex.inflation_within_treated, 2)}×`}
              sub="as-treated estimate ÷ ITT estimate" tone="warn" />
      </div>

      <Card
        title="Four contrasts, one dataset"
        subtitle="All estimates are differences in conversion rate on the same held-out split; only the populations being compared change."
        className="mt-6"
      >
        <div className="space-y-4">
          {ex.contrasts.map((c) => {
            const meta = CONTRAST_LABELS[c.name] ?? { label: c.name, tone: "default" as const };
            return (
              <div key={c.name} className="rounded-lg border border-line bg-panel2 p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-[14px] font-semibold">
                    {meta.label}{" "}
                    <Badge tone={meta.tone}>{meta.tone === "accent" ? "valid" : meta.tone === "warn" ? "biased" : "biased"}</Badge>
                  </h3>
                  <div className="tnum text-[15px] font-semibold">
                    {fmtNum(c.estimate, 5)}{" "}
                    <span className="text-[12px] font-normal text-muted">
                      95% CI [{fmtNum(c.ci_lo, 5)}, {fmtNum(c.ci_hi, 5)}]
                    </span>
                  </div>
                </div>
                <p className="mt-2 text-[13px] leading-relaxed text-muted">{c.note}</p>
                <div className="mt-2 text-[12px] text-muted tnum">
                  n_a={fmtInt(c.n_a)} · n_b={fmtInt(c.n_b)}
                  {isFinite(c.rate_a) ? ` · rate_a=${fmtNum(c.rate_a, 5)}` : ""}
                  {isFinite(c.rate_b) ? ` · rate_b=${fmtNum(c.rate_b, 5)}` : ""}
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card
          title="Why the naive contrast is biased: the exposed subgroup is not comparable"
          subtitle="Standardized mean differences between exposed treated users and the control arm. Randomization balances treated-vs-control, not exposed-vs-control."
        >
          <BarList
            rows={ex.exposed_vs_control_imbalance.rows.map((r) => ({ label: r.feature, value: Math.abs(r.smd) }))}
            fmt={(v) => fmtNum(v, 3)}
            color="#f87171"
          />
          <Note>{ex.exposed_vs_control_imbalance.interpretation}</Note>
        </Card>

        <Card title="Recovering an effect on the exposed" subtitle="Without conditioning on anything post-treatment.">
          <p className="text-[13px] leading-relaxed text-muted">
            With one-sided noncompliance, randomization is an instrument for exposure. The Wald ratio
          </p>
          <p className="tnum my-3 rounded border border-line bg-panel2 px-4 py-3 text-[14px]">
            CACE = (E[Y|T=1] − E[Y|T=0]) / (E[E|T=1] − E[E|T=0])
          </p>
          <p className="text-[13px] leading-relaxed text-muted">
            identifies the effect of <em>being shown the ad</em> among users who would be exposed if targeted —
            the compliers — using only randomized quantities. The numbers differ (a per-exposed effect is
            naturally much larger than a per-assigned one); the logic differs more. The ITT remains the number
            a budget consumes, because budgets buy assignment, not compliance.
          </p>
          <Note>{ex.verdict}</Note>
        </Card>
      </div>
    </>
  );
}
