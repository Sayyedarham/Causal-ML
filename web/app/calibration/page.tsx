import CalibrationChart from "@/components/charts/CalibrationChart";
import { Badge, Card, Note, PageHeader } from "@/components/ui";
import { modelLabel } from "@/lib/colors";
import { getCalibration } from "@/lib/data";
import { fmtInt, fmtNum } from "@/lib/format";

export default async function CalibrationPage() {
  const cal = await getCalibration();
  const models = Object.entries(cal).filter(([, bins]) => bins.length > 0);

  return (
    <>
      <PageHeader
        title="Calibration of predicted effects"
        lead="A model that merely orders users can look great on a Qini curve while its magnitudes are fiction. Calibration asks more: within bins of predicted uplift, does the realized effect match the prediction? Only a calibrated score may be plugged into a budget calculation."
      />

      {models.length === 0 && (
        <Card>
          <p className="text-[13px] text-muted">
            No calibration bins were produced for this run. Calibration is only computed for causal models
            whose scores claim to be effects.
          </p>
        </Card>
      )}

      <div className="space-y-6">
        {models.map(([name, bins]) => {
          const offDiag = bins.filter(
            (b) => b.realized_lo > b.predicted_uplift_mean || b.realized_hi < b.predicted_uplift_mean,
          ).length;
          const covered = bins.length - offDiag;
          return (
            <Card
              key={name}
              title={<span className="flex items-center gap-2">{modelLabel(name)} <Badge tone={covered / bins.length >= 0.7 ? "accent" : "warn"}>{covered}/{bins.length} bins cover prediction</Badge></span>}
              subtitle="Realized uplift per bin with 95% Wald intervals against mean predicted uplift. Bins are equal-count in predicted uplift."
            >
              <CalibrationChart
                points={bins.map((b) => ({
                  bin: b.bin,
                  predicted: b.predicted_uplift_mean,
                  realized: b.realized_uplift,
                  lo: b.realized_lo,
                  hi: b.realized_hi,
                  n: b.n,
                }))}
                xLabel="predicted uplift (conversion probability difference)"
                yLabel="realized uplift"
                xFmt={{ kind: "num", digits: 3 }}
                yFmt={{ kind: "num", digits: 3 }}
              />
              <div className="overflow-x-auto">
                <table className="mt-2 w-full min-w-[640px] text-[13px]">
                  <thead>
                    <tr className="border-b border-line text-left text-muted">
                      <th className="py-1.5 pr-2 font-medium">bin</th>
                      <th className="py-1.5 pr-2 text-right font-medium">n</th>
                      <th className="py-1.5 pr-2 text-right font-medium">predicted</th>
                      <th className="py-1.5 pr-2 text-right font-medium">realized</th>
                      <th className="py-1.5 text-right font-medium">95% CI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bins.map((b) => {
                      const covers = b.realized_lo <= b.predicted_uplift_mean && b.predicted_uplift_mean <= b.realized_hi;
                      return (
                        <tr key={b.bin} className="border-b border-line/40 last:border-0">
                          <td className="tnum py-1.5 pr-2 text-muted">{b.bin}</td>
                          <td className="tnum py-1.5 pr-2 text-right">{fmtInt(b.n)}</td>
                          <td className="tnum py-1.5 pr-2 text-right">{fmtNum(b.predicted_uplift_mean, 4)}</td>
                          <td className="tnum py-1.5 pr-2 text-right">{fmtNum(b.realized_uplift, 4)}</td>
                          <td className={`tnum py-1.5 text-right ${covers ? "text-muted" : "text-warn"}`}>
                            [{fmtNum(b.realized_lo, 4)}, {fmtNum(b.realized_hi, 4)}]
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          );
        })}
      </div>

      <Note>
        Wide intervals in the low bins are expected: at a 0.29% conversion rate even tens of thousands of rows
        produce only hundreds of control conversions to difference against. A bin whose CI misses the diagonal
        is a miscalibration, not noise — but read the CI width before judging.
      </Note>
    </>
  );
}
