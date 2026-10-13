"use client";

// The whole allocation is recomputed client-side from the measured incremental
// grid: incremental conversions do not depend on economics, so cost/value only
// rescale spend and profit. A reader can substitute their own numbers.
import { useMemo, useState } from "react";
import LineBandChart from "@/components/charts/LineBandChart";
import { Card, Note } from "@/components/ui";
import { modelColor, modelLabel } from "@/lib/colors";
import { fmtInt, fmtNum, fmtPct, fmtUsd } from "@/lib/format";

export interface AllocSeries {
  model: string;
  isCausal: boolean;
  phi: number[];
  inc: number[];
  lo: number[];
  hi: number[];
}

interface Props {
  population: number;
  defaultCost: number;
  defaultValue: number;
  defaultModel?: string;
  series: AllocSeries[];
  gridPhis: number[];
}

const BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0];

export default function AllocationExplorer({ population, defaultCost, defaultValue, defaultModel, series, gridPhis }: Props) {
  const [cost, setCost] = useState(defaultCost);
  const [value, setValue] = useState(defaultValue);
  const [selected, setSelected] = useState(defaultModel ?? series[0]?.model ?? "");

  const econ = useMemo(() => {
    const rebuilt = series.map((s) => {
      const n = s.phi.length;
      const spend = s.phi.map((p) => cost * p * population);
      const profit = s.inc.map((inc, i) => value * inc - spend[i]);
      const profitLo = s.lo.map((lo, i) => value * lo - spend[i]);
      const profitHi = s.hi.map((hi, i) => value * hi - spend[i]);
      const roi = profit.map((pr, i) => (spend[i] > 0 ? pr / spend[i] : NaN));

      let jmax = 0;
      for (let j = 1; j < n; j++) if (profit[j] > profit[jmax]) jmax = j;
      const best = profit[jmax];
      let lo_phi = s.phi[jmax], hi_phi = s.phi[jmax];
      if (best > 0) {
        const keep = profit.map((p) => p >= best * 0.95);
        const idx = keep.map((k, j) => (k ? j : -1)).filter((j) => j >= 0);
        if (idx.length) { lo_phi = s.phi[idx[0]]; hi_phi = s.phi[idx[idx.length - 1]]; }
      }
      const reached = s.phi[jmax] * population;
      const incAt = s.inc[jmax];
      return {
        model: s.model, spend, profit, profitLo, profitHi, roi,
        jmax, best, lo_phi, hi_phi, reached, incAt,
        breakeven: incAt > 0 ? (value * incAt) / reached : 0,
      };
    });
    return rebuilt;
  }, [series, cost, value, population]);

  const sel = econ.find((e) => e.model === selected) ?? econ[0];
  const selSeries = series.find((s) => s.model === sel?.model);

  const comparison = useMemo(() => {
    const rows: { phi: number; cells: { model: string; inc: number; profit: number; roi: number }[] }[] = [];
    for (const phi of BUDGET_FRACTIONS) {
      const cells = series.map((s) => {
        let j = 0, bd = Infinity;
        for (let i = 0; i < s.phi.length; i++) {
          const d = Math.abs(s.phi[i] - phi);
          if (d < bd) { bd = d; j = i; }
        }
        const spend = cost * s.phi[j] * population;
        const profit = value * s.inc[j] - spend;
        return { model: s.model, inc: s.inc[j], profit, roi: spend > 0 ? profit / spend : NaN };
      });
      rows.push({ phi, cells });
    }
    return rows;
  }, [series, cost, value, population]);

  if (!sel || !selSeries) return null;

  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card
          title="Profit curve"
          subtitle={`profit(φ) = value × incremental(φ) − cost × φ × ${fmtInt(population)}. Built from the measured test-split uplift of ${modelLabel(sel.model)}; the band is the bootstrap interval on incremental conversions.`}
        >
          <LineBandChart
            series={[
              {
                name: `${modelLabel(sel.model)} profit`,
                color: modelColor(sel.model),
                xs: selSeries.phi,
                ys: sel.profit,
                band: { los: sel.profitLo, his: sel.profitHi },
              },
              {
                name: "target everyone",
                color: "#8b94a7",
                dashed: true,
                xs: selSeries.phi,
                ys: selSeries.phi.map(() => sel.profit[sel.profit.length - 1]),
                band: null,
              },
            ]}
            xLabel="targeting fraction φ"
            yLabel="profit"
            xFmt={{ kind: "num", digits: 2 }}
            yFmt={{ kind: "usd" }}
          />
          <Note>
            The curve is flat near its peak — the first-order condition zeroes marginal profit there — so the
            argmax is poorly identified. The shaded range on the slider below marks the plateau within 5% of
            maximum profit; any budget inside it is as defensible as the peak.
          </Note>
        </Card>

        <div className="space-y-4">
          <Card title="Your economics">
            <label className="block text-[13px]">
              <span className="flex justify-between text-muted">
                cost per impression <span className="tnum text-fg">{fmtUsd(cost, 4)}</span>
              </span>
              <input type="range" min={0.0005} max={0.012} step={0.0001} value={cost}
                     onChange={(e) => setCost(parseFloat(e.target.value))} className="mt-1 w-full accent-teal-300" />
            </label>
            <label className="mt-4 block text-[13px]">
              <span className="flex justify-between text-muted">
                value per incremental conversion <span className="tnum text-fg">{fmtUsd(value, 2)}</span>
              </span>
              <input type="range" min={1} max={50} step={0.5} value={value}
                     onChange={(e) => setValue(parseFloat(e.target.value))} className="mt-1 w-full accent-teal-300" />
            </label>
            <div className="mt-4 border-t border-line pt-3 text-[12px] leading-relaxed text-muted">
              Criteo publishes no campaign economics; the defaults are explicit assumptions
              ({fmtUsd(defaultCost, 3)}/{fmtUsd(defaultValue)}). Everything on this page recomputes from the
              measured uplift grid as you drag.
            </div>
          </Card>

          <Card title="Optimal policy" subtitle={`for ${modelLabel(sel.model)} at your economics`}>
            <div className="space-y-2.5 text-[13px]">
              <KV k="target fraction" v={fmtPct(sel.lo_phi, 1) === fmtPct(sel.hi_phi, 1) ? fmtPct(sel.lo_phi, 1) : `${fmtPct(sel.lo_phi, 0)} – ${fmtPct(sel.hi_phi, 0)} (plateau)`} />
              <KV k="users reached" v={fmtInt(sel.reached)} />
              <KV k="spend" v={fmtUsd(cost * selSeries.phi[sel.jmax] * population)} />
              <KV k="incr. conversions" v={fmtInt(sel.incAt)} tone="accent" />
              <KV k="profit" v={fmtUsd(sel.best)} tone={sel.best >= 0 ? "accent" : "bad"} />
              <KV k="ROI" v={fmtNum(sel.roi[sel.jmax], 2) + "×"} />
              <KV k="breakeven CPM" v={fmtUsd(sel.breakeven, 4)} sub="highest cost/impression that still breaks even here" />
            </div>
          </Card>
        </div>
      </div>

      <Card
        title="Policies at matched budgets"
        subtitle="Every policy reaches the same number of people at the same spend; the only difference is whom each chose. Winners per budget are highlighted."
        className="mt-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th className="py-1.5 pr-2 font-medium">budget</th>
                {series.map((s) => (
                  <th key={s.model} className="py-1.5 pr-2 text-right font-medium">{modelLabel(s.model)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparison.map(({ phi, cells }) => {
                const bestInc = Math.max(...cells.filter((c) => isFinite(c.inc)).map((c) => c.inc));
                return (
                  <tr key={phi} className="border-b border-line/40 last:border-0">
                    <td className="tnum py-2 pr-2 text-muted">{fmtUsd(cost * phi * population)} · {fmtPct(phi, 0)}</td>
                    {cells.map((c) => {
                      const win = isFinite(c.inc) && Math.abs(c.inc - bestInc) < 1e-9;
                      return (
                        <td key={c.model} className={`tnum py-2 pr-2 text-right ${win ? "font-semibold text-accent" : ""}`}>
                          {fmtInt(c.inc)}
                          <span className="ml-1.5 text-[11px] text-muted">{fmtUsd(c.profit)}</span>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Note>
          Cells show incremental conversions with profit beneath. At small budgets the ranking gap between
          causal models and random targeting is largest — that is where targeting actually pays.
        </Note>
      </Card>

    </>
  );

  function KV({ k, v, sub, tone }: { k: string; v: string; sub?: string; tone?: "accent" | "bad" }) {
    return (
      <div className="flex items-baseline justify-between gap-3 border-b border-line/40 pb-1.5">
        <span className="text-muted">{k}</span>
        <span className={`tnum font-medium ${tone === "accent" ? "text-accent" : tone === "bad" ? "text-bad" : ""}`}>
          {v}
          {sub && <div className="mt-0.5 text-[11px] font-normal text-muted">{sub}</div>}
        </span>
      </div>
    );
  }
}
