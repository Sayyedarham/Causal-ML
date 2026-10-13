"use client";

// Scorer playground: the exported tree ensembles are replayed in the browser,
// bit-for-bit with Python (same <= routing, same leaf sums, same links).
import { useEffect, useMemo, useState } from "react";
import { predictUplift } from "@/lib/scorer";
import type { ModelDump } from "@/lib/types";
import { modelColor, modelLabel } from "@/lib/colors";
import { fmtNum, fmtUsd } from "@/lib/format";

interface ManifestEntry { kind: string; is_causal: boolean; description: string }

interface Props {
  manifest: Record<string, ManifestEntry>;
  featureRanges: [number, number][];
  defaultCost: number;
  defaultValue: number;
}

export default function ScorerPlayground({ manifest, featureRanges, defaultCost, defaultValue }: Props) {
  const names = Object.keys(manifest);
  const [dumps, setDumps] = useState<Record<string, ModelDump>>({});
  const [status, setStatus] = useState("loading model dumps…");
  const [x, setX] = useState<number[]>(() => featureRanges.map(([lo, hi]) => (lo + hi) / 2));
  const [cost, setCost] = useState(defaultCost);
  const [value, setValue] = useState(defaultValue);

  useEffect(() => {
    let alive = true;
    (async () => {
      for (const name of names) {
        try {
          const res = await fetch(`/data/models/${name}.json`);
          const dump = (await res.json()) as ModelDump;
          if (!alive) return;
          setDumps((d) => ({ ...d, [name]: dump }));
        } catch {
          if (!alive) return;
          setStatus(`failed to load ${name}`);
          return;
        }
      }
      if (alive) setStatus("");
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rows = useMemo(() => {
    const threshold = cost / value;
    return names.map((name) => {
      const dump = dumps[name];
      const score = dump ? predictUplift(dump, x) : NaN;
      const nTrees = dump ? Object.values(dump.components).reduce((a, c) => a + c.trees.length, 0) : 0;
      return {
        name, score, nTrees,
        target: dump ? score > threshold : false,
        isCausal: manifest[name].is_causal,
      };
    });
  }, [dumps, x, cost, value, names, manifest]);

  const threshold = cost / value;

  return (
    <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
      <div className="space-y-4">
        <div className="rounded-lg border border-line bg-panel p-5">
          <h2 className="text-[15px] font-semibold">User features</h2>
          <p className="mt-1 text-[12px] text-muted">
            Twelve anonymized Criteo features. Drag any slider; every loaded model re-scores this user
            instantly, in your browser.
          </p>
          <div className="mt-4 space-y-3">
            {featureRanges.map(([lo, hi], j) => (
              <label key={j} className="block text-[12px]">
                <span className="flex justify-between text-muted">
                  <span>f{j}</span>
                  <span className="tnum text-fg">{fmtNum(x[j], 2)}</span>
                </span>
                <input
                  type="range" min={lo} max={hi}
                  step={(hi - lo) / 200}
                  value={x[j]}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setX((prev) => prev.map((p, i) => (i === j ? v : p)));
                  }}
                  className="w-full accent-teal-300"
                />
              </label>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-line bg-panel p-5">
          <h2 className="text-[15px] font-semibold">Targeting economics</h2>
          <label className="mt-3 block text-[12px]">
            <span className="flex justify-between text-muted">
              cost / impression <span className="tnum text-fg">{fmtUsd(cost, 4)}</span>
            </span>
            <input type="range" min={0.0005} max={0.012} step={0.0001} value={cost}
                   onChange={(e) => setCost(parseFloat(e.target.value))} className="mt-1 w-full accent-teal-300" />
          </label>
          <label className="mt-3 block text-[12px]">
            <span className="flex justify-between text-muted">
              value / conversion <span className="tnum text-fg">{fmtUsd(value, 2)}</span>
            </span>
            <input type="range" min={1} max={50} step={0.5} value={value}
                   onChange={(e) => setValue(parseFloat(e.target.value))} className="mt-1 w-full accent-teal-300" />
          </label>
          <p className="tnum mt-3 text-[12px] text-muted">
            target when τ(x) &gt; c/v = <span className="text-fg">{fmtNum(threshold, 5)}</span>
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-lg border border-line bg-panel p-5">
          <div className="flex items-baseline justify-between">
            <h2 className="text-[15px] font-semibold">Scores for this user</h2>
            {status && <span className="text-[12px] text-muted">{status}</span>}
          </div>
          <div className="mt-4 space-y-3">
            {rows.map((r) => {
              const lo = Math.min(r.score, 0), hi = Math.max(r.score, 0);
              const scale = Math.max(...rows.filter((q) => q.isCausal).map((q) => Math.abs(q.score) || 1e-9), 1e-9);
              const w = (Math.abs(r.score) / scale) * 50;
              return (
                <div key={r.name} className="flex items-center gap-3">
                  <span className="w-28 shrink-0 text-[13px]">{modelLabel(r.name)}</span>
                  <span className="relative h-4 flex-1 rounded-sm bg-panel2">
                    <span className="absolute inset-y-0 left-1/2 w-px bg-line" />
                    {!isNaN(r.score) && (
                      <span
                        className="absolute inset-y-0 rounded-sm"
                        style={{
                          left: r.score >= 0 ? "50%" : `${50 - w}%`,
                          width: `${Math.max(w, 0.4)}%`,
                          background: modelColor(r.name),
                          opacity: 0.8,
                        }}
                      />
                    )}
                  </span>
                  <span className="tnum w-24 shrink-0 text-right text-[13px]">
                    {r.isCausal ? fmtNum(r.score, 5) : `${fmtNum(r.score, 5)} (P)`}
                  </span>
                  <span
                    className={`w-16 shrink-0 rounded px-1.5 py-0.5 text-center text-[11px] ${
                      !r.isCausal ? "text-muted"
                      : r.target ? "bg-accent/15 text-accent" : "bg-panel2 text-muted"
                    }`}
                  >
                    {r.isCausal ? (r.target ? "target" : "skip") : "—"}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-4 border-l-2 border-line pl-3 text-[12px] leading-relaxed text-muted">
            The response model&apos;s score is P(convert), not an effect — it is shown in parentheses for
            contrast. `target` means this user clears the c/v threshold under that model; disagreement between
            causal models here is exactly what an analyst should investigate before spending.
          </p>
        </div>

        <div className="rounded-lg border border-line bg-panel p-5">
          <h2 className="text-[15px] font-semibold">What is running</h2>
          <p className="mt-2 text-[12px] leading-relaxed text-muted">
            Each model is a set of flattened LightGBM tree dumps (feature / threshold / left / right / value
            arrays) exported by the Python pipeline. This page walks every tree — left when
            <span className="tnum"> x[f] &lt;= threshold</span> — sums leaf values, and applies the same link
            (logit for classifiers, identity for effect regressors). No server, no Python: the deployed site
            scores real users with the exact fitted artifacts.
          </p>
          <dl className="tnum mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-[12px] sm:grid-cols-3">
            {rows.filter((r) => r.nTrees > 0).map((r) => (
              <div key={r.name} className="flex justify-between gap-2 border-b border-line/40 py-0.5">
                <dt className="text-muted">{r.name}</dt>
                <dd>{r.nTrees} trees</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}
