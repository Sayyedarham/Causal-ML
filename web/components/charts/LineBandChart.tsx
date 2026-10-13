"use client";

// Multi-series SVG line chart with optional bootstrap bands and a hover
// crosshair. No chart library: the whole dashboard is hand-drawn SVG.
import { useMemo, useRef, useState } from "react";
import { applyFmt, type FmtSpec } from "@/lib/format";

export interface Series {
  name: string;
  color: string;
  dashed?: boolean;
  xs: number[];
  ys: number[];
  band?: { los: number[]; his: number[] } | null;
}

interface Props {
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xFmt: FmtSpec;
  yFmt: FmtSpec;
  yFromZero?: boolean;
  legend?: boolean;
  xDomain?: [number, number];
}

export default function LineBandChart({
  series, height = 340, xLabel, yLabel, xFmt, yFmt, yFromZero = false, legend = true, xDomain,
}: Props) {
  const W = 900;
  const H = height;
  const M = { top: 16, right: 140, bottom: 40, left: 76 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const xf = applyFmt(xFmt);
  const yf = applyFmt(yFmt);

  const xDomainDefault = useMemo<[number, number]>(() => {
    if (xDomain) return xDomain;
    const all = series.flatMap((s) => s.xs);
    return [Math.min(...all), Math.max(...all)];
  }, [series, xDomain]);

  const yExtent = useMemo<[number, number]>(() => {
    const vals = series.flatMap((s) => {
      const v = [s.ys.filter(isFinite)];
      if (s.band) { v.push(s.band.los.filter(isFinite)); v.push(s.band.his.filter(isFinite)); }
      return v.flat();
    });
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (yFromZero) lo = Math.min(0, lo);
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) { lo = (lo || 0) - 1; hi = hi + 1 || 1; }
    const pad = 0.06 * (hi - lo);
    return [lo - pad, hi + pad];
  }, [series, yFromZero]);

  const px = (x: number) => M.left + ((x - xDomainDefault[0]) / (xDomainDefault[1] - xDomainDefault[0] || 1)) * iw;
  const py = (y: number) => M.top + ih - ((y - yExtent[0]) / (yExtent[1] - yExtent[0] || 1)) * ih;

  const yTicks = niceTicks(yExtent[0], yExtent[1], 5);
  const xTicks = niceTicks(xDomainDefault[0], xDomainDefault[1], 6);

  const path = (xs: number[], ys: number[]) => {
    let d = "";
    let pen = false;
    for (let i = 0; i < xs.length; i++) {
      if (!isFinite(ys[i])) { pen = false; continue; }
      d += (pen ? "L" : "M") + px(xs[i]).toFixed(1) + " " + py(ys[i]).toFixed(1);
      pen = true;
    }
    return d;
  };

  const bandPath = (s: Series) => {
    if (!s.band) return "";
    const { los, his } = s.band;
    const up: string[] = [], down: string[] = [];
    for (let i = 0; i < s.xs.length; i++) {
      if (!isFinite(ys_at(s, i))) continue;
      up.push(`${px(s.xs[i]).toFixed(1)} ${py(his[i]).toFixed(1)}`);
      down.unshift(`${px(s.xs[i]).toFixed(1)} ${py(los[i]).toFixed(1)}`);
    }
    return "M" + up.join("L") + "L" + down.join("L") + "Z";
  };
  const ys_at = (s: Series, i: number) => s.ys[i];

  const nearest = (clientX: number): number | null => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const sx = ((clientX - rect.left) / rect.width) * W;
    if (sx < M.left || sx > M.left + iw) return null;
    const xv = xDomainDefault[0] + ((sx - M.left) / iw) * (xDomainDefault[1] - xDomainDefault[0]);
    const base = series[0];
    let best = 0, bd = Infinity;
    for (let i = 0; i < base.xs.length; i++) {
      const d = Math.abs(base.xs[i] - xv);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full select-none"
        onMouseMove={(e) => setHoverX(nearest(e.clientX))}
        onMouseLeave={() => setHoverX(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + iw} y1={py(t)} y2={py(t)} stroke="#232a3a" strokeWidth={1} />
            <text x={M.left - 8} y={py(t) + 4} textAnchor="end" fontSize={12} fill="#8b94a7" className="tnum">
              {yf(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={px(t)} x2={px(t)} y1={M.top} y2={M.top + ih} stroke="#232a3a" strokeWidth={1} opacity={0.5} />
            <text x={px(t)} y={H - 12} textAnchor="middle" fontSize={12} fill="#8b94a7" className="tnum">
              {xf(t)}
            </text>
          </g>
        ))}
        {yExtent[0] < 0 && yExtent[1] > 0 && (
          <line x1={M.left} x2={M.left + iw} y1={py(0)} y2={py(0)} stroke="#5b6478" strokeWidth={1} />
        )}

        {series.map((s) => (
          <g key={s.name}>
            {s.band && (
              <path d={bandPath(s)} fill={s.color} opacity={0.13} stroke="none" />
            )}
            <path d={path(s.xs, s.ys)} fill="none" stroke={s.color} strokeWidth={s.dashed ? 1.5 : 2.25}
                  strokeDasharray={s.dashed ? "5 4" : undefined} />
          </g>
        ))}

        {hoverX !== null && (
          <g>
            <line x1={px(series[0].xs[hoverX])} x2={px(series[0].xs[hoverX])}
                  y1={M.top} y2={M.top + ih} stroke="#8b94a7" strokeWidth={1} strokeDasharray="3 3" />
            {series.map((s) => isFinite(s.ys[hoverX]) && (
              <circle key={s.name} cx={px(s.xs[hoverX])} cy={py(s.ys[hoverX])} r={3.5} fill={s.color} />
            ))}
          </g>
        )}

        {legend && (
          <g>
            {series.map((s, i) => (
              <g key={s.name} transform={`translate(${M.left + iw + 12}, ${M.top + 4 + i * 20})`}>
                <line x1={0} x2={22} y1={-4} y2={-4} stroke={s.color} strokeWidth={2.25}
                      strokeDasharray={s.dashed ? "5 4" : undefined} />
                {s.band && <rect x={0} y={-8} width={22} height={8} fill={s.color} opacity={0.15} />}
                <text x={28} y={0} fontSize={12} fill="#dde3ee">{s.name}</text>
              </g>
            ))}
          </g>
        )}

        {xLabel && (
          <text x={M.left + iw / 2} y={H + 2} textAnchor="middle" fontSize={12} fill="#8b94a7">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={16} y={M.top + ih / 2} textAnchor="middle" fontSize={12} fill="#8b94a7"
                transform={`rotate(-90 16 ${M.top + ih / 2})`}>
            {yLabel}
          </text>
        )}
      </svg>

      {hoverX !== null && (
        <div className="pointer-events-none absolute left-2 top-2 rounded border border-line bg-ink/95 px-3 py-2 text-xs">
          <div className="tnum mb-1 text-muted">{xf(series[0].xs[hoverX])}</div>
          {series.map((s) => (
            <div key={s.name} className="flex items-center gap-2">
              <span className="inline-block h-[3px] w-4" style={{ background: s.color }} />
              <span className="tnum">{yf(s.ys[hoverX])}</span>
              <span className="text-muted">{s.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(v);
  return out;
}
