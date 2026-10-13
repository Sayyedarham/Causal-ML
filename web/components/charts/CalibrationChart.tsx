// Calibration plot: realized uplift (with CI whiskers) against predicted uplift,
// plus the identity line a perfectly calibrated model would follow.
import { applyFmt, type FmtSpec } from "@/lib/format";

interface Props {
  points: { bin: number; predicted: number; realized: number; lo: number; hi: number; n: number }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xFmt: FmtSpec;
  yFmt: FmtSpec;
}

export default function CalibrationChart({ points, height = 320, xLabel, yLabel, xFmt, yFmt }: Props) {
  const xf = applyFmt(xFmt);
  const yf = applyFmt(yFmt);
  const W = 900, H = height;
  const M = { top: 16, right: 24, bottom: 44, left: 76 };
  const iw = W - M.left - M.right, ih = H - M.top - M.bottom;

  const all = points.flatMap((p) => [p.realized, p.lo, p.hi, p.predicted]).filter(isFinite);
  const lo = Math.min(0, ...all), hi = Math.max(...all, 1e-12);
  const pad = 0.08 * (hi - lo || 1);
  const dom: [number, number] = [lo - pad, hi + pad];
  const px = (v: number) => M.left + ((v - dom[0]) / (dom[1] - dom[0] || 1)) * iw;
  const py = (v: number) => M.top + ih - ((v - dom[0]) / (dom[1] - dom[0] || 1)) * ih;
  const f = (v: number) => `${px(v).toFixed(1)} ${py(v).toFixed(1)}`;

  const ticks = [dom[0], dom[0] + (dom[1] - dom[0]) / 2, dom[1]];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={M.left} x2={M.left + iw} y1={py(t)} y2={py(t)} stroke="#232a3a" />
          <text x={M.left - 8} y={py(t) + 4} textAnchor="end" fontSize={12} fill="#8b94a7" className="tnum">{yf(t)}</text>
          <line x1={px(t)} x2={px(t)} y1={M.top} y2={M.top + ih} stroke="#232a3a" opacity={0.5} />
          <text x={px(t)} y={H - 14} textAnchor="middle" fontSize={12} fill="#8b94a7" className="tnum">{xf(t)}</text>
        </g>
      ))}
      {dom[0] < 0 && dom[1] > 0 && (
        <line x1={M.left} x2={M.left + iw} y1={py(0)} y2={py(0)} stroke="#5b6478" />
      )}
      <line x1={px(dom[0])} y1={py(dom[0])} x2={px(dom[1])} y2={py(dom[1])}
            stroke="#8b94a7" strokeDasharray="5 4" strokeWidth={1.5} />
      {points.map((p) => (
        <g key={p.bin}>
          <line x1={px(p.predicted)} x2={px(p.predicted)} y1={py(p.lo)} y2={py(p.hi)}
                stroke="#5eead4" strokeWidth={2} opacity={0.8} />
          <circle cx={px(p.predicted)} cy={py(p.realized)} r={4.5} fill="#5eead4">
            <title>{`n=${p.n}\npredicted ${xf(p.predicted)}\nrealized ${yf(p.realized)} [${yf(p.lo)}, ${yf(p.hi)}]`}</title>
          </circle>
        </g>
      ))}
      {xLabel && <text x={M.left + iw / 2} y={H + 2} textAnchor="middle" fontSize={12} fill="#8b94a7">{xLabel}</text>}
      {yLabel && (
        <text x={16} y={M.top + ih / 2} textAnchor="middle" fontSize={12} fill="#8b94a7"
              transform={`rotate(-90 16 ${M.top + ih / 2})`}>{yLabel}</text>
      )}
    </svg>
  );
}
