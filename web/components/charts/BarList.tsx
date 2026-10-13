// Horizontal bar list (feature importance, |SMD|, gain shares).
export default function BarList({ rows, fmt, color = "#5eead4", max }: {
  rows: { label: string; value: number; note?: string }[];
  fmt: (v: number) => string;
  color?: string;
  max?: number;
}) {
  const hi = max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-12);
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-3 text-[13px]">
          <span className="w-20 shrink-0 text-muted">{r.label}</span>
          <span className="relative h-3 flex-1 rounded-sm bg-panel2">
            <span
              className="absolute inset-y-0 left-0 rounded-sm"
              style={{ width: `${(100 * Math.abs(r.value)) / hi}%`, background: color, opacity: 0.75 }}
            />
          </span>
          <span className="tnum w-20 shrink-0 text-right">{fmt(r.value)}</span>
        </div>
      ))}
      {rows.length === 0 && <div className="text-[13px] text-muted">no data</div>}
    </div>
  );
}
