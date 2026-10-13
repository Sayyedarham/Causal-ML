// Client-safe formatting helpers (no node imports).
const nz = (v: number) => (Object.is(v, -0) ? 0 : v);

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(nz(v));
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const abs = Math.abs(nz(v));
  if (abs !== 0 && abs < 1e-3) return nz(v).toExponential(2);
  if (abs >= 1e5) return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(nz(v));
  return nz(v).toFixed(digits);
}

export function fmtUsd(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const sign = v < 0 ? "−" : "";
  return sign + "$" + new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }).format(Math.abs(v));
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return (100 * v).toFixed(digits) + "%";
}

// Serializable format descriptor — server components can pass these to client
// chart components (functions cannot cross the server/client boundary).
export type FmtSpec =
  | { kind: "num"; digits?: number }
  | { kind: "usd"; digits?: number }
  | { kind: "pct"; digits?: number }
  | { kind: "int" };

export function applyFmt(spec: FmtSpec): (v: number) => string {
  switch (spec.kind) {
    case "num": return (v) => fmtNum(v, spec.digits ?? 2);
    case "usd": return (v) => fmtUsd(v, spec.digits ?? 0);
    case "pct": return (v) => fmtPct(v, spec.digits ?? 1);
    case "int": return (v) => fmtInt(v);
  }
}
