import type { ReactNode } from "react";

export function Card({ title, subtitle, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-panel p-5 ${className}`}>
      {title && (
        <header className="mb-4">
          <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
          {subtitle && <p className="mt-1 text-[13px] leading-snug text-muted">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone = "default" }: {
  label: string; value: ReactNode; sub?: ReactNode;
  tone?: "default" | "accent" | "warn" | "bad";
}) {
  const toneCls =
    tone === "accent" ? "text-accent" : tone === "warn" ? "text-warn" : tone === "bad" ? "text-bad" : "";
  return (
    <div className="rounded-lg border border-line bg-panel p-4">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className={`tnum mt-1.5 text-xl font-semibold ${toneCls}`}>{value}</div>
      {sub && <div className="mt-1 text-xs leading-snug text-muted">{sub}</div>}
    </div>
  );
}

export function Badge({ children, tone = "default" }: {
  children: ReactNode; tone?: "default" | "accent" | "muted" | "warn" | "bad";
}) {
  const cls = {
    default: "border-line text-fg",
    accent: "border-accent/40 text-accent",
    muted: "border-line text-muted",
    warn: "border-warn/40 text-warn",
    bad: "border-bad/40 text-bad",
  }[tone];
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-[11px] leading-none ${cls}`}>
      {children}
    </span>
  );
}

/** Horizontal confidence-interval whisker around a point estimate. */
export function CiBar({ lo, hi, est, fmt }: {
  lo: number; hi: number; est: number; fmt: (v: number) => string;
}) {
  const span = hi - lo || 1e-12;
  const pad = 0.12 * span;
  const min = lo - pad, width = span + 2 * pad;
  const pct = (v: number) => (100 * (v - min)) / width;
  const estClamped = Math.min(Math.max(est, lo), hi);
  return (
    <span className="inline-flex w-44 items-center" title={`[${fmt(lo)}, ${fmt(hi)}]`}>
      <span className="relative h-2 w-full rounded bg-panel2">
        <span
          className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-accent2/70"
          style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }}
        />
        <span
          className="absolute top-1/2 h-2.5 w-[2px] -translate-y-1/2 bg-accent2"
          style={{ left: `${pct(estClamped)}%` }}
          title={fmt(est)}
        />
      </span>
    </span>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 border-l-2 border-line pl-3 text-[13px] leading-relaxed text-muted">
      {children}
    </p>
  );
}

export function PageHeader({ title, lead }: { title: string; lead: ReactNode }) {
  return (
    <div className="mb-8 max-w-3xl">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">{lead}</p>
    </div>
  );
}
