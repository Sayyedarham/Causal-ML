"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS: [string, string][] = [
  ["/", "Overview"],
  ["/effects", "Effects"],
  ["/curves", "Curves"],
  ["/calibration", "Calibration"],
  ["/allocation", "Budget"],
  ["/exposure", "Exposure"],
  ["/methods", "Methods"],
  ["/scorer", "Scorer"],
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ink/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3">
        <Link href="/" className="mr-2 flex items-baseline gap-2">
          <span className="font-semibold tracking-tight">uplift</span>
          <span className="text-xs text-muted">Criteo v2.1</span>
        </Link>
        <nav className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
          {LINKS.map(([href, label]) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={active
                  ? "text-accent underline decoration-accent/40 underline-offset-4"
                  : "text-muted hover:text-fg"}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
