import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "uplift — incremental-conversion targeting",
  description:
    "Uplift modeling on the Criteo Uplift v2.1 randomized dataset: who converts because of the ad, and what budget should follow.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto max-w-6xl px-5 py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-5 pb-10 pt-4 text-xs text-muted">
          Criteo Uplift Prediction Dataset v2.1 · CC BY-NC-SA 4.0 · Diemert et al., AdKDD 2018.
          Static build — every number is computed by the Python pipeline and committed as JSON.
        </footer>
      </body>
    </html>
  );
}
