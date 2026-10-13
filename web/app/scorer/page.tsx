import { PageHeader } from "@/components/ui";
import ScorerPlayground from "@/components/ScorerPlayground";
import { getModelManifest } from "@/lib/data";

// Feature ranges observed in the raw Criteo v2.1 archive (scripts/validate_raw.py).
const FEATURE_RANGES: [number, number][] = [
  [12.62, 26.75], [10.06, 16.34], [8.21, 9.05], [-8.40, 4.68],
  [10.28, 21.12], [-9.01, 4.12], [-31.43, 0.29], [4.83, 11.99],
  [3.64, 3.97], [13.19, 75.30], [5.30, 6.47], [-1.38, -0.17],
];

export default async function ScorerPage() {
  const manifest = await getModelManifest();
  const entries = Object.fromEntries(
    Object.entries(manifest.models).map(([name, m]) => [name, {
      kind: m.kind, is_causal: m.is_causal, description: m.description,
    }]),
  );

  return (
    <>
      <PageHeader
        title="Scorer playground"
        lead="The fitted models, running in your browser. Same tree dumps the Python pipeline wrote, replayed by a ~40-line TypeScript scorer — move the sliders and watch every policy re-rank this user."
      />
      <ScorerPlayground
        manifest={entries}
        featureRanges={FEATURE_RANGES}
        defaultCost={0.002}
        defaultValue={12}
      />
    </>
  );
}
