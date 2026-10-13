// Copies build-time artifacts into public/ so client components can fetch them
// (tree dumps for the scorer playground), and verifies web/data is populated.
import { cpSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const dataDir = join(process.cwd(), "data");
const publicData = join(process.cwd(), "public", "data");

const required = [
  "dataset.json", "run.json", "effects.json", "curves.json", "allocation.json",
];
const missing = required.filter((f) => !existsSync(join(dataDir, f)));
if (missing.length) {
  console.error(
    `web/data is missing ${missing.join(", ")}.\n` +
    "Run `uplift run` (Python) first so artifacts exist, then rebuild.",
  );
  process.exit(1);
}

cpSync(dataDir, publicData, { recursive: true });
console.log(
  "public/data <-",
  readdirSync(join(publicData, "models")).length + " model dumps +",
  required.length + " core artifacts",
);
