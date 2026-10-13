"""One-off integrity + schema validation of the raw Criteo Uplift v2.1 archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
OUT = ROOT / "data" / "raw" / "validation.json"

EXPECTED_HEADER = "f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure"
N_FEATURES = 12


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def scan() -> dict:
    t0 = time.time()
    n = 0
    malformed = 0
    treated = 0
    conv = visit = expo = 0
    conv_t = conv_c = visit_t = visit_c = expo_t = expo_c = 0
    fmin = [float("inf")] * N_FEATURES
    fmax = [float("-inf")] * N_FEATURES

    with gzip.open(RAW, "rt", encoding="utf-8") as fh:
        header = next(fh).rstrip("\n")
        if header != EXPECTED_HEADER:
            raise SystemExit(f"HEADER MISMATCH\n got: {header}\n exp: {EXPECTED_HEADER}")
        print(f"header OK ({len(header.split(','))} columns)", flush=True)

        for line in fh:
            parts = line.split(",")
            if len(parts) != 16:
                malformed += 1
                continue
            n += 1
            for i in range(N_FEATURES):
                v = float(parts[i])
                if v < fmin[i]:
                    fmin[i] = v
                if v > fmax[i]:
                    fmax[i] = v
            t = parts[12] == "1"
            c = parts[13] == "1"
            v = parts[14] == "1"
            e = parts[15][0] == "1"
            treated += t
            conv += c
            visit += v
            expo += e
            if t:
                conv_t += c
                visit_t += v
                expo_t += e
            else:
                conv_c += c
                visit_c += v
                expo_c += e
            if n % 2_000_000 == 0:
                print(f"  {n:,} rows  ({time.time() - t0:.0f}s)", flush=True)

    control = n - treated
    return {
        "rows": n,
        "malformed_rows": malformed,
        "n_treated": treated,
        "n_control": control,
        "treatment_ratio": treated / n,
        "conversion_rate": conv / n,
        "visit_rate": visit / n,
        "exposure_rate": expo / n,
        "conv_rate_treated": conv_t / treated,
        "conv_rate_control": conv_c / control,
        "visit_rate_treated": visit_t / treated,
        "visit_rate_control": visit_c / control,
        "exposure_rate_treated": expo_t / treated,
        "exposure_rate_control": expo_c / control,
        "naive_ate_conversion": conv_t / treated - conv_c / control,
        "naive_ate_visit": visit_t / treated - visit_c / control,
        "feature_min": fmin,
        "feature_max": fmax,
        "scan_elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}", file=sys.stderr)
        return 1
    print("hashing...", flush=True)
    digest = sha256_of(RAW)
    print(f"sha256={digest}", flush=True)
    stats = scan()
    report = {"file": RAW.name, "bytes": RAW.stat().st_size, "sha256": digest, **stats}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
