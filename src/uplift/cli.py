"""Command-line interface.

    uplift fetch      Download the Criteo Uplift v2.1 archive, verify its checksum.
    uplift validate   Check archive integrity against the pinned sha256.
    uplift run        Execute pipeline stages (data -> fit -> ... -> export).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from .config import DATASET_SHA256, DATASET_URL, Paths, load_config

CHUNK = 1 << 22  # 4 MiB


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args: argparse.Namespace) -> int:
    paths = Paths()
    paths.ensure()
    archive = paths.raw_archive

    if archive.exists() and not args.force:
        digest = _sha256_of(archive)
        if digest == DATASET_SHA256:
            print(f"already downloaded and verified: {archive}")
            return 0
        print(f"archive exists but sha256 mismatch ({digest}); re-downloading")

    part = archive.with_suffix(archive.suffix + ".part")
    print(f"downloading {DATASET_URL}")
    print(f"  -> {part}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0 and blocks % 512 == 0:
            print(f"  {done / 1e6:8.0f} / {total / 1e6:.0f} MB", flush=True)

    tmp_part = part.with_suffix(".download")
    urllib.request.urlretrieve(DATASET_URL, tmp_part, reporthook=hook)
    tmp_part.rename(part)

    digest = _sha256_of(part)
    if digest != DATASET_SHA256:
        part.unlink()
        print(f"FATAL: sha256 mismatch after download\n  got {digest}\n  exp {DATASET_SHA256}",
              file=sys.stderr)
        return 1
    part.rename(archive)
    print(f"verified sha256={digest}")
    print(f"saved {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = Paths()
    archive = paths.raw_archive
    if not archive.exists():
        print(f"missing {archive}; run `uplift fetch` first", file=sys.stderr)
        return 1

    print("hashing archive ...")
    digest = _sha256_of(archive)
    ok = digest == DATASET_SHA256
    print(f"sha256 {'OK' if ok else 'MISMATCH'}: {digest}")
    if not ok:
        print(f"  expected: {DATASET_SHA256}", file=sys.stderr)

    validation = paths.raw_validation
    if validation.exists():
        import json

        v = json.loads(validation.read_text())
        print(f"rows={v.get('rows'):,}  malformed={v.get('malformed_rows')}  "
              f"treated={v.get('treatment_ratio'):.4f}  "
              f"conversion={v.get('conversion_rate'):.5f}")
    else:
        print("(no raw/validation.json; run scripts/validate_raw.py for a full scan)")
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run

    overrides: dict = {}
    if args.train_rows is not None:
        overrides["train_sample_rows"] = args.train_rows
    if args.valid_rows is not None:
        overrides["valid_sample_rows"] = args.valid_rows
    cfg = load_config(args.config, **overrides)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    run(cfg, stages=stages, verbose=not args.quiet)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uplift",
                                     description="Incremental-conversion targeting engine")
    parser.add_argument("--version", action="version", version="uplift 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download the Criteo Uplift v2.1 archive")
    p_fetch.add_argument("--force", action="store_true",
                         help="re-download even if the archive already verifies")
    p_fetch.set_defaults(func=cmd_fetch)

    p_val = sub.add_parser("validate", help="verify archive checksum and report raw facts")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="execute pipeline stages")
    p_run.add_argument("--config", default=None, help="JSON file patching the default Config")
    p_run.add_argument("--stages", default="data,fit,diagnostics,evaluate,allocate,export",
                       help="comma-separated subset of: data,fit,diagnostics,evaluate,allocate,export")
    p_run.add_argument("--train-rows", type=int, default=None,
                       help="override train split sample size")
    p_run.add_argument("--valid-rows", type=int, default=None,
                       help="override valid split sample size")
    p_run.add_argument("--quiet", action="store_true", help="suppress progress logs")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
