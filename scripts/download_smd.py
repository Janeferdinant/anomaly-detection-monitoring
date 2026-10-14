"""Download the Server Machine Dataset (SMD) and verify SHA-256 checksums.

Source: https://github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset, MIT licence), pinned to
one commit. The files (about 460 MB) go to ``data/raw/smd/`` (git-ignored) and are never committed.

Usage::

    python scripts/download_smd.py            # download + verify (skips files already present)
    python scripts/download_smd.py --machines machine-1-1 machine-2-1    # a subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_monitoring import smd  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--machines", nargs="+", default=list(smd.MACHINES))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    try:
        raw = smd.download(machines=args.machines, workers=args.workers)
    except smd.ChecksumError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Checksums OK for {len(args.machines)} machines ({len(smd.file_keys(args.machines))} files), commit {smd.COMMIT[:10]}")
    print(f"Ready: {raw.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
