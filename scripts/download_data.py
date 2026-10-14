"""Download the public dataset used by this repository and verify its checksum.

Dataset: Individual Household Electric Power Consumption (Hebrail & Berard, 2006),
UCI Machine Learning Repository, https://doi.org/10.24432/C58K54, licensed CC BY 4.0.

The data is downloaded at run time into ``data/raw/`` (git-ignored) and is never committed.

Usage::

    python scripts/download_data.py            # download + verify (skips if already valid)
    python scripts/download_data.py --verify   # only verify an existing download
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
# SHA-256 of the ZIP as served by UCI when this repository was built (2026-10-14).
ZIP_SHA256 = "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"
ZIP_NAME = "household_power_consumption.zip"
TXT_NAME = "household_power_consumption.txt"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")


def download(raw_dir: Path = RAW_DIR, verify_only: bool = False) -> Path:
    """Fetch and unpack the dataset; return the path of the extracted text file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / ZIP_NAME
    txt_path = raw_dir / TXT_NAME

    if not zip_path.exists():
        if verify_only:
            raise FileNotFoundError(f"{zip_path} not found; run without --verify to download")
        print(f"Downloading {URL}")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(zip_path)
    verify_checksum(zip_path, ZIP_SHA256)
    print(f"Checksum OK  sha256={ZIP_SHA256}")

    if not txt_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(TXT_NAME, raw_dir)
    print(f"Ready: {txt_path.relative_to(raw_dir.parent.parent)}")
    return txt_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="only verify an existing download")
    args = ap.parse_args(argv)
    try:
        download(verify_only=args.verify)
    except (ChecksumError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
