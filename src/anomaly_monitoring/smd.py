"""Server Machine Dataset (SMD): real server metrics with human-labelled anomaly intervals.

Source: Su et al. (2019), "Robust Anomaly Detection for Multivariate Time Series through
Stochastic Recurrent Neural Network", KDD. Data: https://github.com/NetManAIOps/OmniAnomaly
(``ServerMachineDataset``, MIT licence, see ``THIRD_PARTY_NOTICES.md``).

28 machines, 38 metrics each, one reading per minute for five weeks of training data
(unlabelled, assumed normal) followed by five weeks of test data with expert labels.

Files are downloaded at run time from a pinned commit and verified against SHA-256 checksums
(``smd_manifest.json``); raw files are never committed.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
BASE_URL = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{COMMIT}/ServerMachineDataset"
MACHINES: tuple[str, ...] = tuple(
    [f"machine-1-{i}" for i in range(1, 9)]
    + [f"machine-2-{i}" for i in range(1, 10)]
    + [f"machine-3-{i}" for i in range(1, 12)]
)
SPLITS = ("train", "test", "test_label")
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "smd"
MANIFEST = Path(__file__).with_name("smd_manifest.json")
MINUTES_PER_POINT = 5  # series are averaged to 5-minute bins for the benchmark
POINTS_PER_DAY = 24 * 60 // MINUTES_PER_POINT


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_keys(machines: Iterable[str] = MACHINES) -> list[str]:
    return [f"{s}/{m}.txt" for m in machines for s in SPLITS]


def load_manifest() -> dict[str, str]:
    data: dict[str, str] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data


def _fetch_one(key: str, raw_dir: Path, expected: str | None) -> tuple[str, str]:
    dest = raw_dir / key
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{key}", tmp)
        tmp.replace(dest)
    digest = sha256_of(dest)
    if expected is not None and digest != expected:
        raise ChecksumError(f"checksum mismatch for {key}: expected {expected}, got {digest}")
    return key, digest


def download(raw_dir: Path = RAW_DIR, machines: Iterable[str] = MACHINES, workers: int = 6,
             write_manifest: bool = False) -> Path:
    """Fetch every SMD file for ``machines`` and verify its checksum; returns ``raw_dir``."""
    machines = list(machines)
    keys = file_keys(machines)
    manifest = {} if write_manifest else load_manifest()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda k: _fetch_one(k, raw_dir, manifest.get(k)), keys))
    if write_manifest:
        MANIFEST.write_text(json.dumps(dict(sorted(results)), indent=1) + "\n", encoding="utf-8")
    return raw_dir


def downsample_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive blocks of ``factor`` rows (a trailing partial block is dropped)."""
    n = (a.shape[0] // factor) * factor
    return a[:n].reshape(n // factor, factor, *a.shape[1:]).mean(axis=1)


def downsample_any(y: np.ndarray, factor: int) -> np.ndarray:
    """A coarse bin is anomalous if any of its minutes is."""
    n = (y.shape[0] // factor) * factor
    return np.asarray(y[:n].reshape(n // factor, factor).any(axis=1), dtype=bool)


@dataclass(frozen=True)
class Machine:
    name: str
    train: np.ndarray        # (T_train, 38), 5-minute means, unlabelled, assumed normal
    test: np.ndarray         # (T_test, 38)
    labels: np.ndarray       # (T_test,) bool, expert labels

    @property
    def n_channels(self) -> int:
        return int(self.train.shape[1])


def load_machine(name: str, raw_dir: Path = RAW_DIR, factor: int = MINUTES_PER_POINT) -> Machine:
    train = np.loadtxt(raw_dir / "train" / f"{name}.txt", delimiter=",", dtype=np.float64)
    test = np.loadtxt(raw_dir / "test" / f"{name}.txt", delimiter=",", dtype=np.float64)
    lab = np.loadtxt(raw_dir / "test_label" / f"{name}.txt", dtype=np.int8).astype(bool)
    if lab.shape[0] != test.shape[0]:
        raise ValueError(f"{name}: {lab.shape[0]} labels for {test.shape[0]} test rows")
    return Machine(name, downsample_mean(train, factor), downsample_mean(test, factor),
                   downsample_any(lab, factor))
