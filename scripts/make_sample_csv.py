"""Write a sample CSV for ``monitor stream``: one UCI voltage window with injected incidents.

Usage::

    python scripts/make_sample_csv.py            # writes data/processed/sample_voltage.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from anomaly_monitoring.data import POINTS_PER_DAY, load_hourly, make_units  # noqa: E402
from anomaly_monitoring.inject import inject_anomalies  # noqa: E402
from scripts.download_data import download  # noqa: E402


def main() -> int:
    hourly = load_hourly(download())
    unit = {u["name"]: u for u in make_units(hourly)}["voltage-w0"]
    res = inject_anomalies(unit["values"], seed=0, points_per_day=POINTS_PER_DAY)
    idx = pd.date_range(unit["start"], periods=res.values.size, freq="h")
    out = ROOT / "data" / "processed" / "sample_voltage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": idx, "value": res.values}).to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)} ({res.values.size} hourly points, {len(res.events)} injected incidents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
