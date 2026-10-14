"""Build the static demo ``docs/index.html`` from precomputed detector scores.

The page needs no server and fetches nothing at load time: this script embeds a small JSON
payload (hourly values of the 60-day test window, injected incident labels, per-point scores of
each detector, rule-layer flags, and threshold grids) into ``scripts/demo_template.html``.
The browser then applies the threshold and persistence sliders to those precomputed scores.

Usage (needs the dataset, see scripts/download_data.py)::

    python scripts/build_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from anomaly_monitoring.data import CHANNELS, load_hourly, make_units  # noqa: E402
from anomaly_monitoring.detectors import DETECTOR_LABELS, make_detector  # noqa: E402
from anomaly_monitoring.inject import inject_anomalies  # noqa: E402
from anomaly_monitoring.rules import DataQualityRules  # noqa: E402
from scripts.download_data import download  # noqa: E402

SEED = 0
WARMUP_DAYS = 14
LEVELS = 60  # number of threshold steps on the slider
SAMPLES = [  # series ids (channel-window), chosen to cover the four channels
    "voltage-w0",
    "global_active_power-w0",
    "global_intensity-w1",
    "global_reactive_power-w2",
    "voltage-w1",
    "global_active_power-w2",
]
DETECTORS = ["zscore", "seasonal", "rolling_mad", "ewma", "iforest", "combo"]
NICE = {
    "global_active_power": "Global active power",
    "global_reactive_power": "Global reactive power",
    "global_intensity": "Global current intensity",
    "voltage": "Voltage",
}


def quantile_grid() -> list[float]:
    """Quantile levels from 0.90 to 0.9999, evenly spaced in log(1 - q)."""
    expo = 1 + 3 * np.arange(LEVELS) / (LEVELS - 1)
    return [float(1 - 10 ** -e) for e in expo]


def main() -> int:
    txt = download()
    units = {u["name"]: u for u in make_units(load_hourly(txt))}
    qs = quantile_grid()
    out = {"levels": [round(q, 6) for q in qs], "detectors": {d: DETECTOR_LABELS[d] for d in DETECTORS}, "series": []}
    ppd = 24
    for sid in SAMPLES:
        u = units[sid]
        res = inject_anomalies(u["values"], seed=SEED, points_per_day=ppd, warmup_days=WARMUP_DAYS)
        x, y = res.values, res.labels
        split = x.size // 2
        warm = WARMUP_DAYS * ppd
        rules = DataQualityRules().fit(x[:split])
        dq = rules.flags(x)
        entry = {
            "id": sid,
            "title": f"{NICE[u['channel']]} ({u['unit']})",
            "channel": u["channel"],
            "start": (np.datetime64(u["start"]) + np.timedelta64(split // ppd, "D")).astype(str),
            "x": [None if np.isnan(v) else round(float(v), 3) for v in x[split:]],
            "events": [[e.start - split, e.end - split, e.kind] for e in res.events if e.start >= split],
            "dq": "".join("1" if v else "0" for v in dq[split:]),
            "scores": {}, "grid": {},
        }
        for d in DETECTORS:
            det = make_detector(d, seed=SEED, points_per_day=ppd).fit(x[:split])
            s = det.score(x)
            entry["scores"][d] = [round(float(v), 3) for v in s[split:]]
            entry["grid"][d] = [round(float(np.quantile(s[warm:split], q)), 4) for q in qs]
        out["series"].append(entry)

    payload = json.dumps(out, separators=(",", ":"))
    html = (ROOT / "scripts" / "demo_template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/" in html
    html = html.replace("/*__DATA__*/null", payload)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote docs/index.html ({len(html) / 1024:.0f} KiB, {len(out['series'])} series)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
