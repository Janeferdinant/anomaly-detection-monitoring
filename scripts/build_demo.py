"""Build the static demo ``docs/index.html`` from precomputed detector scores.

The page needs no server and fetches nothing at load time: this script embeds a JSON payload
(values of the test window, labelled incidents, per-point scores of every method, rule-layer flags
and the alert-budget thresholds) into ``scripts/demo_template.html``. The browser then applies
the persistence slider and the data-quality switch to those precomputed scores and shows the
alerts. No benchmark metric is displayed.

Series: six UCI household-power windows with injected incidents and three Server Machine Dataset
machines with human-labelled anomalies. Methods use the parameters chosen on calibration windows
by ``scripts/run_benchmark.py`` (``results/benchmark_<dataset>/chosen.json``) when present.

Usage (needs both datasets, see scripts/download_data.py and scripts/download_smd.py)::

    python scripts/build_demo.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from anomaly_monitoring import benchmark as B  # noqa: E402
from anomaly_monitoring import smd  # noqa: E402
from anomaly_monitoring.alerting import threshold_for_budget  # noqa: E402
from anomaly_monitoring.data import POINTS_PER_DAY, load_hourly, make_units  # noqa: E402
from anomaly_monitoring.detectors import DETECTOR_LABELS  # noqa: E402
from anomaly_monitoring.inject import inject_anomalies  # noqa: E402
from anomaly_monitoring.metrics import segments  # noqa: E402
from scripts.download_data import download  # noqa: E402

SEED = 0
UCI_SAMPLES = [
    "voltage-w0",
    "global_active_power-w0",
    "global_intensity-w1",
    "global_reactive_power-w2",
    "voltage-w1",
    "global_active_power-w2",
]
SMD_SAMPLES = ["machine-1-3", "machine-2-4", "machine-3-3"]
NICE = {
    "global_active_power": "Global active power (kW)",
    "global_reactive_power": "Global reactive power (kVAr)",
    "global_intensity": "Global current intensity (A)",
    "voltage": "Voltage (V)",
}
KS = range(1, 7)


def sig3(a: np.ndarray) -> list[float]:
    return [float(f"{v:.3g}") for v in a]


def load_chosen(ds: str) -> tuple[dict, dict | None]:
    p = ROOT / "results" / f"benchmark_{ds}" / "chosen.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        return j["methods"], j["rules"]
    return {}, None


def entry_for(unit: B.Unit, chosen: dict, rule_params, title: str, source: str, step: str,
              show: np.ndarray, events) -> dict:
    t0, t1 = unit.test
    cal0, cal1 = unit.cal
    dq_all = B.fit_rules(unit, rule_params)["all"]
    out = {
        "title": title, "source": source, "step": step, "ppd": unit.ppd, "mpp": unit.minutes_per_point,
        "x": [None if np.isnan(v) else round(float(v), 3) for v in show[t0:t1]],
        "events": events,
        "dq": "".join("1" if v else "0" for v in dq_all[t0:t1]),
        "scores": {}, "thr": {},
    }
    for m in B.METHODS:
        s = B.fit_and_score(unit, m, B.method_params(m, chosen), SEED)
        out["scores"][m] = sig3(s[t0:t1])
        grid = []
        for use_dq in (False, True):
            extra = dq_all[cal0:cal1] if use_dq else None
            per_k = []
            for k in KS:
                row = []
                for bud in B.BUDGETS:
                    thr = threshold_for_budget(s[cal0:cal1], k, bud, unit.ppd, extra_flags=extra)["threshold"]
                    row.append(None if not np.isfinite(thr) else float(f"{thr:.4g}"))
                per_k.append(row)
            grid.append(per_k)
        out["thr"][m] = grid
        print(f"  {unit.name} {m}", flush=True)
    return out


def main() -> int:
    chosen_u, rules_u = load_chosen("uci")
    chosen_s, rules_s = load_chosen("smd")
    out = {"budgets": list(B.BUDGETS), "methods": {m: DETECTOR_LABELS[m] for m in B.METHODS}, "series": []}

    download()
    units = {u.name: u for u in B.uci_units([SEED])}
    raw_units = {v["name"]: v for v in make_units(load_hourly())}
    for sid in UCI_SAMPLES:
        u = units[sid]
        ch = u.info["channel"]
        t0 = u.test[0]
        # incident kinds come from the seeded generator
        raw = raw_units[sid]
        res = inject_anomalies(raw["values"], seed=SEED, points_per_day=POINTS_PER_DAY, warmup_days=14)
        events = [[e.start - t0, e.end - t0, e.kind] for e in res.events if e.start >= t0]
        out["series"].append(entry_for(
            u, chosen_u, rules_u, f"Household power: {NICE[ch]}, {sid.split('-')[-1]} (injected incidents)",
            "UCI household electricity, incidents injected by a seeded generator", "hourly", u.values, events))

    smd.download(machines=SMD_SAMPLES)
    for u in B.smd_units(SMD_SAMPLES):
        t0, t1 = u.test
        train = u.values[: u.fit_end]
        z = np.abs((u.values[t0:t1] - train.mean(0)) / np.maximum(train.std(0), 1e-6))
        lab = u.labels[t0:t1]
        ch = int(np.argmax(z[lab].mean(0))) if lab.any() else 0  # the metric that moves most inside labelled anomalies
        events = [[a, b, "labelled"] for a, b in segments(lab)]
        out["series"].append(entry_for(
            u, chosen_s, rules_s, f"Server {u.name.replace('machine-', '')}: one metric of {u.n_channels}, "
            "human-labelled anomalies", "Server Machine Dataset, anomalies labelled by operations experts",
            "five-minute", u.values[:, ch], events))

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
