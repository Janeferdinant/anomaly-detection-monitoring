"""Run the full matched-recall evaluation and write results + a dated run log.

Usage (from the repository root, after ``python scripts/download_data.py``)::

    python scripts/run_evaluation.py                    # seeds 0-4, writes results/ and docs/runs/
    python scripts/run_evaluation.py --seeds 0 1        # quick run
    python scripts/run_evaluation.py --no-log           # do not write the markdown run log

Outputs
-------
* ``results/results.csv``  one row per (series, seed, detector, persistence k, rule layer, target)
* ``results/summary.json`` the headline comparison in machine-readable form
* ``docs/runs/<date>-evaluation.md`` the run log with every table and the exact settings

Pre-registered protocol (fixed before the final seeds were run): the primary comparison is
"detector + data-quality rule layer, k=1, calibration recall target 0.80" against the global
z-score + rule layer. Everything else is secondary and is reported regardless of outcome.
Method development used separate seeds (1000-1002); seeds 0-4 were not used for tuning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from anomaly_monitoring import report as R  # noqa: E402
from anomaly_monitoring.data import CHANNELS, load_hourly, make_units  # noqa: E402
from anomaly_monitoring.evaluate import EvalConfig, run_all  # noqa: E402
from scripts.download_data import ZIP_SHA256, download  # noqa: E402

PRIMARY_TARGET = 0.8
TARGETS = (0.5, 0.65, 0.8, 0.9)


def build_log(df, args, runtime_s, date, n_units) -> tuple[str, dict]:
    seeds = ", ".join(map(str, args.seeds))
    prim = R.reduction_table(df, dq=True, targets=[PRIMARY_TARGET])
    prim_auto = R.auto_table(df, [PRIMARY_TARGET])
    summary = {
        "date": date, "seeds": args.seeds, "n_series": n_units, "runtime_seconds": round(runtime_s),
        "primary": {"protocol": f"detector + rule layer, k=1, calibration recall target {PRIMARY_TARGET}, vs global z-score + rule layer",
                    "detectors": prim, "calibration_selected": prim_auto},
    }
    L = []
    a = L.append
    a(f"# Evaluation run {date}")
    a("")
    a(f"Status label: `Verified (run {date}, seeds {seeds})`. Numbers below were produced by "
      "`scripts/run_evaluation.py` on public data with injected, labelled anomalies. They are new "
      "numbers from an independent re-implementation. They are **not** the result reported for the "
      "earlier deployment on the CV, and they do not reproduce or confirm it.")
    a("")
    a("## Development notes (full disclosure)")
    a("")
    a("* Method development (detector parameters, calibration protocol, number of injected incidents per window) used development seeds 1000-1002. Seeds 0-4 were not used for tuning.")
    a("* The protocol and the primary comparison were fixed before the final runs; every table below is reported whatever its outcome.")
    a("* A first complete run on seeds 0-4 showed that stuck-value incidents were rarely caught by the rule layer: the stuck-run length was fitted as a high quantile of run lengths, which the injected stuck incidents in the calibration window inflated. The rule layer is shared by all detectors. It was changed to the median run length plus 2 (floor 4, needs at least 10 runs) and the whole evaluation was re-run. This log is from the re-run; the first run was discarded.")
    a("")
    a("## Settings")
    a("")
    a(f"* Command: `python scripts/run_evaluation.py --seeds {' '.join(map(str, args.seeds))}`")
    a(f"* Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}")
    a(f"* Runtime: {runtime_s / 60:.1f} minutes on CPU ({platform.processor() or 'unknown CPU'}, {platform.system()})")
    a(f"* Data: UCI Individual Household Electric Power Consumption (CC BY 4.0), zip sha256 `{ZIP_SHA256}`")
    a(f"* Series: {n_units} windows of 120 days of hourly data ({len(CHANNELS)} channels x 3 gap-free windows: "
      f"{', '.join(CHANNELS)}). Each window is split 60/60 days into a calibration window and a test window.")
    a(f"* Seeds (injection and model seeds): {seeds}; units evaluated: {len(df[['series','seed']].drop_duplicates())} (series x seed)")
    a("* Injected incidents per window half: 3 each of spike, level shift, drift, stuck value and missing burst (15 per half).")
    a("* Alert-suppression rule: persistence of k consecutive flagged points, k in 1, 2, 3.")
    a("* Event recall = share of injected incidents with at least one alert inside them. False alert = an alert segment that overlaps no incident. FA/day = false alerts per day on the test window.")
    a("* Thresholds are chosen on the calibration window only (highest threshold whose calibration event recall >= target); the test window is never used for choosing them.")
    a("* Matched-recall comparisons use only the (series, seed) units where both the method and the baseline reached the target recall on the calibration window; the `Matched units` column shows how many that was.")
    a("* Percentage columns are the change in false alerts per day (negative = fewer false alerts than the global z-score). CIs are 95% percentile bootstrap intervals over series (10,000 resamples, seed 0). The per-series mean is unstable when the baseline has almost no false alerts; the pooled change is the more reliable column.")
    a("")
    a(f"## Primary comparison: detector + rule layer, k=1, calibration recall target {PRIMARY_TARGET:.2f}")
    a("")
    a(R.md_reduction(prim + prim_auto))
    a("")
    a("## Detector alone (no rule layer), k=1")
    a("")
    a(R.md_reduction(R.reduction_table(df, dq=False, targets=TARGETS)))
    a("")
    a("## Detector + rule layer, k=1, all recall targets")
    a("")
    a(R.md_reduction(R.reduction_table(df, dq=True, targets=TARGETS)))
    a("")
    a("## Calibration-selected pipeline (detector and k chosen on the calibration window), rule layer on")
    a("")
    auto_all = R.auto_table(df, TARGETS)
    a(R.md_reduction(auto_all))
    a("")
    for rec in auto_all:
        a(f"* Target {rec['target']:.2f}, configurations chosen: " + ", ".join(f"{k} x{v}" for k, v in rec.get("chosen", {}).items()))
    a("")
    a("## Effect of the alert-suppression (persistence) rule at a fixed threshold")
    a("")
    a(f"The threshold is the one calibrated for k=1 (target {PRIMARY_TARGET:.2f}); only k changes, so this shows the raw trade-off. Detector alone, all units.")
    a("")
    a(R.md_persistence(R.persistence_table(df, dq=False, target=PRIMARY_TARGET)))
    a("")
    a("## Persistence at matched recall (threshold re-calibrated for each k)")
    a("")
    a("Detector alone, target 0.65. Share of units where the target was reachable and mean false alerts per day on those units.")
    a("")
    a("| Detector | k | Target reached on calibration | Test FA/day (reached units) | Test event recall (reached units) |")
    a("|---|---|---|---|---|")
    d = df[(~df["dq"]) & (df["target"] == 0.65)]
    for det in d["detector"].unique():
        for k in sorted(d["k"].unique()):
            g = d[(d["detector"] == det) & (d["k"] == k)]
            ok = g[g["reached"]]
            a(f"| {R.label(det)} | {k} | {g['reached'].mean():.2f} | "
              f"{ok['test_false_alerts_per_day'].mean() if len(ok) else float('nan'):.2f} | "
              f"{ok['test_event_recall'].mean() if len(ok) else float('nan'):.2f} |")
    a("")
    a("## Ranking-quality check at matched test recall (threshold chosen on the test window; not deployable)")
    a("")
    a("Uses test labels to pick the threshold, so it measures how well each score separates incidents from normal points, independent of threshold transfer. Only units where every detector could reach recall r are compared.")
    a("")
    a("Detector alone:")
    a("")
    a(R.md_oracle(R.oracle_table(df, dq=False)))
    a("")
    a("Detector + rule layer:")
    a("")
    a(R.md_oracle(R.oracle_table(df, dq=True)))
    a("")
    a(f"## Which incident types are caught (test window, k=1, target {PRIMARY_TARGET:.2f})")
    a("")
    a("Detector alone:")
    a("")
    a(R.md_kinds(R.kind_table(df, dq=False, target=PRIMARY_TARGET)))
    a("")
    a("Detector + rule layer:")
    a("")
    a(R.md_kinds(R.kind_table(df, dq=True, target=PRIMARY_TARGET)))
    a("")
    a(f"## Point-adjusted vs event-level vs raw point scoring (same alerts, detector + rule layer, k=1, target {PRIMARY_TARGET:.2f})")
    a("")
    a(R.md_scoring(R.scoring_table(df, dq=True, target=PRIMARY_TARGET)))
    a("")
    a("Point-adjusted (PA) scoring credits a whole incident when one point is flagged, so PA recall is always at least the raw point recall and PA precision at least the raw point precision.")
    a("")
    return "\n".join(L), summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--windows", type=int, default=3, help="gap-free 120-day windows per channel")
    ap.add_argument("--no-log", action="store_true", help="do not write docs/runs/<date>-evaluation.md")
    args = ap.parse_args(argv)

    txt = download()
    t0 = time.time()
    hourly = load_hourly(txt)
    units = make_units(hourly, n_windows=args.windows)
    print(f"{len(units)} series x {len(args.seeds)} seeds")
    cfg = EvalConfig(targets=TARGETS)
    df = run_all(units, args.seeds, cfg)
    runtime = time.time() - t0

    date = dt.date.today().isoformat()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.round(6).to_csv(out / "results.csv", index=False)
    log, summary = build_log(df, args, runtime, date, len(units))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    if not args.no_log:
        runs = ROOT / "docs" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{date}-evaluation.md").write_text(log, encoding="utf-8")
    print(f"done in {runtime / 60:.1f} min; wrote results/results.csv, results/summary.json")
    print()
    print(log.split("## Detector alone")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
