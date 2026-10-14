"""Run the multi-method benchmark on one dataset and write results plus a dated run log.

Usage (from the repository root)::

    python scripts/download_data.py && python scripts/run_benchmark.py --dataset uci
    python scripts/download_smd.py  && python scripts/run_benchmark.py --dataset smd
    python scripts/run_benchmark.py --dataset smd --smoke      # 3 machines, 1 seed, for a quick check

Outputs (``results/benchmark_<dataset>/``): ``matched.csv.gz``, ``budget.csv.gz``, ``tuning.csv``,
``chosen.json``, ``summary.json``; and ``docs/runs/<date>-<dataset>-benchmark.md``.

Pre-registered protocol (fixed before any test window was scored)
-----------------------------------------------------------------
* Windows: fit | calibration | test (see ``anomaly_monitoring.benchmark``). Hyper-parameters are
  chosen on calibration windows only, at the primary target below.
* Primary comparison: each method against the global z-score at matched calibration event recall,
  k=1, with and without the three-rule layer. Primary recall target: 0.80 for UCI (unchanged from
  the first run) and 0.65 for SMD (chosen from calibration-window diagnostics only: at 0.80 several
  methods could no longer reach the target on some machines, at 0.65 all but one reach it on at
  least 86% of machines).
* Secondary, reported whatever the outcome: other targets, alert-budget curves, time to detect,
  persistence k, rule-layer ablation.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from anomaly_monitoring import benchmark as B  # noqa: E402
from anomaly_monitoring import opsreport as R  # noqa: E402

PRIMARY = {"uci": 0.80, "smd": 0.65}
SEEDS = {"uci": [0, 1, 2, 3, 4], "smd": [0, 1, 2, 3, 4]}
FOCUS_BUDGETS = (0.25, 1.0, 4.0)


def table_vs_baseline(df, target, rules, k=1):
    t = R.matched_vs_baseline(df, target, k=k, rules=rules)
    t = t[t["n_units"] > 0].copy()
    t["change"] = [R.pct(-r) for r in t["pooled_reduction"]]
    t["change_ci"] = [f"[{-hi * 100:+.0f}%, {-lo * 100:+.0f}%]" for lo, hi in zip(t["pooled_lo"], t["pooled_hi"])]
    t["fa"] = [f"{a:.2f} vs {b:.2f}" for a, b in zip(t["fa_method"], t["fa_base"])]
    t["recall"] = [f"{a:.2f} vs {b:.2f}" for a, b in zip(t["recall_method"], t["recall_base"])]
    t["ttd"] = [f"{a:.0f} vs {b:.0f}" for a, b in zip(t["ttd_method"], t["ttd_base"])]
    t["better"] = [f"{a} / {b}" for a, b in zip(t["series_better"], t["series_worse"])]
    return R.md_table(t, [("method", "Method", "label"), ("n_units", "Matched units", "d"),
                          ("change", "Change in FA/day", "s"), ("change_ci", "95% CI", "s"),
                          ("fa", "FA/day (method vs z-score)", "s"), ("recall", "Test event recall", "s"),
                          ("ttd", "Median time to detect, min", "s"), ("better", "Series better / worse", "s")]), t


def table_operating(df, target, rules, k=1):
    t = R.operating_point_table(df, target, k=k, rules=rules)
    t["fa"] = [R.fmt_ci(a, b, c) for a, b, c in zip(t["fa_per_day"], t["fa_lo"], t["fa_hi"])]
    t["rc"] = [R.fmt_ci(a, b, c) for a, b, c in zip(t["recall"], t["recall_lo"], t["recall_hi"])]
    t["pr"] = [R.fmt_ci(a, b, c) for a, b, c in zip(t["precision"], t["precision_lo"], t["precision_hi"])]
    return R.md_table(t, [("method", "Method", "label"), ("reached_share", "Target reached (calibration)", ".2f"),
                          ("fa", "Test FA/day [95% CI]", "s"), ("rc", "Test event recall [95% CI]", "s"),
                          ("pr", "Event precision [95% CI]", "s"), ("ttd_median_min", "Median TTD, min", ".0f")]), t


def build_log(ds, args, runtime_s, date, units, chosen, tune_summary, rule_summary, mdf, bdf, extra_notes):
    prim = PRIMARY[ds]
    m = R.prepare(mdf)
    b = R.prepare(bdf)
    L: list[str] = []
    a = L.append
    seeds = ", ".join(map(str, args.seeds))
    a(f"# Benchmark run {date}: {ds.upper()}")
    a("")
    a(f"Status label: `Verified (run {date}, seeds {seeds})`. Produced by `scripts/run_benchmark.py --dataset {ds}`"
      f"{' --smoke' if args.smoke else ''}. These are new numbers from an independent re-implementation; they are not "
      "the result reported for the earlier deployment on the CV.")
    a("")
    a("## Settings")
    a("")
    a(f"* Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}")
    a(f"* Hardware: {platform.processor() or 'unknown CPU'}, {os.cpu_count()} logical cores, {platform.system()}; "
      f"{args.jobs} worker processes; runtime {runtime_s / 60:.1f} minutes")
    a(f"* Units: {len(units)} ({sum(u.matched_ok for u in units)} usable for matched recall: at least "
      f"{B.MIN_CAL_EVENTS} calibration events and one test event; {sum(u.budget_ok for u in units)} usable for alert budgets)")
    a(f"* Model seeds: {seeds} (deterministic methods are run once)")
    a(f"* Primary recall target {prim}; all targets {B.TARGETS}; persistence k in {B.KS}; alert budgets {B.BUDGETS} alerts/day")
    a("* Threshold rules: matched recall = highest threshold whose calibration event recall >= target; alert budget = lowest "
      "threshold that keeps all alerts on the calibration window within the budget (no labels).")
    a("* FA = false alert = an alert overlapping no labelled event. Event recall = share of labelled events with at least one "
      "flagged point. Time to detect (TTD) = minutes from event start to first flagged point, median over detected events.")
    a("* Percent changes are the pooled change in test false alerts per day relative to the global z-score at the same "
      "calibration recall (negative = fewer false alerts); CIs are 95% percentile bootstrap intervals over series "
      "(UCI: 12 series, each averaged over its injection seeds; SMD: machines).")
    for n in extra_notes:
        a(f"* {n}")
    a("")
    a("## Development notes (full disclosure)")
    a("")
    a("* A first complete run of this benchmark exposed a bug in the one-class SVM score: it was `-decision_function`, which is "
      "negative for normal points, so missing values (which score 0 by contract) ranked above normal points and low thresholds "
      "flagged every missing burst for free. The score is now shifted to be non-negative, a test checks that every detector's "
      "scores are non-negative, and both benchmarks were re-run. This log is from the re-run; the first run was discarded.")
    a("* Thresholds, hyper-parameters, rule settings and the primary comparisons were fixed before the test windows of the final "
      "run were read; the SMD primary recall target (0.65) was chosen from calibration-window diagnostics only.")
    a("* The alert budget counts alert segments per day, so a threshold low enough that a detector stays flagged for a very long "
      "stretch counts as one alert. The column `flagged_share` in `budget.csv.gz` records the share of test points flagged.")
    a("")
    a("## Hyper-parameters chosen on calibration windows")
    a("")
    t = tune_summary.copy()
    t["method"] = [R.label(x) for x in t["method"]]
    a(R.md_table(t, [("method", "Method", "s"), ("params", "Parameters", "s"), ("n_units", "Units", "d"),
                     ("share_unreached", "Share not reaching target", ".2f"),
                     ("mean_cal_fa_per_day", "Calibration FA/day", ".2f"), ("chosen", "Chosen", "s")]))
    a("")
    a("Rule-layer settings (z-score + all rules, same criterion):")
    a("")
    a(R.md_table(rule_summary, [("params", "Rule settings", "s"), ("share_unreached", "Share not reaching target", ".2f"),
                                ("mean_cal_fa_per_day", "Calibration FA/day", ".2f"), ("chosen", "Chosen", "s")]))
    a("")
    for rules, title in (("none", "Detector alone"), ("all", "Detector + rule layer (missing, range, stuck)")):
        a(f"## Primary comparison: {title}, k=1, calibration recall target {prim}")
        a("")
        md, _ = table_vs_baseline(m, prim, rules)
        a(md)
        a("")
        a("Operating points on the test window (all methods, including the baseline):")
        a("")
        a(table_operating(m, prim, rules)[0])
        a("")
    a("## Other recall targets, detector alone, k=1")
    a("")
    for tg in B.TARGETS:
        if tg == prim:
            continue
        a(f"Target {tg}:")
        a("")
        a(table_vs_baseline(m, tg, "none")[0])
        a("")
    a("## Alert-fatigue curves (alert budget, labels not used for thresholds)")
    a("")
    a("Test-window event recall (mean over series, 95% CI) when the threshold keeps the calibration-window alert rate within "
      "the budget, detector alone, k=1.")
    a("")
    bc = R.budget_curve(b, k=1, rules="none")
    for rules_name, bcx in (("none", bc), ("all", R.budget_curve(b, k=1, rules="all"))):
        a(f"Rule layer: {rules_name}")
        a("")
        piv = []
        for meth in B.METHODS:
            d = bcx[bcx["method"] == meth]
            if d.empty:
                continue
            row = {"Method": R.label(meth)}
            for bud in B.BUDGETS:
                x = d[d["budget"] == bud]
                row[f"{bud:g}/day"] = R.fmt_ci(float(x["recall"].iloc[0]), float(x["recall_lo"].iloc[0]),
                                                float(x["recall_hi"].iloc[0])) if len(x) else "n/a"
            piv.append(row)
        a(_md_frame(pd.DataFrame(piv)))
        a("")
    a("False alerts per day, precision and median time to detect at three budgets (detector alone, k=1):")
    a("")
    rows = []
    for meth in B.METHODS:
        row = {"Method": R.label(meth)}
        for bud in FOCUS_BUDGETS:
            x = bc[(bc["method"] == meth) & (bc["budget"] == bud)]
            if len(x):
                r0 = x.iloc[0]
                row[f"{bud:g}/day: FA/day, precision, TTD min"] = f"{r0['fa_per_day']:.2f}, {r0['precision']:.2f}, {r0['ttd_median_min']:.0f}"
        rows.append(row)
    a(_md_frame(pd.DataFrame(rows)))
    a("")
    a("## Persistence rule (k consecutive flagged points), threshold re-calibrated per k")
    a("")
    pt = R.persistence_table(m, prim, "none")
    pt["method"] = [R.label(x) for x in pt["method"]]
    a(R.md_table(pt, [("method", "Method", "s"), ("k", "k", "d"), ("reached_share", "Target reached", ".2f"),
                      ("fa_per_day", "Test FA/day", ".2f"), ("recall", "Test event recall", ".2f"),
                      ("ttd_median_min", "Median TTD, min", ".0f")]))
    a("")
    a(f"## Rule-layer ablation (matched recall {prim}, k=1)")
    a("")
    a("Test false alerts per day and event recall of each detector with each subset of the three rules. "
      "`all-X` removes rule X from the full layer.")
    a("")
    ab = R.ablation_table(m, prim, list(B.RULE_SETS))
    ab["method"] = [R.label(x) for x in ab["method"]]
    a(R.md_table(ab, [("method", "Method", "s"), ("rules", "Rules", "s"), ("reached_share", "Target reached", ".2f"),
                      ("fa_per_day", "Test FA/day", ".2f"), ("recall", "Test event recall", ".2f"),
                      ("n_alerts", "Alerts", ".1f")]))
    a("")
    best = R.best_by_calibration(m, prim, 1, "none")
    a("## Best method chosen on calibration windows only")
    a("")
    a("Candidates must reach the target on at least 90% of units; ranking by mean calibration false alerts per day "
      "(detector alone, k=1):")
    a("")
    a(", ".join(f"{R.label(n)} ({v:.2f})" for v, n in best["ranking"]))
    a("")
    return "\n".join(L), best


def _md_frame(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=["uci", "smd"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=B.default_jobs())
    ap.add_argument("--smoke", action="store_true", help="few units, one seed (quick check; does not write the run log)")
    ap.add_argument("--calibration-only", action="store_true", help="tune on calibration windows and stop")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    ds = args.dataset
    args.seeds = args.seeds or ([0] if args.smoke else SEEDS[ds])

    t0 = time.time()
    notes: list[str] = []
    if ds == "uci":
        from scripts.download_data import download

        download()
        units = B.uci_units(args.seeds[:2] if args.smoke else args.seeds)
        if args.smoke:
            units = units[:4]
        notes.append("Data: UCI Individual Household Electric Power Consumption (CC BY 4.0); 12 gap-free 120-day hourly "
                     "windows; injected anomalies (spike, level shift, drift, stuck, missing); window halves are 60 days; "
                     "fit window = calibration half incl. warm-up (as in the first run).")
        eval_seeds = [0]
    else:
        from anomaly_monitoring import smd

        smd.download(machines=smd.MACHINES[:3] if args.smoke else smd.MACHINES)
        units = B.smd_units(smd.MACHINES[:3] if args.smoke else None)
        dropped = sum(u.info["channels_total"] - u.info["channels_used"] for u in units)
        notes.append("Data: Server Machine Dataset (MIT), commit " + smd.COMMIT[:10] + "; real server metrics with expert labels; "
                     "5-minute means (labels: a bin is anomalous if any minute is); fit window = first (unlabelled) part, "
                     "calibration = first half of the labelled part, test = second half; "
                     f"{dropped} constant-in-training channels dropped over {len(units)} machines; "
                     "univariate methods are run per channel and alert on the largest normalised channel score.")
        eval_seeds = args.seeds

    chosen, tune_summary = B.tune(units, target=PRIMARY[ds], jobs=args.jobs)
    rule_params, rule_summary = B.tune_rules(units, PRIMARY[ds], jobs=args.jobs)
    out = ROOT / "results" / f"benchmark_{ds}{'_smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    tune_summary.to_csv(out / "tuning.csv", index=False)
    rule_summary.to_csv(out / "tuning_rules.csv", index=False)
    (out / "chosen.json").write_text(json.dumps({"methods": chosen, "rules": rule_params}, indent=1), encoding="utf-8")
    print("tuned:", chosen, rule_params)
    if args.calibration_only:
        print(tune_summary.to_string())
        return 0

    mdf, bdf = B.evaluate(units, eval_seeds, chosen, PRIMARY[ds], jobs=args.jobs, rule_params=rule_params)
    runtime = time.time() - t0
    mdf.round(6).to_csv(out / "matched.csv.gz", index=False)
    bdf.round(6).to_csv(out / "budget.csv.gz", index=False)
    date = dt.date.today().isoformat()
    log, best = build_log(ds, args, runtime, date, units, chosen, tune_summary, rule_summary, mdf, bdf, notes)
    summary = {"date": date, "dataset": ds, "seeds": args.seeds, "n_units": len(units), "primary_target": PRIMARY[ds],
               "runtime_seconds": round(runtime), "chosen": chosen, "rule_params": rule_params, "calibration_ranking": best["ranking"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    if not args.smoke and not args.no_log:
        runs = ROOT / "docs" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{date}-{ds}-benchmark.md").write_text(log, encoding="utf-8")
    print(f"done in {runtime / 60:.1f} min; wrote {out.relative_to(ROOT)}")
    print(log[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
