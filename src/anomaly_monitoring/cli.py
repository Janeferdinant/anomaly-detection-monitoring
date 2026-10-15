"""Command line interface: ``monitor stream <csv>`` replays a CSV through the online monitor."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from . import __version__
from .alerting import threshold_for_budget
from .detectors import make_detector
from .rules import DataQualityRules
from .streaming import STREAMING_DETECTORS, OnlineMonitor


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="monitor", description="Anomaly monitoring toolkit.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    st = sub.add_parser(
        "stream", help="replay a CSV through the online detector, one point at a time",
        description="Fit a detector on the first --fit-points values, then process every row one "
                    "at a time with bounded state and print an ALERT line when an alert starts. "
                    "Without --threshold, the threshold is the lowest one that keeps the alert rate "
                    "on the fit window within --budget alerts per day (no labels needed).")
    st.add_argument("csv", help="CSV file with a header row")
    st.add_argument("--column", default="value", help="column holding the metric (default: value)")
    st.add_argument("--time-column", default=None, help="optional timestamp column used in alert lines")
    st.add_argument("--points-per-day", type=int, default=24, help="samples per day (default: 24, hourly)")
    st.add_argument("--detector", choices=STREAMING_DETECTORS, default="seasonal")
    st.add_argument("--fit-points", type=int, default=None,
                    help="number of leading points used to fit (default: 30 days)")
    st.add_argument("--budget", type=float, default=1.0, help="alerts per day allowed on the fit window")
    st.add_argument("--threshold", type=float, default=None, help="fixed score threshold (overrides --budget)")
    st.add_argument("--persistence", type=int, default=1, help="flag only after k consecutive points (default: 1)")
    st.add_argument("--rules", action="store_true", help="add the missing / range / stuck rule layer")
    st.add_argument("--quiet", action="store_true", help="print only the summary")
    return ap


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def run_stream(args: argparse.Namespace) -> int:
    try:
        df = pd.read_csv(args.csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return _fail(f"cannot read {args.csv}: {e}")
    if args.column not in df.columns:
        return _fail(f"column {args.column!r} not found; columns are {list(df.columns)}")
    if args.time_column is not None and args.time_column not in df.columns:
        return _fail(f"time column {args.time_column!r} not found; columns are {list(df.columns)}")
    x = pd.to_numeric(df[args.column], errors="coerce").to_numpy(dtype=float)
    ppd = args.points_per_day
    fit_n = args.fit_points if args.fit_points is not None else 30 * ppd
    if not 2 * ppd <= fit_n < x.size:
        return _fail(f"--fit-points must be between {2 * ppd} (two days) and {x.size - 1}, got {fit_n}")
    labels = df[args.time_column].astype(str).to_numpy() if args.time_column else None

    fit = x[:fit_n]
    det = make_detector(args.detector, points_per_day=ppd).fit(fit)
    rules = DataQualityRules().fit(fit) if args.rules else None
    if args.threshold is not None:
        thr = float(args.threshold)
    else:
        fit_flags = rules.flags(fit) if rules is not None else None
        warm = min(8 * ppd, fit_n // 2)
        res = threshold_for_budget(det.score(fit)[warm:], args.persistence, args.budget, ppd,
                                   extra_flags=None if fit_flags is None else fit_flags[warm:])
        thr = res["threshold"]
        if not res["met"]:
            print("warning: the rule layer alone exceeds the alert budget on the fit window", file=sys.stderr)
    mon = OnlineMonitor.from_detector(det, thr, k=args.persistence, rules=rules)

    n_alerts = 0
    for i, v in enumerate(x):
        step = mon.update(v)
        if i >= fit_n and step.alert_started:
            n_alerts += 1
            if not args.quiet:
                when = labels[i] if labels is not None else f"t={i}"
                why = f"  rule:{'+'.join(step.rules)}" if step.rules else ""
                print(f"{when}  ALERT  score={step.score:.2f}  value={step.value:.4g}{why}")
    days = (x.size - fit_n) / ppd
    print(f"processed {x.size} points (fit window {fit_n}); {args.detector}, threshold {thr:.4g}, "
          f"persistence {args.persistence}, rules {'on' if rules else 'off'}")
    print(f"alerts after the fit window: {n_alerts} ({n_alerts / days:.2f} per day over {days:.1f} days)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stream":
        return run_stream(args)
    return 2  # pragma: no cover - argparse enforces a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
