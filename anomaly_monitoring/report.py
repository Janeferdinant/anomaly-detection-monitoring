"""Turn the raw results frame into the summary tables used in the README and the run log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import DETECTOR_LABELS
from .evaluate import (
    matched_pairs,
    per_series_mean,
    reduction_from_series,
    select_on_calibration,
)
from .inject import EVENT_KINDS

BASE = "zscore"


def _sel(df, detector, k=1, dq=False, target=None):
    m = (df["detector"] == detector) & (df["k"] == k) & (df["dq"] == dq)
    if target is not None:
        m &= df["target"] == target
    return df[m]


def label(d: str) -> str:
    return DETECTOR_LABELS.get(d, d if d != "auto" else "Calibration-selected pipeline")


def reduction_table(df: pd.DataFrame, dq: bool, targets, detectors=None, k: int = 1, boot_seed: int = 0):
    """Matched-calibration-recall comparison of every detector against the global z-score.

    Only (series, seed) units where both the method and the baseline reached the target
    recall on the calibration window are used (the units where recall was matched).
    """
    detectors = detectors or [d for d in df["detector"].unique() if d != BASE]
    recs = []
    for t in targets:
        base = _sel(df, BASE, 1, dq, t)
        n_all = len(base)
        for d in detectors:
            m, b = matched_pairs(_sel(df, d, k, dq, t), base)
            if len(m) == 0:
                recs.append({"target": t, "detector": d, "n_units": 0, "n_all": n_all})
                continue
            res = reduction_from_series(per_series_mean(m, "test_false_alerts_per_day"),
                                        per_series_mean(b, "test_false_alerts_per_day"), boot_seed)
            res.update(target=t, detector=d, n_units=len(m), n_all=n_all,
                       recall_method=float(m["test_event_recall"].mean()),
                       recall_base=float(b["test_event_recall"].mean()))
            recs.append(res)
    return recs


def auto_table(df: pd.DataFrame, targets, boot_seed: int = 0):
    """Calibration-selected pipeline (detector and persistence k chosen on the calibration
    window only, rule layer on) against the z-score + rule layer baseline."""
    recs = []
    for t in targets:
        sel = select_on_calibration(df[df["target"] == t], require_dq=True)
        base = _sel(df, BASE, 1, True, t)
        m, b = matched_pairs(sel, base)
        if len(m) == 0:
            continue
        res = reduction_from_series(per_series_mean(m, "test_false_alerts_per_day"),
                                    per_series_mean(b, "test_false_alerts_per_day"), boot_seed)
        res.update(target=t, detector="auto", n_units=len(m), n_all=len(base),
                   recall_method=float(m["test_event_recall"].mean()),
                   recall_base=float(b["test_event_recall"].mean()),
                   chosen=m["chosen"].value_counts().to_dict() if "chosen" in m else {})
        recs.append(res)
    return recs


def persistence_table(df: pd.DataFrame, dq: bool, target: float):
    """Effect of the persistence rule at a *fixed* threshold (the one calibrated for k=1)."""
    d = df[(df["dq"] == dq) & (df["target"] == target)]
    rows = []
    for det in d["detector"].unique():
        g = d[d["detector"] == det]
        k1 = g[g["k"] == 1]
        for k in sorted(g["k"].unique()):
            gk = g[g["k"] == k]
            rows.append({
                "detector": det, "k": int(k),
                "recall": float(gk["fixed_thr_event_recall"].mean()),
                "fa_per_day": float(gk["fixed_thr_false_alerts_per_day"].mean()),
                "fa_change_vs_k1": float(gk["fixed_thr_false_alerts_per_day"].mean()
                                         / k1["fixed_thr_false_alerts_per_day"].mean() - 1)
                if k1["fixed_thr_false_alerts_per_day"].mean() > 0 else float("nan"),
            })
    return rows


def oracle_table(df: pd.DataFrame, dq: bool, levels=(50, 70, 90), boot_seed: int = 0):
    """Ranking-quality check: threshold chosen on the TEST window to hit test recall r.

    Not deployable (uses test labels); it separates "does the score rank anomalies above
    normal points" from "does a calibration-window threshold transfer". Only units where
    every detector reached recall r are compared.
    """
    first_target = df["target"].min()
    d = df[(df["dq"] == dq) & (df["k"] == 1) & (df["target"] == first_target)]
    recs = []
    dets = [x for x in d["detector"].unique()]
    for r in levels:
        col = f"oracle_fa_r{r}"
        piv = d.pivot_table(index=["series", "seed"], columns="detector", values=col, aggfunc="first")
        n_all = len(piv)
        piv = piv.dropna()
        if len(piv) == 0:
            continue
        for det in dets:
            if det == BASE:
                continue
            fm = piv[det].groupby(level="series").mean()
            fb = piv[BASE].groupby(level="series").mean()
            res = reduction_from_series(fm, fb, boot_seed)
            res.update(level=r / 100, detector=det, n_units=len(piv), n_all=n_all)
            recs.append(res)
        recs.append({"level": r / 100, "detector": BASE, "n_units": len(piv), "n_all": n_all,
                     "fa_method": float(piv[BASE].mean()), "fa_base": float(piv[BASE].mean())})
    return recs


def kind_table(df: pd.DataFrame, dq: bool, target: float):
    d = df[(df["dq"] == dq) & (df["target"] == target) & (df["k"] == 1)]
    rows = []
    for det, g in d.groupby("detector"):
        row = {"detector": det}
        for kind in EVENT_KINDS:
            n = g[f"n_{kind}"].sum()
            row[kind] = float(g[f"det_{kind}"].sum() / n) if n else float("nan")
        rows.append(row)
    return rows


def scoring_table(df: pd.DataFrame, dq: bool, target: float):
    """Point-adjusted vs event-level vs raw point-level scoring of the same alerts."""
    d = df[(df["dq"] == dq) & (df["target"] == target) & (df["k"] == 1)]
    g = d.groupby("detector").agg(
        pa_precision=("test_pa_precision", "mean"), pa_recall=("test_pa_recall", "mean"),
        pa_f1=("test_pa_f1", "mean"),
        point_precision=("test_point_precision", "mean"), point_recall=("test_point_recall", "mean"),
        point_fpr=("test_point_fpr", "mean"),
        event_precision=("test_event_precision", "mean"), event_recall=("test_event_recall", "mean"),
        fa_per_day=("test_false_alerts_per_day", "mean"),
    )
    return g.reset_index().to_dict("records")


# ---- markdown helpers -------------------------------------------------------------------

def pct(x, digits=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{digits}f}%"


def ci(mean, lo, hi):
    if any(isinstance(v, float) and np.isnan(v) for v in (mean, lo, hi)):
        return "n/a"
    return f"{pct(mean)} [{pct(lo)}, {pct(hi)}]"


def md_reduction(recs) -> str:
    out = ["| Target recall | Detector | Matched units | Test FA/day (method vs z-score) | "
           "Mean per-series change in FA/day, 95% CI | Pooled change, 95% CI | Series better / worse | "
           "Test recall (method vs z-score) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if r["n_units"] == 0:
            out.append(f"| {r['target']:.2f} | {label(r['detector'])} | 0 / {r['n_all']} | n/a | n/a | n/a | n/a | n/a |")
            continue
        # a reduction is reported as a negative change in false alerts
        out.append(
            f"| {r['target']:.2f} | {label(r['detector'])} | {r['n_units']} / {r['n_all']} | "
            f"{r['fa_method']:.2f} vs {r['fa_base']:.2f} | "
            f"{ci(-r['mean_reduction'], -r['mean_hi'], -r['mean_lo'])} | "
            f"{ci(-r['pooled_reduction'], -r['pooled_hi'], -r['pooled_lo'])} | "
            f"{r['series_better']} / {r['series_worse']} | "
            f"{r['recall_method']:.2f} vs {r['recall_base']:.2f} |")
    return "\n".join(out)


def md_oracle(recs) -> str:
    out = ["| Test recall r | Detector | Common units | FA/day (method vs z-score) | "
           "Mean per-series change, 95% CI | Pooled change, 95% CI |",
           "|---|---|---|---|---|---|"]
    for r in recs:
        if r["detector"] == BASE:
            continue
        out.append(
            f"| {r['level']:.2f} | {label(r['detector'])} | {r['n_units']} / {r['n_all']} | "
            f"{r['fa_method']:.2f} vs {r['fa_base']:.2f} | "
            f"{ci(-r['mean_reduction'], -r['mean_hi'], -r['mean_lo'])} | "
            f"{ci(-r['pooled_reduction'], -r['pooled_hi'], -r['pooled_lo'])} |")
    return "\n".join(out)


def md_persistence(rows) -> str:
    out = ["| Detector | k | Test event recall | False alerts/day | Change in false alerts vs k=1 |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {label(r['detector'])} | {r['k']} | {r['recall']:.2f} | {r['fa_per_day']:.2f} | "
                   f"{'-' if r['k'] == 1 else pct(r['fa_change_vs_k1'])} |")
    return "\n".join(out)


def md_kinds(rows) -> str:
    out = ["| Detector | " + " | ".join(EVENT_KINDS) + " |", "|---|" + "---|" * len(EVENT_KINDS)]
    for r in rows:
        out.append(f"| {label(r['detector'])} | " + " | ".join(f"{r[k]:.2f}" for k in EVENT_KINDS) + " |")
    return "\n".join(out)


def md_scoring(rows) -> str:
    out = ["| Detector | PA precision | PA recall | PA F1 | Raw point precision | Raw point recall | "
           "Point FPR | Event precision | Event recall | False alerts/day |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {label(r['detector'])} | {r['pa_precision']:.2f} | {r['pa_recall']:.2f} | {r['pa_f1']:.2f} | "
                   f"{r['point_precision']:.2f} | {r['point_recall']:.2f} | {r['point_fpr']:.4f} | "
                   f"{r['event_precision']:.2f} | {r['event_recall']:.2f} | {r['fa_per_day']:.2f} |")
    return "\n".join(out)
