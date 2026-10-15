"""Summary tables for the multi-method benchmark (matched recall, alert budgets, ablations)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import BASELINE, METHODS
from .detectors import DETECTOR_LABELS
from .evaluate import bootstrap_ci, reduction_from_series


def label(m: str) -> str:
    return DETECTOR_LABELS.get(m, m)


def add_unit(df: pd.DataFrame) -> pd.DataFrame:
    """A unit is one series (and, for injected UCI data, one injection seed)."""
    out = df.copy()
    out["unit"] = out["series"] + "|" + out["unit_seed"].astype(str)
    return out


def collapse_seeds(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average model seeds within a unit (``reached`` must hold for every seed)."""
    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in keys + ["seed"]]
    g = df.groupby(keys, as_index=False)
    out = g[num].mean()
    if "reached" in df.columns:
        out["reached"] = df.groupby(keys)["reached"].all().to_numpy()
    if "budget_met" in df.columns:
        out["budget_met"] = df.groupby(keys)["budget_met"].all().to_numpy()
    return out


def _ci(values) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    return bootstrap_ci(v, seed=0, n_boot=4000)


def _per_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("series")[col].mean()


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the unit id and average model seeds within a unit."""
    df = add_unit(df)
    keys = [c for c in ["dataset", "series", "unit", "method", "target", "k", "rules", "budget"] if c in df.columns]
    return collapse_seeds(df.drop(columns=["unit_seed"]), keys)


def matched_vs_baseline(df: pd.DataFrame, target: float, k: int = 1, rules: str = "none",
                        methods=METHODS) -> pd.DataFrame:
    """Each method against the global z-score at the same calibration recall target.

    Only units where both reached the target on the calibration window are compared. The change
    in test false alerts per day is pooled over series with a 95% bootstrap interval over series.
    """
    base = _sel(df, method=BASELINE, target=target, k=1, rules=rules)
    rows = []
    for m in methods:
        if m == BASELINE:
            continue
        me = _sel(df, method=m, target=target, k=k, rules=rules)
        common = set(me[me["reached"]]["unit"]) & set(base[base["reached"]]["unit"])
        a, b = me[me["unit"].isin(common)], base[base["unit"].isin(common)]
        if a.empty:
            rows.append({"method": m, "n_units": 0})
            continue
        res = reduction_from_series(_per_series(a, "test_false_alerts_per_day"),
                                    _per_series(b, "test_false_alerts_per_day"))
        res.update(method=m, n_units=len(a), recall_method=float(a["test_event_recall"].mean()),
                   recall_base=float(b["test_event_recall"].mean()),
                   ttd_method=float(a["test_ttd_median_min"].median()),
                   ttd_base=float(b["test_ttd_median_min"].median()))
        rows.append(res)
    return pd.DataFrame(rows)


def operating_point_table(df: pd.DataFrame, target: float, k: int = 1, rules: str = "none",
                          methods=METHODS) -> pd.DataFrame:
    """Test-window operating point of every method (including the baseline) at one target."""
    rows = []
    for m in methods:
        d = _sel(df, method=m, target=target, k=k, rules=rules)
        if d.empty:
            continue
        fa = _ci(_per_series(d, "test_false_alerts_per_day"))
        rc = _ci(_per_series(d, "test_event_recall"))
        pr = _ci(_per_series(d, "test_event_precision"))
        rows.append({"method": m, "reached_share": float(d["reached"].mean()),
                     "fa_per_day": fa[0], "fa_lo": fa[1], "fa_hi": fa[2],
                     "recall": rc[0], "recall_lo": rc[1], "recall_hi": rc[2],
                     "precision": pr[0], "precision_lo": pr[1], "precision_hi": pr[2],
                     "ttd_median_min": float(d["test_ttd_median_min"].median()),
                     "cal_fa_per_day": float(d["cal_false_alerts_per_day"].mean())})
    return pd.DataFrame(rows)


def budget_curve(bdf: pd.DataFrame, k: int = 1, rules: str = "none", methods=METHODS) -> pd.DataFrame:
    """Alert-fatigue curve: test recall, false alerts, precision and delay per alert budget."""
    rows = []
    for m in methods:
        for bud in sorted(bdf["budget"].unique()):
            d = _sel(bdf, method=m, budget=bud, k=k, rules=rules)
            if d.empty:
                continue
            rc = _ci(_per_series(d, "test_event_recall"))
            rows.append({"method": m, "budget": bud, "recall": rc[0], "recall_lo": rc[1], "recall_hi": rc[2],
                         "fa_per_day": float(_per_series(d, "test_false_alerts_per_day").mean()),
                         "alerts_per_day": float(_per_series(d, "test_alerts_per_day").mean()),
                         "precision": float(_per_series(d, "test_event_precision").mean()),
                         "ttd_median_min": float(d["test_ttd_median_min"].median()),
                         "budget_met_share": float(d["budget_met"].mean())})
    return pd.DataFrame(rows)


def ablation_table(df: pd.DataFrame, target: float, rule_sets, methods=METHODS) -> pd.DataFrame:
    """Rule-layer ablation at k=1: false alerts and recall for each subset of {missing, range, stuck}."""
    rows = []
    for m in methods:
        for r in rule_sets:
            d = _sel(df, method=m, target=target, k=1, rules=r)
            if d.empty:
                continue
            rows.append({"method": m, "rules": r, "reached_share": float(d["reached"].mean()),
                         "fa_per_day": float(_per_series(d, "test_false_alerts_per_day").mean()),
                         "recall": float(_per_series(d, "test_event_recall").mean()),
                         "n_alerts": float(d["test_n_alerts"].mean())})
    return pd.DataFrame(rows)


def persistence_table(df: pd.DataFrame, target: float, rules: str = "none", methods=METHODS) -> pd.DataFrame:
    rows = []
    for m in methods:
        for k in sorted(df["k"].unique()):
            d = _sel(df, method=m, target=target, k=k, rules=rules)
            if d.empty:
                continue
            rows.append({"method": m, "k": k, "reached_share": float(d["reached"].mean()),
                         "fa_per_day": float(_per_series(d, "test_false_alerts_per_day").mean()),
                         "recall": float(_per_series(d, "test_event_recall").mean()),
                         "ttd_median_min": float(d["test_ttd_median_min"].median())})
    return pd.DataFrame(rows)


def best_by_calibration(df: pd.DataFrame, target: float, k: int = 1, rules: str = "none",
                        min_reached: float = 0.9) -> dict:
    """Best method chosen on calibration windows only; its test numbers are then reported.

    Candidates must reach the target on at least ``min_reached`` of the units; among them the one
    with the lowest mean calibration false alerts per day wins (the baseline is a candidate).
    """
    cand = []
    for m in df["method"].unique():
        d = _sel(df, method=m, target=target, k=k, rules=rules)
        if d.empty or d["reached"].mean() < min_reached:
            continue
        cand.append((float(d["cal_false_alerts_per_day"].mean()), m))
    cand.sort()
    return {"best": cand[0][1] if cand else None, "ranking": cand}


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    """Markdown table; ``cols`` are (column, header, format) with format e.g. ``.2f`` or ``s``."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c, _, f in cols:
            v = r[c]
            if f == "s":
                cells.append(str(v))
            elif f == "label":
                cells.append(label(str(v)))
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append("n/a")
            else:
                cells.append(format(v, f))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_ci(v: float, lo: float, hi: float, f: str = ".2f") -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:{f}} [{lo:{f}}, {hi:{f}}]"


def pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{100 * v:+.0f}%"
