"""Matched-recall evaluation of detectors, persistence rules and the data-quality rule layer.

Protocol (per series and per seed)
----------------------------------
1. Inject labelled incidents (``inject.py``). The series is split in the middle into a
   *calibration window* (first half) and a *test window* (second half).
2. Detectors and the rule layer are fitted on the calibration values only (no labels).
3. For every configuration (detector, persistence ``k``, rule layer on/off) a threshold is
   chosen on the calibration window as the highest threshold whose event recall is at
   least ``target_recall`` ("matched recall"). Labels of the test window are never used.
4. The frozen configuration is scored on the test window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alerting import apply_persistence, calibrate_threshold
from .detectors import make_detector
from .inject import EVENT_KINDS, inject_anomalies
from .metrics import event_metrics, point_adjust, point_metrics
from .rules import DataQualityRules


@dataclass(frozen=True)
class EvalConfig:
    detectors: tuple = ("zscore", "rolling_mad", "seasonal", "ewma", "iforest", "combo")
    ks: tuple = (1, 2, 3)
    targets: tuple = (0.8,)  # event-recall targets on the calibration window
    points_per_day: int = 24
    dq_options: tuple = (False, True)
    oracle_levels: tuple = (0.5, 0.7, 0.9)  # secondary analysis only, see evaluate_series


def _kind_counts(events, flags, lo: int) -> dict:
    """Detected / total events per kind for events that start at or after ``lo``."""
    out = {}
    for kind in EVENT_KINDS:
        evs = [e for e in (events or []) if e.kind == kind and e.start >= lo]
        out[f"n_{kind}"] = len(evs)
        out[f"det_{kind}"] = int(sum(bool(flags[e.start : e.end].any()) for e in evs))
    return out


def evaluate_series(x, labels, split: int, warmup: int, seed: int, cfg: EvalConfig, events=None) -> list[dict]:
    """Evaluate every configuration on one labelled series; returns one row per configuration.

    ``events`` (optional list of ``inject.Event``) only adds per-kind detection counts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ppd = cfg.points_per_day
    calib = slice(warmup, split)
    test = slice(split, x.size)

    rules = DataQualityRules().fit(x[:split])
    dq_full = rules.flags(x)

    rows = []
    for name in cfg.detectors:
        det = make_detector(name, seed=seed, points_per_day=ppd).fit(x[:split])
        scores = det.score(x)
        oracle = {}
        for dq in cfg.dq_options:
            # SECONDARY ANALYSIS (not deployable): choose the threshold on the *test* window to
            # reach test event recall r, i.e. compare the detectors' ranking quality alone,
            # independent of how well a calibration-window threshold transfers.
            extra_t = dq_full[test] if dq else None
            for r in cfg.oracle_levels:
                oc = calibrate_threshold(scores[test], y[test], 1, r, ppd, extra_flags=extra_t)
                oracle[(dq, r)] = oc["false_alerts_per_day"] if oc["reached"] else float("nan")
        for target in cfg.targets:
            for dq in cfg.dq_options:
                extra_c = dq_full[calib] if dq else None
                thr_k1 = None
                for k in cfg.ks:
                    cal = calibrate_threshold(scores[calib], y[calib], k, target, ppd, extra_flags=extra_c)
                    if k == cfg.ks[0]:
                        thr_k1 = cal["threshold"]
                    flags = apply_persistence(scores > cal["threshold"], k)
                    fixed = apply_persistence(scores > thr_k1, k)
                    if dq:
                        flags = flags | dq_full
                        fixed = fixed | dq_full
                    em = event_metrics(y[test], flags[test], ppd)
                    fm = event_metrics(y[test], fixed[test], ppd)
                    pm = point_metrics(y[test], flags[test])
                    pa = point_metrics(y[test], point_adjust(y[test], flags[test]))
                    row = {
                        "detector": name, "k": k, "dq": dq, "target": target, "seed": seed,
                        "threshold": cal["threshold"], "reached": cal["reached"],
                        "calib_event_recall": cal["recall"],
                        "calib_false_alerts_per_day": cal["false_alerts_per_day"],
                        "test_n_events": em["n_events"],
                        "test_event_recall": em["event_recall"],
                        "test_n_alerts": em["n_alerts"],
                        "test_n_false_alerts": em["n_false_alerts"],
                        "test_false_alerts_per_day": em["false_alerts_per_day"],
                        "test_event_precision": em["event_precision"],
                        "test_pa_precision": pa["precision"],
                        "test_pa_recall": pa["recall"],
                        "test_pa_f1": pa["f1"],
                        "test_point_precision": pm["precision"],
                        "test_point_recall": pm["recall"],
                        "test_point_fpr": pm["fpr"],
                        # same threshold as k=1, only the persistence rule changes
                        "fixed_thr_event_recall": fm["event_recall"],
                        "fixed_thr_false_alerts_per_day": fm["false_alerts_per_day"],
                    }
                    for r in cfg.oracle_levels:
                        row[f"oracle_fa_r{int(round(r * 100))}"] = oracle[(dq, r)] if k == cfg.ks[0] else float("nan")
                    row.update(_kind_counts(events, flags, split))
                    rows.append(row)
    return rows


def run_all(units: list[dict], seeds, cfg: EvalConfig, warmup_days: int = 14) -> pd.DataFrame:
    """Evaluate every (series, seed). ``units`` come from ``data.make_units``."""
    rows = []
    for unit in units:
        for seed in seeds:
            res = inject_anomalies(unit["values"], seed=seed, points_per_day=cfg.points_per_day,
                                   warmup_days=warmup_days)
            split = res.values.size // 2
            for r in evaluate_series(res.values, res.labels, split, warmup_days * cfg.points_per_day,
                                     seed, cfg, events=res.events):
                r.update(series=unit["name"], channel=unit["channel"])
                rows.append(r)
    return pd.DataFrame(rows)


def select_on_calibration(df: pd.DataFrame, require_dq: bool = True, exclude=("zscore",)) -> pd.DataFrame:
    """Per (series, seed): pick the configuration with the fewest *calibration* false alerts
    among those that reached the target recall (ties: smaller k, then detector order)."""
    cand = df[df["reached"] & ~df["detector"].isin(exclude)]
    if require_dq:
        cand = cand[cand["dq"]]
    order = {d: i for i, d in enumerate(df["detector"].drop_duplicates())}
    cand = cand.assign(_o=cand["detector"].map(order))
    cand = cand.sort_values(["series", "seed", "calib_false_alerts_per_day", "k", "_o"])
    sel = cand.groupby(["series", "seed"], as_index=False).head(1).drop(columns="_o")
    sel = sel.assign(chosen=sel["detector"] + "+k" + sel["k"].astype(str))
    sel["detector"] = "auto"
    return sel


def bootstrap_ci(values, seed: int = 0, n_boot: int = 10000, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean. Returns ``(mean, lo, hi)``."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def matched_pairs(method: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only (series, seed) units where *both* configurations reached the target recall
    on the calibration window, i.e. the units where recall was actually matched."""
    key = ["series", "seed"]
    m = method[method["reached"]].set_index(key)
    b = base[base["reached"]].set_index(key)
    common = m.index.intersection(b.index)
    return m.loc[common].reset_index(), b.loc[common].reset_index()


def per_series_mean(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("series")[col].mean()


def reduction_from_series(fm: pd.Series, fb: pd.Series, boot_seed: int = 0, n_boot: int = 10000) -> dict:
    """Reduction of ``fm`` relative to ``fb`` (both: false alerts per day, indexed by series).

    * ``mean_reduction``: mean over series of 1 - fm/fb (series with fb == 0 are skipped),
      with a 95% percentile-bootstrap CI over series;
    * ``pooled_reduction``: 1 - sum(fm)/sum(fb), with a 95% bootstrap CI over series.
    """
    both = pd.concat([fm.rename("m"), fb.rename("b")], axis=1).dropna()
    ok = both[both["b"] > 0]
    per_series = 1 - ok["m"] / ok["b"]
    mean, lo, hi = bootstrap_ci(per_series.to_numpy(), seed=boot_seed, n_boot=n_boot) if len(ok) else (float("nan"),) * 3

    rng = np.random.default_rng(boot_seed + 1)
    m, b = both["m"].to_numpy(), both["b"].to_numpy()
    idx = rng.integers(0, m.size, size=(n_boot, m.size))
    sb = b[idx].sum(axis=1)
    pooled_boot = np.where(sb > 0, 1 - m[idx].sum(axis=1) / np.where(sb > 0, sb, 1), np.nan)
    plo, phi = np.nanquantile(pooled_boot, [0.025, 0.975])
    pooled = 1 - m.sum() / b.sum() if b.sum() > 0 else float("nan")
    return {
        "n_series": int(len(both)),
        "n_series_with_baseline_alerts": int(len(ok)),
        "mean_reduction": mean, "mean_lo": lo, "mean_hi": hi,
        "pooled_reduction": float(pooled), "pooled_lo": float(plo), "pooled_hi": float(phi),
        "series_better": int((per_series > 0).sum()),
        "series_worse": int((per_series < 0).sum()),
        "series_equal": int((per_series == 0).sum()),
        "fa_method": float(m.mean()), "fa_base": float(b.mean()),
    }


def reduction_vs_baseline(method: pd.DataFrame, base: pd.DataFrame, boot_seed: int = 0) -> dict:
    """False-alert reduction of ``method`` relative to ``base`` on matched units.

    Both frames hold test results for the same (series, seed) units. Values are first
    averaged over seeds within each series; the CI is a bootstrap over series.
    """
    return reduction_from_series(per_series_mean(method, "test_false_alerts_per_day"),
                                 per_series_mean(base, "test_false_alerts_per_day"), boot_seed)
