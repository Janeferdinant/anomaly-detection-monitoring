"""Multi-method benchmark with operations-style metrics (matched recall, alert budgets, ablations).

A *unit* is one labelled series cut into three consecutive windows:

    fit window        detectors and rules are fitted here (values only, no labels)
    calibration win.  thresholds are chosen here (labels used for matched recall; no labels for
                      alert budgets) and hyper-parameters are tuned here
    test window       later in time; used only for reporting

Two protocols are run on every unit for every method:

* **matched recall**: per configuration the threshold is the highest one whose calibration event
  recall reaches a target; the frozen configuration is scored on the test window;
* **alert budget**: per configuration the threshold is the lowest one that keeps the alert rate on
  the calibration window within a budget of alerts per day (no labels); scored on the test window.

Hyper-parameters come from small grids (``detectors.PARAM_GRIDS``) and are chosen on the
calibration windows only: first the share of units that cannot reach the target recall, then the
mean calibration false alerts per day. The test window is never scored during tuning.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .alerting import apply_persistence, calibrate_threshold, threshold_for_budget
from .detectors import PARAM_GRIDS, STOCHASTIC, make_detector
from .metrics import (
    alerts_per_day,
    event_metrics,
    point_adjust,
    point_metrics,
    segments,
    time_to_detect,
)
from .rules import COMPONENTS, DataQualityRules, MultiDataQualityRules

BASELINE = "zscore"
# the six detectors of the earlier run first, then the new ones, then the ensemble
METHODS = ("zscore", "rolling_mad", "seasonal", "ewma", "combo", "stl", "iforest", "lof", "ocsvm",
           "pca", "autoencoder", "ensemble")
TARGETS = (0.5, 0.65, 0.8)
KS = (1, 2, 3)
BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # alerts per day
RULE_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "all": COMPONENTS,
    "missing": ("missing",),
    "range": ("range",),
    "stuck": ("stuck",),
    "all-missing": ("range", "stuck"),
    "all-range": ("missing", "stuck"),
    "all-stuck": ("missing", "range"),
}
MAIN_RULES = ("none", "all")
ABLATION_RULES = tuple(r for r in RULE_SETS if r not in MAIN_RULES)
# rule-layer settings tuned on calibration windows: range margin (share of the 0.5-99.5% span) and
# the highest share of repeated values for which the stuck rule is still applied to a channel
RULE_GRID: list[dict[str, Any]] = [
    {"range_margin": 0.5, "max_flat_fraction": 0.5},
    {"range_margin": 0.5, "max_flat_fraction": 0.05},
    {"range_margin": 2.0, "max_flat_fraction": 0.5},
    {"range_margin": 2.0, "max_flat_fraction": 0.05},
    {"range_margin": 5.0, "max_flat_fraction": 0.05},
]
MIN_CAL_EVENTS = 2  # fewer calibration events cannot define a recall target


@dataclass
class Unit:
    """One labelled series with its fit / calibration / test windows."""

    name: str
    dataset: str
    values: np.ndarray            # (T,) or (T, d)
    labels: np.ndarray            # (T,) bool
    fit_end: int
    cal: tuple[int, int]
    test: tuple[int, int]
    ppd: int
    minutes_per_point: float
    seed: int = 0                 # injection seed (UCI) or 0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return 1 if self.values.ndim == 1 else int(self.values.shape[1])

    @property
    def key(self) -> tuple[str, int]:
        return (self.name, self.seed)

    def n_events(self, window: tuple[int, int]) -> int:
        return len(segments(self.labels[window[0]:window[1]]))

    @property
    def matched_ok(self) -> bool:
        return self.n_events(self.cal) >= MIN_CAL_EVENTS and self.n_events(self.test) >= 1

    @property
    def budget_ok(self) -> bool:
        return self.n_events(self.test) >= 1


# ------------------------------------------------------------------------------------------
# unit builders
# ------------------------------------------------------------------------------------------

def uci_units(seeds, n_windows: int = 3, warmup_days: int = 14) -> list[Unit]:
    """12 UCI household-power series x injection seeds, labels injected (as in the first run)."""
    from .data import CHANNELS, POINTS_PER_DAY, load_hourly, make_units
    from .inject import inject_anomalies

    hourly = load_hourly()
    units = []
    for u in make_units(hourly, channels=list(CHANNELS), n_windows=n_windows):
        for seed in seeds:
            res = inject_anomalies(u["values"], seed=seed, points_per_day=POINTS_PER_DAY, warmup_days=warmup_days)
            split = res.values.size // 2
            units.append(Unit(u["name"], "uci", res.values, res.labels, fit_end=split,
                              cal=(warmup_days * POINTS_PER_DAY, split), test=(split, res.values.size),
                              ppd=POINTS_PER_DAY, minutes_per_point=60.0, seed=seed,
                              info={"channel": u["channel"]}))
    return units


def smd_units(machines=None) -> list[Unit]:
    """Server Machine Dataset machines: train (unlabelled) | calibration half | test half of the test file."""
    from . import smd

    units = []
    for name in machines or smd.MACHINES:
        m = smd.load_machine(name)
        n_tr, n_te = m.train.shape[0], m.test.shape[0]
        half = n_te // 2
        keep = m.train.std(axis=0) > 1e-6  # drop channels that are constant in the training data
        values = np.vstack([m.train, m.test])[:, keep]
        labels = np.concatenate([np.zeros(n_tr, dtype=bool), m.labels])
        units.append(Unit(name, "smd", values, labels, fit_end=n_tr, cal=(n_tr, n_tr + half),
                          test=(n_tr + half, n_tr + n_te), ppd=smd.POINTS_PER_DAY,
                          minutes_per_point=float(smd.MINUTES_PER_POINT), seed=0,
                          info={"channels_total": int(keep.size), "channels_used": int(keep.sum())}))
    return units


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------

def method_params(method: str, chosen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Parameters for ``method``; the ensemble is built from the chosen member parameters."""
    p = dict(chosen.get(method, {}))
    if method == "ensemble":
        from .detectors import ENSEMBLE_MEMBERS

        p["member_params"] = {m: chosen.get(m, {}) for m in ENSEMBLE_MEMBERS}
    return p


def fit_and_score(unit: Unit, method: str, params: dict[str, Any], model_seed: int,
                  until: int | None = None) -> np.ndarray:
    """Fit on the fit window, score the series causally (optionally only up to ``until``)."""
    det = make_detector(method, seed=model_seed, points_per_day=unit.ppd, params=params,
                        n_channels=unit.n_channels)
    det.fit(unit.values[: unit.fit_end])
    x = unit.values if until is None else unit.values[:until]
    return det.score(x)


def fit_rules(unit: Unit, rule_params: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    rules: DataQualityRules | MultiDataQualityRules
    p: dict[str, Any] = rule_params or {}
    rules = MultiDataQualityRules(**p) if unit.n_channels > 1 else DataQualityRules(**p)
    rules.fit(unit.values[: unit.fit_end])
    comp = rules.components(unit.values)
    return {name: np.logical_or.reduce([comp[c] for c in use]) if use else np.zeros(unit.values.shape[0], bool)
            for name, use in RULE_SETS.items()}


# ------------------------------------------------------------------------------------------
# tuning on calibration windows
# ------------------------------------------------------------------------------------------

def _tune_task(args):
    unit, method, gi, params, target = args
    cal0, cal1 = unit.cal
    s = fit_and_score(unit, method, params, unit.seed, until=cal1)
    res = calibrate_threshold(s[cal0:cal1], unit.labels[cal0:cal1], 1, target, unit.ppd)
    return method, gi, unit.key, res["reached"], res["false_alerts_per_day"]


def tune(units: list[Unit], methods=METHODS, target: float = 0.65, jobs: int = 1,
         max_units: int | None = None) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Choose each method's parameters on the calibration windows only.

    Returns ``(chosen, log)``; ``log`` holds every grid entry's calibration cost.
    """
    cal_units = [u for u in units if u.matched_ok]
    if max_units is not None:
        cal_units = cal_units[:max_units]
    chosen: dict[str, dict[str, Any]] = {}
    rows = []
    for stage in (tuple(m for m in methods if m != "ensemble"), tuple(m for m in methods if m == "ensemble")):
        tasks = []
        for m in stage:
            for gi, params in enumerate(PARAM_GRIDS[m]):
                p = method_params(m, chosen) if m == "ensemble" else dict(params)
                if m == "ensemble":
                    p.update(params)
                tasks += [(u, m, gi, p, target) for u in cal_units]
        for out in _map(_tune_task, tasks, jobs):
            rows.append(out)
        log = pd.DataFrame(rows, columns=["method", "grid_index", "unit", "reached", "cal_fa_per_day"])
        for m in stage:
            g = log[log["method"] == m].groupby("grid_index").agg(
                unreached=("reached", lambda r: 1.0 - float(np.mean(r))),
                fa=("cal_fa_per_day", "mean"))
            best = g.sort_values(["unreached", "fa"], kind="stable").index[0]
            chosen[m] = dict(PARAM_GRIDS[m][int(best)])
    log = pd.DataFrame(rows, columns=["method", "grid_index", "unit", "reached", "cal_fa_per_day"])
    summary = log.groupby(["method", "grid_index"]).agg(
        n_units=("unit", "nunique"), share_unreached=("reached", lambda r: 1.0 - float(np.mean(r))),
        mean_cal_fa_per_day=("cal_fa_per_day", "mean")).reset_index()
    summary["params"] = [str(PARAM_GRIDS[m][g]) for m, g in zip(summary["method"], summary["grid_index"])]
    summary["chosen"] = [PARAM_GRIDS[m][g] == chosen[m] for m, g in zip(summary["method"], summary["grid_index"])]
    return chosen, summary


def _tune_rules_task(args):
    unit, gi, params, target = args
    cal0, cal1 = unit.cal
    s = fit_and_score(unit, BASELINE, {}, unit.seed, until=cal1)
    extra = fit_rules(unit, params)["all"][cal0:cal1]
    res = calibrate_threshold(s[cal0:cal1], unit.labels[cal0:cal1], 1, target, unit.ppd, extra_flags=extra)
    return gi, res["reached"], res["false_alerts_per_day"]


def tune_rules(units: list[Unit], target: float, jobs: int = 1) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose the rule-layer settings on calibration windows (z-score + all rules, k=1)."""
    cal_units = [u for u in units if u.matched_ok]
    tasks = [(u, gi, p, target) for gi, p in enumerate(RULE_GRID) for u in cal_units]
    log = pd.DataFrame(_map(_tune_rules_task, tasks, jobs), columns=["grid_index", "reached", "cal_fa_per_day"])
    g = log.groupby("grid_index").agg(share_unreached=("reached", lambda r: 1.0 - float(np.mean(r))),
                                      mean_cal_fa_per_day=("cal_fa_per_day", "mean")).reset_index()
    best = int(g.sort_values(["share_unreached", "mean_cal_fa_per_day"], kind="stable")["grid_index"].iloc[0])
    g["params"] = [str(RULE_GRID[i]) for i in g["grid_index"]]
    g["chosen"] = g["grid_index"] == best
    return dict(RULE_GRID[best]), g


# ------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------

def _window_metrics(unit: Unit, flags: np.ndarray, window: tuple[int, int]) -> dict[str, float]:
    a, b = window
    y, f = unit.labels[a:b], flags[a:b]
    em = event_metrics(y, f, unit.ppd)
    ttd = time_to_detect(y, f) * unit.minutes_per_point
    pa = point_metrics(y, point_adjust(y, f))
    return {
        "n_events": em["n_events"], "event_recall": em["event_recall"], "n_alerts": em["n_alerts"],
        "n_false_alerts": em["n_false_alerts"], "false_alerts_per_day": em["false_alerts_per_day"],
        "alerts_per_day": alerts_per_day(f, unit.ppd), "event_precision": em["event_precision"],
        "ttd_median_min": float(np.nanmedian(ttd)) if np.isfinite(ttd).any() else float("nan"),
        "ttd_sum_min": float(np.nansum(ttd)), "n_detected": int(np.isfinite(ttd).sum()),
        "pa_recall": pa["recall"], "point_recall": point_metrics(y, f)["recall"],
    }


def evaluate_unit_method(unit: Unit, method: str, scores: np.ndarray, dq: dict[str, np.ndarray],
                         model_seed: int, cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """All matched-recall and alert-budget rows of one method on one unit."""
    cal0, cal1 = unit.cal
    t0, t1 = unit.test
    base = {"dataset": unit.dataset, "series": unit.name, "unit_seed": unit.seed, "seed": model_seed,
            "method": method}
    matched: list[dict] = []
    budget: list[dict] = []
    primary = cfg["primary_target"]

    if unit.matched_ok:
        combos = [(t, k, r) for t in cfg["targets"] for k in cfg["ks"] for r in MAIN_RULES]
        combos += [(primary, 1, r) for r in ABLATION_RULES]
        for target, k, rname in combos:
            extra = dq[rname]
            cal = calibrate_threshold(scores[cal0:cal1], unit.labels[cal0:cal1], k, target, unit.ppd,
                                      extra_flags=extra[cal0:cal1] if rname != "none" else None)
            flags = apply_persistence(scores > cal["threshold"], k) | extra
            m = _window_metrics(unit, flags, (t0, t1))
            row = dict(base, target=target, k=k, rules=rname, threshold=cal["threshold"],
                       reached=cal["reached"], cal_event_recall=cal["recall"],
                       cal_false_alerts_per_day=cal["false_alerts_per_day"],
                       n_cal_events=unit.n_events(unit.cal))
            row.update({f"test_{n}": v for n, v in m.items()})
            matched.append(row)

    if unit.budget_ok:
        for bud in cfg["budgets"]:
            for k in cfg["ks"]:
                for rname in MAIN_RULES:
                    extra = dq[rname]
                    thr = threshold_for_budget(scores[cal0:cal1], k, bud, unit.ppd,
                                               extra_flags=extra[cal0:cal1] if rname != "none" else None)
                    flags = apply_persistence(scores > thr["threshold"], k) | extra
                    m = _window_metrics(unit, flags, (t0, t1))
                    row = dict(base, budget=bud, k=k, rules=rname, threshold=thr["threshold"],
                               budget_met=thr["met"], cal_alerts_per_day=thr["alerts_per_day"])
                    row.update({f"test_{n}": v for n, v in m.items()})
                    budget.append(row)
    return matched, budget


def _eval_task(args):
    unit, model_seed, methods, chosen, cfg = args
    dq = fit_rules(unit, cfg.get("rule_params"))
    matched: list[dict] = []
    budget: list[dict] = []
    for m in methods:
        if model_seed != 0 and m not in STOCHASTIC and unit.dataset != "uci":
            continue  # deterministic methods do not depend on the model seed
        s = fit_and_score(unit, m, method_params(m, chosen), model_seed if unit.dataset != "uci" else unit.seed)
        a, b = evaluate_unit_method(unit, m, s, dq, model_seed, cfg)
        matched += a
        budget += b
    return matched, budget


def _map(fn, tasks, jobs: int):
    if jobs <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def evaluate(units: list[Unit], seeds, chosen: dict[str, dict[str, Any]], primary_target: float,
             methods=METHODS, jobs: int = 1,
             rule_params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both protocols for every unit and method; returns ``(matched_rows, budget_rows)``."""
    cfg = {"targets": TARGETS, "ks": KS, "budgets": BUDGETS, "primary_target": primary_target,
           "rule_params": rule_params}
    if units and units[0].dataset == "uci":
        tasks = [(u, 0, methods, chosen, cfg) for u in units]  # injection seed = model seed, one pass
    else:
        tasks = [(u, s, methods, chosen, cfg) for u in units for s in seeds]
    matched, budget = [], []
    for a, b in _map(_eval_task, tasks, jobs):
        matched += a
        budget += b
    return pd.DataFrame(matched), pd.DataFrame(budget)


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))
