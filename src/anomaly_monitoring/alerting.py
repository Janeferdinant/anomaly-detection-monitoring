"""Alert suppression (persistence rule) and threshold calibration at a target recall."""
from __future__ import annotations

import numpy as np

from .metrics import alerts_per_day, event_metrics


def apply_persistence(flags, k: int) -> np.ndarray:
    """Alert-suppression rule: keep a flag only after ``k`` consecutive flagged points.

    Causal. The alert is raised at the k-th consecutive flagged point, so isolated blips
    shorter than ``k`` points never alert. ``k == 1`` returns the flags unchanged.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    f = np.asarray(flags, dtype=bool)
    if k == 1:
        return f.copy()
    idx = np.arange(f.size)
    last_false = np.maximum.accumulate(np.where(~f, idx, -1))
    return f & ((idx - last_false) >= k)


def _candidates(scores: np.ndarray, max_n: int = 200) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    u = np.unique(finite)
    if u.size <= max_n:
        return u
    # keep the upper half of the score distribution (thresholds below the median would
    # flag more than half of all points) on a quantile grid
    return np.unique(np.quantile(finite, np.linspace(0.5, 1.0, max_n)))


def calibrate_threshold(
    scores,
    labels,
    k: int,
    target_recall: float,
    points_per_day: float,
    extra_flags=None,
) -> dict:
    """Pick the highest threshold whose *calibration-window* event recall is >= target.

    Parameters are only the calibration-window arrays, so the test window cannot leak in.
    ``extra_flags`` (e.g. the data-quality rule layer) are OR-ed in after the persistence rule.
    If the target is unreachable the threshold with the best recall is returned (ties go
    to the higher threshold) and ``reached`` is False.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    extra = np.zeros(s.size, bool) if extra_flags is None else np.asarray(extra_flags, bool)

    best = None  # (recall, threshold, metrics) for the fallback
    chosen = None
    for thr in _candidates(s)[::-1]:  # highest threshold first
        flags = apply_persistence(s > thr, k) | extra
        m = event_metrics(y, flags, points_per_day)
        rec = m["event_recall"]
        if np.isnan(rec):
            rec = 0.0
        if rec >= target_recall - 1e-12:
            chosen = (thr, m, True)
            break
        if best is None or rec > best[0] + 1e-12:
            best = (rec, thr, m)
    if chosen is None:
        if best is None:  # no finite scores at all
            return {"threshold": float("inf"), "reached": False, "recall": 0.0,
                    "false_alerts_per_day": 0.0, "n_false_alerts": 0, "n_alerts": 0}
        chosen = (best[1], best[2], False)
    thr, m, reached = chosen
    return {
        "threshold": float(thr),
        "reached": bool(reached),
        "recall": float(m["event_recall"]) if not np.isnan(m["event_recall"]) else 0.0,
        "false_alerts_per_day": float(m["false_alerts_per_day"]),
        "n_false_alerts": int(m["n_false_alerts"]),
        "n_alerts": int(m["n_alerts"]),
    }


def threshold_for_budget(scores, k: int, budget_per_day: float, points_per_day: float,
                         extra_flags=None, max_n: int = 300) -> dict:
    """Lowest threshold whose alert rate on the calibration window stays within a budget.

    The alert budget is a number of alerts per day (all alerts, true or not), so no labels
    are needed. ``extra_flags`` (the rule layer) are OR-ed in after persistence and count
    against the budget. If the rule layer alone exceeds the budget no threshold can satisfy
    it, and ``{"met": False, "threshold": inf}`` is returned (the detector is then silent).
    """
    s = np.asarray(scores, dtype=float)
    extra = np.zeros(s.size, bool) if extra_flags is None else np.asarray(extra_flags, bool)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return {"threshold": float("inf"), "met": False, "alerts_per_day": 0.0}
    u = np.unique(finite)
    cand = u if u.size <= max_n else np.unique(np.quantile(finite, np.linspace(0.0, 1.0, max_n)))
    silent = alerts_per_day(extra, points_per_day)
    if silent > budget_per_day + 1e-12:
        return {"threshold": float("inf"), "met": False, "alerts_per_day": silent}
    best = (float("inf"), silent)
    for thr in cand[::-1]:  # from the highest threshold down; stop at the first violation
        rate = alerts_per_day(apply_persistence(s > thr, k) | extra, points_per_day)
        if rate > budget_per_day + 1e-12:
            break
        best = (float(thr), rate)
    return {"threshold": best[0], "met": True, "alerts_per_day": best[1]}
