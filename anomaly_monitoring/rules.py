"""Data-quality rule layer: missing values, range checks and stuck-value checks.

The rules are deliberately simple and explainable. Their parameters are fitted on the
*calibration window only*, without labels (range from robust quantiles, stuck-run length
from the natural run lengths in that window).
"""
from __future__ import annotations

import numpy as np


def missing_flags(x) -> np.ndarray:
    """True where the value is missing (NaN)."""
    return np.isnan(np.asarray(x, dtype=float))


def range_flags(x, lo: float, hi: float) -> np.ndarray:
    """True where a (non-missing) value falls outside ``[lo, hi]``."""
    a = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        return (a < lo) | (a > hi)


def fit_range(x, margin: float = 0.5, q: float = 0.005) -> tuple[float, float]:
    """Plausible range from the ``q`` / ``1-q`` quantiles, widened by ``margin`` of the span."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    lo, hi = np.quantile(a, [q, 1 - q])
    span = hi - lo
    return float(lo - margin * span), float(hi + margin * span)


def _run_lengths(a: np.ndarray) -> list[int]:
    runs, cur = [], 1
    for i in range(1, a.size):
        if not np.isnan(a[i]) and not np.isnan(a[i - 1]) and a[i] == a[i - 1]:
            cur += 1
        else:
            if cur > 1:
                runs.append(cur)
            cur = 1
    if cur > 1:
        runs.append(cur)
    return runs


def stuck_flags(x, min_run: int) -> np.ndarray:
    """Flag a run of identical non-missing values once it reaches ``min_run`` repeats.

    Causal: the first ``min_run - 1`` points of a run are not flagged (the rule cannot
    know yet), every later point of the run is.
    """
    a = np.asarray(x, dtype=float)
    flags = np.zeros(a.size, dtype=bool)
    run = 1
    for i in range(1, a.size):
        if not np.isnan(a[i]) and not np.isnan(a[i - 1]) and a[i] == a[i - 1]:
            run += 1
        else:
            run = 1
        if run >= min_run:
            flags[i] = True
    return flags


def fit_stuck_run(x, floor: int = 4, q: float = 0.99) -> int:
    """Minimum run length that counts as stuck: just above the ``q`` quantile of natural runs."""
    a = np.asarray(x, dtype=float)
    runs = _run_lengths(a)
    if not runs:
        return int(floor)
    return int(max(floor, np.ceil(np.quantile(runs, q))))


class DataQualityRules:
    """Missing + range + stuck rules with parameters fitted on a calibration window."""

    def __init__(self, range_margin: float = 0.5, stuck_floor: int = 4):
        self.range_margin = range_margin
        self.stuck_floor = stuck_floor
        self.lo = self.hi = None
        self.min_run = None

    def fit(self, x_calib) -> "DataQualityRules":
        self.lo, self.hi = fit_range(x_calib, margin=self.range_margin)
        self.min_run = fit_stuck_run(x_calib, floor=self.stuck_floor)
        return self

    def flags(self, x) -> np.ndarray:
        if self.lo is None:
            raise RuntimeError("fit() must be called before flags()")
        return missing_flags(x) | range_flags(x, self.lo, self.hi) | stuck_flags(x, self.min_run)
