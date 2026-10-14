"""Data-quality rule layer: missing values, range checks and stuck-value checks.

The rules are deliberately simple and explainable. Their parameters are fitted on the
*calibration window only*, without labels (range from robust quantiles, stuck-run length
from the natural run lengths in that window).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

COMPONENTS = ("missing", "range", "stuck")


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
    if a.size == 0:
        return float("-inf"), float("inf")
    lo, hi = np.quantile(a, [q, 1 - q])
    span = hi - lo
    if span <= 0:  # a flat channel has no meaningful range; the range rule is switched off
        return float("-inf"), float("inf")
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


def flat_fraction(x) -> float:
    """Share of consecutive non-missing pairs that repeat the previous value."""
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return 0.0
    ok = ~np.isnan(a[1:]) & ~np.isnan(a[:-1])
    if not ok.any():
        return 0.0
    return float(np.mean(a[1:][ok] == a[:-1][ok]))


def fit_stuck_run(x, floor: int = 4, min_runs: int = 10, max_flat_fraction: float = 0.5) -> int | None:
    """Minimum run length that counts as stuck, from the natural repeats in the window.

    Uses the *median* length of repeated-value runs plus 2 (robust to a few genuine
    stuck-sensor episodes hiding in the window), never below ``floor``. With fewer than
    ``min_runs`` repeated-value runs there is no reliable estimate, so ``floor`` is used.
    Returns ``None`` (rule switched off) for channels that repeat their previous value more
    than ``max_flat_fraction`` of the time: for those, "stuck" is normal behaviour.
    """
    if flat_fraction(x) > max_flat_fraction:
        return None
    runs = _run_lengths(np.asarray(x, dtype=float))
    if len(runs) < min_runs:
        return int(floor)
    return int(max(floor, np.ceil(np.median(runs)) + 2))


class DataQualityRules:
    """Missing + range + stuck rules with parameters fitted on a calibration window."""

    def __init__(self, range_margin: float = 0.5, stuck_floor: int = 4):
        self.range_margin = range_margin
        self.stuck_floor = stuck_floor
        self.lo: float | None = None
        self.hi: float | None = None
        self.min_run: int | None = None
        self.fitted = False

    def fit(self, x_calib) -> DataQualityRules:
        self.lo, self.hi = fit_range(x_calib, margin=self.range_margin)
        self.min_run = fit_stuck_run(x_calib, floor=self.stuck_floor)
        self.fitted = True
        return self

    def components(self, x) -> dict[str, np.ndarray]:
        """Flags of each rule on its own (``missing``, ``range``, ``stuck``)."""
        if not self.fitted or self.lo is None or self.hi is None:
            raise RuntimeError("fit() must be called before flags()")
        a = np.asarray(x, dtype=float)
        stuck = np.zeros(a.size, dtype=bool) if self.min_run is None else stuck_flags(a, self.min_run)
        return {"missing": missing_flags(a), "range": range_flags(a, self.lo, self.hi), "stuck": stuck}

    def flags(self, x, use: Iterable[str] = COMPONENTS) -> np.ndarray:
        comp = self.components(x)
        out = np.zeros(np.asarray(x).shape[0], dtype=bool)
        for name in use:
            out |= comp[name]
        return out


class MultiDataQualityRules:
    """The rule layer for a matrix of metrics: one rule set per channel, flagged when any channel is."""

    def __init__(self, range_margin: float = 0.5, stuck_floor: int = 4):
        self.range_margin = range_margin
        self.stuck_floor = stuck_floor
        self.rules: list[DataQualityRules] = []

    def fit(self, x_calib: Any) -> MultiDataQualityRules:
        a = np.asarray(x_calib, dtype=float)
        a = a[:, None] if a.ndim == 1 else a
        self.rules = [DataQualityRules(self.range_margin, self.stuck_floor).fit(a[:, j])
                      for j in range(a.shape[1])]
        return self

    def components(self, x: Any) -> dict[str, np.ndarray]:
        a = np.asarray(x, dtype=float)
        a = a[:, None] if a.ndim == 1 else a
        out = {c: np.zeros(a.shape[0], dtype=bool) for c in COMPONENTS}
        for j, r in enumerate(self.rules):
            for c, f in r.components(a[:, j]).items():
                out[c] |= f
        return out

    def flags(self, x: Any, use: Iterable[str] = COMPONENTS) -> np.ndarray:
        comp = self.components(x)
        n = next(iter(comp.values())).size
        out = np.zeros(n, dtype=bool)
        for name in use:
            out |= comp[name]
        return out
