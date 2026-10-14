"""Online (streaming) monitoring: process one point at a time with bounded state.

``OnlineMonitor`` reproduces the batch pipeline (detector score, threshold, persistence rule,
data-quality rules) point by point. The batch detector is fitted once on a fit window and its
parameters are handed to an online scorer, which keeps only rolling statistics (a fixed-length
buffer of past values and, for the seasonal detector, of past residuals). The tests check that
the online scores and flags equal the batch ones.

Supported detectors: ``zscore``, ``rolling_mad``, ``seasonal``, ``ewma``.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .detectors import (
    MAD_TO_SIGMA,
    Detector,
    EWMAChart,
    GlobalZScore,
    RollingMAD,
    SeasonalResidual,
)
from .rules import DataQualityRules

STREAMING_DETECTORS = ("zscore", "rolling_mad", "seasonal", "ewma")


def _nanmedian(a: np.ndarray) -> float:
    if a.size == 0 or np.isnan(a).all():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmedian(a))


class _OnlineZScore:
    def __init__(self, det: GlobalZScore):
        self.mu, self.sd = det.mu, det.sd

    def update(self, v: float) -> float:
        return 0.0 if np.isnan(v) else abs(v - self.mu) / self.sd

    def state_size(self) -> int:
        return 0


class _OnlineRollingMAD:
    def __init__(self, det: RollingMAD):
        self.w, self.min_valid, self.floor, self.scale0 = det.w, det.min_valid, det.floor, det.scale0
        self.buf: deque[float] = deque(maxlen=self.w)

    def update(self, v: float) -> float:
        win = np.fromiter(self.buf, dtype=float, count=len(self.buf))
        n_valid = int(np.sum(~np.isnan(win)))
        med = _nanmedian(win)
        mad = _nanmedian(np.abs(win - med)) if n_valid else float("nan")
        self.buf.append(v)
        if np.isnan(v):
            return 0.0
        scale = max(MAD_TO_SIGMA * mad, self.floor * self.scale0) if not np.isnan(mad) else float("nan")
        s = abs(v - med) / scale if not np.isnan(scale) else float("nan")
        return 0.0 if (n_valid < self.min_valid or np.isnan(s)) else float(s)

    def state_size(self) -> int:
        return len(self.buf)


class _OnlineSeasonal:
    """Signed seasonal-residual z-score with the same history rules as the batch version."""

    def __init__(self, det: SeasonalResidual):
        self.p, self.profile_days = det.ppd, det.profile_days
        self.scale_days, self.floor = det.scale_days, det.floor
        assert det.scale0 is not None, "fit the detector before streaming"
        self.scale0: float = det.scale0
        self.hist: deque[float] = deque(maxlen=self.profile_days * self.p)
        self.res: deque[float] = deque(maxlen=self.scale_days * self.p)

    def signed(self, v: float) -> float:
        lags = [self.hist[-j * self.p] for j in range(1, self.profile_days + 1) if len(self.hist) >= j * self.p]
        vals = np.array(lags, dtype=float)
        n_valid = int(np.sum(~np.isnan(vals)))
        profile = _nanmedian(vals)
        r = v - profile if n_valid >= 3 else float("nan")
        win = np.fromiter(self.res, dtype=float, count=len(self.res))
        r_valid = int(np.sum(~np.isnan(win)))
        mad = _nanmedian(np.abs(win))
        self.hist.append(v)
        self.res.append(r)
        scale = max(MAD_TO_SIGMA * mad, self.floor * self.scale0) if not np.isnan(mad) else float("nan")
        z = r / scale if not np.isnan(scale) else float("nan")
        if r_valid < 2 * self.p or np.isnan(z) or np.isnan(v):
            return 0.0
        return float(z)

    def update(self, v: float) -> float:
        return abs(self.signed(v))

    def state_size(self) -> int:
        return len(self.hist) + len(self.res)


class _OnlineEWMA:
    def __init__(self, det: EWMAChart):
        self.lam, self.clip = det.lam, det.clip
        self.seasonal = _OnlineSeasonal(det.seasonal)
        self.e = 0.0
        self.sigma = float(np.sqrt(self.lam / (2 - self.lam)))

    def update(self, v: float) -> float:
        z = float(np.clip(self.seasonal.signed(v), -self.clip, self.clip))
        self.e = self.lam * z + (1 - self.lam) * self.e
        return 0.0 if np.isnan(v) else abs(self.e) / self.sigma

    def state_size(self) -> int:
        return self.seasonal.state_size() + 1


class _OnlineRules:
    """Missing / range / stuck checks with the causal run counter of the batch ``stuck_flags``."""

    def __init__(self, rules: DataQualityRules):
        assert rules.lo is not None and rules.hi is not None, "fit the rules before streaming"
        self.lo, self.hi, self.min_run = rules.lo, rules.hi, rules.min_run
        self.prev = float("nan")
        self.run = 1
        self.n = 0

    def update(self, v: float) -> tuple[str, ...]:
        hit: list[str] = []
        if np.isnan(v):
            hit.append("missing")
        elif v < self.lo or v > self.hi:
            hit.append("range")
        if self.n == 0:
            self.run = 1
        elif not np.isnan(v) and not np.isnan(self.prev) and v == self.prev:
            self.run += 1
        else:
            self.run = 1
        if self.n > 0 and self.min_run is not None and self.run >= self.min_run:
            hit.append("stuck")
        self.prev = v
        self.n += 1
        return tuple(hit)


@dataclass(frozen=True)
class StepResult:
    """Outcome of processing one point."""

    index: int
    value: float
    score: float
    flagged: bool
    alert_started: bool
    rules: tuple[str, ...] = ()


class OnlineMonitor:
    """Score, threshold, persistence and rule layer for a stream, one point at a time."""

    def __init__(self, scorer, threshold: float, k: int = 1, rules: DataQualityRules | None = None):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.scorer = scorer
        self.threshold = float(threshold)
        self.k = int(k)
        self.rules = _OnlineRules(rules) if rules is not None else None
        self.t = 0
        self.run = 0          # consecutive points above the threshold
        self.prev_flag = False

    @classmethod
    def from_detector(cls, det: Detector, threshold: float, k: int = 1,
                      rules: DataQualityRules | None = None) -> OnlineMonitor:
        """Build the online scorer from an already fitted batch detector."""
        if isinstance(det, EWMAChart):
            scorer: object = _OnlineEWMA(det)
        elif isinstance(det, SeasonalResidual):
            scorer = _OnlineSeasonal(det)
        elif isinstance(det, RollingMAD):
            scorer = _OnlineRollingMAD(det)
        elif isinstance(det, GlobalZScore):
            scorer = _OnlineZScore(det)
        else:
            raise ValueError(f"streaming is supported for {STREAMING_DETECTORS}, not {det.name!r}")
        return cls(scorer, threshold, k, rules)

    def update(self, value: float) -> StepResult:
        v = float(value)
        score = float(self.scorer.update(v))  # type: ignore[attr-defined]
        self.run = self.run + 1 if score > self.threshold else 0
        flag = self.run >= self.k
        hits: tuple[str, ...] = ()
        if self.rules is not None:
            hits = self.rules.update(v)
            flag = flag or bool(hits)
        res = StepResult(self.t, v, score, flag, flag and not self.prev_flag, hits)
        self.prev_flag = flag
        self.t += 1
        return res

    def state_size(self) -> int:
        """Number of stored values (bounded by the rolling windows, independent of stream length)."""
        return int(self.scorer.state_size())  # type: ignore[attr-defined]
