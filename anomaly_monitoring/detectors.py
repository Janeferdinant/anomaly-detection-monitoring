"""Anomaly detectors that turn a series into a per-point anomaly score (higher = more unusual).

All detectors share one interface::

    det = make_detector("seasonal", seed=0, points_per_day=24)
    det.fit(x_calibration)      # unsupervised: sees values from the calibration window only
    scores = det.score(x)       # same length as x; 0 where x is missing

Every score is *causal*: the score at time t depends only on values up to and including t
(plus parameters fitted on the calibration window), as it would in online monitoring.
Missing values (NaN) always score 0; the rule layer (``rules.py``) is responsible for them.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

MAD_TO_SIGMA = 1.4826
DETECTOR_NAMES = ("zscore", "rolling_mad", "seasonal", "ewma", "iforest", "combo")


def _nanmedian(a: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _robust_scale(a: np.ndarray, fallback: float = 1.0) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return fallback
    s = MAD_TO_SIGMA * float(np.median(np.abs(a - np.median(a))))
    if s <= 0:
        s = float(np.std(a))
    return s if s > 0 else fallback


def _trailing_windows(x: np.ndarray, w: int) -> np.ndarray:
    """Row t holds x[t-w:t] (the w values *before* t, NaN where unavailable)."""
    padded = np.concatenate([np.full(w, np.nan), x])
    return sliding_window_view(padded, w)[: x.size]


class Detector:
    """Base class: ``fit`` on calibration values, ``score`` any series causally."""

    name = "base"

    def __init__(self, points_per_day: int = 24, seed: int = 0):
        self.ppd = int(points_per_day)
        self.seed = int(seed)

    def fit(self, x_calib) -> "Detector":
        return self

    def score(self, x) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class GlobalZScore(Detector):
    """Naive baseline: |x - mean| / std with mean and std taken from the calibration window."""

    name = "zscore"

    def fit(self, x_calib):
        a = np.asarray(x_calib, dtype=float)
        self.mu = float(np.nanmean(a))
        sd = float(np.nanstd(a))
        self.sd = sd if sd > 0 else 1.0
        return self

    def score(self, x):
        a = np.asarray(x, dtype=float)
        s = np.abs(a - self.mu) / self.sd
        return np.where(np.isnan(a), 0.0, s)


class RollingMAD(Detector):
    """Rolling robust z-score: |x - rolling median| / (1.4826 * rolling MAD) over the trailing week."""

    name = "rolling_mad"

    def __init__(self, points_per_day=24, seed=0, window_days=7, min_valid_days=1, floor=0.1):
        super().__init__(points_per_day, seed)
        self.w = window_days * self.ppd
        self.min_valid = min_valid_days * self.ppd
        self.floor = floor

    def fit(self, x_calib):
        self.scale0 = _robust_scale(np.asarray(x_calib, dtype=float))
        return self

    def score(self, x):
        a = np.asarray(x, dtype=float)
        win = _trailing_windows(a, self.w)
        med = _nanmedian(win, axis=1)
        mad = _nanmedian(np.abs(win - med[:, None]), axis=1)
        n_valid = np.sum(~np.isnan(win), axis=1)
        scale = np.maximum(MAD_TO_SIGMA * mad, self.floor * self.scale0)
        s = np.abs(a - med) / scale
        s = np.where((n_valid < self.min_valid) | np.isnan(s), 0.0, s)
        return np.where(np.isnan(a), 0.0, s)


class SeasonalResidual(Detector):
    """Seasonal-decomposition residuals scored with a trailing robust scale.

    The seasonal profile for hour h is the median of the same hour over the previous
    ``profile_days`` days (so a single contaminated day cannot drag it). The residual is
    scaled by 1.4826 * MAD of recent residuals (all hours, trailing ``scale_days``).
    """

    name = "seasonal"

    def __init__(self, points_per_day=24, seed=0, profile_days=7, scale_days=14, floor=0.2):
        super().__init__(points_per_day, seed)
        self.profile_days = profile_days
        self.scale_days = scale_days
        self.floor = floor
        self.scale0 = None

    def _residuals(self, a: np.ndarray) -> np.ndarray:
        n, p = a.size, self.ppd
        stack = np.full((self.profile_days, n), np.nan)
        for j in range(1, self.profile_days + 1):
            if j * p < n:
                stack[j - 1, j * p :] = a[: n - j * p]
        n_valid = np.sum(~np.isnan(stack), axis=0)
        profile = _nanmedian(stack, axis=0)
        r = a - profile
        r[n_valid < 3] = np.nan  # not enough history yet
        return r

    def fit(self, x_calib):
        r = self._residuals(np.asarray(x_calib, dtype=float))
        self.scale0 = _robust_scale(r, fallback=_robust_scale(np.asarray(x_calib, dtype=float)))
        return self

    def signed_z(self, x) -> np.ndarray:
        """Signed standardised residual (0 where undefined); building block for EWMA / IForest."""
        a = np.asarray(x, dtype=float)
        r = self._residuals(a)
        w = self.scale_days * self.ppd
        win = _trailing_windows(r, w)
        mad = _nanmedian(np.abs(win), axis=1)
        n_valid = np.sum(~np.isnan(win), axis=1)
        scale0 = self.scale0 if self.scale0 is not None else _robust_scale(r, 1.0)
        scale = np.maximum(MAD_TO_SIGMA * mad, self.floor * scale0)
        z = r / scale
        z = np.where((n_valid < 2 * self.ppd) | np.isnan(z), 0.0, z)
        return np.where(np.isnan(a), 0.0, z)

    def score(self, x):
        return np.abs(self.signed_z(x))


class EWMAChart(Detector):
    """EWMA control chart on the seasonal residual z-scores (sensitive to small sustained shifts)."""

    name = "ewma"

    def __init__(self, points_per_day=24, seed=0, lam=0.25, clip=8.0):
        super().__init__(points_per_day, seed)
        self.lam = lam
        self.clip = clip
        self.seasonal = SeasonalResidual(points_per_day, seed)

    def fit(self, x_calib):
        self.seasonal.fit(x_calib)
        return self

    def score(self, x):
        z = np.clip(self.seasonal.signed_z(x), -self.clip, self.clip)
        e = np.empty_like(z)
        prev = 0.0
        for i, v in enumerate(z):
            prev = self.lam * v + (1 - self.lam) * prev
            e[i] = prev
        sigma = np.sqrt(self.lam / (2 - self.lam))
        s = np.abs(e) / sigma
        return np.where(np.isnan(np.asarray(x, dtype=float)), 0.0, s)


class IsolationForestDetector(Detector):
    """Isolation Forest on causal window features of the seasonal residual."""

    name = "iforest"

    def __init__(self, points_per_day=24, seed=0, n_estimators=100):
        super().__init__(points_per_day, seed)
        self.seasonal = SeasonalResidual(points_per_day, seed)
        self.n_estimators = n_estimators
        self.model = None

    def _features(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(self.seasonal.signed_z(x), -20, 20)
        # row t of each window matrix ends at t (left-padded with zeros), so features are causal
        w3 = sliding_window_view(np.concatenate([np.zeros(2), z]), 3)
        w6 = sliding_window_view(np.concatenate([np.zeros(5), z]), 6)
        z3, z6, zstd6 = w3.mean(axis=1), w6.mean(axis=1), w6.std(axis=1)
        d = np.abs(np.diff(np.concatenate([[0.0], z])))
        return np.column_stack([z, z3, z6, zstd6, d])

    def fit(self, x_calib):
        a = np.asarray(x_calib, dtype=float)
        self.seasonal.fit(a)
        feats = self._features(a)
        start = min(8 * self.ppd, max(0, a.size - 2 * self.ppd))
        usable = feats[start:]
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1
        ).fit(usable)
        return self

    def score(self, x):
        a = np.asarray(x, dtype=float)
        s = -self.model.score_samples(self._features(a))
        return np.where(np.isnan(a), 0.0, s)


class ComboDetector(Detector):
    """Seasonal residual (spikes, sharp shifts) combined with an EWMA chart (small sustained shifts).

    Each score is divided by its own 99th percentile on the calibration window (unsupervised)
    so the two are comparable, and the larger of the two is used.
    """

    name = "combo"

    def __init__(self, points_per_day=24, seed=0, q=0.99):
        super().__init__(points_per_day, seed)
        self.q = q
        self.seasonal = SeasonalResidual(points_per_day, seed)
        self.ewma = EWMAChart(points_per_day, seed)

    def fit(self, x_calib):
        a = np.asarray(x_calib, dtype=float)
        self.seasonal.fit(a)
        self.ewma.fit(a)
        skip = min(8 * self.ppd, max(0, a.size - 2 * self.ppd))
        self.norm = [max(float(np.quantile(d.score(a)[skip:], self.q)), 1e-6)
                     for d in (self.seasonal, self.ewma)]
        return self

    def score(self, x):
        return np.maximum(self.seasonal.score(x) / self.norm[0], self.ewma.score(x) / self.norm[1])


_REGISTRY = {
    "zscore": GlobalZScore,
    "rolling_mad": RollingMAD,
    "seasonal": SeasonalResidual,
    "ewma": EWMAChart,
    "iforest": IsolationForestDetector,
    "combo": ComboDetector,
}

DETECTOR_LABELS = {
    "zscore": "Global z-score (baseline)",
    "rolling_mad": "Rolling median/MAD",
    "seasonal": "Seasonal residual",
    "ewma": "EWMA control chart",
    "iforest": "Isolation Forest",
    "combo": "Seasonal residual + EWMA",
}


def make_detector(name: str, seed: int = 0, points_per_day: int = 24) -> Detector:
    if name not in _REGISTRY:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](points_per_day=points_per_day, seed=seed)
