"""Anomaly detectors that turn a series into a per-point anomaly score (higher = more unusual).

All detectors share one interface::

    det = make_detector("seasonal", seed=0, points_per_day=24)
    det.fit(x_fit)          # unsupervised: sees values from the fit window only
    scores = det.score(x)   # same length as x; 0 where x is missing

Every score is *causal*: the score at time t depends only on values up to and including t
(plus parameters fitted on the fit window), as it would in online monitoring. Missing values
(NaN) always score 0; the rule layer (``rules.py``) is responsible for them.

Univariate detectors take a 1-D series. For multivariate series (a matrix with one column per
metric) ``make_detector(..., n_channels=d)`` returns either a model that natively consumes all
channels (Isolation Forest, LOF, one-class SVM, PCA, autoencoder; they work on causal features
of the per-channel seasonal residuals) or a ``MultiChannel`` wrapper that runs the univariate
detector on every channel and alerts on the largest channel score.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

MAD_TO_SIGMA = 1.4826
DETECTOR_NAMES = (
    "zscore", "rolling_mad", "seasonal", "ewma", "iforest", "combo",
    "stl", "lof", "ocsvm", "pca", "autoencoder", "ensemble",
)
# detectors that are only defined for one channel and are wrapped for multivariate input
UNIVARIATE = ("zscore", "rolling_mad", "seasonal", "ewma", "combo", "stl")
# detectors whose result depends on the random seed
STOCHASTIC = ("iforest", "autoencoder", "ensemble")

Array = np.ndarray


def _nanmedian(a: Array, axis: int) -> Array:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)  # type: ignore[no-any-return]


def _robust_scale(a: Array, fallback: float = 1.0) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return fallback
    s = MAD_TO_SIGMA * float(np.median(np.abs(a - np.median(a))))
    if s <= 0:
        s = float(np.std(a))
    return s if s > 0 else fallback


def _residual_scale0(r: Array, a: Array) -> float:
    """Robust scale of fit-window residuals, kept above numerical dust relative to the raw spread."""
    s = _robust_scale(r, fallback=_robust_scale(a))
    return max(s, 1e-3 * float(np.nanstd(a)), 1e-12)


def _as_2d(x: Any) -> Array:
    a = np.asarray(x, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _trailing_windows(x: Array, w: int) -> Array:
    """Row t holds x[t-w:t] (the w values *before* t, NaN where unavailable)."""
    padded = np.concatenate([np.full(w, np.nan), x])
    return sliding_window_view(padded, w)[: x.size]  # type: ignore[no-any-return]


def trailing_median(a: Array, w: int) -> Array:
    """Median (ignoring NaN) of the ``w`` values before each position; NaN when none are valid."""
    return pd.Series(a).rolling(w, min_periods=1).median().shift(1).to_numpy(dtype=float)


def trailing_count(a: Array, w: int) -> Array:
    """Number of valid (non-NaN) values among the ``w`` values before each position."""
    c = pd.Series(a).rolling(w, min_periods=1).count().shift(1).fillna(0.0)
    return c.to_numpy(dtype=float)


def trailing_mad(a: Array, w: int, med: Array, chunk: int = 2048) -> Array:
    """Median absolute deviation of each trailing window from that window's own median."""
    n = a.size
    out = np.full(n, np.nan)
    view = _trailing_windows(a, w)
    for s in range(0, n, chunk):
        dev = np.abs(view[s : s + chunk] - med[s : s + chunk, None])
        has_nan = np.isnan(dev).any(axis=1)
        res = np.empty(dev.shape[0])
        if (~has_nan).any():  # complete windows: plain median is much faster than nanmedian
            res[~has_nan] = np.median(dev[~has_nan], axis=1)
        if has_nan.any():
            res[has_nan] = _nanmedian(dev[has_nan], axis=1)
        out[s : s + chunk] = res
    return out


class Detector:
    """Base class: ``fit`` on fit-window values, ``score`` any series causally."""

    name = "base"

    def __init__(self, points_per_day: int = 24, seed: int = 0):
        self.ppd = int(points_per_day)
        self.seed = int(seed)

    def fit(self, x_calib: Any) -> Detector:
        return self

    def score(self, x: Any) -> Array:  # pragma: no cover - interface
        raise NotImplementedError


class GlobalZScore(Detector):
    """Naive baseline: |x - mean| / std with mean and std taken from the fit window."""

    name = "zscore"

    def fit(self, x_calib: Any) -> GlobalZScore:
        a = np.asarray(x_calib, dtype=float)
        self.mu = float(np.nanmean(a))
        sd = float(np.nanstd(a))
        self.sd = sd if sd > 0 else 1.0
        return self

    def score(self, x: Any) -> Array:
        a = np.asarray(x, dtype=float)
        s = np.abs(a - self.mu) / self.sd
        return np.where(np.isnan(a), 0.0, s)


class RollingMAD(Detector):
    """Rolling robust z-score: |x - rolling median| / (1.4826 * rolling MAD) over the trailing week."""

    name = "rolling_mad"

    def __init__(self, points_per_day: int = 24, seed: int = 0, window_days: int = 7,
                 min_valid_days: int = 1, floor: float = 0.1):
        super().__init__(points_per_day, seed)
        self.w = window_days * self.ppd
        self.min_valid = min_valid_days * self.ppd
        self.floor = floor

    def fit(self, x_calib: Any) -> RollingMAD:
        self.scale0 = _robust_scale(np.asarray(x_calib, dtype=float))
        return self

    def score(self, x: Any) -> Array:
        a = np.asarray(x, dtype=float)
        med = trailing_median(a, self.w)
        mad = trailing_mad(a, self.w, med)
        n_valid = trailing_count(a, self.w)
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

    def __init__(self, points_per_day: int = 24, seed: int = 0, profile_days: int = 7,
                 scale_days: int = 14, floor: float = 0.2):
        super().__init__(points_per_day, seed)
        self.profile_days = profile_days
        self.scale_days = scale_days
        self.floor = floor
        self.scale0: float | None = None

    def _residuals(self, a: Array) -> Array:
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

    def fit(self, x_calib: Any) -> SeasonalResidual:
        a = np.asarray(x_calib, dtype=float)
        self.scale0 = _residual_scale0(self._residuals(a), a)
        return self

    def signed_z(self, x: Any) -> Array:
        """Signed standardised residual (0 where undefined); building block for EWMA / IForest."""
        a = np.asarray(x, dtype=float)
        r = self._residuals(a)
        return standardise_residual(r, a, self.scale_days * self.ppd, self.ppd, self.floor, self.scale0)

    def score(self, x: Any) -> Array:
        return np.abs(self.signed_z(x))


def standardise_residual(r: Array, a: Array, w: int, ppd: int, floor: float,
                         scale0: float | None) -> Array:
    """Scale residuals by a trailing robust scale (shared by the seasonal and STL detectors)."""
    mad = trailing_median(np.abs(r), w)
    n_valid = trailing_count(r, w)
    s0 = scale0 if scale0 is not None else _robust_scale(r, 1.0)
    scale = np.maximum(MAD_TO_SIGMA * mad, floor * s0)
    z = r / scale
    z = np.where((n_valid < 2 * ppd) | np.isnan(z), 0.0, z)
    return np.where(np.isnan(a), 0.0, z)


class STLResidual(Detector):
    """Robust STL seasonal profile plus a causal trend, scored like the seasonal residual.

    A robust STL decomposition (statsmodels, ``robust=True``) of the fit window gives the
    seasonal profile; its last full cycle is frozen as the profile. STL's own two-sided trend
    smoother would look into the future, so the trend here is the trailing median of the
    de-seasonalised series over the previous ``trend_days`` days (excluding the current point).
    The residual is scaled by a trailing robust scale as in :class:`SeasonalResidual`.
    """

    name = "stl"

    def __init__(self, points_per_day: int = 24, seed: int = 0, trend_days: int = 3,
                 scale_days: int = 14, floor: float = 0.2, seasonal_window: int = 7):
        super().__init__(points_per_day, seed)
        self.trend_days = trend_days
        self.scale_days = scale_days
        self.floor = floor
        self.seasonal_window = seasonal_window
        self.profile: Array | None = None
        self.scale0: float | None = None

    def _fit_profile(self, a: Array) -> Array:
        p = self.ppd
        s = pd.Series(a).interpolate(limit_direction="both")
        if s.isna().all():
            return np.zeros(p)
        if a.size >= 2 * p:
            from statsmodels.tsa.seasonal import STL

            tail = s.to_numpy()[-10 * p :]  # the most recent ten cycles are enough for the profile
            jump = max(1, p // 10)  # loess evaluated every `jump` points and interpolated (speed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = STL(tail, period=p, seasonal=self.seasonal_window | 1, robust=True,
                          seasonal_jump=jump, trend_jump=jump, low_pass_jump=jump).fit()
            seas = np.asarray(res.seasonal)
            profile = np.empty(p)
            n = tail.size
            for phase in range(p):  # last full cycle, aligned to the phase of each index
                last = n - 1 - ((n - 1 - phase) % p)
                profile[phase] = seas[last]
            return profile
        prof = np.array([np.nanmedian(a[ph::p]) if np.isfinite(a[ph::p]).any() else 0.0 for ph in range(p)])
        return prof - np.nanmedian(prof)

    def _residuals(self, a: Array) -> Array:
        phase = np.arange(a.size) % self.ppd
        assert self.profile is not None
        d = a - self.profile[phase]
        trend = trailing_median(d, self.trend_days * self.ppd)
        n_valid = trailing_count(d, self.trend_days * self.ppd)
        r = d - trend
        r[n_valid < self.ppd // 2] = np.nan
        return r

    def fit(self, x_calib: Any) -> STLResidual:
        a = np.asarray(x_calib, dtype=float)
        self.profile = self._fit_profile(a)
        self.scale0 = _residual_scale0(self._residuals(a), a)
        return self

    def score(self, x: Any) -> Array:
        a = np.asarray(x, dtype=float)
        r = self._residuals(a)
        z = standardise_residual(r, a, self.scale_days * self.ppd, self.ppd, self.floor, self.scale0)
        return np.abs(z)


class EWMAChart(Detector):
    """EWMA control chart on the seasonal residual z-scores (sensitive to small sustained shifts)."""

    name = "ewma"

    def __init__(self, points_per_day: int = 24, seed: int = 0, lam: float = 0.25, clip: float = 8.0):
        super().__init__(points_per_day, seed)
        self.lam = lam
        self.clip = clip
        self.seasonal = SeasonalResidual(points_per_day, seed)

    def fit(self, x_calib: Any) -> EWMAChart:
        self.seasonal.fit(x_calib)
        return self

    def score(self, x: Any) -> Array:
        z = np.clip(self.seasonal.signed_z(x), -self.clip, self.clip)
        e = np.empty_like(z)
        prev = 0.0
        for i, v in enumerate(z):
            prev = self.lam * v + (1 - self.lam) * prev
            e[i] = prev
        sigma = np.sqrt(self.lam / (2 - self.lam))
        s = np.abs(e) / sigma
        return np.where(np.isnan(np.asarray(x, dtype=float)), 0.0, s)


# --------------------------------------------------------------------------------------
# models on causal features of the seasonal residuals (one or many channels)
# --------------------------------------------------------------------------------------

def residual_features(z: Array) -> Array:
    """Causal window features of standardised residuals ``z`` (T, d) -> (T, 5d).

    Per channel: z_t, mean of the last 3, mean and std of the last 6, and |z_t - z_{t-1}|.
    Row t only uses z up to and including t (windows are left-padded with zeros).
    """
    z = np.clip(_as_2d(z), -20, 20)
    cols = []
    for j in range(z.shape[1]):
        c = z[:, j]
        w3 = sliding_window_view(np.concatenate([np.zeros(2), c]), 3)
        w6 = sliding_window_view(np.concatenate([np.zeros(5), c]), 6)
        d = np.abs(np.diff(np.concatenate([[0.0], c])))
        cols += [c, w3.mean(axis=1), w6.mean(axis=1), w6.std(axis=1), d]
    return np.column_stack(cols)


class ResidualFeatureModel(Detector):
    """Base for detectors that learn the normal region of residual features on the fit window."""

    def __init__(self, points_per_day: int = 24, seed: int = 0, max_fit_rows: int = 4000):
        super().__init__(points_per_day, seed)
        self.max_fit_rows = max_fit_rows
        self.channels: list[SeasonalResidual] = []

    def _features(self, x: Array) -> Array:
        z = np.column_stack([ch.signed_z(x[:, j]) for j, ch in enumerate(self.channels)])
        return residual_features(z)

    def _fit_model(self, f: Array) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _raw_score(self, f: Array) -> Array:  # pragma: no cover - interface
        raise NotImplementedError

    def _subsample(self, f: Array) -> Array:
        if f.shape[0] <= self.max_fit_rows:
            return f
        rng = np.random.default_rng(self.seed)
        idx = np.sort(rng.choice(f.shape[0], size=self.max_fit_rows, replace=False))
        return f[idx]

    def fit(self, x_calib: Any) -> ResidualFeatureModel:
        a = _as_2d(x_calib)
        self.channels = [SeasonalResidual(self.ppd, self.seed).fit(a[:, j]) for j in range(a.shape[1])]
        feats = self._features(a)
        start = min(8 * self.ppd, max(0, a.shape[0] - 2 * self.ppd))
        self._fit_model(feats[start:])
        return self

    def score(self, x: Any) -> Array:
        a = _as_2d(x)
        s = self._raw_score(self._features(a))
        return np.where(np.isnan(a).all(axis=1), 0.0, s)


class IsolationForestDetector(ResidualFeatureModel):
    """Isolation Forest on causal window features of the seasonal residual."""

    name = "iforest"

    def __init__(self, points_per_day: int = 24, seed: int = 0, n_estimators: int = 100,
                 max_samples: int | str = "auto"):
        super().__init__(points_per_day, seed)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.model: IsolationForest | None = None

    def _fit_model(self, f: Array) -> None:
        self.model = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                     random_state=self.seed, n_jobs=1).fit(f)

    def _raw_score(self, f: Array) -> Array:
        assert self.model is not None
        return -self.model.score_samples(f)  # type: ignore[no-any-return]


class LOFDetector(ResidualFeatureModel):
    """Local outlier factor (novelty mode) against the fit-window feature cloud."""

    name = "lof"

    def __init__(self, points_per_day: int = 24, seed: int = 0, n_neighbors: int = 20):
        super().__init__(points_per_day, seed)
        self.n_neighbors = n_neighbors
        self.model: LocalOutlierFactor | None = None

    def _fit_model(self, f: Array) -> None:
        f = self._subsample(f)
        k = int(max(1, min(self.n_neighbors, f.shape[0] - 1)))
        self.model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=1).fit(f)

    def _raw_score(self, f: Array) -> Array:
        assert self.model is not None
        return -self.model.score_samples(f)  # type: ignore[no-any-return]


class OneClassSVMDetector(ResidualFeatureModel):
    """One-class SVM (RBF kernel) trained on the fit-window features."""

    name = "ocsvm"

    def __init__(self, points_per_day: int = 24, seed: int = 0, nu: float = 0.05,
                 max_fit_rows: int = 3000):
        super().__init__(points_per_day, seed, max_fit_rows=max_fit_rows)
        self.nu = nu
        self.model: OneClassSVM | None = None
        self.offset = 0.0

    def _fit_model(self, f: Array) -> None:
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=self.nu).fit(self._subsample(f))
        # -decision_function is negative for inliers; shift it so that the most normal fit-window
        # point scores 0 and every score is >= 0 (missing values score 0 and must stay the lowest)
        self.offset = float(np.min(-self.model.decision_function(f)))

    def _raw_score(self, f: Array) -> Array:
        assert self.model is not None
        return np.maximum(-self.model.decision_function(f) - self.offset, 0.0)


class _Standardiser:
    """Mean/std scaling fitted on the fit window (std floored so flat features stay finite)."""

    def fit(self, f: Array) -> _Standardiser:
        self.mu = f.mean(axis=0)
        self.sd = np.maximum(f.std(axis=0), 0.1)  # features are z-scores: a floor of 0.1 sigma
        return self

    def transform(self, f: Array) -> Array:
        return (f - self.mu) / self.sd  # type: ignore[no-any-return]


class PCADetector(ResidualFeatureModel):
    """PCA reconstruction error: squared distance of a feature vector from the principal subspace."""

    name = "pca"

    def __init__(self, points_per_day: int = 24, seed: int = 0, variance: float = 0.9):
        super().__init__(points_per_day, seed)
        self.variance = variance
        self.model: PCA | None = None
        self.scaler = _Standardiser()

    def _fit_model(self, f: Array) -> None:
        g = self.scaler.fit(f).transform(f)
        n_comp = self.variance if g.shape[1] > 1 else 1
        if g.shape[1] > 1 and g.shape[0] <= g.shape[1]:
            n_comp = max(1, g.shape[0] - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # zero-variance toy windows
            self.model = PCA(n_components=n_comp, svd_solver="full").fit(g)

    def _raw_score(self, f: Array) -> Array:
        assert self.model is not None
        g = self.scaler.transform(f)
        rec = self.model.inverse_transform(self.model.transform(g))
        return np.sum((g - rec) ** 2, axis=1)  # type: ignore[no-any-return]


class AutoencoderDetector(ResidualFeatureModel):
    """Small dense autoencoder (scikit-learn MLP, CPU): mean squared reconstruction error.

    Architecture ``in -> h -> b -> h -> in`` with ReLU units, Adam, trained on the fit-window
    features for at most ``max_iter`` epochs (training stops early when the loss plateaus).
    """

    name = "autoencoder"

    def __init__(self, points_per_day: int = 24, seed: int = 0, hidden: int = 32,
                 bottleneck: int = 8, max_iter: int = 120):
        super().__init__(points_per_day, seed)
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.max_iter = max_iter
        self.model: MLPRegressor | None = None
        self.scaler = _Standardiser()

    def _fit_model(self, f: Array) -> None:
        g = self.scaler.fit(f).transform(f)
        b = max(1, min(self.bottleneck, g.shape[1] - 1)) if g.shape[1] > 1 else 1
        h = max(b, self.hidden)
        self.model = MLPRegressor(hidden_layer_sizes=(h, b, h), activation="relu", solver="adam",
                                  learning_rate_init=1e-3, max_iter=self.max_iter,
                                  batch_size=min(128, max(8, g.shape[0])), random_state=self.seed,
                                  tol=1e-4, n_iter_no_change=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(g, g)

    def _raw_score(self, f: Array) -> Array:
        assert self.model is not None
        g = self.scaler.transform(f)
        return np.mean((g - self.model.predict(g)) ** 2, axis=1)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------------------
# combinations
# --------------------------------------------------------------------------------------

def _warmup_skip(n: int, ppd: int) -> int:
    return min(8 * ppd, max(0, n - 2 * ppd))


class ComboDetector(Detector):
    """Seasonal residual (spikes, sharp shifts) combined with an EWMA chart (small sustained shifts).

    Each score is divided by its own 99th percentile on the fit window (unsupervised)
    so the two are comparable, and the larger of the two is used.
    """

    name = "combo"

    def __init__(self, points_per_day: int = 24, seed: int = 0, q: float = 0.99):
        super().__init__(points_per_day, seed)
        self.q = q
        self.seasonal = SeasonalResidual(points_per_day, seed)
        self.ewma = EWMAChart(points_per_day, seed)

    def fit(self, x_calib: Any) -> ComboDetector:
        a = np.asarray(x_calib, dtype=float)
        self.seasonal.fit(a)
        self.ewma.fit(a)
        skip = _warmup_skip(a.size, self.ppd)
        self.norm = [max(float(np.quantile(d.score(a)[skip:], self.q)), 1e-3)
                     for d in (self.seasonal, self.ewma)]
        return self

    def score(self, x: Any) -> Array:
        return np.maximum(self.seasonal.score(x) / self.norm[0], self.ewma.score(x) / self.norm[1])


def combine_votes(scores: Sequence[Array], norms: Sequence[float], min_votes: int) -> Array:
    """Soft vote: ``min_votes``-th largest of the normalised member scores.

    At threshold theta the result exceeds theta exactly when at least ``min_votes`` members
    have ``score / norm > theta``, i.e. a vote of members that each use their own scaled threshold.
    """
    m = np.vstack([np.asarray(s, dtype=float) / n for s, n in zip(scores, norms, strict=True)])
    order = np.sort(m, axis=0)[::-1]
    return order[min(min_votes, m.shape[0]) - 1]  # type: ignore[no-any-return]


class VotingEnsemble(Detector):
    """Voting ensemble: an alert needs ``min_votes`` of the member detectors to agree.

    Members are scaled by their own 99th-percentile score on the fit window. Persistence (k
    consecutive flagged points) is applied afterwards by the alerting layer, like for any detector.
    """

    name = "ensemble"

    def __init__(self, points_per_day: int = 24, seed: int = 0, members: Sequence[str] | None = None,
                 member_params: dict[str, dict[str, Any]] | None = None, min_votes: int = 2,
                 q: float = 0.99, n_channels: int = 1):
        super().__init__(points_per_day, seed)
        self.member_names = tuple(members or ENSEMBLE_MEMBERS)
        self.member_params = member_params or {}
        self.min_votes = min_votes
        self.q = q
        self.n_channels = n_channels
        self.members = [make_detector(n, seed, points_per_day, self.member_params.get(n), n_channels)
                        for n in self.member_names]
        self.norms: list[float] = []

    def fit(self, x_calib: Any) -> VotingEnsemble:
        a = np.asarray(x_calib, dtype=float)
        skip = _warmup_skip(a.shape[0], self.ppd)
        self.norms = []
        for m in self.members:
            m.fit(a)
            self.norms.append(max(float(np.quantile(m.score(a)[skip:], self.q)), 1e-3))
        return self

    def score(self, x: Any) -> Array:
        return combine_votes([m.score(x) for m in self.members], self.norms, self.min_votes)


ENSEMBLE_MEMBERS = ("seasonal", "ewma", "iforest", "pca", "lof")


class MultiChannel(Detector):
    """Run a univariate detector on every channel and alert on the largest normalised score.

    Each channel's score is divided by its own ``q``-quantile on the fit window (at least 1, so
    flat channels cannot blow up), which makes channels comparable; the maximum over channels is
    the score. This is the usual "any metric breaches its rule" monitoring logic.
    """

    def __init__(self, factory: Callable[[], Detector], points_per_day: int = 24, seed: int = 0,
                 q: float = 0.995, name: str = "multichannel"):
        super().__init__(points_per_day, seed)
        self.factory = factory
        self.q = q
        self.name = name
        self.dets: list[Detector] = []
        self.norms: list[float] = []

    def fit(self, x_calib: Any) -> MultiChannel:
        a = _as_2d(x_calib)
        skip = _warmup_skip(a.shape[0], self.ppd)
        self.dets, self.norms = [], []
        for j in range(a.shape[1]):
            d = self.factory().fit(a[:, j])
            self.dets.append(d)
            self.norms.append(max(float(np.quantile(d.score(a[:, j])[skip:], self.q)), 1.0))
        return self

    def score(self, x: Any) -> Array:
        a = _as_2d(x)
        s = np.max(np.vstack([d.score(a[:, j]) / n for j, (d, n) in enumerate(zip(self.dets, self.norms, strict=True))]),
                   axis=0)
        return np.where(np.isnan(a).all(axis=1), 0.0, s)


_REGISTRY: dict[str, Callable[..., Detector]] = {
    "zscore": GlobalZScore,
    "rolling_mad": RollingMAD,
    "seasonal": SeasonalResidual,
    "ewma": EWMAChart,
    "iforest": IsolationForestDetector,
    "combo": ComboDetector,
    "stl": STLResidual,
    "lof": LOFDetector,
    "ocsvm": OneClassSVMDetector,
    "pca": PCADetector,
    "autoencoder": AutoencoderDetector,
    "ensemble": VotingEnsemble,
}

DETECTOR_LABELS = {
    "zscore": "Global z-score (baseline)",
    "rolling_mad": "Rolling median/MAD",
    "seasonal": "Seasonal residual",
    "ewma": "EWMA control chart",
    "iforest": "Isolation Forest",
    "combo": "Seasonal residual + EWMA",
    "stl": "Robust STL residual",
    "lof": "Local outlier factor",
    "ocsvm": "One-class SVM",
    "pca": "PCA reconstruction error",
    "autoencoder": "Autoencoder",
    "ensemble": "Voting ensemble",
}

# Small hyper-parameter grids, tuned on calibration windows only (see ``tuning.py``).
# The first entry of every grid is the default of the class.
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "zscore": [{}],
    "rolling_mad": [{}, {"window_days": 3}, {"window_days": 1}],
    "seasonal": [{}, {"profile_days": 3}, {"scale_days": 7}, {"profile_days": 3, "scale_days": 7}],
    "ewma": [{}, {"lam": 0.1}, {"lam": 0.5}],
    "combo": [{}],
    "stl": [{}, {"trend_days": 1}],
    "iforest": [{}, {"n_estimators": 200, "max_samples": 1024}],
    "lof": [{}, {"n_neighbors": 50}],
    "ocsvm": [{}, {"nu": 0.01}],
    "pca": [{}, {"variance": 0.8}],
    "autoencoder": [{}, {"hidden": 16, "bottleneck": 4}],
    "ensemble": [{}, {"min_votes": 3}],
}


def make_detector(name: str, seed: int = 0, points_per_day: int = 24,
                  params: dict[str, Any] | None = None, n_channels: int = 1) -> Detector:
    """Build a detector by name. ``n_channels > 1`` selects the multivariate variant."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(_REGISTRY)}")
    params = dict(params or {})
    if n_channels > 1 and name in UNIVARIATE:
        cls = _REGISTRY[name]
        return MultiChannel(lambda: cls(points_per_day=points_per_day, seed=seed, **params),
                            points_per_day, seed, name=name)
    if name == "ensemble":
        return VotingEnsemble(points_per_day=points_per_day, seed=seed, n_channels=n_channels, **params)
    return _REGISTRY[name](points_per_day=points_per_day, seed=seed, **params)
