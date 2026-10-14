"""Seeded, documented anomaly injection for data-quality and monitoring benchmarks.

The public sensor data has no ground-truth incident labels, so labelled incidents are
injected by a small generator that is fully determined by ``seed``. Five incident types
cover typical data-quality / monitoring failures:

============== =================================================== ========================
kind           what is done to the clean series                    duration (hourly data)
============== =================================================== ========================
spike          1-2 points shifted by +/- 5-10 sigma                1-2 points
level_shift    constant offset of +/- 3-6 sigma, then back         6-24 points
drift          linear ramp up to +/- 4-8 sigma, then abrupt reset  24-72 points
stuck          value frozen at the last good reading               6-24 points
missing        a burst of missing values (NaN)                     3-12 points
============== =================================================== ========================

``sigma`` is the robust scale (1.4826 * MAD) of the residual from the hour-of-day median
of the clean series, so magnitudes are comparable across channels. Events never overlap,
are at least ``min_gap_days`` apart, never straddle the calibration/test boundary (the
middle of the series) and never fall in the first ``warmup_days`` of the series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVENT_KINDS = ("spike", "level_shift", "drift", "stuck", "missing")

# (min, max) duration in hours; scaled by points_per_day / 24
_DURATION_H = {
    "spike": (1, 2),
    "level_shift": (6, 24),
    "drift": (24, 72),
    "stuck": (6, 24),
    "missing": (3, 12),
}
_MAGNITUDE_SIGMA = {
    "spike": (5.0, 10.0),
    "level_shift": (3.0, 6.0),
    "drift": (4.0, 8.0),
}


@dataclass(frozen=True)
class Event:
    kind: str
    start: int
    end: int  # exclusive
    magnitude: float = 0.0  # signed, in sigma units (0 for stuck / missing)


@dataclass
class InjectionResult:
    values: np.ndarray
    labels: np.ndarray
    events: list
    sigma: float


def robust_sigma(x: np.ndarray, points_per_day: int) -> float:
    """Robust scale of the residual from the hour-of-day median."""
    x = np.asarray(x, dtype=float)
    hours = np.arange(x.size) % points_per_day
    prof = np.array([np.nanmedian(x[hours == h]) for h in range(points_per_day)])
    r = x - prof[hours]
    r = r[~np.isnan(r)]
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(s if s > 0 else (np.std(r) or 1.0))


def _place(rng, kinds, lo, hi, ppd, min_gap, occupied):
    placed = []
    for kind in kinds:
        d_lo, d_hi = _DURATION_H[kind]
        d_lo = max(1, int(round(d_lo * ppd / 24)))
        d_hi = max(d_lo, int(round(d_hi * ppd / 24)))
        for _ in range(2000):
            dur = int(rng.integers(d_lo, d_hi + 1))
            if hi - lo - dur < 1:
                continue
            start = int(rng.integers(lo, hi - dur))
            end = start + dur
            if all(end + min_gap <= s or start >= e + min_gap for s, e in occupied):
                occupied.append((start, end))
                placed.append((kind, start, end))
                break
        else:  # pragma: no cover - only for pathologically short series
            raise RuntimeError("could not place all events; use a longer series or fewer events")
    return placed


def inject_anomalies(
    values,
    seed: int,
    points_per_day: int = 24,
    n_per_type: int = 2,
    warmup_days: int = 14,
    min_gap_days: int = 1,
) -> InjectionResult:
    """Return a copy of ``values`` with labelled incidents injected (deterministic in ``seed``)."""
    x = np.asarray(values, dtype=float).copy()
    clean = x.copy()
    n = x.size
    ppd = int(points_per_day)
    rng = np.random.default_rng(seed)
    sigma = robust_sigma(clean, ppd)
    half = n // 2
    min_gap = min_gap_days * ppd

    occupied: list[tuple[int, int]] = []
    placed = []
    for lo, hi in ((warmup_days * ppd, half), (half, n - 1)):
        kinds = list(EVENT_KINDS) * n_per_type
        rng.shuffle(kinds)
        placed += _place(rng, kinds, lo, hi, ppd, min_gap, occupied)

    events = []
    for kind, s, e in sorted(placed, key=lambda t: t[1]):
        mag = 0.0
        if kind in _MAGNITUDE_SIGMA:
            m_lo, m_hi = _MAGNITUDE_SIGMA[kind]
            mag = float(rng.uniform(m_lo, m_hi) * rng.choice([-1.0, 1.0]))
        if kind == "spike":
            x[s:e] = clean[s:e] + mag * sigma
        elif kind == "level_shift":
            x[s:e] = clean[s:e] + mag * sigma
        elif kind == "drift":
            ramp = np.arange(1, e - s + 1) / (e - s)
            x[s:e] = clean[s:e] + mag * sigma * ramp
        elif kind == "stuck":
            x[s:e] = clean[s - 1]
        elif kind == "missing":
            x[s:e] = np.nan
        events.append(Event(kind, s, e, mag))

    labels = np.zeros(n, dtype=bool)
    for ev in events:
        labels[ev.start : ev.end] = True
    return InjectionResult(values=x, labels=labels, events=events, sigma=sigma)
