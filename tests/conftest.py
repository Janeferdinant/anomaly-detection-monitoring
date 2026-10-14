"""Shared toy series for the unit tests (no real data needed)."""
import numpy as np
import pytest

PPD = 24  # points per day (hourly data)


def seasonal_series(n_days=40, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * PPD)
    base = 10 + 3 * np.sin(2 * np.pi * t / PPD) + 1.5 * np.sin(4 * np.pi * t / PPD)
    return base + rng.normal(0, noise, size=t.size)


@pytest.fixture
def clean():
    return seasonal_series()


@pytest.fixture
def with_spike(clean):
    x = clean.copy()
    idx = 30 * PPD + 5
    x[idx] += 12.0
    return x, idx


@pytest.fixture
def with_shift(clean):
    x = clean.copy()
    s, e = 30 * PPD + 3, 30 * PPD + 21
    x[s:e] += 3.0
    return x, (s, e)
