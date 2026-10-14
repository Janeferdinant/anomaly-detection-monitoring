import numpy as np
import pytest

from anomaly_monitoring.detectors import DETECTOR_NAMES, make_detector

from conftest import PPD, seasonal_series

WARM = 14 * PPD  # ignore the first two weeks (no history for rolling statistics)


def fitted(name, x, seed=0):
    det = make_detector(name, seed=seed, points_per_day=PPD)
    det.fit(x[: 20 * PPD])
    return det


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_scores_have_series_length_and_are_finite(name, clean):
    s = fitted(name, clean).score(clean)
    assert s.shape == clean.shape
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_spike_has_the_top_score(name, with_spike):
    x, idx = with_spike
    s = fitted(name, x).score(x)
    ranked = np.argsort(-s[WARM:])[:3] + WARM
    assert idx in ranked


@pytest.mark.parametrize("name", ["seasonal", "ewma", "iforest", "combo"])
def test_level_shift_scores_higher_inside_than_outside(name, with_shift):
    x, (a, b) = with_shift
    s = fitted(name, x).score(x)
    outside = np.concatenate([s[WARM:a], s[b + 24 :]])
    # more than half of the shifted points outrank 90% of the normal points
    assert (s[a:b] > np.quantile(outside, 0.9)).mean() > 0.5


def test_rolling_mad_catches_a_shift_on_a_non_seasonal_series():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, size=40 * PPD)
    a, b = 30 * PPD, 30 * PPD + 12
    x[a:b] += 6.0
    det = make_detector("rolling_mad", seed=0, points_per_day=PPD).fit(x[: 20 * PPD])
    s = det.score(x)
    outside = np.concatenate([s[WARM:a], s[b:]])
    assert (s[a:b] > np.quantile(outside, 0.99)).mean() > 0.5


def test_seasonal_residual_separates_a_shift_that_zscore_cannot(with_shift):
    """A +3 shift hides inside the normal daily swing for a global z-score."""
    x, (a, b) = with_shift
    zs = fitted("zscore", x).score(x)
    ss = fitted("seasonal", x).score(x)
    others = np.concatenate([np.arange(WARM, a), np.arange(b, x.size)])
    # seasonal: every shifted point scores above 95% of normal points
    assert (ss[a:b] > np.quantile(ss[others], 0.95)).mean() > 0.9
    # z-score: most shifted points sit inside the normal range of scores
    assert (zs[a:b] > np.quantile(zs[others], 0.95)).mean() < 0.5


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_missing_values_score_zero_and_do_not_break(name, clean):
    x = clean.copy()
    x[25 * PPD : 25 * PPD + 6] = np.nan
    s = fitted(name, x).score(x)
    assert np.isfinite(s).all()
    assert (s[25 * PPD : 25 * PPD + 6] == 0).all()


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_scores_are_causal(name, clean):
    """Changing the future must not change any earlier score."""
    t0 = 30 * PPD
    det = fitted(name, clean)
    s1 = det.score(clean)
    changed = clean.copy()
    changed[t0:] += np.random.default_rng(3).normal(0, 5, size=clean.size - t0)
    s2 = det.score(changed)
    np.testing.assert_allclose(s1[:t0], s2[:t0])


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_deterministic_for_fixed_seed(name, with_spike):
    x, _ = with_spike
    a = fitted(name, x, seed=7).score(x)
    b = fitted(name, x, seed=7).score(x)
    np.testing.assert_array_equal(a, b)


def test_zscore_uses_calibration_statistics_only():
    rng = np.random.default_rng(0)
    calib = rng.normal(5, 2, size=500)
    det = make_detector("zscore", seed=0, points_per_day=PPD).fit(calib)
    x = np.array([5.0, 9.0, 1.0, 11.0])
    np.testing.assert_allclose(det.score(x), np.abs(x - calib.mean()) / calib.std(), rtol=1e-9)


def test_unknown_detector_name_raises():
    with pytest.raises(ValueError):
        make_detector("nope", seed=0, points_per_day=PPD)


def test_series_without_history_does_not_crash():
    x = seasonal_series(n_days=3)
    for name in DETECTOR_NAMES:
        det = make_detector(name, seed=0, points_per_day=PPD).fit(x)
        assert np.isfinite(det.score(x)).all()
