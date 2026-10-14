"""The online (point-at-a-time) monitor must reproduce the batch detectors exactly."""
import numpy as np
import pytest

from anomaly_monitoring.alerting import apply_persistence
from anomaly_monitoring.detectors import make_detector
from anomaly_monitoring.rules import DataQualityRules
from anomaly_monitoring.streaming import STREAMING_DETECTORS, OnlineMonitor
from conftest import PPD, seasonal_series


def messy_series(seed=0):
    x = seasonal_series(n_days=45, seed=seed, noise=0.4)
    x[30 * PPD + 3] += 15.0                       # spike
    x[35 * PPD : 35 * PPD + 10] += 4.0            # level shift
    x[36 * PPD + 5 : 36 * PPD + 11] = np.nan      # missing burst
    x[40 * PPD : 40 * PPD + 12] = x[40 * PPD]     # stuck
    x[10] = np.nan                                # early NaN, inside the warm-up
    return x


@pytest.mark.parametrize("name", STREAMING_DETECTORS)
def test_online_scores_equal_batch_scores(name):
    x = messy_series()
    det = make_detector(name, seed=0, points_per_day=PPD).fit(x[: 20 * PPD])
    batch = det.score(x)
    mon = OnlineMonitor.from_detector(det, threshold=1e9)
    online = np.array([mon.update(v).score for v in x])
    np.testing.assert_allclose(online, batch, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", STREAMING_DETECTORS)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_online_flags_equal_batch_flags_with_persistence_and_rules(name, k):
    x = messy_series(seed=1)
    fit = x[: 20 * PPD]
    det = make_detector(name, seed=0, points_per_day=PPD).fit(fit)
    rules = DataQualityRules().fit(fit)
    thr = float(np.quantile(det.score(fit)[8 * PPD :], 0.97))
    batch = apply_persistence(det.score(x) > thr, k) | rules.flags(x)
    mon = OnlineMonitor.from_detector(det, threshold=thr, k=k, rules=rules)
    online = np.array([mon.update(v).flagged for v in x])
    assert (online == batch).all()


def test_alert_started_marks_the_first_point_of_each_alert_segment():
    x = messy_series()
    det = make_detector("seasonal", points_per_day=PPD).fit(x[: 20 * PPD])
    mon = OnlineMonitor.from_detector(det, threshold=3.0, k=2)
    steps = [mon.update(v) for v in x]
    flags = np.array([s.flagged for s in steps])
    starts = np.array([s.alert_started for s in steps])
    expected = flags & ~np.concatenate([[False], flags[:-1]])
    assert (starts == expected).all()
    assert starts.sum() >= 1


def test_online_state_is_bounded():
    """Memory must not grow with the stream length (rolling statistics only)."""
    x = seasonal_series(n_days=60)
    det = make_detector("seasonal", points_per_day=PPD).fit(x[: 20 * PPD])
    mon = OnlineMonitor.from_detector(det, threshold=5.0)
    for v in x[: 30 * PPD]:
        mon.update(v)
    size_mid = mon.state_size()
    for v in x[30 * PPD :]:
        mon.update(v)
    assert mon.state_size() == size_mid


def test_online_monitor_reports_rule_names():
    x = messy_series()
    fit = x[: 20 * PPD]
    det = make_detector("zscore", points_per_day=PPD).fit(fit)
    mon = OnlineMonitor.from_detector(det, threshold=1e9, rules=DataQualityRules().fit(fit))
    steps = [mon.update(v) for v in x]
    assert "missing" in steps[36 * PPD + 6].rules
    assert "stuck" in steps[40 * PPD + 11].rules


def test_unsupported_detector_is_rejected():
    det = make_detector("iforest", points_per_day=PPD).fit(seasonal_series())
    with pytest.raises(ValueError):
        OnlineMonitor.from_detector(det, threshold=1.0)
