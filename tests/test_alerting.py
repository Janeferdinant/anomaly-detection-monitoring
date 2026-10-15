import numpy as np
import pytest

from anomaly_monitoring.alerting import apply_persistence, calibrate_threshold


def arr(s):
    return np.array([c == "1" for c in s])


def test_persistence_k1_is_identity():
    f = arr("01101")
    assert apply_persistence(f, 1).tolist() == f.tolist()


def test_persistence_requires_k_consecutive_and_is_causal():
    f = arr("0110111100")
    assert apply_persistence(f, 3).tolist() == arr("0000001100").tolist()


def test_persistence_removes_isolated_blips():
    f = arr("0100100100")
    assert not apply_persistence(f, 2).any()


def test_persistence_rejects_bad_k():
    with pytest.raises(ValueError):
        apply_persistence(arr("01"), 0)


def _toy_scores():
    # 3 events of length 2 with high scores; low background with two medium blips
    n = 100
    labels = np.zeros(n, bool)
    scores = np.full(n, 0.1)
    for s in (10, 40, 70):
        labels[s : s + 2] = True
        scores[s : s + 2] = 5.0
    scores[[20, 55]] = 2.0
    return scores, labels


def test_calibrate_picks_highest_threshold_meeting_target_recall():
    scores, labels = _toy_scores()
    cal = calibrate_threshold(scores, labels, k=1, target_recall=1.0, points_per_day=24)
    assert cal["reached"] is True
    flags = scores > cal["threshold"]
    assert flags[labels].all() and not flags[~labels].any()


def test_calibrate_reports_unreachable_target():
    scores, labels = _toy_scores()
    scores[10:12] = 0.0  # event invisible to the detector at any threshold
    cal = calibrate_threshold(scores, labels, k=1, target_recall=1.0, points_per_day=24)
    assert cal["reached"] is False
    assert cal["recall"] == pytest.approx(2 / 3)


def test_calibrate_with_persistence_never_increases_false_alerts_here():
    scores, labels = _toy_scores()
    c1 = calibrate_threshold(scores, labels, k=1, target_recall=1.0, points_per_day=24)
    c2 = calibrate_threshold(scores, labels, k=2, target_recall=1.0, points_per_day=24)
    assert c2["reached"]
    assert c2["false_alerts_per_day"] <= c1["false_alerts_per_day"]


def test_calibration_is_a_pure_function_of_the_calibration_slice():
    """No leakage: callers pass only the calibration window; later data cannot matter."""
    scores, labels = _toy_scores()
    a = calibrate_threshold(scores, labels, 1, 0.9, 24)["threshold"]
    rng = np.random.default_rng(0)
    later_scores = np.concatenate([scores, rng.normal(size=50) * 9])
    later_labels = np.concatenate([labels, np.ones(50, bool)])
    b = calibrate_threshold(later_scores[: scores.size], later_labels[: labels.size], 1, 0.9, 24)["threshold"]
    assert a == b


def test_calibrate_extra_flags_count_toward_recall():
    scores, labels = _toy_scores()
    scores[10:12] = 0.0
    extra = np.zeros_like(labels)
    extra[10] = True  # the rule layer catches the invisible event
    cal = calibrate_threshold(scores, labels, 1, 1.0, 24, extra_flags=extra)
    assert cal["reached"] is True


def test_budget_threshold_respects_the_alert_budget_and_is_label_free():
    from anomaly_monitoring.alerting import threshold_for_budget
    from anomaly_monitoring.metrics import alerts_per_day

    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(0, 1, size=24 * 40))
    for budget in (0.5, 1.0, 3.0):
        res = threshold_for_budget(s, k=1, budget_per_day=budget, points_per_day=24)
        assert res["met"]
        flags = apply_persistence(s > res["threshold"], 1)
        assert alerts_per_day(flags, 24) <= budget + 1e-12
    # a larger budget never gives a higher threshold
    t = [threshold_for_budget(s, 1, b, 24)["threshold"] for b in (0.5, 1.0, 3.0)]
    assert t[0] >= t[1] >= t[2]


def test_budget_threshold_when_rule_alerts_alone_exceed_the_budget():
    from anomaly_monitoring.alerting import threshold_for_budget

    s = np.ones(24 * 10)
    extra = np.zeros(s.size, dtype=bool)
    extra[::5] = True  # about 5 alerts per day from the rule layer alone
    res = threshold_for_budget(s, 1, 1.0, 24, extra_flags=extra)
    assert not res["met"] and res["threshold"] == float("inf")
