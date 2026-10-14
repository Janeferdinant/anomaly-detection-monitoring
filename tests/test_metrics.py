import numpy as np
import pytest

from anomaly_monitoring.metrics import (
    event_metrics,
    false_alerts_per_day,
    point_adjust,
    point_metrics,
    segments,
)


def arr(s):
    return np.array([c == "1" for c in s])


def test_segments_basic():
    assert segments(arr("0011100110")) == [(2, 5), (7, 9)]
    assert segments(arr("0000")) == []
    assert segments(arr("1111")) == [(0, 4)]
    assert segments(arr("1")) == [(0, 1)]


def test_point_metrics_counts():
    y = arr("0011110000")
    p = arr("0001100110")
    m = point_metrics(y, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 2, 4)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(2 / 6)


def test_point_metrics_no_predictions_is_zero_not_nan():
    m = point_metrics(arr("0110"), arr("0000"))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["fpr"] == 0.0


def test_point_adjust_expands_hit_events_only():
    y = arr("0111110011100")
    p = arr("0010000000000")  # hits event 1 at one point, misses event 2
    adj = point_adjust(y, p)
    assert adj.tolist() == arr("0111110000000").tolist()


def test_point_adjust_never_removes_false_positives():
    y = arr("00110000")
    p = arr("10010001")
    adj = point_adjust(y, p)
    assert adj[0] and adj[7]  # false positives kept
    assert adj[2] and adj[3]  # event expanded


def test_point_adjusted_scoring_is_more_generous_than_raw():
    y = arr("0011111100")
    p = arr("0010000000")
    raw = point_metrics(y, p)["recall"]
    pa = point_metrics(y, point_adjust(y, p))["recall"]
    assert pa > raw and pa == 1.0


def test_event_metrics_detection_and_false_alerts():
    # events at [2,5) and [8,10); alerts: [3,4) hits event 1, [6,7) false, [8,9) hits event 2
    y = np.concatenate([arr("0011100011"), np.zeros(2, bool)])
    p = arr("000100100100")
    m = event_metrics(y, p, points_per_day=12)
    assert m["n_events"] == 2 and m["n_detected"] == 2
    assert m["event_recall"] == 1.0
    assert m["n_alerts"] == 3 and m["n_false_alerts"] == 1
    assert m["event_precision"] == pytest.approx(2 / 3)
    assert m["false_alerts_per_day"] == pytest.approx(1.0)


def test_event_metrics_missed_event():
    y = arr("0110000110")
    p = arr("0000000000")
    m = event_metrics(y, p, points_per_day=10)
    assert m["n_events"] == 2 and m["n_detected"] == 0 and m["event_recall"] == 0.0
    assert m["false_alerts_per_day"] == 0.0


def test_alert_spanning_two_events_is_not_a_false_alert():
    y = arr("0110110")
    p = arr("0111110")
    m = event_metrics(y, p, points_per_day=7)
    assert m["n_alerts"] == 1 and m["n_false_alerts"] == 0 and m["n_detected"] == 2


def test_false_alerts_per_day_scaling():
    y = np.zeros(48, bool)
    p = np.zeros(48, bool)
    p[[3, 10, 30]] = True
    assert false_alerts_per_day(y, p, points_per_day=24) == pytest.approx(1.5)


def test_no_events_event_recall_is_nan_safe():
    m = event_metrics(arr("0000"), arr("0100"), points_per_day=4)
    assert m["n_events"] == 0 and np.isnan(m["event_recall"])
    assert m["n_false_alerts"] == 1
