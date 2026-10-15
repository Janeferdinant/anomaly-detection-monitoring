"""Scoring: point-level, point-adjusted and event-level metrics.

Conventions
-----------
* ``y_true`` / ``y_pred`` are boolean arrays of equal length (one entry per time step).
* An *event* is a maximal run of True in ``y_true``; an *alert* is a maximal run of True
  in ``y_pred``.
* A *false alert* is an alert that overlaps no true event. ``false_alerts_per_day`` is the
  headline false-positive measure used in this repository.
* Point-adjusted (PA) scoring (Xu et al., 2018) credits a whole event as detected if any
  single point inside it is flagged. It is widely used but generous, so the event-level
  numbers are reported next to it.
"""
from __future__ import annotations

import numpy as np


def _segment_arrays(mask) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def segments(mask) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), ...]`` for each run of True in ``mask``."""
    starts, ends = _segment_arrays(mask)
    return list(zip(starts.tolist(), ends.tolist()))


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def point_metrics(y_true, y_pred) -> dict:
    """Raw point-level confusion counts, precision, recall, F1 and false-positive rate."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": _div(fp, fp + tn),
    }


def point_adjust(y_true, y_pred) -> np.ndarray:
    """Point-adjust predictions: any hit inside an event marks the whole event as flagged.

    False positives outside events are kept unchanged, and missed events stay missed.
    """
    y = np.asarray(y_true, dtype=bool)
    adj = np.asarray(y_pred, dtype=bool).copy()
    for s, e in segments(y):
        if adj[s:e].any():
            adj[s:e] = True
    return adj


def event_metrics(y_true, y_pred, points_per_day: float) -> dict:
    """Event-level metrics: event recall, alert precision and false alerts per day."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    es, ee = _segment_arrays(y)
    as_, ae = _segment_arrays(p)
    zero = np.zeros(1, dtype=np.int64)
    p_cum = np.concatenate([zero, np.cumsum(p, dtype=np.int64)])
    y_cum = np.concatenate([zero, np.cumsum(y, dtype=np.int64)])
    n_events = int(es.size)
    n_detected = int(np.sum(p_cum[ee] - p_cum[es] > 0))
    n_false = int(np.sum(y_cum[ae] - y_cum[as_] == 0))
    n_alerts = int(as_.size)
    days = y.size / float(points_per_day)
    return {
        "n_events": n_events,
        "n_detected": n_detected,
        "event_recall": n_detected / n_events if n_events else float("nan"),
        "n_alerts": n_alerts,
        "n_false_alerts": n_false,
        "event_precision": (n_alerts - n_false) / n_alerts if n_alerts else 0.0,
        "false_alerts_per_day": n_false / days if days > 0 else 0.0,
    }


def time_to_detect(y_true, y_pred) -> np.ndarray:
    """Delay, in points, from the start of each event to its first flagged point.

    One entry per event (in time order); NaN for events with no flagged point. A delay of 0
    means a flag was already raised when the event started (or at its first point).
    """
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    es, ee = _segment_arrays(y)
    n = p.size
    idx = np.where(p, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]  # nxt[i]: first flagged index >= i
    first = nxt[es] if es.size else np.zeros(0, dtype=np.int64)
    delay = np.where(first < ee, first - es, np.nan) if es.size else np.zeros(0)
    return np.asarray(delay, dtype=float)


def alerts_per_day(y_pred, points_per_day: float) -> float:
    """Number of alerts (maximal runs of flagged points) per day, whatever they overlap."""
    p = np.asarray(y_pred, dtype=bool)
    as_, _ = _segment_arrays(p)
    days = p.size / float(points_per_day)
    return float(as_.size) / days if days > 0 else 0.0


def false_alerts_per_day(y_true, y_pred, points_per_day: float) -> float:
    """Number of alerts that overlap no true event, divided by the length in days."""
    return event_metrics(y_true, y_pred, points_per_day)["false_alerts_per_day"]
