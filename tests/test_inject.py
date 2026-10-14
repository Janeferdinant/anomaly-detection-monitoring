import numpy as np

from anomaly_monitoring.inject import EVENT_KINDS, inject_anomalies
from anomaly_monitoring.metrics import segments

from conftest import PPD, seasonal_series


def run(seed=0, n_days=120, **kw):
    x = seasonal_series(n_days=n_days, seed=99)
    return x, inject_anomalies(x, seed=seed, points_per_day=PPD, **kw)


def test_same_seed_gives_identical_output():
    _, a = run(seed=5)
    _, b = run(seed=5)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.events == b.events


def test_different_seeds_differ():
    _, a = run(seed=1)
    _, b = run(seed=2)
    assert a.events != b.events


def test_input_is_not_mutated():
    x = seasonal_series(n_days=120, seed=99)
    before = x.copy()
    inject_anomalies(x, seed=3, points_per_day=PPD)
    np.testing.assert_array_equal(x, before)


def test_every_kind_appears_in_both_halves_with_requested_count():
    _, r = run(seed=0, n_per_type=2)
    half = r.values.size // 2
    for kind in EVENT_KINDS:
        cal = [e for e in r.events if e.kind == kind and e.start < half]
        tst = [e for e in r.events if e.kind == kind and e.start >= half]
        assert len(cal) == 2 and len(tst) == 2


def test_events_do_not_overlap_keep_gap_and_respect_warmup():
    _, r = run(seed=4, warmup_days=14)
    evs = sorted(r.events, key=lambda e: e.start)
    assert evs[0].start >= 14 * PPD
    for a, b in zip(evs, evs[1:]):
        assert b.start - a.end >= PPD  # at least a day apart
    # an event never straddles the calibration/test boundary
    half = r.values.size // 2
    assert all(not (e.start < half < e.end) for e in evs)


def test_labels_match_events_exactly():
    _, r = run(seed=6)
    expected = np.zeros(r.values.size, bool)
    for e in r.events:
        expected[e.start : e.end] = True
    np.testing.assert_array_equal(r.labels, expected)
    assert segments(r.labels) == sorted((e.start, e.end) for e in r.events)


def test_values_outside_events_are_untouched():
    x, r = run(seed=7)
    np.testing.assert_array_equal(r.values[~r.labels], x[~r.labels])


def test_event_shapes():
    x, r = run(seed=8)
    for e in r.events:
        seg = r.values[e.start : e.end]
        if e.kind == "missing":
            assert np.isnan(seg).all()
        elif e.kind == "stuck":
            assert np.all(seg == seg[0]) and seg.size >= 6
        elif e.kind == "spike":
            assert e.end - e.start <= 2 and np.abs(seg - x[e.start : e.end]).max() > 0
        else:  # level_shift / drift
            assert np.abs(seg - x[e.start : e.end]).max() > 0
