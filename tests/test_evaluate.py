import numpy as np
import pandas as pd
import pytest

from anomaly_monitoring.evaluate import EvalConfig, bootstrap_ci, evaluate_series
from anomaly_monitoring.inject import inject_anomalies
from conftest import PPD, seasonal_series

CFG = EvalConfig(detectors=("zscore", "seasonal"), ks=(1, 2), targets=(0.8,), points_per_day=PPD)


def make_unit(seed=0):
    x = seasonal_series(n_days=120, seed=11, noise=0.4)
    r = inject_anomalies(x, seed=seed, points_per_day=PPD)
    return r.values, r.labels


def rows(x, y, seed=0):
    return pd.DataFrame(evaluate_series(x, y, split=60 * PPD, warmup=14 * PPD, seed=seed, cfg=CFG))


def test_rows_cover_the_grid_and_have_expected_columns():
    x, y = make_unit()
    df = rows(x, y)
    assert len(df) == 2 * 2 * 2  # detectors x k x with/without rule layer
    for col in ["detector", "k", "dq", "threshold", "calib_event_recall", "test_event_recall",
                "test_false_alerts_per_day", "test_pa_precision", "test_pa_recall", "test_point_fpr"]:
        assert col in df.columns


def test_deterministic_for_fixed_seed():
    x, y = make_unit()
    pd.testing.assert_frame_equal(rows(x, y, seed=3), rows(x, y, seed=3))


def test_no_label_leakage_test_window_cannot_change_calibration():
    """Replace the whole test window (values and labels): all calibration outputs must be identical."""
    x, y = make_unit()
    base = rows(x, y)
    x2, y2 = x.copy(), y.copy()
    rng = np.random.default_rng(5)
    x2[60 * PPD :] = rng.normal(100, 30, size=x2.size - 60 * PPD)
    y2[60 * PPD :] = rng.random(y2.size - 60 * PPD) < 0.3
    other = rows(x2, y2)
    for col in ["threshold", "calib_event_recall", "calib_false_alerts_per_day", "reached"]:
        np.testing.assert_array_equal(base[col].to_numpy(), other[col].to_numpy())


def test_calibration_recall_meets_target_when_reached():
    x, y = make_unit()
    df = rows(x, y)
    ok = df[df["reached"]]
    assert len(ok) > 0
    assert (ok["calib_event_recall"] >= CFG.targets[0] - 1e-12).all()


def test_rule_layer_cannot_reduce_calibration_recall():
    x, y = make_unit()
    df = rows(x, y)
    for _, g in df.groupby(["detector", "k"]):
        no, yes = g[~g["dq"]].iloc[0], g[g["dq"]].iloc[0]
        assert yes["calib_event_recall"] >= no["calib_event_recall"] - 1e-12


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    vals = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.35])
    a = bootstrap_ci(vals, seed=0)
    b = bootstrap_ci(vals, seed=0)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi
    assert mean == pytest.approx(vals.mean())
