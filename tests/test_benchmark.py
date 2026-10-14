import json

import numpy as np
import pandas as pd
import pytest

from anomaly_monitoring import benchmark as B
from anomaly_monitoring import opsreport as R
from anomaly_monitoring import smd
from anomaly_monitoring.inject import inject_anomalies
from conftest import PPD, seasonal_series

METHODS = ("zscore", "seasonal", "iforest", "pca")


def make_unit(seed=0, channels=1, name="toy", days=120):
    base = seasonal_series(n_days=days, seed=11, noise=0.4)
    if channels == 1:
        res = inject_anomalies(base, seed=seed, points_per_day=PPD)
        values, labels = res.values, res.labels
    else:
        res = inject_anomalies(base, seed=seed, points_per_day=PPD)
        rng = np.random.default_rng(1)
        values = np.column_stack([res.values] + [seasonal_series(n_days=days, seed=20 + j, noise=0.4)
                                                 + rng.normal(0, 0.01, base.size) for j in range(channels - 1)])
        labels = res.labels
    half = base.size // 2
    return B.Unit(name, "toy", values, labels, fit_end=half, cal=(14 * PPD, half), test=(half, base.size),
                  ppd=PPD, minutes_per_point=60.0, seed=seed)


def run(unit, chosen=None, rule_params=None):
    return B.evaluate([unit], [0], chosen or {}, primary_target=0.8, methods=METHODS, jobs=1, rule_params=rule_params)


def test_rows_cover_the_grid_and_have_the_expected_columns():
    m, b = run(make_unit())
    per_method = len(B.TARGETS) * len(B.KS) * len(B.MAIN_RULES) + len(B.ABLATION_RULES)
    assert len(m) == len(METHODS) * per_method
    assert len(b) == len(METHODS) * len(B.BUDGETS) * len(B.KS) * len(B.MAIN_RULES)
    for col in ["method", "target", "k", "rules", "threshold", "reached", "test_false_alerts_per_day",
                "test_event_recall", "test_ttd_median_min", "test_event_precision"]:
        assert col in m.columns
    for col in ["budget", "budget_met", "cal_alerts_per_day", "test_alerts_per_day"]:
        assert col in b.columns


def test_no_leakage_changing_the_test_window_cannot_change_calibration_outputs():
    u = make_unit()
    m1, b1 = run(u)
    u2 = make_unit()
    rng = np.random.default_rng(5)
    t0, t1 = u2.test
    u2.values[t0:t1] += rng.normal(0, 3, size=t1 - t0)
    u2.labels[t0:t1] = rng.random(t1 - t0) < 0.2
    m2, b2 = run(u2)
    for col in ["threshold", "reached", "cal_event_recall", "cal_false_alerts_per_day"]:
        pd.testing.assert_series_equal(m1[col], m2[col])
    for col in ["threshold", "budget_met", "cal_alerts_per_day"]:
        pd.testing.assert_series_equal(b1[col], b2[col])
    assert not m1["test_false_alerts_per_day"].equals(m2["test_false_alerts_per_day"])


def test_alert_budget_is_respected_on_the_calibration_window():
    _, b = run(make_unit())
    ok = b[b["budget_met"]]
    assert len(ok) > 0
    assert (ok["cal_alerts_per_day"] <= ok["budget"] + 1e-9).all()


def test_higher_budget_never_lowers_test_recall_much_for_the_baseline():
    _, b = run(make_unit())
    d = b[(b["method"] == "zscore") & (b["k"] == 1) & (b["rules"] == "none")].sort_values("budget")
    assert d["test_event_recall"].iloc[-1] >= d["test_event_recall"].iloc[0]


def test_multichannel_unit_runs_through_every_path():
    u = make_unit(channels=3)
    m, b = run(u)
    assert len(m) > 0 and len(b) > 0
    assert np.isfinite(m["test_false_alerts_per_day"]).all()


def test_tuning_uses_only_calibration_windows_and_returns_a_grid_entry():
    u = make_unit()
    chosen, summary = B.tune([u], methods=("zscore", "seasonal", "pca", "ensemble"), target=0.65)
    u2 = make_unit()
    t0, t1 = u2.test
    u2.values[t0:t1] = 1e6  # destroy the test window
    chosen2, _ = B.tune([u2], methods=("zscore", "seasonal", "pca", "ensemble"), target=0.65)
    assert chosen == chosen2
    assert set(chosen) == {"zscore", "seasonal", "pca", "ensemble"}
    assert summary["chosen"].sum() == 4


def test_rule_tuning_picks_one_grid_entry():
    params, log = B.tune_rules([make_unit()], 0.8)
    assert params in B.RULE_GRID and log["chosen"].sum() == 1


def test_units_without_enough_calibration_events_are_flagged():
    u = make_unit()
    u.labels[u.cal[0]:u.cal[1]] = False
    assert not u.matched_ok and u.budget_ok
    m, b = run(u)
    assert m.empty and not b.empty


def test_deterministic_for_fixed_seed():
    a, _ = run(make_unit(seed=2))
    b, _ = run(make_unit(seed=2))
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- SMD loader


def test_smd_manifest_lists_every_file_with_a_sha256():
    manifest = smd.load_manifest()
    assert set(manifest) == set(smd.file_keys())
    assert len(manifest) == 28 * 3
    assert all(len(v) == 64 for v in manifest.values())


def test_smd_downsampling_averages_values_and_ors_labels():
    a = np.arange(20, dtype=float).reshape(10, 2)
    d = smd.downsample_mean(a, 5)
    assert d.shape == (2, 2) and d[0, 0] == 4.0 and d[1, 1] == 15.0
    y = np.zeros(11, dtype=bool)
    y[7] = True
    assert smd.downsample_any(y, 5).tolist() == [False, True]


def test_smd_load_machine_from_files(tmp_path):
    for split, arr in (("train", np.ones((20, 3))), ("test", np.ones((20, 3)) * 2)):
        (tmp_path / split).mkdir()
        np.savetxt(tmp_path / split / "m.txt", arr, delimiter=",")
    (tmp_path / "test_label").mkdir()
    lab = np.zeros(20, dtype=int)
    lab[12:14] = 1
    np.savetxt(tmp_path / "test_label" / "m.txt", lab, fmt="%d")
    m = smd.load_machine("m", raw_dir=tmp_path, factor=5)
    assert m.train.shape == (4, 3) and m.test.shape == (4, 3)
    assert m.labels.tolist() == [False, False, True, False]


def test_smd_checksum_mismatch_is_detected(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("abc")
    with pytest.raises(smd.ChecksumError):
        smd._fetch_one("x.txt", tmp_path, "0" * 64)
    assert smd._fetch_one("x.txt", tmp_path, smd.sha256_of(f))[1] == smd.sha256_of(f)


# ---------------------------------------------------------------- report helpers


def test_matched_vs_baseline_uses_only_units_where_both_reached():
    rows = []
    for unit, series in (("a|0", "a"), ("b|0", "b"), ("c|0", "c")):
        for method, fa, reached in (("zscore", 2.0, True), ("seasonal", 1.0, unit != "c|0")):
            rows.append({"dataset": "toy", "series": series, "unit": unit, "method": method, "target": 0.8,
                         "k": 1, "rules": "none", "reached": reached, "test_false_alerts_per_day": fa,
                         "test_event_recall": 0.8, "test_ttd_median_min": 10.0})
    t = R.matched_vs_baseline(pd.DataFrame(rows), 0.8, methods=("zscore", "seasonal"))
    assert t["n_units"].iloc[0] == 2
    assert t["pooled_reduction"].iloc[0] == pytest.approx(0.5)


def test_chosen_json_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"methods": {"lof": {"n_neighbors": 50}}}))
    assert json.loads(p.read_text())["methods"]["lof"]["n_neighbors"] == 50
