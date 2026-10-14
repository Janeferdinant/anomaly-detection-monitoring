import numpy as np

from anomaly_monitoring.rules import (
    DataQualityRules,
    fit_range,
    fit_stuck_run,
    missing_flags,
    range_flags,
    stuck_flags,
)


def test_missing_flags():
    x = np.array([1.0, np.nan, np.nan, 2.0])
    assert missing_flags(x).tolist() == [False, True, True, False]


def test_range_flags_ignore_nan():
    x = np.array([0.0, 5.0, 11.0, np.nan, -1.0])
    assert range_flags(x, lo=0.0, hi=10.0).tolist() == [False, False, True, False, True]


def test_fit_range_widens_with_margin():
    x = np.linspace(0, 10, 1001)
    lo, hi = fit_range(x, margin=0.5)
    assert lo < 0.0 and hi > 10.0
    lo0, hi0 = fit_range(x, margin=0.0)
    assert lo < lo0 and hi > hi0


def test_stuck_flags_flags_from_kth_repeat_causally():
    x = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0])
    f = stuck_flags(x, min_run=3)
    # run of four 3.0s at idx 2..5, flagged from the 3rd identical value (idx 4)
    assert f.tolist() == [False, False, False, False, True, True, False, False]


def test_stuck_flags_short_runs_ok_and_nan_breaks_runs():
    x = np.array([1.0, 1.0, np.nan, 1.0, 1.0, 2.0])
    assert not stuck_flags(x, min_run=3).any()


def test_fit_stuck_run_has_floor_and_exceeds_natural_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert fit_stuck_run(x, floor=4) == 4
    y = x.copy()
    for s in range(0, 400, 20):
        y[s : s + 7] = y[s]  # many natural runs of length 7
    assert fit_stuck_run(y, floor=4) >= 7


def test_fit_stuck_run_ignores_a_few_genuine_stuck_episodes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    for s in (50, 200, 350):
        x[s : s + 20] = x[s]  # three long stuck episodes, no other repeats
    assert fit_stuck_run(x, floor=4) == 4  # too few runs to estimate: floor
    y = x.copy()
    for s in range(400, 580, 12):
        y[s : s + 2] = y[s]  # many natural repeats of length 2
    assert fit_stuck_run(y, floor=4) == 4  # median 2 + 2


def test_data_quality_rules_combine():
    rng = np.random.default_rng(1)
    calib = rng.normal(50, 1, size=600)
    rules = DataQualityRules().fit(calib)
    x = rng.normal(50, 1, size=100)
    x[10] = np.nan
    x[30] = 90.0
    x[50:70] = x[50]
    flags = rules.flags(x)
    assert flags[10] and flags[30] and flags[69]
    assert not flags[0:10].any()


def test_components_and_ablation_subsets_add_up():
    rng = np.random.default_rng(2)
    rules = DataQualityRules().fit(rng.normal(50, 1, size=600))
    x = rng.normal(50, 1, size=100)
    x[10] = np.nan
    x[30] = 90.0
    x[50:70] = x[50]
    comp = rules.components(x)
    assert comp["missing"][10] and not comp["missing"][30]
    assert comp["range"][30] and not comp["range"][10]
    assert comp["stuck"][69] and not comp["stuck"][30]
    union = rules.flags(x)
    assert (union == (comp["missing"] | comp["range"] | comp["stuck"])).all()
    assert not rules.flags(x, use=("missing", "range"))[69]  # leave-one-out removes the stuck flag
    assert rules.flags(x, use=()).sum() == 0


def test_flat_channels_switch_the_stuck_and_range_rules_off():
    flat = np.full(500, 3.0)
    rules = DataQualityRules().fit(flat)
    assert rules.min_run is None
    x = np.concatenate([flat[:50], np.full(50, 4.0)])
    assert not rules.flags(x).any()  # nothing is "stuck" or "out of range" on a constant channel


def test_multichannel_rules_flag_when_any_channel_is_flagged():
    from anomaly_monitoring.rules import MultiDataQualityRules

    rng = np.random.default_rng(3)
    calib = rng.normal(0, 1, size=(600, 3))
    rules = MultiDataQualityRules().fit(calib)
    x = rng.normal(0, 1, size=(50, 3))
    x[7, 2] = 50.0
    x[20, 0] = np.nan
    f = rules.flags(x)
    assert f[7] and f[20] and f.sum() == 2
    assert rules.flags(x, use=("range",))[7] and not rules.flags(x, use=("range",))[20]
