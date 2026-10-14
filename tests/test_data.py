import numpy as np
import pandas as pd
import pytest

from anomaly_monitoring.data import POINTS_PER_DAY, make_units


def hourly_frame(n_days, nan_ranges=(), channels=("voltage", "global_intensity")):
    idx = pd.date_range("2007-01-01", periods=n_days * POINTS_PER_DAY, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=idx.size) for c in channels}, index=idx)
    for a, b in nan_ranges:
        df.iloc[a:b, :] = np.nan
    return df


def test_windows_are_gap_free_non_overlapping_and_right_length():
    df = hourly_frame(400)
    units = make_units(df, channels=["voltage"], n_windows=3, window_days=100)
    assert len(units) == 3
    for u in units:
        assert u["values"].size == 100 * POINTS_PER_DAY
        assert not np.isnan(u["values"]).any()
    starts = [u["start"] for u in units]
    assert starts == sorted(set(starts))


def test_windows_with_natural_gaps_are_skipped():
    gap = (150 * POINTS_PER_DAY, 150 * POINTS_PER_DAY + 5)  # inside the second 100-day block
    df = hourly_frame(500, nan_ranges=[gap])
    units = make_units(df, channels=["voltage"], n_windows=3, window_days=100)
    assert [u["start"] for u in units][1] != "2007-04-11"  # block 2 (days 100-199) was skipped
    assert all(not np.isnan(u["values"]).any() for u in units)


def test_too_few_clean_windows_raises():
    df = hourly_frame(150)
    with pytest.raises(RuntimeError):
        make_units(df, channels=["voltage"], n_windows=3, window_days=100)


def test_hourly_cache_round_trips_floats_exactly(tmp_path):
    """The fast CSV float parser can differ in the last bit; scores of flat series are sensitive to that."""
    from anomaly_monitoring.data import load_hourly

    rng = np.random.default_rng(1)
    idx = pd.date_range("2007-01-01", periods=48, freq="min")
    raw = pd.DataFrame({"Date": idx.strftime("%d/%m/%Y"), "Time": idx.strftime("%H:%M:%S"),
                        "Global_active_power": rng.normal(1, 0.3, idx.size),
                        "Global_reactive_power": rng.normal(0.1, 0.05, idx.size),
                        "Voltage": rng.normal(240, 2, idx.size), "Global_intensity": rng.normal(4, 1, idx.size)})
    txt = tmp_path / "raw.txt"
    raw.to_csv(txt, sep=";", index=False)
    cache = tmp_path / "hourly.csv"
    first = load_hourly(txt, cache)      # computed from the raw file, written to the cache
    second = load_hourly(txt, cache)     # read back from the cache
    pd.testing.assert_frame_equal(first, second, check_freq=False, check_exact=True)
