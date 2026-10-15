import pandas as pd

from anomaly_monitoring.cli import main
from conftest import PPD, seasonal_series


def write_csv(tmp_path, with_time=True):
    x = seasonal_series(n_days=50, seed=4, noise=0.3)
    x[40 * PPD + 5] += 20.0
    df = pd.DataFrame({"value": x})
    if with_time:
        df.insert(0, "timestamp", pd.date_range("2026-01-01", periods=x.size, freq="h"))
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)
    return path


def test_stream_prints_the_injected_spike_as_an_alert(tmp_path, capsys):
    path = write_csv(tmp_path)
    rc = main(["stream", str(path), "--column", "value", "--time-column", "timestamp",
               "--points-per-day", str(PPD), "--fit-points", str(20 * PPD), "--budget", "1", "--rules"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALERT" in out and "2026-02-10 05:00" in out  # day 40, hour 5
    assert "processed 1200 points" in out


def test_stream_threshold_option_and_index_labels(tmp_path, capsys):
    path = write_csv(tmp_path, with_time=False)
    rc = main(["stream", str(path), "--points-per-day", str(PPD), "--fit-points", str(20 * PPD),
               "--detector", "ewma", "--threshold", "4", "--persistence", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "t=" in out and "threshold 4" in out


def test_stream_missing_file_and_bad_column_fail_cleanly(tmp_path, capsys):
    assert main(["stream", str(tmp_path / "nope.csv")]) == 2
    path = write_csv(tmp_path)
    assert main(["stream", str(path), "--column", "absent"]) == 2
    assert "absent" in capsys.readouterr().err


def test_stream_needs_a_fit_window_shorter_than_the_series(tmp_path, capsys):
    path = write_csv(tmp_path)
    assert main(["stream", str(path), "--fit-points", "999999"]) == 2
