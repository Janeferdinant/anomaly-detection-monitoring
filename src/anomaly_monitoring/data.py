"""Load the UCI household power data as hourly series and cut non-overlapping benchmark windows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CHANNELS = {
    "global_active_power": "kW",
    "voltage": "V",
    "global_intensity": "A",
    "global_reactive_power": "kVAr",
}
POINTS_PER_DAY = 24
ROOT = Path(__file__).resolve().parents[2]
RAW_TXT = ROOT / "data" / "raw" / "household_power_consumption.txt"
CACHE = ROOT / "data" / "processed" / "hourly.csv"


def load_hourly(txt_path: Path = RAW_TXT, cache: Path | None = CACHE) -> pd.DataFrame:
    """Hourly means of the selected channels (NaN for hours with no valid minute at all)."""
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True, float_precision="round_trip")
    wanted = {"date", "time", *CHANNELS}
    raw = pd.read_csv(txt_path, sep=";", usecols=lambda c: c.lower() in wanted, na_values="?",
                      low_memory=False)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"date": "Date", "time": "Time"})
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw = raw.drop(columns=["Date", "Time"]).set_index(ts).astype(float)
    hourly = raw.resample("h").mean()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(cache)
    return hourly


def make_units(hourly: pd.DataFrame, channels=None, n_windows: int = 3, window_days: int = 120) -> list[dict]:
    """Pick ``n_windows`` consecutive-in-time, non-overlapping, gap-free windows per channel.

    A window is accepted only if the *raw* data has no missing hour in it, so every missing
    value in an evaluation series is an injected, labelled one. Windows are scanned in time
    order from the start of the data with a stride of ``window_days``.
    """
    channels = list(channels or CHANNELS)
    length = window_days * POINTS_PER_DAY
    units = []
    for ch in channels:
        series = hourly[ch]
        picked = 0
        for start in range(0, len(series) - length + 1, length):
            win = series.iloc[start : start + length]
            if win.isna().any():
                continue
            units.append({
                "name": f"{ch}-w{picked}",
                "channel": ch,
                "unit": CHANNELS[ch],
                "start": win.index[0].strftime("%Y-%m-%d"),
                "values": win.to_numpy(dtype=float),
            })
            picked += 1
            if picked == n_windows:
                break
        if picked < n_windows:
            raise RuntimeError(f"only {picked} gap-free windows found for {ch}")
    return units
