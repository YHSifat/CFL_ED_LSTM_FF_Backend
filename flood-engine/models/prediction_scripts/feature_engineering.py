"""Create the seven-feature CSV used by the federated vanilla LSTM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OFFICIAL_TIMES = ("06:00:00", "09:00:00", "12:00:00", "15:00:00", "18:00:00")
RAW_FEATURE_COLUMNS = ("WL(mMSL)", "RAINFALL(mm)")
TIME_FEATURE_COLUMNS = ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "delta_hours")
OUTPUT_COLUMNS = (
    "station_id", "date", "time", "timestamp",
    *RAW_FEATURE_COLUMNS, *TIME_FEATURE_COLUMNS,
)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the temporal channels used by ``load_st_gnn_station_data``."""
    frame = frame.copy()
    hour = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60.0
    day_of_year = frame["timestamp"].dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["delta_hours"] = (
        frame.groupby("station_id", sort=False)["timestamp"]
        .transform(lambda values: values.diff().dt.total_seconds().fillna(0.0) / 3600.0 / 12.0)
    )
    return frame


def fill_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the training pipeline's interpolation and fallback policy."""
    frame = frame.sort_values(["station_id", "timestamp"]).copy()
    for column in RAW_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame.groupby("station_id", sort=False)[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
        frame[column] = (
            frame.groupby("station_id", sort=False)[column]
            .transform(lambda values: values.ffill().bfill())
            .fillna(0.0)
            .astype("float32")
        )
    return frame


def build_complete_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every station on the shared official timestamp grid."""
    station_ranges = frame.groupby("station_id")["timestamp"].agg(["min", "max"])
    start = station_ranges["min"].max().normalize()
    end = station_ranges["max"].min().normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Stations do not have an overlapping timestamp range.")

    dates = pd.date_range(start, end, freq="D")
    times = pd.to_timedelta([time[:2] + ":00:00" for time in OFFICIAL_TIMES])
    timestamps = dates.repeat(len(times)) + times.tolist() * len(dates)
    stations = sorted(frame["station_id"].astype(str).unique())
    grid = pd.MultiIndex.from_product(
        [stations, timestamps], names=["station_id", "timestamp"]
    ).to_frame(index=False)
    return grid.merge(frame, on=["station_id", "timestamp"], how="left")


def build_features(input_path: Path) -> pd.DataFrame:
    """Read processed data and return a complete model-ready feature table."""
    frame = pd.read_csv(input_path)
    frame.columns = frame.columns.str.strip()
    required = {"station_id", "timestamp", *RAW_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    frame["station_id"] = frame["station_id"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["station_id", "timestamp"])
    frame = frame.drop_duplicates(["station_id", "timestamp"], keep="last")
    frame = frame[["station_id", "timestamp", *RAW_FEATURE_COLUMNS]]
    frame = build_complete_grid(frame)
    frame = fill_missing_features(frame)
    frame = add_time_features(frame)
    frame["date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["time"] = frame["timestamp"].dt.strftime("%H:%M:%S")
    frame = frame.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    #change station_id SW177 to SW179 lakhya--lakhpur
    frame["station_id"] = frame["station_id"].replace("SW177", "SW179")
    return frame[list(OUTPUT_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV (default: Prediction/input_csv/processed_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: Prediction/input_csv/engineered_data.csv)",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = args.input or script_dir / "input_csv" / "processed_data.csv"
    output_path = args.output or script_dir / "input_csv" / "engineered_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = build_features(input_path)
    features.to_csv(output_path, index=False)
    print(
        f"Wrote {len(features):,} rows for "
        f"{features['station_id'].nunique()} stations to {output_path}"
    )


if __name__ == "__main__":
    main()
