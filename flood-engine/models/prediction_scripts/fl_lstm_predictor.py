"""Run clustered federated vanilla-LSTM water-level forecasts.

Example:
    py Prediction/fl_lstm_predictor.py --input Prediction/input_csv/engineered_data.csv \
        --start-date 2026-08-23 --stations SW137A SW147.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "clustered_federated_vanilla_lstm_model_10-10_less_ep_lr.pt"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "Prediction" / "input_csv" / "engineered_data.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Prediction" / "result" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_INPUT_LEN = 150
DEFAULT_OUTPUT_LEN = 75
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2


class VanillaLSTM(nn.Module):
    def __init__(self, input_size: int, output_len: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0.0,
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, output_len),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(inputs)
        return self.forecast_head(encoded[:, -1])


def load_input(path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    required = {"station_id", "timestamp", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns required by the checkpoint: {missing}")

    frame["station_id"] = frame["station_id"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["station_id", "timestamp"])
    frame = frame.drop_duplicates(["station_id", "timestamp"], keep="last")
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    # Training uses five observations per day. Keep only timestamps shared by all stations
    # so each station window has the same temporal shape as the training tensor.
    counts = frame.groupby("timestamp")["station_id"].nunique()
    station_count = frame["station_id"].nunique()
    shared_timestamps = sorted(counts[counts == station_count].index.tolist())
    if not shared_timestamps:
        raise ValueError("The CSV does not contain a complete shared timestamp grid.")
    frame = frame[frame["timestamp"].isin(shared_timestamps)].copy()
    # print(f"shape of frame: {frame.shape}, number of shared timestamps: {len(shared_timestamps)}")

    return frame, shared_timestamps


def cluster_for_station(station: str, clusters: dict) -> str:
    for cluster_name, stations in clusters.items():
        if station in stations:
            return cluster_name
    raise ValueError(f"Station {station!r} is not present in the checkpoint clusters.")


def predict(
    input_path: Path,
    start_date: str,
    station_names: list[str] | None,
    model_path: Path = MODEL_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feature_columns = list(checkpoint.get("feature_columns", [
        "WL(mMSL)", "RAINFALL(mm)", "hour_sin", "hour_cos",
        "doy_sin", "doy_cos", "delta_hours",
    ]))
    input_len = int(checkpoint.get("input_len", DEFAULT_INPUT_LEN))
    output_len = int(checkpoint.get("output_len", DEFAULT_OUTPUT_LEN))
    clusters = checkpoint.get("clusters")
    if not clusters or "cluster_model_state_dicts" not in checkpoint:
        raise KeyError("Checkpoint must contain clusters and cluster_model_state_dicts.")
    if "input_scaler" not in checkpoint or "target_scaler" not in checkpoint:
        raise KeyError("Checkpoint must contain input_scaler and target_scaler.")

    frame, shared_timestamps = load_input(input_path, feature_columns)
    requested_date = pd.Timestamp(start_date).normalize()
    forecast_timestamps = [
        timestamp for timestamp in shared_timestamps
        if timestamp.normalize() == requested_date
    ]
    if not forecast_timestamps:
        raise ValueError(f"No complete prediction timestamp is available for {start_date!r}.")
    forecast_start = forecast_timestamps[0]
    start_index = shared_timestamps.index(forecast_start)
    if start_index < input_len:
        raise ValueError(
            f"Only {start_index} prior timestamps are available before {forecast_start}; "
            f"the model requires {input_len}."
        )



    available_stations = set(frame["station_id"])


    checkpoint_stations = [station for stations in clusters.values() for station in stations]
    print(f"len checkpoint_stations: {len(checkpoint_stations)}")
    if station_names is None:
        station_names = [station for station in checkpoint_stations if station in available_stations]
    else:
        station_names = [station.strip() for station in station_names]
    unknown = sorted(set(station_names) - available_stations)
    if unknown:
        raise ValueError(f"Requested stations are not present in the CSV: {unknown}")
    if not station_names:
        raise ValueError("No stations were selected for prediction.")

    print(f"stations to predict: {station_names}")
    print(f"available stations: {available_stations}")

    station_order = sorted(available_stations)
    print(f"len station_order: {len(station_order)}")
    print(f"station name len: {len(station_names)}")
    station_to_index = {station: index for index, station in enumerate(station_order)}
    
    pivot = frame.pivot(
        index="timestamp",
        columns="station_id",
        values=feature_columns
    )

    # pivot() creates columns in (feature, station_id) order.
    # Change them to (station_id, feature) before reindexing.
    pivot.columns = pivot.columns.swaplevel(0, 1)

    expected_columns = pd.MultiIndex.from_product(
        [station_order, feature_columns],
        names=pivot.columns.names
    )

    pivot = pivot.reindex(
        index=shared_timestamps,
        columns=expected_columns
    )

    # print("Pivot shape:", pivot.shape)
    # print("Expected columns:", len(station_order) * len(feature_columns))
    # print("NaN count:", pivot.isna().sum().sum())
    # print("First columns:", pivot.columns[:10].tolist())

    # print(f"first row of pivot: {pivot.iloc[0]}")
    input_window = pivot.iloc[start_index-input_len:start_index]
    missing_cells = input_window.isna()
    
    if missing_cells.any().any():
        missing_timestamps = missing_cells.any(axis=1)
        print(f"input window shape {input_window.shape}")
        for timestamp in input_window.index[missing_timestamps]:
            missing_columns = missing_cells.loc[timestamp]
            missing_columns = missing_columns[missing_columns].index.tolist()
            missing_details = [
                f"{station_id}.{feature}"
                for station_id, feature in missing_columns
            ]
            # print(f"  {timestamp}: missing features: {', '.join(missing_details)}\n")
        raise ValueError(
            f"The 30-day input window contains missing feature values at "
            f"{int(missing_timestamps.sum())} timestamps."
        )

    values = pivot.to_numpy(dtype=np.float32).reshape(len(shared_timestamps), len(station_order), len(feature_columns))
    scaler = checkpoint["input_scaler"]
    values = scaler.transform(values.reshape(-1, len(feature_columns))).reshape(values.shape)
    target_scaler = checkpoint["target_scaler"]
    rows = []

    # The model predicts 75 values directly; repeat the five observed daily time slots.
    daily_times = sorted({timestamp.time() for timestamp in shared_timestamps})
    if not daily_times:
        raise ValueError("Input timestamps do not contain any daily time slots.")
    future_timestamps = [
        forecast_start.normalize() + pd.Timedelta(days=offset // len(daily_times))
        + pd.Timedelta(
            hours=daily_times[offset % len(daily_times)].hour,
            minutes=daily_times[offset % len(daily_times)].minute,
            seconds=daily_times[offset % len(daily_times)].second,
        )
        for offset in range(output_len)
    ]
    print(f"station len {len(station_names)}, future timestamps len {len(future_timestamps)}")
    for station in station_names:
        print(f"Predicting for station {station}...")
        cluster_name = cluster_for_station(station, clusters)
        state = checkpoint["cluster_model_state_dicts"].get(cluster_name)
        if state is None:
            raise KeyError(f"Checkpoint has no model state for cluster {cluster_name!r}.")
        model = VanillaLSTM(
            len(feature_columns), output_len,
            int(checkpoint.get("hidden_size", HIDDEN_SIZE)),
            int(checkpoint.get("num_layers", NUM_LAYERS)),
        ).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        station_index = station_to_index[station]
        window = torch.tensor(
            values[start_index - input_len:start_index, station_index, :],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)
        with torch.no_grad():
            prediction = model(window).cpu().numpy().reshape(-1, 1)
        prediction = target_scaler.inverse_transform(prediction).ravel()
        rows.extend({
                "station_id": station,
                "timestamp": timestamp,
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "predicted_WL(mMSL)": float(value),
            } for timestamp, value in zip(future_timestamps, prediction))

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path / f"lstm_prediction_{start_date}.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--start-date", required=True, help="First forecast date, for example 2026-08-23")
    parser.add_argument("--stations", nargs="*", help="Station IDs; omit to predict all available checkpoint stations")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    result = predict(args.input, args.start_date, args.stations, args.model, args.output)
    print(f"Wrote {len(result):,} predictions for {result['station_id'].nunique()} stations to {args.output}")


if __name__ == "__main__":
    main()
