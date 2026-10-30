"""Runs the clustered federated vanilla LSTM (models/train_test_scripts/fl_lstm_claude.py).

Input:  the last 30 days of 3-hourly WL readings (06/09/12/15/18 Asia/Dhaka) plus that
        day's rainfall copied to each of its 5 readings  -> 150 steps x 7 features.
Output: the next 75 readings, from 06:00 the day after the input ends to 18:00 on day 15.

Each station uses its cluster's model and its own scaler row, both read from the checkpoint.
Preprocessing must stay identical to the training script; check_model_parity.py verifies it.
"""
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config

SLOT_HOURS = (6, 9, 12, 15, 18)          # FFWC reading times, local Bangladesh time
INPUT_DAYS, FORECAST_DAYS = 30, 15
# FFWC station -> checkpoint station whose model and scaler it uses. SW179 (Demra, Lakhya)
# left FFWC's list; SW177 (Lakhpur, Lakhya, similar levels) takes its place, as in the
# earlier pipeline (models/prediction_scripts/feature_engineering.py).
STATION_ALIASES = {"SW177": "SW179"}
_REQUIRED_KEYS = ("cluster_model_state_dicts", "scaler_mean", "scaler_std", "stations",
                  "clusters", "feature_columns", "input_len", "output_len")


class InputError(ValueError):
    """The station's data can't form a valid input window (the station is skipped)."""


class VanillaLSTM(nn.Module):
    """Same layers and state-dict keys as VanillaLSTM in the training script."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_len: int,
                 residual: bool):
        super().__init__()
        self.residual = residual
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(),
            nn.Linear(hidden_size, output_len),
        )

    def forward(self, inputs):
        encoded, _ = self.lstm(inputs)
        output = self.forecast_head(encoded[:, -1])
        if self.residual:
            output = output + inputs[:, -1, 0:1]   # scaled last WL = persistence baseline
        return output


def input_slots(end_day: date) -> pd.DatetimeIndex:
    """150 local reading times: day end_day-29 06:00 .. end_day 18:00."""
    days = [end_day - timedelta(days=k) for k in range(INPUT_DAYS - 1, -1, -1)]
    return pd.DatetimeIndex([datetime.combine(d, time(h)) for d in days for h in SLOT_HOURS])


def target_slots(end_day: date) -> pd.DatetimeIndex:
    """75 local reading times: end_day+1 06:00 .. end_day+15 18:00."""
    days = [end_day + timedelta(days=k) for k in range(1, FORECAST_DAYS + 1)]
    return pd.DatetimeIndex([datetime.combine(d, time(h)) for d in days for h in SLOT_HOURS])


def time_features(timestamps: pd.DatetimeIndex, previous: pd.Timestamp) -> np.ndarray:
    """hour_sin, hour_cos, doy_sin, doy_cos, delta_hours, as in the training load_data()."""
    hour = timestamps.hour + timestamps.minute / 60.0
    doy = timestamps.dayofyear
    times = pd.DatetimeIndex([previous]).append(timestamps).to_series()
    gaps = times.diff().dt.total_seconds().to_numpy()[1:] / 3600.0
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        gaps / 12.0,
    ]).astype(np.float32)


def _local_naive(index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_convert(config.TZ).tz_localize(None)
    return index


class Forecaster:
    def __init__(self, path: str | Path = config.MODEL_PATH):
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        ck = torch.load(path, map_location="cpu", weights_only=False)
        missing = [k for k in _REQUIRED_KEYS if k not in ck]
        if missing:
            raise KeyError(f"{path.name} is missing checkpoint keys: {missing}")

        self.path = path
        self.version = path.stem
        self.input_len, self.output_len = ck["input_len"], ck["output_len"]
        if (self.input_len, self.output_len) != (INPUT_DAYS * len(SLOT_HOURS),
                                                 FORECAST_DAYS * len(SLOT_HOURS)):
            raise ValueError(f"Unexpected checkpoint lengths {self.input_len}/{self.output_len}")
        self.log_rainfall = ck.get("log_rainfall", True)
        self.stations: list[str] = list(ck["stations"])
        self.station_index = {s: i for i, s in enumerate(self.stations)}
        self.cluster_of = {s: c for c, members in ck["clusters"].items() for s in members}
        self.mean = np.asarray(ck["scaler_mean"], dtype=np.float32)   # (stations, features)
        self.std = np.asarray(ck["scaler_std"], dtype=np.float32)

        n_features = len(ck["feature_columns"])
        self.models = {}
        for cluster, state in ck["cluster_model_state_dicts"].items():
            model = VanillaLSTM(n_features, ck.get("hidden_size", 128), ck.get("num_layers", 2),
                                self.output_len, ck.get("residual_forecast", True))
            model.load_state_dict(state)
            self.models[cluster] = model.eval()

    def has_station(self, station_id: str) -> bool:
        return STATION_ALIASES.get(station_id, station_id) in self.station_index

    # ---- input -----------------------------------------------------------
    def build_raw_window(self, wl: pd.Series, rain: pd.Series,
                         end_day: date) -> tuple[np.ndarray, dict]:
        """wl: readings indexed by time; rain: mm indexed by date -> (150, 2) [wl, rain]
        plus counts of filled values.

        WL is matched to the reading slots; runs of up to MAX_WL_INTERP_STEPS missing
        readings are interpolated (as in training), longer gaps raise InputError.
        Missing readings at the end (e.g. 18:00 not posted yet) repeat the last reading.
        """
        slots = input_slots(end_day)
        # Snap to the nearest hour so e.g. 05:59 still lands on the 06:00 slot.
        wl = pd.Series(wl.to_numpy(dtype=float), index=_local_naive(wl.index).round("h"))
        wl = wl[~wl.index.duplicated(keep="first")].reindex(slots)
        if wl.loc[str(end_day)].isna().all():
            raise InputError(f"no WL reading on {end_day}")
        gaps = wl.isna()
        longest = gaps.groupby((~gaps).cumsum()).sum().max()
        if longest > config.MAX_WL_INTERP_STEPS:
            raise InputError(f"WL gap of {longest} readings (max {config.MAX_WL_INTERP_STEPS})")
        wl = wl.interpolate(limit=config.MAX_WL_INTERP_STEPS, limit_direction="both")

        rain = pd.Series(rain.to_numpy(dtype=float), index=pd.to_datetime(rain.index).date)
        rain = rain[~rain.index.duplicated(keep="first")]
        # Daily rainfall copied to that day's 5 readings; missing rain counts as 0 mm.
        rf = rain.reindex(slots.date)
        info = {"wl_filled": int(gaps.sum()),
                "rain_missing_days": int(rf.isna().sum()) // len(SLOT_HOURS)}
        rf = rf.fillna(0.0).clip(lower=0.0).to_numpy()
        return np.column_stack([wl.to_numpy(), rf]).astype(np.float32), info

    # ---- predict ---------------------------------------------------------
    def predict_window(self, station_id: str, raw: np.ndarray, timestamps: pd.DatetimeIndex,
                       previous: pd.Timestamp) -> np.ndarray:
        """raw: (150, 2) unscaled [wl, rainfall_mm] at `timestamps`; `previous` is the
        reading time before the window (for delta_hours). Returns 75 WL values in metres."""
        if raw.shape != (self.input_len, 2) or not np.isfinite(raw).all():
            raise InputError(f"bad input window shape {raw.shape} or non-finite values")
        station_id = STATION_ALIASES.get(station_id, station_id)
        s = self.station_index[station_id]
        inputs = raw.copy()
        if self.log_rainfall:
            inputs[:, 1] = np.log1p(inputs[:, 1])
        features = np.concatenate([inputs, time_features(timestamps, previous)], axis=1)
        features = (features - self.mean[s]) / self.std[s]
        with torch.no_grad():
            out = self.models[self.cluster_of[station_id]](
                torch.from_numpy(features[None].astype(np.float32))).numpy()[0]
        y = out * self.std[s, 0] + self.mean[s, 0]
        if not np.isfinite(y).all():
            raise ValueError("model returned NaN/inf")
        return y.astype(float)

    def predict(self, station_id: str, wl: pd.Series, rain: pd.Series,
                end_day: date) -> tuple[pd.Series, dict]:
        """Forecast issued on the night of end_day. Returns WL (m) indexed by tz-aware
        local target time, and the build_raw_window fill counts."""
        if not self.has_station(station_id):
            raise InputError(f"{station_id} is not in the model's station list")
        raw, info = self.build_raw_window(wl, rain, end_day)
        slots = input_slots(end_day)
        previous = slots[0] - pd.Timedelta(hours=12)   # previous day's 18:00 reading
        y = self.predict_window(station_id, raw, slots, previous)
        return pd.Series(y, index=target_slots(end_day).tz_localize(config.TZ)), info
