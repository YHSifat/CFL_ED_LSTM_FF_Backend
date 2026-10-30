"""Wraps the pretrained model: builds the 30-day input and returns a 15-day forecast.

Model files go in MODEL_DIR. For each station it looks for
  <station_id>.<ext>  (per-station model)  else  global.<ext>
with ext in .keras/.h5 (Keras), .pt (TorchScript), .joblib/.pkl (sklearn etc.).
Optional scalers saved with joblib next to it:
  <name>_scaler_x.joblib  (fit on [wl, rainfall] columns)
  <name>_scaler_y.joblib  (fit on the wl target)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config

_EXTS = (".keras", ".h5", ".pt", ".joblib", ".pkl")


class Predictor:
    def __init__(self, model_dir: str = config.MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._cache: dict[str, tuple] = {}

    # ---- loading ---------------------------------------------------------
    def _find(self, station_id: str) -> tuple[str, Path]:
        for name in (station_id, "global"):
            for ext in _EXTS:
                p = self.model_dir / f"{name}{ext}"
                if p.exists():
                    return name, p
        raise FileNotFoundError(f"No model for station {station_id} in {self.model_dir}")

    def _load(self, station_id: str):
        name, path = self._find(station_id)
        if name in self._cache:
            return self._cache[name]
        ext = path.suffix
        if ext in (".keras", ".h5"):
            from tensorflow import keras
            model, kind = keras.models.load_model(path, compile=False), "keras"
        elif ext == ".pt":
            import torch
            model, kind = torch.jit.load(str(path), map_location="cpu").eval(), "torch"
        else:
            model, kind = joblib.load(path), "sklearn"
        sx = self.model_dir / f"{name}_scaler_x.joblib"
        sy = self.model_dir / f"{name}_scaler_y.joblib"
        entry = (kind, model,
                 joblib.load(sx) if sx.exists() else None,
                 joblib.load(sy) if sy.exists() else None)
        self._cache[name] = entry
        return entry

    # ---- input -----------------------------------------------------------
    @staticmethod
    def build_input(df: pd.DataFrame) -> np.ndarray | None:
        """df: INPUT_DAYS rows of [day, wl, rainfall_mm] -> array (INPUT_DAYS, 2) or None."""
        if len(df) != config.INPUT_DAYS:
            return None
        wl = df["wl"].astype(float)
        if wl.isna().sum() > config.MAX_MISSING_WL_DAYS:
            return None
        wl = wl.interpolate(limit_direction="both")
        rf = df["rainfall_mm"].astype(float).fillna(0.0)  # missing rain treated as 0 mm
        return np.column_stack([wl.to_numpy(), rf.to_numpy()])

    # ---- predict ---------------------------------------------------------
    def predict(self, station_id: str, df: pd.DataFrame) -> np.ndarray | None:
        x = self.build_input(df)
        if x is None:
            return None
        kind, model, sx, sy = self._load(station_id)

        xi = sx.transform(x) if sx is not None else x
        inp = xi[None, ...] if config.INPUT_LAYOUT == "sequence" else xi.reshape(1, -1)
        inp = inp.astype(np.float32)

        if kind == "keras":
            y = model.predict(inp, verbose=0)
        elif kind == "torch":
            import torch
            with torch.no_grad():
                y = model(torch.from_numpy(inp)).cpu().numpy()
        else:
            y = model.predict(inp)

        y = np.asarray(y, dtype=float).reshape(-1)
        if sy is not None:
            y = sy.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        if config.OUTPUT_IS_DELTA:
            y = x[-1, 0] + y
        if y.shape[0] != config.HORIZON_DAYS:
            raise ValueError(f"Model returned {y.shape[0]} values, expected {config.HORIZON_DAYS}")
        if not np.all(np.isfinite(y)):
            raise ValueError("Model returned NaN/inf")
        return y
