"""Checks forecaster.py against the training script's own preprocessing and model.

  python check_model_parity.py --data "<thesis>/Datasets/Processed/Station_wise_combined"

1. time features:  forecaster.time_features == training load_station_data() columns
2. model:          Forecaster.predict_window == training VanillaLSTM on the same windows
3. live path:      Forecaster.predict fed raw CSV readings + daily rainfall (the way the
                   engine will feed it) == training pipeline, for day-aligned windows
Also prints model vs persistence MAE on the sampled test-period windows.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import forecaster as fc

TRAIN_SCRIPT = Path(__file__).resolve().parent / "models" / "train_test_scripts" / "fl_lstm_claude.py"
TOLERANCE_M = 1e-4


def load_training_module(data_dir: Path):
    if importlib.util.find_spec("tqdm") is None:     # only used by the training loop
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=None)
    spec = importlib.util.spec_from_file_location("fl_lstm_claude", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = data_dir
    module.DEVICE = torch.device("cpu")
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path, help="Station_wise_combined folder")
    p.add_argument("--windows", type=int, default=300, help="random test windows to compare")
    args = p.parse_args()

    train = load_training_module(args.data)
    model = fc.Forecaster()
    stations, index, raw, _ = train.load_data()
    _, _, features, targets, _ = train.load_station_data()
    if stations != model.stations:
        sys.exit(f"Station order differs: data {stations} vs checkpoint {model.stations}")

    # Reference: the training script's model on its own scaled features.
    ck = torch.load(model.path, map_location="cpu", weights_only=False)
    mean, std = ck["scaler_mean"], ck["scaler_std"]
    scaled = ((features - mean) / std).astype(np.float32)
    ref_models = {}
    for cluster, state in ck["cluster_model_state_dicts"].items():
        m = train.VanillaLSTM(features.shape[-1])
        train.load_trainable_state(m, state)
        ref_models[cluster] = m.eval()

    def reference(i, s):
        x = torch.from_numpy(scaled[i:i + model.input_len, s][None])
        with torch.no_grad():
            y = ref_models[model.cluster_of[stations[s]]](x).numpy()[0]
        return y * std[s, 0] + mean[s, 0]

    rng = np.random.default_rng(0)
    test_start = int(len(index) * 0.85)
    last_start = len(index) - model.input_len - model.output_len
    starts = rng.integers(test_start, last_start + 1, args.windows)
    sts = rng.integers(0, len(stations), args.windows)

    # 1 + 2: time features and model output on identical windows
    tf_diff, model_diff, mae_model, mae_persist = 0.0, 0.0, [], []
    for i, s in zip(starts, sts):
        ts = index[i:i + model.input_len]
        tf = fc.time_features(ts, index[i - 1])
        tf_diff = max(tf_diff, float(np.abs(tf - features[i:i + model.input_len, s, 2:]).max()))
        ours = model.predict_window(stations[s], raw[i:i + model.input_len, s], ts, index[i - 1])
        ref = reference(i, s)
        model_diff = max(model_diff, float(np.abs(ours - ref).max()))
        actual = targets[i + model.input_len:i + model.input_len + model.output_len, s]
        mae_model.append(np.abs(ours - actual).mean())
        mae_persist.append(np.abs(raw[i + model.input_len - 1, s, 0] - actual).mean())
    print(f"[1] time features   max |diff| = {tf_diff:.2e}")
    print(f"[2] model output    max |diff| = {model_diff:.2e} m over {args.windows} windows")
    print(f"    MAE model {np.mean(mae_model):.4f} m | persistence {np.mean(mae_persist):.4f} m")

    # 3: live path from raw CSV readings, windows ending at 18:00 on a regular grid
    csv = {}
    for s in stations:
        df = pd.read_csv(args.data / f"{s}.csv")
        df.columns = df.columns.str.strip()
        df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        wl = pd.to_numeric(df[train.WL_COLUMN], errors="coerce")
        rain = pd.to_numeric(df[train.RAIN_COLUMN], errors="coerce")
        csv[s] = (pd.Series(wl.to_numpy(), index=df["ts"]).dropna(),
                  pd.Series(rain.to_numpy(), index=pd.to_datetime(df["date"]).dt.date).dropna())

    pos = pd.Series(np.arange(len(index)), index=index)
    checked, skipped, live_diff = 0, 0, 0.0
    for end in pd.date_range(index[test_start].normalize(), index[last_start + model.input_len - 1], freq="7D"):
        slots = fc.input_slots(end.date())
        if not slots.isin(index).all() or slots[-1] not in pos:
            continue
        i = int(pos[slots[0]])
        if i == 0 or index[i - 1] != slots[0] - pd.Timedelta(hours=12):
            continue
        for s, sid in enumerate(stations):
            wl, rain = csv[sid]
            try:
                ours = model.predict(sid, wl, rain, end.date())[0].to_numpy()
            except fc.InputError:
                skipped += 1
                continue
            live_diff = max(live_diff, float(np.abs(ours - reference(i, s)).max()))
            checked += 1
    print(f"[3] live path       max |diff| = {live_diff:.2e} m over {checked} windows "
          f"({skipped} skipped for WL gaps)")

    ok = tf_diff < 1e-5 and model_diff < TOLERANCE_M and checked and live_diff < TOLERANCE_M
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
