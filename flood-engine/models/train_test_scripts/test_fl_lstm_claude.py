
"""Evaluate the clustered federated vanilla LSTM on held-out station windows.

Matches fl_vanilla_lstm_local_weighted.py (the revised training script):
- reuses its data loader, dataset and model so preprocessing is identical
- restores preprocessing flags, split ratio and per-station scaler from the checkpoint
- reports per-station metrics, a persistence baseline and MAE by lead day
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# The revised training script, saved as models/main/fl_vanilla_lstm_local_weighted.py
from models.main import fl_lstm_claude as train_module
from models.test.common import SAVED_DIR, capture_output, report


CHECKPOINT_NAME = "clustered_fl_vanilla_lstm-local-weighted-0.5.pt"
CHECKPOINT_PATH = SAVED_DIR / CHECKPOINT_NAME
LOG_FILENAME = "test_clustered_fl_vanilla_lstm-local-weighted-0.5_split_70.txt"
BATCH_SIZE = 256

# checkpoint key -> training-module attribute it controls
CHECKPOINT_SETTINGS = {
	"per_station_scaling": "PER_STATION_SCALING",
	"residual_forecast": "RESIDUAL_FORECAST",
	"log_rainfall": "LOG_RAINFALL",
	"max_wl_interp_steps": "MAX_WL_INTERP_STEPS",
	"hidden_size": "HIDDEN_SIZE",
	"num_layers": "NUM_LAYERS",
	"dropout": "DROPOUT",
}


class _IdentityScaler:
	"""Values passed to report() are already in metres; per-station scaling cannot
	be expressed as one sklearn scaler, so inverse-transform is done here instead."""

	def transform(self, values):
		return values

	def inverse_transform(self, values):
		return values


def resolve_checkpoint_path():
	if CHECKPOINT_PATH.exists():
		return CHECKPOINT_PATH
	if train_module.CHECKPOINT_PATH.exists():
		return train_module.CHECKPOINT_PATH
	raise FileNotFoundError(
		f"Checkpoint not found at {CHECKPOINT_PATH} or {train_module.CHECKPOINT_PATH}"
	)


def apply_checkpoint_settings(checkpoint):
	"""Make the training module's loader and model use the settings the checkpoint was trained with."""
	applied = {}
	for key, attribute in CHECKPOINT_SETTINGS.items():
		if key in checkpoint:
			setattr(train_module, attribute, checkpoint[key])
			applied[attribute] = checkpoint[key]
		else:
			applied[attribute] = f"{getattr(train_module, attribute)} (not in checkpoint, using script default)"
	if checkpoint.get("input_len", train_module.INPUT_LEN) != train_module.INPUT_LEN:
		raise ValueError("Checkpoint input_len does not match INPUT_LEN.")
	if checkpoint.get("output_len", train_module.OUTPUT_LEN) != train_module.OUTPUT_LEN:
		raise ValueError("Checkpoint output_len does not match OUTPUT_LEN.")
	return applied


def predict_test_windows(checkpoint, features, targets, stations, validation_end):
	"""Returns predictions, actuals and persistence in metres, plus the station of each window."""
	mean, std = checkpoint["scaler_mean"], checkpoint["scaler_std"]
	target_mean, target_std = mean[:, 0], std[:, 0]
	station_to_index = {station: index for index, station in enumerate(stations)}
	device = train_module.DEVICE

	predictions, actual, persistence, window_stations = [], [], [], []
	for cluster_name, cluster_stations in checkpoint["clusters"].items():
		model = train_module.VanillaLSTM(features.shape[-1]).to(device)
		train_module.load_trainable_state(model, checkpoint["cluster_model_state_dicts"][cluster_name])
		model.eval()

		for station in cluster_stations:
			s = station_to_index[station]
			dataset = train_module.StationWindowDataset(features, targets, [s], validation_end, len(features))
			station_predictions, station_persistence = [], []
			with torch.no_grad():
				for start in range(0, len(dataset), BATCH_SIZE):
					batch = dataset.windows[start:start + BATCH_SIZE]
					station_predictions.append(model(batch.to(device)).cpu().numpy())
					last = batch[:, -1, 0:1].numpy()
					station_persistence.append(np.repeat(last, train_module.OUTPUT_LEN, axis=1))

			scale, shift = target_std[s], target_mean[s]
			predictions.append(np.concatenate(station_predictions) * scale + shift)
			persistence.append(np.concatenate(station_persistence) * scale + shift)
			actual.append(dataset.targets.numpy() * scale + shift)
			window_stations.extend([station] * len(dataset))

	return (
		np.concatenate(predictions),
		np.concatenate(actual),
		np.concatenate(persistence),
		window_stations,
	)


def print_detailed_metrics(predictions, actual, persistence, window_stations, clusters):
	metrics = train_module.regression_metrics
	station_array = np.asarray(window_stations)

	print("\nPer-station test results (metres):")
	print(f"  {'cluster':<7} {'station':<10} {'MAE':>7} {'RMSE':>7} {'NSE':>7} | "
	      f"{'Pers MAE':>8} {'Pers RMSE':>9} {'Skill':>6}")
	for cluster_name, cluster_stations in clusters.items():
		for station in cluster_stations:
			mask = station_array == station
			mae, rmse, nse = metrics(actual[mask], predictions[mask])
			p_mae, p_rmse, _ = metrics(actual[mask], persistence[mask])
			skill = 1.0 - mae / p_mae if p_mae > 0 else float("nan")
			print(f"  {cluster_name:<7} {station:<10} {mae:7.4f} {rmse:7.4f} {nse:7.3f} | "
			      f"{p_mae:8.4f} {p_rmse:9.4f} {skill:6.2f}")

	print("\nPer-cluster test results (metres):")
	for cluster_name, cluster_stations in clusters.items():
		mask = np.isin(station_array, cluster_stations)
		mae, rmse, nse = metrics(actual[mask], predictions[mask])
		p_mae, _, _ = metrics(actual[mask], persistence[mask])
		print(f"  {cluster_name}: MAE {mae:.4f} | RMSE {rmse:.4f} | NSE {nse:.3f} | "
		      f"Persistence MAE {p_mae:.4f} | Skill {1.0 - mae / p_mae:.3f}")

	mae, rmse, nse = metrics(actual, predictions)
	p_mae, p_rmse, p_nse = metrics(actual, persistence)
	print(f"\nOverall: MAE {mae:.4f} m | RMSE {rmse:.4f} m | NSE {nse:.3f}")
	print(f"Persistence: MAE {p_mae:.4f} m | RMSE {p_rmse:.4f} m | NSE {p_nse:.3f}")
	print(f"Skill vs persistence (MAE): {1.0 - mae / p_mae:.3f}")

	days, per_day = train_module.FORECAST_DAYS, train_module.OBS_PER_DAY
	daily_mae = np.abs(predictions - actual).reshape(-1, days, per_day).mean(axis=(0, 2))
	daily_p_mae = np.abs(persistence - actual).reshape(-1, days, per_day).mean(axis=(0, 2))
	print("\nMAE by lead day (model | persistence):")
	for day, (m, p) in enumerate(zip(daily_mae, daily_p_mae), start=1):
		print(f"  Day {day:2d}: {m:.4f} | {p:.4f}")


def main():
	checkpoint_path = resolve_checkpoint_path()
	checkpoint = torch.load(checkpoint_path, map_location=train_module.DEVICE, weights_only=False)
	for key in ("cluster_model_state_dicts", "scaler_mean", "scaler_std"):
		if key not in checkpoint:
			raise KeyError(
				f"Checkpoint is missing '{key}'. It was probably made by the old training "
				"script; retrain with fl_vanilla_lstm_local_weighted.py."
			)
	applied = apply_checkpoint_settings(checkpoint)

	stations, timestamps, features, targets, feature_columns = train_module.load_station_data()
	train_module.validate_clusters(stations)
	if checkpoint.get("stations", stations) != stations:
		raise ValueError("Checkpoint stations do not match the current dataset.")
	if checkpoint.get("feature_columns", feature_columns) != feature_columns:
		raise ValueError("Checkpoint feature columns do not match the current dataset.")

	mean, std = checkpoint["scaler_mean"], checkpoint["scaler_std"]
	if mean.shape != (features.shape[1], features.shape[2]):
		raise ValueError(f"Scaler shape {mean.shape} does not match data {features.shape[1:]}.")

	split = (0.7,0.15)
	validation_end = int(len(features) * (split[0] + split[1]))

	# Use the training-period statistics stored in the checkpoint; never refit on test data.
	features = ((features - mean) / std).astype(np.float32)
	targets = ((targets - mean[:, 0]) / std[:, 0]).astype(np.float32)

	print(f"Device: {train_module.DEVICE}")
	print(f"Checkpoint: {checkpoint_path}")
	print(f"Best training round: {checkpoint.get('best_round', 'unknown')}")
	print(f"Local weight aggregation (alpha): {checkpoint.get('local_weight_aggregation', 'unknown')}")
	print("Settings restored from checkpoint:")
	for attribute, value in applied.items():
		print(f"  {attribute}: {value}")
	print(f"Split ratio: {tuple(split)} | Test period: {timestamps[validation_end]} to {timestamps[-1]}")
	print(f"Feature columns: {len(feature_columns)} ({', '.join(feature_columns)})")

	predictions, actual, persistence, window_stations = predict_test_windows(
		checkpoint, features, targets, stations, validation_end
	)
	print(f"Test windows: {len(actual)} | Stations: {len(stations)}")

	print_detailed_metrics(predictions, actual, persistence, window_stations, checkpoint["clusters"])

	print("\nShared report:")
	report(predictions, actual, _IdentityScaler(), station_ids=window_stations)


if __name__ == "__main__":
	with capture_output(LOG_FILENAME):
		main()