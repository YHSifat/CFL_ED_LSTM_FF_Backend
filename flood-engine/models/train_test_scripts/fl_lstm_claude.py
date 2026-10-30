import glob
import random
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Datasets" / "Processed" / "Station_wise_combined"
TRAINING_RESULT_DIR = PROJECT_ROOT / "models" / "main" / "training_result"

# ============================================================
# Experiment config (single source of truth)
# ============================================================
LOCAL_WEIGHT_AGGREGATION = 0.5   # alpha: 0.0 = start each round from global model, 1.0 = pure local
RUN_NAME = f"fl_vanilla_lstm-local-weighted-{LOCAL_WEIGHT_AGGREGATION}_fedavg"
CHECKPOINT_PATH = PROJECT_ROOT / f"clustered_{RUN_NAME}.pt"
LOG_FILENAME = f"{RUN_NAME}.txt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DAYS = 30
FORECAST_DAYS = 15
OBS_PER_DAY = 5
INPUT_LEN = INPUT_DAYS * OBS_PER_DAY
OUTPUT_LEN = FORECAST_DAYS * OBS_PER_DAY

BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5               # try 1e-5 .. 1e-4

FEDERATED_ALGORITHM = "fedavg"   # "fedavg" or "fedprox"
FEDERATED_ROUNDS = 30
LOCAL_EPOCHS = 2
LOCAL_LEARNING_RATE = 1e-3
FEDPROX_MU = 0.01
EARLY_STOPPING_PATIENCE = 6

DATASET_SPLIT_RATIO = (0.7, 0.15, 0.15)

# Improvement toggles (turn off one at a time for your ablation table)
PER_STATION_SCALING = True       # normalize each station with its own mean/std
RESIDUAL_FORECAST = True         # model predicts change from last observed water level
LOG_RAINFALL = True              # log1p transform on rainfall
MAX_WL_INTERP_STEPS = OBS_PER_DAY * 2   # interpolate WL gaps up to ~2 days only

WL_COLUMN = "WL(mMSL)"
RAIN_COLUMN = "RAINFALL(mm)"
FEATURE_COLUMNS = [WL_COLUMN, RAIN_COLUMN]
TIME_COLUMNS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "delta_hours"]

CLUSTERS = {
	"c1": ("SW291.5R", "SW294", "SW77", "SW62", "SW45", "SW45.5"),
	"c2": ("SW63", "SW97", "SW46.9R", "SW65", "SW49", "SW49A", "SW15J", "SW225"),
	"c3": ("SW151", "SW147.5", "SW228.5", "SW68A", "SW137A"),
	"c4": ("SW68.5", "SW42", "SW302", "SW179", "SW180", "SW7.5"),
}


# ============================================================
# Utilities
# ============================================================
class _Tee:
	def __init__(self, console, file_handle):
		self.console = console
		self.file_handle = file_handle

	def write(self, text):
		self.console.write(text)
		self.file_handle.write(text)
		self.file_handle.flush()

	def flush(self):
		self.console.flush()
		self.file_handle.flush()


@contextmanager
def capture_output(filename):
	TRAINING_RESULT_DIR.mkdir(parents=True, exist_ok=True)
	output_path = TRAINING_RESULT_DIR / filename
	with output_path.open("w", encoding="utf-8") as file_handle:
		original_stdout = sys.stdout
		sys.stdout = _Tee(original_stdout, file_handle)
		try:
			yield output_path
		finally:
			sys.stdout = original_stdout


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def validate_clusters(stations):
	cluster_stations = [station for cluster in CLUSTERS.values() for station in cluster]
	if len(cluster_stations) != len(set(cluster_stations)):
		duplicated = sorted(s for s in set(cluster_stations) if cluster_stations.count(s) > 1)
		raise ValueError("Stations are assigned to more than one cluster: " + ", ".join(duplicated))
	missing = sorted(set(stations) - set(cluster_stations))
	extra = sorted(set(cluster_stations) - set(stations))
	if missing or extra:
		message = []
		if missing:
			message.append("missing from clusters: " + ", ".join(missing))
		if extra:
			message.append("not found in processed data: " + ", ".join(extra))
		raise ValueError("Invalid cluster definition (" + "; ".join(message) + ")")


# ============================================================
# Data
# ============================================================
def load_data():
	paths = sorted(
		path for path in glob.glob(str(DATA_DIR / "*.csv"))
		if Path(path).name != "station_descriptive_statistics.csv"
	)
	if not paths:
		raise FileNotFoundError(f"No station CSV files found in {DATA_DIR!s}")

	station_frames = {}
	for path in paths:
		station_id = Path(path).stem
		frame = pd.read_csv(path)
		frame.columns = frame.columns.str.strip()
		missing_columns = [c for c in FEATURE_COLUMNS if c not in frame]
		if missing_columns:
			raise ValueError(f"{path} is missing columns: {missing_columns}")
		frame["timestamp"] = pd.to_datetime(
			frame["date"].astype(str) + " " + frame["time"].astype(str), errors="coerce"
		)
		frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
		frame = frame[~frame.index.duplicated(keep="first")]
		station_frames[station_id] = (
			frame[FEATURE_COLUMNS]
			.apply(pd.to_numeric, errors="coerce")
			.replace([np.inf, -np.inf], np.nan)
		)

	stations = sorted(station_frames)
	common_index = station_frames[stations[0]].index
	for station in stations[1:]:
		common_index = common_index.intersection(station_frames[station].index)
	common_index = common_index.sort_values()
	if len(common_index) < INPUT_LEN + OUTPUT_LEN:
		raise ValueError("Not enough common timestamps for one training window.")

	print("\nMissing-value handling:")
	arrays = []
	for station in stations:
		frame = station_frames[station].reindex(common_index).copy()
		wl_missing = int(frame[WL_COLUMN].isna().sum())
		frame[WL_COLUMN] = frame[WL_COLUMN].interpolate(
			limit=MAX_WL_INTERP_STEPS, limit_direction="both"
		)
		wl_long_gap = int(frame[WL_COLUMN].isna().sum())
		frame[WL_COLUMN] = frame[WL_COLUMN].ffill().bfill()
		rain_missing = int(frame[RAIN_COLUMN].isna().sum())
		# Rainfall is not interpolated: a missing reading is treated as no rain.
		frame[RAIN_COLUMN] = frame[RAIN_COLUMN].fillna(0.0).clip(lower=0.0)
		if wl_missing or rain_missing:
			print(
				f"  {station}: WL missing {wl_missing} "
				f"(beyond {MAX_WL_INTERP_STEPS}-step interpolation: {wl_long_gap}), "
				f"rain missing {rain_missing} -> 0"
			)
		arrays.append(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

	values = np.stack(arrays, axis=1)

	gap_hours = common_index.to_series().diff().dt.total_seconds().div(3600.0).dropna()
	expected = 24.0 / OBS_PER_DAY
	print("\nSampling regularity of the common timeline:")
	print(f"  Median gap: {gap_hours.median():.2f} h (expected ~{expected:.2f} h)")
	print(f"  Gaps > 2x expected: {(gap_hours > 2 * expected).mean() * 100:.2f}%")
	print(f"  Gaps > 1 day: {(gap_hours > 24).sum()} | Max gap: {gap_hours.max():.1f} h")

	time_features = pd.DataFrame(index=common_index)
	hour = common_index.hour + common_index.minute / 60.0
	day_of_year = common_index.dayofyear
	time_features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
	time_features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
	time_features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
	time_features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
	time_features["delta_hours"] = (
		common_index.to_series().diff().dt.total_seconds().fillna(0) / 3600.0 / 12.0
	).to_numpy()
	time_values = time_features[TIME_COLUMNS].to_numpy(dtype=np.float32)

	if not np.isfinite(values).all() or not np.isfinite(time_values).all():
		raise ValueError("Non-finite values remain after data preparation.")
	return stations, common_index, values, time_values


def load_station_data():
	stations, timestamps, raw_features, time_features = load_data()
	targets = raw_features[:, :, 0].copy()
	model_inputs = raw_features.copy()
	if LOG_RAINFALL:
		model_inputs[:, :, 1] = np.log1p(model_inputs[:, :, 1])
	features = np.concatenate(
		[model_inputs, np.repeat(time_features[:, None, :], len(stations), axis=1)], axis=2
	)
	feature_columns = FEATURE_COLUMNS + TIME_COLUMNS
	return stations, timestamps, features, targets, feature_columns


def fit_scaler(features, train_end):
	"""Returns mean/std of shape (stations, features), fitted on the training period only."""
	train = features[:train_end].astype(np.float64)
	n_stations, n_features = features.shape[1], features.shape[2]
	if PER_STATION_SCALING:
		mean = train.mean(axis=0)
		std = train.std(axis=0)
	else:
		flat = train.reshape(-1, n_features)
		mean = np.broadcast_to(flat.mean(axis=0), (n_stations, n_features)).copy()
		std = np.broadcast_to(flat.std(axis=0), (n_stations, n_features)).copy()
	std = np.where(std < 1e-6, 1.0, std)
	return mean.astype(np.float32), std.astype(np.float32)


class StationWindowDataset(Dataset):
	"""Windows whose forecast targets lie in [target_start, end).

	Inputs may look back before target_start (into the previous split). This uses
	only past observations, so it is not leakage, and it avoids discarding the
	first INPUT_LEN steps of the validation and test periods.
	"""

	def __init__(self, features, targets, station_indices, target_start, end):
		first = max(0, target_start - INPUT_LEN)
		last = end - INPUT_LEN - OUTPUT_LEN
		windows, window_targets = [], []
		for station_index in station_indices:
			for index in range(first, last + 1):
				input_end = index + INPUT_LEN
				windows.append(features[index:input_end, station_index, :])
				window_targets.append(targets[input_end:input_end + OUTPUT_LEN, station_index])
		if not windows:
			raise ValueError("Not enough observations for a station window.")
		self.windows = torch.tensor(np.stack(windows), dtype=torch.float32)
		self.targets = torch.tensor(np.stack(window_targets), dtype=torch.float32)

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, index):
		return self.windows[index], self.targets[index]


# ============================================================
# Model
# ============================================================
class VanillaLSTM(nn.Module):
	def __init__(self, input_size):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=HIDDEN_SIZE,
			num_layers=NUM_LAYERS,
			batch_first=True,
			dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
		)
		self.forecast_head = nn.Sequential(
			nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
			nn.ReLU(),
			nn.Dropout(DROPOUT),
			nn.Linear(HIDDEN_SIZE, OUTPUT_LEN),
		)
		if RESIDUAL_FORECAST:
			# Start exactly at the persistence forecast; training learns the corrections.
			nn.init.zeros_(self.forecast_head[-1].weight)
			nn.init.zeros_(self.forecast_head[-1].bias)

	def forward(self, inputs):
		encoded, _ = self.lstm(inputs)
		output = self.forecast_head(encoded[:, -1])
		if RESIDUAL_FORECAST:
			# Feature 0 is WL scaled with the same statistics as the target,
			# so it can be added directly as the persistence baseline.
			output = output + inputs[:, -1, 0:1]
		return output


# ============================================================
# Federated helpers
# ============================================================
def trainable_state(model):
	return OrderedDict(
		(name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
	)


def clone_state(state):
	return OrderedDict((name, value.clone()) for name, value in state.items())


def mix_states(local_state, global_state, alpha):
	mixed = OrderedDict()
	for name, global_value in global_state.items():
		if torch.is_floating_point(global_value):
			mixed[name] = alpha * local_state[name] + (1.0 - alpha) * global_value
		else:
			mixed[name] = global_value.clone()
	return mixed


def load_trainable_state(model, state):
	current = model.state_dict()
	for name, value in state.items():
		if name in current:
			current[name] = value.to(DEVICE)
	model.load_state_dict(current)


def federated_average(global_state, client_states, client_weights):
	if not client_states or sum(client_weights) <= 0:
		raise ValueError("At least one non-empty client is required for federated averaging.")
	total_weight = float(sum(client_weights))
	averaged = OrderedDict()
	for name in global_state:
		if not torch.is_floating_point(global_state[name]):
			averaged[name] = client_states[0][name].clone()
			continue
		value = torch.zeros_like(global_state[name], dtype=torch.float32)
		for client_state, weight in zip(client_states, client_weights):
			value += client_state[name].float() * (weight / total_weight)
		averaged[name] = value.to(global_state[name].dtype)
	return averaged


def train_local_model(model, loader, criterion, local_epochs, local_learning_rate,
                      global_state=None, proximal_mu=0.0):
	model.train()
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=local_learning_rate, weight_decay=WEIGHT_DECAY
	)
	anchor = None
	if global_state is not None and proximal_mu > 0.0:
		# Move the FedProx anchor to the device once, not on every batch.
		anchor = {name: global_state[name].to(DEVICE) for name, _ in model.named_parameters()}

	epoch_loss = 0.0
	for _ in range(local_epochs):
		epoch_loss = 0.0
		for inputs, targets in loader:
			inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
			optimizer.zero_grad()
			prediction_loss = criterion(model(inputs), targets)
			loss = prediction_loss
			if anchor is not None:
				penalty = sum(
					torch.sum((parameter - anchor[name]) ** 2)
					for name, parameter in model.named_parameters()
				)
				loss = loss + 0.5 * proximal_mu * penalty
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			epoch_loss += prediction_loss.item()   # report prediction loss only
	return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
	model.eval()
	total_loss = 0.0
	with torch.no_grad():
		for inputs, targets in loader:
			inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
			total_loss += criterion(model(inputs), targets).item()
	return total_loss / len(loader)


# ============================================================
# Test evaluation
# ============================================================
def regression_metrics(actual, predicted):
	error = predicted - actual
	mae = float(np.mean(np.abs(error)))
	rmse = float(np.sqrt(np.mean(error ** 2)))
	denominator = float(np.sum((actual - actual.mean()) ** 2))
	nse = 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 0 else float("nan")
	return mae, rmse, nse


def evaluate_test(cluster_models, features, targets, mean, std, station_to_index, validation_end):
	target_mean, target_std = mean[:, 0], std[:, 0]
	all_actual, all_model, all_persist = [], [], []
	print("\nPer-station test results (metres):")
	print(f"  {'cluster':<7} {'station':<10} {'MAE':>7} {'RMSE':>7} {'NSE':>7} | "
	      f"{'Pers MAE':>8} {'Pers RMSE':>9} {'Skill':>6}")
	for cluster_name, model in cluster_models.items():
		model.eval()
		for station in CLUSTERS[cluster_name]:
			s = station_to_index[station]
			dataset = StationWindowDataset(features, targets, [s], validation_end, len(features))
			loader = DataLoader(dataset, batch_size=BATCH_SIZE * 4, shuffle=False)
			predictions, actual, persistence = [], [], []
			with torch.no_grad():
				for inputs, y in loader:
					predictions.append(model(inputs.to(DEVICE)).cpu().numpy())
					actual.append(y.numpy())
					last = inputs[:, -1, 0:1].numpy()
					persistence.append(np.repeat(last, OUTPUT_LEN, axis=1))
			scale, shift = target_std[s], target_mean[s]
			predictions = np.concatenate(predictions) * scale + shift
			actual = np.concatenate(actual) * scale + shift
			persistence = np.concatenate(persistence) * scale + shift

			mae, rmse, nse = regression_metrics(actual, predictions)
			p_mae, p_rmse, _ = regression_metrics(actual, persistence)
			skill = 1.0 - mae / p_mae if p_mae > 0 else float("nan")
			print(f"  {cluster_name:<7} {station:<10} {mae:7.4f} {rmse:7.4f} {nse:7.3f} | "
			      f"{p_mae:8.4f} {p_rmse:9.4f} {skill:6.2f}")
			all_actual.append(actual)
			all_model.append(predictions)
			all_persist.append(persistence)

	actual = np.concatenate(all_actual)
	predictions = np.concatenate(all_model)
	persistence = np.concatenate(all_persist)
	mae, rmse, nse = regression_metrics(actual, predictions)
	p_mae, p_rmse, p_nse = regression_metrics(actual, persistence)
	print(f"\nTest MAE: {mae:.4f} m | Test RMSE: {rmse:.4f} m")
	print(f"Persistence MAE: {p_mae:.4f} m | Persistence RMSE: {p_rmse:.4f} m")
	print(f"Skill vs persistence (MAE): {1.0 - mae / p_mae:.3f}")

	daily_mae = np.abs(predictions - actual).reshape(-1, FORECAST_DAYS, OBS_PER_DAY).mean(axis=(0, 2))
	daily_p_mae = np.abs(persistence - actual).reshape(-1, FORECAST_DAYS, OBS_PER_DAY).mean(axis=(0, 2))
	print("\nMAE by lead day (model | persistence):")
	for day, (m, p) in enumerate(zip(daily_mae, daily_p_mae), start=1):
		print(f"  Day {day:2d}: {m:.4f} | {p:.4f}")


# ============================================================
# Main
# ============================================================
def main():
	if FEDERATED_ALGORITHM not in ("fedavg", "fedprox"):
		raise ValueError("FEDERATED_ALGORITHM must be 'fedavg' or 'fedprox'.")
	set_seed(SEED)
	print("Device:", DEVICE)

	stations, timestamps, features, targets, feature_columns = load_station_data()
	validate_clusters(stations)
	station_to_index = {station: index for index, station in enumerate(stations)}
	train_end = int(len(features) * DATASET_SPLIT_RATIO[0])
	validation_end = int(len(features) * (DATASET_SPLIT_RATIO[0] + DATASET_SPLIT_RATIO[1]))

	mean, std = fit_scaler(features, train_end)
	features = ((features - mean) / std).astype(np.float32)
	# The target uses the WL feature's statistics, so scaled feature 0 == scaled target.
	targets = ((targets - mean[:, 0]) / std[:, 0]).astype(np.float32)

	print(f"\nLoaded data for {len(stations)} stations with {len(timestamps)} observations.")
	print(f"Ratio:- Train: {DATASET_SPLIT_RATIO[0]}, Validation: {DATASET_SPLIT_RATIO[1]}, "
	      f"Test: {DATASET_SPLIT_RATIO[2]}")
	print("\nModel configuration:")
	print("  Model: Clustered Federated Vanilla LSTM")
	print(f"  Federated algorithm: {FEDERATED_ALGORITHM}"
	      + (f" (mu={FEDPROX_MU})" if FEDERATED_ALGORITHM == "fedprox" else ""))
	print(f"  Federated rounds: {FEDERATED_ROUNDS} (early stopping patience {EARLY_STOPPING_PATIENCE})")
	print(f"  Local epochs: {LOCAL_EPOCHS}")
	print(f"  Local learning rate: {LOCAL_LEARNING_RATE} | Weight decay: {WEIGHT_DECAY}")
	print(f"  Batch size: {BATCH_SIZE}")
	print(f"  Input length: {INPUT_LEN} time steps | Output length: {OUTPUT_LEN} time steps")
	print(f"  Hidden size: {HIDDEN_SIZE} | LSTM layers: {NUM_LAYERS} | Dropout: {DROPOUT}")
	print(f"  Local weight aggregation (alpha): {LOCAL_WEIGHT_AGGREGATION}")
	print(f"  Per-station scaling: {PER_STATION_SCALING} | Residual forecast: {RESIDUAL_FORECAST} "
	      f"| log1p rainfall: {LOG_RAINFALL}")
	print(f"  Seed: {SEED}")
	print("\nTraining data summary:")
	print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
	print(f"  Input features: {features.shape[-1]} ({', '.join(feature_columns)})")
	print(f"  Feature shape: {features.shape} | Target shape: {targets.shape}")
	print(f"  Train / Validation / Test observations: {train_end} / "
	      f"{validation_end - train_end} / {len(features) - validation_end}")

	clients = {}
	for cluster_name, cluster_stations in CLUSTERS.items():
		indices = [station_to_index[station] for station in cluster_stations]
		train_dataset = StationWindowDataset(features, targets, indices, 0, train_end)
		validation_dataset = StationWindowDataset(features, targets, indices, train_end, validation_end)
		clients[cluster_name] = {
			"stations": cluster_stations,
			"train_loader": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
			"validation_loader": DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
		}

	input_size = features.shape[-1]
	global_model = VanillaLSTM(input_size).to(DEVICE)
	criterion = nn.HuberLoss()
	global_state = trainable_state(global_model)

	# Persistent per-cluster weights. These must be created ONCE, outside the
	# round loop, otherwise every round starts from the global model and
	# alpha has no effect.
	cluster_states = {name: clone_state(global_state) for name in clients}

	best_validation_loss = float("inf")
	best_round = 0
	rounds_without_improvement = 0

	for round_number in tqdm(range(1, FEDERATED_ROUNDS + 1), desc="Federated rounds", unit="round"):
		client_states, client_weights = [], []
		local_models, train_loss_by_cluster = {}, {}

		for cluster_name, client in clients.items():
			local_model = VanillaLSTM(input_size).to(DEVICE)
			start_state = mix_states(cluster_states[cluster_name], global_state, LOCAL_WEIGHT_AGGREGATION)
			load_trainable_state(local_model, start_state)
			train_loss = train_local_model(
				local_model,
				client["train_loader"],
				criterion,
				local_epochs=LOCAL_EPOCHS,
				local_learning_rate=LOCAL_LEARNING_RATE,
				global_state=global_state if FEDERATED_ALGORITHM == "fedprox" else None,
				proximal_mu=FEDPROX_MU if FEDERATED_ALGORITHM == "fedprox" else 0.0,
			)
			trained_state = trainable_state(local_model)
			cluster_states[cluster_name] = trained_state
			client_states.append(trained_state)
			client_weights.append(len(client["train_loader"].dataset))
			local_models[cluster_name] = local_model
			train_loss_by_cluster[cluster_name] = train_loss

		# FedProx changes only the client objective; server aggregation is FedAvg.
		global_state = federated_average(global_state, client_states, client_weights)
		load_trainable_state(global_model, global_state)

		validation_total, validation_weight = 0.0, 0
		for cluster_name, client in clients.items():
			validation_loss = evaluate(local_models[cluster_name], client["validation_loader"], criterion)
			weight = len(client["validation_loader"].dataset)
			validation_total += validation_loss * weight
			validation_weight += weight
			tqdm.write(
				f"Round {round_number:03d} | {cluster_name} | "
				f"Train Loss: {train_loss_by_cluster[cluster_name]:.6f} | "
				f"Evaluation Loss: {validation_loss:.6f}"
			)
		train_loss = sum(
			train_loss_by_cluster[name] * client_weights[i] for i, name in enumerate(clients)
		) / sum(client_weights)
		validation_loss = validation_total / validation_weight
		print(f"Round {round_number:03d} | Train Loss: {train_loss:.6f} | Evaluation Loss: {validation_loss:.6f}")

		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			best_round = round_number
			rounds_without_improvement = 0
			torch.save({
				"model_state_dict": global_model.state_dict(),
				"cluster_model_state_dicts": {name: trainable_state(m) for name, m in local_models.items()},
				"scaler_mean": mean,
				"scaler_std": std,
				"stations": stations,
				"clusters": CLUSTERS,
				"feature_columns": feature_columns,
				"input_len": INPUT_LEN,
				"output_len": OUTPUT_LEN,
				"model_type": "Clustered-FL-Vanilla-LSTM",
				"hidden_size": HIDDEN_SIZE,
				"num_layers": NUM_LAYERS,
				"local_weight_aggregation": LOCAL_WEIGHT_AGGREGATION,
				"per_station_scaling": PER_STATION_SCALING,
				"residual_forecast": RESIDUAL_FORECAST,
				"log_rainfall": LOG_RAINFALL,
				"best_round": best_round,
			}, CHECKPOINT_PATH)
		else:
			rounds_without_improvement += 1
			if rounds_without_improvement >= EARLY_STOPPING_PATIENCE:
				print(f"Early stopping at round {round_number:03d}.")
				break

	print(f"\nBest round: {best_round:03d} | Best evaluation loss: {best_validation_loss:.6f}")

	checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
	cluster_models = {}
	for cluster_name in clients:
		model = VanillaLSTM(input_size).to(DEVICE)
		load_trainable_state(model, checkpoint["cluster_model_state_dicts"][cluster_name])
		cluster_models[cluster_name] = model

	evaluate_test(cluster_models, features, targets, mean, std, station_to_index, validation_end)
	print(f"\nSaved federated vanilla LSTM model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
	with capture_output(LOG_FILENAME):
		main()