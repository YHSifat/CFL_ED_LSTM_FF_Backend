"""Combine prediction-time water-level and rainfall JSON data.

The output contains one row per water-level station and official observation
time. Rainfall is daily, so its value is repeated for each official time on
the same date. Missing observations are kept as NaN instead of being filled
with a synthetic value.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


OFFICIAL_TIMES = ("06:00:00", "09:00:00", "12:00:00", "15:00:00", "18:00:00")
WATER_FILE_PATTERN = re.compile(r"^wl_(?P<station>.+?)_\d{4}-\d{2}-\d{2}\.json$")
RAINFALL_FILE_PATTERN = re.compile(r"^rf_(?P<station>.+?)\.json$")


def snap_to_official_time(timestamp: pd.Timestamp) -> pd.Timestamp:
	"""Snap a timestamp to the nearest official Bangladesh observation time."""
	if pd.isna(timestamp):
		return pd.NaT

	candidates = [
		timestamp.normalize() + pd.Timedelta(hours=hour)
		for hour in (6, 9, 12, 15, 18)
	]
	candidates.extend(
		[
			timestamp.normalize() - pd.Timedelta(hours=6),
			timestamp.normalize() + pd.Timedelta(days=1, hours=6),
		]
	)
	return min(candidates, key=lambda candidate: abs(candidate - timestamp))


def load_station_mapping(mapping_path: Path) -> dict[str, str]:
	"""Return a water-level station to rainfall station mapping."""
	with mapping_path.open(encoding="utf-8") as mapping_file:
		mapping_rows = json.load(mapping_file)

	water_to_rainfall = {}
	for row in mapping_rows:
		rainfall_station = str(row["station_id"]).strip()
		for water_station in str(row.get("mapped_wl_station_id", "")).split(","):
			water_station = water_station.strip()
			if water_station:
				water_to_rainfall[water_station] = rainfall_station
				print(f"Mapping water-level station {water_station} to rainfall station {rainfall_station}")
	
	print(f"Loaded {len(water_to_rainfall)} water-level to rainfall station mappings from {mapping_path}")
	return water_to_rainfall


def load_rainfall(rainfall_dir: Path) -> pd.DataFrame:
	"""Load all rainfall files into one daily station-indexed table."""
	records = []
	for file_path in sorted(rainfall_dir.glob("rf_*.json")):
		match = RAINFALL_FILE_PATTERN.match(file_path.name)
		if not match:
			continue

		with file_path.open(encoding="utf-8") as rainfall_file:
			rows = json.load(rainfall_file)
		rainfall_station = match.group("station")
		for row in rows:
			records.append(
				{
					"rainfall_station_id": rainfall_station,
					"date": pd.to_datetime(
						row.get("rf_date"), format="mixed", dayfirst=False, errors="coerce"
					).normalize(),
					"rainfall": pd.to_numeric(row.get("rainfall"), errors="coerce"),
				}
			)

	if not records:
		return pd.DataFrame(columns=["rainfall_station_id", "date", "rainfall"])

	rainfall = pd.DataFrame(records).dropna(subset=["date"])
	return (
		rainfall.groupby(["rainfall_station_id", "date"], as_index=False, sort=True)["rainfall"]
		.mean()
	)


def load_water_levels(water_level_dir: Path) -> pd.DataFrame:
	"""Load, normalize, de-duplicate, and complete-grid water-level data."""
	records = []
	for file_path in sorted(water_level_dir.glob("wl_*.json")):
		match = WATER_FILE_PATTERN.match(file_path.name)
		if not match:
			continue

		with file_path.open(encoding="utf-8") as water_file:
			rows = json.load(water_file)
		station_id = match.group("station")
		for row in rows:
			timestamp = pd.to_datetime(row.get("wl_date"), format="mixed", errors="coerce", utc=True)
			if pd.notna(timestamp):
				timestamp = timestamp.tz_localize(None)
			records.append(
				{
					"station_id": station_id,
					"timestamp": snap_to_official_time(timestamp),
					"water_level": pd.to_numeric(row.get("waterlevel"), errors="coerce"),
				}
			)

	if not records:
		return pd.DataFrame(columns=["station_id", "date", "time", "timestamp", "water_level"])

	water_levels = pd.DataFrame(records).dropna(subset=["timestamp"])
	water_levels = (
		water_levels.groupby(["station_id", "timestamp"], as_index=False, sort=True)["water_level"]
		.mean()
	)
	water_levels["date"] = water_levels["timestamp"].dt.normalize()

	completed = []
	for station_id, station_data in water_levels.groupby("station_id", sort=True):
		dates = pd.date_range(
			station_data["date"].min(), station_data["date"].max(), freq="D"
		)
		grid = pd.MultiIndex.from_product(
			[dates, pd.to_timedelta([f"{time[:2]}:00:00" for time in OFFICIAL_TIMES])],
			names=["date", "time_delta"],
		).to_frame(index=False)
		grid["timestamp"] = grid["date"] + grid["time_delta"]
		grid["station_id"] = station_id
		completed.append(grid.merge(station_data, on=["station_id", "timestamp", "date"], how="left"))

	water_levels = pd.concat(completed, ignore_index=True)
	water_levels["date"] = water_levels["date"].dt.strftime("%Y-%m-%d")
	water_levels["time"] = water_levels["timestamp"].dt.strftime("%H:%M:%S")
	return water_levels[["station_id", "date", "time", "timestamp", "water_level"]]


def fill_missing_features(dataset: pd.DataFrame) -> pd.DataFrame:
	"""Apply the model's numeric missing-value policy station by station."""
	dataset = dataset.sort_values(["station_id", "timestamp"]).copy()
	for column in ("waterlevel", "rainfall"):
		dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
		dataset[column] = (
			dataset.groupby("station_id", sort=False)[column]
			.transform(lambda values: values.interpolate(limit_direction="both"))
		)
		dataset[column] = (
			dataset.groupby("station_id", sort=False)[column]
			.transform(lambda values: values.ffill().bfill())
			.fillna(0.0)
		.astype("float32")
		)
	return dataset


def build_dataset(base_dir: Path) -> pd.DataFrame:
	"""Build the combined dataset from the prediction raw-data directory."""
	prediction_dir = base_dir / "Prediction"
	mapping = load_station_mapping(prediction_dir / "raw_data" / "rainfall_stations.json")
	water_levels = load_water_levels(prediction_dir / "raw_data" / "wl")
	rainfall = load_rainfall(prediction_dir / "raw_data" / "rainfall")

	water_levels["rainfall_station_id"] = water_levels["station_id"].map(mapping)
	water_levels["date_key"] = pd.to_datetime(water_levels["date"], errors="coerce")
	combined = water_levels.merge(
		rainfall,
		left_on=["rainfall_station_id", "date_key"],
		right_on=["rainfall_station_id", "date"],
		how="left",
		suffixes=("", "_rainfall"),
	)
	combined = combined.rename(columns={"water_level": "waterlevel"})
	combined = fill_missing_features(combined)
	combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
	combined["time"] = combined["timestamp"].dt.strftime("%H:%M:%S")
	combined = combined.rename(
		columns={"waterlevel": "WL(mMSL)", "rainfall": "RAINFALL(mm)"}
	)
	return combined[
		[
			"station_id", "date", "time", "timestamp",
			"WL(mMSL)", "RAINFALL(mm)",
		]
	].sort_values(["station_id", "timestamp"], ignore_index=True)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output CSV path (default: Prediction/input_csv/processed_data.csv)",
	)
	args = parser.parse_args()

	script_dir = Path(__file__).resolve().parent
	output_path = args.output or script_dir / "input_csv" / "processed_data.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataset = build_dataset(script_dir.parent)
	dataset.to_csv(output_path, index=False)
	print(f"Wrote {len(dataset):,} rows for {dataset['station_id'].nunique()} stations to {output_path}")


if __name__ == "__main__":
	main()
