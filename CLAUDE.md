# Flood engine

Backend for a Bangladesh flood forecasting website (modelled on ffwc.gov.bd).
The frontend is already built and hosted separately; it reads from Supabase.

All code is in `flood-engine/`; see @README.md for setup and the frontend queries.

## Architecture
- **Local engine (this repo, Python)**: scrapes FFWC, runs a pretrained water level (WL)
  model, writes everything to Supabase Postgres.
- **Supabase (free tier)**: Postgres plus auto REST API. The frontend uses the anon key, which is
  read-only through RLS. The engine connects as `postgres` (bypasses RLS) via the
  Session pooler URL in `.env`.
- **Admin panel**: `uvicorn admin:app --host 127.0.0.1 --port 8000`. No auth, so it stays on localhost.
- Must stay within free tiers (Supabase: 500 MB DB, pauses after 7 idle days, no backups).

## Files
- `schema.sql`: tables, views, RLS. Idempotent, so re-run after edits. Views are dropped and recreated.
- `scraper.py`: FFWC API client and parsers (pure functions, no DB).
- `ffwc_pdf.py`: parses FFWC forecast bulletin PDFs (5-day, 10-day) and matches rows to stations.
- `db.py`: all SQL. `engine.py`: orchestration and CLI. `forecaster.py`: runs the model checkpoint.
- `check_model_parity.py`: proves `forecaster.py` matches the training pipeline (run after any change to it).
- `run_nightly.cmd`: Task Scheduler entry point (`engine.py all` at 23:50, log in `logs/`).
- `wl_stations.json` / `rainfall_stations.json`: station lists for `sync-stations`; the rainfall
  file's `mapped_wl_station_id` follows the training mapping (see Model).
- `admin.py` + `admin.html`: admin API and single-page UI (vanilla JS).
- `config.py`: everything configurable, read from `.env`.

## Data model
- `stations.id` = FFWC WL code (`SW7.5`); `stations.web_id` = numeric id used in API URLs.
- `rf_stations.id` = rainfall code (`CL6`); `rf_stations.api_id` = numeric id in API URLs.
- `stations.rf_station_id` links a WL station to its model's rainfall input.
- Flags: `scrape_enabled` (tracked, shown on site), `has_model` (forecast it).
  User-owned columns must never be overwritten by station sync.
- `forecasts` keeps every issued forecast (`source` = 'ours' | 'ffwc', `issued_on`, `target_time`;
  `target_date` and `lead_days` are generated), which powers the comparison and past-performance graphs.
  Ours is 3-hourly: 75 rows per station per night. FFWC rows carry the product in `model_version`
  (`ffwc-5day`, `ffwc-10day-mean`) and are kept for `has_model` stations only.
- `forecast_comparison` view pairs ours and FFWC's at the same station/issue day/target time;
  `comparison_skill()` and `comparison_issues()` aggregate it for the site (avoids the 1000-row API cap).
- `forecast_steps` view numbers our forecast rows by step (1 = next day 06:00 ... 75); `horizon_performance()`
  and `horizon_skill()` feed the site's performance page. Website horizons map to steps 1/2/4/5/15/35/75.
- `daily_wl` view = mean of the day's readings. It must match how the model's training data defined daily WL.

## FFWC API (endpoints the ffwc.gov.bd site itself calls)
- WL: `https://api.ffwc.gov.bd/data_load/observed-waterlevel-by-station-and-date/{web_id}/{YYYY-MM-DD}/`
  returns 40 days ending at the date, readings at 06/09/12/15/18 h.
- Rainfall: `https://api.ffwc.gov.bd/data_load/rainfall-by-station/{api_id}/` returns ~40 days, daily, no date param.
- Requires header `x-ffwc-internal-key` (in `.env` as `FFWC_API_KEY`); 401/403 means it rotated.
- FFWC forecast: `/data_load/five-days-forecast-waterlevel/` = all stations' current 5-day forecast,
  `{web_id: [{"YYYY-MM-DD HH:MM:SSZ": wl}]}` at 06-18h (local despite `Z`), first step = 06:00 the day
  after issue. Fetched by every full `scrape`. Covers 21 of our 25 stations.
- FFWC's 10-day (medium range) forecast is monsoon-only (JSON stops at early Oct); past bulletins
  come from PDFs (`import-ffwc-pdf`). No PDF download URL is known; PDFs are added by hand.
- Keep `REQUEST_DELAY_S` at about 3 s; requests are sequential.

## Gotchas (verified against real responses)
- `wl_date` like `2026-10-11T15:00:00Z` is **Bangladesh local time** despite the `Z`. Store as Asia/Dhaka.
- `rf_date` is **MM-DD-YYYY** (`09-30-2026`).
- Numbers often arrive as strings; missing values are `"-"` or `"#N/A"`.
- In responses, `st_id` is the numeric id (web_id / api_id), not the SW/CL code.
- 14 WL stations have no `dl` in the station list; the WL response's `dangerlevel` fills it in.
- `SW179` (Demra, Lakhya) is not in FFWC's list; `SW177` (Lakhpur) is the replacement.
- Rainfall older than ~40 days can't be re-fetched; history comes from training data CSVs
  or saved responses (`engine.py import-rf-json rf_*.json`, e.g. the thesis `scraping/scraped_data/rainfall/`).
- PDF bulletins spell some stations differently (`NAME_ALIASES` in `ffwc_pdf.py`, e.g. Elasin = Elasinghat)
  and wrap long river names onto the next line; 10-day rows that don't have exactly 10 values are skipped.

## Model (clustered federated vanilla LSTM)
- Checkpoint `models/clustered_fl_vanilla_lstm-local-weighted-0.5.pt` (`MODEL_PATH`); training and
  test scripts in `models/train_test_scripts/`. Training data lives outside this repo, in the thesis
  folder's `Datasets/Processed/Station_wise_combined/`.
- Input: 150 steps = 30 days x 5 readings (06/09/12/15/18 local), features WL, log1p rainfall
  (the day's value copied to its 5 readings), hour/day-of-year sin/cos, delta_hours/12.
  Output: 75 steps = next day 06:00 .. day 15 18:00.
- 4 cluster models (c1..c4) and a per-station scaler, all in the checkpoint. Only its 25 stations
  can be forecast. SW179 is gone from FFWC, so SW177 (same river, similar levels) runs through
  SW179's model and scaler (`STATION_ALIASES` in `forecaster.py`); its forecasts are stored as SW177.
- `models/prediction_scripts/` is the earlier offline pipeline (JSON -> CSV -> older checkpoint with
  global scalers). Reference only; it can't load the current checkpoint.
- Preprocessing must match `fl_lstm_claude.py` exactly: WL gaps up to 10 readings interpolated,
  longer gaps skip the station; missing rain = 0. Verify with `check_model_parity.py --data <dir>`.
- Rainfall station per WL station = training `STATION_MAPPING` (thesis `Preprocessing/map_wl_with_all.py`).
  CL42 (SW137A, SW68.5, SW302) has no known FFWC api id, so CL9 stands in.
- At 23:50 FFWC hasn't posted the day's 18:00 reading (it appears after 02:00), so it is filled
  with the 15:00 value; run stats record fills per station (`filled`).

## Comparison data (as of 2026-10-31)
- FFWC PDFs imported: 5-day 2026-10-03 and 10-17, 10-day 2026-09-25 and 10-09 (in `models/ffwc_predictions/`,
  gitignored). Our forecasts for issue days 2026-09-25 and 2026-10-01..10-29 are hindcasts
  (`engine.py predict --from ... --to ...`, or the admin panel's "Run model for forecasts starting on").
- WL history backfilled from 2026-08-01; rainfall from 2026-08-31 (saved JSON) + API.

## Open tasks
1. Find the FFWC rainfall station list endpoint so all rainfall stations can be imported.
2. Register `run_nightly.cmd` in Task Scheduler (daily 23:50); runs are manual from the admin panel for now.
3. Find CL42's FFWC rainfall api id and switch SW137A, SW68.5, SW302 back to it.
4. Find where FFWC publishes the bulletin PDFs, to automate `import-ffwc-pdf`.

## Conventions
- Python 3.11+, psycopg 3, no ORM. All SQL lives in `db.py`.
- Each scrape step and each station commits separately so one failure doesn't lose the rest.
- Test parsers against saved sample responses before calling the live API.
