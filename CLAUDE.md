# Flood engine

Backend for a Bangladesh flood forecasting website (modelled on ffwc.gov.bd).
The frontend is already built and hosted separately; it reads from Supabase.

See @README.md for setup and the frontend queries.

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
- `db.py`: all SQL. `engine.py`: orchestration and CLI. `predictor.py`: model wrapper.
- `admin.py` + `admin.html`: admin API and single-page UI (vanilla JS).
- `config.py`: everything configurable, read from `.env`.

## Data model
- `stations.id` = FFWC WL code (`SW7.5`); `stations.web_id` = numeric id used in API URLs.
- `rf_stations.id` = rainfall code (`CL6`); `rf_stations.api_id` = numeric id in API URLs.
- `stations.rf_station_id` links a WL station to its model's rainfall input.
- Flags: `scrape_enabled` (tracked, shown on site), `has_model` (forecast it).
  User-owned columns must never be overwritten by station sync.
- `forecasts` keeps every issued forecast (`source` = 'ours' | 'ffwc', `issued_on`, `target_date`),
  which powers the comparison and past-performance graphs.
- `daily_wl` view = mean of the day's readings. It must match how the model's training data defined daily WL.

## FFWC API (endpoints the ffwc.gov.bd site itself calls)
- WL: `https://api.ffwc.gov.bd/data_load/observed-waterlevel-by-station-and-date/{web_id}/{YYYY-MM-DD}/`
  returns ~60 days ending at the date, readings at 06/09/12/15/18 h.
- Rainfall: `https://api.ffwc.gov.bd/data_load/rainfall-by-station/{api_id}/` returns ~40 days, daily, no date param.
- Requires header `x-ffwc-internal-key` (in `.env` as `FFWC_API_KEY`); 401/403 means it rotated.
- Keep `REQUEST_DELAY_S` at about 3 s; requests are sequential.

## Gotchas (verified against real responses)
- `wl_date` like `2026-10-11T15:00:00Z` is **Bangladesh local time** despite the `Z`. Store as Asia/Dhaka.
- `rf_date` is **MM-DD-YYYY** (`09-30-2026`).
- Numbers often arrive as strings; missing values are `"-"` or `"#N/A"`.
- In responses, `st_id` is the numeric id (web_id / api_id), not the SW/CL code.
- 14 WL stations have no `dl` in the station list; the WL response's `dangerlevel` fills it in.
- `SW179` (Demra, Lakhya) is not in FFWC's list; `SW177` (Lakhpur) is the replacement.
- Rainfall older than ~40 days can't be re-fetched; history comes from training data CSVs
  or the user's old `scraping/scraped_data/rainfall/rf_*.json` files.

## Model
- Input: last 30 days daily [wl, rainfall_mm]; output: next 15 days WL.
- Files in `models/`: `<station_code>.<ext>` or `global.<ext>` (.keras/.h5/.pt TorchScript/.joblib),
  with optional `<name>_scaler_x.joblib` / `_scaler_y.joblib`.
- Gaps of up to 3 WL days are interpolated; missing rainfall is filled with 0.

## Open tasks
1. Importer for the old scraper's JSON files (`scraping/scraped_data/wl/*.json`, `.../rainfall/rf_*.json`).
2. Scrape FFWC's own forecasts for the comparison graph (`scraper.fetch_ffwc_forecasts` is a stub;
   needs a sample response from the forecast endpoint).
3. Find the FFWC rainfall station list endpoint so all rainfall stations can be imported.
4. Schedule `python engine.py all` (Task Scheduler / cron) once manual runs work.

## Conventions
- Python 3.11+, psycopg 3, no ORM. All SQL lives in `db.py`.
- Each scrape step and each station commits separately so one failure doesn't lose the rest.
- Test parsers against saved sample responses before calling the live API.
