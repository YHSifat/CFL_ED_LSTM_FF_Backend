# Bonna flood engine (backend)

Backend of **Bonna**, the water-level forecasting system built for the B.Sc. thesis
*Clustered Federated Learning with Encoder-Decoder LSTM for Multi-Horizon Water-Level
Forecasting in Bangladesh* (Reshma Kakoli and Md Yusuf Hasan, supervised by Dr Sajeeb Saha,
Department of CSE, Jagannath University).

The engine runs on a local machine. It scrapes observed water levels and rainfall from the
Flood Forecasting and Warning Centre (FFWC, [ffwc.gov.bd](https://www.ffwc.gov.bd)), runs the
trained clustered federated model to forecast the next 15 days, and writes observations and
forecasts to a Supabase Postgres database. The [frontend](../CFL_ED_LSTM_FF_Frontend) reads
that database and shows the results.

**Live site: <https://bonna-bd.vercel.app/>**

## Overview

Bangladesh sits at the bottom of the Ganges-Brahmaputra-Meghna basin, and much of its river
flow comes from upstream catchments in India, Nepal, Bhutan and China. FFWC forecasts floods
with MIKE 11, a physics-based river model. The thesis tests whether a deep-learning model can
give useful forecasts at lead times of up to 15 days, while handling the fact that stations
behave very differently from one another.

The approach, **CFL-EDLSTM**:

- **25 BWDB gauge stations on 13 rivers** (20 non-tidal, 5 tidal), grouped into **4 hydrological
  clusters**: Upstream (6 stations), Mid-river (8), Downstream (5) and Urban (6).
- **Clustered federated learning.** Each cluster is one federated client. In every round the
  server sends the global weights to the four clients, each trains on its own stations, and the
  server combines the updates with FedAvg weighted by station count (6/25, 8/25, 5/25, 6/25).
  Unlike plain FL, each cluster keeps its own locally adapted model. Training runs 30 rounds.
- **Multi-horizon output.** The model reads a 30-day window (150 readings, 5 a day at 06, 09,
  12, 15 and 18 h) and predicts 15 days ahead (75 readings).
- **Inputs.** Water level, rainfall, temperature, evaporation and an MDD variable, expanded to
  42 features in the thesis experiments (rainfall totals, lags, rolling means and time encodings).

### Thesis results (held-out test windows)

| Model | MAE (m) | RMSE (m) | NSE |
|---|---|---|---|
| XGBoost | 0.540 | 1.025 | 0.9919 |
| Vanilla LSTM | 0.496 | 0.722 | 0.9960 |
| GRU | 0.502 | 0.743 | 0.9957 |
| TCN | 0.574 | 0.853 | 0.9944 |
| **CFL-EDLSTM** | **0.321** | **0.521** | **0.9979** |

CFL-EDLSTM is the most accurate model from the 3-day horizon onward. At 15 days its MAE is
0.436 m, against 1.149 m for XGBoost. In a live check at station SW15J (forecast issued
3 October 2026), its mean absolute error over 5 lead days was 0.054 m, against 0.112 m for
FFWC's own forecast.

### The model this engine runs

The engine loads `models/clustered_fl_vanilla_lstm-local-weighted-0.5.pt`: the clustered
federated set-up above (4 cluster models plus a scaler per station) with an LSTM and a dense
forecast head that outputs all 75 steps at once. Its inputs are the ones FFWC publishes live:
water level, log rainfall and time features (hour and day-of-year sin/cos, gap since the last
reading). The training script is `models/train_test_scripts/fl_lstm_claude.py`, and
`check_model_parity.py` checks that `forecaster.py` prepares data exactly as training did.

## Features

- **FFWC scraper**: observed water level (3-hourly, 06:00–18:00) and daily rainfall from the
  same API the FFWC website calls, plus FFWC's current 5-day forecast on every full scrape.
- **Forecasting**: a 15-day, 3-hourly forecast (75 values) for each station with *Model* ticked,
  using that station's cluster model and scaler.
- **Hindcasts**: run the model for any past issue day to build forecast-performance history.
- **FFWC bulletin import**: parse FFWC's 5-day and 10-day forecast PDFs so both forecasts can
  be compared on the same issue days.
- **Backfill and import**: walk water-level history back in 50-day steps; import rainfall
  history from CSV or saved FFWC responses.
- **Admin panel** (localhost only): choose which stations to track and forecast, set each
  station's rainfall input, and run scrapes, backfills and forecasts with a button.
- **Database views and functions** for the website: station status against danger level,
  ours vs FFWC on matched times, and error per forecast horizon.
- **Nightly job**: `run_nightly.cmd` for Windows Task Scheduler (scrape, then forecast).

## Architecture

```
 ffwc.gov.bd API ──► [ local engine: scraper → forecaster → db ] ──► Supabase Postgres ◄── frontend
                                       ▲                              tables, views, RLS     (anon key,
                             admin panel 127.0.0.1:8000               (schema.sql)            read-only)
```

The website needs no backend server: Supabase generates a REST API over the tables and views,
and row-level security makes them read-only for the public anon key. The engine connects as
`postgres` through the session pooler URL, so it can write.

All code is in `flood-engine/`:

| File | Role |
|---|---|
| `engine.py` | Orchestration and CLI (sync stations, scrape, backfill, predict, import). |
| `scraper.py` | FFWC API client and parsers (no database code). |
| `forecaster.py` | Loads the checkpoint, builds the 150-step input, returns 75 forecast steps. |
| `ffwc_pdf.py` | Parses FFWC 5-day and 10-day bulletin PDFs and matches rows to stations. |
| `db.py` | All SQL (psycopg 3, no ORM). |
| `schema.sql` | Tables, views, functions and RLS. This is the contract with the frontend. Safe to re-run. |
| `admin.py`, `admin.html` | Admin API (FastAPI) and single-page UI. |
| `config.py` | Settings read from `.env`. |
| `wl_stations.json`, `rainfall_stations.json` | Station lists for `sync-stations`. |
| `models/` | Model checkpoint, training/test scripts, FFWC bulletin PDFs (gitignored). |

Main tables: `stations`, `rf_stations`, `observed_wl`, `observed_rf`, `forecasts` (every issued
forecast, `source` = `ours` or `ffwc`) and `engine_runs` (admin only). Station ids are FFWC
water-level codes such as `SW7.5`, the same as in the frontend.

## How to run

Requirements: Python 3.11+, a free Supabase project, and an FFWC API key.

### 1. Database (Supabase free tier)
1. Create a project at supabase.com (Singapore or Mumbai region, the closest to Bangladesh).
2. SQL Editor → paste `schema.sql` → Run.
3. Project Settings → Database → copy the **Session pooler** connection string into `.env` as `DATABASE_URL`.
4. Project Settings → API → the **anon** key goes in the frontend. The service_role key and
   database password never leave your machine.

### 2. Install and configure
From the repo root (`run_nightly.cmd` expects `.venv` there):
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
cd flood-engine                        # all following commands run from here
pip install -r requirements.txt        # torch CPU: add --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env                   # fill in DATABASE_URL and FFWC_API_KEY
```

`FFWC_API_KEY` is the value of the `x-ffwc-internal-key` header the FFWC website sends. It
changes from time to time: when runs fail with HTTP 401 or 403, copy the new value from the
browser's Network tab.

### 3. Load stations and start the admin panel
```bash
# load station lists; tracks the rainfall stations and the WL stations they map to
python engine.py sync-stations --wl wl_stations.json --rf rainfall_stations.json --apply-targets

uvicorn admin:app --host 127.0.0.1 --port 8000     # admin panel, no login: keep it on localhost
```

In the admin panel:
- **Water level stations tab**: tick *Track* for stations to scrape and show on the site,
  *Model* for stations to forecast, and pick each one's rainfall input. Each row has a one-off
  *Scrape* button.
- **Rainfall stations tab**: track or add stations. The API id is the number in the FFWC site's
  `rainfall-by-station/{id}/` request.
- **Backfill** walks the water-level endpoint back in 50-day steps to fill history. Rainfall has
  no date parameter, so rainfall older than about 40 days must come from training data:
  `python engine.py import-rf rf.csv`.

### 4. Command line
```bash
python engine.py scrape                          # all tracked stations
python engine.py scrape --wl SW7.5 --rf CL9      # selected stations
python engine.py backfill-wl --from 2026-05-01   # water-level history
python engine.py predict                         # forecast from today's data
python engine.py predict --from 2026-10-15 --to 2026-10-29   # hindcasts, one per issue day
python engine.py all                             # scrape, then predict
python engine.py import-ffwc-pdf models/ffwc_predictions/*.pdf   # FFWC bulletins
```
The docstring in `engine.py` lists every command.

### 5. Run it every night
`run_nightly.cmd` runs `engine.py all` and logs to `logs/nightly.log`. Register it once:
```
schtasks /create /tn "FloodEngineNightly" /sc daily /st 23:50 /tr "\"%CD%\run_nightly.cmd\""
```
On Linux or macOS, use cron. A lock in `engine.py` stops runs from overlapping.

## FFWC API notes

| Data | URL (under `https://api.ffwc.gov.bd/data_load`) | Returns |
|---|---|---|
| Water level | `/observed-waterlevel-by-station-and-date/{web_id}/{YYYY-MM-DD}/` | 40 days ending at the date, 5 readings a day |
| Rainfall | `/rainfall-by-station/{api_id}/` | about 40 days, daily |
| FFWC forecast | `/five-days-forecast-waterlevel/` | all stations' current 5-day forecast |

- `wl_date` such as `2026-10-11T15:00:00Z` is **Bangladesh time** despite the `Z`. It is stored
  as Asia/Dhaka time; set `FFWC_TIME_IS_LOCAL=false` if that turns out to be wrong.
- `rf_date` is month-first (`09-30-2026`).
- Keep `REQUEST_DELAY_S` at about 3 seconds; requests are sent one at a time.

## Queries the frontend uses (supabase-js)
```js
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// cache check: changes when there is a new reading, forecast or station edit
await sb.rpc('data_stamp');

// map markers + warning status
await sb.from('station_status').select('*');

// history graph
await sb.from('daily_wl').select('day,wl').eq('station_id', id).gte('day', from).order('day');

// latest 15-day forecast: ours and FFWC
await sb.from('latest_forecasts').select('source,target_time,wl,issued_on').eq('station_id', id).order('target_time');

// past performance at a fixed lead time
await sb.from('forecast_vs_observed').select('*').eq('station_id', id).eq('lead_days', 3).order('target_time');

// ours vs FFWC on matched times: issue days for a station, then error by lead day
await sb.rpc('comparison_issues', { p_station: id });
await sb.rpc('comparison_skill', { p_station: id });   // null = all stations

// performance per horizon (step 1 = next day 06:00, 5 = 1 day, 75 = 15 days)
await sb.rpc('horizon_performance', { p_station: id, p_step: 5, p_from: '2026-10-16', p_to: '2026-10-30' });
await sb.rpc('horizon_skill', { p_station: id, p_from: '2026-10-16', p_to: '2026-10-30' });   // null = all
```

## Free-tier notes
- 500 MB database: about 100 stations × 15-day forecasts × 2 sources ≈ 1.1M forecast rows a year
  (100–150 MB with indexes), enough for a couple of years.
- Free projects pause after a week without activity. Nightly runs keep it awake, but check the
  dashboard if you stop running the engine.
- No automatic backups on the free plan: run `pg_dump "$DATABASE_URL" > backup.sql` weekly.

## Limitations
- Data gaps: some stations report mainly during the monsoon, and upstream data from India is
  not available.
- Rainfall older than about 40 days cannot be re-fetched from FFWC.
- Federation is simulated on one machine; training used single runs without a hyperparameter search.
- Only the 25 stations in the checkpoint can be forecast. SW179 (Demra) is no longer published
  by FFWC, so SW177 (Lakhpur, same river) is forecast through SW179's model.
