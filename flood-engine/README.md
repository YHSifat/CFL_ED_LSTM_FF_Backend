# Flood engine (local) + Supabase (hosted DB)

```
 ffwc API ──► [ local engine: scraper → model → db ] ──► Supabase Postgres ◄── frontend (anon key, read-only)
                         ▲
               local admin panel (127.0.0.1)
```

The frontend needs no backend server: Supabase auto-generates a REST API over
the tables/views, and Row Level Security makes it read-only for the public key.

## 1. Database (Supabase free tier)
1. Create a project at supabase.com (pick the Singapore or Mumbai region — closest to Bangladesh).
2. SQL Editor → paste `schema.sql` → Run.
3. Project Settings → Database → copy the **Session pooler** connection string into `.env` as `DATABASE_URL`.
4. Project Settings → API → the **anon** key goes in the frontend. The service_role key and DB password never leave your machine.

## 2. FFWC API
The engine calls the same endpoints the ffwc.gov.bd site uses:

| Data | URL | Returns |
|---|---|---|
| Water level | `/data_load/observed-waterlevel-by-station-and-date/{web_id}/{YYYY-MM-DD}/` | ~60 days ending at the date, 5 readings a day |
| Rainfall | `/data_load/rainfall-by-station/{api_id}/` | ~40 days, daily |

Put the `x-ffwc-internal-key` value in `.env` as `FFWC_API_KEY`. It will probably change;
when runs fail with HTTP 401/403, copy the new value from the browser's Network tab.

Two format details the scraper handles:
- `wl_date` like `2026-10-11T15:00:00Z` is **Bangladesh time**, despite the `Z`
  (readings are at 06:00–18:00). It is stored as Asia/Dhaka time; set
  `FFWC_TIME_IS_LOCAL=false` if you find otherwise.
- `rf_date` is month-first (`09-30-2026`).

## 3. Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # + tensorflow / torch / scikit-learn
cp .env.example .env                   # fill it in

# load station lists; tracks your rainfall stations and the WL stations they map to
python engine.py sync-stations --wl wl_stations.json --rf rainfall_stations.json --apply-targets

uvicorn admin:app --host 127.0.0.1 --port 8000     # admin panel
```
In the admin panel:
- **Water level stations tab**: tick *Track* for stations to scrape and show on the
  site, *Model* for stations to forecast, and pick each one's rainfall input.
  Each row also has a one-off *Scrape* button.
- **Rainfall stations tab**: track or add stations. The API id is the number in
  the site's `rainfall-by-station/{id}/` request.
- **Backfill** walks the water level endpoint back in 50-day steps to fill history.
  Rainfall has no date parameter, so rainfall history beyond ~40 days has to come
  from your training data: `python engine.py import-rf rf.csv`.

Command line equivalents: `python engine.py scrape`, `scrape --wl SW7.5 --rf CL9`,
`backfill-wl --from 2026-05-01`, `predict`, `all`. The docstring in `engine.py` lists them.

Backfill past forecasts for the performance graph (hindcast):
`python engine.py predict --date 2026-09-01` (loop over dates).

## 4. Frontend queries (supabase-js)
```js
const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// map markers + warning sign
await sb.from('station_status').select('*');

// history graph
await sb.from('daily_wl').select('day,wl').eq('station_id', id).gte('day', from).order('day');

// 15-day forecast: ours vs FFWC
await sb.from('latest_forecasts').select('source,target_date,wl,issued_on').eq('station_id', id).order('target_date');

// past performance: forecast vs observed at a fixed lead time
await sb.from('forecast_vs_observed').select('*').eq('station_id', id).eq('lead_days', 3).order('target_date');

// summary skill per lead day
await sb.rpc('forecast_skill', { p_station: id, p_from: '2026-06-01', p_to: '2026-10-30' });
```

## 5. Automating later
Windows Task Scheduler or cron: `python engine.py all` a few times a day.
It scrapes the tracked stations, then forecasts the stations with *Model* ticked. The lock in `engine.py` stops overlapping runs.

## Free-tier notes
- 500 MB DB: ~100 stations × 15-day forecasts × 2 sources ≈ 1.1M forecast rows/year
  (~100–150 MB with indexes) — fine for a couple of years.
- Free projects pause after 1 week of inactivity; regular engine runs keep it busy,
  but check the dashboard if you stop running it.
- No automatic backups on free: run `pg_dump "$DATABASE_URL" > backup.sql` weekly.
