"""All database reads/writes for the engine and admin panel."""
from contextlib import contextmanager
from datetime import date

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

import config


@contextmanager
def connect():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    with psycopg.connect(config.DATABASE_URL, prepare_threshold=None) as conn:
        yield conn


def _dicts(cur) -> list[dict]:
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---- stations: sync from FFWC lists -----------------------------------------
def upsert_stations(conn, rows: list[dict]) -> int:
    """FFWC-owned fields only; scrape_enabled / has_model / rf_station_id /
    warning_level are yours and never overwritten."""
    with conn.cursor() as cur:
        cur.executemany("""
            insert into stations (id, web_id, name, river, basin, district, division,
                                  lat, lon, danger_level, rhwl, ffwc_active)
            values (%(id)s, %(web_id)s, %(name)s, %(river)s, %(basin)s, %(district)s, %(division)s,
                    %(lat)s, %(lon)s, %(danger_level)s, %(rhwl)s, %(ffwc_active)s)
            on conflict (id) do update set
              web_id       = excluded.web_id,
              name         = excluded.name,
              river        = coalesce(excluded.river, stations.river),
              basin        = coalesce(excluded.basin, stations.basin),
              district     = coalesce(excluded.district, stations.district),
              division     = coalesce(excluded.division, stations.division),
              lat          = coalesce(excluded.lat, stations.lat),
              lon          = coalesce(excluded.lon, stations.lon),
              danger_level = coalesce(excluded.danger_level, stations.danger_level),
              rhwl         = coalesce(excluded.rhwl, stations.rhwl),
              ffwc_active  = coalesce(excluded.ffwc_active, stations.ffwc_active),
              updated_at   = now()
        """, rows)
    return len(rows)


def upsert_rf_stations(conn, rows: list[dict]) -> int:
    with conn.cursor() as cur:
        cur.executemany("""
            insert into rf_stations (id, api_id, name, basin, lat, lon)
            values (%(id)s, %(api_id)s, %(name)s, %(basin)s, %(lat)s, %(lon)s)
            on conflict (id) do update set
              api_id = coalesce(excluded.api_id, rf_stations.api_id),
              name   = coalesce(excluded.name, rf_stations.name),
              basin  = coalesce(excluded.basin, rf_stations.basin),
              lat    = coalesce(excluded.lat, rf_stations.lat),
              lon    = coalesce(excluded.lon, rf_stations.lon)
        """, [{"basin": None, "lat": None, "lon": None, **r} for r in rows])
    return len(rows)


def update_station_levels(conn, code: str, danger_level, rhwl):
    """Keep DL / RHWL current from the values the WL endpoint returns."""
    conn.execute("""
        update stations set danger_level = coalesce(%s, danger_level),
                            rhwl = coalesce(%s, rhwl), updated_at = now()
        where id = %s
    """, (danger_level, rhwl, code))


# ---- stations: admin settings ---------------------------------------------
def list_wl_stations(conn) -> list[dict]:
    return _dicts(conn.execute("""
        select s.id, s.web_id, s.name, s.river, s.basin, s.district, s.danger_level,
               s.ffwc_active, s.scrape_enabled, s.has_model, s.rf_station_id,
               o.last_obs
        from stations s
        left join (select station_id, max(obs_time) as last_obs
                   from observed_wl group by 1) o on o.station_id = s.id
        order by s.scrape_enabled desc, s.basin nulls last, s.name
    """))


def list_rf_stations(conn) -> list[dict]:
    return _dicts(conn.execute("""
        select r.id, r.api_id, r.name, r.scrape_enabled,
               coalesce(m.mapped_wl, '{}') as mapped_wl, o.last_obs
        from rf_stations r
        left join (select rf_station_id, array_agg(id order by id) as mapped_wl
                   from stations where rf_station_id is not null group by 1) m
               on m.rf_station_id = r.id
        left join (select rf_station_id, max(obs_date) as last_obs
                   from observed_rf group by 1) o on o.rf_station_id = r.id
        order by r.scrape_enabled desc, r.name
    """))


def update_wl_settings(conn, rows: list[dict]) -> int:
    """rows: [{id, scrape_enabled?, has_model?, rf_station_id?}] (missing keys unchanged)"""
    n = 0
    for r in rows:
        sets, vals = [], []
        for col in ("scrape_enabled", "has_model", "rf_station_id"):
            if col in r:
                sets.append(f"{col} = %s")
                vals.append(r[col] or None if col == "rf_station_id" else bool(r[col]))
        if sets:
            n += conn.execute(f"update stations set {', '.join(sets)} where id = %s",
                              (*vals, r["id"])).rowcount
    return n


def update_rf_settings(conn, rows: list[dict]) -> int:
    n = 0
    for r in rows:
        if "scrape_enabled" in r:
            n += conn.execute("update rf_stations set scrape_enabled = %s where id = %s",
                              (bool(r["scrape_enabled"]), r["id"])).rowcount
    return n


def map_rf_to_wl(conn, rf_id: str, wl_codes: list[str], enable: bool = False) -> list[str]:
    """Point the given WL stations at this rainfall station; returns unknown codes."""
    known = station_ids(conn)
    missing = [c for c in wl_codes if c not in known]
    for c in wl_codes:
        if c in known:
            conn.execute(
                "update stations set rf_station_id = %s, scrape_enabled = scrape_enabled or %s where id = %s",
                (rf_id, enable, c))
    if enable:
        conn.execute("update rf_stations set scrape_enabled = true where id = %s", (rf_id,))
    return missing


# ---- scrape targets --------------------------------------------------------
def wl_targets(conn, codes: list[str] | None = None) -> list[tuple[str, int]]:
    """(code, web_id) for the given codes, or for all enabled stations."""
    if codes:
        q = "select id, web_id from stations where id = any(%s) and web_id is not null order by id"
        return conn.execute(q, (codes,)).fetchall()
    return conn.execute("select id, web_id from stations where scrape_enabled "
                        "and web_id is not null order by id").fetchall()


def rf_targets(conn, codes: list[str] | None = None) -> list[tuple[str, str]]:
    if codes:
        q = "select id, api_id from rf_stations where id = any(%s) and api_id is not null order by id"
        return conn.execute(q, (codes,)).fetchall()
    return conn.execute("select id, api_id from rf_stations where scrape_enabled "
                        "and api_id is not null order by id").fetchall()


def station_ids(conn) -> set[str]:
    return {r[0] for r in conn.execute("select id from stations")}


# ---- observations / forecasts -------------------------------------------------
def upsert_observed_wl(conn, rows: list[tuple]) -> int:
    """rows: [(station_code, obs_time, wl)]"""
    with conn.cursor() as cur:
        cur.executemany("""
            insert into observed_wl (station_id, obs_time, wl) values (%s, %s, %s)
            on conflict (station_id, obs_time) do update set wl = excluded.wl
        """, rows)
    return len(rows)


def upsert_observed_rf(conn, rows: list[tuple]) -> int:
    """rows: [(rf_station_code, obs_date, rainfall_mm)]"""
    with conn.cursor() as cur:
        cur.executemany("""
            insert into observed_rf (rf_station_id, obs_date, rainfall_mm) values (%s, %s, %s)
            on conflict (rf_station_id, obs_date) do update set rainfall_mm = excluded.rainfall_mm
        """, rows)
    return len(rows)


def upsert_forecasts(conn, source: str, rows: list[tuple], model_version: str | None) -> int:
    """rows: [(station_code, issued_on, target_time, wl)], target_time tz-aware"""
    with conn.cursor() as cur:
        cur.executemany("""
            insert into forecasts (source, station_id, issued_on, target_time, wl, model_version)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (source, station_id, issued_on, target_time)
            do update set wl = excluded.wl, model_version = excluded.model_version, created_at = now()
        """, [(source, *r, model_version) for r in rows])
    return len(rows)


def model_station_info(conn) -> list[dict]:
    """[{id, web_id, name, river}] for has_model stations (FFWC forecasts are kept for these only)."""
    return _dicts(conn.execute(
        "select id, web_id, name, river from stations where has_model order by id"))


def model_stations(conn, codes: list[str] | None = None) -> list[tuple[str, str | None]]:
    if codes:
        return conn.execute("select id, rf_station_id from stations where id = any(%s) order by id",
                            (codes,)).fetchall()
    return conn.execute("select id, rf_station_id from stations where has_model order by id").fetchall()


def last_wl_day(conn, station_id: str) -> date | None:
    return conn.execute("select max(day) from daily_wl where station_id = %s",
                        (station_id,)).fetchone()[0]


def get_wl_readings(conn, station_id: str, start: date, end: date) -> pd.Series:
    """WL readings from start 00:00 to end 23:59 local, indexed by tz-aware obs_time."""
    rows = conn.execute("""
        select obs_time, wl from observed_wl
        where station_id = %(sid)s
          and (obs_time at time zone 'Asia/Dhaka')::date between %(start)s and %(end)s
        order by obs_time
    """, {"sid": station_id, "start": start, "end": end}).fetchall()
    return pd.Series([r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows]),
                     dtype=float)


def get_rainfall(conn, rf_station_id: str, start: date, end: date) -> pd.Series:
    """Daily rainfall (mm) indexed by date; days FFWC didn't report are absent."""
    rows = conn.execute("""
        select obs_date, rainfall_mm from observed_rf
        where rf_station_id = %s and obs_date between %s and %s order by obs_date
    """, (rf_station_id, start, end)).fetchall()
    return pd.Series([r[1] for r in rows], index=[r[0] for r in rows], dtype=float)


# ---- run log -------------------------------------------------------------
def start_run(conn, task: str, params: dict | None = None) -> int:
    run_id = conn.execute("insert into engine_runs (task, stats) values (%s, %s) returning id",
                          (task, Jsonb({"params": params or {}}))).fetchone()[0]
    conn.commit()
    return run_id


def set_progress(conn, run_id: int, text: str):
    conn.execute("update engine_runs set progress = %s where id = %s", (text, run_id))
    conn.commit()


def finish_run(conn, run_id: int, status: str, message: str = "", stats: dict | None = None):
    conn.execute("""
        update engine_runs set finished_at = now(), status = %s, message = %s,
               stats = coalesce(stats, '{}'::jsonb) || %s, progress = null
        where id = %s
    """, (status, message[:8000], Jsonb(stats or {}), run_id))
    conn.commit()


def recent_runs(conn, limit: int = 20) -> list[dict]:
    return _dicts(conn.execute("""
        select id, task, started_at, finished_at, status, progress, message, stats
        from engine_runs order by id desc limit %s
    """, (limit,)))
