-- =====================================================================
-- Flood forecasting DB schema (Supabase / Postgres 15+)
-- Safe to re-run: Supabase Dashboard -> SQL Editor -> paste -> Run
-- (re-running also upgrades a database created with the first version)
-- =====================================================================

-- ---------- Water-level stations (id = FFWC code, e.g. 'SW7.5') ----------
create table if not exists stations (
  id             text primary key,
  name           text not null,
  river          text,
  basin          text,
  lat            double precision,
  lon            double precision,
  danger_level   real,
  warning_level  real,                       -- your threshold; null => DL - 0.5 m
  rhwl           real,
  rf_station_id  text,                       -- rainfall station used as model input
  has_model      boolean not null default false,
  updated_at     timestamptz not null default now()
);
alter table stations add column if not exists web_id         integer;   -- numeric id used in FFWC API URLs
alter table stations add column if not exists district       text;
alter table stations add column if not exists division       text;
alter table stations add column if not exists ffwc_active    boolean;
alter table stations add column if not exists scrape_enabled boolean not null default false;
create unique index if not exists stations_web_id_key on stations (web_id);

-- ---------- Rainfall stations (id = FFWC code, e.g. 'CL6') ----------
create table if not exists rf_stations (
  id     text primary key,
  name   text,
  basin  text,
  lat    double precision,
  lon    double precision
);
alter table rf_stations add column if not exists api_id         text;   -- numeric id used in FFWC API URLs
alter table rf_stations add column if not exists scrape_enabled boolean not null default false;
create unique index if not exists rf_stations_api_id_key on rf_stations (api_id);

-- ---------- Observations ----------
create table if not exists observed_wl (
  station_id  text not null references stations(id) on delete cascade,
  obs_time    timestamptz not null,
  wl          real not null,
  primary key (station_id, obs_time)
);

create table if not exists observed_rf (
  rf_station_id  text not null references rf_stations(id) on delete cascade,
  obs_date       date not null,
  rainfall_mm    real not null,
  primary key (rf_station_id, obs_date)
);

-- ---------- Forecasts (ours + FFWC) ----------
-- 3-hourly: our model forecasts each FFWC reading time (06,09,12,15,18 local) for
-- 15 days. The first version of this table was daily (target_date as a plain
-- column); it never held data, so it is dropped and recreated once.
do $$
begin
  if exists (select 1 from information_schema.columns
             where table_schema = 'public' and table_name = 'forecasts'
               and column_name = 'target_date' and is_generated = 'NEVER') then
    drop table forecasts cascade;
  end if;
end $$;

create table if not exists forecasts (
  source         text not null check (source in ('ours', 'ffwc')),
  station_id     text not null references stations(id) on delete cascade,
  issued_on      date not null,                -- local day the input data ends
  target_time    timestamptz not null,
  wl             real not null,
  target_date    date generated always as ((target_time at time zone 'Asia/Dhaka')::date) stored,
  lead_days      smallint generated always as
                   (((target_time at time zone 'Asia/Dhaka')::date - issued_on)) stored,
  model_version  text,
  created_at     timestamptz not null default now(),
  primary key (source, station_id, issued_on, target_time)
);
create index if not exists forecasts_station_target_idx on forecasts (station_id, target_time);

-- ---------- Engine run log (admin only) ----------
create table if not exists engine_runs (
  id           bigserial primary key,
  task         text not null,
  started_at   timestamptz not null default now(),
  finished_at  timestamptz,
  status       text not null default 'running',
  message      text,
  stats        jsonb
);
alter table engine_runs add column if not exists progress text;

-- =====================================================================
-- Views (dropped and recreated so changes apply on re-run)
-- =====================================================================
drop function if exists forecast_skill(text, date, date);
drop view if exists forecast_vs_observed, station_status, latest_forecasts, daily_wl cascade;

-- FFWC reads 5 times a day (06,09,12,15,18 local). "Daily WL" below is the
-- mean of the day's readings. If your model was trained on the 06:00 reading,
-- replace avg(wl) with:
--   avg(wl) filter (where extract(hour from obs_time at time zone 'Asia/Dhaka') = 6)
create view daily_wl with (security_invoker = true) as
select station_id,
       (obs_time at time zone 'Asia/Dhaka')::date as day,
       avg(wl)::real as wl,
       max(wl)::real as wl_max,
       count(*)      as n_obs
from observed_wl
group by 1, 2;

create view latest_forecasts with (security_invoker = true) as
select f.*
from forecasts f
join (select source, station_id, max(issued_on) as issued_on
      from forecasts group by 1, 2) m using (source, station_id, issued_on);

create view station_status with (security_invoker = true) as
with last_obs as (
  select distinct on (station_id) station_id, obs_time, wl
  from observed_wl order by station_id, obs_time desc
),
fc as (
  select station_id, max(wl) as peak_wl,
         (array_agg(target_time order by wl desc))[1] as peak_time
  from latest_forecasts where source = 'ours' group by station_id
)
select s.id, s.name, s.river, s.basin, s.district, s.lat, s.lon,
       s.danger_level,
       coalesce(s.warning_level, s.danger_level - 0.5) as warning_level,
       s.rhwl,
       l.wl as current_wl, l.obs_time as last_obs_time,
       round(((l.wl - s.danger_level) * 100)::numeric) as cm_vs_danger,
       case
         when l.wl is null or s.danger_level is null then 'unknown'
         when l.wl >= s.danger_level then 'danger'
         when l.wl >= coalesce(s.warning_level, s.danger_level - 0.5) then 'warning'
         else 'normal'
       end as status,
       fc.peak_wl as forecast_peak_wl, fc.peak_time as forecast_peak_time,
       coalesce(fc.peak_wl >= s.danger_level, false) as forecast_exceeds_danger
from stations s
left join last_obs l on l.station_id = s.id
left join fc on fc.station_id = s.id
where s.scrape_enabled;          -- only stations you track appear on the site

-- Matched on the exact reading time, so a forecast only scores once FFWC posts that reading.
create view forecast_vs_observed with (security_invoker = true) as
select f.source, f.station_id, f.issued_on, f.target_time, f.target_date, f.lead_days,
       f.wl as forecast_wl, o.wl as observed_wl, (f.wl - o.wl) as error
from forecasts f
join observed_wl o on o.station_id = f.station_id and o.obs_time = f.target_time;

create function forecast_skill(p_station text, p_from date, p_to date)
returns table (source text, lead_days smallint, n bigint, mae real, rmse real, bias real)
language sql stable as $$
  select source, lead_days, count(*),
         avg(abs(error))::real, sqrt(avg(error * error))::real, avg(error)::real
  from forecast_vs_observed
  where station_id = p_station and target_date between p_from and p_to
  group by source, lead_days order by source, lead_days;
$$;

-- =====================================================================
-- Row Level Security: anon key can read, never write.
-- =====================================================================
alter table stations    enable row level security;
alter table rf_stations enable row level security;
alter table observed_wl enable row level security;
alter table observed_rf enable row level security;
alter table forecasts   enable row level security;
alter table engine_runs enable row level security;

do $$
declare t text;
begin
  foreach t in array array['stations','rf_stations','observed_wl','observed_rf','forecasts'] loop
    execute format('drop policy if exists public_read on %I', t);
    execute format('create policy public_read on %I for select to anon, authenticated using (true)', t);
  end loop;
end $$;

grant select on stations, rf_stations, observed_wl, observed_rf, forecasts,
                daily_wl, latest_forecasts, station_status, forecast_vs_observed
  to anon, authenticated;
revoke all on engine_runs from anon, authenticated;
grant execute on function forecast_skill(text, date, date) to anon, authenticated;
