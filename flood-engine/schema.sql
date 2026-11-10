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
-- For data_stamp(): max() over these is a single index lookup.
create index if not exists forecasts_created_at_idx on forecasts (created_at);
create index if not exists observed_wl_obs_time_idx on observed_wl (obs_time);

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
drop function if exists data_stamp();
drop function if exists comparison_skill(text);
drop function if exists comparison_issues(text);
drop function if exists horizon_performance(text, int, date, date);
drop function if exists horizon_skill(text, date, date);
drop view if exists forecast_steps cascade;
drop view if exists forecast_comparison, forecast_vs_observed, station_status, latest_forecasts, daily_wl cascade;

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

-- Ours vs FFWC on the same station, issue day and target time (FFWC's 5-day forecast is
-- at 06-18h, its PDF bulletins at 09:00 / 06:00), with the reading once FFWC posts it.
-- ffwc_product = forecasts.model_version of the FFWC row ('ffwc-5day', 'ffwc-10day-mean').
create view forecast_comparison with (security_invoker = true) as
select f.station_id, f.issued_on, f.target_time, f.target_date, f.lead_days,
       f.model_version as ffwc_product,
       o.wl as ours_wl, f.wl as ffwc_wl, w.wl as observed_wl,
       (o.wl - w.wl) as ours_error, (f.wl - w.wl) as ffwc_error
from forecasts f
join forecasts o on o.source = 'ours' and o.station_id = f.station_id
                and o.issued_on = f.issued_on and o.target_time = f.target_time
left join observed_wl w on w.station_id = f.station_id and w.obs_time = f.target_time
where f.source = 'ffwc';

-- Matched error per FFWC product and lead day (all stations when p_station is null).
create function comparison_skill(p_station text default null)
returns table (ffwc_product text, lead_days smallint, n bigint,
               mae_ours real, mae_ffwc real, rmse_ours real, rmse_ffwc real)
language sql stable as $$
  select ffwc_product, lead_days, count(*),
         avg(abs(ours_error))::real, avg(abs(ffwc_error))::real,
         sqrt(avg(ours_error * ours_error))::real, sqrt(avg(ffwc_error * ffwc_error))::real
  from forecast_comparison
  where observed_wl is not null and (p_station is null or station_id = p_station)
  group by 1, 2 order by 1, 2;
$$;

-- Issue days that have both forecasts for a station, newest first.
create function comparison_issues(p_station text)
returns table (issued_on date, ffwc_product text, n bigint, n_observed bigint)
language sql stable as $$
  select issued_on, ffwc_product, count(*), count(observed_wl)
  from forecast_comparison where station_id = p_station
  group by 1, 2 order by 1 desc, 2;
$$;

-- Our forecasts numbered by step: 1 = first reading after issue (next day 06:00),
-- 5 = next day 18:00, ... 75 = day 15 18:00. Website horizons: 3h=1, 6h=2, 12h=4,
-- 1D=5, 3D=15, 7D=35, 15D=75. persistence_wl = last reading at or before issue day 18:00.
create view forecast_steps with (security_invoker = true) as
select f.station_id, f.issued_on, f.target_time, f.target_date,
       ((f.target_date - f.issued_on - 1) * 5
        + array_position(array[6, 9, 12, 15, 18],
                         extract(hour from f.target_time at time zone 'Asia/Dhaka')::int))::int as step,
       f.wl as predicted, o.wl as observed
from forecasts f
left join observed_wl o on o.station_id = f.station_id and o.obs_time = f.target_time
where f.source = 'ours';

-- One step's forecasts for targets in [p_from, p_to], with the reading (chart data).
create function horizon_performance(p_station text, p_step int, p_from date, p_to date)
returns table (issued_on date, target_time timestamptz, predicted real, observed real)
language sql stable as $$
  select issued_on, target_time, predicted, observed from forecast_steps
  where station_id = p_station and step = p_step and target_date between p_from and p_to
  order by target_time;
$$;

-- Error per step for targets in [p_from, p_to] (all stations when p_station is null).
create function horizon_skill(p_station text, p_from date, p_to date)
returns table (step int, n bigint, mae real, rmse real, nse real, mae_persistence real)
language sql stable as $$
  with base as (
    select station_id, issued_on, step, predicted, observed from forecast_steps
    where observed is not null and target_date between p_from and p_to
      and (p_station is null or station_id = p_station)
  ),
  persistence as (     -- one lookup per station and issue day, not per row
    select k.station_id, k.issued_on,
           (select w.wl from observed_wl w
            where w.station_id = k.station_id
              and w.obs_time <= (k.issued_on + time '18:00') at time zone 'Asia/Dhaka'
            order by w.obs_time desc limit 1) as wl
    from (select distinct station_id, issued_on from base) k
  ),
  s as (
    select b.*, p.wl as persistence_wl, avg(b.observed) over (partition by b.step) as observed_mean
    from base b join persistence p using (station_id, issued_on)
  )
  select step, count(*),
         avg(abs(predicted - observed))::real,
         sqrt(avg((predicted - observed) ^ 2))::real,
         -- NSE per station only: across stations it mostly measures their level differences
         case when p_station is not null and count(*) > 1 and var_pop(observed) > 0
              then (1 - sum((predicted - observed) ^ 2) / sum((observed - observed_mean) ^ 2))::real
         end,
         avg(abs(persistence_wl - observed))::real
  from s group by step order by step;
$$;

create function forecast_skill(p_station text, p_from date, p_to date)
returns table (source text, lead_days smallint, n bigint, mae real, rmse real, bias real)
language sql stable as $$
  select source, lead_days, count(*),
         avg(abs(error))::real, sqrt(avg(error * error))::real, avg(error)::real
  from forecast_vs_observed
  where station_id = p_station and target_date between p_from and p_to
  group by source, lead_days order by source, lead_days;
$$;

-- Newest reading and newest forecast write (created_at is reset on re-runs), plus the last
-- station edit. The website caches query results in the browser until this value changes.
create function data_stamp()
returns table (last_obs timestamptz, last_forecast timestamptz, last_station timestamptz)
language sql stable as $$
  select (select max(obs_time) from observed_wl),
         (select max(created_at) from forecasts),
         (select max(updated_at) from stations);
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
                daily_wl, latest_forecasts, station_status, forecast_vs_observed, forecast_comparison,
                forecast_steps
  to anon, authenticated;
revoke all on engine_runs from anon, authenticated;
grant execute on function forecast_skill(text, date, date), data_stamp() to anon, authenticated;
grant execute on function comparison_skill(text), comparison_issues(text) to anon, authenticated;
grant execute on function horizon_performance(text, int, date, date), horizon_skill(text, date, date)
  to anon, authenticated;
