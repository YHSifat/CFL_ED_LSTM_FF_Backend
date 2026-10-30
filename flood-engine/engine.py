"""Main engine: sync stations -> scrape -> predict -> store.

  python engine.py sync-stations --wl wl_stations.json --rf rainfall_stations.json --apply-targets
  python engine.py scrape                         # all enabled stations
  python engine.py scrape --wl SW7.5 SW42 --rf CL9
  python engine.py backfill-wl --from 2026-05-01 --to 2026-10-11 [--wl SW7.5]
  python engine.py predict [--date 2026-10-01] [--wl SW7.5]
  python engine.py predict --from 2026-10-15 --to 2026-10-29   # one forecast per issue day
  python engine.py all                            # scrape enabled + predict
  python engine.py import-wl history_wl.csv      # station_id,date,wl
  python engine.py import-rf history_rf.csv      # rf_station_id,date,rainfall_mm
  python engine.py import-rf-json old/rf_CL9.json ...        # saved FFWC rainfall responses
  python engine.py import-ffwc-pdf models/ffwc_predictions/*.pdf   # FFWC forecast bulletins
"""
import argparse
import json
import logging
import re
import threading
import traceback
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

import config
import db
import ffwc_pdf
import scraper
from forecaster import INPUT_DAYS, Forecaster, InputError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("engine")

LOCK = threading.Lock()   # one engine job at a time
TASKS = ("scrape", "predict", "all", "backfill-wl")


class EngineBusy(RuntimeError):
    pass


def _today() -> date:
    return datetime.now(config.TZ).date()


# ---- station sync (quick, runs outside the task lock) --------------------
def sync_stations(wl_records: list | None = None, rf_records: list | None = None,
                  apply_targets: bool = False) -> dict:
    """Load station lists from JSON records (or from the FFWC URLs if set and
    no records are given). With apply_targets, the WL stations named in a
    rainfall list's mapped_wl_station_id are linked to it and enabled."""
    out = {}
    wl = scraper.parse_wl_stations(wl_records) if wl_records is not None else (
        scraper.fetch_wl_stations() if config.FFWC_WL_STATIONS_URL else None)
    rf = scraper.parse_rf_stations(rf_records) if rf_records is not None else (
        scraper.fetch_rf_stations() if config.FFWC_RF_STATIONS_URL else None)
    with db.connect() as conn:
        if wl is not None:
            out["wl_stations"] = db.upsert_stations(conn, wl)
        if rf is not None:
            out["rf_stations"] = db.upsert_rf_stations(conn, rf)
            missing = []
            for r in rf:
                if r["mapped_wl"]:
                    missing += db.map_rf_to_wl(conn, r["id"], r["mapped_wl"], enable=apply_targets)
            if missing:
                out["unknown_wl_codes_in_mapping"] = sorted(set(missing))
        conn.commit()
    log.info("sync: %s", out)
    return out


# ---- scraping --------------------------------------------------------------
def scrape(conn, run_id: int, wl_codes=None, rf_codes=None, end_date: date | None = None,
           do_wl: bool = True, do_rf: bool = True) -> tuple[dict, list]:
    end_date = end_date or _today()
    stats = {"wl_rows": 0, "rf_rows": 0, "wl_stations": 0, "rf_stations": 0, "empty": []}
    errors = []

    # None = all tracked stations, [] = none of this kind, [codes] = just those
    wl_list = db.wl_targets(conn, wl_codes) if do_wl and wl_codes != [] else []
    rf_list = db.rf_targets(conn, rf_codes) if do_rf and rf_codes != [] else []
    total, i = len(wl_list) + len(rf_list), 0

    for code, web_id in wl_list:
        i += 1
        db.set_progress(conn, run_id, f"{i}/{total} WL {code}")
        try:
            rows, meta = scraper.fetch_wl(web_id, end_date)
            if not rows:
                stats["empty"].append(code)
                continue
            db.upsert_observed_wl(conn, [(code, t, wl) for t, wl in rows])
            if meta:
                db.update_station_levels(conn, code, meta.get("danger_level"), meta.get("rhwl"))
            conn.commit()
            stats["wl_rows"] += len(rows)
            stats["wl_stations"] += 1
        except scraper.FFWCError as exc:
            conn.rollback()
            errors.append(f"WL {code}: {exc}")
            if "x-ffwc-internal-key" in str(exc):
                break                   # every request will fail the same way
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            log.exception("WL %s failed", code)
            errors.append(f"WL {code}: {exc}")

    for code, api_id in rf_list:
        i += 1
        db.set_progress(conn, run_id, f"{i}/{total} RF {code}")
        try:
            rows = scraper.fetch_rf(api_id)
            if not rows:
                stats["empty"].append(code)
                continue
            db.upsert_observed_rf(conn, [(code, d, mm) for d, mm in rows])
            conn.commit()
            stats["rf_rows"] += len(rows)
            stats["rf_stations"] += 1
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            log.exception("RF %s failed", code)
            errors.append(f"RF {code}: {exc}")
            if "x-ffwc-internal-key" in str(exc):
                break

    # FFWC's own 5-day forecast, one request for all stations (full scrapes only)
    if do_wl and wl_codes is None:
        db.set_progress(conn, run_id, "FFWC 5-day forecast")
        try:
            stats["ffwc_rows"] = scrape_ffwc_5day(conn)
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            log.exception("FFWC 5-day forecast failed")
            errors.append(f"FFWC forecast: {exc}")
    return stats, errors


def scrape_ffwc_5day(conn) -> int:
    code_of = {s["web_id"]: s["id"] for s in db.model_station_info(conn)}
    rows = [(code_of[w], issued, t, wl) for w, issued, t, wl in scraper.fetch_ffwc_5day() if w in code_of]
    n = db.upsert_forecasts(conn, "ffwc", rows, ffwc_pdf.PRODUCT_5DAY)
    conn.commit()
    return n


def backfill_wl(conn, run_id: int, start: date, end: date, wl_codes=None) -> tuple[dict, list]:
    """The WL endpoint returns 40 days ending at a date, so walk backwards."""
    stats, errors = {"wl_rows": 0, "requests": 0}, []
    cursor = end
    while cursor >= start:
        s, e = scrape(conn, run_id, wl_codes=wl_codes, end_date=cursor, do_rf=False)
        stats["wl_rows"] += s["wl_rows"]
        stats["requests"] += s["wl_stations"] + len(s["empty"])
        errors += [f"{cursor}: {m}" for m in e]
        cursor -= timedelta(days=config.WL_BACKFILL_STEP_DAYS)
    return stats, errors


# ---- prediction ------------------------------------------------------------
def predict(conn, run_id: int, issue_date: date | None = None, wl_codes=None) -> tuple[dict, list]:
    """Forecast the 75 readings after each station's input day (issue_date, else the
    last day with WL data) and store them as source 'ours'."""
    today = _today()
    model = Forecaster()
    stats, errors = {"predicted": 0, "skipped": {}, "filled": {}}, []
    stations = db.model_stations(conn, wl_codes)
    for n, (sid, rf_sid) in enumerate(stations, 1):
        db.set_progress(conn, run_id, f"{n}/{len(stations)} predict {sid}")
        if not model.has_station(sid):
            stats["skipped"][sid] = "not a station the model was trained on"
            continue
        if not rf_sid:
            stats["skipped"][sid] = "no rainfall station mapped"
            continue
        end = issue_date or db.last_wl_day(conn, sid)
        if end is None:
            stats["skipped"][sid] = "no data"
            continue
        if issue_date is None and (today - end).days > config.MAX_STALENESS_DAYS:
            stats["skipped"][sid] = f"stale data (last {end})"
            continue
        try:
            start = end - timedelta(days=INPUT_DAYS - 1)
            y, info = model.predict(sid, db.get_wl_readings(conn, sid, start, end),
                                    db.get_rainfall(conn, rf_sid, start, end), end)
            if info["wl_filled"] or info["rain_missing_days"]:
                stats["filled"][sid] = info
            rows = [(sid, end, t.to_pydatetime(), float(v)) for t, v in y.items()]
            db.upsert_forecasts(conn, "ours", rows, model.version)
            conn.commit()
            stats["predicted"] += 1
        except InputError as exc:
            conn.rollback()
            stats["skipped"][sid] = str(exc)
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            log.exception("prediction failed for %s", sid)
            errors.append(f"{sid}: {exc}")
    return stats, errors


# ---- logged task runner (CLI, admin panel, scheduler) --------------------
def _days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date is before start date")
    return [start + timedelta(days=k) for k in range((end - start).days + 1)]


def run_task(task: str, wl: list[str] | None = None, rf: list[str] | None = None,
             end_date: date | None = None, start_date: date | None = None,
             issue_date: date | None = None) -> dict:
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    if not LOCK.acquire(blocking=False):
        raise EngineBusy("Another engine task is already running")
    params = {k: v for k, v in dict(wl=wl, rf=rf, end_date=end_date, start_date=start_date,
                                     issue_date=issue_date).items() if v}
    try:
        with db.connect() as conn:
            run_id = db.start_run(conn, task, json.loads(json.dumps(params, default=str)))
            stats, errors = {}, []
            try:
                if task in ("scrape", "all"):
                    stats["scrape"], e = scrape(conn, run_id, wl, rf, end_date)
                    errors += e
                if task == "backfill-wl":
                    if not start_date:
                        raise ValueError("backfill-wl needs a start date")
                    stats["backfill"], e = backfill_wl(conn, run_id, start_date, end_date or _today(), wl)
                    errors += e
                if task == "predict" and start_date:
                    # One forecast per issue day in [start_date, end_date] (hindcasts for past days)
                    stats["predict"] = {"predicted": 0, "days": {}}
                    for day in _days(start_date, end_date or start_date):
                        s, e = predict(conn, run_id, day, wl)
                        stats["predict"]["predicted"] += s["predicted"]
                        stats["predict"]["days"][day.isoformat()] = {
                            "predicted": s["predicted"], "skipped": s["skipped"]}
                        errors += [f"{day}: {m}" for m in e]
                elif task in ("predict", "all"):
                    codes = wl if task == "predict" else None
                    stats["predict"], e = predict(conn, run_id, issue_date, codes)
                    errors += e
                done = any(v.get("wl_rows") or v.get("rf_rows") or v.get("predicted")
                           for v in stats.values())
                status = "ok" if not errors else ("partial" if done else "failed")
                db.finish_run(conn, run_id, status, "\n".join(errors), stats)
            except Exception:
                conn.rollback()
                db.finish_run(conn, run_id, "failed", traceback.format_exc(), stats)
                raise
        log.info("%s finished: %s", task, status)
        return {"task": task, "status": status, "stats": stats, "errors": errors}
    finally:
        LOCK.release()


# ---- one-off CSV history import -------------------------------------------
def import_wl_csv(path: str):
    df = pd.read_csv(path, dtype={"station_id": str}).dropna(subset=["station_id", "date", "wl"])
    with db.connect() as conn:
        df = df[df["station_id"].isin(db.station_ids(conn))]
        rows = [(r.station_id, datetime.combine(pd.Timestamp(r.date).date(), time(6), config.TZ),
                 float(r.wl)) for r in df.itertuples()]
        n = db.upsert_observed_wl(conn, rows)
        conn.commit()
    log.info("imported %d WL rows (stations not in the DB were dropped)", n)


def import_ffwc_pdf(paths: list[str]):
    """FFWC forecast bulletins (5-day / 10-day PDFs) -> forecasts, source 'ffwc'."""
    with db.connect() as conn:
        stations = db.model_station_info(conn)
        for path in paths:
            b = ffwc_pdf.parse(path)
            rows, unmatched = [], []
            for sf in b.stations:
                sid = ffwc_pdf.match_station(sf, stations)
                if sid is None:
                    unmatched.append(" ".join(sf.words))
                    continue
                rows += [(sid, b.issued_on, t, wl) for t, wl in sf.values]
            n = db.upsert_forecasts(conn, "ffwc", rows, b.product)
            conn.commit()
            log.info("%s: %s issued %s, %d rows for %d model stations (%d other stations skipped)",
                     path, b.product, b.issued_on, n, len({r[0] for r in rows}), len(unmatched))


def import_rf_json(paths: list[str]):
    """Saved FFWC rainfall responses named rf_<CODE>.json (e.g. the old scraper's files)."""
    with db.connect() as conn:
        api_ids = {r["id"]: r["api_id"] for r in db.list_rf_stations(conn)}
        for path in paths:
            code = re.match(r"rf_(.+)\.json$", Path(path).name)
            if not code or code.group(1) not in api_ids:
                log.warning("%s: unknown rainfall station, skipped", path)
                continue
            code = code.group(1)
            rows = scraper.parse_rf(json.load(open(path, encoding="utf-8")), api_ids[code])
            n = db.upsert_observed_rf(conn, [(code, d, mm) for d, mm in rows])
            conn.commit()
            log.info("%s: %d rainfall rows for %s", path, n, code)


def import_rf_csv(path: str):
    df = pd.read_csv(path, dtype={"rf_station_id": str}).dropna(
        subset=["rf_station_id", "date", "rainfall_mm"])
    with db.connect() as conn:
        db.upsert_rf_stations(conn, [{"id": s, "api_id": None, "name": None}
                                     for s in df["rf_station_id"].unique()])
        rows = [(r.rf_station_id, pd.Timestamp(r.date).date(), float(r.rainfall_mm))
                for r in df.itertuples()]
        n = db.upsert_observed_rf(conn, rows)
        conn.commit()
    log.info("imported %d rainfall rows", n)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=[*TASKS, "sync-stations", "import-wl", "import-rf",
                                    "import-rf-json", "import-ffwc-pdf"])
    p.add_argument("path", nargs="*", help="file(s) for the import-* tasks")
    p.add_argument("--wl", nargs="*", help="WL station codes, e.g. SW7.5 SW42 (sync: JSON file)")
    p.add_argument("--rf", nargs="*", help="rainfall station codes, e.g. CL9 (sync: JSON file)")
    p.add_argument("--from", dest="start", type=date.fromisoformat)
    p.add_argument("--to", dest="end", type=date.fromisoformat)
    p.add_argument("--date", type=date.fromisoformat, help="issue date for predict")
    p.add_argument("--apply-targets", action="store_true")
    a = p.parse_args()

    def _load(paths):
        return json.load(open(paths[0], encoding="utf-8")) if paths else None

    if a.task == "sync-stations":
        print(sync_stations(_load(a.wl), _load(a.rf), a.apply_targets))
    elif a.task == "import-wl":
        import_wl_csv(a.path[0])
    elif a.task == "import-rf":
        import_rf_csv(a.path[0])
    elif a.task == "import-rf-json":
        import_rf_json(a.path)
    elif a.task == "import-ffwc-pdf":
        import_ffwc_pdf(a.path)
    else:
        print(run_task(a.task, a.wl, a.rf, end_date=a.end, start_date=a.start, issue_date=a.date))
