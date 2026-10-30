"""Main engine: sync stations -> scrape -> predict -> store.

  python engine.py sync-stations --wl wl_stations.json --rf rainfall_stations.json --apply-targets
  python engine.py scrape                         # all enabled stations
  python engine.py scrape --wl SW7.5 SW42 --rf CL9
  python engine.py backfill-wl --from 2026-05-01 --to 2026-10-11 [--wl SW7.5]
  python engine.py predict [--date 2026-10-01] [--wl SW7.5]
  python engine.py all                            # scrape enabled + predict
  python engine.py import-wl history_wl.csv      # station_id,date,wl
  python engine.py import-rf history_rf.csv      # rf_station_id,date,rainfall_mm
"""
import argparse
import json
import logging
import threading
import traceback
from datetime import date, datetime, time, timedelta

import pandas as pd

import config
import db
import scraper
from predictor import Predictor

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
    return stats, errors


def backfill_wl(conn, run_id: int, start: date, end: date, wl_codes=None) -> tuple[dict, list]:
    """The WL endpoint returns ~60 days ending at a date, so walk backwards."""
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
    today = _today()
    model = Predictor()
    stats, errors = {"predicted": 0, "skipped": {}}, []
    stations = db.model_stations(conn, wl_codes)
    for n, (sid, rf_sid) in enumerate(stations, 1):
        db.set_progress(conn, run_id, f"{n}/{len(stations)} predict {sid}")
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
            df = db.get_input_window(conn, sid, rf_sid, end, config.INPUT_DAYS)
            y = model.predict(sid, df)
            if y is None:
                stats["skipped"][sid] = "too many missing days in input window"
                continue
            rows = [(sid, end, end + timedelta(days=k + 1), float(v)) for k, v in enumerate(y)]
            db.upsert_forecasts(conn, "ours", rows, config.MODEL_VERSION)
            conn.commit()
            stats["predicted"] += 1
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            log.exception("prediction failed for %s", sid)
            errors.append(f"{sid}: {exc}")
    return stats, errors


# ---- logged task runner (CLI, admin panel, scheduler) --------------------
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
                if task in ("predict", "all"):
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
    p.add_argument("task", choices=[*TASKS, "sync-stations", "import-wl", "import-rf"])
    p.add_argument("path", nargs="?", help="CSV for import-wl / import-rf")
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
        import_wl_csv(a.path)
    elif a.task == "import-rf":
        import_rf_csv(a.path)
    else:
        print(run_task(a.task, a.wl, a.rf, end_date=a.end, start_date=a.start, issue_date=a.date))
