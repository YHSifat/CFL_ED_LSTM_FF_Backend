"""Local admin panel.  Run:  uvicorn admin:app --host 127.0.0.1 --port 8000
No login - keep it bound to 127.0.0.1."""
from datetime import date
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
import db
import engine

app = FastAPI(title="Flood engine admin")
PAGE = (Path(__file__).parent / "admin.html").read_text(encoding="utf-8")


class WLSetting(BaseModel):
    id: str
    scrape_enabled: bool | None = None
    has_model: bool | None = None
    rf_station_id: str | None = None


class RFSetting(BaseModel):
    id: str
    scrape_enabled: bool


class Settings(BaseModel):
    wl: list[WLSetting] = []
    rf: list[RFSetting] = []


class NewRF(BaseModel):
    id: str
    api_id: str
    name: str | None = None
    mapped_wl: list[str] = []
    enable: bool = True


class SyncReq(BaseModel):
    wl_records: list[dict] | None = None
    rf_records: list[dict] | None = None
    apply_targets: bool = False


class RunReq(BaseModel):
    task: str
    wl: list[str] | None = None
    rf: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    issue_date: date | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/stations")
def stations():
    with db.connect() as conn:
        return jsonable_encoder({
            "wl": db.list_wl_stations(conn),
            "rf": db.list_rf_stations(conn),
            "ffwc_urls": {"wl": bool(config.FFWC_WL_STATIONS_URL), "rf": bool(config.FFWC_RF_STATIONS_URL)},
        })


@app.post("/api/stations/settings")
def save_settings(s: Settings):
    with db.connect() as conn:
        n = db.update_wl_settings(conn, [w.model_dump(exclude_unset=True) for w in s.wl])
        n += db.update_rf_settings(conn, [r.model_dump() for r in s.rf])
        conn.commit()
    return {"updated": n}


@app.post("/api/rf-stations")
def add_rf(r: NewRF):
    with db.connect() as conn:
        db.upsert_rf_stations(conn, [{"id": r.id.strip(), "api_id": r.api_id.strip(), "name": r.name}])
        missing = db.map_rf_to_wl(conn, r.id.strip(), [c.strip() for c in r.mapped_wl if c.strip()],
                                  enable=r.enable)
        conn.commit()
    return {"added": r.id, "unknown_wl_codes": missing}


@app.post("/api/sync")
def sync(req: SyncReq):
    try:
        return engine.sync_stations(req.wl_records, req.rf_records, req.apply_targets)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/run")
def run(req: RunReq, background: BackgroundTasks):
    if req.task not in engine.TASKS:
        raise HTTPException(404, "unknown task")
    if req.task == "backfill-wl" and not req.start_date:
        raise HTTPException(400, "backfill needs a start date")
    if engine.LOCK.locked():
        raise HTTPException(409, "A task is already running")
    background.add_task(_safe_run, req)
    return {"started": req.task}


def _safe_run(req: RunReq):
    try:
        engine.run_task(req.task, req.wl, req.rf, end_date=req.end_date,
                        start_date=req.start_date, issue_date=req.issue_date)
    except engine.EngineBusy:
        pass
    except Exception:  # noqa: BLE001 - recorded in engine_runs
        engine.log.exception("task %s failed", req.task)


@app.get("/api/runs")
def runs():
    with db.connect() as conn:
        return jsonable_encoder(db.recent_runs(conn))
