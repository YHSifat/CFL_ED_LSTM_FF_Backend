"""FFWC API client: station lists, observed water level, rainfall.

Endpoints and formats taken from the ffwc.gov.bd site's own requests:
  WL : /observed-waterlevel-by-station-and-date/{web_id}/{YYYY-MM-DD}/  (40 days, 3-hourly)
  RF : /rainfall-by-station/{api_id}/                                  (~40 days, daily)
  FC : /five-days-forecast-waterlevel/   (FFWC 5-day forecast, all stations, 3-hourly)
"""
import logging
import time
from datetime import date, datetime, timedelta

import requests

import config

log = logging.getLogger(__name__)


class FFWCError(RuntimeError):
    pass


_session = requests.Session()
_session.headers.update({
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,bn;q=0.8",
    "origin": "https://www.ffwc.gov.bd",
    "referer": "https://www.ffwc.gov.bd/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": config.USER_AGENT,
})
if config.FFWC_API_KEY:
    _session.headers["x-ffwc-internal-key"] = config.FFWC_API_KEY

_last_request = 0.0


def _throttle():
    global _last_request
    wait = config.REQUEST_DELAY_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def get_json(url: str, retries: int = 3):
    err = None
    for attempt in range(1, retries + 1):
        _throttle()
        try:
            resp = _session.get(url, timeout=30)
        except requests.RequestException as exc:
            err = exc
        else:
            if resp.status_code in (401, 403):
                raise FFWCError(
                    f"HTTP {resp.status_code} from FFWC. The x-ffwc-internal-key value has "
                    "probably changed: copy the new one from the browser Network tab into .env")
            if resp.status_code == 404:
                return []
            if resp.ok:
                try:
                    return resp.json()
                except ValueError as exc:
                    err = exc
            else:
                err = FFWCError(f"HTTP {resp.status_code}")
        if attempt < retries:
            log.warning("GET %s failed (%s), retry %d", url, err, attempt)
            time.sleep(2 ** attempt)
    raise FFWCError(f"GET {url} failed: {err}")


def _as_list(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise FFWCError(f"Unexpected response shape: {str(payload)[:200]}")


# ---- value cleaning (FFWC uses "-", "#N/A", strings for numbers) ----------
_NULLS = {"", "-", "#N/A", "N/A", "NA", "null", "None"}


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in _NULLS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _text(v):
    if v is None:
        return None
    s = str(v).strip()
    return None if s in _NULLS else s


def parse_wl_time(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if config.FFWC_TIME_IS_LOCAL or dt.tzinfo is None:
        dt = dt.replace(tzinfo=config.TZ)
    return dt


def parse_rf_date(s: str) -> date | None:
    s = str(s).strip()
    for fmt in (config.RF_DATE_FORMAT, "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


# ---- station lists ------------------------------------------------------
def parse_wl_stations(records: list[dict]) -> list[dict]:
    """FFWC water-level station list (format of wl_stations.json)."""
    out = []
    for r in records:
        code, web_id = _text(r.get("st_id")), r.get("web_id")
        if not code or web_id in (None, ""):
            continue
        dl = _num(r.get("dl"))
        out.append({
            "id": code,
            "web_id": int(web_id),
            "name": _text(r.get("station")) or code,
            "river": _text(r.get("river")),
            "basin": _text(r.get("basin")),
            "district": _text(r.get("district")),
            "division": _text(r.get("division")),
            "lat": _num(r.get("latitude")),
            "lon": _num(r.get("longitude")),
            "danger_level": dl if dl is not None else _num(r.get("pmdl")),
            "rhwl": _num(r.get("rhwl")),
            "ffwc_active": bool(r.get("status")) if r.get("status") is not None else None,
        })
    return out


def parse_rf_stations(records: list[dict]) -> list[dict]:
    """Rainfall station list. Accepts your rainfall_stations.json format
    ({id, station_id, name, mapped_wl_station_id}); extend the key tuples if
    FFWC's own list uses other names."""
    out = []
    for r in records:
        code = _text(next((r[k] for k in ("station_id", "st_code", "code") if r.get(k)), None))
        api_id = _text(next((r[k] for k in ("id", "web_id", "st_id") if r.get(k) is not None), None))
        if not code or not api_id:
            continue
        mapped = _text(r.get("mapped_wl_station_id"))
        out.append({
            "id": code,
            "api_id": api_id,
            "name": _text(r.get("name") or r.get("station")),
            "basin": _text(r.get("basin")),
            "lat": _num(r.get("latitude") or r.get("lat")),
            "lon": _num(r.get("longitude") or r.get("long") or r.get("lon")),
            "mapped_wl": [m.strip() for m in mapped.split(",") if m.strip()] if mapped else [],
        })
    return out


def fetch_wl_stations() -> list[dict]:
    if not config.FFWC_WL_STATIONS_URL:
        raise FFWCError("FFWC_WL_STATIONS_URL not set; import wl_stations.json instead")
    return parse_wl_stations(_as_list(get_json(config.FFWC_WL_STATIONS_URL)))


def fetch_rf_stations() -> list[dict]:
    if not config.FFWC_RF_STATIONS_URL:
        raise FFWCError("FFWC_RF_STATIONS_URL not set; import a rainfall stations JSON instead")
    return parse_rf_stations(_as_list(get_json(config.FFWC_RF_STATIONS_URL)))


# ---- observations -------------------------------------------------------
def parse_wl(records: list[dict], web_id: int) -> tuple[list[tuple], dict]:
    """-> ([(obs_time, wl)], {"danger_level", "rhwl"} from the latest record)"""
    rows, meta, skipped = {}, {}, 0
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("st_id") is not None and str(r["st_id"]) != str(web_id):
            skipped += 1
            continue
        t, wl = parse_wl_time(r.get("wl_date")), _num(r.get("waterlevel"))
        if t is None or wl is None:
            continue
        rows[t] = wl                       # de-duplicates repeated timestamps
        if not meta or t > meta["_t"]:
            meta = {"_t": t, "danger_level": _num(r.get("dangerlevel")),
                    "rhwl": _num(r.get("riverhighestwaterlevel"))}
    if skipped:
        log.warning("web_id %s: skipped %d records for other stations", web_id, skipped)
    meta.pop("_t", None)
    return sorted(rows.items()), meta


def fetch_wl(web_id: int, end_date: date) -> tuple[list[tuple], dict]:
    url = config.WL_URL.format(web_id=web_id, end_date=end_date.isoformat())
    return parse_wl(_as_list(get_json(url)), web_id)


def parse_rf(records: list[dict], api_id: str) -> list[tuple]:
    """-> [(obs_date, rainfall_mm)]"""
    rows = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("st_id") is not None and str(r["st_id"]) != str(api_id):
            continue
        d, mm = parse_rf_date(r.get("rf_date")), _num(r.get("rainfall"))
        if d is not None and mm is not None and mm >= 0:
            rows[d] = mm
    return sorted(rows.items())


def fetch_rf(api_id: str) -> list[tuple]:
    return parse_rf(_as_list(get_json(config.RF_URL.format(api_id=api_id))), api_id)


# ---- FFWC's own forecasts ----------------------------------------------------
def parse_ffwc_5day(payload: dict) -> list[tuple]:
    """five-days-forecast-waterlevel: {"<web_id>": [{"2026-10-31 06:00:00Z": 2.31}, ...]}
    -> [(web_id, issued_on, target_time, wl)]. Times are local despite the "Z" (as for WL).
    The response only says when values are for, not when they were issued: the first
    target is 06:00 the day after issue (same as our model), so issued_on = that day - 1."""
    rows = []
    for web_id, steps in (payload or {}).items():
        if not isinstance(steps, list):
            continue
        points = []
        for step in steps:
            for ts, wl in (step.items() if isinstance(step, dict) else ()):
                t, v = parse_wl_time(str(ts).replace(" ", "T")), _num(wl)
                if t is not None and v is not None:
                    points.append((t, v))
        if points:
            issued_on = min(t for t, _ in points).date() - timedelta(days=1)
            rows += [(int(web_id), issued_on, t, v) for t, v in points]
    return rows


def fetch_ffwc_5day() -> list[tuple]:
    return parse_ffwc_5day(get_json(config.FFWC_5DAY_URL))
