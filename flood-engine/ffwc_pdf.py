"""Parse FFWC's forecast bulletins (PDF) into forecast rows.

Two products, both text PDFs (no OCR needed):
  5-day  "3 & 5 Days Deterministic Flood Forecast as on 17 October, 2026"
         rows: SL, station, river, DL, today (09:00), then per 24h step:
         forecast (2 decimals), rise/fall cm, above/below DL cm. Targets are 09:00 local.
  10-day "Medium Range 1-10 days Probabilistic Forecast ... As of 09 October, 2026"
         per station 3 lines: "river station DL today d1..d9 Upper Range", then
         "Lower Range" and "Mean" lines. Targets are 06:00 local; we keep the Mean.

Rows identify stations by words (name/river), resolved to station ids by match_station().
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pdfplumber

import config

PRODUCT_5DAY = "ffwc-5day"
PRODUCT_10DAY = "ffwc-10day-mean"

_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")
_ISSUE_5DAY = re.compile(r"Forecast as on (\d{1,2} \w+,? \d{4})", re.I)
_ISSUE_10DAY = re.compile(r"Forecast As of (\d{1,2} \w+,? \d{4})", re.I)
# Bulletin spelling -> FFWC station-list spelling (normalised: lowercase, letters/digits only)
NAME_ALIASES = {"elasin": "elasinghat"}


@dataclass
class StationForecast:
    words: list[str]               # station + river words as printed
    river_first: bool              # 10-day lists river before station
    values: list[tuple] = field(default_factory=list)   # [(target_time, wl)]


@dataclass
class Bulletin:
    product: str
    issued_on: date
    stations: list[StationForecast]


def _issue_date(text: str, pattern: re.Pattern) -> date | None:
    m = pattern.search(text)
    return datetime.strptime(m.group(1).replace(",", ""), "%d %B %Y").date() if m else None


def _local(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=config.TZ)


def parse(path: str | Path) -> Bulletin:
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n".join(pages)
    if (issued := _issue_date(text, _ISSUE_5DAY)) is not None:
        return Bulletin(PRODUCT_5DAY, issued, _parse_5day(text, issued))
    if (issued := _issue_date(text, _ISSUE_10DAY)) is not None:
        return Bulletin(PRODUCT_10DAY, issued, _parse_10day(text, issued))
    raise ValueError(f"{path}: not a recognised FFWC 5-day or 10-day forecast PDF")


def _parse_5day(text: str, issued: date) -> list[StationForecast]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^\*?(\d+)\s*([A-Za-z].*)$", line.strip())
        if not m:
            continue
        tokens = m.group(2).split()
        first_num = next((i for i, t in enumerate(tokens) if _NUMBER.match(t)), None)
        if not first_num:
            continue
        numbers = tokens[first_num:]
        if not all(_NUMBER.match(t) for t in numbers) or len(numbers) < 3:
            continue
        # numbers: DL, today, then (forecast, rise/fall[, above/below]) per day.
        # Forecasts are the only values with decimals after "today".
        forecasts = [float(t) for t in numbers[2:] if "." in t]
        sf = StationForecast(tokens[:first_num], river_first=False)
        sf.values = [(_local(issued + timedelta(days=k), 9), v) for k, v in enumerate(forecasts, 1)]
        if sf.values:
            out.append(sf)
    return out


def _parse_10day(text: str, issued: date) -> list[StationForecast]:
    out, current = [], None
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 3 or tokens[-1] != "Range" and tokens[-1] != "Mean":
            continue
        label = "Mean" if tokens[-1] == "Mean" else tokens[-2]        # Upper / Lower / Mean
        body = tokens[:-1] if label == "Mean" else tokens[:-2]
        first_num = next((i for i, t in enumerate(body) if _NUMBER.match(t)), len(body))
        numbers = body[first_num:]
        if label == "Upper" and first_num > 0:   # "River Station DL today d1..d9 Upper Range"
            current = StationForecast(body[:first_num], river_first=True)
            numbers = numbers[1:]                # drop DL
            out.append(current)
        elif first_num > 0 and current is not None:
            current.words += body[:first_num]    # long names wrap onto the next line
        if label != "Mean" or current is None:
            continue
        # numbers: today (06:00, issue day), then 1..9-day forecasts. Rows with a
        # different count can't be aligned to days reliably, so they are skipped.
        if len(numbers) != 10:
            continue
        current.values = [(_local(issued + timedelta(days=k), 6), float(v))
                          for k, v in enumerate(numbers[1:], 1)]
    return [s for s in out if s.values]


# ---- station matching ---------------------------------------------------------
def _norm(s: str | None) -> str:
    n = re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return NAME_ALIASES.get(n, n)


def match_station(sf: StationForecast, stations: list[dict]) -> str | None:
    """stations: [{id, name, river}] -> id of the station whose name and river match the
    printed words (any split point); falls back to a unique name-only match."""
    by_pair, by_name = {}, {}
    for s in stations:
        by_pair.setdefault((_norm(s["name"]), _norm(s["river"])), []).append(s["id"])
        by_name.setdefault(_norm(s["name"]), []).append(s["id"])
    words = sf.words
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        name, river = (b, a) if sf.river_first else (a, b)
        ids = by_pair.get((_norm(name), _norm(river)))
        if ids and len(ids) == 1:
            return ids[0]
    # Wrapped names can leave the river split around the station name, so try every
    # contiguous run of words as the name, accepting only a unique match.
    for size in range(len(words), 0, -1):
        for i in range(len(words) - size + 1):
            ids = by_name.get(_norm(" ".join(words[i:i + size])))
            if ids and len(ids) == 1:
                return ids[0]
    return None
