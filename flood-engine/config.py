"""Central configuration. Values come from .env (never commit .env)."""
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# Supabase -> Project Settings -> Database -> Connection string -> "Session pooler"
DATABASE_URL = os.getenv("DATABASE_URL", "")   # checked in db.connect()
TZ = ZoneInfo("Asia/Dhaka")

# ---- FFWC API --------------------------------------------------------------
FFWC_API_BASE = os.getenv("FFWC_API_BASE", "https://api.ffwc.gov.bd/data_load")
WL_URL = FFWC_API_BASE + "/observed-waterlevel-by-station-and-date/{web_id}/{end_date}/"
RF_URL = FFWC_API_BASE + "/rainfall-by-station/{api_id}/"

# Header the ffwc.gov.bd site sends with its API calls. It is likely to change
# (it has a version in it); on 401/403 copy the new value from the Network tab.
FFWC_API_KEY = os.getenv("FFWC_API_KEY", "")
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
)

# Optional: URLs of the station-list endpoints (else import the JSON files)
FFWC_WL_STATIONS_URL = os.getenv("FFWC_WL_STATIONS_URL", "")
FFWC_RF_STATIONS_URL = os.getenv("FFWC_RF_STATIONS_URL", "")
# Optional: FFWC forecast endpoint (not wired up until we have a sample response)
FFWC_FORECAST_URL = os.getenv("FFWC_FORECAST_URL", "")

REQUEST_DELAY_S = float(os.getenv("REQUEST_DELAY_S", "3"))   # pause between requests
# The WL endpoint returns ~60 days ending at the requested date; backfill steps by this
WL_BACKFILL_STEP_DAYS = int(os.getenv("WL_BACKFILL_STEP_DAYS", "50"))
# FFWC writes local Bangladesh times with a "Z" suffix (readings at 06:00-18:00),
# so the suffix is ignored and times are treated as Asia/Dhaka.
FFWC_TIME_IS_LOCAL = os.getenv("FFWC_TIME_IS_LOCAL", "true").lower() == "true"
RF_DATE_FORMAT = os.getenv("RF_DATE_FORMAT", "%m-%d-%Y")       # e.g. "09-30-2026"

# ---- Model -----------------------------------------------------------------
MAX_STALENESS_DAYS = int(os.getenv("MAX_STALENESS_DAYS", "2"))

# Clustered federated LSTM checkpoint (forecaster.py). Its architecture, scaler,
# stations and clusters are read from the checkpoint itself.
MODEL_PATH = os.getenv("MODEL_PATH", "models/clustered_fl_vanilla_lstm-local-weighted-0.5.pt")
# Longest run of missing WL readings filled by interpolation (training used 10 = ~2 days)
MAX_WL_INTERP_STEPS = int(os.getenv("MAX_WL_INTERP_STEPS", "10"))
