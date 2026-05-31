"""
Central configuration for the Grid Resilience Strategy.
All modules import from here to avoid scattered magic constants.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Backtest window ───────────────────────────────────────────────────────────
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2025-12-31"

# ── Supported ISOs ────────────────────────────────────────────────────────────
SUPPORTED_ISOS = ["ERCOT", "PJM", "MISO", "CAISO", "SPP"]

# gridstatus class names — verified against gridstatus 0.34.0
# (import via getattr(gridstatus, cls_name)())
ISO_CLASS_MAP = {
    "ERCOT": "Ercot",
    "PJM": "PJM",  # requires PJM_API_KEY env var (free — pjm.com/api)
    "MISO": "MISO",  # no credentials needed
    "CAISO": "CAISO",  # no credentials needed
    "SPP": "SPP",  # no credentials; uses custom method names (see grid_data.py)
    "NYISO": "NYISO",  # no credentials needed
    "ISO-NE": "ISONE",  # no credentials needed
}

# Default location type for gridstatus .get_lmp() per ISO
# SPP uses separate DA/RT methods so this field is ignored for SPP
ISO_LOCATION_TYPE = {
    "ERCOT": "settlement point",  # gridstatus ≥0.35 dropped "hub"; HB_* are settlement points
    "PJM": "zone",
    "MISO": "LMP",
    "CAISO": "trading_hub",
    "SPP": "hub",  # SPP uses get_lmp_day_ahead_hourly() instead
    "NYISO": "zone",
    "ISO-NE": "hub",
}

# ISOs for which inter-zonal spread fills congestion_frac when component data absent
CONGESTION_SPREAD_ISOS = {"ERCOT", "PJM"}

# Location type used to fetch zone-level LMPs for spread computation.
# PJM already uses "zone" in ISO_LOCATION_TYPE so needs no entry here.
ISO_ZONE_LOCATION_TYPE = {
    "ERCOT": "settlement point",  # LZ_* zones are also settlement points
}

# Primary hub/zone names to pull per ISO for the aggregate stress signal
ISO_BENCHMARK_NODES = {
    "ERCOT": ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"],
    "PJM": ["WESTERN HUB", "EASTERN HUB", "AEP GEN HUB", "APS GEN HUB"],
    "MISO": ["ILLINOIS HUB", "MICHIGAN HUB", "MINNESOTA HUB", "ARKANSAS HUB"],
    "CAISO": ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND"],
    "SPP": ["SPPNORTH_HUB", "SPPSOUTH_HUB"],
}

# ── LMP stress thresholds ─────────────────────────────────────────────────────
# Hours above this $/MWh count toward the "spike hours" stress sub-signal
LMP_SPIKE_THRESHOLD = {
    "ERCOT": 200.0,
    "PJM": 200.0,
    "MISO": 150.0,
    "CAISO": 300.0,
    "SPP": 150.0,
}

# Rolling window (calendar days) for computing LMP z-score baseline
LMP_ZSCORE_WINDOW = 90

# ── Grid Stress Index weights (sum to 1) ──────────────────────────────────────
GSI_WEIGHTS = {
    "lmp_zscore": 0.40,  # normalized daily-max LMP vs rolling history
    "congestion_frac": 0.25,  # congestion component share of total LMP
    "reserve_tightness": 0.20,  # load / available-capacity proxy
    "event_flag": 0.15,  # named-event binary with ±5-day decay
}

# ── Portfolio construction ────────────────────────────────────────────────────
PORTFOLIO_LONG_N = 5  # number of names to go long
PORTFOLIO_SHORT_N = 5  # number of names to go short
REBALANCE_FREQ = "ME"  # pandas offset alias: month-end

# Replace short book with XLU sector-ETF hedge (default on)
# Set to False to restore original long/short individual-name construction
XLU_HEDGE: bool = True

# ── Conditional beta estimation ───────────────────────────────────────────────
# Event window (trading days) around each stress event for the event study
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10

# Minimum number of stress observations required to estimate a reliable beta
MIN_STRESS_OBS = 10  # SPP has ~12 stress days/year; PJM requires PJM_API_KEY (pjm.com/api)

# ── Factor neutralization ─────────────────────────────────────────────────────
WINSOR_LIMITS = (0.025, 0.975)  # clip factor scores at these percentiles
