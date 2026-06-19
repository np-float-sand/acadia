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
    "PJM": "DAY_AHEAD_HOURLY",
    "MISO": "DAY_AHEAD_HOURLY",
    "CAISO": "DAY_AHEAD_HOURLY",
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

# ── Stress event detection thresholds ────────────────────────────────────────
# LMP spike events: target genuine grid emergencies (99th pct, 3+ days)
STRESS_SPIKE_PCT = 0.97   # percentile of lmp_max to qualify as a spike day
STRESS_SPIKE_MIN_DAYS = 3  # min consecutive spike days to form an event
STRESS_SPIKE_MERGE_GAP = 1  # merge events separated by fewer days

# Congestion events: requires 30%+ congestion fraction for 10+ consecutive days.
# Grid-search optimised (2026-06-17): cong_threshold=0.30 + min_days=10 gives
# the best Sharpe — the duration filter does the heavy lifting against noise.
STRESS_CONG_THRESHOLD = 0.30  # congestion_frac must exceed this level
STRESS_CONG_MIN_DAYS = 10     # min consecutive high-congestion days

# ── Grid Stress Index weights (sum to 1) ──────────────────────────────────────
GSI_WEIGHTS = {
    "lmp_zscore": 0.40,  # normalized daily-max LMP vs rolling history
    "congestion_frac": 0.25,  # congestion component share of total LMP
    "reserve_tightness": 0.20,  # load / available-capacity proxy
    "event_flag": 0.15,  # named-event binary with ±5-day decay
}

# ── Portfolio construction ────────────────────────────────────────────────────
PORTFOLIO_LONG_N = 3  # number of names to go long  (grid-search optimised 2026-06-17)
PORTFOLIO_SHORT_N = 5  # number of names to go short
REBALANCE_FREQ = "ME"  # pandas offset alias: month-end

# Replace short book with XLU sector-ETF hedge (default on)
# Set to False to restore original long/short individual-name construction
XLU_HEDGE: bool = False  # True

# Enable interest coverage ratio as a factor component (default off — see CLAUDE.md)
USE_ICR: bool = False

# ── Conditional beta estimation ───────────────────────────────────────────────
# Event window (trading days) around each stress event for the event study
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10

# Minimum number of stress observations required to estimate a reliable beta
MIN_STRESS_OBS = (
    10  # SPP has ~12 stress days/year; PJM requires PJM_API_KEY (pjm.com/api)
)

# ── Factor neutralization ─────────────────────────────────────────────────────
WINSOR_LIMITS = (0.025, 0.975)  # clip factor scores at these percentiles
