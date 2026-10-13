"""Frozen configuration for the electrification strategy v1.

Every parameter here is frozen from the 2026-09-06 probe
(docs/electrification-short-leg-insurance-probe-results.md) and is NEVER revised
from results. Simple-return convention throughout.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RF: float = 0.04
ANN: int = 252
START_DEFAULT: str = "2017-06-01"

# -- basket construction --------------------------------------------------
REBALANCE_LAG_DAYS: int = 42
MAX_SINGLE_NAME_WEIGHT: float = 0.25
LISTING_MIN_DAYS: int = 252          # a name enters only after this many trading days

MARQUEE_UNIVERSE: list[str] = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
SUB_INDUSTRIES: set[str] = {
    "electrical_equipment", "heavy_electrical", "electrical_e&c",
    "dc_thermal", "grid_scale_storage", "data_center_reit",
}

# -- vol target (reuses grid_equipment_basket.overlay.vol_target_scalar) --
VOL_TARGET: float = 0.20
VOL_LOOKBACK: int = 21
VOL_MAX_LEVERAGE: float = 1.5

# -- valuation / extension de-risk overlay -------------------------------
VAL_MA_DAYS: int = 200
VAL_RS_DAYS: int = 252
VAL_EXT_LO: float = 0.15
VAL_EXT_HI: float = 0.35
VAL_RS_LO: float = 0.25
VAL_RS_HI: float = 0.50
VAL_MULT_MID: float = 0.8
VAL_MULT_LOW: float = 0.6
VAL_PLATEAU_SCALES: tuple[float, ...] = (0.8, 1.0, 1.2)

# -- hedge overlay -----------------------------------------------------
HEDGE_SLEEVE_TICKERS: tuple[str, ...] = ("GLD", "IEF")
HEDGE_SLEEVE_WEIGHT: float = 0.15
HEDGE_SHORT_TICKERS: tuple[str, ...] = ("DLR", "EQIX")
HEDGE_SHORT_WEIGHT: float = 0.25
HEDGE_SHORT_SIGNAL: str = "DFII10"
HEDGE_SHORT_LOOKBACK_DAYS: int = 126
HEDGE_SHORT_LAG_DAYS: int = 5

# -- solar-rotation overlay (open item #17: 2026-09-06 probe, `screen +val+sleeve`
# baseline; frozen from `scratchpad/utility_rotation.py` + `docs/electrification-
# strategy-v1-results.md` "rotate INTO solar when rates fall" -- full Sharpe
# 0.69->0.77, CAGR 14.4->16.8%, MaxDD -24%->-26.8%, +7% in the feared solar-squeeze
# months). Opposite sign of the rejected DER-sleeve short. Reuses the same 6-month
# real-yield-change / 5-day-lag signal as the hedge short above (HEDGE_SHORT_
# LOOKBACK_DAYS / LAG_DAYS / SIGNAL), just triggered on FALLING rather than rising.
SOLAR_ROTATION_TICKERS: tuple[str, ...] = ("ENPH", "SEDG", "CHPT", "RUN", "BLNK", "STEM")
SOLAR_ROTATION_WEIGHT: float = 0.10

# -- short interest diagnostic (spec 3.1) --------------------------------
SHORT_INTEREST_CACHE_FILE: str = "short_interest_history.parquet"

# -- evaluation ------------------------------------------------------
BENCHMARKS: list[str] = ["VOLT", "PAVE", "GRID", "SPY", "XLI"]
FEARED_PROXY: str = "TAN"

# (peak-search window), trough-end -- same anchors as the probe
EPISODES: dict[str, tuple[tuple[str, str], str]] = {
    "COVID 20": (("2020-02-01", "2020-02-25"), "2020-04-30"),
    "Rate shock 22": (("2021-11-01", "2022-01-31"), "2022-12-31"),
    "DeepSeek 25": (("2024-11-01", "2025-01-15"), "2025-05-31"),
    "Tariff Apr-25": (("2025-03-15", "2025-04-03"), "2025-04-30"),
    "Selloff 26": (("2026-05-01", "2026-06-20"), "2026-08-31"),
}
CALM_WINDOWS: dict[str, tuple[str, str]] = {
    "pre-COVID 17-19": ("2017-06-01", "2019-12-31"),
    "AI-bull 23-24": ("2023-01-01", "2024-10-31"),
}
SUBWINDOWS: dict[str, tuple[str, str]] = {
    "2019-22": ("2019-01-01", "2022-12-31"),
    "2023-26": ("2023-01-01", "2026-08-31"),
}

# -- pre-registered winner rule (spec 6.3) -----------------------------
WINNER_MAXDD_MAX: float = -0.27
WINNER_SUBWINDOW_SHARPE_MIN: float = 0.4
WINNER_FEARED_PNL_MIN: float = -0.02
WINNER_DRAG_MAX: float = 0.06

# -- mechanical `screen_auto` construction (2026-09-23) -------------------
# Point-in-time replacements for the hand-recorded seed list and its three
# judgment columns. Every term is an EXACT phrase -- unquoted, EDGAR full-text
# search matches any word in the string.
AUTO_LOOKBACK_YEARS: int = 2           # universe-year Y reads filings from [Y-2, Y-1]
AUTO_MIN_HITS: int = 1                 # at least one filing in the window uses the term
AUTO_POLICY_MIN_HITS: int = 1          # filings in the window that must flag policy
AUTO_CAPEX_QUANTILE: float = 0.5       # "sells into the buildout" = light half
AUTO_FORMS: str = "10-K"

# Who the filer sells to, in its own words. Replaces `customer_institutional`.
# These are NOT exclusive to suppliers -- a utility writes "data center" about its
# customers -- which is why this filter only ever runs after the capex-light cut.
AUTO_CUSTOMER_TERMS: tuple[str, ...] = (
    '"data center"',
    '"data centers"',
    '"substation"',
    '"switchgear"',
    '"transmission and distribution"',
    '"grid modernization"',
    '"electric utilities"',
)
# Subsidy vocabulary. Replaces `low_policy_dependence` (a hit FLAGS the name).
AUTO_POLICY_TERMS: tuple[str, ...] = (
    '"investment tax credit"',
    '"production tax credit"',
    '"net energy metering"',
    '"advanced manufacturing production credit"',
)

# -- vocabulary clean-up (2026-10-12; docs/survivorship-audit-results.md) ----------------
# "data center(s)" is kept as a customer term, but a full-text hit counts only when the
# filing's Item 1 (Business) uses it for a market, not for the filer's own IT
# ("our data centers", cybersecurity boilerplate). Spurious hits are listed in
# data/vocabulary_hits_dc_excluded.csv, a build artifact of `build_vocab_filter`.
AUTO_DC_TERMS: tuple[str, ...] = ('"data center"', '"data centers"')
# Gas pipelines and gas distributors pass rule 2 on gas "transmission and distribution"
# wording. Their hits are dropped, so they can't enter the short pool (the long book
# already excludes the POWER group). Electric-and-gas combination utilities (4931) stay.
RULE2_EXCLUDED_SICS: tuple[int, ...] = (4922, 4923, 4924)
