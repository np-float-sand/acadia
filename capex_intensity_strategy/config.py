"""Frozen configuration for the capex-intensity strategy.

Every parameter is frozen from the 2026-09-21 discovery run
(docs/capex-intensity-rule-results.md) and is NEVER revised from results.
Simple-return convention throughout; metrics use the repo-standard RF = 4%.

The rule in one line: inside a mechanically-defined power/electrification
universe, go long the companies that SELL INTO the capital cycle (low capital
spending relative to revenue) and short the ones that ARE the capital cycle.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RF: float = 0.04
ANN: int = 252
START_DEFAULT: str = "2019-01-01"      # first month with >=4 filed quarters for most names

# -- universe -------------------------------------------------------------
# A name is a candidate if its SEC filer SIC code is in sic_codes.csv, OR its
# prior-year annual report matches the fuel-cycle lexicon below. Both are
# point-in-time: SIC is assigned at filing, and the lexicon is queried per year.
LIQUIDITY_MIN_DOLLAR_VOLUME: float = 10e6   # trailing 1-yr median, evaluated monthly
LIQUIDITY_WINDOW_DAYS: int = 252
LIQUIDITY_MIN_PERIODS: int = 120
MIN_PRICE_HISTORY_DAYS: int = 400

# Filer-authored vocabulary that identifies nuclear-fuel-cycle businesses whose
# SIC code is generic mining (1040/1090/1400) and so misses the SIC screen.
FUEL_CYCLE_LEXICON: tuple[str, ...] = (
    '"uranium concentrates"',
    '"nuclear fuel cycle"',
    '"uranium enrichment"',
    '"yellowcake"',
    '"enriched uranium"',
)
LEXICON_FORMS: str = "10-K,40-F,20-F"
LEXICON_LOOKBACK_YEARS: int = 2        # filings from [Y-2, Y-1] define the year-Y universe

# -- score ----------------------------------------------------------------
# Point-in-time TTM capital expenditure / TTM revenue, built from EDGAR
# companyfacts using each fact's `filed` date (never its period end).
TTM_QUARTERS: int = 4
QUARTER_MIN_DAYS: int = 60             # a discrete quarter spans 60..110 days
QUARTER_MAX_DAYS: int = 110            # was 200, which admitted 6-month YTD facts
# Cash-flow facts in a 10-Q are year-to-date (3/6/9 months) and in a 10-K annual,
# so capex is fetched at every span up to a full year and differenced back into
# discrete quarters by `fundamentals.derive_quarters`.
FLOW_MAX_SPAN_DAYS: int = 400

# -- portfolio ------------------------------------------------------------
QUANTILE: float = 0.25                 # long lightest 25%, short heaviest 25%
SHORT_WEIGHT: float = 0.50             # short leg sized at half the long leg
REBALANCE_DAYS: int = 21               # monthly
COST_PER_TURNOVER: float = 0.0010      # 10bp each way on traded notional
VOL_TARGET: float = 0.20
VOL_LOOKBACK_DAYS: int = 42
MAX_LEVERAGE: float = 2.0

BENCHMARKS: tuple[str, ...] = ("SPY", "XLU", "XLI")

# The "could I have bought this cheaper?" set -- what a person would actually buy
# instead, with published expense ratios. VOLT listed 2024-12-05, so it is only
# comparable over a short window and is charted separately.
COMPARISON_ETFS: tuple[str, ...] = ("SPY", "XLI", "XLU", "GRID", "PAVE", "VOLT")

# -- reporting windows ----------------------------------------------------
WINDOWS: dict[str, tuple[str, str]] = {
    "2019-2022 (fit)": ("2019-01-01", "2022-12-31"),
    "2023-2026 (oos)": ("2023-01-01", "2026-12-31"),
    "2025-2026 (oos)": ("2025-01-01", "2026-12-31"),
    "full": ("2019-01-01", "2026-12-31"),
}

# SEC fair-access policy REQUIRES a contact address in the User-Agent; requests
# without one are answered with HTTP 403. Matches the convention already used in
# research/*/edgar*.py. Override with SEC_CONTACT_EMAIL if running as someone else.
SEC_CONTACT: str = os.environ.get("SEC_CONTACT_EMAIL", "sand.gh1902@gmail.com")
SEC_USER_AGENT: str = f"acadia-research {SEC_CONTACT}"
SEC_RATE_LIMIT_SLEEP: float = 0.12

# -- fact-quality guards (added 2026-09-23) -------------------------------
# A trailing-four-quarter sum must actually span about a year, and must be
# current as of the observation date. Without these, a filer that stopped
# tagging an item (AEP's capex facts stop in 2021) was scored in 2026 off its
# 2021 numbers, and four same-quarter facts from four years counted as a TTM.
TTM_SPAN_MIN_DAYS: int = 330
TTM_SPAN_MAX_DAYS: int = 400
MAX_FACT_STALENESS_DAYS: int = 270     # ~2 quarters of reporting lag, then out
