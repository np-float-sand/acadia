"""
Ticker universe + fixed weights for the satellite-sleeve diversification test
(2026-09-18).

This is NOT a signal test. `docs/electricity-price-volatility-miner-results.md`
already ran (and failed) the predictive-signal version of this question on the
same names. This probe asks a purely mechanical question instead: does holding
these names as a plain, equal-weight, non-timed satellite sleeve at a fixed % of
NAV improve the shipped electrification book's combined Sharpe/Calmar/MaxDD --
via low correlation, independent of whether the sleeve has any alpha of its own.

Sleeve membership (all fixed, no re-selection):
  miners      -- CIFR, WULF, MARA, RIOT, IREN, BTDR, CLSK (all 7; CLSK was
                 excluded from the predictive probe only because no ERCOT/PJM
                 hub was a defensible proxy for its fleet -- that exclusion
                 reason is irrelevant to a plain stock basket).
  generators  -- VST, NRG, CEG, TLN.
  combined    -- equal-weight union of both (11 names).

Data-quality note -- SPAC / reverse-merger ticker history (see data.py):
CIFR, WULF and BTDR all traded on their current ticker for months-to-years
*before* the crypto-mining business they are held for today existed --  a SPAC
shell sitting near its ~$10 trust NAV (CIFR, BTDR) or an unrelated reverse-
merger target (WULF). Using yfinance's raw first-valid-date for these three
would silently splice near-zero-volatility, business-irrelevant data onto the
front of the sleeve's return series. EFFECTIVE_START_OVERRIDES below pins each
of those three to the date its price series departs the shell/pre-merger
regime (verified directly against the fetched price history -- see the
one-line evidence comment on each). IREN's raw first-valid date needed no
correction (real, moving prices from day one).
"""
from __future__ import annotations

MINERS = ["CIFR", "WULF", "MARA", "RIOT", "IREN", "BTDR", "CLSK"]
GENERATORS = ["VST", "NRG", "CEG", "TLN"]
COMBINED = sorted(set(MINERS) | set(GENERATORS))

SLEEVES = {
    "miners": MINERS,
    "generators": GENERATORS,
    "combined": COMBINED,
}

WEIGHTS = (0.05, 0.10, 0.15)

# Base-book cells tested against: primary first (used everywhere else in this
# project, incl. the just-finished crypto-miner signal probe), screen_reit
# second (docs/electrification-reit-breadth-probe-results.md: strictly
# dominates `screen +val+sleeve` in this lineage).
BASE_CELLS = [("screen", "+val+sleeve"), ("screen_reit", "+val+sleeve")]

RECENT_WINDOW_START = "2023-01-01"

# Evidence (fetched via grid_equipment_basket.data.prices.fetch_prices, full
# history from 2005/2010/2012):
#   CIFR: flat ~$9.9-10.0 SPAC-trust band through 2021-08-19; jumps to $11.34
#         on 2021-08-26 (Cipher Mining business-combination close) and moves
#         freely thereafter.
#   WULF: flat/quasi-flat ~$8-9.6 band through 2021-06-24; jumps to $15.82 on
#         2021-06-25 and moves freely thereafter (reverse-merger uplisting).
#   BTDR: flat ~$10.7-10.85 SPAC-trust band through 2023-04-13; resets to
#         $7.03 on 2023-04-14 (Bitdeer business-combination close) and moves
#         freely thereafter.
# IREN's raw first_valid_index (2021-11-17) already shows real, moving prices
# from day one -- no override needed. MARA/RIOT/CLSK also carry older,
# business-irrelevant history (pre-2017/2018 pivots to crypto mining), but
# none of the three is the sleeve's binding constraint (see run_probe.py) so
# it does not affect the reported windows.
EFFECTIVE_START_OVERRIDES = {
    "CIFR": "2021-08-26",
    "WULF": "2021-06-25",
    "BTDR": "2023-04-14",
}
