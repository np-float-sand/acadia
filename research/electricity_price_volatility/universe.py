"""
Universe + pre-registered decisions for the electricity-price-volatility /
crypto-miner probe (2026-09-16).

Source: docs/electricity-price-volatility-miner-preregistration.md (Decisions 1-7).
Do not change hub lists, name lists, or sign predictions after looking at results --
that is exactly the data-snooping trap Decision 2 exists to prevent.
"""

# ---------------------------------------------------------------------------
# Decision 1 -- hub universe (19 hubs, all cached at native hourly resolution)
# ---------------------------------------------------------------------------

ERCOT_HUBS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
]

PJM_HUBS = [
    "AEP GEN HUB", "AEP-DAYTON HUB", "ATSI GEN HUB", "CHICAGO GEN HUB", "CHICAGO HUB",
    "DOMINION HUB", "EASTERN HUB", "N ILLINOIS HUB", "NEW JERSEY HUB", "OHIO HUB",
    "WEST INT HUB", "WESTERN HUB",
]

# (iso, hub_name) tuples -- the full 19-hub scan universe used for hub SELECTION
# (train window only, Decision 2). Every name is scanned against all 19 regardless
# of which ISO its disclosed fleet actually sits in -- geographic plausibility is
# judged AFTER selection (Decision 2/3/7), never used to restrict the scan itself.
ALL_HUBS = [("ERCOT", h) for h in ERCOT_HUBS] + [("PJM", h) for h in PJM_HUBS]

assert len(ALL_HUBS) == 19

# ---------------------------------------------------------------------------
# Decision 3 -- miner universe
# ---------------------------------------------------------------------------

# Primary (named in the source paper summary) -- headline-verdict-bearing.
MINERS_PRIMARY = ["CIFR", "WULF"]

# Widened for robustness -- reported separately, cannot rescue a primary failure.
MINERS_WIDENED = ["MARA", "RIOT", "IREN", "BTDR"]

MINERS_ALL = MINERS_PRIMARY + MINERS_WIDENED

# CLSK explicitly excluded (Decision 3 / pre-reg item 7): GA/MS/TN/WY fleet, no
# ERCOT/PJM hub is a defensible proxy. Do not test it.

# Disclosed-fleet plausibility notes, fixed BEFORE seeing any hub-selection result.
MINER_FLEET_NOTES = {
    "CIFR": "100% ERCOT/TX (Odessa/Alborz, Bear, Chief, Barber Lake). Only an ERCOT hub is plausible.",
    "WULF": "NY-heavy (Lake Mariner, NYISO) + ~20% PA (Nautilus, PJM). NYISO is NOT in this "
            "repo's hub universe -- any PJM pick is a KNOWN-WEAK proxy (only ~20% of "
            "disclosed capacity), stated up front per the pre-registration.",
    "MARA": "TX-heavy (~55%) + ND/OH/NE. ERCOT hub plausible; a PJM pick would be a flag "
            "(OH exposure is a minority of capacity, not obviously PJM-zoned).",
    "RIOT": "TX (~85%) + KY (~15%, PJM territory). ERCOT hub plausible; PJM pick weakly "
            "plausible via the KY minority stake.",
    "IREN": "100% TX/ERCOT (per electricity_earnings_exposure/universe.py exposure vector). "
            "Only an ERCOT hub is plausible.",
    "BTDR": "TX (~60%) + WA (~40%, neither ERCOT nor PJM -- WECC territory, not covered here). "
            "ERCOT hub plausible for the TX majority; no hub covers the WA share.",
}

# ---------------------------------------------------------------------------
# Decision 4 -- generator arm (parallel mechanism, opposite sign)
# ---------------------------------------------------------------------------

GENERATORS = ["VST", "NRG", "CEG", "TLN"]

# Disclosed fleet mix by ISO (GW, from research/electricity_earnings_exposure/probe_a.py
# FLEET table -- the same hand-coded fleet used in Probe A, reused here only for the
# plausibility narrative, not the regression itself).
GENERATOR_FLEET_NOTES = {
    "VST": "Mixed: ~26.4 GW ERCOT (gas/nuclear/coal/renew) vs ~16 GW PJM (gas/coal, + "
           "Energy Harbor nuclear from 2024-04). ~62% ERCOT / 38% PJM -- both an ERCOT "
           "and a PJM hub pick are plausible.",
    "NRG": "ERCOT-heavy: ~13 GW ERCOT (gas/coal) vs ~3 GW PJM (gas). ~81% ERCOT -- an "
           "ERCOT hub pick is plausible; a PJM pick is a mild (not fatal) flag.",
    "CEG": "~100% PJM: ~25.5 GW (17.5 nuclear + 6 gas + 2 renew), zero ERCOT rows in the "
           "Probe A fleet table. An ERCOT hub pick would be a genuine red flag.",
    "TLN": "PJM-heavy: ~8.2 GW PJM (nuclear/gas/coal) vs ~1.0 GW ERCOT (gas). ~89% PJM -- "
           "a PJM hub pick is plausible; an ERCOT pick is a mild flag.",
}

# ---------------------------------------------------------------------------
# Decision 2 / 4 -- sign predictions (locked before any output is seen)
# ---------------------------------------------------------------------------

# Miners: lagged hub price-return VARIANCE -> LOWER subsequent equity variance
# (curtailment-hedging mechanism: miners power down / resell pre-purchased power
# when wholesale prices spike, stabilizing cash flow / reducing stock variance).
MINER_PREDICTED_SIGN = -1

# Generators: lagged hub price-return VARIANCE -> HIGHER subsequent equity variance
# (more volatile realised wholesale revenue -> more volatile equity returns).
GENERATOR_PREDICTED_SIGN = +1

# ---------------------------------------------------------------------------
# Decision 5 -- controls
# ---------------------------------------------------------------------------

BTC_TICKER = "BTC-USD"      # miner-arm mandatory control (Probe B confound guard)
SPY_TICKER = "SPY"          # generator-arm control

# ---------------------------------------------------------------------------
# Windows (Decision 2 / 5)
# ---------------------------------------------------------------------------

TRAIN_START, TRAIN_END = "2023-01-01", "2024-12-31"   # hub SELECTION only
TEST_START = "2025-01-01"                              # gate-bearing
YTD_START = "2026-01-01"                               # fragility check, not gating

# ---------------------------------------------------------------------------
# Decision 6 -- gate
# ---------------------------------------------------------------------------

GATE_ABS_T = 2.0
