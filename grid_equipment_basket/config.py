"""Central configuration for the grid-equipment-suppliers thematic basket.

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Verified include-set — see candidate_research.md. Inclusion is decided on the
# latest-10-K business description only, never on historical returns. All nine
# spec §2 candidates passed the business test (PRIM is borderline but retained;
# FLNC is retained despite weak returns). Foreign-listed names (ABB, Siemens
# Energy, Prysmian, Nexans) are logged there and excluded — US access is thin
# OTC ADRs only. GEV has price history only from 2024-04-02 and enters the
# equal-weight basket on its first available date (spec §4).
UNIVERSE: list[str] = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]

BENCHMARKS: list[str] = ["XLI", "SPY", "XLU", "GRID", "PAVE"]
PRIMARY_BENCHMARK: str = "XLI"
HONESTY_BENCHMARKS: list[str] = ["GRID", "PAVE"]

PRIMARY_START: str = "2023-01-01"
PRIOR_REGIME_START: str = "2020-01-01"
PRIOR_REGIME_END: str = "2022-12-31"

RISK_FREE_RATE: float = 0.04
ANN_FACTOR: int = 252

REBALANCE_LAG_DAYS: int = 42          # ~6 weeks after each calendar quarter-end
MAX_SINGLE_NAME_WEIGHT: float = 0.25

# Step 2 (gated) — fixed, documented, not fitted.
BACKLOG_TILT_TOP: float = 1.25
BACKLOG_TILT_BOTTOM: float = 0.75

# ── Value-chain reframe (spec 2026-08-29) ──────────────────────────────────
# Frozen before any backtest, from 10-K business descriptions only. Never revised from results.
BUCKET_MAKERS: list[str] = ["ETN", "HUBB", "GEV", "VRT", "NVT"]
BUCKET_CONTRACTORS: list[str] = ["PWR", "MYRG", "PRIM", "FLNC"]

# Construction 1 — long-only tilt. Documented constants, not optimized.
VC_BASE_MAKER: float = 1.25
VC_BASE_CONTRACTOR: float = 0.75
VC_WITHIN_TOP: float = 1.10
VC_WITHIN_BOTTOM: float = 0.90

# Signal guards (mirror backlog_data's span / staleness guards).
VC_SIGNAL_MIN_QUARTERS: int = 8      # need 2 full TTM windows for a YoY margin change
VC_SPAN_MIN_DAYS: int = 300          # quarter_end[-1]..quarter_end[-5] lower bound
VC_SPAN_MAX_DAYS: int = 430          # ...upper bound (gappy series -> NaN)
VC_STALENESS_MAX_DAYS: int = 200     # latest disclosed quarter must be this fresh vs asof

# Construction 2 + hedges.
PAIR_OVERLAY_WEIGHT: float = 0.30
COND_SHORT_TICKER: str = "QQQ"
COND_SHORT_WEIGHT: float = 0.30
COND_SHORT_MA_DAYS: int = 100
COND_SHORT_VOL_DAYS: int = 20
COND_SHORT_VOL_REF_DAYS: int = 252

# Fixed drawdown episode for the Gate 2 comparison (spec §7.3): the DeepSeek scare.
DRAWDOWN_PEAK_WINDOW: tuple[str, str] = ("2024-07-01", "2024-12-31")
DRAWDOWN_TROUGH_END: str = "2025-06-30"

# Robustness pass (spec §7.1): makers to drop when --drop-winners is set.
VC_DROP_WINNERS: list[str] = ["VRT", "GEV"]
