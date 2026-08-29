"""Central configuration for the grid-equipment-suppliers thematic basket.

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Provisional list — Task 6 (candidate_research.md) replaces this with the
# verified include-set. Inclusion is decided on 10-K business description only.
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
