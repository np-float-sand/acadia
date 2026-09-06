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
    "dc_thermal", "grid_scale_storage",
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
