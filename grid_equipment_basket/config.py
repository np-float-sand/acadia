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
VC_STALENESS_MAX_DAYS: int = 200     # latest disclosed *quarterly* (xbrl_rpo) row must be this fresh vs asof
# Spec §4.2: annual-only disclosers (HUBB, NVT — yearly `nongaap_backlog_total`) carry the most
# recent annual figure forward between updates, so their backlog rows get a wider staleness bound.
VC_STALENESS_MAX_DAYS_ANNUAL: int = 400

# Construction 2 + hedges.
PAIR_GROSS: float = 1.00             # spec §6/§10: 100% gross long / 100% gross short (dollar-neutral).
                                     # Implicit — each leg is renormalized to sum to 1 in simulate_pair —
                                     # kept here for spec traceability.
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

# ── Layer-1 risk overlay (2026-08-31) — trend gate + vol target ────────────
# Frozen, documented, plateau-justified (MA 50-150 x target-vol 15-25% all give
# Sharpe 1.7-2.0 in-sample; not a knife-edge). See overlay.py for the honesty
# caveat and docs/handoff_2026-08-29-grid-equipment-basket.md for provenance.
OVERLAY_MA_DAYS: int = 100
OVERLAY_VOL_LOOKBACK: int = 20
OVERLAY_TARGET_VOL: float = 0.20
OVERLAY_MAX_LEVERAGE: float = 1.5

# ── Layer-2 grid-congestion regime signal (2026-08-31) ─────────────────────
# Replaces layer-1's price trend gate with a physical "is the grid bottleneck
# tightening or easing" read:  exposure = regime_multiplier x vol_target_scalar
#
# Gate outcome: the pre-registered ladder was run and NO rung strictly passed
# -- every rung beats layer-1-only on Sharpe but not on primary-window Calmar
# (2.16 vs 2.32). ADOPTED ANYWAY as the recommended overlay by PM decision
# (2026-08-31): the rationale is generalisation (rung 1 is Sharpe-positive on
# BOTH windows -- 0.61 prior vs layer-1's 0.20 -- fixing layer 1's 2020-2022
# collapse) plus the physical-grid differentiator, accepting a ~5pp deeper
# primary-window drawdown. The shipped config is ladder rung 1 (discrete,
# PJM-4 DC-heavy zones, congestion only) -- see grid_regime.shipped_config().
# Full write-up: docs/grid-regime-layer2-results.md.
REGIME_ENABLED: bool = True        # layer 2 (rung 1) is the recommended overlay
REGIME_ZONES_CORE: list[str] = ["DOM", "AEP", "COMED", "PPL"]   # DC-heavy PJM zones
REGIME_ZONES_WIDE: list[str] = ["DOM", "AEP", "COMED", "PPL", "PSEG", "ATSI"]
REGIME_DC_ZONES: list[str] = ["DOM", "AEP", "COMED", "PPL"]     # data-center set for the option-A relative signal
REGIME_ZSCORE_WINDOW: int = 756      # ~3y trailing window for the per-zone z-scores
REGIME_ZSCORE_MINP: int = 252       # ~1y minimum obs before the signal is active
REGIME_ZSCORE_WINSOR: float = 3.0   # clip z-scores to +/- this
REGIME_MONTH_LOOKBACK: int = 20     # trailing trading days averaged at each month-end
REGIME_THRESH: float = 0.5         # discrete-mode: |composite| >= this leaves neutral
REGIME_HI: float = 1.25            # discrete-mode multiplier when tightening
REGIME_LO: float = 0.6            # discrete-mode multiplier when easing
REGIME_K: float = 0.35            # continuous-mode slope: clip(1 + k*composite, 0.5, 1.5)
REGIME_W_CONG: float = 1.0         # rung-1 sub-signal weights (congestion only)
REGIME_W_RESERVE: float = 0.0

# Signal is z-scored over this full span (long warm-up); basket returns are
# evaluated only on the two windows below (which is why the prior test is
# effectively 2021-2022 -- see spec s11).
REGIME_SIGNAL_START: str = "2018-01-01"
REGIME_PRIMARY_WINDOW: tuple[str, str] = ("2023-01-01", "2025-12-31")
REGIME_PRIOR_WINDOW: tuple[str, str] = ("2020-01-01", "2022-12-31")   # matches layer-1's panel

# The frozen pre-registered ladder (spec s5). Run top to bottom; stop at the
# first rung whose verdict is "PASS" (gates + plateau, non-marginal). Each rung
# changes exactly one thing from the one above; `neighbours` are the plateau
# probes (must also clear G1 & G2). NOTHING here is revised from results.
REGIME_LADDER: list[dict] = [
    {"name": "1: discrete, PJM-4, congestion-only",
     "mode": "discrete", "zones": REGIME_ZONES_CORE, "w_cong": 1.0, "w_reserve": 0.0,
     "zone_weight": "equal", "thresh": 0.5,
     "neighbours": [{"thresh": 0.25}, {"thresh": 0.75},
                    {"zones": ["DOM", "AEP", "COMED"]},
                    {"zones": ["DOM", "AEP", "COMED", "PPL", "PSEG"]}]},
    {"name": "2: + reserve-tightness",
     "mode": "discrete", "zones": REGIME_ZONES_CORE, "w_cong": 0.7, "w_reserve": 0.3,
     "zone_weight": "equal", "thresh": 0.5,
     "neighbours": [{"thresh": 0.25}, {"thresh": 0.75},
                    {"w_cong": 0.55, "w_reserve": 0.45}, {"w_cong": 0.85, "w_reserve": 0.15}]},
    {"name": "3: continuous multiplier",
     "mode": "continuous", "zones": REGIME_ZONES_CORE, "w_cong": 0.7, "w_reserve": 0.3,
     "zone_weight": "equal", "k": 0.35,
     "neighbours": [{"k": 0.25}, {"k": 0.45}]},
    {"name": "4: widen zones to 6",
     "mode": "continuous", "zones": REGIME_ZONES_WIDE, "w_cong": 0.7, "w_reserve": 0.3,
     "zone_weight": "equal", "k": 0.35,
     "neighbours": [{"k": 0.25}, {"k": 0.45},
                    {"zones": ["DOM", "AEP", "COMED", "PPL", "PSEG"]}]},
    {"name": "5: load-weighted zones",
     "mode": "continuous", "zones": REGIME_ZONES_WIDE, "w_cong": 0.7, "w_reserve": 0.3,
     "zone_weight": "load", "k": 0.35,
     "neighbours": [{"k": 0.25}, {"k": 0.45}]},
    {"name": "6: + ERCOT West spread proxy",
     "mode": "continuous", "zones": REGIME_ZONES_WIDE, "w_cong": 0.7, "w_reserve": 0.3,
     "zone_weight": "load", "k": 0.35, "ercot_west": True,
     "neighbours": [{"k": 0.25}, {"k": 0.45}]},
    {"name": "7: + RT/DA spread sub-signal",
     "deferred": True,
     "reason": "needs a fetch_lmp_rt() + a live gridstatus RT-LMP availability probe "
               "over 2018-2025 (spec s7); not wired this pass."},
]
