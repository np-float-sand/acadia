from __future__ import annotations

import math

PUDL_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/"

FERC_TABLES: dict[str, str] = {
    "plant_in_service": "core_ferc1__yearly_plant_in_service_sched204",
    "dep_by_function":  "core_ferc1__yearly_depreciation_by_function_sched219",
    "plant_summary":    "core_ferc1__yearly_utility_plant_summary_sched200",
    "utility_xwalk":    "core_pudl__assn_ferc1_pudl_utilities",
}

# Frozen seed universe (finalised in data/utility_map.py). US-listed regulated
# electric + electric-heavy multi-utilities that file FERC Form 1.
UNIVERSE_SEED: tuple[str, ...] = (
    "AEP", "AEE", "AES", "AVA", "BKH", "CMS", "CNP", "D", "DTE", "DUK",
    "ED", "EIX", "ES", "ETR", "EVRG", "EXC", "FE", "HE", "IDA", "LNT",
    "MGEE", "NEE", "NWE", "OGE", "OTTR", "PCG", "PEG", "PNW", "POR", "PPL",
    "SO", "TXNM", "WEC", "XEL",
)

PRIMARY_START, PRIMARY_END = "2011-01-01", "2025-12-31"
PRE_THESIS_START, PRE_THESIS_END = "2003-01-01", "2010-12-31"

REBALANCE_MONTH = 5
SIGNAL_CAGR_YEARS = 3
MIN_HISTORY_YEARS = 4          # need Y-3..Y inclusive for one 3-yr growth figure
QUINTILES = 5
WINSOR_SIGMA = 2.5
STRUCTURAL_BREAK_LOG = math.log(2.0)
BETA_WINDOW = 252
BETA_CLIP = (0.5, 2.0)

GATE: dict[str, float] = {
    "min_mean_ic": 0.03,
    "min_ic_t": 2.0,
    "min_ls_sharpe": 0.40,
    "max_quintile_inversions": 1,
    "min_positive_years_frac": 0.60,
    "min_alpha_ann": 0.03,
    "min_alpha_t": 2.0,
    "max_additivity_r2": 0.60,
    "pre_thesis_min_t": -2.0,
}
