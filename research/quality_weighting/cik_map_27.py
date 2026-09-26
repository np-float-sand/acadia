"""Ticker -> CIK map for the `screen` 27-name electrification-supplier universe.

Copied verbatim from `research/backlog_pricing_power/cik_map.py` (built this session
by a sibling agent for the backlog-pricing-power probe, pulled from SEC's
company_tickers.json). That module lives in the *main* repo working tree as an
untracked file and is not visible from this worktree checkout (git worktrees have
separate working directories; untracked files aren't shared) -- so the static
ticker->CIK data is copied here rather than imported cross-worktree at runtime.
Not production code. Throwaway research script.
"""
from __future__ import annotations

SCREEN_27 = [
    "AEIS", "AME", "APG", "ATKR", "AYI", "CARR", "EME", "EMR", "ENS", "ETN",
    "FIX", "GEV", "HUBB", "IESC", "JCI", "MOD", "MTZ", "MYRG", "NVT", "POWL",
    "PRIM", "PWR", "RRX", "SPXC", "TT", "VRT", "WCC",
]

CIK_BY_TICKER: dict[str, str] = {
    "AEIS": "927003", "AME": "1037868", "APG": "1796209", "ATKR": "1666138",
    "AYI": "1144215", "CARR": "1783180", "EME": "105634", "EMR": "32604",
    "ENS": "1289308", "ETN": "1551182", "FIX": "1035983", "GEV": "1996810",
    "HUBB": "48898", "IESC": "1048268", "JCI": "833444", "MOD": "67347",
    "MTZ": "15615", "MYRG": "700923", "NVT": "1720635", "POWL": "80420",
    "PRIM": "1361538", "PWR": "1050915", "RRX": "82811", "SPXC": "88205",
    "TT": "1466258", "VRT": "1674101", "WCC": "929008",
}

assert set(SCREEN_27) == set(CIK_BY_TICKER)
