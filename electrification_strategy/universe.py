"""Universe constructions for the electrification strategy.

Three constructions, all quarterly-reconstituted with a 42-day reporting lag and
a 252-trading-day listing gate, all equal-weight with a 25% single-name cap:

  marquee  -- grid_equipment_basket.config.UNIVERSE verbatim (reference)
  frozen   -- our quality screen: seed pool & sub-industry map & (in >=1 of 5 ETFs)
              & profitable_2026
  thematic -- thematic-ETF consensus: in >=2 of {VOLT,ELFY,ZAP,GRID}, filtered to
              the sub-industry map + listing gate. No profitability screen.

CAVEAT: seed membership, the ETF snapshot, and profitable_2026 are 2026-vintage,
back-cast. Only the listing-date gate is genuinely point-in-time. True
point-in-time holdings/GICS + time-varying profitability are a pre-live gate.
"""
from __future__ import annotations

import pandas as pd

from grid_equipment_basket.basket import apply_cap

from electrification_strategy import config

THEMATIC_ETFS = ["VOLT", "ELFY", "ZAP", "GRID"]
_MEMBERSHIP_COLS = ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]


def load_seed() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "universe_seed.csv", comment="#")
    df["profitable_2026"] = (
        df["profitable_2026"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False}).astype(bool)
    )
    df = df.set_index("ticker")
    bad = set(df["sub_industry"]) - config.SUB_INDUSTRIES
    if bad:
        raise ValueError(f"universe_seed.csv has unknown sub_industry values: {sorted(bad)}")
    return df


def load_membership() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "etf_membership_2026.csv", comment="#").set_index("ticker")
    for col in _MEMBERSHIP_COLS:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False}).astype(bool)
        )
    return df[_MEMBERSHIP_COLS]


def _listed_eligible(prices: pd.DataFrame, asof: pd.Timestamp) -> set[str]:
    upto = prices.loc[:asof]
    counts = upto.notna().sum()
    return set(counts[counts >= config.LISTING_MIN_DAYS].index)


def members(construction: str, asof: pd.Timestamp, prices: pd.DataFrame) -> list[str]:
    listed = _listed_eligible(prices, asof)

    if construction == "marquee":
        return sorted(set(config.MARQUEE_UNIVERSE) & listed)

    seed = load_seed()
    memb = load_membership()
    in_map = set(seed.index)                       # sub_industry validated in load_seed
    if construction == "frozen":
        any_etf = set(memb.index[memb[_MEMBERSHIP_COLS].any(axis=1)])
        profitable = set(seed.index[seed["profitable_2026"]])
        pool = in_map & any_etf & profitable
    elif construction == "thematic":
        ge2 = set(memb.index[memb[THEMATIC_ETFS].sum(axis=1) >= 2])
        pool = in_map & ge2
    else:
        raise ValueError(f"unknown construction {construction!r}")
    return sorted(pool & listed)


def target_fn(construction: str, prices: pd.DataFrame):
    def _fn(available, asof):
        elig = [t for t in members(construction, pd.Timestamp(asof), prices) if t in set(available)]
        if not elig:
            return pd.Series(dtype=float)
        raw = pd.Series(1.0 / len(elig), index=sorted(elig))
        return apply_cap(raw, config.MAX_SINGLE_NAME_WEIGHT)

    return _fn
