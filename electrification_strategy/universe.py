"""Universe constructions for the electrification strategy.

Five constructions, all quarterly-reconstituted with a 42-day reporting lag and
a 252-trading-day listing gate, all equal-weight with a 25% single-name cap:

  marquee     -- grid_equipment_basket.config.UNIVERSE verbatim (reference)
  frozen      -- our quality screen: seed pool & sub-industry map & (in >=1 of 5 ETFs)
  thematic    -- thematic-ETF consensus: in >=2 of {VOLT,ELFY,ZAP,GRID}, filtered to
                 the sub-industry map + listing gate. No profitability screen.
  screen      -- OUR fundamental supplier rule, NO ETF gate: seed pool & sub-industry
                 map & customer_institutional & (earnings_valued OR
                 low_policy_dependence). This is the capex-cycle-pair-classifier
                 rule (memory `capex-cycle-pair-classifier`), restricted to the
                 long/supplier side. Names are picked on stated properties, not on
                 what VOLT holds.
                 REIT_SUB_INDUSTRY rows are explicitly excluded from this pool, so
                 adding IRM/AMT to the seed CSV for `screen_reit` (below) leaves
                 `screen`'s own membership provably unchanged.
  screen_reit -- Task 7 REIT-breadth probe: the same rule as `screen`, but WITHOUT
                 the REIT_SUB_INDUSTRY exclusion, i.e. `screen`'s pool union any
                 data_center_reit-tagged names that clear the same customer +
                 quality bar (IRM, AMT). Tests whether data-center-exposed REITs
                 add breadth without hurting the strategy's risk profile
                 (docs/electrification-reit-breadth-probe-results.md).

CAVEAT: seed membership, the ETF snapshot, and the judgment columns are
2026-vintage, back-cast. Only the listing-date gate is genuinely point-in-time.
The binding criterion is `customer_institutional` -- a stable structural fact
about who a firm sells to -- so the back-cast exposure sits mainly in *seed-list
membership*, not in the filter (30 of 33 seed names pass). `thematic`/`frozen`
additionally depend on the ETF snapshot, which is top-holdings-only for
GRID/PAVE/ELFY (128/102/116 holdings, top ~25 visible) -- a full point-in-time
holdings feed is a pre-live gate for those two. `screen`/`screen_reit` have no
ETF dependency.

CHANGED 2026-09-22 -- `profitable_2026` removed.
  We used to do: `customer_institutional` AND >=2 of 3 of
      {`profitable_2026`, `earnings_valued`, `low_policy_dependence`}.
  We now do:     `customer_institutional` AND (`earnings_valued` OR
      `low_policy_dependence`).  `frozen` also dropped it as a hard filter.
Two reasons. (1) It was LOOK-AHEAD: a single profitability verdict recorded in
2026, applied backwards to every backtest date from 2017 on, so a firm that was
loss-making in 2019 but profitable by 2026 counted as profitable in 2019 -- the
column name said as much. (2) It was NOT HELPFUL: an ablation through
`backtest.run_comparison` showed it fully redundant -- `screen`/`screen_reit`/
`frozen` give identical pools with and without it (28/30/15 names, zero adds,
zero drops) and byte-identical returns (max abs diff 0.00e+00 over 2,421 days),
because the only two names it excluded (FLNC, STEM) already fail the other
criteria or sit in no ETF. Removing it cost nothing and removed the one
explicitly year-stamped input.
"""
from __future__ import annotations

import pandas as pd

from grid_equipment_basket.basket import apply_cap

from electrification_strategy import auto_screen, config

THEMATIC_ETFS = ["VOLT", "ELFY", "ZAP", "GRID"]
_MEMBERSHIP_COLS = ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]
_BOOL_COLS = ["customer_institutional", "earnings_valued", "low_policy_dependence"]
_SCREEN_QUALITY = ["earnings_valued", "low_policy_dependence"]

CONSTRUCTIONS = ("marquee", "frozen", "thematic", "screen", "screen_reit")
# `screen_auto` needs a build artifact (`data/screen_auto_members.csv`), so it is
# selectable but not part of the default comparison grid -- a plain CLI run must not
# depend on a file that only exists after a rebuild.
EXTRA_CONSTRUCTIONS = ("screen_auto",)
ALL_CONSTRUCTIONS = CONSTRUCTIONS + EXTRA_CONSTRUCTIONS
REIT_SUB_INDUSTRY = "data_center_reit"


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False}).astype(bool)


def load_seed() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "universe_seed.csv", comment="#")
    for col in _BOOL_COLS:
        df[col] = _to_bool(df[col])
    df = df.set_index("ticker")
    bad = set(df["sub_industry"]) - config.SUB_INDUSTRIES
    if bad:
        raise ValueError(f"universe_seed.csv has unknown sub_industry values: {sorted(bad)}")
    return df


def load_membership() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "etf_membership_2026.csv", comment="#").set_index("ticker")
    for col in _MEMBERSHIP_COLS:
        df[col] = _to_bool(df[col])
    return df[_MEMBERSHIP_COLS]


def candidate_tickers(constructions) -> list[str]:
    """Every name any requested construction can hold -- what must be priced.

    The seed-list constructions draw from marquee + seed; `screen_auto` draws from the
    SEC universe, so its build artifact's names are added only when it is requested."""
    cand = set(config.MARQUEE_UNIVERSE) | set(load_seed().index)
    if "screen_auto" in constructions:
        held = auto_screen.load_membership()
        cand |= set(held.columns[held.astype(bool).any()])
    return sorted(cand)


def _listed_eligible(prices: pd.DataFrame, asof: pd.Timestamp) -> set[str]:
    upto = prices.loc[:asof]
    counts = upto.notna().sum()
    return set(counts[counts >= config.LISTING_MIN_DAYS].index)


def members(construction: str, asof: pd.Timestamp, prices: pd.DataFrame) -> list[str]:
    listed = _listed_eligible(prices, asof)

    if construction == "marquee":
        return sorted(set(config.MARQUEE_UNIVERSE) & listed)

    seed = load_seed()
    in_map = set(seed.index)                       # sub_industry validated in load_seed

    if construction == "screen_auto":
        # Mechanical: SEC SIC population + capex-light + filer vocabulary + filed
        # earnings + policy exclusion. No seed list, no judgment column.
        held = auto_screen.load_membership()
        upto = held.loc[:asof]
        if upto.empty:
            return []
        row = upto.iloc[-1]
        return sorted(set(row.index[row.astype(bool)]) & listed)

    if construction in ("screen", "screen_reit"):
        cust = set(seed.index[seed["customer_institutional"]])
        quality = set(seed.index[seed[_SCREEN_QUALITY].any(axis=1)])
        base_pool = in_map & cust & quality
        reit_tickers = set(seed.index[seed["sub_industry"] == REIT_SUB_INDUSTRY])
        if construction == "screen":
            return sorted((base_pool - reit_tickers) & listed)
        return sorted(base_pool & listed)   # screen_reit: screen ∪ qualifying REITs

    memb = load_membership()
    if construction == "frozen":
        any_etf = set(memb.index[memb[_MEMBERSHIP_COLS].any(axis=1)])
        pool = in_map & any_etf
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
