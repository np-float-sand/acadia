"""Mechanical, point-in-time universe construction.

Membership is never hand-picked. A ticker is eligible in month *m* when:

  1. its SEC filer SIC code is in `data/sic_codes.csv`, OR its annual report
     from the prior years matched the fuel-cycle lexicon by that year; AND
  2. its trailing-1yr median dollar volume clears the liquidity floor at *m*.

Rule (1) is why uranium names appear at all: CCJ and UEC file under SIC 1090
(Misc Metal Ores) and LEU/UUUU under 1400 (Nonmetallic Minerals), alongside
hundreds of gold miners, so no SIC screen can isolate them. What separates them
is their own filing language.
"""
from __future__ import annotations

import pandas as pd

from capex_intensity_strategy import config


def load_sic_codes() -> pd.DataFrame:
    """The SIC code table that defines the structural half of the universe."""
    df = pd.read_csv(config.DATA_DIR / "sic_codes.csv", comment="#")
    if not set(df["group"]) <= {"ELEC", "POWER"}:
        raise ValueError(f"sic_codes.csv has unknown group values: {sorted(set(df['group']))}")
    return df


def liquidity_mask(dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """Monthly boolean mask: trailing median dollar volume above the floor.

    Evaluated on a rolling window so a name enters when it becomes liquid and
    leaves when it stops -- the universe breathes rather than being fixed.
    """
    rolling = dollar_volume.rolling(
        config.LIQUIDITY_WINDOW_DAYS, min_periods=config.LIQUIDITY_MIN_PERIODS
    ).median()
    return (rolling > config.LIQUIDITY_MIN_DOLLAR_VOLUME).resample("ME").last().fillna(False)


def lexicon_mask(index: pd.DatetimeIndex, tickers: list[str],
                 membership_years: dict[str, list[int]]) -> pd.DataFrame:
    """Monthly mask for lexicon-discovered names, effective from their first year.

    `membership_years` maps ticker -> the universe-years whose lookback filings
    matched. A name stays eligible after its first qualifying year (a company
    does not stop being a uranium miner because one year's search missed it).
    Tickers absent from the mapping are structural (SIC) members and always pass.
    """
    mask = pd.DataFrame(True, index=index, columns=tickers)
    years = pd.Series(index.year, index=index)
    for ticker, qualifying in membership_years.items():
        if ticker not in mask.columns:
            continue
        if not qualifying:
            mask[ticker] = False
            continue
        first = min(qualifying)
        mask[ticker] = (years >= first).to_numpy()
    return mask


def membership(dollar_volume: pd.DataFrame, tickers: list[str],
               membership_years: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Final monthly eligibility mask: liquidity AND lexicon-effective-date."""
    cols = [t for t in tickers if t in dollar_volume.columns]
    liquid = liquidity_mask(dollar_volume[cols])
    if not membership_years:
        return liquid
    lex = lexicon_mask(liquid.index, cols, membership_years)
    return liquid & lex.reindex(index=liquid.index, columns=cols).fillna(True)


def discover_lexicon_members(fetch_ciks, cik_to_ticker: dict[str, str],
                             years: range) -> dict[str, list[int]]:
    """Run the fuel-cycle lexicon per universe-year and map CIKs to tickers.

    `fetch_ciks(phrase, start, end)` is injected so this is testable offline.
    For universe-year Y the search covers filings published in
    [Y - LEXICON_LOOKBACK_YEARS, Y - 1] -- strictly before Y begins.
    """
    out: dict[str, list[int]] = {}
    for year in years:
        start = f"{year - config.LEXICON_LOOKBACK_YEARS}-01-01"
        end = f"{year - 1}-12-31"
        ciks: set[str] = set()
        for phrase in config.FUEL_CYCLE_LEXICON:
            ciks |= fetch_ciks(phrase, start, end)
        for cik in ciks:
            ticker = cik_to_ticker.get(cik)
            if ticker:
                out.setdefault(ticker, []).append(year)
    return {k: sorted(v) for k, v in out.items()}
