"""Mechanical replacement for the hand-recorded seed list and judgment columns.

`screen` picks names from a 33-row CSV whose membership and three boolean columns
were recorded by hand in 2026 and applied backwards to every date since 2017. That
seed list -- not the columns, which 30 of 33 names pass -- is the strategy's one
genuinely non-point-in-time input, and `electrification_strategy/README.md` names a
point-in-time seed list as a pre-live gate.

This module replaces it with four filters, each of which reads only filings
published before the universe-year it decides:

| hand-recorded input      | mechanical replacement                              |
|--------------------------|-----------------------------------------------------|
| the 33-name seed list    | SEC SIC population + fuel-cycle lexicon + liquidity |
| (implicitly) not a utility | TTM capex / revenue in the light half              |
| `customer_institutional` | buildout vocabulary in the prior-year 10-K          |
| `earnings_valued`        | TTM `NetIncomeLoss` > 0                             |
| `low_policy_dependence`  | absence of ITC / 45X / PTC / NEM / NEVI vocabulary   |

Network access is injected (`fetch_count`) so the whole module is testable offline,
matching `capex_intensity_strategy.universe.discover_lexicon_members`.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from electrification_strategy import config

MEMBERSHIP_FILE = "screen_auto_members.csv"
VOCABULARY_FILE = "vocabulary_hits.csv"
DC_EXCLUDED_FILE = "vocabulary_hits_dc_excluded.csv"


def load_vocabulary_hits(files=(VOCABULARY_FILE,), clean: bool = True) -> pd.DataFrame:
    """The cached EDGAR full-text hits behind the vocabulary filters.

    `clean=True` (every production caller) applies `clean_hits`: spurious "data center(s)"
    hits and gas transmission/distribution filers are removed. `clean=False` returns the raw
    full-text hits.
    """
    frames = [pd.read_csv(config.DATA_DIR / f, comment="#") for f in files
              if (config.DATA_DIR / f).exists()]
    if not frames:
        return pd.DataFrame(columns=["ticker", "phrase", "file_date"])
    hits = pd.concat(frames, ignore_index=True).drop_duplicates()
    if not clean:
        return hits
    return clean_hits(hits, excluded=load_dc_exclusions(), excluded_tickers=rule2_excluded_tickers())


_DC_ANY = re.compile(r"data cent(?:er|re)s?", re.I)
_DC_OWN = re.compile(
    r"\b(?:our|its|the company's|third[- ]party|co-?located|cloud|own)\s+(?:\w+\s+)?data cent(?:er|re)s?"
    r"|networks?\s+and\s+data cent(?:er|re)s?"
    r"|data cent(?:er|re)s?\s+(?:resiliency|redundancy|and networks?)", re.I)


def is_market_mention(text: str) -> bool:
    """Does `text` (a filing's Item 1) use "data center(s)" for a market, not the filer's own IT?

    True when at least one mention is NOT self-referential ("our/its/third-party/cloud ... data
    centers", "network and data center resiliency" and the like).
    """
    return len(_DC_ANY.findall(text)) > len(_DC_OWN.findall(text))


def load_dc_exclusions() -> pd.DataFrame | None:
    """Spurious "data center(s)" hits (ticker, phrase, file_date), built by `build_vocab_filter`."""
    path = config.DATA_DIR / DC_EXCLUDED_FILE
    return pd.read_csv(path, comment="#") if path.exists() else None


def rule2_excluded_tickers() -> set[str]:
    """Tickers whose SEC SIC code is in `config.RULE2_EXCLUDED_SICS` (gas T&D)."""
    from capex_intensity_strategy import config as cis_config
    path = cis_config.CACHE_DIR / "candidates.parquet"
    if not path.exists():
        return set()
    cand = pd.read_parquet(path)
    return set(cand.loc[cand["sic"].isin(config.RULE2_EXCLUDED_SICS), "ticker"])


def clean_hits(hits: pd.DataFrame, excluded: pd.DataFrame | None,
               excluded_tickers: set[str]) -> pd.DataFrame:
    """Drop the listed spurious data-center hits and every hit of `excluded_tickers`."""
    out = hits[~hits["ticker"].isin(excluded_tickers)]
    if excluded is not None and len(excluded):
        key = ["ticker", "phrase", "file_date"]
        ex = excluded[key].astype(str).drop_duplicates()
        hit = out[key].astype(str).merge(ex.assign(_x=1), on=key, how="left")["_x"].notna().to_numpy()
        out = out[~hit]
    return out.reset_index(drop=True)


def load_membership() -> pd.DataFrame:
    """Precomputed date x ticker membership, written by `build_auto_screen`.

    A build artifact, not a hand-recorded input: every row is reproducible from
    SEC filings by `python -m electrification_strategy.build_auto_screen`.
    """
    path = config.DATA_DIR / MEMBERSHIP_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m electrification_strategy.build_auto_screen`")
    long = pd.read_csv(path, comment="#", parse_dates=["date"])
    wide = long.assign(v=True).pivot_table(index="date", columns="ticker", values="v",
                                           fill_value=False)
    wide.columns.name = None
    return wide.astype(bool).sort_index()


def vocabulary_members(fetch_count, ciks: dict[str, str], terms,
                       years, min_hits: int | None = None) -> dict[str, list[int]]:
    """Ticker -> universe-years in which its prior filings used the vocabulary.

    `fetch_count(phrase, cik, start, end)` returns how many of that filer's filings
    in the window contain the exact phrase. For universe-year Y the window is
    [Y - AUTO_LOOKBACK_YEARS, Y - 1], strictly before Y begins, so a name joins the
    pool only on language it had already published.
    """
    min_hits = config.AUTO_MIN_HITS if min_hits is None else min_hits
    out: dict[str, list[int]] = {}
    for year in years:
        start = f"{year - config.AUTO_LOOKBACK_YEARS}-01-01"
        end = f"{year - 1}-12-31"
        for ticker, cik in ciks.items():
            for phrase in terms:
                if fetch_count(phrase, cik, start, end) >= min_hits:
                    out.setdefault(ticker, []).append(year)
                    break
    return {k: sorted(v) for k, v in out.items()}


def vocabulary_members_from_dates(fetch_dates, ciks: dict[str, str], terms,
                                  years) -> dict[str, list[int]]:
    """`vocabulary_members`, but one query per (filer, term) instead of per year.

    `fetch_dates(phrase, cik, start, end)` returns the filing dates that matched.
    A filing published in year F decides universe-years F+1 .. F+AUTO_LOOKBACK_YEARS,
    which is the same [Y - lookback, Y - 1] window read from the other end.
    """
    years = list(years)
    lo, hi = min(years), max(years)
    start = f"{lo - config.AUTO_LOOKBACK_YEARS}-01-01"
    end = f"{hi - 1}-12-31"
    out: dict[str, set[int]] = {}
    for ticker, cik in ciks.items():
        hit_years: set[int] = set()
        for phrase in terms:
            for date in fetch_dates(phrase, cik, start, end):
                hit_years.add(int(str(date)[:4]))
        covered = {y for f in hit_years
                   for y in range(f + 1, f + 1 + config.AUTO_LOOKBACK_YEARS)}
        covered &= set(years)
        if covered:
            out[ticker] = covered
    return {k: sorted(v) for k, v in out.items()}


def profitable_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """date x ticker: is trailing-twelve-month net income positive, as then filed."""
    if panel.empty or "net_income" not in panel.columns:
        return pd.DataFrame()
    wide = panel.pivot_table(index="date", columns="ticker", values="net_income",
                             dropna=False)
    wide.columns.name = None
    return (wide > 0).fillna(False).sort_index()


def light_capex_mask(score: pd.DataFrame, mask: pd.DataFrame,
                     quantile: float | None = None) -> pd.DataFrame:
    """date x ticker: is this name in the light `quantile` of the eligible pool.

    Ranking uses eligible names only, so an illiquid or unscored name never moves
    the cut.
    """
    quantile = config.AUTO_CAPEX_QUANTILE if quantile is None else quantile
    eligible = score.where(mask.reindex_like(score).fillna(False))
    ranks = eligible.rank(axis=1, method="first")
    n_scored = eligible.notna().sum(axis=1)
    n_pick = np.maximum(1, np.round(n_scored * quantile)).where(n_scored >= 2, 0)
    return (ranks.le(n_pick, axis=0) & eligible.notna()).fillna(False)


def vocabulary_years(hits: pd.DataFrame, terms, years, lookback: int | None = None,
                     min_hits: int = 1) -> dict[str, list[int]]:
    """`vocabulary_members_from_dates`, read off a cached hit table instead of the wire.

    Universe-year Y counts the matching filings published in [Y - lookback, Y - 1]
    and keeps the name only once that count reaches `min_hits`. The count is
    per-window, not lifetime: a name with one hit a year never clears `min_hits=2`
    under a two-year lookback.
    """
    lookback = config.AUTO_LOOKBACK_YEARS if lookback is None else lookback
    years = set(years)
    if hits.empty:
        return {}
    sub = hits[hits["phrase"].isin(terms)]
    out: dict[str, list[int]] = {}
    for ticker, group in sub.groupby("ticker"):
        filed = [int(str(d)[:4]) for d in group["file_date"]]
        covered = sorted(y for y in years
                         if sum(1 for f in filed if y - lookback <= f <= y - 1) >= min_hits)
        if covered:
            out[ticker] = covered
    return out


def build_membership(score: pd.DataFrame, mask: pd.DataFrame, panel: pd.DataFrame,
                     hits: pd.DataFrame, power_tickers=frozenset(),
                     quantile: float | None = None, lookback: int | None = None,
                     policy_min_hits: int | None = None) -> pd.DataFrame:
    """date x ticker membership for `screen_auto`: every filter, intersected.

    `power_tickers` (SIC group POWER -- utilities, pipelines, generators) are
    dropped outright. The capex-light cut already removes almost all of them; the
    handful that survive are midstream names whose 10-Ks talk about data centres as
    customers, and they are off-thesis by construction rather than by performance.
    """
    light = light_capex_mask(score, mask, quantile=quantile)
    cols = [c for c in light.columns if c not in power_tickers]
    light = light[cols]
    years = sorted({d.year for d in light.index})
    policy_min_hits = (config.AUTO_POLICY_MIN_HITS if policy_min_hits is None
                       else policy_min_hits)
    vocab = vocabulary_years(hits, config.AUTO_CUSTOMER_TERMS, years, lookback=lookback)
    policy = vocabulary_years(hits, config.AUTO_POLICY_TERMS, years, lookback=lookback,
                              min_hits=policy_min_hits)

    def year_frame(mapping):
        data = {dt: pd.Series(True, index=[t for t, ys in mapping.items()
                                           if dt.year in ys and t in cols])
                for dt in light.index}
        return pd.DataFrame(data).T.reindex(columns=cols).fillna(False).astype(bool)

    profitable = profitable_mask(panel)
    profitable = (profitable.reindex(index=light.index, columns=cols)
                  .ffill().fillna(False).astype(bool)) if not profitable.empty \
        else pd.DataFrame(False, index=light.index, columns=cols)
    return light & year_frame(vocab) & profitable & ~year_frame(policy)


def _active(frame: pd.DataFrame, asof: pd.Timestamp) -> set[str]:
    if frame is None or frame.empty:
        return set()
    upto = frame.loc[:asof]
    if upto.empty:
        return set()
    row = upto.iloc[-1]
    return set(row.index[row.astype(bool)])


def members(asof, light=None, profitable=None, vocabulary=None,
            policy_flagged=None, quality_any: bool = False) -> list[str]:
    """The mechanical pool as of one date: every active filter, intersected.

    `quality_any` mirrors `screen`'s own shape -- pass on profitability OR policy
    independence rather than both -- and is off by default so each filter can be
    ablated on its own.
    """
    asof = pd.Timestamp(asof)
    year = asof.year
    pool = _active(light, asof)
    if vocabulary is not None:
        pool &= {t for t, yrs in vocabulary.items() if year in yrs}
    flagged = {t for t, yrs in (policy_flagged or {}).items() if year in yrs}
    profit = _active(profitable, asof)
    if profitable is not None and not quality_any:
        pool &= profit
    if quality_any:
        pool &= profit | (pool - flagged)
    else:
        pool -= flagged
    return sorted(pool)
