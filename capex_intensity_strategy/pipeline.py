"""End-to-end build: universe -> facts -> score -> book.

Separated from `__main__` so the whole pipeline can be driven from a notebook
or a test with injected data instead of live network calls.
"""
from __future__ import annotations

import json

import pandas as pd
import yfinance as yf

from capex_intensity_strategy import backtest, config, edgar, fundamentals, universe


def build_candidates(refresh: bool = False) -> pd.DataFrame:
    """Ticker -> group for every filer in the SIC table, plus lexicon names."""
    path = config.CACHE_DIR / "candidates.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    codes = universe.load_sic_codes()
    cik_to_ticker = {v: k for k, v in edgar.ticker_to_cik(refresh=refresh).items()}
    rows = []
    for _, row in codes.iterrows():
        for cik in edgar.sic_population(int(row["sic"])):
            ticker = cik_to_ticker.get(cik)
            if ticker:
                rows.append({"ticker": ticker, "cik": cik,
                             "sic": int(row["sic"]), "group": row["group"],
                             "source": "sic"})
        print(f"  sic {int(row['sic'])}: cumulative {len(rows)}", flush=True)
    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
    out.to_parquet(path)
    return out


def build_lexicon_members(years: range, refresh: bool = False) -> dict[str, list[int]]:
    """Point-in-time fuel-cycle membership by universe-year."""
    path = config.CACHE_DIR / "lexicon_members.json"
    if path.exists() and not refresh:
        return {k: list(v) for k, v in json.loads(path.read_text()).items()}
    cik_to_ticker = {v: k for k, v in edgar.ticker_to_cik().items()}
    members = universe.discover_lexicon_members(
        edgar.full_text_ciks, cik_to_ticker, years)
    path.write_text(json.dumps(members))
    return members


def fetch_prices(tickers: list[str], start: str, end: str,
                 refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted closes and dollar volume for the candidate pool."""
    px_path, dv_path = config.CACHE_DIR / "prices.parquet", config.CACHE_DIR / "dollar_volume.parquet"
    if px_path.exists() and dv_path.exists() and not refresh:
        return pd.read_parquet(px_path), pd.read_parquet(dv_path)
    raw = yf.download(sorted(set(tickers)), start=start, end=end,
                      auto_adjust=True, progress=False)
    close, volume = raw["Close"], raw["Volume"]
    keep = [c for c in close.columns
            if close[c].dropna().shape[0] > config.MIN_PRICE_HISTORY_DAYS]
    close, dollar_volume = close[keep], (close[keep] * volume[keep])
    close.to_parquet(px_path)
    dollar_volume.to_parquet(dv_path)
    return close, dollar_volume


def build(start: str = config.START_DEFAULT, end: str | None = None,
          refresh: bool = False, short_weight: float | None = None) -> dict:
    """Run the whole thing and return every intermediate for inspection."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = build_candidates(refresh=refresh)
    years = range(int(start[:4]), int(end[:4]) + 1)
    lexicon = build_lexicon_members(years, refresh=refresh)
    tickers = sorted(set(candidates["ticker"]) | set(lexicon))
    close, dollar_volume = fetch_prices(
        tickers + list(config.BENCHMARKS) + list(config.COMPARISON_ETFS),
                                        start="2015-01-01", end=end, refresh=refresh)
    tradable = [t for t in tickers if t in close.columns]

    cik_map = edgar.ticker_to_cik()
    facts = edgar.fetch_facts({t: cik_map[t] for t in tradable if t in cik_map},
                              refresh=refresh)
    dates = pd.date_range(start, end, freq="ME")
    panel = fundamentals.ttm_panel(facts, dates)
    score = fundamentals.capex_intensity(panel)

    mask = universe.membership(dollar_volume, tradable, lexicon)
    index = score.index.union(mask.index)
    score = score.reindex(index)
    mask = mask.reindex(index).ffill().fillna(False)

    weights = backtest.target_weights(score, mask, short_weight=short_weight)
    returns = close.pct_change()
    net = backtest.run(weights, returns)
    return {
        "candidates": candidates, "lexicon": lexicon, "score": score, "mask": mask,
        "weights": weights, "returns": returns, "net": net, "prices": close,
        "summary": backtest.summarise(net),
        "diagnostics": backtest.diagnostics(weights, returns),
        "benchmarks": {b: backtest.summarise(returns[b])
                       for b in config.BENCHMARKS if b in returns.columns},
    }
