"""THE decisive test for the B1 short: is 'capital-heavy AND persistently unprofitable'
a factor, or was it the 2021-24 hydrogen bubble?

    python research/electrification_strategy_probes/broad_universe_short_test.py

Within the electricity universe the rule produces -21.4%/yr standalone. The 77-name
non-electricity theme set was too thin to judge (1.5 qualifying names/month). This
builds a ~1000-name multi-sector universe (SEC company_tickers.json is ordered by
size) and asks whether the same rule selects losers OUTSIDE electricity, where the
phenotype should have a real population.

Writes its own facts/price caches -- capex_intensity_strategy.edgar.fetch_facts(
refresh=True) would OVERWRITE the main 280-ticker cache.
"""
from __future__ import annotations

import json, os, sys, time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from capex_intensity_strategy import config as cc, edgar, fundamentals
from electrification_strategy import backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
FACTS = os.path.join(OUT, "_broad1k_facts.parquet")
PX = os.path.join(OUT, "_broad1k_px.parquet")
DV = os.path.join(OUT, "_broad1k_dv.parquet")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN
N_UNIVERSE = 1000


def universe():
    cik = json.load(open(os.path.join(ROOT, "capex_intensity_strategy/data/cache/ticker_cik.json")))
    return list(cik.items())[:N_UNIVERSE], cik


def get_prices(tickers):
    if os.path.exists(PX) and os.path.exists(DV):
        return pd.read_parquet(PX), pd.read_parquet(DV)
    import yfinance as yf
    closes, vols = [], []
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        raw = yf.download(chunk, start="2015-01-01", end=END, auto_adjust=True,
                          progress=False, threads=True)
        closes.append(raw["Close"])
        vols.append(raw["Volume"])
        print(f"  prices {i + len(chunk)}/{len(tickers)}", flush=True)
    close = pd.concat(closes, axis=1)
    vol = pd.concat(vols, axis=1)
    keep = [c for c in close.columns
            if close[c].dropna().shape[0] > cc.MIN_PRICE_HISTORY_DAYS]
    close, dv = close[keep], (close[keep] * vol[keep])
    close.to_parquet(PX)
    dv.to_parquet(DV)
    return close, dv


def get_facts(tickers, cik):
    if os.path.exists(FACTS):
        return pd.read_parquet(FACTS)
    frames = []
    for i, t in enumerate(tickers):
        if t not in cik:
            continue
        try:
            f = edgar.company_facts(cik[t])
        except Exception:
            continue
        if len(f):
            f.insert(0, "ticker", t)
            frames.append(f)
        if i % 50 == 0:
            print(f"  edgar {i}/{len(tickers)}  ({len(frames)} with facts)", flush=True)
        time.sleep(cc.SEC_RATE_LIMIT_SLEEP)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(FACTS)
    return out


def main():
    pairs, cik = universe()
    tickers = [t for t, _ in pairs]
    print(f"universe: top {len(tickers)} US filers by size")
    close, dv = get_prices(tickers)
    print(f"  with usable price history: {close.shape[1]}")
    have = [t for t in tickers if t in close.columns]
    facts = get_facts(have, cik)
    print(f"  with EDGAR facts: {facts['ticker'].nunique()}")

    dates = pd.date_range(START, END, freq="ME")
    panel = fundamentals.ttm_panel(facts, dates)
    score = fundamentals.capex_intensity(panel)
    cols = [c for c in score.columns if c in close.columns]
    score = score[cols]
    idx = score.index

    ni = panel.pivot_table(index="date", columns="ticker", values="net_income", dropna=False)
    nip = ni.reindex(index=idx, columns=cols).ffill()
    loss = (nip < 0).fillna(False)
    persistent = loss.rolling(12, min_periods=6).min().fillna(0).astype(bool)
    liq = (dv[[c for c in cols if c in dv.columns]].rolling(252, min_periods=120).median()
           >= cc.LIQUIDITY_MIN_DOLLAR_VOLUME)
    liq = liq.reindex(index=idx, columns=cols).ffill().fillna(False)
    elig = liq & score.notna()

    # the electricity universe this rule was discovered in -- excluded for the clean read
    cand = pd.read_parquet(os.path.join(ROOT, "capex_intensity_strategy/data/cache/candidates.parquet"))
    electric = set(cand["ticker"])
    non_elec = elig.copy()
    for t in electric:
        if t in non_elec.columns:
            non_elec[t] = False

    from grid_equipment_basket.basket import simulate_basket, apply_cap

    def run(memb):
        c2 = [c for c in memb.columns if c in close.columns]
        m = memb[c2]
        if not m.any().any():
            return None

        def tfn(available, asof):
            asof = pd.Timestamp(asof)
            upto = m.loc[:asof]
            if upto.empty:
                return pd.Series(dtype=float)
            row = upto.iloc[-1]
            held = set(row.index[row.astype(bool)])
            cnt = close.loc[:asof].notna().sum()
            listed = set(cnt[cnt >= config.LISTING_MIN_DAYS].index)
            e = sorted(held & set(available) & listed)
            if not e:
                return pd.Series(dtype=float)
            return apply_cap(pd.Series(1.0 / len(e), index=e), config.MAX_SINGLE_NAME_WEIGHT)

        return simulate_basket(close[c2], START, END, config.REBALANCE_LAG_DAYS,
                               config.MAX_SINGLE_NAME_WEIGHT, tfn)

    res = {}
    for pool_name, pool in [("ALL sectors", elig), ("EX-electricity", non_elec)]:
        rank = score.where(pool).rank(axis=1, pct=True)
        print(f"\n{'='*92}\n{pool_name}: {pool.loc['2019':].sum(axis=1).mean():.0f} eligible/month, "
              f"{(pool & persistent).loc['2019':].sum(axis=1).mean():.1f} capital-heavy-and-losing")
        for cut, clab in [(0.75, "heaviest 25%"), (0.60, "heaviest 40%"), (0.50, "heaviest 50%")]:
            memb = (rank >= cut) & persistent & pool
            br = run(memb)
            if br is None or br.returns.empty:
                print(f"  {clab:<13}: no names"); continue
            m = backtest.compute_metrics(br.returns, RF, ANN)
            held = (br.weights > 0).sum(axis=1)
            res[f"{pool_name}|{clab}"] = {"names": round(float(held.mean()), 1),
                                          "cagr": m["cagr"], "sharpe": m["sharpe"]}
            print(f"  {clab:<13}: {held.mean():5.1f} names   standalone {m['cagr']*100:+7.1f}%/yr"
                  f"   Sharpe {m['sharpe']:+.2f}   vol {m['ann_vol']*100:.0f}%")
        # control: capital-heavy but profitable
        memb = (rank >= 0.60) & ~loss & pool
        br = run(memb)
        if br is not None and not br.returns.empty:
            m = backtest.compute_metrics(br.returns, RF, ANN)
            print(f"  CONTROL heaviest 40% + PROFITABLE: {(br.weights>0).sum(axis=1).mean():.0f} names"
                  f"   {m['cagr']*100:+.1f}%/yr   Sharpe {m['sharpe']:+.2f}")

    json.dump(res, open(os.path.join(OUT, "broad_universe_short.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
