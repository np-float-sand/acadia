"""The mirror of the short test: is the LONG book's edge the theme, or just quality?

The broad-universe test rejected the short leg by showing the same screen returns
+52%/yr outside electricity. The same question has to be asked of the long book:
the screen is (capital-light AND profitable AND grid vocabulary AND liquid). If
"capital-light AND profitable AND liquid" alone does just as well on 950 stocks
across every sector, then the electrification theme is decoration and the strategy
is a generic quality factor.

Reuses the caches built by broad_universe_short_test.py.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from grid_equipment_basket.basket import simulate_basket, apply_cap
from grid_equipment_basket.data.prices import fetch_prices
from capex_intensity_strategy import config as cc, fundamentals
from electrification_strategy import backtest, config, hedge_overlay
from electrification_strategy.fred import fetch_series

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN

facts = pd.read_parquet(os.path.join(OUT, "_broad1k_facts.parquet"))
close = pd.read_parquet(os.path.join(OUT, "_broad1k_px.parquet"))
dv = pd.read_parquet(os.path.join(OUT, "_broad1k_dv.parquet"))
panel = fundamentals.ttm_panel(facts, pd.date_range(START, END, freq="ME"))
score = fundamentals.capex_intensity(panel)
cols = [c for c in score.columns if c in close.columns]
score = score[cols]
idx = score.index

ni = panel.pivot_table(index="date", columns="ticker", values="net_income", dropna=False)
nip = ni.reindex(index=idx, columns=cols).ffill()
profitable = (nip > 0).fillna(False)
liq = (dv[[c for c in cols if c in dv.columns]].rolling(252, min_periods=120).median()
       >= cc.LIQUIDITY_MIN_DOLLAR_VOLUME)
liq = liq.reindex(index=idx, columns=cols).ffill().fillna(False)
elig = liq & score.notna()

cand = pd.read_parquet(os.path.join(ROOT, "capex_intensity_strategy/data/cache/candidates.parquet"))
electric = set(cand["ticker"])
non_elec = elig.copy()
for t in electric:
    if t in non_elec.columns:
        non_elec[t] = False

# overlay inputs
extra = fetch_prices(sorted(set(config.HEDGE_SLEEVE_TICKERS) | {"SPY"}
                            | set(config.HEDGE_SHORT_TICKERS)), "2015-01-01", END)
px_all = close.copy()
for t in extra.columns:
    if t not in px_all.columns:
        px_all[t] = extra[t].reindex(px_all.index)
spy_ret = px_all["SPY"].loc[START:END].pct_change()
sleeve_r = hedge_overlay.sleeve_returns(px_all)
short_r = hedge_overlay.short_returns(px_all)
dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)


def run(memb, overlay=True):
    c2 = [c for c in memb.columns if c in px_all.columns]
    m = memb[c2]
    if not m.any().any():
        return None, None

    def tfn(available, asof):
        asof = pd.Timestamp(asof)
        upto = m.loc[:asof]
        if upto.empty:
            return pd.Series(dtype=float)
        row = upto.iloc[-1]
        held = set(row.index[row.astype(bool)])
        cnt = px_all.loc[:asof].notna().sum()
        listed = set(cnt[cnt >= config.LISTING_MIN_DAYS].index)
        e = sorted(held & set(available) & listed)
        if not e:
            return pd.Series(dtype=float)
        return apply_cap(pd.Series(1.0 / len(e), index=e), config.MAX_SINGLE_NAME_WEIGHT)

    br = simulate_basket(px_all[c2], START, END, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, tfn)
    if not overlay:
        return br.returns, br
    core = backtest.compose(br.returns, spy_ret, use_valuation=True)
    rising = hedge_overlay.real_yield_rising_mask(dfii10, core.index)
    return hedge_overlay.apply_hedge(core, sleeve_r, short_r, rising, True, False), br


def M(r, label, n=None):
    x = backtest.compute_metrics(r.dropna(), RF, ANN)
    s = spy_ret.reindex(r.dropna().index).fillna(0.0)
    return {"book": label, "names": n, "CAGR": f"{x['cagr']*100:.1f}%",
            "Vol": f"{x['ann_vol']*100:.1f}%", "Sharpe": x["sharpe"],
            "MaxDD": f"{x['max_dd']*100:.1f}%",
            "Calmar": round(x["cagr"] / abs(x["max_dd"]), 3) if x["max_dd"] < 0 else None,
            "beta": round(float(np.cov(r.dropna(), s)[0, 1] / s.var()), 2)}


rank = score.where(elig).rank(axis=1, pct=True)
rank_ne = score.where(non_elec).rank(axis=1, pct=True)

rows = []
book = pd.read_pickle(os.path.join(OUT, "_build_cache.pkl"))["book"]
rows.append(M(book, "The electrification book (A1+A4)", 20.9))

for lab, rk, pool in [("broad, ALL sectors", rank, elig),
                      ("broad, EX-electricity", rank_ne, non_elec)]:
    for q, qlab in [(0.60, "lightest 60%"), (0.50, "lightest 50%"), (0.25, "lightest 25%")]:
        memb = (rk <= q) & profitable & pool
        r, br = run(memb)
        if r is None:
            continue
        rows.append(M(r, f"{lab}: {qlab} + profitable",
                      round(float((br.weights > 0).sum(axis=1).mean()), 1)))

# the theme with no quality screen at all -- just the electricity universe, EW
memb = elig.copy()
for t in list(memb.columns):
    if t not in electric:
        memb[t] = False
r, br = run(memb)
if r is not None:
    rows.append(M(r, "electricity universe, no screen at all",
                  round(float((br.weights > 0).sum(axis=1).mean()), 1)))

df = pd.DataFrame(rows)
pd.set_option("display.width", 230, "display.max_columns", 30)
print(df.to_string(index=False))
df.to_json(os.path.join(OUT, "broad_universe_long.json"), orient="records", indent=1)
