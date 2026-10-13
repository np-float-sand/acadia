"""Run the production one-score long/short on a shadow cache, with and without the step-aside rule.

    python research/survivorship_audit/run_compare.py base     # rebuild on production data (control)
    python research/survivorship_audit/run_compare.py shadow   # same rebuild + priced dropouts

Every stage is rebuilt from the shadow caches: SIC/liquidity universe, screen_auto membership,
A1+A4 members, the fourteen-signal score (fundamental panel and power betas recomputed), the
rule-2 short pool. One deliberate change vs production, applied to BOTH runs: a name must have a
real (not forward-filled) price on the rebalance day to be picked, so a dropout cannot be chosen
after it has stopped trading. Held positions are still valued at the last price (cash after exit).
"""
import json, os, pathlib, pickle, sys
MODE = sys.argv[1]
D = pathlib.Path(__file__).resolve().parent; ROOT = D.parents[1]
R = D/{"shadow": "_shadow", "base": "_shadow_base", "dcfix": "_shadow_dcfix", "gas": "_shadow_gas", "both": "_shadow_both"}[MODE]
os.environ["ACADIA_DATA_ROOT"] = str(R)
sys.path[:0] = [str(ROOT), str(ROOT/"research"/"electrification_strategy_probes")]
import pandas as pd
from capex_intensity_strategy import config as cis_config
from electrification_strategy import config as es_config
cis_config.CACHE_DIR = R/"capex_intensity_strategy"/"data"/"cache"
es_config.DATA_DIR = R/"es_data"
from research.xsec_alpha_probe import data
data.CACHE = R/"xsec_cache"
import grid_equipment_basket.basket as gb
_sim = gb.simulate_basket
def sim_live(prices, start, end, lag_days=42, cap=0.25, target_fn=None):
    raw = prices.loc[start:end].sort_index()
    def fn(available, asof):
        live = raw.loc[asof].dropna().index if asof in raw.index else available
        return target_fn([a for a in available if a in live], asof)
    return _sim(prices, start, end, lag_days, cap, fn if target_fn else None)
import investor_book as ib, build_tophalf_proposal as t, build_a1a4_proposal as a14, build_onescore_proposal as one
ib.simulate_basket = t.simulate_basket = a14.simulate_basket = sim_live
from electrification_strategy import build_auto_screen
from electrification_strategy.fred import fetch_series

build_auto_screen.main([])                                  # screen_auto membership -> shadow es_data
a14.CACHE = str(R/"a1a4_build_cache.pkl")
if os.path.exists(a14.CACHE): os.remove(a14.CACHE)
pub = a14.build_returns()                                   # A1+A4 members on shadow data
members = pub["members"]
px = data.prices().copy()
extra = sorted(set(t.BENCH) | set(es_config.HEDGE_SLEEVE_TICKERS) | set(es_config.HEDGE_SHORT_TICKERS))
spx = t.fetch_prices(extra, "2015-01-01", t.END)
for c in spx.columns:
    if c not in px.columns: px[c] = spx[c].reindex(px.index)
comp = one.onescore_composite()
bench = {b: px[b].loc[t.START:t.END].pct_change() for b in t.BENCH if b in px.columns}
spy = bench["SPY"]
dfii10 = fetch_series(es_config.HEDGE_SHORT_SIGNAL, "2003-01-01", t.END)
ry = fetch_series(t.RATE_SIGNAL, "2003-01-01", t.END).dropna()
cash = fetch_series(t.CASH_SERIES, "2003-01-01", t.END).dropna()
sgrowth = t.supplier_growth(members)
step = ib.build(members, comp, px, spy, dfii10, ry, cash, growth=sgrowth)          # production: with step-aside
always = ib.build(members, comp, px, spy, dfii10, ry.clip(lower=1.0), cash)        # step-aside never fires
wins = {"full": (t.START, t.END), "2019-22": ("2019-01-01", "2022-12-31"), "2023+": ("2023-01-01", t.END)}
out = {"mode": MODE, "stats": {}, "alpha_vs_GRID": {}}
for lab, b in (("with_step_aside", step), ("without_step_aside", always)):
    r = b["returns"]
    out["stats"][lab] = {w: t.a1.stats(r.loc[a:e], spy.loc[a:e]) for w, (a, e) in wins.items()}
    out["alpha_vs_GRID"][lab] = {w: t.alpha(r.loc[a:e], bench["GRID"].loc[a:e]) for w, (a, e) in wins.items()}
    out["stats"][lab]["short_basket_full"] = t.a1.stats(b["short_returns"].loc[t.START:t.END])
# Which dropouts entered each stage, and what they earned while held
drop = pd.read_csv(D/"dropout_universe_shadow.csv") if MODE == "shadow" else pd.DataFrame(columns=["col"])
dcols = set(drop["col"])
univ = data.universe()
rets = px.pct_change(fill_method=None)
sel = {}
for c in sorted(dcols):
    W = step["long_weights"].get(c)
    short_q = [str(k.date()) for k, v in step["short_log"].items() if c in v]
    long_days = int((W > 0).sum()) if W is not None else 0
    rec = {"eligible_months": int(univ[c].sum()) if c in univ.columns else 0,
           "screen_member_months": int(members[c].sum()) if c in members.columns else 0,
           "long_days": long_days, "short_quarters": short_q}
    if long_days:
        rec["long_contrib"] = round(float((W.shift(1) * rets[c].reindex(W.index)).sum()), 4)
    sel[c] = rec
out["dropouts"] = sel
out["names"] = {"long_mean": round(float(step["long_held"].loc[t.START:t.END].mean()), 1),
                "short_mean": round(float(step["short_held"][step["short_held"] > 0].mean()), 1)}
(D/f"compare_{MODE}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: {w: v[w]["sharpe"] for w in wins} for k, v in out["stats"].items()}))
