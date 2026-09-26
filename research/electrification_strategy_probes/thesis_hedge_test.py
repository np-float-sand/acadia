"""Test the three thesis-driven hedge baskets against the book.

The proposal (telecom-glut analogy: the fiber build-out bankrupted the equipment
layer while the optimisation layer captured the value):

  1. POWER-CONSTRAINED  CEG, VST     -- IPPs with contracted generation win if the
                                        public grid blocks data-centre build-out
  2. ARCHITECTURAL      AAPL, QCOM, ARM -- if centralised DCs stall, AI moves on-device
  3. CAPEX FATIGUE      GOOGL, MSFT  -- optimisation software replaces machinery

Two tests, and they are different questions:
  (a) is it a HEDGE in general -- negative down-day beta, positive in drawdowns?
  (b) does it work in the SPECIFIC feared scenario? The `DeepSeek 25` episode already
      in config.EPISODES is a real instance of "efficient AI means less hardware",
      which is precisely the thesis's trigger.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from grid_equipment_basket.data.prices import fetch_prices
from electrification_strategy import backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN
CACHE = os.path.join(OUT, "_thesis_px.parquet")

BASKETS = {
    "1. Power-constrained (CEG, VST)": ["CEG", "VST"],
    "2. Architectural / edge AI (AAPL, QCOM, ARM)": ["AAPL", "QCOM", "ARM"],
    "3. Capex fatigue (GOOGL, MSFT)": ["GOOGL", "MSFT"],
    "all three combined": ["CEG", "VST", "AAPL", "QCOM", "ARM", "GOOGL", "MSFT"],
}
NAMES = sorted({t for v in BASKETS.values() for t in v})


def px():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    p = fetch_prices(NAMES + ["BTAL", "GLD", "SPY"], "2015-01-01", END)
    p.to_parquet(CACHE)
    return p


d = pd.read_pickle(os.path.join(OUT, "_build_cache.pkl"))
book, core = d["book"], d["core"]
p = px()
rets = p.reindex(book.index).pct_change().fillna(0.0)
spy = d["bench"]["SPY"].reindex(book.index).fillna(0.0)

down = book < 0
curve = (1 + book).cumprod()
in_dd = (curve / curve.cummax() - 1) < -0.08


def basket(tk):
    tk = [t for t in tk if t in rets.columns]
    return rets[tk].mean(axis=1)


print("=" * 104)
print("(a) IS IT A HEDGE IN GENERAL?   listed-late names start when they list")
print("=" * 104)
rows = []
for name, tks in BASKETS.items():
    r = basket(tks)
    live = r != 0
    m = backtest.compute_metrics(r[live], RF, ANN)
    rows.append({"basket": name,
                 "drift_%yr": round(m["cagr"] * 100, 1),
                 "corr": round(float(r.corr(book)), 2),
                 "corr_down": round(float(r[down].corr(book[down])), 2),
                 "beta_down": round(float(np.cov(r[down], book[down])[0, 1] / book[down].var()), 2),
                 "in_drawdown_%yr": round(float(r[in_dd].mean() * 252 * 100), 1)})
for t in ["BTAL", "GLD"]:
    r = rets[t]
    m = backtest.compute_metrics(r, RF, ANN)
    rows.append({"basket": f"[reference] {t}", "drift_%yr": round(m["cagr"] * 100, 1),
                 "corr": round(float(r.corr(book)), 2),
                 "corr_down": round(float(r[down].corr(book[down])), 2),
                 "beta_down": round(float(np.cov(r[down], book[down])[0, 1] / book[down].var()), 2),
                 "in_drawdown_%yr": round(float(r[in_dd].mean() * 252 * 100), 1)})
df = pd.DataFrame(rows)
pd.set_option("display.width", 240, "display.max_columns", 30)
print(df.to_string(index=False))

print("\n" + "=" * 104)
print("(b) DOES IT WORK IN THE FEARED SCENARIO?  total return through each episode")
print("    'DeepSeek 25' is the real instance of 'efficient AI needs less hardware'")
print("=" * 104)
ep_rows = []
for ep, ((pa, pb), tend) in config.EPISODES.items():
    seg = slice(pa, tend)
    rec = {"episode": ep, "THE BOOK": f"{((1+book.loc[seg]).prod()-1)*100:+.1f}%"}
    for name, tks in BASKETS.items():
        r = basket(tks).loc[seg]
        rec[name.split(".")[0] if name[0].isdigit() else name] = \
            f"{((1+r).prod()-1)*100:+.1f}%"
    for t in ["BTAL", "GLD"]:
        rec[t] = f"{((1+rets[t].loc[seg]).prod()-1)*100:+.1f}%"
    ep_rows.append(rec)
print(pd.DataFrame(ep_rows).to_string(index=False))

print("\n" + "=" * 104)
print("(c) AS A SLEEVE — replacing the shipped GLD/IEF, against the bar 1.050 / 1.118")
print("=" * 104)
out = []
w_s = config.HEDGE_SLEEVE_WEIGHT
out.append({"sleeve": "shipped 15% GLD/IEF", "wt": 0.15,
            **{k: v for k, v in zip(["Sharpe", "Calmar", "CAGR", "MaxDD"], [None] * 4)}})
rows2 = []
mb = backtest.compute_metrics(book.dropna(), RF, ANN)
rows2.append({"sleeve": "-- shipped 15% GLD/IEF --", "wt": 0.15, "Sharpe": mb["sharpe"],
              "Calmar": round(mb["cagr"] / abs(mb["max_dd"]), 3),
              "CAGR": f"{mb['cagr']*100:.1f}%", "MaxDD": f"{mb['max_dd']*100:.1f}%",
              "beta": round(float(np.cov(book.dropna(), spy.reindex(book.dropna().index))[0, 1] / spy.var()), 2)})
for name, tks in list(BASKETS.items()) + [("[reference] GLD/BTAL 50/50", ["GLD", "BTAL"])]:
    sl = basket(tks)
    for w in (0.15, 0.25):
        r = (1 - w) * core + w * sl.reindex(core.index).fillna(0.0)
        x = backtest.compute_metrics(r.dropna(), RF, ANN)
        rows2.append({"sleeve": name, "wt": w, "Sharpe": x["sharpe"],
                      "Calmar": round(x["cagr"] / abs(x["max_dd"]), 3),
                      "CAGR": f"{x['cagr']*100:.1f}%", "MaxDD": f"{x['max_dd']*100:.1f}%",
                      "beta": round(float(np.cov(r.dropna(), spy.reindex(r.dropna().index))[0, 1] / spy.var()), 2)})
df2 = pd.DataFrame(rows2)
df2["PASS"] = np.where((df2["Sharpe"] >= 1.050) & (df2["Calmar"] >= 1.118), "Y", "")
print(df2.to_string(index=False))
df2.to_json(os.path.join(OUT, "thesis_hedge_test.json"), orient="records", indent=1)


# ---------------------------------------------------------------------------
# CORRECTION: EPISODES encodes a (peak-search window, trough-end). Slicing the
# whole window includes the run-up INTO the episode, which flatters anything that
# rallied first (CEG/VST doubled before DeepSeek, then crashed on it). Measure
# peak -> trough using the repo's own episode finder instead.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from grid_equipment_basket import hedges as _geh
    print("\n" + "=" * 104)
    print("(b-corrected) PEAK-TO-TROUGH of the book's own drawdown, and what each basket did")
    print("=" * 104)
    rows3 = []
    for ep, (peak_win, tend) in config.EPISODES.items():
        try:
            peak, trough = _geh.find_drawdown_episode(book, peak_win, tend)
        except Exception as e:
            print(f"  {ep}: {e}")
            continue
        seg = slice(peak, trough)
        rec = {"episode": f"{ep} ({str(peak)[:10]}→{str(trough)[:10]})",
               "THE BOOK": f"{((1+book.loc[seg]).prod()-1)*100:+.1f}%"}
        for name, tks in BASKETS.items():
            key = name.split(".")[0] if name[0].isdigit() else "all 3"
            rec[key] = f"{((1+basket(tks).loc[seg]).prod()-1)*100:+.1f}%"
        for t in ["BTAL", "GLD"]:
            rec[t] = f"{((1+rets[t].loc[seg]).prod()-1)*100:+.1f}%"
        rows3.append(rec)
    print(pd.DataFrame(rows3).to_string(index=False))
