"""Test the hedge candidates that survived the down-day screen, as a sleeve.

Bar: keep or improve BOTH Sharpe (1.050) and Calmar (1.118) of the shipped book.

The shipped book is 85% core + 15% GLD/IEF. To test alternative sleeves fairly we
rebuild from `core` (vol-targeted + valuation-delevered, PRE-sleeve) rather than
layering on top of the existing sleeve.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from electrification_strategy import backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN

d = pd.read_pickle(os.path.join(OUT, "_build_cache.pkl"))
book, core = d["book"], d["core"]
px = pd.read_parquet(os.path.join(OUT, "_hedge_px.parquet"))
rets = px.reindex(core.index).pct_change().fillna(0.0)
spy = d["bench"]["SPY"].reindex(core.index).fillna(0.0)

# the shipped sleeve, exactly
w_s = config.HEDGE_SLEEVE_WEIGHT
shipped_sleeve = (book - (1 - w_s) * core) / w_s

curve = (1 + book).cumprod()
in_dd = (curve / curve.cummax() - 1) < -0.08


def M(r, label, w=None):
    r = r.dropna()
    x = backtest.compute_metrics(r, RF, ANN)
    s = spy.reindex(r.index).fillna(0.0)
    c = (1 + r).cumprod()
    return {"sleeve": label, "wt": w, "CAGR": f"{x['cagr']*100:.1f}%",
            "Vol": f"{x['ann_vol']*100:.1f}%", "Sharpe": x["sharpe"],
            "MaxDD": f"{x['max_dd']*100:.1f}%",
            "Calmar": round(x["cagr"] / abs(x["max_dd"]), 3) if x["max_dd"] < 0 else None,
            "beta": round(float(np.cov(r, s)[0, 1] / s.var()), 2),
            "worst_yr": f"{min((1+r.loc[f'{y}']).prod()-1 for y in range(2019,2027))*100:.0f}%"}


def sleeve_from(tickers, weights=None):
    tk = [t for t in tickers if t in rets.columns]
    if not tk:
        return None
    w = np.array(weights if weights else [1 / len(tk)] * len(tk))
    return (rets[tk] * w).sum(axis=1)


CANDS = {
    "GLD/IEF (shipped)": shipped_sleeve,
    "BTAL": sleeve_from(["BTAL"]),
    "GDX": sleeve_from(["GDX"]),
    "GLD": sleeve_from(["GLD"]),
    "TAIL": sleeve_from(["TAIL"]),
    "GLD/GDX 50/50": sleeve_from(["GLD", "GDX"]),
    "GLD/IEF/BTAL 40/30/30": sleeve_from(["GLD", "IEF", "BTAL"], [0.4, 0.3, 0.3]),
    "GLD/BTAL 50/50": sleeve_from(["GLD", "BTAL"]),
    "GLD/GDX/BTAL 40/30/30": sleeve_from(["GLD", "GDX", "BTAL"], [0.4, 0.3, 0.3]),
    "GDX/BTAL 50/50": sleeve_from(["GDX", "BTAL"]),
}

rows = [M(book, "-- shipped book (15% GLD/IEF) --", 0.15)]
print("bar: Sharpe >= 1.050 and Calmar >= 1.118\n")
for name, sl in CANDS.items():
    if sl is None:
        continue
    for w in (0.10, 0.15, 0.20, 0.25):
        r = (1 - w) * core + w * sl.reindex(core.index).fillna(0.0)
        rows.append(M(r, name, w))

df = pd.DataFrame(rows)
pd.set_option("display.width", 240, "display.max_columns", 30)
df["PASS"] = np.where((df["Sharpe"] >= 1.050) & (df["Calmar"].astype(float) >= 1.118), "Y", "")
print(df.to_string(index=False))
df.to_json(os.path.join(OUT, "hedge_sleeve_test.json"), orient="records", indent=1)

print("\ntop 10 by Calmar:")
print(df.sort_values("Calmar", ascending=False).head(10).to_string(index=False))

# what each candidate did during the book's own drawdowns
print("\nsleeve return during the book's drawdown episodes (annualised):")
for name, sl in CANDS.items():
    if sl is None:
        continue
    s = sl.reindex(core.index).fillna(0.0)
    print(f"  {name:<26} {s[in_dd].mean()*252*100:+7.1f}%/yr   "
          f"standalone {backtest.compute_metrics(s, RF, ANN)['cagr']*100:+6.1f}%/yr")
