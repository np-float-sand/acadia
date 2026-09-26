"""Robustness on the surviving hedge sleeves, before recommending any of them.

The screen found BTAL (anti-beta) is the only candidate with genuinely negative
down-day beta AND negative standalone drift -- i.e. it behaves like insurance:
it costs carry and pays out in drawdowns. GLD and GDX look better in the table but
both had exceptional 2019-26 runs, so the obvious trap is fitting a hedge to a gold
bull market. This separates the two.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from electrification_strategy import backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
RF, ANN = config.RF, config.ANN
d = pd.read_pickle(os.path.join(OUT, "_build_cache.pkl"))
book, core = d["book"], d["core"]
px = pd.read_parquet(os.path.join(OUT, "_hedge_px.parquet"))
rets = px.reindex(core.index).pct_change().fillna(0.0)
spy = d["bench"]["SPY"].reindex(core.index).fillna(0.0)
w_s = config.HEDGE_SLEEVE_WEIGHT
shipped_sleeve = (book - (1 - w_s) * core) / w_s

EPISODES = {"COVID 2020": ("2020-02-01", "2020-06-30"),
            "2022 rate shock": ("2022-01-01", "2022-12-31"),
            "2025 tariffs": ("2025-02-01", "2025-07-31")}
SUBS = {"2019-2022": ("2019-01-01", "2022-12-31"), "2023-2026": ("2023-01-01", "2026-08-31")}


def mk(tickers, weights=None):
    tk = [t for t in tickers if t in rets.columns]
    w = np.array(weights if weights else [1 / len(tk)] * len(tk))
    return (rets[tk] * w).sum(axis=1)


BOOKS = {
    "shipped: 15% GLD/IEF": (shipped_sleeve, 0.15),
    "drop IEF: 15% GLD": (mk(["GLD"]), 0.15),
    "20% GLD": (mk(["GLD"]), 0.20),
    "20% BTAL": (mk(["BTAL"]), 0.20),
    "20% GLD/BTAL 50/50": (mk(["GLD", "BTAL"]), 0.20),
    "25% GLD/BTAL 50/50": (mk(["GLD", "BTAL"]), 0.25),
    "20% GLD/GDX/BTAL 40/30/30": (mk(["GLD", "GDX", "BTAL"], [.4, .3, .3]), 0.20),
    "25% GLD/GDX/BTAL 40/30/30": (mk(["GLD", "GDX", "BTAL"], [.4, .3, .3]), 0.25),
}


def compose(sleeve, w):
    return (1 - w) * core + w * sleeve.reindex(core.index).fillna(0.0)


rows = []
for name, (sl, w) in BOOKS.items():
    r = compose(sl, w)
    x = backtest.compute_metrics(r.dropna(), RF, ANN)
    rec = {"book": name, "Sharpe": x["sharpe"],
           "Calmar": round(x["cagr"] / abs(x["max_dd"]), 3),
           "CAGR": f"{x['cagr']*100:.1f}%", "MaxDD": f"{x['max_dd']*100:.1f}%",
           "beta": round(float(np.cov(r.dropna(), spy.reindex(r.dropna().index))[0, 1]
                              / spy.var()), 2)}
    for k, (a, z) in SUBS.items():
        rec[f"Sh {k}"] = backtest.compute_metrics(r.loc[a:z], RF, ANN)["sharpe"]
    for k, (a, z) in EPISODES.items():
        seg = r.loc[a:z]
        c = (1 + seg).cumprod()
        rec[k] = f"{(c/c.cummax()-1).min()*100:.1f}%"
    rows.append(rec)

df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print("Sharpe/Calmar bar: 1.050 / 1.118\n")
print(df.to_string(index=False))

print("\n--- the honest question: how much of each sleeve's help is the gold bull market? ---")
for t in ["GLD", "GDX", "BTAL", "IEF"]:
    if t not in rets.columns:
        continue
    s = rets[t]
    full = backtest.compute_metrics(s, RF, ANN)["cagr"]
    a = backtest.compute_metrics(s.loc["2019":"2022"], RF, ANN)["cagr"]
    b = backtest.compute_metrics(s.loc["2023":], RF, ANN)["cagr"]
    print(f"  {t:<5} standalone  full {full*100:+6.1f}%/yr   2019-22 {a*100:+6.1f}%   2023-26 {b*100:+6.1f}%")

print("\n--- stress: what if the hedge sleeve returns ZERO (gold rally does not repeat)? ---")
for name, (sl, w) in BOOKS.items():
    zero = pd.Series(0.0, index=core.index)
    r = compose(zero, w)
    x = backtest.compute_metrics(r.dropna(), RF, ANN)
    r2 = compose(sl, w)
    x2 = backtest.compute_metrics(r2.dropna(), RF, ANN)
    print(f"  {name:<28} actual Sharpe {x2['sharpe']:.3f} -> with a dead sleeve {x['sharpe']:.3f}"
          f"   (Calmar {round(x2['cagr']/abs(x2['max_dd'])   ,3)} -> "
          f"{round(x['cagr']/abs(x['max_dd']),3)})")
