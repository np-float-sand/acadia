"""Selection book (TOP_HALF) hedged with the 14-signal bottom-quintile short vs GRID / PAVE / XLI.

    python -m research.xsec_alpha_probe.hedge_compare

Monthly, pre-overlay. Each hedge sized three ways: fixed 0.25x; "matched" = TOP_HALF's full-sample
OLS beta to the hedge (IN-SAMPLE, look-ahead); "trailing" = beta over the prior TRAIL months of
completed returns only (no look-ahead; 1.0x prior for the first 12 months). Borrow: ETFs
0.5%/yr (as in the selection page), single-name short 1%/yr.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import data, evaluate as ev, run, short_compare, tilt

ETF_BORROW, NAME_BORROW = 0.005, tilt.BORROW
TRAIL, TRAIL_MIN, PRIOR = 24, 12, 1.0  # 1.0x until 12 months of history exist (book starts 2019-01)


def main() -> None:
    br = tilt.book_returns()
    top = br["TOP_HALF"]
    bm = data.monthly_prices(data.benchmarks()).pct_change().shift(-1).reindex(br.index)
    s14, _ = short_compare.all14_short()
    hedges = {"GRID": (bm["GRID"], ETF_BORROW), "PAVE": (bm["PAVE"], ETF_BORROW),
              "XLI": (bm["XLI"], ETF_BORROW), "S_all14": (s14.reindex(br.index), NAME_BORROW)}
    books = {"TOP_HALF long only": top}
    info = []
    for name, (h, borrow) in hedges.items():
        ev_top, ev_h = top.loc[run.START:run.END], h.loc[run.START:run.END]
        k = np.polyfit(ev_h, ev_top, 1)[0]
        tt = top < 0
        info.append({"hedge": name, "corr w/ book": ev_h.corr(ev_top), "matched size": k,
                     "hedge ann": ev_h.mean() * 12,
                     "down-month beta (full/19-22/23-26)": " / ".join(
                         f"{np.polyfit(top.loc[a:e][tt.loc[a:e]], h.loc[a:e][tt.loc[a:e]], 1)[0]:.2f}"
                         for a, e in run.WINDOWS.values())})
        for kk, lab in ((0.25, "0.25x"), (k, "matched")):
            books[f"- {lab} {name}"] = top - kk * h - kk * borrow / 12
        # row t's return is realised at t+1, so the beta used at t may only see rows <= t-1
        kt = (top.rolling(TRAIL, min_periods=TRAIL_MIN).cov(h)
              / h.rolling(TRAIL, min_periods=TRAIL_MIN).var()).shift(1).fillna(PRIOR)
        books[f"- trailing {name}"] = top - kt * h - kt * borrow / 12
        info[-1]["trailing size (mean 19-26)"] = kt.loc[run.START:run.END].mean()
    rows = []
    for name, r in books.items():
        row = {"book": name}
        for w, (a, e) in run.WINDOWS.items():
            p = ev.perf(r.loc[a:e])
            row[f"SR {w}"] = p["sharpe"]; row[f"Calmar {w}"] = tilt.calmar(r.loc[a:e])
            row[f"maxDD {w}"] = p["maxdd"]; row[f"ann {w}"] = p["ann"]
        rows.append(row)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(info).set_index("hedge").round(2).to_string(), "\n")
    res = pd.DataFrame(rows).set_index("book")
    cols = [f"{m} {w}" for m in ("SR", "Calmar", "maxDD", "ann") for w in run.WINDOWS]
    print(res[cols].round(2).to_string())
    res.to_csv(data.CACHE / "results_hedge_compare.csv")


if __name__ == "__main__":
    main()
