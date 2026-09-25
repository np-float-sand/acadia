"""Selection book (TOP_HALF, capex_light+rev_accel) vs two candidate short legs, 2026-09-25.

    python -m research.xsec_alpha_probe.short_compare

Short legs = EW bottom quintile of the eligible universe, monthly, 10bp + 1%/yr borrow:
  S_capex : ranked by the frozen capex_light+rev_accel composite (tilt.short_leg)
  S_all14 : ranked by the equal-weight 14-signal composite (composites.py)
Books: TOP_HALF alone; TOP_HALF - 0.25x short; TOP_HALF - matched-GRID-beta short.
Monthly, before the vol-target / valuation / sleeve overlay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import composites as cp, data, evaluate as ev, run, signals as sg, tilt


def all14_short() -> pd.Series:
    px_d = data.prices(); univ = data.universe(); px_m = data.monthly_prices(px_d)
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    bm = data.monthly_prices(data.benchmarks())
    sigs = {}
    sigs.update(sg.fundamental_signals()); sigs.update(sg.price_signals(px_m))
    sigs.update(sg.power_signals(px_m, bm["SPY"]))
    comp = cp.composite(sigs, list(sigs), univ)
    return run.book_returns(-comp, fwd1, univ, None)["long"], comp


def main() -> None:
    br = tilt.book_returns()
    top, grid = br["TOP_HALF"], br["GRID"]
    s14, comp14 = all14_short()
    legs = {"S_capex": tilt.short_leg().reindex(br.index), "S_all14": s14.reindex(br.index)}
    b_top = np.polyfit(grid, top, 1)[0]
    rows, legrows = [], []
    books = {"TOP_HALF (selection, long only)": top}
    for name, s in legs.items():
        k = b_top / np.polyfit(grid, s, 1)[0]
        for kk, lab in ((0.25, "0.25x"), (k, f"{k:.2f}x matched-beta")):
            books[f"TOP_HALF - {lab} {name}"] = top - kk * s - kk * tilt.BORROW / 12
    for name, r in books.items():
        for w, (a, e) in run.WINDOWS.items():
            p = ev.perf(r.loc[a:e])
            rows.append({"book": name, "window": w, "SR": p["sharpe"], "ann": p["ann"],
                         "maxDD": p["maxdd"], "Calmar": tilt.calmar(r.loc[a:e])})
    for name, s in legs.items():
        for w, (a, e) in run.WINDOWS.items():
            ss, tt = s.loc[a:e], top.loc[a:e]
            dn = tt < 0
            legrows.append({"leg": name, "window": w, "leg ann (long)": ss.mean() * 12,
                            "short outright SR": ev.perf(-ss)["sharpe"],
                            "corr w/ TOP_HALF": ss.corr(tt),
                            "beta on TOP_HALF down months": np.polyfit(tt[dn], ss[dn], 1)[0],
                            "leg ann when TOP_HALF down": ss[dn].mean() * 12})
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(legrows).set_index(["leg", "window"]).round(2).to_string(), "\n")
    print(pd.DataFrame(rows).set_index(["book", "window"]).round(2).to_string(), "\n")
    # name overlap between the two short lists
    univ = data.universe()
    compc = tilt.composite(univ, tilt.book_membership(univ.index))
    ov = []
    for dt in univ.index[univ.index >= run.START]:
        a = (-compc.loc[dt].where(univ.loc[dt])).dropna(); b = (-comp14.loc[dt].where(univ.loc[dt])).dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        qa = set(a[a >= a.quantile(0.8)].index); qb = set(b[b >= b.quantile(0.8)].index)
        ov.append(len(qa & qb) / max(1, len(qa)))
    print(f"avg overlap of the two bottom-quintile lists: {np.mean(ov):.0%}")


if __name__ == "__main__":
    main()
