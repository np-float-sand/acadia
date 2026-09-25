"""Does the LONG side (TOP_HALF selection book) benefit from power-market signals? 2026-09-25.

    python -m research.xsec_alpha_probe.long_power

Pre-set: score = 0.5 x frozen composite (capex_light + rev_accel) + 0.5 x mean z of the chosen power
signals (same universe-or-book mask as tilt.composite). 9 variants: power_all (6), power_betas (3),
power_shocks (3), and each single power signal. Monthly TOP_HALF of the `screen_auto` book, before the
overlay. ADDS only if Sharpe AND Calmar beat the frozen TOP_HALF in BOTH 2019-22 and 2023-26.
"""
from __future__ import annotations

import pandas as pd

from research.xsec_alpha_probe import composites as cp, data, evaluate as ev, run, signals as sg, tilt


def main() -> pd.DataFrame:
    univ = data.universe()
    book = tilt.book_membership(univ.index)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(index=univ.index, columns=cols,
                                                                     fill_value=False)
    px_m = data.monthly_prices(data.prices())
    power = sg.power_signals(px_m, data.monthly_prices(data.benchmarks())["SPY"])
    z = {k: ev.prep(v.reindex(index=univ.index, columns=cols).where(mask)) for k, v in power.items()}
    base = tilt.composite(univ, book)
    groups = {"power_all": cp.POWER_BETAS + cp.POWER_SHOCKS, "power_betas": cp.POWER_BETAS,
              "power_shocks": cp.POWER_SHOCKS, **{k: [k] for k in cp.POWER_BETAS + cp.POWER_SHOCKS}}
    books = {"frozen (capex+rev)": tilt.book_returns()["TOP_HALF"]}
    grid = tilt.book_returns()["GRID"]
    for g, members in groups.items():
        pz = pd.concat([z[m] for m in members], keys=range(len(members))).groupby(level=1).mean()
        score = base.mul(0.5).add(pz.mul(0.5), fill_value=0)
        books[f"+ {g}"] = tilt.book_returns(score=score)["TOP_HALF"]
    rows = []
    ref = books["frozen (capex+rev)"]
    for name, r in books.items():
        row = {"book": name}
        for w, (a, e) in run.WINDOWS.items():
            p = ev.perf(r.loc[a:e])
            row[f"SR {w}"], row[f"Calmar {w}"] = p["sharpe"], tilt.calmar(r.loc[a:e])
        hedged, _ = tilt.grid_hedge(r, grid.reindex(r.index))
        row["GRID-hedged SR"] = ev.perf(hedged)["sharpe"]
        row["active vs frozen, %/yr"] = (r - ref).mean() * 12 * 100
        row["active t (NW1)"] = ev.nw_tstat((r - ref).dropna(), 1) if name != "frozen (capex+rev)" else float("nan")
        row["ADDS"] = name != "frozen (capex+rev)" and all(
            row[f"{m} {w}"] > (ev.perf(ref.loc[a:e])["sharpe"] if m == "SR" else tilt.calmar(ref.loc[a:e]))
            for m in ("SR", "Calmar") for w, (a, e) in run.WINDOWS.items() if w != "full")
        rows.append(row)
    res = pd.DataFrame(rows).set_index("book")
    res.to_csv(data.CACHE / "results_long_power.csv")
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(main().round(2).to_string())
