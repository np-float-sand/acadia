"""Plan step 1: tilt the `screen_auto` book by the frozen composite.

    python -m research.xsec_alpha_probe.tilt

Composite FROZEN 2026-09-24: mean of prep(capex_light) and prep(rev_accel), each
winsorized 5/95 + z-scored across the month's eligible universe plus the book's own
members. Nothing about it is tuned here.

Books (monthly, month-end `screen_auto` membership, 10bp one-way on turnover):
  EW        equal weight (baseline; the shipped book before vol-target/valuation)
  TILT_0.5  w ~ max(0, 1 + 0.5 z)   <- primary
  TILT_1    w ~ max(0, 1 + 1.0 z)
  TOP_HALF  equal weight, top half of the book by composite

Bar (fixed 2026-09-24, before results): the tilt ADDS if the active return
(TILT_0.5 - EW) has Newey-West t >= 1.5 pooled with a positive mean in both
2019-22 and 2023-26, AND TILT_0.5's GRID-hedged Sharpe beats EW's in both halves.
GRID hedge = r - beta * GRID, beta = full-sample OLS on monthly returns (matched beta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from electrification_strategy import auto_screen
from research.xsec_alpha_probe import data, evaluate as ev, run, signals as sg

TILTS = {"TILT_0.5": 0.5, "TILT_1": 1.0}


def composite(univ: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    fund = sg.fundamental_signals()
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(index=univ.index, columns=cols,
                                                                     fill_value=False)

    def z(x):
        return ev.prep(x.reindex(index=univ.index, columns=cols).where(mask))

    return z(fund["capex_light"]).add(z(fund["rev_accel"]), fill_value=0) / 2


def book_membership(index: pd.DatetimeIndex) -> pd.DataFrame:
    m = auto_screen.load_membership().astype(bool)
    return m.reindex(index, method="ffill").fillna(False).astype(bool)


def weights(members: list[str], z: pd.Series) -> dict[str, pd.Series]:
    zz = z.reindex(members).fillna(0.0)
    out = {"EW": pd.Series(1.0, index=members)}
    for name, k in TILTS.items():
        out[name] = (1 + k * zz).clip(lower=0)
    ranked = zz.sort_values(ascending=False)
    out["TOP_HALF"] = pd.Series(1.0, index=ranked.index[: max(1, len(ranked) // 2)])
    return {k: v / v.sum() for k, v in out.items() if v.sum() > 0}


def book_returns() -> pd.DataFrame:
    px_m = data.monthly_prices(data.prices())
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    univ = data.universe()
    book = book_membership(univ.index)
    comp = composite(univ, book)
    rows, prev = {}, {}
    for dt in univ.index:
        if dt not in fwd1.index:
            continue
        r = fwd1.loc[dt]
        members = [t for t in book.columns[book.loc[dt]] if pd.notna(r.get(t))]
        if len(members) < 3:
            continue
        row = {"n": len(members)}
        for name, w in weights(members, comp.loc[dt]).items():
            p = prev.get(name)
            tv = w.abs().sum() if p is None else w.sub(p, fill_value=0).abs().sum()
            prev[name] = w
            row[name] = float((w * r.reindex(w.index)).sum()) - tv * run.COST
        rows[dt] = row
    out = pd.DataFrame(rows).T
    grid = px_m["GRID"].pct_change().shift(-1)
    out["GRID"] = grid.reindex(out.index)
    return out.loc[run.START:run.END].dropna(subset=["EW", "GRID"])


def report(br: pd.DataFrame) -> pd.DataFrame:
    books = ["EW", *TILTS, "TOP_HALF"]
    rows = []
    for b in books:
        beta = np.polyfit(br["GRID"], br[b], 1)[0]
        hedged = br[b] - beta * br["GRID"]
        active = br[b] - br["EW"]
        for w, (a, e) in run.WINDOWS.items():
            act = active.loc[a:e]
            rows.append({"book": b, "window": w, "beta_GRID": beta,
                         "SR": ev.perf(br[b].loc[a:e])["sharpe"],
                         "ann": ev.perf(br[b].loc[a:e])["ann"],
                         "GRID-hedged SR": ev.perf(hedged.loc[a:e])["sharpe"],
                         "active ann": act.mean() * 12,
                         "active t (NW1)": ev.nw_tstat(act, 1) if b != "EW" else np.nan})
    return pd.DataFrame(rows).set_index(["book", "window"])


def verdict(rep: pd.DataFrame) -> str:
    t = rep.loc[("TILT_0.5", "full"), "active t (NW1)"]
    halves_pos = all(rep.loc[("TILT_0.5", w), "active ann"] > 0 for w in ("2019-22", "2023-26"))
    hedged_beats = all(rep.loc[("TILT_0.5", w), "GRID-hedged SR"] > rep.loc[("EW", w), "GRID-hedged SR"]
                       for w in ("2019-22", "2023-26"))
    ok = t >= ev.BAR_T and halves_pos and hedged_beats
    return (f"{'PASS' if ok else 'FAIL'} — active t {t:.2f} (need >= {ev.BAR_T}), "
            f"active > 0 both halves: {halves_pos}, hedged SR beats EW both halves: {hedged_beats}")


if __name__ == "__main__":
    br = book_returns()
    br.to_csv(data.CACHE / "tilt_returns.csv")
    rep = report(br)
    rep.to_csv(data.CACHE / "results_tilt.csv")
    print(f"months {len(br)}, names/month {br['n'].min():.0f}-{br['n'].max():.0f}")
    print(rep.round(2).to_string())
    print(verdict(rep))
