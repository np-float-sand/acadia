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

Alpha-vs-beta label (fixed 2026-09-24, before results): a book is ALPHA if its
GRID-hedged return (= Jensen alpha vs GRID) has Newey-West t >= 1.5 pooled and a
positive mean in both halves; otherwise BETA (the theme). Calmar = ann / |maxDD|.

Step 2 (`--hedges`): hedge the TOP_HALF book three ways -- GRID short at matched
beta (the recommended policy), short the bottom quintile of the eligible universe
by composite (the capex-heavy avoid list as a short), and both 50/50. Single-name
shorts pay BORROW per year on the short notional on top of trading costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from electrification_strategy import auto_screen
from research.xsec_alpha_probe import data, evaluate as ev, run, signals as sg

TILTS = {"TILT_0.5": 0.5, "TILT_1": 1.0}
BORROW = 0.01          # 1%/yr general-collateral-plus borrow on single-name shorts


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


def book_returns(score: pd.DataFrame | None = None, start: str = run.START) -> pd.DataFrame:
    """Monthly book returns; `score` replaces the frozen composite (for later steps).
    `start` earlier than run.START gives warm-up history for trailing estimates."""
    px_m = data.monthly_prices(data.prices())
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    univ = data.universe()
    book = book_membership(univ.index)
    comp = composite(univ, book) if score is None else score.reindex(univ.index)
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
    return out.loc[start:run.END].dropna(subset=["EW", "GRID"])


def calmar(r: pd.Series) -> float:
    p = ev.perf(r)
    return p["ann"] / abs(p["maxdd"]) if p["maxdd"] < 0 else np.nan


def label(hedged: pd.Series) -> tuple[str, float]:
    t = ev.nw_tstat(hedged.loc[run.START:run.END], 1)
    halves = all(hedged.loc[a:e].mean() > 0 for w, (a, e) in run.WINDOWS.items() if w != "full")
    return ("ALPHA" if t >= ev.BAR_T and halves else "BETA"), t


def stats(r: pd.Series, hedged: pd.Series) -> list[dict]:
    lab, t = label(hedged)
    rows = []
    for w, (a, e) in run.WINDOWS.items():
        raw, h = ev.perf(r.loc[a:e]), ev.perf(hedged.loc[a:e])
        rows.append({"window": w, "SR": raw["sharpe"], "ann": raw["ann"], "maxDD": raw["maxdd"],
                     "Calmar": calmar(r.loc[a:e]), "hedged SR": h["sharpe"],
                     "alpha ann": h["ann"], "hedged Calmar": calmar(hedged.loc[a:e]),
                     "alpha t (NW1)": t if w == "full" else ev.nw_tstat(hedged.loc[a:e], 1),
                     "label": lab if w == "full" else ""})
    return rows


def grid_hedge(r: pd.Series, grid: pd.Series) -> tuple[pd.Series, float]:
    beta = np.polyfit(grid, r, 1)[0]
    return r - beta * grid, beta


def report(br: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in ["EW", *TILTS, "TOP_HALF"]:
        hedged, beta = grid_hedge(br[b], br["GRID"])
        active = br[b] - br["EW"]
        for row in stats(br[b], hedged):
            a, e = run.WINDOWS[row["window"]]
            rows.append({"book": b, "beta_GRID": beta, **row,
                         "active ann": active.loc[a:e].mean() * 12,
                         "active t (NW1)": ev.nw_tstat(active.loc[a:e], 1) if b != "EW" else np.nan})
    return pd.DataFrame(rows).set_index(["book", "window"])


def short_leg() -> pd.Series:
    """EW bottom quintile of the eligible universe by the frozen composite, net of costs."""
    px_m = data.monthly_prices(data.prices())
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    univ = data.universe()
    comp = composite(univ, book_membership(univ.index))
    bot = run.book_returns(-comp, fwd1, univ, None)
    return bot["long"]


def hedges(br: pd.DataFrame) -> pd.DataFrame:
    top, grid = br["TOP_HALF"], br["GRID"]
    heavy = short_leg().reindex(br.index)
    grid_h, _ = grid_hedge(top, grid)
    # short the capex-heavy bottom quintile at the notional that matches its GRID beta to top's
    b_top, b_heavy = np.polyfit(grid, top, 1)[0], np.polyfit(grid, heavy, 1)[0]
    k = b_top / b_heavy
    heavy_h = top - k * heavy - k * BORROW / 12
    both = 0.5 * grid_h + 0.5 * heavy_h
    rows = []
    for name, h in {"TOP_HALF - GRID (matched beta)": grid_h,
                    f"TOP_HALF - {k:.2f}x capex-heavy quintile": heavy_h,
                    "50/50 of the two": both}.items():
        for row in stats(h, h):
            rows.append({"hedge": name, **{c: row[c] for c in
                         ("window", "SR", "ann", "maxDD", "Calmar", "alpha t (NW1)", "label")}})
    return pd.DataFrame(rows).set_index(["hedge", "window"])


def verdict(rep: pd.DataFrame) -> str:
    t = rep.loc[("TILT_0.5", "full"), "active t (NW1)"]
    halves_pos = all(rep.loc[("TILT_0.5", w), "active ann"] > 0 for w in ("2019-22", "2023-26"))
    hedged_beats = all(rep.loc[("TILT_0.5", w), "hedged SR"] > rep.loc[("EW", w), "hedged SR"]
                       for w in ("2019-22", "2023-26"))
    ok = t >= ev.BAR_T and halves_pos and hedged_beats
    return (f"{'PASS' if ok else 'FAIL'} — active t {t:.2f} (need >= {ev.BAR_T}), "
            f"active > 0 both halves: {halves_pos}, hedged SR beats EW both halves: {hedged_beats}")


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    br = book_returns()
    br.to_csv(data.CACHE / "tilt_returns.csv")
    if "--hedges" in sys.argv:
        h = hedges(br)
        h.to_csv(data.CACHE / "results_hedges.csv")
        print(h.round(2).to_string())
    else:
        rep = report(br)
        rep.to_csv(data.CACHE / "results_tilt.csv")
        print(f"months {len(br)}, names/month {br['n'].min():.0f}-{br['n'].max():.0f}")
        print(rep.round(2).to_string())
        print(verdict(rep))
