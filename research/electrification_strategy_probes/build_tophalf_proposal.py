"""Figures + investor stats for the SELECTION proposal (top half by composite).

    python research/electrification_strategy_probes/build_tophalf_proposal.py

The book: the published `screen_auto` A1+A4 membership (same as
`build_a1a4_proposal.py`), but each rebalance holds only the half of the eligible
names with the highest frozen composite score (mean z of capex-light and revenue
acceleration, `research/xsec_alpha_probe`), equal weight, then the same shipped
overlay (20% vol target + valuation de-lever + 15% GLD/IEF sleeve).

The hedged version shorts an index ETF against the overlaid book at a TRAILING
252-day beta (lagged a day; estimated on the same book run from WARMUP_START so
2019 has a real trailing year -- nothing back-filled, no look-ahead), paying
HEDGE_BORROW a year on the short notional and 10bp on every change in it.

Single-name short option: an equal-weight basket of the BOTTOM QUINTILE of the eligible
universe by the 14-signal equal-weight composite (`research/xsec_alpha_probe/composites.py`),
re-formed at each month-end from filings/prices known that day, held (weights drifting) through
the next month-end. Shorted two ways: a fixed 0.25x, and at the same trailing 252-day beta as
the ETFs. Pays NAME_BORROW a year, 10bp on basket turnover and on every change in hedge size.

The trailing year needs a real 2018 book. The published membership starts 2019-01, so the warm-up
books use `_warmup_members.pkl` (the identical rule run over 2018 by `build_warmup_members.py`)
for the months before 2019 and the published membership from 2019 on. Without it the 2018 book is
empty and every hedge ratio ramps up from ~0 through 2019.

Stats and figures reuse `build_a1a4_proposal` so every number is computed the
same way as the published proposal. Writes output_html/img/tophalf/.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_a1a4_proposal as a1  # noqa: E402  (sets the dark matplotlib style)
from electrification_strategy import backtest, config, hedge_overlay  # noqa: E402
from electrification_strategy.fred import fetch_series  # noqa: E402
from grid_equipment_basket.basket import simulate_basket  # noqa: E402
from grid_equipment_basket.data.prices import fetch_prices  # noqa: E402
from research.xsec_alpha_probe import data, realbook, short_compare, tilt  # noqa: E402

plt, mdates = a1.plt, a1.mdates

# Light mode: a1 sets a dark style at import; override it and the colours its figure
# functions read at call time. Series hues = dataviz reference palette, light steps,
# slots 1-3 (validated: all checks pass; aqua needs relief -> legends + direct labels).
a1.SURFACE = "#fcfcfb"
a1.C_BOOK, a1.C_SPY, a1.C_GRID, a1.C_NEG = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
a1.INK, a1.SOFT, a1.FAINT, a1.LINE = "#0b0b0b", "#52514e", "#8b8a85", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": a1.SURFACE, "axes.facecolor": a1.SURFACE, "savefig.facecolor": a1.SURFACE,
    "axes.edgecolor": a1.LINE, "axes.labelcolor": a1.SOFT, "text.color": a1.INK,
    "xtick.color": a1.SOFT, "ytick.color": a1.SOFT, "grid.color": a1.LINE,
})
OUT = os.path.join(ROOT, "output_html", "img", "tophalf")
os.makedirs(OUT, exist_ok=True)
a1.OUT = OUT                                   # reuse a1's figure code, write here
START, END = a1.START, a1.END
BENCH = ["SPY", "GRID", "PAVE", "XLU", "VOLT", "XLI"]
HEDGES = {"GRID": {"GRID": 1.0}, "PAVE": {"PAVE": 1.0}, "XLI": {"XLI": 1.0},
          "GRID+PAVE": {"GRID": 0.5, "PAVE": 0.5},
          "GRID+PAVE+XLI": {"GRID": 1 / 3, "PAVE": 1 / 3, "XLI": 1 / 3}}   # daily-rebalanced blends
HEDGE_BORROW = 0.005                           # assumed borrow on a large ETF short, a year
HEDGE_COST_BP = 10
NAME_BORROW = 0.01                             # assumed general-collateral-plus borrow on single names
NAME_Q = 0.2                                   # short the bottom quintile
FIXED_K = 0.25                                 # fixed-size sleeve: $0.25 short per $1 of book
BETA_WINDOW = 252
WARMUP_START = "2017-10-01"                    # history for the first trailing hedge beta
WINDOWS = {"full": (START, END), "2019-22": ("2019-01-01", "2022-12-31"),
           "2022+": ("2022-01-01", END), "2023+": ("2023-01-01", END)}


# ------------------------------------------------------------------ data ----
def load():
    pub = pickle.load(open(os.path.join(ROOT, "output_html", "img", "a1a4", "_build_cache.pkl"), "rb"))
    members = pub["members"]
    px = data.prices().copy()
    extra = sorted(set(BENCH) | set(config.HEDGE_SLEEVE_TICKERS) | set(config.HEDGE_SHORT_TICKERS))
    spx = fetch_prices(extra, "2015-01-01", END)
    for t in spx.columns:
        if t not in px.columns:
            px[t] = spx[t].reindex(px.index)
    univ = data.universe()
    comp = tilt.composite(univ, tilt.book_membership(univ.index))
    return pub, members, px, comp


def build_book(spec, members, px, spy_ret, dfii10, start=START):
    cand = px[[c for c in members.columns if c in px.columns]]
    br = simulate_basket(cand, start, END, config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT,
                         realbook.target_fn(spec, members, px))
    raw = br.returns
    vol_only = backtest.compose(raw, spy_ret, use_valuation=False)
    core = backtest.compose(raw, spy_ret, use_valuation=True)
    rising = hedge_overlay.real_yield_rising_mask(dfii10, core.index)
    book = hedge_overlay.apply_hedge(core, hedge_overlay.sleeve_returns(px),
                                     hedge_overlay.short_returns(px), rising, True, False)
    return {"raw": raw, "vol_only": vol_only, "core": core, "book": book,
            "weights": br.weights, "held": (br.weights > 0).sum(axis=1)}


def hedge(book: pd.Series, h: pd.Series, beta_src: pd.Series, borrow: float = HEDGE_BORROW,
          unit_cost: pd.Series | None = None, fixed: float | None = None) -> tuple[pd.Series, pd.Series]:
    """book minus trailing-beta x h, net of borrow and hedge trading. Returns (net, beta).

    No look-ahead: the hedge ratio on day t is the beta of `beta_src` (the same book
    run from WARMUP_START, so 2019 already has a full trailing year) on h over the
    BETA_WINDOW days ending t-1. Nothing is back-filled.
    `fixed` replaces the beta with a constant size; `unit_cost` is a per-$1-of-hedge daily cost
    (the single-name basket's own rebalancing) charged at the hedge size."""
    if fixed is None:
        hs = h.reindex(beta_src.index)
        beta = (beta_src.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(hs)
                / hs.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()).shift(1).clip(0, 2)
        beta = beta.reindex(book.index)
    else:
        beta = pd.Series(fixed, index=book.index)
    assert beta.loc[START:END].notna().all(), "hedge beta has no trailing year at the start"
    h = h.reindex(book.index)
    cost = beta * borrow / config.ANN + beta.diff().abs().fillna(0) * HEDGE_COST_BP / 1e4
    if unit_cost is not None:
        cost = cost + beta * unit_cost.reindex(book.index).fillna(0)
    return (book - beta * h - cost).dropna(), beta


def name_short_basket(px: pd.DataFrame, start: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Daily return of the bottom-quintile basket (long side, i.e. what is shorted), its daily
    rebalancing cost per $1, and the number of names held.

    At each month-end the 14-signal composite is ranked across that month's eligible universe
    (names priced that day); the bottom NAME_Q is bought equal-weight on the next trading day
    and held with weights drifting until the next month-end. No future information: the score
    uses filings and prices known at the month-end, and membership never looks at later prices."""
    _, comp = short_compare.all14_short()
    univ = data.universe()
    rets = px.pct_change(fill_method=None)
    days = rets.loc[start:END].index
    out, cost, held = pd.Series(0.0, index=days), pd.Series(0.0, index=days), pd.Series(0, index=days)
    w_prev = pd.Series(dtype=float)
    ends = [d for d in comp.index if pd.Timestamp(start) - pd.offsets.MonthEnd(1) <= d <= pd.Timestamp(END)]
    for i, me in enumerate(ends):
        seg = days[(days > me) & (days <= (ends[i + 1] if i + 1 < len(ends) else pd.Timestamp(END)))]
        if seg.empty or me not in univ.index:
            continue
        priced = px.loc[:me].iloc[-1].notna()
        sc = comp.loc[me].where(univ.loc[me].reindex(comp.columns, fill_value=False)
                                & priced.reindex(comp.columns, fill_value=False)).dropna()
        if len(sc) < 10:
            continue
        names = sc[sc <= sc.quantile(NAME_Q)].index
        w = pd.Series(1.0 / len(names), index=names)
        cost.loc[seg[0]] = w.sub(w_prev, fill_value=0).abs().sum() * HEDGE_COST_BP / 1e4
        val = w.copy()
        for d in seg:
            r = rets.loc[d, names].fillna(0.0)
            new = val * (1 + r)
            out.loc[d] = new.sum() / val.sum() - 1
            val = new
            held.loc[d] = len(names)
        w_prev = val / val.sum()
    return out, cost, held


def alpha(r: pd.Series, h: pd.Series) -> dict:
    """Jensen alpha of r on h (daily OLS), annualised, Newey-West t (5 lags)."""
    d = pd.concat([r.rename("r"), h.rename("h")], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d["h"].values])
    coef, *_ = np.linalg.lstsq(X, d["r"].values, rcond=None)
    e = d["r"].values - X @ coef
    n, L = len(d), 5
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        G = Xe[lag:].T @ Xe[:-lag]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return {"alpha": float(coef[0] * config.ANN), "beta": float(coef[1]), "t": float(coef[0] / se[0])}


def names_turnover(weights: pd.DataFrame, exposure: pd.Series) -> float:
    """Annual two-way turnover from changing names, scaled by the overlay's exposure."""
    w = weights.loc[START:END]
    chg = w.diff().abs().sum(axis=1)
    chg = chg[chg > 1e-9].iloc[1:]
    years = (w.index[-1] - w.index[0]).days / 365.25
    return float((chg * exposure.reindex(chg.index).fillna(1.0)).sum() * (1 - config.HEDGE_SLEEVE_WEIGHT) / years)


def overlay_turnover(d: dict) -> dict:
    raw, core, book = d["raw"], d["core"], d["book"]
    w_s = config.HEDGE_SLEEVE_WEIGHT
    rf = config.RF / config.ANN
    denom = (raw - rf).where((raw - rf).abs() > 1e-9)
    exp = ((core - rf) / denom).clip(0, config.VOL_MAX_LEVERAGE).ffill().fillna(1.0)
    years = (raw.index[-1] - raw.index[0]).days / 365.25
    sleeve = (book - (1 - w_s) * core) / w_s
    turn = {"names": names_turnover(d["weights"], exp),
            "resize": float(exp.diff().abs().sum()) * (1 - w_s) / years,
            "sleeve": float((w_s * (1 - w_s) * (sleeve - exp * raw).abs()).sum()) / years}
    turn["total"] = sum(turn.values())
    return turn


# ------------------------------------------------------------- figures ----
def fig_selection(top, ew, bench):
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ends = []
    for label, r, c, lw in [("Selection book", top, a1.C_BOOK, 2.3),
                            ("Published book (all names)", ew, a1.FAINT, 1.7),
                            ("GRID ETF", bench["GRID"], a1.C_GRID, 1.5)]:
        cv = a1.curve(r.loc[START:END])
        ax.plot(cv.index, cv.values, color=c, lw=lw, label=label)
        ends.append([float(cv.iloc[-1]), cv.index[-1], c, float(cv.iloc[-1])])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                 # push near-equal endpoints apart (log space)
        if np.log10(ends[i][0]) - np.log10(ends[i - 1][0]) < 0.035:
            ends[i][0] = 10 ** (np.log10(ends[i - 1][0]) + 0.035)
    for ypos, xend, c, val in ends:
        ax.annotate(f"  {val:.2f}x", (xend, ypos), color=a1.INK, fontsize=9.5, va="center", weight="bold")
    ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    ax.text(pd.Timestamp("2023-02-01"), ax.get_ylim()[1] * 0.97, "grid tight (2023 on)",
            color=a1.SOFT, fontsize=9, va="top")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4, 6])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel("growth of $1 (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(cv.index[0], cv.index[-1] + pd.Timedelta(days=330))
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=a1.INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/selection.png", bbox_inches="tight")
    plt.close(fig)


def fig_hedged(hedged_top, hedged_ew):
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for label, r, c, lw in [("Selection book, short GRID", hedged_top, a1.C_BOOK, 2.3),
                            ("Published book, short GRID", hedged_ew, a1.FAINT, 1.7)]:
        cv = a1.curve(r.loc[START:END])
        ax.plot(cv.index, cv.values, color=c, lw=lw, label=label)
        ax.annotate(f"  {cv.iloc[-1]:.2f}x", (cv.index[-1], cv.iloc[-1]), color=a1.INK, fontsize=9.5,
                    va="center", weight="bold")
    ax.axhline(1.0, color=a1.FAINT, lw=1)
    ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    ax.set_ylabel("growth of $1, theme hedged out")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(cv.index[0], cv.index[-1] + pd.Timedelta(days=330))
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=a1.INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hedged.png", bbox_inches="tight")
    plt.close(fig)


def fig_short_options(series: dict, betas: dict):
    """Growth, drawdown and hedge size of each hedged version; stats in the legend."""
    fig, (a1x, a2x, a3x) = plt.subplots(3, 1, figsize=(8.4, 9.2), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.6, 1.4]})
    cols = {"short PAVE": a1.C_SPY, "short GRID": a1.C_GRID, "short XLI": a1.FAINT,
            "short single names, beta-sized": a1.C_BOOK, "short single names, 0.25x": "#7b4fd6"}
    for label, r in series.items():
        r = r.loc[START:END]
        full, t23 = a1.stats(r), a1.stats(r.loc["2023-01-01":END])
        lab = (f"{label}   Sharpe {full['sharpe']:.2f} · Calmar {full['calmar']:.2f} · "
               f"maxDD {full['maxdd']:.0%}   |   2023+: Sharpe {t23['sharpe']:.2f} · Calmar {t23['calmar']:.2f}")
        cv = a1.curve(r)
        lw = 2.3 if "single" in label else 1.5
        a1x.plot(cv.index, cv.values, color=cols[label], lw=lw, label=lab)
        a1x.annotate(f"  {cv.iloc[-1]:.2f}x", (cv.index[-1], cv.iloc[-1]), color=a1.INK, fontsize=8.5,
                     va="center", weight="bold")
        dd = cv / cv.cummax() - 1
        a2x.plot(dd.index, dd.values * 100, color=cols[label], lw=lw * 0.7)
        if label in betas:
            b = betas[label].loc[START:END]
            a3x.plot(b.index, b.values, color=cols[label], lw=lw * 0.7)
    for ax in (a1x, a2x, a3x):
        ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    a1x.axhline(1.0, color=a1.FAINT, lw=1)
    a1x.set_ylabel(r"growth of \$1, hedged")
    a1x.legend(loc="upper left", fontsize=7.4, labelcolor=a1.INK, frameon=False)
    a2x.set_ylabel("drawdown, %")
    a3x.set_ylabel(r"\$ short per \$1")
    a3x.xaxis.set_major_locator(mdates.YearLocator())
    a3x.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a3x.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=330))
    fig.tight_layout()
    fig.savefig(f"{OUT}/short_options.png", bbox_inches="tight")
    plt.close(fig)


def fig_short_leg(book: pd.Series, basket: pd.Series, pave: pd.Series, held: pd.Series):
    """What is being shorted: the bottom-quintile basket vs PAVE vs the book, with stats."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    for label, r, c in [("Selection book", book, a1.C_BOOK),
                        ("PAVE ETF", pave, a1.C_SPY),
                        ("Bottom-quintile basket (the single-name short)", basket, "#7b4fd6")]:
        r = r.loc[START:END].dropna()
        st = a1.stats(r)
        cv = a1.curve(r)
        ax.plot(cv.index, cv.values, color=c, lw=2.0,
                label=f"{label}   {st['cagr']:.1%}/yr · Sharpe {st['sharpe']:.2f} · maxDD {st['maxdd']:.0%}")
        ax.annotate(f"  {cv.iloc[-1]:.2f}x", (cv.index[-1], cv.iloc[-1]), color=a1.INK, fontsize=8.5,
                    va="center", weight="bold")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("growth of \\$1 (log)")
    ax.legend(loc="upper left", fontsize=8, labelcolor=a1.INK, frameon=False)
    h = held.loc[START:END]
    bx.plot(h.index, h.values, color="#7b4fd6", lw=1.4)
    bx.set_ylabel("names shorted")
    bx.xaxis.set_major_locator(mdates.YearLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bx.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=330))
    for a in (ax, bx):
        a.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/short_leg.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def window_stats(r, spy):
    return {w: a1.stats(r.loc[a:b], spy) for w, (a, b) in WINDOWS.items()}


def main():
    pub, members, px, comp = load()
    spy_ret = px["SPY"].loc[START:END].pct_change()
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    bench = {b: px[b].loc[START:END].pct_change() for b in BENCH if b in px.columns}
    top = build_book(("top", None, comp), members, px, spy_ret, dfii10)
    ew = build_book(None, members, px, spy_ret, dfii10)
    spy_full = px["SPY"].loc[WARMUP_START:END].pct_change()
    warm = pickle.load(open(os.path.join(OUT, "_warmup_members.pkl"), "rb"))
    members_w = pd.concat([warm.loc[:"2018-12-31"], members.loc["2019-01-01":]]).fillna(False).astype(bool)
    top_w = build_book(("top", None, comp), members_w, px, spy_full, dfii10, start=WARMUP_START)["book"]
    ew_w = build_book(None, members_w, px, spy_full, dfii10, start=WARMUP_START)["book"]
    hedge_px = {hn: sum(w * px[t].loc[WARMUP_START:END].pct_change() for t, w in mix.items())
                for hn, mix in HEDGES.items()}
    spy = bench["SPY"]

    out = {"window": [START, END], "book": a1.stats(top["book"], spy),
           "published": a1.stats(ew["book"], spy),
           "bench": {b: a1.stats(r, spy) for b, r in bench.items()},
           "windows": {"book": window_stats(top["book"], spy), "published": window_stats(ew["book"], spy)},
           "stages": {k: a1.stats(top[k], spy) for k in ["raw", "vol_only", "core", "book"]},
           "ytd": a1.stats(top["book"].loc["2026"], spy), "by_year": {}}
    for y in range(2019, 2027):
        seg, eseg = top["book"].loc[f"{y}"], ew["book"].loc[f"{y}"]
        out["by_year"][y] = {
            "book_ret": round(float((1 + seg).prod() - 1), 4),
            "pub_ret": round(float((1 + eseg).prod() - 1), 4),
            "spy_ret": round(float((1 + spy.loc[f"{y}"].dropna()).prod() - 1), 4),
            "grid_ret": round(float((1 + bench["GRID"].loc[f"{y}"].dropna()).prod() - 1), 4),
            "book_sharpe": a1.stats(seg)["sharpe"], "book_maxdd": a1.stats(seg)["maxdd"],
            "book_calmar": a1.stats(seg)["calmar"],
            "names": round(float(top["held"].loc[f"{y}"].mean()), 1)}

    # hedged versions: trailing beta, borrow + trading charged
    out["hedged"] = {}
    hedged_series = {}
    for hname in HEDGES:
        for lab, bk, src in (("book", top["book"], top_w), ("published", ew["book"], ew_w)):
            net, beta = hedge(bk, hedge_px[hname], src)
            hedged_series[(hname, lab)] = net
            out["hedged"][f"{lab} - {hname}"] = {
                **{w: a1.stats(net.loc[a:b]) for w, (a, b) in WINDOWS.items()},
                "beta_mean": round(float(beta.loc[START:END].mean()), 2),
                "beta_last": round(float(beta.iloc[-1]), 2)}
            if lab == "book":        # how well the hedge tracks the book: the ex-ante criterion
                fit = {}
                for fw, (fa, fb) in {"2018-22": ("2018-01-01", "2022-12-31"),
                                     "2023+": ("2023-01-01", END)}.items():
                    dd = pd.concat([src.loc[fa:fb], hedge_px[hname].loc[fa:fb]], axis=1).dropna()
                    c = float(dd.corr().iloc[0, 1])
                    bb = dd.cov().iloc[0, 1] / dd.iloc[:, 1].var()
                    fit[fw] = {"corr": round(c, 3), "resid_vol": round(float((dd.iloc[:, 0] - bb * dd.iloc[:, 1]).std()
                                                                            * np.sqrt(config.ANN)), 4)}
                out["hedged"][f"{lab} - {hname}"]["fit"] = fit
    # single-name short option (bottom quintile by the 14-signal composite) + 0.25x ETF comparators
    basket, basket_cost, basket_held = name_short_basket(px, WARMUP_START)
    variants = {"names - beta": dict(h=basket, borrow=NAME_BORROW, unit_cost=basket_cost),
                "names - 0.25x": dict(h=basket, borrow=NAME_BORROW, unit_cost=basket_cost, fixed=FIXED_K)}
    for hname in ("GRID", "PAVE", "XLI"):
        variants[f"{hname} - 0.25x"] = dict(h=hedge_px[hname], fixed=FIXED_K)
    betas = {}
    for vname, kw in variants.items():
        net, beta = hedge(top["book"], beta_src=top_w, **kw)
        hedged_series[(vname, "book")] = net
        betas[vname] = beta
        out["hedged"][f"book - {vname}"] = {
            **{w: a1.stats(net.loc[a:b]) for w, (a, b) in WINDOWS.items()},
            "beta_mean": round(float(beta.loc[START:END].mean()), 2),
            "beta_last": round(float(beta.iloc[-1]), 2)}
    dd = pd.concat([top_w, basket], axis=1).dropna()
    for fw, (fa, fb) in {"2018-22": ("2018-01-01", "2022-12-31"), "2023+": ("2023-01-01", END)}.items():
        x = dd.loc[fa:fb]
        bb = x.cov().iloc[0, 1] / x.iloc[:, 1].var()
        out["hedged"]["book - names - beta"].setdefault("fit", {})[fw] = {
            "corr": round(float(x.corr().iloc[0, 1]), 3),
            "resid_vol": round(float((x.iloc[:, 0] - bb * x.iloc[:, 1]).std() * np.sqrt(config.ANN)), 4)}
    out["short_basket"] = {
        "stats": {w: a1.stats(basket.loc[a:b]) for w, (a, b) in WINDOWS.items()},
        "names_mean": round(float(basket_held.loc[START:END].mean()), 1),
        "names_last": int(basket_held.iloc[-1]),
        "turnover_yr": round(float(basket_cost.loc[START:END].sum() / (HEDGE_COST_BP / 1e4)
                                   / ((pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25)), 2)}
    for hname in ("GRID", "PAVE", "XLI"):
        betas[f"{hname} - beta"] = hedge(top["book"], hedge_px[hname], top_w)[1]

    # alpha: Jensen alpha of each book on GRID; and selection = book minus published book
    out["alpha"] = {}
    for w, (a, b) in WINDOWS.items():
        out["alpha"][w] = {
            "book_vs_GRID": alpha(top["book"].loc[a:b], bench["GRID"].loc[a:b]),
            "published_vs_GRID": alpha(ew["book"].loc[a:b], bench["GRID"].loc[a:b]),
            "book_vs_published": alpha(top["book"].loc[a:b], ew["book"].loc[a:b]),
            "book_vs_SPY": alpha(top["book"].loc[a:b], spy.loc[a:b])}

    # costs
    turn = overlay_turnover(top)
    turn_pub = overlay_turnover(ew)
    net = {bp: a1.stats(top["book"] - turn["total"] * bp / 1e4 / config.ANN)
           for bp in (2, 10, 20, 30)}
    net_fee = {fee: a1.stats(top["book"] - turn["total"] * 10 / 1e4 / config.ANN - fee / config.ANN)
               for fee in (0.01, 0.02)}
    net23 = {bp: a1.stats((top["book"] - turn["total"] * bp / 1e4 / config.ANN).loc["2023-01-01":END])
             for bp in (10, 20)}
    hg = hedged_series[("GRID", "book")]
    hedged_net = {bp: {w: a1.stats((hg - turn["total"] * bp / 1e4 / config.ANN).loc[a:b])
                       for w, (a, b) in WINDOWS.items()} for bp in (10, 20)}
    out["cost"] = {"turnover": turn, "turnover_published": turn_pub, "net": net, "net_fee": net_fee,
                   "net_2023": net23, "hedged_net": hedged_net}

    w = top["weights"].iloc[-1]
    out["holdings"] = sorted(w[w > 0].index)
    wp = ew["weights"].iloc[-1]
    out["holdings_published"] = sorted(wp[wp > 0].index)
    out["composite_last"] = {t: round(float(v), 2) for t, v in
                             comp.dropna(how="all").iloc[-1].reindex(out["holdings_published"]).fillna(0).sort_values(ascending=False).items()}
    out["breadth"] = {"mean": round(float(top["held"].loc[START:END].mean()), 1),
                      "mean_2023": round(float(top["held"].loc["2023-01-01":END].mean()), 1),
                      "last": int(top["held"].iloc[-1]), "min": int(top["held"].loc[START:END].min()),
                      "published_last": int(ew["held"].iloc[-1])}
    liq = data.dollar_volume()[out["holdings"]].loc[:END].tail(252).median()
    out["liquidity"] = {"min_name": liq.idxmin(), "min_dollar_volume_m": round(float(liq.min()) / 1e6, 1),
                        "median_dollar_volume_m": round(float(liq.median()) / 1e6, 1)}
    json.dump(out, open(f"{OUT}/stats.json", "w"), indent=1, default=float)

    d = {"book": top["book"], "raw": top["raw"], "vol_only": top["vol_only"], "core": top["core"],
         "bench": bench, "held": top["held"].loc[START:END]}
    a1.fig_growth(d); a1.fig_drawdown(d); a1.fig_years(d); a1.fig_riskreturn(d)
    a1.fig_breadth(d); a1.fig_overlay(d); a1.fig_beta(d, out)
    fig_selection(top["book"], ew["book"], bench)
    fig_hedged(hedged_series[("GRID", "book")], hedged_series[("GRID", "published")])
    lab = {"PAVE": "short PAVE", "GRID": "short GRID", "XLI": "short XLI",
           "names - beta": "short single names, beta-sized", "names - 0.25x": "short single names, 0.25x"}
    fig_short_options({v: hedged_series[(k, "book")] for k, v in lab.items()},
                      {lab[k]: betas[k if k.startswith("names") else f"{k} - beta"] for k in lab})
    fig_short_leg(top["book"], basket, bench["PAVE"], basket_held)
    print(f"wrote figures + stats.json to {OUT}")
    print(json.dumps({"book": out["book"], "published": out["published"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
