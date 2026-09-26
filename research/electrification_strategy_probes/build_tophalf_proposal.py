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
# The proposed long/short: the short is restricted to names passing the long side's rule 2 (10-K
# buildout wording, AUTO_CUSTOMER_TERMS, 4-year lookback) BEFORE the bottom quintile is formed.
SHORT_VOCAB_FILES = ("vocabulary_hits.csv", "vocabulary_hits_short_universe.csv")
C_LS, C_LONG, C_PAVE = "#2a78d6", "#7b4fd6", "#8b8a85"
C_GROWTH = "#b07a12"                           # supplier revenue-growth overlay (amber, distinct from series)
C_RATE = "#9b2226"                             # 10-year real yield overlay (dark red)
RATE_SIGNAL = "DFII10"                         # rule 11: hold the strategy only while the 10y real yield > 0
CASH_SERIES = "DTB3"                           # what idle capital earns when rule 11 is off (3m T-bill)
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


def name_short_basket(px: pd.DataFrame, start: str, keep=None,
                      log: dict | None = None, exclude: set | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Daily return of the bottom-quintile basket (long side, i.e. what is shorted), its daily
    rebalancing cost per $1, and the number of names held.

    At each month-end the 14-signal composite is ranked across that month's eligible universe
    (names priced that day); the bottom NAME_Q is bought equal-weight on the next trading day
    and held with weights drifting until the next month-end. No future information: the score
    uses filings and prices known at the month-end, and membership never looks at later prices.
    `keep(month_end, names) -> names` optionally drops names AFTER the quintile is formed (e.g. one
    SIC group); `exclude` (a set, or `f(month_end, names) -> names to drop`) removes names from the
    universe BEFORE ranking; `log`, if given,
    receives {month_end: names shorted}."""
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
        if exclude:
            drop = exclude(me, sc.index) if callable(exclude) else exclude
            sc = sc.drop([x for x in drop if x in sc.index])
        if len(sc) < 10:
            continue
        names = sc[sc <= sc.quantile(NAME_Q)].index
        if keep is not None:
            names = pd.Index(keep(me, names))
            if names.empty:
                continue
        if log is not None:
            log[me] = list(names)
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
def fig_selection(top, ew, bench, fname="selection.png", label="Selection book"):
    """Growth of $1. ew=None swaps the published book for the S&P 500 (the investor page)."""
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ends = []
    second = ("Published book (all names)", ew, a1.FAINT, 1.7) if ew is not None else \
        ("S&P 500", bench["SPY"], a1.C_SPY, 1.5)
    for label, r, c, lw in [(label, top, a1.C_BOOK, 2.3), second,
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
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)


def fig_sharpe(top, ew, bench, fname="sharpe.png", label="Selection book", regimes=False):
    """Lead figure: Sharpe by window (bars) over rolling 12-month Sharpe (lines).
    ew=None drops the published book (the investor page)."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.4, 6.6), gridspec_kw={"height_ratios": [1.15, 1]})
    rows = [(label, top, a1.C_BOOK), ("Published book", ew, a1.FAINT),
            ("GRID ETF", bench["GRID"], a1.C_GRID), ("PAVE ETF", bench["PAVE"], "#7b4fd6"),
            ("S&P 500", bench["SPY"], a1.C_SPY)]
    rows = [row for row in rows if row[1] is not None]
    wins = [("Full period, 2019–2026", START, 0.55), ("Grid tight, 2023 on", "2023-01-01", 1.0)]
    y = np.arange(len(rows))[::-1]
    h = 0.38
    for j, (wlab, a, alpha_) in enumerate(wins):
        vals = [a1.stats(r.loc[a:END])["sharpe"] for _, r, _ in rows]
        yy = y + (h / 2 if j == 0 else -h / 2)
        ax.barh(yy, vals, height=h * 0.92, color=[c for *_, c in rows], alpha=alpha_,
                label=wlab, edgecolor="none")
        for yv, v in zip(yy, vals):
            ax.text(v + 0.02, yv, f"{v:.2f}", va="center", fontsize=9, color=a1.INK,
                    weight="bold" if j == 1 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, *_ in rows], fontsize=10)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.set_xlabel("Sharpe ratio (4% risk-free)")
    ax.set_xlim(0, 2.0)
    ax.grid(axis="y", visible=False)
    leg = ax.legend(loc="lower right", fontsize=9, labelcolor=a1.INK, frameon=False)
    for p in leg.legend_handles:
        p.set_color(a1.SOFT)
    leg.legend_handles[0].set_alpha(0.55)

    win = 252
    rf = config.RF / config.ANN
    lines = [(label, top, a1.C_BOOK, 2.3), ("Published book", ew, a1.FAINT, 1.6),
             ("GRID ETF", bench["GRID"], a1.C_GRID, 1.5)]
    for lab, r, c, lw in [ln for ln in lines if ln[1] is not None]:
        ex = r.loc[START:END].dropna() - rf
        rs = (ex.rolling(win).mean() / ex.rolling(win).std() * np.sqrt(config.ANN)).dropna()
        bx.plot(rs.index, rs.values, color=c, lw=lw, label=lab)
    bx.axhline(0, color=a1.SOFT, lw=0.9)
    bx.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    bx.set_ylabel("rolling 12-month Sharpe")
    if regimes:                                   # qualitative market-regime labels (investor page)
        bx.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color=a1.C_GRID, alpha=0.07, lw=0)
        bx.set_ylim(top=bx.get_ylim()[1] + 1.6)
        yt = bx.get_ylim()[1] - 0.15
        for x0, txt, c in [("2020-01-20", "Negative real yields\nclean-energy rally\ngrid not yet tight", a1.C_GRID),
                           ("2022-01-15", "Rate\nshock", a1.SOFT),
                           ("2023-01-20", "Positive real yields · grid tight\nsuppliers paid for the buildout", a1.C_BOOK)]:
            bx.text(pd.Timestamp(x0), yt, txt, color=c, fontsize=8.3, va="top", linespacing=1.25)
    bx.xaxis.set_major_locator(mdates.YearLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bx.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=60))
    bx.legend(loc="lower left" if regimes else "upper left", fontsize=9, labelcolor=a1.INK,
              frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)


def fig_hedged(hedged_top, hedged_ew, fname="hedged.png",
               labels=("Selection book, short GRID", "Published book, short GRID")):
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for label, r, c, lw in [(labels[0], hedged_top, a1.C_BOOK, 2.3),
                            (labels[1], hedged_ew, a1.FAINT, 1.7)]:
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
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
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


def fig_short_leg(book: pd.Series, basket: pd.Series, pave: pd.Series, held: pd.Series,
                  fname="short_leg.png", label="Selection book"):
    """What is being shorted: the bottom-quintile basket vs PAVE vs the book, with stats."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    for label, r, c in [(label, book, a1.C_BOOK),
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
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- proposed book ----
def rule2_filter():
    """exclude-callable for `name_short_basket`: drop names without rule-2 wording that year."""
    from electrification_strategy import auto_screen
    hits = pd.concat([pd.read_csv(os.path.join(config.DATA_DIR, f), comment="#") for f in SHORT_VOCAB_FILES],
                     ignore_index=True)
    vocab = auto_screen.vocabulary_years(hits, config.AUTO_CUSTOMER_TERMS, list(range(2017, 2027)),
                                         lookback=a1.LOOKBACK)
    by_year: dict[int, set] = {}
    for tk, ys in vocab.items():
        for y in ys:
            by_year.setdefault(y, set()).add(tk)
    return lambda me, idx: [x for x in idx if x not in by_year.get(me.year, set())]


def supplier_growth(members: pd.DataFrame, real: bool = True) -> pd.Series:
    """Median trailing-twelve-month revenue growth (YoY, as filed) across the names that pass the
    long book's rules 1-5 that month -- the buildout, read from the suppliers' own filings.
    real=True subtracts CPI inflation (YoY), lagged a month for its publication date."""
    from research.xsec_alpha_probe import signals as sg
    rg = sg.fundamental_signals()["rev_growth"]
    elig = (members.reindex(index=rg.index, method="ffill").reindex(columns=rg.columns, fill_value=False)
            .fillna(False).astype(bool))
    g = rg.where(elig).median(axis=1).loc[START:END].dropna()
    if real:
        cpi = fetch_series("CPIAUCSL", "2015-01-01", END).pct_change(12)
        cpi.index = cpi.index + pd.offsets.MonthEnd(2)       # month m's CPI is usable at the end of m+1
        g = (g - cpi.reindex(g.index, method="ffill")).dropna()
    return g


def _growth_axis(ax, g: pd.Series, ry: pd.Series | None = None, rlim=(-2, 6)):
    """Overlay supplier revenue growth (right axis) and, if given, the 10y real yield on a second,
    offset right axis in its own colour. Returns the axes whose legend handles to add."""
    axes = []
    if ry is not None:
        rx = ax.twinx()
        rx.spines["right"].set_position(("outward", 52))
        rx.spines["right"].set_visible(True)
        rx.spines["right"].set_color(C_RATE)
        rx.plot(ry.index, ry.values, color=C_RATE, lw=1.0, alpha=0.75, ls=(0, (1, 1.5)),
                label="10-year real yield (right axis, red dotted)")
        rx.set_ylim(*rlim)
        rx.set_ylabel("10-year real yield, %", color=C_RATE)
        rx.tick_params(axis="y", colors=C_RATE)
        rx.grid(False)
        axes.append(rx)
    gx = ax.twinx()
    gx.step(g.index, g.values * 100, where="post", color=C_GROWTH, lw=1.2, alpha=0.8, ls=(0, (1, 1.5)),
            label="Supplier revenue growth after inflation, median (buildout proxy, right axis, amber dotted)")
    gx.axhline(0, color=a1.FAINT, lw=0.6, ls=":")
    gx.set_ylabel("supplier revenue growth after inflation, %", color=C_GROWTH)
    gx.tick_params(axis="y", colors=C_GROWTH)
    gx.grid(False)
    gx.spines["right"].set_visible(True)
    gx.spines["right"].set_color(C_GROWTH)
    return [gx] + axes


def _years(series: dict) -> dict:
    return {y: {k: round(float((1 + r.loc[f"{y}"].dropna()).prod() - 1), 4) for k, r in series.items()}
            for y in range(2019, 2027)}


def _shade_cash(ax, w: pd.Series | None):
    """Grey band wherever rule 11 has the strategy in T-bills."""
    if w is None:
        return
    off = w.loc[START:END] == 0
    grp = (off != off.shift()).cumsum()
    first = True
    for _, g in off[off].groupby(grp[off]):
        if len(g) < 5:
            continue
        ax.axvspan(g.index[0], g.index[-1], color="#6b6b6b", alpha=0.13, lw=0, hatch="//",
                   label="Strategy in T-bills (rule 11: regime not green)" if first else None)
        first = False


def fig_inv_sharpe(rows, lines, growth=None, ry=None, cash_w=None, regime=None, always_on=None):
    """Investor lead figures: Sharpe bars (full / 2023 on) -> inv_sharpe.png, and a separate, wide
    rolling 12-month Sharpe with regimes, the buildout proxy and real yields -> inv_rolling.png."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(rows))[::-1]
    h = 0.38
    first = {lab: r.dropna().index[0] for lab, r, _ in rows}
    for j, (wlab, a, alpha_) in enumerate([("Full period, 2019–2026", START, 0.55),
                                           ("Grid tight, 2023 on", "2023-01-01", 1.0)]):
        # a series that starts after the window opens (VOLT) gets only its since-listing bar
        vals = [np.nan if (first[lab] > pd.Timestamp(a) + pd.Timedelta(days=40) and j == 0)
                else a1.stats(r.loc[a:END])["sharpe"] for lab, r, _ in rows]
        yy = y + (h / 2 if j == 0 else -h / 2)
        ax.barh(yy, np.nan_to_num(vals), height=h * 0.92, color=[c for *_, c in rows], alpha=alpha_,
                label=wlab, edgecolor="none")
        for yv, v in zip(yy, vals):
            if not np.isnan(v):
                ax.text(v + 0.02, yv, f"{v:.2f}", va="center", fontsize=9, color=a1.INK,
                        weight="bold" if j == 1 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, *_ in rows], fontsize=10)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.set_xlabel("Sharpe ratio (4% risk-free)")
    ax.set_xlim(0, 2.05)
    ax.grid(axis="y", visible=False)
    leg = ax.legend(loc="lower right", fontsize=9, labelcolor=a1.INK, frameon=False)
    for p_ in leg.legend_handles:
        p_.set_color(a1.SOFT)
    leg.legend_handles[0].set_alpha(0.55)
    fig.tight_layout()
    fig.savefig(f"{OUT}/inv_sharpe.png", bbox_inches="tight")
    plt.close(fig)

    fig, (bx, sx) = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 0.075], "hspace": 0.06})
    rf = config.RF / config.ANN
    for k, (lab, r, c, lw) in enumerate(lines):
        ex = r.loc[START:END].dropna() - rf
        rs = (ex.rolling(252).mean() / ex.rolling(252).std() * np.sqrt(config.ANN)).dropna()
        if k == 0 and cash_w is not None:
            # solid only where the trailing year was (almost) fully invested -- there it equals the
            # always-invested line, so the dashed "kept invested" line joins it without a jump
            on = cash_w.reindex(rs.index).rolling(252, min_periods=1).mean()
            rs = rs.where(on >= 0.97)
            if always_on is not None:
                ea = always_on.loc[START:END].dropna() - rf
                ra = (ea.rolling(252).mean() / ea.rolling(252).std() * np.sqrt(config.ANN)).dropna()
                gap = rs.reindex(ra.index).isna()
                gap = gap | gap.shift(1, fill_value=False) | gap.shift(-1, fill_value=False)
                bx.plot(ra.index, ra.where(gap).values, color=c, lw=1.8, ls=(0, (5, 2.5)),
                        label="Strategy if rule 11 had kept it invested (dashed)")
        bx.plot(rs.index, rs.values, color=c, lw=lw, label=lab)
    _shade_cash(bx, cash_w)
    bx.set_ylim(-1.8, 3.8)
    bx.axhline(0, color=a1.SOFT, lw=0.9)
    bx.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color=a1.C_GRID, alpha=0.07, lw=0)
    bx.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    bx.set_ylim(top=bx.get_ylim()[1] + 1.6)
    yt = bx.get_ylim()[1] - 0.15
    for x0, txt, c in [("2020-01-20", "Negative real yields\nclean-energy rally\ngrid not yet tight", a1.C_GRID),
                       ("2022-01-15", "Rate\nshock", a1.SOFT),
                       ("2023-01-20", "Positive real yields · grid tight\nsuppliers paid for the buildout", a1.C_BOOK)]:
        bx.text(pd.Timestamp(x0), yt, txt, color=c, fontsize=10, va="top", linespacing=1.25)
    bx.set_ylabel("rolling 12-month Sharpe")
    bx.xaxis.set_major_locator(mdates.YearLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bx.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=60))
    hs, ls_ = bx.get_legend_handles_labels()
    if growth is not None:
        extra = _growth_axis(bx, growth, ry)
        lo, hi = bx.get_ylim()
        f = -lo / (hi - lo)                        # put every axis's zero on the Sharpe zero line
        extra[0].set_ylim(-f / (1 - f) * 45, 45)
        if len(extra) > 1:
            extra[1].set_ylim(-f / (1 - f) * 9, 9)
        for gx in extra:
            h2, l2 = gx.get_legend_handles_labels()
            hs, ls_ = hs + h2, ls_ + l2
    if regime is not None:                         # green: both conditions met, amber: one, red: neither
        cols = {2: "#bfe3c9", 1: "#f6e1a8", 0: "#f3bcc3"}   # pastel: green / amber / red
        rg = regime.loc[START:END]
        grp = (rg != rg.shift()).cumsum()
        for _, g in rg.groupby(grp):
            sx.axvspan(g.index[0], g.index[-1] + pd.Timedelta(days=1), color=cols[int(g.iloc[0])], lw=0)
        sx.set_yticks([])
        sx.set_ylabel("regime", rotation=0, ha="right", va="center", fontsize=9.5, color=a1.SOFT)
        for sp in sx.spines.values():
            sp.set_visible(False)
        from matplotlib.patches import Patch
        hs += [Patch(color=cols[2]), Patch(color=cols[1]), Patch(color=cols[0])]
        ls_ += ["Regime green: real yield > 0 and real supplier growth > 0 (invested)",
                "Regime amber: one of the two", "Regime red: neither"]
    sx.xaxis.set_major_locator(mdates.YearLocator())
    sx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.legend(hs, ls_, loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=9.5, labelcolor=a1.INK,
               frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/inv_rolling.png", bbox_inches="tight")
    plt.close(fig)


def _regime_strip(sx, regime):
    """Pastel green/amber/red strip on `sx`; returns (handles, labels) for the legend."""
    from matplotlib.patches import Patch
    cols = {2: "#bfe3c9", 1: "#f6e1a8", 0: "#f3bcc3"}
    rg = regime.loc[START:END]
    for _, g in rg.groupby((rg != rg.shift()).cumsum()):
        sx.axvspan(g.index[0], g.index[-1] + pd.Timedelta(days=1), color=cols[int(g.iloc[0])], lw=0)
    sx.set_yticks([])
    sx.set_ylabel("regime", rotation=0, ha="right", va="center", fontsize=9.5, color=a1.SOFT)
    for sp in sx.spines.values():
        sp.set_visible(False)
    return ([Patch(color=cols[2]), Patch(color=cols[1]), Patch(color=cols[0])],
            ["Regime green: real yield > 0 and real supplier growth > 0 (invested)",
             "Regime amber: one of the two", "Regime red: neither"])


def fig_inv_growth(lines, growth=None, ry=None, cash_w=None, regime=None, always_on=None):
    """Wide growth-of-$1 chart with the regime strip beneath and, dashed, the strategy's path had
    rule 11 kept it invested (each T-bill spell starts from the strategy's level on the switch day)."""
    fig, (ax, sx) = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 0.075], "hspace": 0.06})
    ends = []
    base = a1.curve(lines[0][1].loc[START:END])
    for k, (lab, r, c, lw) in enumerate(lines):
        cv = a1.curve(r.loc[START:END])
        late = cv.index[0] > pd.Timestamp(START) + pd.Timedelta(days=40)
        if late:                                   # start it on the strategy's line
            cv = cv * float(base.loc[:cv.index[0]].iloc[-1])
        ax.plot(cv.index, cv.values, color=c, lw=lw, label=lab)
        if not late:                               # a rebased line's level is not a growth of $1
            ends.append([float(cv.iloc[-1]), cv.index[-1], float(cv.iloc[-1])])
        if k == 0 and always_on is not None and cash_w is not None:
            off = cash_w.loc[START:END] == 0
            first = True
            for _, g in off[off].groupby((off != off.shift()).cumsum()[off]):
                if len(g) < 5:
                    continue
                start = cv.loc[:g.index[0]].index[-2] if len(cv.loc[:g.index[0]]) > 1 else g.index[0]
                seg = always_on.loc[g.index[0]:g.index[-1]]
                path = float(cv.loc[start]) * (1 + seg).cumprod()
                path = pd.concat([pd.Series([float(cv.loc[start])], index=[start]), path])
                ax.plot(path.index, path.values, color=c, lw=1.8, ls=(0, (5, 2.5)),
                        label="Strategy if rule 11 had kept it invested (dashed)" if first else None)
                first = False
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if np.log10(ends[i][0]) - np.log10(ends[i - 1][0]) < 0.035:
            ends[i][0] = 10 ** (np.log10(ends[i - 1][0]) + 0.035)
    for ypos, xend, val in ends:
        ax.annotate(f"  {val:.2f}x", (xend, ypos), color=a1.INK, fontsize=10, va="center", weight="bold")
    ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp(END), color=a1.C_BOOK, alpha=0.06, lw=0)
    _shade_cash(ax, cash_w)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4, 6])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=200))
    hs, ls_ = ax.get_legend_handles_labels()
    if growth is not None:
        extra = _growth_axis(ax, growth, ry, rlim=(-4, 6))   # zeros aligned at 40% of both axes
        extra[0].set_ylim(-40, 60)
        for gx in extra:
            h2, l2 = gx.get_legend_handles_labels()
            hs, ls_ = hs + h2, ls_ + l2
    if regime is not None:
        h3, l3 = _regime_strip(sx, regime)
        hs, ls_ = hs + h3, ls_ + l3
    sx.xaxis.set_major_locator(mdates.YearLocator())
    sx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.legend(hs, ls_, loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=9.5, labelcolor=a1.INK,
               frameon=False, ncol=3)
    fig.savefig(f"{OUT}/inv_growth.png", bbox_inches="tight")
    plt.close(fig)


def fig_inv_years(series, counterfactual=None):
    """Calendar-year returns. `counterfactual=(always_on_returns, invested_weight)`: for each year rule 11
    spent at least 5% of in T-bills, a hatched bar in the strategy's colour shows the always-invested
    return behind the actual one."""
    yrs = list(range(2019, 2027))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    n = len(series)
    wbar = 0.8 / n
    x = np.arange(len(yrs))
    for i, (lab, r, c) in enumerate(series):
        v = [float((1 + r.loc[f"{y}"].dropna()).prod() - 1) * 100 for y in yrs]
        ax.bar(x - 0.4 + wbar * (i + 0.5), v, wbar * 0.92, color=c, label=lab, zorder=3)
        if i == 0:
            for xi, vi in zip(x - 0.4 + wbar * 0.5, v):
                ax.text(xi, vi + (1.4 if vi >= 0 else -3.8), f"{vi:.1f}" if abs(vi) < 1 else f"{vi:.0f}",
                        ha="center", fontsize=8.3,
                        color=a1.INK, weight="bold", zorder=5)
            if counterfactual is not None:
                alw, wgt = counterfactual
                first = True
                for xi, y, vi in zip(x - 0.4 + wbar * 0.5, yrs, v):
                    if float(1 - wgt.loc[f"{y}"].mean()) < 0.05:
                        continue
                    va = float((1 + alw.loc[f"{y}"].dropna()).prod() - 1) * 100
                    ax.bar(xi, va, wbar * 0.92, facecolor="none", edgecolor=c, hatch="////", lw=1.1, zorder=4,
                           label="Long/short had rule 11 kept it invested (hatched)" if first else None)
                    first = False
                    top = max(va, vi)
                    ax.text(xi, (va if va >= 0 else va) + (1.4 if va >= 0 else -3.8) + (3.6 if abs(va - vi) < 4 else 0),
                            f"{va:.1f}" if abs(va) < 1 else f"{va:.0f}", ha="center", fontsize=8.0, color=c,
                            style="italic", zorder=5)
    ax.axhline(0, color=a1.FAINT, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) if y != 2026 else "2026*" for y in yrs])
    ax.set_ylabel("calendar-year return (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.8, labelcolor=a1.INK, frameon=False, ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/inv_years.png", bbox_inches="tight")
    plt.close(fig)


def fig_inv_drawdown(lines):
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for lab, r, c, lw, *style in lines:
        dd = a1.drawdown(r.loc[START:END]) * 100
        ax.plot(dd.index, dd.values, color=c, lw=lw, ls=style[0] if style else "-",
                label=f"{lab}  (worst {dd.min():.0f}%)")
    ax.set_ylabel("drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right", fontsize=8.8, labelcolor=a1.INK, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/inv_drawdown.png", bbox_inches="tight")
    plt.close(fig)


def fig_inv_breadth(long_held, short_held):
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.step(long_held.index, long_held.values, where="post", color=C_LONG, lw=1.6, label="names held long")
    sh = short_held.loc[START:END]
    ax.step(sh.index, sh.values, where="post", color=a1.C_NEG, lw=1.4, label="names held short")
    ax.set_ylabel("names")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9, labelcolor=a1.INK, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/inv_breadth.png", bbox_inches="tight")
    plt.close(fig)


def rate_gate(r: pd.Series, ry: pd.Series, cash: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Rule 11: hold r only while the 10-year real yield is above zero, known the day before; otherwise
    hold 3-month T-bills. 10bp is charged on each switch. Returns (gated returns, on/off weight)."""
    w = (ry.reindex(r.index, method="ffill").shift(1) > 0).astype(float)
    w.iloc[0] = 1.0 if pd.isna(ry.reindex(r.index, method="ffill").iloc[0]) else w.iloc[0]
    tb = cash.reindex(r.index, method="ffill").fillna(0) / 100 / config.ANN
    return w * r + (1 - w) * tb - w.diff().abs().fillna(0) * HEDGE_COST_BP / 1e4, w


def time_under_water(r: pd.Series) -> dict:
    """Longest stretch below a previous peak, in months; `recovered` False if still under at the end."""
    cv = (1 + r.dropna()).cumprod()
    peak, best, cur, start = 1.0, pd.Timedelta(0), None, None
    prev = cv.index[0] - pd.Timedelta(days=1)
    for d, v in cv.items():
        if v < peak - 1e-12:
            cur = cur or prev
        else:
            if cur is not None and d - cur > best:
                best, start = d - cur, (cur, d, True)
            cur, peak = None, max(peak, v)
        prev = d
    if cur is not None and cv.index[-1] - cur > best:
        best, start = cv.index[-1] - cur, (cur, cv.index[-1], False)
    return {"months": round(best.days / 30.44, 1), "recovered": bool(start[2]) if start else True,
            "from": str(start[0].date()) if start else None, "to": str(start[1].date()) if start else None}


def proposed(top, top_w, px, bench, turn, members, comp, dfii10):
    """The investor proposal (`investor_book`): buffered top-half long book, 20-point no-trade band,
    monthly sleeve and short resets, rule-2 short re-formed quarterly at $0.25, rule 11 -- with every
    trade charged. `top` is kept only for the always-on reference and the short-leg chart."""
    import investor_book as ib
    ry = fetch_series(RATE_SIGNAL, "2003-01-01", END).dropna()
    cash = fetch_series(CASH_SERIES, "2003-01-01", END).dropna()
    spy = bench["SPY"]
    sgrowth = supplier_growth(members)
    run = lambda **kw: ib.build(members, comp, px, spy, dfii10, ry, cash, growth=sgrowth, **kw)
    main_ = run()
    ls, gate_w = main_["returns"], main_["invested"]
    pave = run(short="PAVE")["returns"]
    gross = run(bp=0.0)["returns"]
    always = ib.build(members, comp, px, spy, dfii10, ry.clip(lower=1.0), cash)["returns"]   # rule 11 never fires
    yield_only = ib.build(members, comp, px, spy, dfii10, ry, cash)["returns"]                # rule 11 on real yield alone
    books = {"long_short": ls, "long_short_gross": gross, "long_short_always_on": always,
             "long_short_yield_only": yield_only,
             "long_short_pave": pave, **bench}
    wins = {"full": (START, END), "2019-22": ("2019-01-01", "2022-12-31"), "2023+": ("2023-01-01", END)}
    held_l, held_s = main_["long_held"].loc[START:END], main_["short_held"].loc[START:END]
    log = main_["short_log"]
    out = {"windows": {k: {w: a1.stats(r.loc[a:e], spy.loc[a:e]) for w, (a, e) in wins.items()}
                       for k, r in books.items()},
           "by_year": _years({"long_short": ls, "GRID": bench["GRID"], "PAVE": bench["PAVE"], "SPY": spy}),
           "by_year_sharpe": {y: a1.stats(ls.loc[f"{y}"])["sharpe"] for y in range(2019, 2027)},
           "by_year_maxdd": {y: a1.stats(ls.loc[f"{y}"])["maxdd"] for y in range(2019, 2027)},
           "alpha": {w: {"ls_vs_GRID": alpha(ls.loc[a:e], bench["GRID"].loc[a:e]),
                         "ls_vs_SPY": alpha(ls.loc[a:e], spy.loc[a:e])} for w, (a, e) in wins.items()},
           "beta_to_GRID": round(float(ls.loc[START:END].cov(bench["GRID"].loc[START:END])
                                       / bench["GRID"].loc[START:END].var()), 2),
           "names": {"long_mean": round(float(held_l.mean()), 1), "long_min": int(held_l[held_l > 0].min()),
                     "long_mean_2023": round(float(held_l.loc["2023-01-01":].mean()), 1),
                     "long_last": int(held_l.iloc[-1]),
                     "short_mean": round(float(held_s[held_s > 0].mean()), 1),
                     "short_min": int(held_s[held_s > 0].min()),
                     "short_mean_2023": round(float(held_s.loc["2023-01-01":].mean()), 1),
                     "short_last": int(held_s.iloc[-1])},
           "short_basket": {w: a1.stats(main_["short_returns"].loc[a:e]) for w, (a, e) in wins.items()},
           "short_last": {"formed": str(max(log).date()), "names": log[max(log)]},
           "turnover": {k: round(v, 3) for k, v in main_["turnover"].items()},
           "turnover_invested": round(main_["turnover_invested"], 3)}
    wl = main_["long_weights"].iloc[-1]
    out["long_last"] = sorted(wl[wl > 0].index)
    out["cost"] = {bp: {w: a1.stats(run(bp=float(bp))["returns"].loc[a:e]) for w, (a, e) in wins.items()}
                   for bp in (0, 2, 10, 20)}
    out["cost"]["fee"] = {f: a1.stats(run(fee=f)["returns"]) for f in (0.01, 0.02)}
    out["tuw"] = {k: {w: time_under_water(r.loc[a:e]) for w, (a, e) in wins.items()}
                  for k, r in books.items() if k != "VOLT"}
    on = gate_w.loc[START:END]
    out["rate_rule"] = {"share_on": round(float(on.mean()), 3), "switches": int((on.diff().abs() > 0).sum()),
                        "real_yield_last": [str(ry.index[-1].date()), round(float(ry.iloc[-1]), 2)]}
    r3 = fetch_prices(["IWV"], "2015-01-01", END)["IWV"].pct_change()
    out["beta_russell3000"] = {k: {w: round(float(r.loc[a:e].cov(r3.reindex(r.index).loc[a:e])
                                                   / r3.reindex(r.index).loc[a:e].var()), 2)
                                   for w, (a, e) in wins.items()} for k, r in books.items()}
    ls_lab = "Long/short (the strategy)"
    out["supplier_growth"] = {str(d.date()): round(float(v), 4) for d, v in sgrowth.resample("QE").last().items()}
    # regime: 2 = real yield > 0 and real supplier growth > 0; 1 = one of them; 0 = neither
    dix = ls.index
    regime = ((ry.reindex(dix, method="ffill").shift(1) > 0).astype(int)
              + (sgrowth.reindex(dix, method="ffill") > 0).astype(int))
    out["regime_by_year"] = {y: {str(k): round(float(v), 2) for k, v in
                                 regime.loc[str(y)].value_counts(normalize=True).items()} for y in range(2019, 2027)}
    volt = bench["VOLT"].dropna()
    vlab = "VOLT ETF*"
    fig_inv_sharpe([(ls_lab, ls, C_LS), ("GRID ETF", bench["GRID"], a1.C_GRID), ("PAVE ETF", bench["PAVE"], C_PAVE),
                    ("S&P 500", spy, a1.C_SPY), (vlab, volt, C_LONG)],
                   [(ls_lab, ls, C_LS, 2.3), ("GRID ETF", bench["GRID"], a1.C_GRID, 1.3),
                    ("PAVE ETF", bench["PAVE"], C_PAVE, 1.3), ("S&P 500", spy, a1.C_SPY, 1.3), (vlab, volt, C_LONG, 1.3)],
                   growth=sgrowth, ry=ry.loc[START:END], cash_w=gate_w, regime=regime, always_on=always)
    fig_inv_growth([(ls_lab, ls, C_LS, 2.3), ("GRID ETF", bench["GRID"], a1.C_GRID, 1.4),
                    ("PAVE ETF", bench["PAVE"], C_PAVE, 1.4), ("S&P 500", spy, a1.C_SPY, 1.4),
                    ("VOLT ETF* (from its Dec 2024 launch)", volt, C_LONG, 1.4)], growth=sgrowth,
                   ry=ry.loc[START:END], cash_w=gate_w, regime=regime, always_on=always)
    v0 = str(volt.index[0].date())
    out["volt_window"] = {"start": v0, **{k: a1.stats(r.loc[v0:END], spy.loc[v0:END])
                                          for k, r in (("long_short", ls), ("VOLT", volt), ("GRID", bench["GRID"]),
                                                       ("PAVE", bench["PAVE"]), ("SPY", spy))}}
    fig_inv_years([(ls_lab, ls, C_LS), ("GRID ETF", bench["GRID"], a1.C_GRID), ("PAVE ETF", bench["PAVE"], C_PAVE),
                   ("S&P 500", spy, a1.C_SPY)], counterfactual=(always, gate_w))
    out["by_year_always_on"] = {y: round(float((1 + always.loc[f"{y}"].dropna()).prod() - 1), 4)
                                for y in range(2019, 2027)}
    out["by_year_share_off"] = {y: round(float(1 - gate_w.loc[f"{y}"].mean()), 2) for y in range(2019, 2027)}
    fig_inv_drawdown([(ls_lab, ls, C_LS, 2.0),
                      ("Had the step-aside rule kept it invested", always, C_LS, 1.3, (0, (4, 2))),
                      ("GRID ETF", bench["GRID"], a1.C_GRID, 1.2),
                      ("S&P 500", spy, a1.C_SPY, 1.2)])
    fig_inv_breadth(held_l, held_s)
    fig_short_leg(top["book"], main_["short_returns"], bench["PAVE"], held_s, "inv_short_leg.png", "Long book")
    return out


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
    fig_sharpe(top["book"], ew["book"], bench)
    fig_selection(top["book"], ew["book"], bench)
    fig_hedged(hedged_series[("GRID", "book")], hedged_series[("GRID", "published")])
    lab = {"PAVE": "short PAVE", "GRID": "short GRID", "XLI": "short XLI",
           "names - beta": "short single names, beta-sized", "names - 0.25x": "short single names, 0.25x"}
    fig_short_options({v: hedged_series[(k, "book")] for k, v in lab.items()},
                      {lab[k]: betas[k if k.startswith("names") else f"{k} - beta"] for k in lab})
    fig_short_leg(top["book"], basket, bench["PAVE"], basket_held)
    # investor page: the proposed long/short and the long-only smart-beta book
    out["proposed"] = proposed(top, top_w, px, bench, turn, members, comp, dfii10)
    json.dump(out, open(f"{OUT}/stats.json", "w"), indent=1, default=float)
    print(f"wrote figures + stats.json to {OUT}")
    print(json.dumps({"book": out["book"], "published": out["published"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
