"""Figures + investor stats for the `screen_auto` A1+A4 proposal (dark mode).

    python research/electrification_strategy_probes/build_a1a4_proposal.py

Writes:
    output_html/img/a1a4/*.png     -- dark-mode figures for the proposal
    output_html/img/a1a4/stats.json -- every number the write-up cites

The book: `screen_auto` with AUTO_LOOKBACK_YEARS=4 and AUTO_CAPEX_QUANTILE=0.60,
evaluated SEED-FREE (on the mechanical universe it selects from, not the
33-name hand panel), shipped overlay = 20% vol target + valuation de-lever +
15% GLD/IEF sleeve.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from grid_equipment_basket.basket import simulate_basket, apply_cap
from grid_equipment_basket.data.prices import fetch_prices
from capex_intensity_strategy import edgar, fundamentals, pipeline
from electrification_strategy import (auto_screen, backtest, config, hedge_overlay,
                                      universe, valuation_overlay)
from grid_equipment_basket.basket import rebalance_dates
from electrification_strategy.fred import fetch_series

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_build_cache.pkl")
START, END = "2019-01-01", "2026-08-31"
LOOKBACK, QUANTILE = 4, 0.60
BENCH = ["SPY", "GRID", "PAVE", "XLU", "VOLT"]

# -- dark palette (validated: node scripts/validate_palette.js --mode dark --surface #1b1e23)
SURFACE = "#1b1e23"
C_BOOK, C_SPY, C_GRID = "#3987e5", "#d95926", "#199e70"
INK, SOFT, FAINT, LINE = "#e6e9ec", "#9aa3ad", "#727a84", "#2c313a"
C_NEG = "#e66767"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0, "axes.labelcolor": SOFT,
    "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 160,
})


# ------------------------------------------------------------------ data ----
def build_returns():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))

    built = pipeline.build(start=START, end=END)
    score, mask, cands = built["score"], built["mask"], built["candidates"]
    power = set(cands.loc[cands["group"] == "POWER", "ticker"])
    panel = fundamentals.ttm_panel(edgar.fetch_facts({}),
                                   pd.date_range(START, END, freq="ME"))
    hits = auto_screen.load_vocabulary_hits()

    seed = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    extra = sorted(set(seed) | set(BENCH) | set(config.HEDGE_SLEEVE_TICKERS)
                   | set(config.HEDGE_SHORT_TICKERS) | {config.FEARED_PROXY})
    spx = fetch_prices(extra, "2015-01-01", END)

    px = built["prices"].copy()                      # full history: the listing gate needs it
    for t in spx.columns:
        if t not in px.columns:
            px[t] = spx[t].reindex(px.index)

    def target_fn(members):
        def _fn(available, asof):
            asof = pd.Timestamp(asof)
            upto = members.loc[:asof]
            if upto.empty:
                return pd.Series(dtype=float)
            row = upto.iloc[-1]
            held = set(row.index[row.astype(bool)])
            counts = px.loc[:asof].notna().sum()
            listed = set(counts[counts >= config.LISTING_MIN_DAYS].index)
            elig = sorted(held & set(available) & listed)
            if not elig:
                return pd.Series(dtype=float)
            return apply_cap(pd.Series(1.0 / len(elig), index=elig),
                             config.MAX_SINGLE_NAME_WEIGHT)
        return _fn

    members = auto_screen.build_membership(score, mask, panel, hits, power_tickers=power,
                                           quantile=QUANTILE, lookback=LOOKBACK)
    cols = [c for c in members.columns if c in px.columns]
    br = simulate_basket(px[cols], START, END, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, target_fn(members))

    spy_ret = px["SPY"].loc[START:END].pct_change()
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    raw = br.returns
    vol_only = backtest.compose(raw, spy_ret, use_valuation=False)
    core = backtest.compose(raw, spy_ret, use_valuation=True)
    rising = hedge_overlay.real_yield_rising_mask(dfii10, core.index)
    book = hedge_overlay.apply_hedge(core, hedge_overlay.sleeve_returns(px),
                                     hedge_overlay.short_returns(px), rising, True, False)

    # the hand-picked book, on its own panel, for the like-for-like comparison
    cand_px = spx[[c for c in seed if c in spx.columns]]
    hb = simulate_basket(cand_px, START, END, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn("screen", cand_px))
    hcore = backtest.compose(hb.returns, spx["SPY"].loc[START:END].pct_change(), True)
    hand = hedge_overlay.apply_hedge(hcore, hedge_overlay.sleeve_returns(spx),
                                     hedge_overlay.short_returns(spx),
                                     hedge_overlay.real_yield_rising_mask(dfii10, hcore.index),
                                     True, False)

    bench = {b: px[b].loc[START:END].pct_change() for b in BENCH if b in px.columns}
    out = {"raw": raw, "vol_only": vol_only, "core": core, "book": book, "hand": hand,
           "bench": bench, "held": (br.weights > 0).sum(axis=1),
           "weights_last": br.weights.iloc[-1], "members": members}
    pickle.dump(out, open(CACHE, "wb"))
    return out


# --------------------------------------------------------------- metrics ----
def stats(r: pd.Series, spy: pd.Series | None = None) -> dict:
    r = r.dropna()
    m = backtest.compute_metrics(r, config.RF, config.ANN)
    out = {"cagr": m["cagr"], "vol": m["ann_vol"], "sharpe": m["sharpe"],
           "sortino": m["sortino"], "maxdd": m["max_dd"], "hit_rate": m["hit_rate"],
           "calmar": round(m["cagr"] / abs(m["max_dd"]), 3) if m["max_dd"] < 0 else None}
    if spy is not None:
        s = spy.reindex(r.index).fillna(0.0)
        out["beta"] = round(float(np.cov(r, s)[0, 1] / s.var()), 2)
        out["corr"] = round(float(r.corr(s)), 2)
        up, dn = s > 0, s < 0
        out["up_capture"] = round(float(r[up].mean() / s[up].mean()), 2)
        out["down_capture"] = round(float(r[dn].mean() / s[dn].mean()), 2)
    return out


def curve(r):
    return (1.0 + r.dropna()).cumprod()


def drawdown(r):
    c = curve(r)
    return c / c.cummax() - 1.0


# --------------------------------------------------------------- figures ----
def fig_growth(d):
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    series = [("The book", d["book"], C_BOOK, 2.2),
              ("S&P 500", d["bench"]["SPY"], C_SPY, 1.6),
              ("GRID ETF", d["bench"]["GRID"], C_GRID, 1.6)]
    ends = []
    for label, r, c, lw in series:
        cv = curve(r)
        ax.plot(cv.index, cv.values, color=c, lw=lw, label=label, solid_capstyle="round")
        ends.append([float(cv.iloc[-1]), cv.index[-1], label, c, float(cv.iloc[-1])])
    # push labels apart in log space so near-identical endpoints stay legible
    ends.sort(key=lambda e: e[0])
    gap = 0.052
    for i in range(1, len(ends)):
        if np.log10(ends[i][0]) - np.log10(ends[i - 1][0]) < gap:
            ends[i][0] = 10 ** (np.log10(ends[i - 1][0]) + gap)
    for ypos, xend, label, c, val in ends:
        ax.annotate(f"  {label}  {val:.2f}x", (xend, ypos), color=c, fontsize=9.5,
                    va="center", weight="bold")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4, 6])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel("growth of $1 (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(cv.index[0], cv.index[-1] + pd.Timedelta(days=430))
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/growth.png", bbox_inches="tight")
    plt.close(fig)


def fig_drawdown(d):
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for label, r, c, fill in [("S&P 500", d["bench"]["SPY"], C_SPY, False),
                              ("The book", d["book"], C_BOOK, True)]:
        dd = drawdown(r) * 100
        if fill:
            ax.fill_between(dd.index, dd.values, 0, color=c, alpha=0.20, linewidth=0)
        ax.plot(dd.index, dd.values, color=c, lw=1.7, label=label,
                alpha=1.0 if fill else 0.85)
    ax.set_ylabel("drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/drawdown.png", bbox_inches="tight")
    plt.close(fig)


def fig_years(d):
    yrs = list(range(2019, 2027))
    bk = [float((1 + d["book"].loc[f"{y}"]).prod() - 1) * 100 for y in yrs]
    sp = [float((1 + d["bench"]["SPY"].loc[f"{y}"].dropna()).prod() - 1) * 100 for y in yrs]
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(x - 0.21, bk, 0.40, color=C_BOOK, label="The book", zorder=3)
    ax.bar(x + 0.21, sp, 0.40, color=C_SPY, label="S&P 500", zorder=3)
    for xi, v in zip(x - 0.21, bk):
        ax.text(xi, v + (1.4 if v >= 0 else -3.6), f"{v:.0f}", ha="center",
                fontsize=8.5, color=INK, weight="bold")
    ax.axhline(0, color=FAINT, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) if y != 2026 else "2026*" for y in yrs])
    ax.set_ylabel("calendar-year return (%)")
    ax.set_ylim(min(sp + bk) - 6, max(sp + bk) + 9)
    ax.legend(loc="upper right", fontsize=9.5, labelcolor=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/years.png", bbox_inches="tight")
    plt.close(fig)


def fig_riskreturn(d):
    pts = [("The book", d["book"], C_BOOK)]
    pts += [(b, d["bench"][b], C_SPY if b == "SPY" else FAINT)
            for b in ["SPY", "GRID", "PAVE", "XLU"] if b in d["bench"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, r, c in pts:
        m = stats(r)
        big = label == "The book"
        ax.scatter(m["vol"] * 100, m["cagr"] * 100, s=210 if big else 110,
                   color=c, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"  {label}", (m["vol"] * 100, m["cagr"] * 100), color=INK if big else SOFT,
                    fontsize=10 if big else 9.5, va="center",
                    weight="bold" if big else "normal")
    ax.set_xlabel("volatility (% a year)")
    ax.set_ylabel("return (% a year)")
    xs = [stats(r)["vol"] * 100 for _, r, _ in pts]
    ys = [stats(r)["cagr"] * 100 for _, r, _ in pts]
    ax.set_xlim(0, max(xs) * 1.32)
    ax.set_ylim(min(ys) - 2.5, max(ys) + 2.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/risk_return.png", bbox_inches="tight")
    plt.close(fig)


def fig_breadth(d):
    h = d["held"]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.fill_between(h.index, h.values, 0, color=C_BOOK, alpha=0.18, step="post", linewidth=0)
    ax.step(h.index, h.values, where="post", color=C_BOOK, lw=2.0)
    ax.annotate(f"  {int(h.iloc[-1])} names", (h.index[-1], h.iloc[-1]), color=C_BOOK,
                fontsize=9.5, va="center", weight="bold")
    ax.set_ylabel("names held")
    ax.set_ylim(0, max(h.values) * 1.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(h.index[0], h.index[-1] + pd.Timedelta(days=330))
    fig.tight_layout()
    fig.savefig(f"{OUT}/breadth.png", bbox_inches="tight")
    plt.close(fig)


def fig_overlay(d):
    """What each layer of risk control does -- one axis, drawdown only."""
    stages = [("No risk control", d["raw"]), ("+ volatility target", d["vol_only"]),
              ("+ valuation de-lever", d["core"]), ("+ gold/Treasury sleeve", d["book"])]
    vals = [stats(r)["maxdd"] * 100 for _, r in stages]
    labels = [s for s, _ in stages]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    bars = ax.barh(range(len(vals)), vals, 0.56,
                   color=[C_NEG, "#b4553a", "#7a6a4a", C_BOOK], zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", ha="left",
                fontsize=9.5, color="#ffffff", weight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("worst peak-to-trough loss (%)")
    ax.set_xlim(min(vals) * 1.08, 0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/overlay.png", bbox_inches="tight")
    plt.close(fig)


def fig_beta(d, out):
    """Market sensitivity: beta, and the up/down capture asymmetry behind it.

    Two panels sharing one scale (all three measures are unitless ratios), rather
    than a dual axis. Left = how much market risk each book carries; right = how
    that risk splits between rising and falling days, which is the part that
    matters and the part beta alone hides.
    """
    order = ["The book", "GRID", "PAVE", "VOLT", "XLU"]
    src = {"The book": out["book"], "GRID": out["bench"].get("GRID"),
           "PAVE": out["bench"].get("PAVE"), "VOLT": out["bench"].get("VOLT"),
           "XLU": out["bench"].get("XLU")}
    order = [k for k in order if src.get(k)]
    beta = [src[k]["beta"] for k in order]
    up = [src[k]["up_capture"] for k in order]
    dn = [src[k]["down_capture"] for k in order]
    y = np.arange(len(order))
    is_book = [k == "The book" for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    ax1.barh(y, beta, 0.6, color=[C_BOOK if b else FAINT for b in is_book], zorder=3)
    for yi, v, b in zip(y, beta, is_book):
        ax1.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=9.5,
                 color=INK if b else SOFT, weight="bold" if b else "normal")
    ax1.axvline(1.0, color=FAINT, lw=1, ls=":", zorder=2)
    ax1.text(1.0, -0.62, " market", color=FAINT, fontsize=8.5, va="bottom")
    ax1.set_xlabel("beta to the S&P 500")
    ax1.set_xlim(0, max(beta) * 1.25)
    ax1.set_yticks(y)
    ax1.set_yticklabels(order, fontsize=10.5,
                        color=INK)
    ax1.set_ylim(-0.85, len(order) - 0.45)
    ax1.invert_yaxis()
    ax1.grid(axis="y", visible=False)

    h = 0.30
    ax2.barh(y - h / 2 - 0.02, up, h, color=C_BOOK, zorder=3, label="up-capture")
    ax2.barh(y + h / 2 + 0.02, dn, h, color=C_SPY, zorder=3, label="down-capture")
    for yi, u, v in zip(y, up, dn):
        ax2.text(u + 0.03, yi - h / 2 - 0.02, f"{u:.2f}", va="center", fontsize=8.5, color=INK)
        ax2.text(v + 0.03, yi + h / 2 + 0.02, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax2.axvline(1.0, color=FAINT, lw=1, ls=":", zorder=2)
    ax2.set_xlabel("share of the market's move captured")
    ax2.set_xlim(0, max(up + dn) * 1.22)
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/beta.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- costs & vol window ----
def cost_analysis(d):
    """Turnover and cost, split by source, across volatility-window settings.

    The overlay resizes the WHOLE book daily off a trailing realised-vol estimate,
    and that resizing -- not name selection -- is the bulk of the trading bill.
    """
    raw, mem = d["raw"], d["members"]
    spy = d["bench"]["SPY"].reindex(raw.index).fillna(0.0)
    years = (raw.index[-1] - raw.index[0]).days / 365.25
    w_s = config.HEDGE_SLEEVE_WEIGHT
    sleeve = (d["book"] - (1 - w_s) * d["core"]) / w_s      # exact, backed out
    level = (1.0 + raw).cumprod()
    vmult = valuation_overlay.extension_multiplier(level, spy, 1.0).reindex(raw.index).fillna(1.0)
    dates = [pd.Timestamp(x) for x in rebalance_dates(START, END, config.REBALANCE_LAG_DAYS)]

    def one(window):
        rv = raw.rolling(window).std() * np.sqrt(config.ANN)
        vscal = (config.VOL_TARGET / rv).clip(upper=config.VOL_MAX_LEVERAGE).shift(1)
        exp = (vmult * vscal.reindex(raw.index)).clip(upper=config.VOL_MAX_LEVERAGE).fillna(1.0)
        core = exp * raw.fillna(0.0) + (1.0 - exp) * config.RF / config.ANN
        book = (1 - w_s) * core + w_s * sleeve.reindex(core.index).fillna(0.0)

        prev, names_tw = None, 0.0
        for dt in dates:
            upto = mem.loc[:dt]
            if upto.empty or not upto.iloc[-1].any():
                continue
            nm = sorted(upto.iloc[-1].index[upto.iloc[-1].astype(bool)])
            w = pd.Series(1.0 / len(nm), index=nm)
            if prev is not None:
                alln = prev.index.union(w.index)
                churn = float((w.reindex(alln).fillna(0)
                               - prev.reindex(alln).fillna(0)).abs().sum())
                names_tw += churn * float(exp.asof(dt)) * (1 - w_s)
            prev = w
        resize = float(exp.diff().abs().sum()) * (1 - w_s)
        slv = float((w_s * (1 - w_s)
                     * (sleeve.fillna(0) - (exp * raw).fillna(0)).abs()).sum())
        turn = {"names": names_tw / years, "resize": resize / years, "sleeve": slv / years}
        turn["total"] = sum(turn.values())
        m = stats(book)
        net = {bp: stats(book - turn["total"] * bp / 1e4 / config.ANN)["sharpe"]
               for bp in (2, 10, 20, 30)}
        eps = {}
        for lab, (a, z) in [("covid_2020", ("2020-01-01", "2020-12-31")),
                            ("rates_2022", ("2022-01-01", "2022-12-31")),
                            ("tariffs_2025", ("2025-01-01", "2025-12-31"))]:
            c = (1 + book.loc[a:z]).cumprod()
            eps[lab] = round(float((c / c.cummax() - 1).min()), 4)
        return {"turnover": turn, "gross": m, "net_sharpe": net, "episodes": eps,
                "exposure_mean": round(float(exp.mean()), 3)}

    return {w: one(w) for w in (21, 42, 63, 126, 252)}


def fig_volwindow(cost):
    """Slower window = less trading AND worse risk-adjusted return. Strictly worse."""
    ws = sorted(cost)
    x = [cost[w]["turnover"]["total"] * 100 for w in ws]
    y = [cost[w]["net_sharpe"][10] for w in ws]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x, y, color=FAINT, lw=1.4, ls="--", zorder=2)
    for xi, yi, w in zip(x, y, ws):
        cur = w == 21
        ax.scatter(xi, yi, s=220 if cur else 110, color=C_BOOK if cur else FAINT,
                   zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"  {w}-day{'  (current)' if cur else ''}\n  MaxDD {cost[w]['gross']['maxdd']*100:.1f}%",
                    (xi, yi), color=INK if cur else SOFT, fontsize=9.5,
                    va="center", ha="left", weight="bold" if cur else "normal")
    ax.set_xlabel("annual turnover (% of NAV traded, two-way)")
    ax.set_ylabel("Sharpe, net of 10bp trading")
    ax.set_xlim(0, max(x) * 1.34)
    ax.set_ylim(min(y) - 0.06, max(y) + 0.06)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vol_window.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def main():
    d = build_returns()
    spy = d["bench"]["SPY"]
    out = {"window": [START, END], "book": stats(d["book"], spy),
           "hand": stats(d["hand"], spy),
           "stages": {k: stats(d[k], spy) for k in ["raw", "vol_only", "core", "book"]},
           "bench": {b: stats(r, spy) for b, r in d["bench"].items()},
           "ytd": stats(d["book"].loc["2026"], spy),
           "by_year": {}, "breadth": {}}
    for y in range(2019, 2027):
        seg, hseg = d["book"].loc[f"{y}"], d["hand"].loc[f"{y}"]
        h = d["held"].loc[f"{y}"]
        out["by_year"][y] = {
            "book_ret": round(float((1 + seg).prod() - 1), 4),
            "spy_ret": round(float((1 + spy.loc[f"{y}"].dropna()).prod() - 1), 4),
            "book_sharpe": stats(seg)["sharpe"], "hand_sharpe": stats(hseg)["sharpe"],
            "book_maxdd": stats(seg)["maxdd"], "hand_maxdd": stats(hseg)["maxdd"],
            "book_calmar": stats(seg)["calmar"], "hand_calmar": stats(hseg)["calmar"],
            "names": round(float(h.mean()), 1) if len(h) else None}
    w = d["weights_last"]
    out["holdings"] = sorted(w[w > 0].index)
    out["breadth"] = {"mean": round(float(d["held"].mean()), 1),
                      "last": int(d["held"].iloc[-1]), "min": int(d["held"].min())}
    json.dump(out, open(f"{OUT}/stats.json", "w"), indent=1, default=float)

    cost = cost_analysis(d)
    out["cost"] = cost
    json.dump(out, open(f"{OUT}/stats.json", "w"), indent=1, default=float)

    fig_growth(d); fig_drawdown(d); fig_years(d)
    fig_riskreturn(d); fig_breadth(d); fig_overlay(d); fig_volwindow(cost)
    fig_beta(d, out)
    print(f"wrote figures + stats.json to {OUT}")
    print(json.dumps(out["book"], indent=1))
    print("YTD:", json.dumps(out["ytd"], indent=1))


if __name__ == "__main__":
    main()
