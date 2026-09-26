"""Part B — search for a short basket that keeps or improves Sharpe AND Calmar.

    python research/electrification_strategy_probes/build_a1a4_short_probe.py

Pre-registered bar, fixed before the run: combined Sharpe >= the long book's 1.050
AND Calmar >= its 1.118, with turnover/borrow not materially worse.

Writes output_html/img/a1a4/short_probe.{png,json}. Candidates are built from the
same point-in-time panel as the long book -- filed-date-aware net income and capex,
so nothing uses information the strategy would not have had.
"""
from __future__ import annotations

import json, os, pickle, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from grid_equipment_basket.basket import simulate_basket, apply_cap, rebalance_dates
from capex_intensity_strategy import edgar, fundamentals, pipeline
from electrification_strategy import auto_screen, backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
CACHE = os.path.join(OUT, "_short_cache.pkl")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN

SURFACE = "#1b1e23"
C_BOOK, C_SHORT, C_LONG = "#3987e5", "#d95926", "#199e70"
INK, SOFT, FAINT, LINE = "#e6e9ec", "#9aa3ad", "#727a84", "#2c313a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0, "axes.labelcolor": SOFT,
    "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 160,
})


def M(r):
    x = backtest.compute_metrics(r.dropna(), RF, ANN)
    return {"sharpe": x["sharpe"], "cagr": x["cagr"], "vol": x["ann_vol"],
            "maxdd": x["max_dd"],
            "calmar": round(x["cagr"] / abs(x["max_dd"]), 3) if x["max_dd"] < 0 else None}


def build():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    built = pipeline.build(start=START, end=END)
    panel = fundamentals.ttm_panel(edgar.fetch_facts({}), pd.date_range(START, END, freq="ME"))
    score, mask = built["score"], built["mask"]
    px = built["prices"]
    power = set(built["candidates"].loc[built["candidates"]["group"] == "POWER", "ticker"])
    idx = score.index
    elig = mask.reindex(index=idx, columns=score.columns).fillna(False)
    ni = panel.pivot_table(index="date", columns="ticker", values="net_income", dropna=False)
    nip = ni.reindex(index=idx, columns=score.columns).ffill()
    # unprofitable for 12 consecutive months, as filed
    persistent = (nip < 0).fillna(False).rolling(12, min_periods=6).min().fillna(0).astype(bool)
    rank = score.where(elig).rank(axis=1, pct=True)
    cands = {"B1s": (rank >= 0.75) & persistent & elig,     # heaviest quartile -- sharpest
             "B1m": (rank >= 0.60) & persistent & elig,     # heaviest 40% -- RECOMMENDED
             "B1h": (rank >= 0.50) & persistent & elig}     # heaviest half -- broadest
    out = {"px": px, "cands": cands, "power": power}
    pickle.dump(out, open(CACHE, "wb"))
    return out


def basket(memb, px):
    cols = [c for c in memb.columns if c in px.columns]
    m = memb[cols]

    def tfn(available, asof):
        asof = pd.Timestamp(asof)
        upto = m.loc[:asof]
        if upto.empty:
            return pd.Series(dtype=float)
        row = upto.iloc[-1]
        held = set(row.index[row.astype(bool)])
        counts = px.loc[:asof].notna().sum()
        listed = set(counts[counts >= config.LISTING_MIN_DAYS].index)
        e = sorted(held & set(available) & listed)
        if not e:
            return pd.Series(dtype=float)
        return apply_cap(pd.Series(1.0 / len(e), index=e), config.MAX_SINGLE_NAME_WEIGHT)

    br = simulate_basket(px[cols], START, END, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, tfn)
    dates = [pd.Timestamp(x) for x in rebalance_dates(START, END, config.REBALANCE_LAG_DAYS)]
    prev, tw = None, 0.0
    for dt in dates:
        upto = m.loc[:dt]
        if upto.empty or not upto.iloc[-1].any():
            continue
        nm = sorted(upto.iloc[-1].index[upto.iloc[-1].astype(bool)])
        w = pd.Series(1.0 / len(nm), index=nm)
        if prev is not None:
            a = prev.index.union(w.index)
            tw += float((w.reindex(a).fillna(0) - prev.reindex(a).fillna(0)).abs().sum())
        prev = w
    yrs = (px.loc[START:END].index[-1] - px.loc[START:END].index[0]).days / 365.25
    return br, tw / yrs


def fig(bk, sr, res):
    comb = bk - 0.15 * sr
    fig_, ax = plt.subplots(figsize=(8.4, 4.1))
    for lab, r, c, lw in [("Long only", bk, C_LONG, 1.8),
                          ("Long − 0.15× short basket", comb, C_BOOK, 2.2)]:
        cv = (1 + r.dropna()).cumprod()
        ax.plot(cv.index, cv.values, color=c, lw=lw, label=lab)
        ax.annotate(f"  {cv.iloc[-1]:.2f}x", (cv.index[-1], cv.iloc[-1]), color=c,
                    fontsize=9.5, va="center", weight="bold")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4, 6])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylabel("growth of $1 (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    cv = (1 + comb.dropna()).cumprod()
    ax.set_xlim(cv.index[0], cv.index[-1] + pd.Timedelta(days=260))
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK)
    fig_.tight_layout()
    fig_.savefig(f"{OUT}/short_probe.png", bbox_inches="tight")
    plt.close(fig_)


def main():
    d = build()
    bk = pickle.load(open(os.path.join(OUT, "_build_cache.pkl"), "rb"))["book"]
    res = {"bar": M(bk)}
    for key, memb in d["cands"].items():
        br, turn = basket(memb, d["px"])
        sr = br.returns.reindex(bk.index).fillna(0.0)
        held = (br.weights > 0).sum(axis=1)
        down = bk < 0
        e = {"names_mean": round(float(held.mean()), 1), "names_min": int(held.min()),
             "standalone": M(sr), "turnover": round(turn, 3),
             "corr": round(float(sr.corr(bk)), 2),
             "corr_down": round(float(sr[down].corr(bk[down])), 2),
             "ever": sorted(set(br.weights.columns[(br.weights > 0).any()])),
             "weights": {}, "borrow": {}, "squeeze": {}}
        for w in (0.10, 0.15, 0.25):
            c = M(bk - w * sr)
            c["pass"] = bool(c["sharpe"] >= res["bar"]["sharpe"]
                             and c["calmar"] >= res["bar"]["calmar"])
            e["weights"][w] = c
        for b in (0.0, 0.05, 0.10, 0.20, 0.40):
            c = M(bk - 0.15 * (sr + b / ANN))
            c["pass"] = bool(c["sharpe"] >= res["bar"]["sharpe"]
                             and c["calmar"] >= res["bar"]["calmar"])
            e["borrow"][b] = c
        mo = (1 + sr).resample("ME").prod() - 1
        r21 = (1 + sr).rolling(21).apply(np.prod, raw=True) - 1
        e["squeeze"] = {"worst_month": round(float(mo.max()), 4),
                        "worst_month_date": str(mo.idxmax().date()),
                        "worst_21d": round(float(r21.max()), 4),
                        "worst_21d_date": str(r21.idxmax().date())}
        e["by_year"] = {y: round(float((1 + sr.loc[f"{y}"]).prod() - 1), 4)
                        for y in range(2019, 2027)}
        res[key] = e
        if key == "B1m":
            fig(bk, sr, e)
    json.dump(res, open(f"{OUT}/short_probe.json", "w"), indent=1, default=float)
    print(json.dumps({k: (v if k == "bar" else
                          {"names": v["names_mean"], "standalone": v["standalone"],
                           "w15": v["weights"][0.15]}) for k, v in res.items()},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
