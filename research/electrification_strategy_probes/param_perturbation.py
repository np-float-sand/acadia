"""One-at-a-time parameter perturbation of the one-score screened-electrification long/short book.

    python research/electrification_strategy_probes/param_perturbation.py

Builds the canonical `_vpowerlong` book (`investor_book.build` on the 14-signal one-score composite,
step-aside to T-bills when the 10y real yield or suppliers' real revenue growth is <= 0, 10bp on every trade) and then moves ONE setting at a time
around its live value, holding everything else fixed. Settings come from three places: `build()` kwargs,
`electrification_strategy.config` (vol target, stretch cut, sleeve, cap, rebalance lag) and
`build_tophalf_proposal` (borrow). Reports Sharpe / Calmar / CAGR / vol / MaxDD for 2023+ (primary) and
the full 2019+ window, and draws one row of panels per setting.

Writes output_html/img/param_sweep/{sweep.json, <param>.png} (--out NAME for another folder; --plot redraws only).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import build_onescore_proposal as bo  # noqa: E402
import build_tophalf_proposal as t  # noqa: E402
import investor_book as ib  # noqa: E402
from electrification_strategy import config  # noqa: E402

OUT = os.path.join(t.ROOT, "output_html", "img", "param_sweep")
WINS = {"2023+": ("2023-01-01", t.END), "full": (t.START, t.END)}
VAL_KEYS = ("VAL_EXT_LO", "VAL_EXT_HI", "VAL_RS_LO", "VAL_RS_HI")

# (name, where, values, base, group, what it does). where: "kw" build kwarg, "cfg" config attribute,
# "t" build_tophalf_proposal attribute, "val" scale on the four stretch-cut thresholds.
# Settings that change the long book's names need the long-book/short-basket cache cleared ("rebuild").
PARAMS = [
    ("VOL_TARGET", "cfg", [0.12, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30], 0.20, "Sizing",
     "Annualised volatility the long book is levered/de-levered to (exposure = target / trailing realised vol)."),
    ("VOL_LOOKBACK", "cfg", [10, 15, 21, 42, 63], 21, "Sizing",
     "Trading days of returns used to measure the realised vol that sets exposure."),
    ("VOL_MAX_LEVERAGE", "cfg", [1.0, 1.25, 1.5, 1.75, 2.0], 1.5, "Sizing",
     "Ceiling on long-book exposure from the vol target and stretch cut combined."),
    ("band", "kw", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 0.20, "Sizing",
     "No-trade band: long exposure is resized only when the target moves more than this from the drifted level."),
    ("stretch scale", "val", [0.6, 0.8, 1.0, 1.2, 1.4], 1.0, "Stretch cut",
     "Multiplier on all four stretch-cut thresholds (extension above 200d MA 15%/35%, 12m excess vs SPY 25%/50%); lower = cuts sooner."),
    ("VAL_MULT_MID", "cfg", [0.6, 0.7, 0.8, 0.9, 1.0], 0.8, "Stretch cut",
     "Exposure multiplier when the book is moderately stretched (between the low and high thresholds)."),
    ("VAL_MULT_LOW", "cfg", [0.4, 0.5, 0.6, 0.7, 0.8], 0.6, "Stretch cut",
     "Exposure multiplier when the book is very stretched (above the high threshold)."),
    ("VAL_MA_DAYS", "cfg", [100, 150, 200, 250], 200, "Stretch cut",
     "Moving-average length the book's extension is measured against."),
    ("VAL_RS_DAYS", "cfg", [126, 189, 252, 378], 252, "Stretch cut",
     "Look-back for the book's excess return over SPY in the stretch test."),
    ("HEDGE_SLEEVE_WEIGHT", "cfg", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25], 0.15, "Sleeve",
     "Share of capital in the GLD/IEF defensive sleeve, reset monthly."),
    ("top_frac", "kw", [0.33, 0.40, 0.50, 0.60, 0.67], 0.50, "Long book",
     "Share of the ranked eligible list that new long names are drawn from (0.5 = top half)."),
    ("long_hold", "kw", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75], 0.65, "Long book",
     "Hold buffer: a held name stays while it ranks inside this top share of the list (0.50 = no buffer)."),
    ("MAX_SINGLE_NAME_WEIGHT", "cfg", [0.10, 0.15, 0.20, 0.25, 0.33], 0.25, "Long book",
     "Cap on any one long name's weight (equal weight otherwise)."),
    ("REBALANCE_LAG_DAYS", "cfg", [21, 42, 63], 42, "Long book",
     "Calendar days after quarter-end the long book rebalances (inputs are filed-date gated, so a shorter lag just uses staler data, no look-ahead)."),
    ("short_k", "kw", [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50], 0.25, "Short book",
     "Dollars short per $1 of fund, reset monthly."),
    ("short_q", "kw", [0.10, 0.15, 0.20, 0.25, 0.30], 0.20, "Short book",
     "Short the bottom this-quantile of the 14-signal score, drawn from companies whose 10-Ks DO use grid-customer wording ('data center', 'substation', 'switchgear', 'transmission and distribution', …) and that the long book doesn't hold."),
    ("short_freq", "kw", ["M", "2M", "Q", "S"], "Q", "Short book",
     "How often the short basket is re-formed (monthly, bi-monthly, quarterly, semi-annual)."),
    ("NAME_BORROW", "t", [0.0, 0.01, 0.02, 0.04], 0.01, "Short book",
     "Annual borrow fee charged on the single-name short notional."),
    ("ry_thr", "kw", [-0.5, -0.25, 0.0, 0.25, 0.5], 0.0, "Step-aside switch",
     "Step-aside switch, part 1: the fund is invested only while the 10-year real yield (DFII10, %) is above this; otherwise it all sits in 3-month T-bills."),
    ("g_thr", "kw", [-0.02, -0.01, 0.0, 0.01, 0.02], 0.0, "Step-aside switch",
     "Step-aside switch, part 2: …and only while the eligible suppliers' median revenue growth after inflation is above this."),
    ("bp", "kw", [0.0, 5.0, 10.0, 20.0, 30.0], 10.0, "Costs",
     "Trading cost in basis points charged on every trade (long names, long resizing, sleeve resets, short book, and moving in/out of T-bills)."),
]
REBUILD = {"top_frac", "long_hold", "MAX_SINGLE_NAME_WEIGHT", "REBALANCE_LAG_DAYS"}


def metrics(r: pd.Series) -> dict:
    out = {}
    for w, (a, e) in WINS.items():
        s = t.a1.stats(r.loc[a:e])
        out[w] = {k: round(float(s[k]), 4) for k in ("sharpe", "calmar", "cagr", "vol", "maxdd")}
    return out


def clear_books():
    for k in [k for k in ib._CACHE if k != "comp14"]:
        del ib._CACHE[k]


def main():
    os.makedirs(OUT, exist_ok=True)
    pub, members, px, frozen = t.load()
    comp = bo.onescore_composite()
    comp = comp.reindex(index=frozen.index.union(comp.index)).sort_index()
    spy = px["SPY"].loc[t.START:t.END].pct_change()
    dfii = t.fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", t.END)
    ry = t.fetch_series(t.RATE_SIGNAL, "2003-01-01", t.END).dropna()
    cash = t.fetch_series(t.CASH_SERIES, "2003-01-01", t.END).dropna()
    growth = t.supplier_growth(members)

    def run(**kw):
        return ib.build(members, comp, px, spy, dfii, ry, cash, growth=growth, **kw)["returns"]

    base = metrics(run())
    print("base", base)
    res = {"base": base, "params": []}
    for name, where, vals, b, group, desc in PARAMS:
        rows = []
        for v in vals:
            saved = {}
            kw = {}
            if where == "kw":
                kw = {name: (None if (name == "long_hold" and v == 0.50) else v)}
            elif where == "cfg":
                saved[(config, name)] = getattr(config, name)
                setattr(config, name, v)
            elif where == "t":
                saved[(t, name)] = getattr(t, name)
                setattr(t, name, v)
            elif where == "val":
                for k in VAL_KEYS:
                    saved[(config, k)] = getattr(config, k)
                    setattr(config, k, getattr(config, k) * v)
            if name in REBUILD:
                clear_books()
            try:
                m = metrics(run(**kw))
            finally:
                for (mod, k), old in saved.items():
                    setattr(mod, k, old)
                if name in REBUILD:
                    clear_books()
            rows.append({"value": v, **m})
            print(f"{name:24s} {str(v):6s} SR23 {m['2023+']['sharpe']:.2f}  DD23 {m['2023+']['maxdd']:.1%}"
                  f"  SRfull {m['full']['sharpe']:.2f}")
        res["params"].append({"name": name, "base": b, "group": group, "desc": desc, "rows": rows})
        json.dump(res, open(os.path.join(OUT, "sweep.json"), "w"), indent=1)
    plot(res)
    print("wrote", OUT)


C_SR, C_CAL, C_CAGR, C_VOL, C_DD = "#2f6fdb", "#d97706", "#15803d", "#7c3aed", "#b91c1c"


def plot(res):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for p in res["params"]:
        rows = p["rows"]
        cat = isinstance(rows[0]["value"], str)
        xs = list(range(len(rows))) if cat else [r["value"] for r in rows]
        bx = rows.index(next(r for r in rows if r["value"] == p["base"]))
        fig, axes = plt.subplots(1, 3, figsize=(11, 2.5))
        panels = [("Sharpe & Calmar", [("sharpe", "Sharpe", C_SR), ("calmar", "Calmar", C_CAL)], False),
                  ("CAGR & vol", [("cagr", "CAGR", C_CAGR), ("vol", "Vol", C_VOL)], True),
                  ("Max drawdown", [("maxdd", "MaxDD", C_DD)], True)]
        for ax, (title, series, pct) in zip(axes, panels):
            for key, lab, c in series:
                y23 = [r["2023+"][key] for r in rows]
                yf = [r["full"][key] for r in rows]
                ax.plot(xs, y23, "-o", color=c, ms=3.5, lw=1.8, label=f"{lab} 2023+")
                ax.plot(xs, yf, "--", color=c, lw=1.0, alpha=0.55, label=f"{lab} 2019+")
            ax.axvline(xs[bx], color="#888", lw=0.8, ls=":")
            ax.set_title(title, fontsize=9, loc="left")
            if pct:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if cat:
                ax.set_xticks(xs, [r["value"] for r in rows])
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, frameon=False, ncol=2 if len(series) > 1 else 1)
        fig.suptitle(f"{p['name']}  (live = {p['base']})", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{p['name'].replace(' ', '_')}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    if "--out" in sys.argv:                          # e.g. --out param_sweep_clean
        OUT = os.path.join(t.ROOT, "output_html", "img", sys.argv[sys.argv.index("--out") + 1])
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "sweep.json"))))
    else:
        main()
