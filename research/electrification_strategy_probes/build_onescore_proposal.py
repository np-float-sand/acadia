"""PRODUCTION build (adopted 2026-09-26): the one-score investor long/short.

    python research/electrification_strategy_probes/build_onescore_proposal.py

The long book is ranked on the same fourteen-signal score the short book uses (mean of each
signal's winsorized 5/95 z-score over the eligible universe plus the `screen_auto` book; a missing
signal is skipped, not counted as zero), instead of `build_tophalf_proposal`'s two-signal score
(capex-light + revenue acceleration). Everything else is `build_tophalf_proposal.main()` /
`investor_book.build` unchanged: top half with the 65% buffer, equal weight, vol target + stretch cut
+ 15% GLD/IEF sleeve, $0.25 short per $1 of the fund, step-aside rule, 10bp costs.

Writes to output_html/img/tophalf_powerlong/:
  * every chart and stats.json      -- via build_tophalf_proposal.main() with OUT redirected;
  * extras.json                     -- the write-up tables stats.json does not hold: alpha split,
                                       risk-control stages (full + 2023+), short/step-aside stages, style-factor
                                       regressions (2023+), momentum-sell-off episodes, and the
                                       up/down-month short rationale.

The investor write-up that reports these numbers is
output_html/screened_elec_selection_investor_edit_vpowerlong.md (edited by hand from these outputs);
rule book output_html/screened_elec_strategy_log_vpowerlong.md.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import build_tophalf_proposal as t  # noqa: E402
import investor_book as ib  # noqa: E402
from electrification_strategy import config  # noqa: E402
from electrification_strategy.fred import fetch_series  # noqa: E402
from research.xsec_alpha_probe import data, evaluate as ev, signals as sg, tilt  # noqa: E402

OUT = os.path.join(t.ROOT, "output_html", "img", "tophalf_powerlong")
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}.zip"
EPISODES = [("2019-08-27", "2019-09-16"), ("2019-10-09", "2019-11-08"), ("2020-04-16", "2020-06-08"),
            ("2020-10-14", "2020-11-24"), ("2021-02-12", "2021-03-29"), ("2022-06-14", "2022-08-10"),
            ("2022-12-27", "2023-02-02"), ("2023-05-12", "2023-06-16"), ("2024-07-10", "2024-08-05"),
            ("2025-01-24", "2025-01-27"), ("2025-02-07", "2025-03-10"), ("2026-06-22", "2026-08-31")]


def onescore_composite() -> pd.DataFrame:
    """The fourteen-signal score, over the universe plus the screen_auto book (as tilt.composite)."""
    univ = data.universe()
    book = tilt.book_membership(univ.index)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(index=univ.index, columns=cols,
                                                                     fill_value=False)
    px_m = data.monthly_prices(data.prices())
    sigs = {}
    sigs.update(sg.fundamental_signals())
    sigs.update(sg.price_signals(px_m))
    sigs.update(sg.power_signals(px_m, data.monthly_prices(data.benchmarks())["SPY"]))
    z = [ev.prep(v.reindex(index=univ.index, columns=cols).where(mask)) for v in sigs.values()]
    return pd.concat(z, keys=range(len(z))).groupby(level=1).mean()


def _ff_daily(name: str, cols: list[str]) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(FF_URL.format(name)).read()))
    txt = zf.read(zf.namelist()[0]).decode("latin-1")
    rows = [ln.split(",") for ln in txt.splitlines() if ln.strip()[:8].isdigit() and len(ln.split(",")) == len(cols) + 1]
    d = pd.DataFrame([[float(x) for x in r[1:]] for r in rows], columns=cols,
                     index=pd.to_datetime([r[0].strip() for r in rows]))
    return d / 100


def extras(comp: pd.DataFrame) -> dict:
    pub, members, px, _ = t.load()
    spy = px["SPY"].loc[t.START:t.END].pct_change()
    grid = px["GRID"].loc[t.START:t.END].pct_change()
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", t.END)
    ry = fetch_series(t.RATE_SIGNAL, "2003-01-01", t.END).dropna()
    cash = fetch_series(t.CASH_SERIES, "2003-01-01", t.END).dropna()
    g = t.supplier_growth(members)
    off = ry.clip(lower=1.0)
    wins = {"full": (t.START, t.END), "2023+": ("2023-01-01", t.END)}

    def st(r, a, z):
        return {k: round(float(v), 4) for k, v in t.a1.stats(r.loc[a:z], spy.loc[a:z]).items() if v is not None}

    def cum(x):
        return float((1 + x.fillna(0)).prod() - 1)

    ib._CACHE.clear()
    bb = t.build_book(("top", None, comp), members, px, spy, dfii10)
    lo = ib.build(members, comp, px, spy, dfii10, off, cash, short=None)["returns"]
    al = ib.build(members, comp, px, spy, dfii10, off, cash)
    fu = ib.build(members, comp, px, spy, dfii10, ry, cash, growth=g)
    out = {"stages": {w: {k: st(bb[k], a, z) for k in ("raw", "vol_only", "core", "book")} for w, (a, z) in wins.items()},
           "short_stages": {w: {"long_only": st(lo, a, z), "plus_short": st(al["returns"], a, z),
                                "strategy": st(fu["returns"], a, z)} for w, (a, z) in wins.items()},
           "episodes": [{"a": a, "z": z, "strategy": round(cum(fu["returns"].loc[a:z].iloc[1:]), 4),
                         "always": round(cum(al["returns"].loc[a:z].iloc[1:]), 4)} for a, z in EPISODES]}
    mo = (pd.DataFrame({"basket": al["short_returns"], "long": lo, "grid": grid}).loc["2023-01-01":t.END]
          .resample("ME").apply(lambda x: (1 + x).prod() - 1))
    mo["q"], mo["spread"] = mo["long"] - 0.25 * mo["basket"], mo["long"] - mo["basket"]
    out["updown"] = {lab: {c: round(float(m[c].mean()), 4) for c in ("basket", "long", "q", "spread", "grid")} | {"n": int(len(m))}
                     for lab, m in (("down", mo[mo.grid < 0]), ("up", mo[mo.grid > 0]), ("all", mo))}
    # style factors, 2023 on
    F = _ff_daily("F-F_Research_Data_5_Factors_2x3_daily_CSV", ["MKT", "SMB", "HML", "RMW", "CMA", "RF"]).join(
        _ff_daily("F-F_Momentum_Factor_daily_CSV", ["MOM"]), how="inner")
    import yfinance as yf
    e = yf.download(["SPY", "GRID", "MTUM", "IWF", "IWD", "QUAL", "USMV", "IWM"], start="2022-12-01", end=t.END,
                    auto_adjust=True, progress=False)["Close"].pct_change()
    D = F.join(pd.DataFrame({"y": fu["returns"], "GRID": e["GRID"], "SPYe": e["SPY"], "MOMe": e["MTUM"] - e["SPY"],
                             "GRWe": e["IWF"] - e["IWD"], "QUALe": e["QUAL"] - e["SPY"], "LOVOLe": e["USMV"] - e["SPY"],
                             "SIZEe": e["IWM"] - e["SPY"]}), how="inner").loc["2023-01-01":t.END].dropna()
    D["GRIDx"] = D.GRID - D.RF

    def reg(y, cols):
        m = sm.OLS(y, sm.add_constant(D[cols])).fit()
        return {"alpha": round(m.params["const"] * 252, 4), "t": round(m.tvalues["const"], 2), "r2": round(m.rsquared, 2),
                **{c + "_b": round(m.params[c], 2) for c in cols}, **{c + "_t": round(m.tvalues[c], 1) for c in cols}}
    ff6 = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    out["style"] = {"grid": reg(D.y, ["GRID"]), "spy": reg(D.y, ["SPYe"]), "ff6": reg(D.y - D.RF, ff6),
                    "ff6_grid": reg(D.y - D.RF, ff6 + ["GRIDx"]),
                    "etf_grid": reg(D.y, ["GRID", "MOMe", "GRWe", "QUALe", "LOVOLe", "SIZEe"])}
    return out


def alpha_split(comp: pd.DataFrame, start: str = "2023-01-01") -> dict:
    """Split the strategy's alpha vs GRID into additive sources (write-up §Sources of the Alpha).

    Re-runs investor_book.build's daily loop (same order of updates, 10bp) and books each day's return
    as long selection at the window's mean exposure, cash on the unexposed part, risk-control timing
    (exposure minus its mean, times the long book's excess over cash), sleeve, short, borrow and costs.
    The pieces sum to build()'s returns; OLS alpha is linear, so their alphas sum to the total."""
    from electrification_strategy import hedge_overlay
    pub, members, px, _ = t.load()
    spy = px["SPY"].loc[t.START:t.END].pct_change()
    grid = px["GRID"].pct_change()
    ib._CACHE.clear()
    cost, rf, k = 10 / 1e4, config.RF / config.ANN, t.FIXED_K
    raw, W = ib.long_book(members, comp, px)
    days = raw.loc[t.START:t.END].index
    raw, W = raw.reindex(days).fillna(0.0), W.reindex(days).ffill().fillna(0.0)
    hx, rtrade = ib.banded_exposure(raw, spy.reindex(days))
    sl = hedge_overlay.sleeve_returns(px).reindex(days).fillna(0.0)
    ws = config.HEDGE_SLEEVE_WEIGHT
    month_end = set(pd.Series(days, index=days).resample("ME").last().values)
    b, btrade, _, _ = ib.short_basket(px, days, long_weights=W, key=(ib.LONG_HOLD, True))
    hbar = float(hx.loc[start:].mean())
    rows, sleeve_tr, short_tr, wsl, s = [], [], [], ws, k
    for d in days:
        rb = (1 - wsl) * (hx[d] * raw[d] + (1 - hx[d]) * rf) + wsl * sl[d]
        r = rb - s * b[d] - s * t.NAME_BORROW / config.ANN
        rows.append({"long_selection": (1 - wsl) * hbar * raw[d], "cash": (1 - wsl) * (1 - hbar) * rf,
                     "timing": (1 - wsl) * (hx[d] - hbar) * (raw[d] - rf), "sleeve": wsl * sl[d],
                     "short": -s * b[d], "borrow": -s * t.NAME_BORROW / config.ANN})
        wd, sd = wsl * (1 + sl[d]) / (1 + rb), s * (1 + b[d]) / (1 + r)
        sleeve_tr.append(abs(wd - ws) * 2 if d in month_end else 0.0)
        short_tr.append(abs(sd - k) if d in month_end else 0.0)
        wsl, s = (ws, k) if d in month_end else (wd, sd)
    C = pd.DataFrame(rows, index=days)
    trades = [W.diff().abs().sum(axis=1).fillna(0.0) * hx * (1 - ws), rtrade * (1 - ws),
              pd.Series(sleeve_tr, index=days), btrade * k, pd.Series(short_tr, index=days)]
    C["costs"] = -sum(trades) * cost

    def alpha(y):
        d = pd.concat([y.rename("y"), grid.rename("g")], axis=1).loc[start:t.END].dropna()
        return float(np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.g]), d.y, rcond=None)[0][0] * config.ANN)
    out = {c: round(alpha(C[c]), 4) for c in C.columns}
    out["total"] = round(alpha(C.sum(axis=1)), 4)
    out["mean_exposure"] = round(hbar, 3)
    out["per_dollar"] = {"long_names": round(alpha(raw), 4), "short_basket": round(alpha(b), 4)}
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    warm = os.path.join(t.OUT, "_warmup_members.pkl")          # the warm-up membership main() reads from OUT
    if os.path.exists(warm) and not os.path.exists(os.path.join(OUT, "_warmup_members.pkl")):
        import shutil
        shutil.copy(warm, OUT)
    comp = onescore_composite()
    t.OUT = t.a1.OUT = OUT                                      # every chart and stats.json go here
    _load = t.load

    def load():
        pub, members, px, frozen = _load()
        return pub, members, px, comp.reindex(index=frozen.index.union(comp.index)).sort_index()
    t.load = load
    t.main()
    json.dump(extras(comp) | {"alpha_split_2023": alpha_split(comp)}, open(os.path.join(OUT, "extras.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
