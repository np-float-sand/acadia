"""Search for a genuine HEDGE to the capital-light long book.

Every short leg tested so far failed the same way: it was POSITIVELY correlated with
the book on the book's own down days (capex-heavy +0.60, DLR/EQIX +0.36, B1 +0.32).
Those are drag, not insurance. This screens explicitly on down-day behaviour.

Two candidate pools:
  A. liquid cross-asset instruments (duration, metals, FX, vol, defensives, inverse,
     anti-beta) -- held LONG as a sleeve
  B. every name in the ~950-stock broad universe -- ranked by down-day beta, to find
     whether any INDUSTRY is systematically anti-correlated (shorted if high-beta)

Metric that matters: behaviour when the book is falling, not average correlation.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from grid_equipment_basket.data.prices import fetch_prices
from electrification_strategy import backtest, config

OUT = os.path.join(ROOT, "output_html", "img", "a1a4")
START, END = "2019-01-01", "2026-08-31"
RF, ANN = config.RF, config.ANN
CANDIDATE_PX = os.path.join(OUT, "_hedge_px.parquet")

# ---- pool A: cross-asset instruments, by the mechanism they would hedge -------
POOL_A = {
    "duration":   ["TLT", "IEF", "SHY", "ZROZ", "EDV", "TIP"],
    "metals":     ["GLD", "SLV", "IAU", "GDX"],
    "fx":         ["UUP", "FXY", "FXF", "UDN"],
    "volatility": ["VIXY", "VXX", "SVXY"],
    "defensive":  ["XLP", "XLV", "XLU", "USMV", "SPLV"],
    "inverse":    ["SH", "PSQ", "RWM", "DOG"],
    "anti_beta":  ["BTAL", "QMN", "CAOS", "TAIL"],
    "commodity":  ["DBC", "USO", "UNG", "DBA"],
    "credit":     ["HYG", "LQD", "TLH"],
    # elec-adjacent, high-beta -> candidates to SHORT rather than hold
    "elec_hibeta": ["TAN", "ICLN", "LIT", "ARKK", "SMH", "IGV", "XHB", "IWM", "COPX", "URA"],
}
ALL_A = sorted({t for v in POOL_A.values() for t in v})


def get_px():
    if os.path.exists(CANDIDATE_PX):
        return pd.read_parquet(CANDIDATE_PX)
    px = fetch_prices(ALL_A + ["SPY"], "2015-01-01", END)
    px.to_parquet(CANDIDATE_PX)
    return px


def main():
    book = pd.read_pickle(os.path.join(OUT, "_build_cache.pkl"))["book"].dropna()
    px = get_px()
    rets = px.reindex(book.index).pct_change()

    # the days that actually matter
    down = book < 0
    worst = book <= book.quantile(0.10)          # the book's worst 10% of days
    # the book's real drawdown episodes
    curve = (1 + book).cumprod()
    dd = curve / curve.cummax() - 1
    in_dd = dd < -0.08                            # meaningfully underwater

    def profile(r, label, group=""):
        r = r.reindex(book.index).fillna(0.0)
        if r.abs().sum() == 0:
            return None
        b_all = float(np.cov(r, book)[0, 1] / book.var())
        b_dn = float(np.cov(r[down], book[down])[0, 1] / book[down].var())
        m = backtest.compute_metrics(r, RF, ANN)
        return {"group": group, "candidate": label,
                "drift_%yr": round(m["cagr"] * 100, 1),
                "corr": round(float(r.corr(book)), 2),
                "corr_down": round(float(r[down].corr(book[down])), 2),
                "beta": round(b_all, 2), "beta_down": round(b_dn, 2),
                "worst10_ret_%": round(float(r[worst].mean() * 100 * 252), 1),
                "in_dd_%yr": round(float(r[in_dd].mean() * 252 * 100), 1)}

    rows = []
    for grp, tks in POOL_A.items():
        for t in tks:
            if t in rets.columns:
                p = profile(rets[t], t, grp)
                if p:
                    rows.append(p)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("=" * 110)
    print("POOL A — cross-asset candidates, sorted by DOWN-DAY beta to the book")
    print("  beta_down < 0  => it rises when the book falls (a real hedge)")
    print("  worst10_ret_%  => annualised return on the book's worst 10% of days")
    print("=" * 110)
    print(df.sort_values("beta_down").to_string(index=False))

    df.to_json(os.path.join(OUT, "hedge_search_poolA.json"), orient="records", indent=1)

    # ---- pool B: does any part of the broad universe hedge? ------------------
    bpx = pd.read_parquet(os.path.join(OUT, "_broad1k_px.parquet"))
    br = bpx.reindex(book.index).pct_change()
    keep = [c for c in br.columns if br[c].notna().sum() > 1000]
    bb = {}
    for c in keep:
        r = br[c].fillna(0.0)
        if r.abs().sum() == 0:
            continue
        bb[c] = float(np.cov(r[down], book[down])[0, 1] / book[down].var())
    s = pd.Series(bb).sort_values()
    print("\n" + "=" * 110)
    print(f"POOL B — {len(s)} single stocks ranked by down-day beta to the book")
    print("=" * 110)
    print("most NEGATIVE down-day beta (would hedge if held LONG):")
    print(s.head(15).round(2).to_string())
    print("\nmost POSITIVE down-day beta (would hedge if held SHORT):")
    print(s.tail(15).round(2).to_string())
    print(f"\nhow many of {len(s)} names have NEGATIVE down-day beta: {(s < 0).sum()}")
    s.to_json(os.path.join(OUT, "hedge_search_poolB.json"), indent=1)


if __name__ == "__main__":
    main()
