"""Run every candidate signal through one pre-registered evaluation.

    python -m research.xsec_alpha_probe.run [--refresh]

Bar (fixed 2026-09-24, before results): monthly rank-IC Newey-West |t| >= 1.5 in
the pooled window AND both halves (2019-22, 2023-26), same sign -> PASS; pooled
clears but one half short -> CLOSE. Evaluated at 1m and 3m horizons.
Books: top quintile minus equal-weight eligible universe, monthly, 10bp one-way.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import data, evaluate as ev, signals as sg

START, SPLIT, END = "2019-01-31", "2022-12-31", "2026-07-31"
WINDOWS = {"full": (START, END), "2019-22": (START, SPLIT), "2023-26": ("2023-01-31", END)}
COST = 0.0010
RISK_DAYS = 63


def _slice(x, w):
    a, b = WINDOWS[w]
    return x.loc[a:b]


def risk_panels(px_d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = px_d.pct_change(fill_method=None)
    vol = r.rolling(RISK_DAYS, min_periods=40).std().resample("ME").last()
    semi = np.sqrt((r.clip(upper=0) ** 2).rolling(RISK_DAYS, min_periods=40).mean()).resample("ME").last()
    return {"eq": None, "invvol": vol, "semivol": semi}


def book_returns(sig: pd.DataFrame, fwd1: pd.DataFrame, univ: pd.DataFrame,
                 risk: pd.DataFrame | None, q: float = 0.2) -> pd.DataFrame:
    """Monthly: top-quintile long return, universe EW return, net spread after costs."""
    rows, prev = {}, None
    for dt in sig.index:
        s = sig.loc[dt].where(univ.loc[dt]).dropna() if dt in univ.index else pd.Series(dtype=float)
        r = fwd1.loc[dt].reindex(s.index) if dt in fwd1.index else None
        if r is None or s.size < ev.MIN_NAMES:
            continue
        s = s[r.notna()]
        if s.size < ev.MIN_NAMES:
            continue
        scale = risk.loc[dt].reindex(s.index) if risk is not None and dt in risk.index else None
        w = ev.bucket_weights(s, q=q, top=True, scale=scale)
        tv = w.abs().sum() if prev is None else w.sub(prev, fill_value=0).abs().sum()
        prev = w
        long_r = float((w * r.reindex(w.index)).sum())
        uni_r = float(r.reindex(s.index).mean())
        rows[dt] = {"long": long_r - tv * COST, "univ": uni_r, "spread": long_r - uni_r - tv * COST,
                    "turnover": tv}
    return pd.DataFrame(rows).T


def evaluate_signal(name: str, raw: pd.DataFrame, univ: pd.DataFrame, fwd1: pd.DataFrame,
                    fwd3: pd.DataFrame, risks: dict, grid_m: pd.Series | None) -> dict:
    cols = raw.columns.intersection(univ.columns)
    raw = raw.reindex(index=univ.index, columns=cols)
    sig = ev.prep(raw.where(univ[cols]))
    out = {"signal": name, "names/mo": float(sig.loc[START:END].notna().sum(axis=1).mean())}
    for h, fwd, lags in (("1m", fwd1, 1), ("3m", fwd3, 2)):
        ic = ev.ic_series(sig, fwd.reindex(columns=cols))
        ts = {w: ev.nw_tstat(_slice(ic, w), lags=lags) for w in WINDOWS}
        out[f"IC{h}"] = float(_slice(ic, "full").mean())
        for w in WINDOWS:
            out[f"t{h}_{w}"] = ts[w]
        out[f"verdict_{h}"] = ev.verdict(ts["full"], ts["2019-22"], ts["2023-26"])
    for scheme, risk in risks.items():
        b = book_returns(sig, fwd1.reindex(columns=cols), univ[cols], risk)
        if b.empty:          # signal never covers enough names (e.g. no NI facts in control)
            continue
        for w in WINDOWS:
            p = ev.perf(_slice(b["spread"], w))
            out[f"SR_{scheme}_{w}"] = p["sharpe"]
        out[f"ann_{scheme}"] = ev.perf(_slice(b["spread"], "full"))["ann"]
        if scheme == "semivol":
            out["turnover/yr"] = float(_slice(b["turnover"], "full").mean() * 12)
        if grid_m is not None and scheme == "eq":
            lr = _slice(b["long"], "full")
            g = grid_m.reindex(lr.index)
            beta = lr.rolling(36, min_periods=12).cov(g) / g.rolling(36, min_periods=12).var()
            hedged = lr - beta.shift(1).fillna(1.0) * g
            out["SR_gridhedged"] = ev.perf(hedged)["sharpe"]
    return out


def main(refresh: bool = False, control: bool = False) -> pd.DataFrame:
    px_d = data.prices(control)
    univ = data.universe(control)
    px_m = data.monthly_prices(px_d)
    ret_m = px_m.pct_change(fill_method=None)
    fwd1 = ret_m.shift(-1)
    fwd3 = px_m.shift(-3) / px_m - 1
    bm = data.monthly_prices(data.benchmarks())
    grid_fwd = bm["GRID"].pct_change().shift(-1)

    sigs = {}
    sigs.update(sg.fundamental_signals(control=control, refresh=refresh))
    sigs.update(sg.price_signals(px_m))
    sigs.update(sg.power_signals(px_m, bm["SPY"], refresh=refresh,
                                 tag="_control" if control else ""))
    risks = risk_panels(px_d)
    rows = []
    for name, raw in sigs.items():
        print(f"  evaluating {name} ...", flush=True)
        rows.append(evaluate_signal(name, raw, univ, fwd1, fwd3, risks,
                                    None if control else grid_fwd))
    res = pd.DataFrame(rows).set_index("signal")
    tag = "_control" if control else ""
    res.to_csv(data.CACHE / f"results{tag}.csv")
    return res


def _print(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40)
    ic_cols = ["names/mo", "IC1m", "t1m_full", "t1m_2019-22", "t1m_2023-26", "verdict_1m",
               "IC3m", "t3m_full", "t3m_2019-22", "t3m_2023-26", "verdict_3m"]
    print(res[ic_cols].round(3).to_string())
    bk = [c for c in res.columns if c.startswith("SR_") or c.startswith("ann_") or c == "turnover/yr"]
    print(res[bk].round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    _print(main(a.refresh, a.control))
