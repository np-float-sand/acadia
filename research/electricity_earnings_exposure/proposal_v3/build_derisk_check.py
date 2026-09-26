"""Would 'SPY + the 15% gold/short-Treasury sleeve' have beaten the book?
Also: VOLT/GRID + the same sleeve, to isolate the electrification screen from the sleeve.

sleeve return = equal-weight daily pct_change of GLD & IEF  (matches
electrification_strategy/hedge_overlay.py).  wrapped = 0.85*core + 0.15*sleeve.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
RF = 0.0376
START = "2019-01-02"
PER = {"2019-2026 (full)": ("2019-01-02", "2026-08-31"),
       "2023-2024": ("2023-01-01", "2024-12-31"),
       "2025-2026": ("2025-01-01", "2026-08-31")}


def m(r, a, b):
    r = r.loc[a:b].dropna()
    if len(r) < 20:
        return (np.nan, np.nan, np.nan)
    cum = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (cum / cum.cummax() - 1).min()
    return ((cagr - RF) / vol, cum.iloc[-1] - 1, dd)


def main():
    import yfinance as yf
    rr = pd.read_csv("output_electrification/returns.csv", parse_dates=["Date"]).set_index("Date")
    book = rr["screen|+val+sleeve"].loc[START:]
    book_plain = rr["screen|plain"].loc[START:]
    book_val = rr["screen|+val"].loc[START:]

    px = yf.download(["SPY", "VOLT", "GRID", "GLD", "IEF"], start="2018-06-01",
                     end="2026-09-01", auto_adjust=True, group_by="column",
                     progress=False, threads=True)
    px = px.xs("Close", axis=1, level=0) if isinstance(px.columns, pd.MultiIndex) else px
    ret = px.pct_change().reindex(book.index)
    sleeve = ret[["GLD", "IEF"]].mean(axis=1)

    def wrap(core):  # 0.85 core + 0.15 sleeve, daily rebal
        return 0.85 * core.fillna(0) + 0.15 * sleeve.fillna(0)

    rows = [
        ("BOOK  (screen + valuation trim + sleeve)", book),
        ("book, selection only (no trim, no sleeve)", book_plain),
        ("book, selection + valuation trim (no sleeve)", book_val),
        ("SPY  plain", ret["SPY"]),
        ("SPY  + 15% GLD/IEF sleeve", wrap(ret["SPY"])),
        ("VOLT + 15% GLD/IEF sleeve", wrap(ret["VOLT"])),
        ("GRID + 15% GLD/IEF sleeve", wrap(ret["GRID"])),
    ]
    hdr = f"{'':44s} | " + " | ".join(f"{k:>26s}" for k in PER)
    print(hdr)
    print(f"{'':44s} | " + " | ".join(f"{'Sharpe':>7s} {'Ret':>8s} {'DD':>7s}" for _ in PER))
    for lab, s in rows:
        line = f"{lab:44s} |"
        for a, b in PER.values():
            sh, tot, dd = m(s, a, b)
            line += f" {sh:7.2f} {tot:+8.0%} {dd:+7.0%} |"
        print(line)


if __name__ == "__main__":
    main()
