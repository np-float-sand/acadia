"""
Build data + figures for Proposal v3 (Screened Electrification Equipment + short-offset
analysis).

Re-run:   poetry run python research/electricity_earnings_exposure/proposal_v3/build_figures.py
Outputs:  research/electricity_earnings_exposure/proposal_v3/out/
            proposal_v3.json     — every number the write-up cites
            *.png / *.b64        — the five figures (raw + base64 for inlining)
Then:     poetry run python research/electricity_earnings_exposure/proposal_v3/assemble.py --deploy --light --pdf

Inputs (repo-relative, resolved from ROOT — run from anywhere):
  output_electrification/returns.csv, metrics.csv, episode_drawdowns.csv
  yfinance adjusted closes for the benchmarks / short instruments (cached in out/pv3_px.parquet)
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
PXC = os.path.join(OUT, "pv3_px.parquet")

BOOK = "screen|+val+sleeve"
BOOK_ROW = "screen +val+sleeve"          # metrics.csv / episode_drawdowns.csv use spaces
BOOKPLAIN = "screen|plain"
BENCH = ["VOLT", "PAVE", "GRID", "SPY", "XLI"]
SHORTS_PX = ["TAN", "FXI", "DLR", "EQIX", "VST", "NRG", "CEG", "TLN", "POWL", "ATKR", "AZZ",
             "GLD", "IEF"]
FAIR0, FAIR1 = "2025-01-02", "2026-08-31"

# ---------- palette (matches the artifact) ----------
C_BOOK = "#1f5fa8"; C_SPY = "#8b929c"; C_VOLT = "#a06a1f"
C_GRID = "#2f7d5d"; C_PAVE = "#7a6f9c"; C_NEG = "#a23f3a"
INK = "#1b1f24"; SOFT = "#5a626c"; LINE = "#e2e2dd"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7, "axes.axisbelow": True,
})


def fetch_px() -> pd.DataFrame:
    if os.path.exists(PXC):
        return pd.read_parquet(PXC)
    import yfinance as yf
    tks = sorted(set(BENCH + SHORTS_PX))
    raw = yf.download(tks, start="2016-01-01", end="2026-09-01", auto_adjust=True,
                      group_by="column", progress=False, threads=True)
    px = raw.xs("Close", axis=1, level=0) if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.ffill(limit=5).dropna(how="all")
    px.to_parquet(PXC)
    return px


RF = 0.0376   # repo convention: Sharpe = (CAGR - rf) / ann_vol, rf ~ mean 3M T-bill
              # over the backtest; reproduces metrics.csv / RESEARCH-LOG (book Sharpe 0.69).


def metrics(r: pd.Series, rf_ann=RF) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 20:
        return dict(total=np.nan, cagr=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan)
    cum = (1 + r).cumprod()
    yrs = n / 252
    total = cum.iloc[-1] - 1
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (cagr - rf_ann) / vol
    mdd = (cum / cum.cummax() - 1).min()
    return dict(total=float(total), cagr=float(cagr), vol=float(vol),
                sharpe=float(sharpe), mdd=float(mdd))


def beta_to(r: pd.Series, m: pd.Series) -> float:
    d = pd.concat([r, m], axis=1).dropna()
    if len(d) < 40:
        return np.nan
    return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / np.var(d.iloc[:, 1]))


def main():
    rr = pd.read_csv(os.path.join(ROOT, "output_electrification", "returns.csv"), parse_dates=["Date"]).set_index("Date")
    # Backtest effectively begins 2019 — before that the screen universe has <8 listed
    # names (GEV/VRT/FLNC/etc. not yet public), so the pre-2019 series is a near-empty
    # warmup and is excluded from all headline stats and figures.
    START = "2019-01-02"
    book = rr[BOOK].loc[START:]
    bookp = rr[BOOKPLAIN].loc[START:]
    px = fetch_px()
    bret = px.pct_change().reindex(book.index)
    spy = bret["SPY"]

    J = {"window": {}, "long_only": {}, "episodes": {}, "shorts": {}, "meta": {}}
    full_idx = book.index
    J["meta"] = dict(book=BOOK, full_start=str(full_idx.min().date()),
                     full_end=str(full_idx.max().date()), fair=[FAIR0, FAIR1])

    # ---------- table 1: long-only vs benchmarks, both windows ----------
    def row(series, name, win):
        s = series.loc[FAIR0:FAIR1] if win == "fair" else series
        m = metrics(s)
        m["beta"] = beta_to(s, spy.loc[s.index])
        m["name"] = name
        return m
    spy_sleeve = 0.85 * spy.fillna(0) + 0.15 * bret[["GLD", "IEF"]].mean(axis=1).fillna(0)
    for win in ("full", "fair"):
        rows = [row(book, BOOK, win)]
        for b in BENCH:
            rows.append(row(bret[b], b, win))
        rows.append(row(spy_sleeve, "SPY + 15% GLD/IEF sleeve", win))
        J["long_only"][win] = rows

    # ---------- named episode drawdowns (from file) ----------
    ep = pd.read_csv(os.path.join(ROOT, "output_electrification", "episode_drawdowns.csv"), index_col=0)
    J["episodes"] = {c: float(ep.loc[BOOK_ROW, c]) for c in ep.columns}
    J["episodes_bench"] = {}
    # quick approx episode DDs for SPY/VOLT over same calendar spans
    EP_SPANS = {"COVID 20": ("2020-02-01", "2020-04-30"),
                "Rate shock 22": ("2022-01-01", "2022-10-31"),
                "DeepSeek 25": ("2025-01-20", "2025-04-10"),
                "Tariff Apr-25": ("2025-04-01", "2025-05-15"),
                "Selloff 26": ("2026-01-01", "2026-08-31")}
    for nm, (a, b) in EP_SPANS.items():
        J["episodes_bench"][nm] = {}
        for k, s in [("SPY", spy), ("VOLT", bret["VOLT"])]:
            w = s.loc[a:b].dropna()
            if len(w) > 3:
                c = (1 + w).cumprod()
                J["episodes_bench"][nm][k] = float((c / c.cummax() - 1).min())

    # ---------- short-offset overlays ----------
    baseline = metrics(book); baseline["beta"] = beta_to(book, spy); baseline["name"] = "Book (no short)"
    J["shorts"]["baseline"] = baseline

    def ew(cols):
        return bret[cols].mean(axis=1)

    CANDS = [
        ("SPY",        "broad", 0.25, spy,                       "Short the market. Sized to cut the book's 0.59 SPY beta by ~half. The blunt beta hedge."),
        ("XLI",        "broad", 0.25, bret["XLI"],               "Short industrials. Same idea as SPY but a higher-correlation leg — shares the cyclical beta, not the theme."),
        ("PAVE",       "broad", 0.25, bret["PAVE"],              "Short the infrastructure complex. Keeps the concentrated name selection, strips the picks-and-shovels sector beta."),
        ("GRID",       "broad", 0.25, bret["GRID"],              "Short the grid-infra ETF — the closest listed proxy to the book itself. Purest theme-relative construction."),
        ("TAN solar",  "thematic", 0.15, bret["TAN"],           "Short solar / DER — the 'benefits from high power prices & policy' names. The classic proposed offset."),
        ("Merchant gen","thematic",0.15, ew(["VST","NRG","CEG","TLN"]), "Short the merchant generators — the names actually long the wholesale power price."),
        ("Crowding",   "thematic", 0.15, ew(["POWL","ATKR","AZZ"]), "Short the expensive, crowded electrification adjacents that ran with the theme."),
        ("China FXI",  "thematic", 0.15, bret["FXI"],           "Short China large-cap — least correlated global-industrial cyclical, used before as a diversifier."),
        ("DC REITs",   "thematic", 0.10, ew(["DLR","EQIX"]),    "Short data-center REITs at small fixed notional — the runner-up 'insurance' leg from the hedge search."),
    ]
    # feared-scenario proxy: months where clean-energy rips vs the market (TAN - SPY, monthly)
    m_tan = (1 + bret["TAN"]).resample("ME").prod() - 1
    m_spy = (1 + spy).resample("ME").prod() - 1
    feared_months = (m_tan - m_spy).nlargest(15).index
    m_book = (1 + book).resample("ME").prod() - 1

    for name, fam, w, sr, why in CANDS:
        sr = sr.reindex(book.index).fillna(0.0)
        comb = book - w * sr
        mm = metrics(comb)
        mm["beta"] = beta_to(comb, spy)
        mm["name"] = name; mm["family"] = fam; mm["w"] = w; mm["why"] = why
        mm["d_sharpe"] = mm["sharpe"] - baseline["sharpe"]
        mm["d_mdd"] = mm["mdd"] - baseline["mdd"]
        mm["d_total"] = mm["total"] - baseline["total"]
        m_comb = (1 + comb).resample("ME").prod() - 1
        mm["feared_book"] = float(m_book.reindex(feared_months).mean())
        mm["feared_comb"] = float(m_comb.reindex(feared_months).mean())
        J["shorts"].setdefault("candidates", []).append(mm)

    # ================= FIGURES =================
    def _fin(ax):
        ax.margins(x=0.01)
        for s in ("left",):
            ax.spines[s].set_color(LINE)

    # A: growth of $1, fair window — Book vs VOLT vs SPY vs GRID
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=200)
    win = book.loc[FAIR0:FAIR1]
    for s, lab, col, lw in [(win, "Smart long-only book", C_BOOK, 2.2),
                            (bret["VOLT"].loc[FAIR0:FAIR1], "VOLT (electrification ETF)", C_VOLT, 1.6),
                            (bret["GRID"].loc[FAIR0:FAIR1], "GRID (grid-infra ETF)", C_GRID, 1.4),
                            (bret["SPY"].loc[FAIR0:FAIR1], "SPY", C_SPY, 1.4)]:
        c = (1 + s.dropna()).cumprod()
        ax.plot(c.index, c.values, color=col, lw=lw, label=lab)
        ax.scatter([c.index[-1]], [c.values[-1]], color=col, s=18, zorder=5)
    ax.axhline(1, color=LINE, lw=1)
    ax.set_title("Growth of $1 — Jan 2025 to Aug 2026 (VOLT-fair window)", fontsize=12.5, loc="left", pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _fin(ax); fig.tight_layout(); fig.savefig(f"{OUT}/growth_fair.png"); plt.close(fig)

    # B: growth of $1, full window, log — Book vs SPY vs PAVE
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=200)
    for s, lab, col, lw in [(book, "Smart long-only book", C_BOOK, 2.2),
                            (bret["PAVE"], "PAVE (infrastructure)", C_PAVE, 1.4),
                            (bret["SPY"], "SPY", C_SPY, 1.4)]:
        c = (1 + s.reindex(book.index).fillna(0)).cumprod()
        ax.plot(c.index, c.values, color=col, lw=lw, label=lab)
        ax.scatter([c.index[-1]], [c.values[-1]], color=col, s=18, zorder=5)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(["1×", "2×", "3×", "4×", "5×"])
    ax.set_title("Growth of $1 — Jan 2019 to Aug 2026 (log scale)", fontsize=12.5, loc="left", pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _fin(ax); fig.tight_layout(); fig.savefig(f"{OUT}/growth_full.png"); plt.close(fig)

    # C: drawdown, full window — Book vs SPY vs VOLT
    fig, ax = plt.subplots(figsize=(8.2, 3.7), dpi=200)
    for s, lab, col, lw in [(book, "Smart long-only book", C_BOOK, 2.0),
                            (bret["VOLT"].reindex(book.index), "VOLT", C_VOLT, 1.3),
                            (spy, "SPY", C_SPY, 1.3)]:
        c = (1 + s.fillna(0)).cumprod(); dd = c / c.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=col, lw=lw, label=lab)
        ax.fill_between(dd.index, dd.values * 100, 0, color=col, alpha=0.06)
    ax.set_title("Drawdown from prior peak (%)", fontsize=12.5, loc="left", pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    _fin(ax); fig.tight_layout(); fig.savefig(f"{OUT}/drawdown.png"); plt.close(fig)

    # D: rolling 12m Sharpe — Book vs SPY, full
    fig, ax = plt.subplots(figsize=(8.2, 3.7), dpi=200)
    for s, lab, col, lw in [(book, "Smart long-only book", C_BOOK, 2.0),
                            (spy, "SPY", C_SPY, 1.3)]:
        rs = s.fillna(0)
        rsh = (rs.rolling(252).mean() * 252) / (rs.rolling(252).std() * np.sqrt(252))
        ax.plot(rsh.index, rsh.values, color=col, lw=lw, label=lab)
    ax.axhline(0, color=LINE, lw=1); ax.axhline(1, color=LINE, lw=0.8, ls=":")
    ax.set_title("Rolling 12-month Sharpe", fontsize=12.5, loc="left", pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _fin(ax); fig.tight_layout(); fig.savefig(f"{OUT}/rollsharpe.png"); plt.close(fig)

    # E: short candidates — delta Sharpe & delta MaxDD vs baseline
    cands = J["shorts"]["candidates"]
    names = [c["name"] for c in cands]
    dsh = [c["d_sharpe"] for c in cands]
    dmd = [c["d_mdd"] for c in cands]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=200, sharey=True)
    for ax, vals, ttl in [(axes[0], dsh, "Δ Sharpe vs book"), (axes[1], dmd, "Δ Max drawdown vs book (pp)")]:
        v = np.array(vals) * (100 if "drawdown" in ttl else 1)
        cols = [C_GRID if x > 0 else C_NEG for x in v] if "Sharpe" in ttl else \
               [C_GRID if x > 0 else C_NEG for x in v]
        ax.barh(y, v, color=cols, height=0.62)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_title(ttl, fontsize=11.5, loc="left", pad=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Every short-offset candidate, applied to the book (full window)",
                 fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUT}/shorts_bars.png"); plt.close(fig)

    # ================= PERIOD-SPLIT TABLE (2023-24 vs 2025-26) =================
    PER = {"2023-2024": ("2023-01-01", "2024-12-31"),
           "2025-2026": ("2025-01-01", "2026-08-31")}

    def pstat(r, a, b):
        m = metrics(r.loc[a:b])
        return m["sharpe"], m["total"], m["mdd"]

    sr = px.pct_change().reindex(book.index).fillna(0.0)
    def ew2(cols): return sr[cols].mean(axis=1)
    rows_tbl = []
    rows_tbl.append(("Electrification Equipment — screened & risk-managed  (the book)", book))
    rows_tbl.append(("  variant · raw book (no overlays)", rr["screen|plain"].loc[START:]))
    rows_tbl.append(("  variant · + built-in DLR/EQIX insurance short", rr["screen|+val+sleeve+short"].loc[START:]))
    rows_tbl.append(("  variant · thematic universe + overlays", rr["thematic|+val+sleeve"].loc[START:]))
    rows_tbl.append(("book + short · XLI 0.25x", book - 0.25 * sr["XLI"]))
    rows_tbl.append(("book + short · China FXI 0.15x", book - 0.15 * sr["FXI"]))
    rows_tbl.append(("book + short · TAN solar 0.15x", book - 0.15 * sr["TAN"]))
    rows_tbl.append(("book + short · merchant gen 0.15x", book - 0.15 * ew2(["VST","NRG","CEG","TLN"])))
    for b in ["VOLT", "GRID", "PAVE", "SPY", "XLI"]:
        rows_tbl.append((f"benchmark · {b}", sr[b]))
    rows_tbl.append(("benchmark · SPY + 15% GLD/IEF sleeve",
                     0.85 * sr["SPY"].fillna(0) + 0.15 * sr[["GLD", "IEF"]].mean(axis=1).fillna(0)))

    J["period_table"] = []
    lines = ["", "PERIOD-SPLIT TABLE",
             f"{'':52s} | {'2023-2024':>22s} | {'2025-2026':>22s}",
             f"{'':52s} | {'Sharpe':>6s} {'Return':>7s} {'MaxDD':>7s} | {'Sharpe':>6s} {'Return':>7s} {'MaxDD':>7s}"]
    for lab, series in rows_tbl:
        cells = {}
        out = f"{lab[:52]:52s} |"
        for pk, (a, b) in PER.items():
            sh, tot, dd = pstat(series, a, b)
            cells[pk] = dict(sharpe=sh, total=tot, mdd=dd)
            out += f" {sh:6.2f} {tot:+6.0%} {dd:+6.0%} |"
        lines.append(out)
        J["period_table"].append(dict(label=lab.strip(), **{k: v for k, v in cells.items()}))
    print("\n".join(lines))

    with open(os.path.join(OUT, "proposal_v3.json"), "w") as f:
        json.dump(J, f, indent=2)
    print("wrote proposal_v3.json and", len(os.listdir(OUT)), "figures to", OUT)
    # console summary
    print("\nLONG-ONLY (full / fair):")
    for win in ("full", "fair"):
        print(f"  [{win}]")
        for r in J["long_only"][win]:
            print(f"    {r['name']:22s} tot {r['total']:+.0%}  cagr {r['cagr']:+.1%}  "
                  f"vol {r['vol']:.1%}  sharpe {r['sharpe']:.2f}  mdd {r['mdd']:.0%}  beta {r.get('beta',float('nan')):.2f}")
    print("\nSHORT OVERLAYS (full window):")
    b = J["shorts"]["baseline"]
    print(f"  {'Book (no short)':16s} tot {b['total']:+.0%}  sharpe {b['sharpe']:.2f}  mdd {b['mdd']:.0%}  beta {b['beta']:.2f}")
    for c in cands:
        print(f"  {c['name']:16s} w{c['w']:.2f}  tot {c['total']:+.0%}  sharpe {c['sharpe']:.2f} "
              f"({c['d_sharpe']:+.2f})  mdd {c['mdd']:.0%} ({c['d_mdd']*100:+.1f}pp)  "
              f"feared {c['feared_comb']:+.1%} vs {c['feared_book']:+.1%}")


if __name__ == "__main__":
    main()
