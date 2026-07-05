"""
Generate output/backtest_performance_v2.png

Panel 1 (full width): Hard-switch L/S, Non-PJM baseline L/S, Long book,
                       Short book, XLU buy-and-hold, Equal-weight universe
Panel 2: Rolling 63-day Sharpe (strategy vs baseline)
Panel 3: Drawdown (strategy vs baseline)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False

ROOT = Path(__file__).parent
OUT  = ROOT / "output"

# ── Universe tickers (18 active names) ───────────────────────────────────────
UNIVERSE = ["NRG","VST","CNP","AEP","EXC","PPL","FE","D","PEG",
            "ETR","WEC","DTE","CMS","AEE","PCG","EIX","XEL","EVRG"]

# ── Load PnL series ───────────────────────────────────────────────────────────
hs   = pd.read_csv(OUT / "pnl_hard_switch.csv",    parse_dates=["Date"], index_col="Date")
base = pd.read_csv(OUT / "pnl_baseline_nopjm.csv", parse_dates=["Date"], index_col="Date")

start = hs.index[hs["strategy"] != 0].min()
end   = hs.index[-1]

# ── Fetch XLU + universe from yfinance ───────────────────────────────────────
print("Fetching XLU and universe returns from yfinance...")
if HAVE_YF:
    raw = yf.download(UNIVERSE + ["XLU"], start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    daily_ret = raw.pct_change().dropna(how="all")
    xlu_ret   = daily_ret["XLU"].dropna()
    ew_ret    = daily_ret[UNIVERSE].dropna(how="all").mean(axis=1)
else:
    xlu_ret = pd.Series(dtype=float)
    ew_ret  = pd.Series(dtype=float)
    print("  yfinance not available — XLU and EW universe omitted")

# ── Cumulative return helper ──────────────────────────────────────────────────
def cum(s):
    s = s.dropna()
    return ((1 + s).cumprod() - 1) * 100

def drawdown(s):
    c = (1 + s).cumprod()
    return ((c - c.cummax()) / c.cummax()) * 100

# Build cumulative series
hs_cum    = cum(hs["strategy"])
base_cum  = cum(base["strategy"])
long_cum  = cum(hs["long_book"])
short_cum = cum(hs["short_book"])
xlu_cum   = cum(xlu_ret) if not xlu_ret.empty else None
ew_cum    = cum(ew_ret)  if not ew_ret.empty  else None

# ── Colours & styles ──────────────────────────────────────────────────────────
NAVY   = "#0f2744"
TEAL   = "#1a7f7a"
GREEN  = "#2e7d32"
RED    = "#b71c1c"
AMBER  = "#e65100"
GREY   = "#757575"
PURPLE = "#6a1b9a"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                       height_ratios=[1.6, 1])

# ── Panel 1: Cumulative returns — all series ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(hs_cum.index,   hs_cum,   color=NAVY,   lw=2.2, label=f"Hard-Switch L/S  (Sharpe 0.33, {hs_cum.iloc[-1]:+.0f}%)", zorder=5)
ax1.plot(base_cum.index, base_cum, color=TEAL,   lw=1.6, ls="--", label=f"Baseline L/S — no PJM  (Sharpe 0.27, {base_cum.iloc[-1]:+.0f}%)", zorder=4)
ax1.plot(long_cum.index, long_cum, color=GREEN,  lw=1.4, ls="-.", label=f"Long book only  ({long_cum.iloc[-1]:+.0f}%)", zorder=3)
ax1.plot(short_cum.index, short_cum, color=RED,  lw=1.4, ls="-.", label=f"Short book only  ({short_cum.iloc[-1]:+.0f}%)", zorder=3)

if xlu_cum is not None:
    ax1.plot(xlu_cum.index, xlu_cum, color=AMBER, lw=1.4, ls=":", label=f"XLU buy & hold  ({xlu_cum.iloc[-1]:+.0f}%)", zorder=2)
if ew_cum is not None:
    ax1.plot(ew_cum.index, ew_cum, color=GREY, lw=1.2, ls=":", label=f"Equal-weight universe  ({ew_cum.iloc[-1]:+.0f}%)", zorder=2)

ax1.axhline(0, color="black", lw=0.6, ls=":")

ax1.set_title("Cumulative Returns — Hard-Switch Strategy, Legs, and Benchmarks",
              fontsize=11, fontweight="bold", color=NAVY, pad=10)
ax1.set_ylabel("Cumulative Return (%)", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax1.legend(fontsize=8, framealpha=0.92, loc="upper left", ncol=2)
ax1.set_facecolor("#ffffff")
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.25, ls="--")

# ── Panel 2: Rolling Sharpe ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
W = 63

def roll_sharpe(s, w=W):
    return (s.rolling(w).mean() / s.rolling(w).std() * np.sqrt(252))

ax2.plot(roll_sharpe(hs["strategy"]).index,   roll_sharpe(hs["strategy"]),   color=NAVY, lw=1.5, label="Hard-Switch")
ax2.plot(roll_sharpe(base["strategy"]).index, roll_sharpe(base["strategy"]), color=TEAL, lw=1.2, ls="--", label="Baseline")
ax2.axhline(0,   color="black", lw=0.5, ls=":")
ax2.axhline(0.3, color=AMBER,   lw=0.7, ls=":", alpha=0.7, label="0.3 target")
ax2.set_title("Rolling 63-day Sharpe", fontsize=10, fontweight="bold", color=NAVY)
ax2.set_ylabel("Annualised Sharpe", fontsize=8)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.set_facecolor("#ffffff")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.25, ls="--")

# ── Panel 3: Drawdown ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

ax3.fill_between(hs["strategy"].index,   drawdown(hs["strategy"]),   0, color=NAVY,  alpha=0.30, label=f"Hard-Switch  ({drawdown(hs['strategy']).min():.1f}% max DD)")
ax3.fill_between(base["strategy"].index, drawdown(base["strategy"]), 0, color=TEAL,  alpha=0.18, label=f"Baseline  ({drawdown(base['strategy']).min():.1f}% max DD)")
ax3.plot(hs["strategy"].index,   drawdown(hs["strategy"]),   color=NAVY, lw=1.2)
ax3.plot(base["strategy"].index, drawdown(base["strategy"]), color=TEAL, lw=1.0, ls="--")

if xlu_ret is not None and not xlu_ret.empty:
    ax3.plot(xlu_ret.index, drawdown(xlu_ret), color=AMBER, lw=0.9, ls=":", label=f"XLU  ({drawdown(xlu_ret).min():.1f}% max DD)", alpha=0.8)

ax3.set_title("Drawdown", fontsize=10, fontweight="bold", color=NAVY)
ax3.set_ylabel("Drawdown (%)", fontsize=8)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.legend(fontsize=7.5, framealpha=0.9)
ax3.set_facecolor("#ffffff")
ax3.spines[["top","right"]].set_visible(False)
ax3.grid(axis="y", alpha=0.25, ls="--")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("Grid Resilience Equity Strategy — Business-Model-Aware Architecture (July 2026)",
             fontsize=12, fontweight="bold", color=NAVY, y=0.99)

out_path = OUT / "backtest_performance_v2.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Written: {out_path}")
