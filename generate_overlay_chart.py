"""
Generate output/backtest_performance_v2.png

Adds a dotted non-PJM baseline L/S line to the hard-switch cumulative return panel.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / "output"

# ── Load PnL series ───────────────────────────────────────────────────────────
hs   = pd.read_csv(OUT / "pnl_hard_switch.csv",    parse_dates=["Date"], index_col="Date")
base = pd.read_csv(OUT / "pnl_baseline_nopjm.csv", parse_dates=["Date"], index_col="Date")

# Cumulative returns (already cum_strategy column; recompute from daily for safety)
hs_cum   = (1 + hs["strategy"]).cumprod() - 1
base_cum = (1 + base["strategy"]).cumprod() - 1

# ── Load original chart to replicate other panels ────────────────────────────
# We'll regenerate everything from scratch for clean styling

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

navy  = "#0f2744"
teal  = "#1a7f7a"
amber = "#e8a020"
red   = "#c94040"
grey  = "#888888"

# ── Panel 1: Cumulative return with overlay ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])  # full width top row

ax1.plot(hs_cum.index,   hs_cum   * 100, color=navy,  lw=2,   label="Hard-Switch Strategy (PJM included) — Sharpe 0.33")
ax1.plot(base_cum.index, base_cum * 100, color=teal,  lw=1.5, ls="--", label="Baseline (no PJM, ERCOT/MISO/CAISO/SPP) — Sharpe 0.27")
ax1.axhline(0, color="black", lw=0.5, ls=":")
ax1.fill_between(hs_cum.index, 0, hs_cum * 100, where=hs_cum >= 0, alpha=0.08, color=navy)
ax1.fill_between(hs_cum.index, 0, hs_cum * 100, where=hs_cum < 0,  alpha=0.08, color=red)

ax1.set_title("Cumulative L/S Return — Business-Model-Aware Strategy vs Non-PJM Baseline",
              fontsize=11, fontweight="bold", color=navy, pad=10)
ax1.set_ylabel("Cumulative Return (%)", fontsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax1.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
ax1.set_facecolor("#ffffff")
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.3, ls="--")

# Annotate final values
for cum, label, color, offset in [
    (hs_cum,   f"+{hs_cum.iloc[-1]*100:.0f}%",   navy, 2),
    (base_cum, f"+{base_cum.iloc[-1]*100:.0f}%", teal, -5),
]:
    ax1.annotate(label, xy=(cum.index[-1], cum.iloc[-1]*100),
                 xytext=(cum.index[-1], cum.iloc[-1]*100 + offset),
                 fontsize=8.5, fontweight="bold", color=color, ha="right")

# ── Panel 2: Rolling Sharpe ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

window = 63
hs_roll_sharpe = (
    hs["strategy"].rolling(window).mean() /
    hs["strategy"].rolling(window).std()
) * np.sqrt(252)
base_roll_sharpe = (
    base["strategy"].rolling(window).mean() /
    base["strategy"].rolling(window).std()
) * np.sqrt(252)

ax2.plot(hs_roll_sharpe.index,   hs_roll_sharpe,   color=navy, lw=1.5, label="Hard-Switch")
ax2.plot(base_roll_sharpe.index, base_roll_sharpe, color=teal, lw=1.2, ls="--", label="Baseline")
ax2.axhline(0, color="black", lw=0.5, ls=":")
ax2.axhline(0.3, color=amber, lw=0.8, ls=":", alpha=0.7)
ax2.set_title("Rolling 63-day Sharpe", fontsize=10, fontweight="bold", color=navy)
ax2.set_ylabel("Sharpe (annualised)", fontsize=8)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.set_facecolor("#ffffff")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3, ls="--")

# ── Panel 3: Drawdown ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

def drawdown(ret):
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak * 100

ax3.fill_between(hs_cum.index,   drawdown(hs["strategy"]),   0, color=navy, alpha=0.35, label="Hard-Switch")
ax3.fill_between(base_cum.index, drawdown(base["strategy"]), 0, color=teal, alpha=0.20, label="Baseline")
ax3.plot(hs_cum.index,   drawdown(hs["strategy"]),   color=navy, lw=1)
ax3.plot(base_cum.index, drawdown(base["strategy"]), color=teal, lw=1, ls="--")

ax3.set_title("Drawdown", fontsize=10, fontweight="bold", color=navy)
ax3.set_ylabel("Drawdown (%)", fontsize=8)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.legend(fontsize=7.5, framealpha=0.9)
ax3.set_facecolor("#ffffff")
ax3.spines[["top","right"]].set_visible(False)
ax3.grid(axis="y", alpha=0.3, ls="--")

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("Grid Resilience Equity Strategy — Hard-Switch Architecture vs Non-PJM Baseline",
             fontsize=12, fontweight="bold", color=navy, y=0.98)

out_path = OUT / "backtest_performance_v2.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Written: {out_path}")
