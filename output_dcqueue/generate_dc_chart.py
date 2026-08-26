"""
Generate output_dcqueue/dc_signal_comparison.png

Three panels:
  1. Total strategy cumulative return — ICR vs. DC load signal vs. XLU
     (both hard-switch, both on today's complete/corrected data).
  2. DC load signal's long book vs. short book legs — which leg is
     actually driving the (negative) total return.
  3. Same leg breakdown for the ICR signal, for comparison.

long_book/short_book in pnl.csv are each day's weight-scaled contribution
to total portfolio return (not a fully-invested return), so their
cumulative curves show dollar-for-dollar how much each leg is
contributing to (or dragging down) total P&L — directly answering
"which leg is losing."
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# This script lives in output_dcqueue/ — its own directory holds the
# dc-queue backtest artifacts (pnl.csv etc); the icr comparison run's
# artifacts live in the sibling output_icr/ directory at repo root.
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUT = HERE
sys.path.insert(0, str(REPO_ROOT))

from grid_resilience.config import BACKTEST_START, BACKTEST_END
from grid_resilience.data.equity_prices import fetch_returns

icr = pd.read_csv(REPO_ROOT / "output_icr" / "pnl.csv", parse_dates=["Date"], index_col="Date")
dcq = pd.read_csv(HERE / "pnl.csv", parse_dates=["Date"], index_col="Date")


def cum_pct(daily_log_ret: pd.Series) -> pd.Series:
    """Convert a daily log-return series to cumulative % return, matching
    the convention already used for cum_strategy in portfolio/backtest.py."""
    return (np.exp(daily_log_ret.cumsum()) - 1) * 100


xlu_ret = fetch_returns(["XLU"], BACKTEST_START, BACKTEST_END)["XLU"].dropna()
xlu_ret = xlu_ret.reindex(dcq.index).fillna(0)
xlu_cum = cum_pct(xlu_ret)

fig, axes = plt.subplots(3, 1, figsize=(11, 13), height_ratios=[1.3, 1, 1])

# ── Panel 1: total strategy vs. XLU ──────────────────────────────────────────
ax = axes[0]
ax.plot(icr.index, cum_pct(icr["strategy"]), label="Hard switch + ICR", color="#c94040", linewidth=1.6)
ax.plot(dcq.index, cum_pct(dcq["strategy"]), label="Hard switch + DC load signal", color="#1a7f7a", linewidth=1.8)
ax.plot(xlu_cum.index, xlu_cum, label="XLU (utility ETF, buy & hold)", color="#555555", linewidth=1.3, linestyle=":")
ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
ax.set_title("Total Strategy vs. XLU — Cumulative Return (both signals corrected, same data)", fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Return (%)")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.25)

# ── Panel 2: DC load signal's long vs. short legs ────────────────────────────
ax = axes[1]
ax.plot(dcq.index, cum_pct(dcq["long_book"]), label="Long book", color="#2e7d32", linewidth=1.6)
ax.plot(dcq.index, cum_pct(dcq["short_book"]), label="Short book", color="#c94040", linewidth=1.6)
ax.plot(dcq.index, cum_pct(dcq["strategy"]), label="Total (long + short)", color="#1a2332", linewidth=1.2, linestyle="--")
ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
ax.set_title("DC Load Signal — Long Book vs. Short Book Contribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Contribution to Cum. Return (%)")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.25)

# ── Panel 3: ICR's long vs. short legs, for comparison ───────────────────────
ax = axes[2]
ax.plot(icr.index, cum_pct(icr["long_book"]), label="Long book", color="#2e7d32", linewidth=1.6)
ax.plot(icr.index, cum_pct(icr["short_book"]), label="Short book", color="#c94040", linewidth=1.6)
ax.plot(icr.index, cum_pct(icr["strategy"]), label="Total (long + short)", color="#1a2332", linewidth=1.2, linestyle="--")
ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
ax.set_title("ICR Signal — Long Book vs. Short Book Contribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Contribution to Cum. Return (%)")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.25)

for ax in axes:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.tight_layout()
out_path = OUT / "dc_signal_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Written: {out_path}")

# ── Print leg-level summary stats for the executive summary doc ─────────────
def leg_stats(daily_ret: pd.Series, label: str) -> dict:
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else float("nan")
    total = cum_pct(daily_ret).iloc[-1]
    return {"label": label, "total_return_pct": round(total, 2),
            "ann_return_pct": round(ann_ret * 100, 2), "sharpe": round(sharpe, 3)}


print("\nLeg-level stats:")
for df, name in [(icr, "ICR"), (dcq, "DC-queue")]:
    for col, leg_label in [("long_book", "long"), ("short_book", "short"), ("strategy", "total")]:
        stats = leg_stats(df[col], f"{name} {leg_label}")
        print(f"  {stats}")

xlu_stats = leg_stats(xlu_ret, "XLU")
print(f"  {xlu_stats}")
